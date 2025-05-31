"""
Caching system for embedding generation.
"""

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from .exceptions import EmbeddingCacheError

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    Caching system for embeddings with memory and disk storage.

    Provides both in-memory LRU cache for fast access and persistent
    disk cache for long-term storage.
    """

    def __init__(
        self,
        memory_size: int = 1000,
        disk_cache_dir: Path | None = None,
        enable_disk_cache: bool = True,
    ):
        """
        Initialize the embedding cache.

        Args:
            memory_size: Maximum number of embeddings in memory cache
            disk_cache_dir: Directory for disk cache storage
            enable_disk_cache: Whether to enable persistent disk caching
        """
        self.memory_size = memory_size
        self.enable_disk_cache = enable_disk_cache

        # Memory cache: key -> (embedding, timestamp, access_count)
        self._memory_cache: dict[str, tuple[list[float], float, int]] = {}
        self._access_order: list[str] = []

        # Disk cache setup
        if enable_disk_cache:
            if disk_cache_dir is None:
                self.disk_cache_dir: Path = (
                    Path.home() / ".game_loop" / "embedding_cache"
                )
            else:
                self.disk_cache_dir = Path(disk_cache_dir)
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.disk_cache_dir = None  # type: ignore

        # Cache statistics
        self._stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
            "writes": 0,
            "evictions": 0,
        }

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def get(self, text_hash: str) -> list[float] | None:
        """
        Retrieve embedding from cache.

        Args:
            text_hash: Hash of the text to retrieve embedding for

        Returns:
            Embedding vector if found, None otherwise
        """
        async with self._lock:
            try:
                # Check memory cache first
                if text_hash in self._memory_cache:
                    embedding, timestamp, access_count = self._memory_cache[text_hash]

                    # Update access information
                    self._memory_cache[text_hash] = (
                        embedding,
                        timestamp,
                        access_count + 1,
                    )

                    # Update access order for LRU
                    if text_hash in self._access_order:
                        self._access_order.remove(text_hash)
                    self._access_order.append(text_hash)

                    self._stats["memory_hits"] += 1
                    logger.debug(f"Memory cache hit for {text_hash}")
                    return embedding

                # Check disk cache if enabled
                if self.enable_disk_cache and self.disk_cache_dir:
                    disk_embedding = await self._get_from_disk(text_hash)
                    if disk_embedding is not None:
                        # Add to memory cache
                        await self._add_to_memory(text_hash, disk_embedding)
                        self._stats["disk_hits"] += 1
                        logger.debug(f"Disk cache hit for {text_hash}")
                        return disk_embedding

                # Cache miss
                self._stats["misses"] += 1
                logger.debug(f"Cache miss for {text_hash}")
                return None

            except Exception as e:
                logger.error(f"Error retrieving from cache: {e}")
                raise EmbeddingCacheError(f"Cache retrieval failed: {e}") from e

    async def set(self, text_hash: str, embedding: list[float]) -> None:
        """
        Store embedding in cache.

        Args:
            text_hash: Hash of the text
            embedding: Embedding vector to store
        """
        async with self._lock:
            try:
                # Add to memory cache
                await self._add_to_memory(text_hash, embedding)

                # Add to disk cache if enabled
                if self.enable_disk_cache and self.disk_cache_dir:
                    await self._save_to_disk(text_hash, embedding)

                self._stats["writes"] += 1
                logger.debug(f"Cached embedding for {text_hash}")

            except Exception as e:
                logger.error(f"Error storing in cache: {e}")
                raise EmbeddingCacheError(f"Cache storage failed: {e}") from e

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache performance statistics
        """
        total_requests = (
            self._stats["memory_hits"]
            + self._stats["disk_hits"]
            + self._stats["misses"]
        )

        hit_rate = 0.0
        if total_requests > 0:
            hit_rate = (
                self._stats["memory_hits"] + self._stats["disk_hits"]
            ) / total_requests

        return {
            **self._stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "memory_cache_size": len(self._memory_cache),
            "memory_cache_max": self.memory_size,
            "disk_cache_enabled": self.enable_disk_cache,
            "disk_cache_dir": str(self.disk_cache_dir) if self.disk_cache_dir else None,
        }

    async def clear_cache(self, cache_type: str = "all") -> None:
        """
        Clear cache contents.

        Args:
            cache_type: Type of cache to clear ("memory", "disk", or "all")
        """
        async with self._lock:
            try:
                if cache_type in ("memory", "all"):
                    self._memory_cache.clear()
                    self._access_order.clear()
                    logger.info("Memory cache cleared")

                if cache_type in ("disk", "all") and self.disk_cache_dir:
                    await self._clear_disk_cache()
                    logger.info("Disk cache cleared")

                # Reset statistics
                if cache_type == "all":
                    self._stats = {
                        "memory_hits": 0,
                        "disk_hits": 0,
                        "misses": 0,
                        "writes": 0,
                        "evictions": 0,
                    }

            except Exception as e:
                logger.error(f"Error clearing cache: {e}")
                raise EmbeddingCacheError(f"Cache clearing failed: {e}") from e

    async def _add_to_memory(self, text_hash: str, embedding: list[float]) -> None:
        """Add embedding to memory cache with LRU eviction."""
        current_time = time.time()

        # If already in cache, update it
        if text_hash in self._memory_cache:
            _, _, access_count = self._memory_cache[text_hash]
            self._memory_cache[text_hash] = (embedding, current_time, access_count)

            # Update access order
            if text_hash in self._access_order:
                self._access_order.remove(text_hash)
            self._access_order.append(text_hash)
            return

        # Check if cache is full
        if len(self._memory_cache) >= self.memory_size:
            await self._evict_lru()

        # Add new entry
        self._memory_cache[text_hash] = (embedding, current_time, 0)
        self._access_order.append(text_hash)

    async def _evict_lru(self) -> None:
        """Evict least recently used item from memory cache."""
        if not self._access_order:
            return

        # Remove least recently used item
        lru_key = self._access_order.pop(0)
        if lru_key in self._memory_cache:
            del self._memory_cache[lru_key]
            self._stats["evictions"] += 1
            logger.debug(f"Evicted {lru_key} from memory cache")

    async def _get_from_disk(self, text_hash: str) -> list[float] | None:
        """Retrieve embedding from disk cache."""
        if not self.disk_cache_dir:
            return None

        cache_file = self.disk_cache_dir / f"{text_hash}.json"

        try:
            if cache_file.exists():
                with open(cache_file) as f:
                    data = json.load(f)
                    embedding_data = data.get("embedding")
                    if isinstance(embedding_data, list) and all(
                        isinstance(x, float) for x in embedding_data
                    ):
                        return embedding_data
                    elif embedding_data is not None:
                        logger.warning(f"Invalid embedding format in {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to read disk cache file {cache_file}: {e}")

        return None

    async def _save_to_disk(self, text_hash: str, embedding: list[float]) -> None:
        """Save embedding to disk cache."""
        if not self.disk_cache_dir:
            return

        cache_file = self.disk_cache_dir / f"{text_hash}.json"

        try:
            data = {
                "embedding": embedding,
                "timestamp": time.time(),
                "hash": text_hash,
            }

            with open(cache_file, "w") as f:
                json.dump(data, f, separators=(",", ":"))

        except Exception as e:
            logger.warning(f"Failed to write disk cache file {cache_file}: {e}")

    async def _clear_disk_cache(self) -> None:
        """Clear all files from disk cache directory."""
        if not self.disk_cache_dir:
            return

        try:
            for cache_file in self.disk_cache_dir.glob("*.json"):
                cache_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to clear disk cache: {e}")


def create_text_hash(text: str, model: str = "", entity_type: str = "") -> str:
    """
    Create a hash for text to use as cache key.

    Args:
        text: The input text
        model: Model name used for embedding
        entity_type: Type of entity for context

    Returns:
        Hash string to use as cache key
    """
    # Combine text, model, and entity_type for unique hash
    combined = f"{model}:{entity_type}:{text}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()
