"""
Search cache manager for semantic search.

This module provides caching functionality for search results
to improve performance for repeated or similar queries.
"""

import hashlib
import json
import logging
import re
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class SearchCacheManager:
    """Manage caching of search results for performance optimization."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """
        Initialize the search cache manager.

        Args:
            max_size: Maximum number of items in the cache
            ttl_seconds: Default time-to-live for cached items in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, tuple[Any, float]] = (
            OrderedDict()
        )  # {key: (value, expiry_time)}
        self._hit_stats = {"hits": 0, "misses": 0}
        self._last_cleanup = time.time()

    async def get(self, key: str) -> Any | None:
        """
        Get an item from the cache if it exists and is not expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        if key in self._cache:
            value, expiry_time = self._cache[key]

            # Check if expired
            if expiry_time > time.time():
                # Item is valid, move to end to maintain LRU order
                self._cache.move_to_end(key)
                self._hit_stats["hits"] += 1
                return value
            else:
                # Item is expired, remove it
                del self._cache[key]

        self._hit_stats["misses"] += 1
        return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        Add an item to the cache with optional custom TTL.

        Args:
            key: Cache key
            value: Value to store
            ttl: Time-to-live in seconds (uses default if None)
        """
        # Enforce size limit if needed before adding new item
        if len(self._cache) >= self.max_size:
            self._enforce_size_limit()

        # Calculate expiry time
        expiry_time = time.time() + (ttl if ttl is not None else self.ttl_seconds)

        # Add or update cache
        self._cache[key] = (value, expiry_time)
        self._cache.move_to_end(key)  # Move to end to mark as most recently used

        # Periodically clean up expired items
        if time.time() - self._last_cleanup > 60:  # Every minute
            await self._cleanup_expired()

    async def invalidate(self, key: str) -> bool:
        """
        Remove a specific item from the cache.

        Args:
            key: Cache key to remove

        Returns:
            True if key was found and removed, False otherwise
        """
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    async def invalidate_by_pattern(self, pattern: str) -> int:
        """
        Remove items matching a pattern from the cache.

        Args:
            pattern: Regex pattern to match against keys

        Returns:
            Number of keys removed
        """
        keys_to_remove = []
        try:
            regex = re.compile(pattern)

            for key in self._cache.keys():
                if regex.search(key):
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._cache[key]

            return len(keys_to_remove)
        except re.error as e:
            logger.error(f"Invalid regex pattern: {e}")
            return 0

    async def clear(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary of cache statistics
        """
        total_requests = self._hit_stats["hits"] + self._hit_stats["misses"]
        hit_rate = self._hit_stats["hits"] / total_requests if total_requests > 0 else 0

        # Calculate size
        size_items = len(self._cache)

        # Estimate memory usage (rough approximation)
        estimated_memory = sum(
            len(str(key)) + self._estimate_size(value)
            for key, (value, _) in self._cache.items()
        )

        return {
            "size": size_items,
            "max_size": self.max_size,
            "hit_rate": hit_rate,
            "hits": self._hit_stats["hits"],
            "misses": self._hit_stats["misses"],
            "estimated_memory_bytes": estimated_memory,
            "ttl_seconds": self.ttl_seconds,
        }

    def normalize_key(self, query: str, params: dict[str, Any]) -> str:
        """
        Normalize a query and params into a consistent cache key.

        Args:
            query: Search query string
            params: Additional search parameters

        Returns:
            Normalized cache key
        """
        # Normalize query
        normalized_query = query.lower().strip()

        # Sort params by key and serialize to JSON for consistency
        serialized_params = json.dumps(params, sort_keys=True) if params else "{}"

        # Create key
        key = f"{normalized_query}:{serialized_params}"

        # If the key is very long, hash it
        if len(key) > 256:
            key = hashlib.md5(key.encode()).hexdigest()

        return key

    async def _cleanup_expired(self) -> int:
        """
        Remove expired items from the cache.

        Returns:
            Number of items removed
        """
        now = time.time()
        keys_to_remove = []

        for key, (_, expiry_time) in self._cache.items():
            if expiry_time <= now:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._cache[key]

        self._last_cleanup = now
        return len(keys_to_remove)

    def _enforce_size_limit(self) -> int:
        """
        Remove oldest items if cache exceeds max size.

        Returns:
            Number of items removed
        """
        if len(self._cache) <= self.max_size:
            return 0

        # Calculate how many items to remove (30% of max size or at least 1)
        items_to_remove = max(1, int(self.max_size * 0.3))
        items_to_remove = min(items_to_remove, len(self._cache) - self.max_size + 10)

        # Remove oldest items (beginning of OrderedDict)
        for _ in range(items_to_remove):
            self._cache.popitem(last=False)

        return items_to_remove

    def _estimate_size(self, value: Any) -> int:
        """
        Estimate the size in bytes of a Python object.

        Args:
            value: Object to estimate size for

        Returns:
            Estimated size in bytes
        """
        # This is a rough approximation
        if isinstance(value, str | bytes):
            return len(value)
        elif isinstance(value, int | float | bool | None.__class__):
            return 8
        elif isinstance(value, list):
            return sum(self._estimate_size(item) for item in value) + 8
        elif isinstance(value, dict):
            return (
                sum(
                    self._estimate_size(k) + self._estimate_size(v)
                    for k, v in value.items()
                )
                + 8
            )
        else:
            # For other objects, use a reasonable default
            return 64
