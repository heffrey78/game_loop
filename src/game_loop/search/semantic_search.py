"""
Main semantic search service for game entities.

This module provides the primary interface for performing semantic searches
across game entities using various search strategies.
"""

import logging
import time
from collections import OrderedDict
from collections import OrderedDict as OrderedDictType
from typing import Any

from ..database.managers.embedding_manager import EmbeddingDatabaseManager
from ..embeddings.entity_embeddings import EntityEmbeddingGenerator
from ..embeddings.entity_registry import EntityEmbeddingRegistry

logger = logging.getLogger(__name__)


class SemanticSearchService:
    """Centralized service for all semantic search operations."""

    def __init__(
        self,
        embedding_db_manager: EmbeddingDatabaseManager,
        embedding_registry: EntityEmbeddingRegistry,
        embedding_generator: EntityEmbeddingGenerator,
        cache_size: int = 1000,
    ):
        """
        Initialize the semantic search service.

        Args:
            embedding_db_manager: Manager for database operations with embeddings
            embedding_registry: Registry for in-memory entity embeddings
            embedding_generator: Generator for creating embeddings from text
            cache_size: Maximum size of the results cache
        """
        self.db_manager = embedding_db_manager
        self.registry = embedding_registry
        self.generator = embedding_generator
        self._results_cache: OrderedDictType[str, list[dict[str, Any]]] = (
            OrderedDict()
        )  # LRU cache for search results
        self._cache_size = cache_size
        self._search_metrics: dict[str, Any] = {
            "total_searches": 0,
            "cache_hits": 0,
            "avg_search_time": 0.0,
            "search_times_by_strategy": {
                "semantic": [],
                "keyword": [],
                "hybrid": [],
                "exact": [],
            },
        }

    async def search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        strategy: str = "hybrid",
        top_k: int = 10,
        threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """
        Primary search method supporting multiple strategies.

        Args:
            query: The search query string
            entity_types: Limit search to these entity types
            strategy: Search strategy to use (semantic, keyword, hybrid, exact)
            top_k: Maximum number of results to return
            threshold: Minimum similarity score threshold

        Returns:
            List of search results with relevance scores

        Strategies:
        - "semantic": Pure vector similarity search
        - "keyword": Traditional keyword matching
        - "hybrid": Combined semantic and keyword search
        - "exact": Exact match on entity attributes
        """
        start_time = time.time()
        total_searches = self._search_metrics["total_searches"]
        if isinstance(total_searches, int | float):
            self._search_metrics["total_searches"] = total_searches + 1

        # Check cache first
        cache_key = self._create_cache_key(
            query, entity_types, strategy, top_k, threshold
        )
        cached_result = self._get_from_cache(cache_key)
        if cached_result:
            cache_hits = self._search_metrics["cache_hits"]
            if isinstance(cache_hits, int | float):
                self._search_metrics["cache_hits"] = cache_hits + 1
            return cached_result

        # Preprocess query
        query = self._preprocess_query(query)

        # Select best search strategy if "auto" is specified
        actual_strategy = self._select_search_strategy(query, strategy)

        # Execute search based on strategy
        if actual_strategy == "semantic":
            results = await self.semantic_search(query, entity_types, top_k, threshold)
        elif actual_strategy == "keyword":
            results = await self.keyword_search(query, entity_types, top_k)
        elif actual_strategy == "hybrid":
            results = await self.hybrid_search(
                query, entity_types, top_k, 0.7
            )  # 0.7 weight to semantic
        elif actual_strategy == "exact":
            # For exact match, we need to know which field to match on
            # Default to name if not specified in query
            field = "name"
            results = await self.exact_match_search(query, field, entity_types)
        else:
            logger.warning(
                f"Unknown search strategy: {strategy}, falling back to hybrid"
            )
            results = await self.hybrid_search(query, entity_types, top_k)

        # Add to cache
        self._cache_results(cache_key, results)

        # Update metrics
        search_time = time.time() - start_time
        self._update_metrics(search_time, actual_strategy)

        return results

    async def semantic_search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        top_k: int = 10,
        threshold: float = 0.7,
    ) -> list[dict[str, Any]]:
        """
        Perform semantic search using vector embeddings.

        Args:
            query: The search query string
            entity_types: Limit search to these entity types
            top_k: Maximum number of results to return
            threshold: Minimum similarity score threshold

        Returns:
            List of search results with relevance scores
        """
        # Generate embedding for the query
        query_embedding = await self._generate_query_embedding(query)

        # Perform similarity search against database
        try:
            if entity_types:
                results = []
                for entity_type in entity_types:
                    # Get embeddings by entity type
                    entity_embeddings = (
                        await self.db_manager.get_embeddings_by_entity_type(entity_type)
                    )
                    # Calculate similarities
                    entity_results = self._calculate_similarities(
                        query_embedding, entity_embeddings
                    )
                    results.extend(entity_results)
            else:
                # Search across all entity types
                all_embeddings = {}
                # Retrieve from registry or database based on availability
                if self.registry and hasattr(self.registry, "embeddings"):
                    all_embeddings = dict(self.registry.embeddings)
                else:
                    # We need to implement this method in the database manager
                    # to get all embeddings across entity types
                    all_embeddings = await self.db_manager.get_all_embeddings()

                results = self._calculate_similarities(query_embedding, all_embeddings)

            # Filter, rank, and format results
            results = self._filter_by_threshold(results, threshold)
            results = self._rank_results(results, query)
            return results[:top_k]

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    async def keyword_search(
        self, query: str, entity_types: list[str] | None = None, top_k: int = 10
    ) -> list[dict[str, Any]]:
        """
        Perform keyword-based search.

        Args:
            query: The search query string
            entity_types: Limit search to these entity types
            top_k: Maximum number of results to return

        Returns:
            List of search results with relevance scores
        """
        # This is a basic implementation. In a real system, you might
        # use a full-text search engine or more sophisticated techniques.
        keywords = query.lower().split()
        results = []

        try:
            # Get entities from registry or database
            entities = await self._get_searchable_entities(entity_types)

            # Calculate keyword match score for each entity
            for entity_id, entity_data in entities.items():
                score = self._calculate_keyword_match_score(entity_data, keywords)
                if score > 0:
                    results.append(
                        {
                            "entity_id": entity_id,
                            "entity_type": entity_data.get("entity_type", "unknown"),
                            "score": score,
                            "data": entity_data,
                        }
                    )

            # Rank results
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []

    async def hybrid_search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        top_k: int = 10,
        semantic_weight: float = 0.7,
    ) -> list[dict[str, Any]]:
        """
        Combine results from semantic and keyword search with weighting.

        Args:
            query: The search query string
            entity_types: Limit search to these entity types
            top_k: Maximum number of results to return
            semantic_weight: Weight to give to semantic results (0-1)

        Returns:
            List of search results with combined relevance scores
        """
        # Run both searches
        semantic_results = await self.semantic_search(query, entity_types, top_k * 2)
        keyword_results = await self.keyword_search(query, entity_types, top_k * 2)

        # Combine results with weighting
        combined_scores = {}

        for result in semantic_results:
            entity_id = result["entity_id"]
            combined_scores[entity_id] = {
                "entity_id": entity_id,
                "entity_type": result["entity_type"],
                "semantic_score": result["score"],
                "keyword_score": 0,
                "data": result["data"],
            }

        for result in keyword_results:
            entity_id = result["entity_id"]
            if entity_id in combined_scores:
                combined_scores[entity_id]["keyword_score"] = result["score"]
            else:
                combined_scores[entity_id] = {
                    "entity_id": entity_id,
                    "entity_type": result["entity_type"],
                    "semantic_score": 0,
                    "keyword_score": result["score"],
                    "data": result["data"],
                }

        # Calculate combined scores
        results = []
        for entity_id, scores in combined_scores.items():
            combined_score = (
                semantic_weight * scores["semantic_score"]
                + (1 - semantic_weight) * scores["keyword_score"]
            )

            results.append(
                {
                    "entity_id": entity_id,
                    "entity_type": scores["entity_type"],
                    "score": combined_score,
                    "semantic_score": scores["semantic_score"],
                    "keyword_score": scores["keyword_score"],
                    "data": scores["data"],
                }
            )

        # Rank and return top results
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    async def exact_match_search(
        self, query: str, field: str, entity_types: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """
        Find exact matches for a specific field.

        Args:
            query: Value to match
            field: Field name to match on
            entity_types: Limit search to these entity types

        Returns:
            List of exact matches
        """
        results = []

        try:
            # Get entities from registry or database
            entities = await self._get_searchable_entities(entity_types)

            # Find exact matches
            for entity_id, entity_data in entities.items():
                if (
                    field in entity_data
                    and str(entity_data[field]).lower() == query.lower()
                ):
                    results.append(
                        {
                            "entity_id": entity_id,
                            "entity_type": entity_data.get("entity_type", "unknown"),
                            "score": 1.0,  # Exact match has perfect score
                            "data": entity_data,
                        }
                    )

            return results

        except Exception as e:
            logger.error(f"Error in exact match search: {e}")
            return []

    def _preprocess_query(self, query: str) -> str:
        """
        Clean and normalize the search query.

        Args:
            query: The search query string

        Returns:
            Preprocessed query string
        """
        # Simple preprocessing for now
        # Remove extra whitespace
        query = " ".join(query.split())
        # Convert to lowercase
        query = query.lower()

        return query

    def _select_search_strategy(self, query: str, strategy: str) -> str:
        """
        Determine the best search strategy based on the query and specified strategy.

        Args:
            query: The search query string
            strategy: Requested strategy or "auto"

        Returns:
            Selected strategy name
        """
        if strategy != "auto":
            return strategy

        # Auto-select based on query characteristics
        words = query.split()

        # If query is very short, favor exact match
        if len(words) == 1 and len(query) < 20:
            return "exact"
        # If query contains quotes, it might be looking for exact phrases
        elif '"' in query:
            return "keyword"
        # If query is longer, semantic search might be better
        elif len(words) > 3 or len(query) > 30:
            return "semantic"
        # Default to hybrid for balanced approach
        else:
            return "hybrid"

    async def _generate_query_embedding(self, query: str) -> list[float]:
        """
        Generate embedding for the search query.

        Args:
            query: The search query string

        Returns:
            Vector embedding for the query
        """
        try:
            # Use the embedding generator to create an embedding for the query
            embedding = await self.generator.generate_entity_embedding(
                {"name": query, "description": query}, "general"
            )
            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            # Return a zero embedding as fallback (not ideal but prevents crashing)
            # Ideally, the embedding dimension should be retrieved from the generator
            embedding_dim = getattr(self.generator, "embedding_dimension", 384)
            return [0.0] * embedding_dim

    def _rank_results(
        self, results: list[dict[str, Any]], query: str
    ) -> list[dict[str, Any]]:
        """
        Rank and sort search results by relevance.

        Args:
            results: Search results to rank
            query: Original search query

        Returns:
            Ranked search results
        """
        # For now, simply sort by score which should already be calculated
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def _filter_by_threshold(
        self, results: list[dict[str, Any]], threshold: float
    ) -> list[dict[str, Any]]:
        """
        Filter results that don't meet the minimum similarity threshold.

        Args:
            results: Search results to filter
            threshold: Minimum score threshold

        Returns:
            Filtered search results
        """
        return [result for result in results if result["score"] >= threshold]

    def _cache_results(self, cache_key: str, results: list[dict[str, Any]]) -> None:
        """
        Cache search results for future queries.

        Args:
            cache_key: Cache key for the query and parameters
            results: Search results to cache
        """
        # Implement simple LRU cache
        if len(self._results_cache) >= self._cache_size:
            # Remove oldest item
            self._results_cache.popitem(last=False)

        # Add new results to cache
        self._results_cache[cache_key] = results

    def _get_from_cache(self, cache_key: str) -> list[dict[str, Any]] | None:
        """
        Retrieve results from cache if available.

        Args:
            cache_key: Cache key for the query and parameters

        Returns:
            Cached results or None if not in cache
        """
        if cache_key in self._results_cache:
            # Move to end to mark as recently used
            results = self._results_cache.pop(cache_key)
            self._results_cache[cache_key] = results
            return results
        return None

    def _create_cache_key(
        self,
        query: str,
        entity_types: list[str] | None,
        strategy: str,
        top_k: int,
        threshold: float,
    ) -> str:
        """
        Create a unique cache key for the search parameters.

        Args:
            query: Search query
            entity_types: Entity types to search
            strategy: Search strategy
            top_k: Maximum results
            threshold: Similarity threshold

        Returns:
            Cache key string
        """
        # Normalize input for consistent keys
        normalized_query = query.lower().strip()
        entity_types_str = ",".join(sorted(entity_types)) if entity_types else "all"

        return f"{normalized_query}|{entity_types_str}|{strategy}|{top_k}|{threshold}"

    def get_search_metrics(self) -> dict[str, Any]:
        """
        Return performance metrics for recent searches.

        Returns:
            Dictionary of search metrics
        """
        # Calculate average search times by strategy
        avg_times = {}
        search_times_by_strategy = self._search_metrics["search_times_by_strategy"]
        if isinstance(search_times_by_strategy, dict):
            for strategy, times in search_times_by_strategy.items():
                if (
                    times
                    and isinstance(times, list)
                    and all(isinstance(t, int | float) for t in times)
                ):
                    avg_times[strategy] = sum(times) / len(times)
                else:
                    avg_times[strategy] = 0

        # Return aggregated metrics
        return {
            "total_searches": self._search_metrics["total_searches"],
            "cache_hits": self._search_metrics["cache_hits"],
            "cache_hit_rate": (
                (
                    float(self._search_metrics["cache_hits"])
                    / float(self._search_metrics["total_searches"])
                )
                if (
                    isinstance(self._search_metrics["total_searches"], int | float)
                    and isinstance(self._search_metrics["cache_hits"], int | float)
                    and self._search_metrics["total_searches"] > 0
                )
                else 0.0
            ),
            "avg_search_time": self._search_metrics["avg_search_time"],
            "avg_times_by_strategy": avg_times,
        }

    def _update_metrics(self, search_time: float, strategy: str) -> None:
        """
        Update performance metrics for this search.

        Args:
            search_time: Time taken for this search in seconds
            strategy: Search strategy used
        """
        # Update total average
        old_avg = self._search_metrics["avg_search_time"]
        old_count = self._search_metrics["total_searches"]

        # Update the running average
        if isinstance(old_count, int | float) and isinstance(old_avg, int | float):
            if old_count > 1:  # Avoid division by zero for first search
                self._search_metrics["avg_search_time"] = (
                    old_avg * (old_count - 1) + search_time
                ) / old_count
            else:
                self._search_metrics["avg_search_time"] = search_time

        # Update strategy-specific times
        search_times_by_strategy = self._search_metrics["search_times_by_strategy"]
        if (
            isinstance(search_times_by_strategy, dict)
            and strategy in search_times_by_strategy
        ):
            strategy_times = search_times_by_strategy[strategy]
            if isinstance(strategy_times, list):
                strategy_times.append(search_time)
                # Keep only the last 100 searches for each strategy
                if len(strategy_times) > 100:
                    strategy_times.pop(0)

    def _calculate_similarities(
        self, query_embedding: list[float], entity_embeddings: dict[str, list[float]]
    ) -> list[dict[str, Any]]:
        """
        Calculate similarities between a query embedding and entity embeddings.

        Args:
            query_embedding: Query vector embedding
            entity_embeddings: Dictionary mapping entity IDs to embeddings

        Returns:
            List of entities with similarity scores
        """
        results = []

        for entity_id, embedding in entity_embeddings.items():
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, embedding)

            # Get entity data
            entity_data = self._get_entity_data(entity_id)
            entity_type = (
                entity_data.get("entity_type", "unknown") if entity_data else "unknown"
            )

            results.append(
                {
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "score": similarity,
                    "data": entity_data or {"entity_id": entity_id},
                }
            )

        return results

    def _cosine_similarity(self, v1: list[float], v2: list[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            v1: First vector
            v2: Second vector

        Returns:
            Cosine similarity (-1 to 1, where 1 is identical)
        """
        # Simple implementation of cosine similarity
        # In practice, you might want to use a library like numpy for performance

        # Check if vectors are same length
        if len(v1) != len(v2):
            logger.warning(f"Vectors have different dimensions: {len(v1)} vs {len(v2)}")
            min_len = min(len(v1), len(v2))
            v1 = v1[:min_len]
            v2 = v2[:min_len]

        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(v1, v2, strict=False))

        # Calculate magnitudes
        magnitude1 = sum(a * a for a in v1) ** 0.5
        magnitude2 = sum(b * b for b in v2) ** 0.5

        # Handle zero magnitudes
        if magnitude1 == 0 or magnitude2 == 0:
            return 0

        return float(dot_product / (magnitude1 * magnitude2))

    def _calculate_keyword_match_score(
        self, entity_data: dict[str, Any], keywords: list[str]
    ) -> float:
        """
        Calculate keyword match score for an entity.

        Args:
            entity_data: Entity data dictionary
            keywords: List of lowercase keywords to match

        Returns:
            Keyword match score (0 to 1)
        """
        if not entity_data or not keywords:
            return 0

        # Fields to search in order of importance
        searchable_fields = ["name", "title", "description", "content", "text"]

        # Collect text from the entity
        entity_text = ""
        for field in searchable_fields:
            if field in entity_data and isinstance(entity_data[field], str):
                entity_text += " " + entity_data[field].lower()

        if not entity_text:
            # No searchable text found
            return 0

        # Count matching keywords
        matched_keywords = 0
        total_keywords = len(keywords)

        for keyword in keywords:
            if keyword in entity_text:
                matched_keywords += 1

        # Return proportion of keywords matched
        return matched_keywords / total_keywords if total_keywords > 0 else 0

    async def _get_searchable_entities(
        self, entity_types: list[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """
        Get entities that can be searched, optionally filtered by type.

        Args:
            entity_types: List of entity types to include

        Returns:
            Dictionary of entity_id to entity data
        """
        # This is a placeholder implementation
        # In a real system, this would likely involve database queries

        # Try to get entities from the registry first
        entities = {}

        # Implement based on the game's entity retrieval system
        # This is just a placeholder
        if hasattr(self.registry, "get_entities"):
            entities = await self.registry.get_entities(entity_types)

        # If registry doesn't have entities, try to get them from database
        # This would require additional implementation in the database manager

        return entities

    def _get_entity_data(self, entity_id: str) -> dict[str, Any] | None:
        """
        Get entity data for an entity ID.

        Args:
            entity_id: ID of the entity

        Returns:
            Entity data dictionary or None if not found
        """
        # This is a placeholder implementation
        # In a real system, this would likely involve registry or database lookup

        # Try to get entity from the registry first
        if hasattr(self.registry, "metadata") and entity_id in self.registry.metadata:
            return dict(self.registry.metadata[entity_id])

        # If not in registry, we might need to query the database
        # This would require additional implementation

        return None
