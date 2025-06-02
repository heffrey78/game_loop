"""
Search API endpoints for semantic search.

This module provides RESTful API endpoints for semantic search capabilities.
"""

import logging
from typing import Any

from fastapi import (  # type: ignore[import-not-found]
    APIRouter,
    HTTPException,
    Query,
    Request,
)

from ...search.query_processor import QueryProcessor
from ...search.results_processor import SearchResultsProcessor
from ...search.semantic_search import SemanticSearchService

logger = logging.getLogger(__name__)


class SearchEndpoints:
    """Expose search functionality through REST API endpoints."""

    def __init__(self, search_service: SemanticSearchService):
        """
        Initialize the search API endpoints.

        Args:
            search_service: Semantic search service for executing searches
        """
        self.search_service = search_service
        self.router = APIRouter(prefix="/api/search", tags=["search"])
        self._register_routes()

    def _register_routes(self) -> None:
        """Register all API routes."""
        self.router.add_api_route(
            "",
            self.search_entities,
            methods=["GET"],
            response_model=dict[str, Any],
            summary="Search entities",
            description="Search for game entities using text queries",
        )

        self.router.add_api_route(
            "/similar/{entity_id}",
            self.search_by_similarity,
            methods=["GET"],
            response_model=dict[str, Any],
            summary="Find similar entities",
            description="Find entities similar to a reference entity",
        )

        self.router.add_api_route(
            "/context/{entity_id}",
            self.get_entity_context,
            methods=["GET"],
            response_model=dict[str, Any],
            summary="Get entity context",
            description="Get contextually relevant entities for a specific entity",
        )

        self.router.add_api_route(
            "/by-example",
            self.search_by_example,
            methods=["POST"],
            response_model=dict[str, Any],
            summary="Search by example",
            description="Search using an example entity as the query",
        )

        self.router.add_api_route(
            "/suggestions",
            self.get_search_suggestions,
            methods=["GET"],
            response_model=dict[str, Any],
            summary="Get search suggestions",
            description="Get search query suggestions based on partial input",
        )

        self.router.add_api_route(
            "/recent",
            self.get_recent_searches,
            methods=["GET"],
            response_model=dict[str, Any],
            summary="Get recent searches",
            description="Get user's recent searches",
        )

    async def search_entities(
        self,
        request: Request,
        query: str = Query(..., description="Search query text"),
        entity_types: str | None = Query(
            None, description="Comma-separated list of entity types to search"
        ),
        strategy: str = Query(
            "hybrid", description="Search strategy (semantic, keyword, hybrid, exact)"
        ),
        top_k: int = Query(10, description="Maximum number of results to return"),
        threshold: float = Query(0.7, description="Minimum similarity threshold"),
        page: int = Query(1, description="Page number for pagination"),
        page_size: int = Query(10, description="Number of results per page"),
        format: str = Query(
            "detailed", description="Result format (detailed, summary, compact)"
        ),
    ) -> dict[str, Any]:
        """
        Main search endpoint for entities.

        Args:
            request: HTTP request
            query: Search query text
            entity_types: Optional comma-separated list of entity types to filter by
            strategy: Search strategy to use
            top_k: Maximum number of results
            threshold: Minimum similarity threshold
            page: Page number for pagination
            page_size: Number of results per page
            format: Result format type

        Returns:
            Search results with pagination info
        """
        try:
            # Parse parameters
            entity_type_list = None
            if entity_types:
                entity_type_list = [t.strip() for t in entity_types.split(",")]

            # Perform search
            results = await self.search_service.search(
                query=query,
                entity_types=entity_type_list if entity_type_list else [],
                strategy=strategy,
                top_k=top_k,
                threshold=threshold,
            )

            # Get results processor to format and paginate
            results_processor = SearchResultsProcessor(self.search_service.registry)

            # Format results
            formatted_results = results_processor.format_results(results, format)

            # Paginate results
            paginated_results = results_processor.paginate_results(
                formatted_results, page=page, page_size=page_size
            )

            # Add search metadata
            response = {
                **paginated_results,
                "metadata": {
                    "query": query,
                    "strategy": strategy,
                    "entity_types": entity_type_list,
                    "threshold": threshold,
                },
            }

            return response

        except Exception as e:
            logger.error(f"Error in search_entities: {e}")
            raise HTTPException(
                status_code=500, detail=f"Search error: {str(e)}"
            ) from e

    async def search_by_similarity(
        self,
        request: Request,
        entity_id: str,
        top_k: int = Query(10, description="Maximum number of results to return"),
        threshold: float = Query(0.7, description="Minimum similarity threshold"),
        entity_types: str | None = Query(
            None, description="Comma-separated list of entity types to search"
        ),
        page: int = Query(1, description="Page number for pagination"),
        page_size: int = Query(10, description="Number of results per page"),
    ) -> dict[str, Any]:
        """
        Search entities similar to a reference entity.

        Args:
            request: HTTP request
            entity_id: Reference entity ID
            top_k: Maximum number of results
            threshold: Minimum similarity threshold
            entity_types: Optional comma-separated list of entity types to filter by
            page: Page number for pagination
            page_size: Number of results per page

        Returns:
            Similar entities with similarity scores
        """
        try:
            # Get similarity analyzer
            from ...search.similarity import EntitySimilarityAnalyzer

            similarity_analyzer = EntitySimilarityAnalyzer(
                self.search_service.registry, self.search_service.db_manager
            )

            # Parse entity types
            entity_type_list = None
            if entity_types:
                entity_type_list = [t.strip() for t in entity_types.split(",")]

            # Find similar entities
            similar_entities = await similarity_analyzer.find_similar_entities(
                entity_id=entity_id, top_k=top_k, min_similarity=threshold
            )

            # Convert to results format
            results = []
            for similar_id, score in similar_entities:
                # Retrieve entity data
                entity_data = self._get_entity_data(similar_id)
                entity_type = (
                    entity_data.get("entity_type", "unknown")
                    if entity_data
                    else "unknown"
                )

                # Filter by entity type if specified
                if entity_type_list and entity_type not in entity_type_list:
                    continue

                results.append(
                    {
                        "entity_id": similar_id,
                        "entity_type": entity_type,
                        "score": score,
                        "data": entity_data or {"entity_id": similar_id},
                    }
                )

            # Get results processor to paginate
            results_processor = SearchResultsProcessor(self.search_service.registry)

            # Paginate results
            paginated_results = results_processor.paginate_results(
                results, page=page, page_size=page_size
            )

            # Add metadata
            response = {
                **paginated_results,
                "metadata": {
                    "reference_entity": entity_id,
                    "threshold": threshold,
                    "entity_types": entity_type_list,
                },
            }

            return response

        except Exception as e:
            logger.error(f"Error in search_by_similarity: {e}")
            raise HTTPException(
                status_code=500, detail=f"Similarity search error: {str(e)}"
            ) from e

    async def get_entity_context(
        self,
        request: Request,
        entity_id: str,
        context_types: str | None = Query(
            "related,similar", description="Comma-separated list of context types"
        ),
        top_k: int = Query(5, description="Maximum number of results per context type"),
    ) -> dict[str, Any]:
        """
        Get contextually relevant entities for a specific entity.

        Args:
            request: HTTP request
            entity_id: Entity ID to get context for
            context_types: Types of context to include
            top_k: Maximum number of results per context type

        Returns:
            Different types of contextually relevant entities
        """
        try:
            # Parse context types
            context_type_list = (
                [t.strip() for t in context_types.split(",")]
                if context_types
                else ["related", "similar"]
            )

            # Get entity data
            entity_data = self._get_entity_data(entity_id)
            if not entity_data:
                raise HTTPException(
                    status_code=404, detail=f"Entity not found: {entity_id}"
                )

            context_results = {}

            # Get similar entities if requested
            if "similar" in context_type_list:
                from ...search.similarity import EntitySimilarityAnalyzer

                similarity_analyzer = EntitySimilarityAnalyzer(
                    self.search_service.registry, self.search_service.db_manager
                )

                similar = await similarity_analyzer.find_similar_entities(
                    entity_id=entity_id, top_k=top_k, min_similarity=0.7
                )

                context_results["similar"] = [
                    {
                        "entity_id": similar_id,
                        "similarity": score,
                        "data": self._get_entity_data(similar_id),
                    }
                    for similar_id, score in similar
                ]

            # Get related entities if requested
            if "related" in context_type_list:
                # This would be based on your game's entity relationships
                # For now, we'll provide a placeholder implementation
                related = await self._get_related_entities(entity_id, top_k)
                context_results["related"] = related

            # Get parent/child entities if requested and applicable
            if "hierarchy" in context_type_list:
                hierarchy = await self._get_entity_hierarchy(entity_id)
                context_results = hierarchy

            return {
                "entity_id": entity_id,
                "entity_type": entity_data.get("entity_type", "unknown"),
                "contexts": context_results,
            }

        except Exception as e:
            logger.error(f"Error in get_entity_context: {e}")
            raise HTTPException(
                status_code=500, detail=f"Context retrieval error: {str(e)}"
            ) from e

    async def search_by_example(self, request: Request) -> dict[str, Any]:
        """
        Search using an example entity as the query.

        Args:
            request: HTTP request containing example entity data

        Returns:
            Search results based on example entity
        """
        try:
            # Get example entity from request body
            example = await request.json()

            # Extract relevant text from example to form query
            query_text = self._extract_query_from_example(example)

            # Define entity types to search based on example
            entity_type = example.get("entity_type")
            entity_type_list = [entity_type] if entity_type else []

            # Search using example-derived query
            results = await self.search_service.search(
                query=query_text,
                entity_types=entity_type_list,
                strategy="hybrid",
                top_k=10,
                threshold=0.6,  # Lower threshold for example searches
            )

            return {
                "results": results,
                "metadata": {"derived_query": query_text, "entity_type": entity_type},
            }
        except Exception as e:
            logger.error(f"Error in search_by_example: {e}")
            raise HTTPException(
                status_code=500, detail=f"Example search error: {str(e)}"
            ) from e

    async def get_search_suggestions(
        self,
        request: Request,
        partial_query: str = Query(..., description="Partial search query"),
        max_suggestions: int = Query(
            5, description="Maximum number of suggestions to return"
        ),
    ) -> dict[str, Any]:
        """
        Get search query suggestions based on partial input.

        Args:
            request: HTTP request
            partial_query: Partial search query
            max_suggestions: Maximum number of suggestions

        Returns:
            Query suggestions
        """
        try:
            # This is a placeholder implementation
            # In a real system, you might use historical queries, popular entities, etc.

            # Get query processor
            query_processor = QueryProcessor(self.search_service.registry)

            # Generate query variations (uncomment if needed later)
            # expanded = query_processor.expand_query(partial_query)

            # Create suggestions
            suggestions = []

            # Add auto-completion suggestions based on entity names
            entity_types = query_processor.extract_entity_types(partial_query)
            if entity_types:
                # Get entities of the detected types
                for entity_type in entity_types:
                    type_entities = self._get_entities_by_type(
                        entity_type, max_suggestions
                    )

                    for entity in type_entities:
                        if "name" in entity and entity["name"].lower().startswith(
                            partial_query.lower()
                        ):
                            suggestions.append(entity["name"])

            # Add common query patterns
            common_patterns = [
                f"find {partial_query}",
                f"{partial_query} near",
                f"similar to {partial_query}",
                f"{partial_query} with high",
            ]

            suggestions.extend(common_patterns)

            # Ensure we return unique suggestions up to max_suggestions
            unique_suggestions = list(dict.fromkeys(suggestions))

            return {
                "partial_query": partial_query,
                "suggestions": unique_suggestions[:max_suggestions],
            }
        except Exception as e:
            logger.error(f"Error in get_search_suggestions: {e}")
            raise HTTPException(
                status_code=500, detail=f"Suggestion error: {str(e)}"
            ) from e

    async def get_recent_searches(
        self,
        request: Request,
        max_results: int = Query(
            10, description="Maximum number of recent searches to return"
        ),
    ) -> dict[str, Any]:
        """
        Get user's recent searches.

        Args:
            request: HTTP request
            max_results: Maximum number of recent searches

        Returns:
            Recent search queries
        """
        try:
            # This is a placeholder implementation
            # In a real system, you would store and retrieve user's recent searches
            recent_searches: list[dict[str, Any]] = []
            return {"recent_searches": recent_searches, "count": len(recent_searches)}
        except Exception as e:
            logger.error(f"Error in get_recent_searches: {e}")
            raise HTTPException(
                status_code=500, detail=f"Recent searches error: {str(e)}"
            ) from e

    def _parse_search_params(self, request: Request) -> dict[str, Any]:
        """
        Parse and validate search parameters from request.

        Args:
            request: HTTP request

        Returns:
            Dictionary of parsed parameters
        """
        # This would be implemented based on your API framework
        # For now, we'll provide a placeholder implementation
        return {}

    def _format_search_response(
        self, results: list[dict[str, Any]], format_type: str = "detailed"
    ) -> dict[str, Any]:
        """
        Format search results for API response.

        Args:
            results: Search results
            format_type: Format type (detailed, summary, compact)

        Returns:
            Formatted search results
        """
        # Use the SearchResultsProcessor to format results
        results_processor = SearchResultsProcessor(self.search_service.registry)
        return results_processor.format_results(results, format_type)  # type: ignore[no-any-return]

    def _get_entity_data(self, entity_id: str) -> dict[str, Any] | None:
        """
        Get entity data for an entity ID.

        Args:
            entity_id: ID of the entity

        Returns:
            Entity data or None if not found
        """
        # Try to get from the registry
        if hasattr(self.search_service.registry, "get_entity"):
            return self.search_service.registry.get_entity(entity_id)  # type: ignore[no-any-return]

        return None

    def _extract_query_from_example(self, example: dict[str, Any]) -> str:
        """
        Extract a search query from an example entity.

        Args:
            example: Example entity data

        Returns:
            Extracted query string
        """
        query_parts = []

        # Include name if available
        if "name" in example:
            query_parts.append(example["name"])

        # Include description if available
        if "description" in example and len(example["description"]) < 100:
            query_parts.append(example["description"])
        elif "description" in example:
            # Take just the beginning of long descriptions
            query_parts.append(example["description"][:100])

        # Include key attributes
        for key in ["type", "category", "class", "role", "function"]:
            if key in example and isinstance(example[key], str):
                query_parts.append(example[key])

        # If we don't have anything useful, use the whole example
        if not query_parts:
            query = " ".join(
                str(v) for v in example.values() if isinstance(v, str | int | float)
            )
            return query[:200]  # Limit length

        return " ".join(query_parts)

    async def _get_related_entities(
        self, entity_id: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """
        Get entities related to the given entity.

        Args:
            entity_id: Entity ID
            limit: Maximum number of related entities

        Returns:
            List of related entities with relationship info
        """
        # This is a placeholder implementation
        # In a real system, this would query your game's relationship graph
        return []

    async def _get_entity_hierarchy(self, entity_id: str) -> dict[str, Any]:
        """
        Get parent/child relationship hierarchy for an entity.

        Args:
            entity_id: Entity ID

        Returns:
            Hierarchy information
        """
        # This is a placeholder implementation
        # In a real system, this would query your game's entity hierarchy
        return {"parents": [], "children": []}

    def _get_entities_by_type(
        self, entity_type: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """
        Get entities of a specific type.

        Args:
            entity_type: Entity type
            limit: Maximum number of entities

        Returns:
            List of entities
        """
        # This is a placeholder implementation
        # In a real system, this would query your entity registry or database
        return []
