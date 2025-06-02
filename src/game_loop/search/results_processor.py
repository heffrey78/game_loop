"""
Search results processor for semantic search.

This module handles post-processing, formatting, and enhancement
of search results for improved quality and relevance.
"""

import logging
import re
from typing import Any

from ..embeddings.entity_registry import EntityEmbeddingRegistry

logger = logging.getLogger(__name__)


class SearchResultsProcessor:
    """Process, format, and enhance search results."""

    def __init__(self, entity_registry: EntityEmbeddingRegistry):
        """
        Initialize the search results processor.

        Args:
            entity_registry: Registry for entity information
        """
        self.entity_registry = entity_registry

    def deduplicate_results(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Remove duplicate entities from results.

        Args:
            results: List of search results

        Returns:
            Deduplicated search results
        """
        seen_ids = set()
        deduplicated = []

        for result in results:
            entity_id = result.get("entity_id")

            if entity_id and entity_id not in seen_ids:
                seen_ids.add(entity_id)
                deduplicated.append(result)

        return deduplicated

    def enrich_results(
        self, results: list[dict[str, Any]], include_fields: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """
        Add additional entity information to search results.

        Args:
            results: Search results to enrich
            include_fields: Specific fields to include in enriched data

        Returns:
            Enriched search results
        """
        enriched_results = []

        for result in results:
            entity_id = result.get("entity_id")

            if not entity_id:
                # Skip results without entity_id
                enriched_results.append(result)
                continue

            # Get complete entity data from registry
            entity_data = self._get_full_entity_data(entity_id)

            if entity_data:
                # Add or update entity data in the result
                if "data" not in result:
                    result["data"] = {}

                # Add or overwrite fields
                if include_fields:
                    for field in include_fields:
                        if field in entity_data:
                            result["data"][field] = entity_data[field]
                else:
                    # Include all fields if no specific fields requested
                    result["data"].update(entity_data)

            enriched_results.append(result)

        return enriched_results

    def calculate_relevance_scores(
        self, results: list[dict[str, Any]], query: str
    ) -> list[dict[str, Any]]:
        """
        Calculate and add relevance scores to results.

        Args:
            results: Search results to score
            query: Original search query

        Returns:
            Search results with relevance scores
        """
        query_keywords = set(query.lower().split())
        scored_results = []

        for result in results:
            # Start with existing score if available
            base_score = result.get("score", 0.5)
            relevance = base_score

            # Get entity data
            entity_data = result.get("data", {})

            # Calculate keyword match relevance
            if entity_data:
                # Fields to check for keyword matches, in order of importance
                fields_to_check = ["name", "title", "description", "content", "text"]
                field_weights = {
                    "name": 1.0,
                    "title": 0.9,
                    "description": 0.7,
                    "content": 0.5,
                    "text": 0.5,
                }

                keyword_boost = 0.0
                matches_found = 0

                for field in fields_to_check:
                    if field in entity_data and isinstance(entity_data[field], str):
                        field_text = entity_data[field].lower()
                        field_weight = field_weights.get(field, 0.5)

                        # Count keyword matches
                        for keyword in query_keywords:
                            if keyword in field_text:
                                matches_found += 1
                                keyword_boost += float(field_weight)

                # Normalize keyword boost
                if matches_found > 0:
                    keyword_relevance = min(1.0, keyword_boost / len(query_keywords))
                    # Blend with base score (70% original score, 30% keyword relevance)
                    relevance = (base_score * 0.7) + (keyword_relevance * 0.3)

            # Update the result with new score
            result["score"] = relevance
            result["relevance_details"] = {
                "base_score": base_score,
                "final_score": relevance,
            }

            scored_results.append(result)

        return scored_results

    def generate_result_snippets(
        self, results: list[dict[str, Any]], query: str
    ) -> list[dict[str, Any]]:
        """
        Generate contextual snippets highlighting match relevance.

        Args:
            results: Search results to generate snippets for
            query: Original search query

        Returns:
            Search results with snippets added
        """
        query_terms = query.lower().split()
        results_with_snippets = []

        for result in results:
            entity_data = result.get("data", {})

            # Check if we have content to create a snippet from
            snippet_source = None
            for field in ["description", "content", "text", "body"]:
                if field in entity_data and isinstance(entity_data[field], str):
                    snippet_source = entity_data[field]
                    break

            if not snippet_source:
                # No content suitable for snippet
                results_with_snippets.append(result)
                continue

            # Find best snippet containing query terms
            snippet = self._extract_best_snippet(snippet_source, query_terms)

            # Highlight matching terms
            highlighted_snippet = self._highlight_terms(snippet, set(query_terms))

            # Add snippet to result
            result["snippet"] = highlighted_snippet
            results_with_snippets.append(result)

        return results_with_snippets

    def group_results_by_type(
        self, results: list[dict[str, Any]]
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Group results by entity type.

        Args:
            results: Search results to group

        Returns:
            Dictionary mapping entity types to results lists
        """
        grouped: dict[str, list[dict[str, Any]]] = {}

        for result in results:
            entity_type = result.get("entity_type", "unknown")

            if entity_type not in grouped:
                grouped[entity_type] = []

            grouped[entity_type].append(result)

        return grouped

    def sort_results(
        self, results: list[dict[str, Any]], sort_by: str = "relevance"
    ) -> list[dict[str, Any]]:
        """
        Sort results by specified criterion.

        Args:
            results: Search results to sort
            sort_by: Criterion to sort by (relevance, name, type, etc.)

        Returns:
            Sorted search results
        """
        if sort_by == "relevance":
            return sorted(results, key=lambda x: x.get("score", 0), reverse=True)

        elif sort_by == "name":
            return sorted(
                results, key=lambda x: x.get("data", {}).get("name", ""), reverse=False
            )

        elif sort_by == "type":
            return sorted(
                results, key=lambda x: x.get("entity_type", "unknown"), reverse=False
            )

        elif sort_by == "created":
            return sorted(
                results,
                key=lambda x: x.get("data", {}).get("created_at", 0),
                reverse=True,
            )

        else:
            # Default to relevance
            return sorted(results, key=lambda x: x.get("score", 0), reverse=True)

    def format_results(
        self, results: list[dict[str, Any]], format_type: str = "detailed"
    ) -> Any:
        """
        Format results according to specified format type.

        Args:
            results: Search results to format
            format_type: Format type (detailed, summary, compact)

        Returns:
            Formatted search results
        """
        if format_type == "detailed":
            return self._format_detailed(results)
        elif format_type == "summary":
            return self._format_summary(results)
        elif format_type == "compact":
            return self._format_compact(results)
        else:
            # Default to detailed
            return self._format_detailed(results)

    def paginate_results(
        self, results: list[dict[str, Any]], page: int = 1, page_size: int = 10
    ) -> dict[str, Any]:
        """
        Paginate results for display.

        Args:
            results: Search results to paginate
            page: Current page number (1-indexed)
            page_size: Number of results per page

        Returns:
            Dictionary with pagination info and current page results
        """
        total_results = len(results)
        total_pages = max(1, (total_results + page_size - 1) // page_size)

        # Ensure page is valid
        page = max(1, min(page, total_pages))

        # Calculate slice indices
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_results)

        # Get results for this page
        page_results = results[start_idx:end_idx]

        return {
            "results": page_results,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_results": total_results,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1,
            },
        }

    def highlight_matching_terms(
        self, results: list[dict[str, Any]], query: str
    ) -> list[dict[str, Any]]:
        """
        Highlight terms in results that match the query.

        Args:
            results: Search results to highlight
            query: Original search query

        Returns:
            Search results with highlighted terms in snippets and fields
        """
        query_terms = set(query.lower().split())
        highlighted_results = []

        for result in results:
            # Copy the result to avoid modifying original
            highlighted_result = dict(result)

            # Highlight terms in snippet if available
            if "snippet" in highlighted_result:
                highlighted_result["snippet"] = self._highlight_terms(
                    highlighted_result["snippet"], query_terms
                )

            # Add highlights to data fields
            if "data" in highlighted_result:
                data = highlighted_result["data"]
                highlighted_data = {}

                # Process name and description fields
                for field in ["name", "title", "description"]:
                    if field in data and isinstance(data[field], str):
                        highlighted_data[field] = data[field]
                        highlighted_data[f"{field}_highlighted"] = (
                            self._highlight_terms(data[field], query_terms)
                        )

                # Add highlighted fields to the result
                highlighted_result["data"].update(highlighted_data)

            highlighted_results.append(highlighted_result)

        return highlighted_results

    def _get_full_entity_data(self, entity_id: str) -> dict[str, Any] | None:
        """
        Get complete entity data from the registry.

        Args:
            entity_id: ID of the entity

        Returns:
            Entity data or None if not found
        """
        # Attempt to get entity data from registry
        if hasattr(self.entity_registry, "get_entity"):
            return self.entity_registry.get_entity(entity_id)  # type: ignore[no-any-return]

        # Fallback if registry doesn't have this method
        if hasattr(self.entity_registry, "get_entity_data"):
            return self.entity_registry.get_entity_data(entity_id)  # type: ignore[no-any-return]

        # If entity registry has entities as an attribute
        if hasattr(self.entity_registry, "entities"):
            entities = self.entity_registry.entities
            if isinstance(entities, dict) and entity_id in entities:
                return entities[entity_id]  # type: ignore[no-any-return]

        return None

    def _extract_best_snippet(
        self, text: str, query_terms: list[str], snippet_length: int = 160
    ) -> str:
        """
        Extract the best text snippet containing query terms.

        Args:
            text: Source text to extract snippet from
            query_terms: Query terms to search for
            snippet_length: Maximum length of snippet

        Returns:
            Best text snippet
        """
        if not text or not query_terms:
            return text[:snippet_length] if text else ""

        text_lower = text.lower()

        # Find all term positions
        term_positions = []
        for term in query_terms:
            positions = [
                m.start()
                for m in re.finditer(r"\b" + re.escape(term) + r"\b", text_lower)
            ]
            term_positions.extend(positions)

        if not term_positions:
            # No terms found, return beginning of text
            return text[:snippet_length]

        # Sort positions
        term_positions.sort()

        # Find position with most terms in range
        best_pos = 0
        max_terms = 0

        for pos in term_positions:
            # Count terms within snippet_length of this position
            terms_in_range = sum(
                1 for p in term_positions if pos <= p < pos + snippet_length
            )

            if terms_in_range > max_terms:
                max_terms = terms_in_range
                best_pos = pos

        # Get snippet with some context before the first match
        context_before = min(50, best_pos)
        start_pos = max(0, best_pos - context_before)

        # Ensure we don't exceed text length
        end_pos = min(len(text), start_pos + snippet_length)

        # Adjust start to include complete words
        if start_pos > 0:
            while start_pos > 0 and text[start_pos - 1].isalnum():
                start_pos -= 1

        # Adjust end to include complete words
        if end_pos < len(text):
            while end_pos < len(text) and text[end_pos].isalnum():
                end_pos += 1

        snippet = text[start_pos:end_pos]

        # Add ellipsis if needed
        if start_pos > 0:
            snippet = "..." + snippet
        if end_pos < len(text):
            snippet = snippet + "..."

        return snippet

    def _highlight_terms(
        self,
        text: str,
        query_terms: set[str],
        prefix: str = "<em>",
        suffix: str = "</em>",
    ) -> str:
        """
        Highlight query terms in text.

        Args:
            text: Text to highlight terms in
            query_terms: Set of terms to highlight
            prefix: String to add before highlighted terms
            suffix: String to add after highlighted terms

        Returns:
            Text with highlighted terms
        """
        if not text or not query_terms:
            return text

        # Create highlighted version
        highlighted = text

        # Use regex to match whole words only
        for term in query_terms:
            pattern = r"\b(" + re.escape(term) + r")\b"
            replacement = prefix + r"\1" + suffix
            highlighted = re.sub(pattern, replacement, highlighted, flags=re.IGNORECASE)

        return highlighted

    def _format_detailed(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Format results with all available details.

        Args:
            results: Search results to format

        Returns:
            Formatted search results
        """
        # For detailed format, we include all information
        formatted = []

        for result in results:
            formatted_result = {
                "entity_id": result.get("entity_id"),
                "entity_type": result.get("entity_type"),
                "score": result.get("score"),
                "data": result.get("data", {}),
            }

            # Add snippet if available
            if "snippet" in result:
                formatted_result["snippet"] = result["snippet"]

            # Add any relevance details
            if "relevance_details" in result:
                formatted_result["relevance_details"] = result["relevance_details"]

            formatted.append(formatted_result)

        return formatted

    def _format_summary(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Format results with summary information.

        Args:
            results: Search results to format

        Returns:
            Formatted search results with summary information
        """
        # For summary format, we include essential fields
        formatted = []

        for result in results:
            data = result.get("data", {})

            formatted_result = {
                "entity_id": result.get("entity_id"),
                "entity_type": result.get("entity_type"),
                "score": result.get("score"),
                "name": data.get("name", ""),
                "description": data.get("description", ""),
            }

            # Add snippet if available
            if "snippet" in result:
                formatted_result["snippet"] = result["snippet"]

            formatted.append(formatted_result)

        return formatted

    def _format_compact(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Format results with minimal information.

        Args:
            results: Search results to format

        Returns:
            Formatted search results with minimal information
        """
        # For compact format, we include only essential information
        formatted = []

        for result in results:
            data = result.get("data", {})

            formatted_result = {
                "entity_id": result.get("entity_id"),
                "entity_type": result.get("entity_type"),
                "score": result.get("score"),
                "name": data.get("name", ""),
            }

            formatted.append(formatted_result)

        return formatted
