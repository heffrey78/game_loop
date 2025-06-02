"""
Query processor module for semantic search.

This module provides preprocessing, analysis, and optimization
of search queries before they are executed.
"""

import logging
import re
import unicodedata
from typing import Any

from ..embeddings.entity_registry import EntityEmbeddingRegistry

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Handle query preprocessing, analysis, and transformation for optimal search."""

    def __init__(
        self, entity_registry: EntityEmbeddingRegistry, nlp_processor: Any = None
    ) -> None:
        """
        Initialize the query processor.

        Args:
            entity_registry: Registry for entity embeddings and metadata
            nlp_processor: Optional NLP processor for advanced query analysis
        """
        self.entity_registry = entity_registry
        self.nlp_processor = nlp_processor
        self._entity_type_patterns: dict[str, list[str]] = {}
        self._stop_words: set[str] = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "if",
            "because",
            "as",
            "what",
            "when",
            "where",
            "how",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "then",
            "just",
            "so",
            "than",
            "such",
            "both",
            "through",
            "about",
            "for",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "not",
            "no",
            "nor",
            "on",
            "in",
            "to",
            "from",
            "with",
            "by",
            "at",
            "of",
        }
        self._commands = {
            "find": "lookup",
            "search": "lookup",
            "get": "lookup",
            "show": "lookup",
            "list": "list",
            "compare": "comparison",
            "similar": "similarity",
            "like": "similarity",
            "related": "related",
        }
        self._initialize_patterns()

    def normalize_query(self, query: str) -> str:
        """
        Clean and normalize the query text.

        Args:
            query: The search query string

        Returns:
            Normalized query string
        """
        # Convert to lowercase
        query = query.lower()

        # Normalize unicode characters
        query = unicodedata.normalize("NFKD", query)

        # Remove extra whitespace
        query = " ".join(query.split())

        # Remove special characters except quotes (keep them for phrase searches)
        query = re.sub(r'[^\w\s"\']', " ", query)

        # Remove extra whitespace again after character removal
        query = " ".join(query.split())

        return query

    def extract_entity_types(self, query: str) -> list[str]:
        """
        Identify potential entity types mentioned in the query.

        Args:
            query: The search query string

        Returns:
            List of entity types
        """
        entity_types = []

        # Check for explicit type indicators
        for entity_type, patterns in self._entity_type_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query.lower()):
                    entity_types.append(entity_type)

        # If no explicit types found, check for known entity types in the registry
        if not entity_types and hasattr(self.entity_registry, "get_entity_types"):
            all_entity_types = self.entity_registry.get_entity_types()
            for entity_type in all_entity_types:
                if entity_type.lower() in query.lower():
                    entity_types.append(entity_type)

        return list(set(entity_types))  # Remove duplicates

    def expand_query(self, query: str) -> str:
        """
        Expand query with synonyms and related terms.

        Args:
            query: The search query string

        Returns:
            Expanded query string
        """
        # This is a placeholder implementation
        # In a real system, you would use a thesaurus or ML model to expand terms

        # Simple expansion mappings
        expansions = {
            "weapon": "sword axe bow dagger mace hammer",
            "armor": "shield helmet plate mail leather",
            "npc": "character person villager enemy",
            "monster": "enemy creature beast demon",
            "item": "object thing artifact",
            "skill": "ability power talent spell",
            "location": "place area zone room dungeon",
        }

        expanded_terms = []
        words = query.lower().split()

        for word in words:
            expanded_terms.append(word)
            if word in expansions:
                expanded_terms.append(expansions[word])

        return " ".join(expanded_terms)

    def classify_query_intent(self, query: str) -> str:
        """
        Classify the query as lookup, exploration, comparison, etc.

        Args:
            query: The search query string

        Returns:
            Query intent classification
        """
        # Lower case for consistent matching
        query_lower = query.lower()

        # Check for command words that indicate intent
        for command, intent in self._commands.items():
            if command in query_lower.split():
                return intent

        # Check for comparison words
        if any(
            word in query_lower
            for word in ["vs", "versus", "compare", "better", "stronger", "weaker"]
        ):
            return "comparison"

        # Check for similarity words
        if any(word in query_lower for word in ["like", "similar", "same"]):
            return "similarity"

        # Check for relationship words
        if any(word in query_lower for word in ["related", "connected", "associated"]):
            return "related"

        # Check for listing intent
        if any(word in query_lower for word in ["list", "all", "every"]):
            return "list"

        # Default intent
        return "lookup"

    def extract_constraints(self, query: str) -> dict[str, Any]:
        """
        Extract filtering constraints from the query.

        Args:
            query: The search query string

        Returns:
            Dictionary of constraints
        """
        constraints: dict[str, list[Any]] = {}

        # Look for numeric constraints
        # Example: "weapons with damage > 50" or "armor with defense >= 20"
        numeric_pattern = r"(\w+)\s*(>|<|>=|<=|=)\s*(\d+)"
        matches = re.findall(numeric_pattern, query)

        for match in matches:
            field, operator, value = match

            if field not in constraints:
                constraints[field] = []

            constraints[field].append({"operator": operator, "value": int(value)})

        # Look for categorial constraints
        # Example: "weapons of type sword" or "potions with effect healing"
        category_pattern = r"(\w+) (of|with|having) (\w+) (\w+)"
        matches = re.findall(category_pattern, query)

        for match in matches:
            item_type, _, field, value = match

            if field not in constraints:
                constraints[field] = []

            constraints[field].append({"operator": "=", "value": value})

        return constraints

    def generate_query_variants(self, query: str) -> list[str]:
        """
        Generate variations of the query for better matching.

        Args:
            query: The search query string

        Returns:
            List of query variants
        """
        variants = [query]

        # Remove stop words
        words = query.split()
        without_stop = " ".join(
            [word for word in words if word.lower() not in self._stop_words]
        )
        if without_stop and without_stop != query:
            variants.append(without_stop)

        # Word order variants (for up to 4 word queries)
        if len(words) > 1 and len(words) <= 4:
            from itertools import permutations

            for perm in permutations(words):
                variant = " ".join(perm)
                if variant != query:
                    variants.append(variant)

        return variants[:5]  # Limit to 5 variants to avoid explosion

    def estimate_query_complexity(self, query: str) -> float:
        """
        Estimate the complexity of a query to optimize search strategy.

        Args:
            query: The search query string

        Returns:
            Complexity score (0-1)
        """
        # Simple complexity heuristics
        words = query.split()

        # Base complexity on word count
        if len(words) <= 2:
            complexity = 0.2
        elif len(words) <= 5:
            complexity = 0.5
        else:
            complexity = 0.8

        # Adjust for quotes (exact phrases)
        if '"' in query:
            complexity += 0.1

        # Adjust for numeric constraints
        if any(c in query for c in [">", "<", "="]):
            complexity += 0.1

        # Adjust for special operators
        if any(op in query.lower() for op in ["and", "or", "not"]):
            complexity += 0.1

        # Cap at 1.0
        return min(complexity, 1.0)

    def _initialize_patterns(self) -> None:
        """Initialize regex patterns for entity type detection."""
        # These patterns are examples and should be customized for your game entities
        self._entity_type_patterns = {
            "character": [r"character[s]?", r"npc[s]?", r"person", r"people"],
            "weapon": [
                r"weapon[s]?",
                r"sword[s]?",
                r"axe[s]?",
                r"bow[s]?",
                r"dagger[s]?",
                r"mace[s]?",
            ],
            "armor": [r"armor[s]?", r"shield[s]?", r"helmet[s]?", r"plate[s]?"],
            "item": [r"item[s]?", r"object[s]?", r"thing[s]?"],
            "location": [
                r"location[s]?",
                r"place[s]?",
                r"area[s]?",
                r"zone[s]?",
                r"room[s]?",
            ],
            "quest": [r"quest[s]?", r"mission[s]?", r"task[s]?"],
        }

    def _detect_special_tokens(self, query: str) -> dict[str, Any]:
        """
        Detect special tokens or commands in the query.

        Args:
            query: The search query string

        Returns:
            Dictionary of special tokens and their values
        """
        tokens: dict[str, Any] = {}

        # Check for exact phrase search (quoted text)
        exact_phrases = re.findall(r'"([^"]*)"', query)
        if exact_phrases:
            tokens["exact_phrases"] = exact_phrases

        # Check for field specifications (field:value)
        field_values = re.findall(r"(\w+):(\w+)", query)
        if field_values:
            tokens["field_values"] = dict(field_values)

        # Check for boolean operators
        if " AND " in query.upper():
            tokens["boolean_and"] = True
        if " OR " in query.upper():
            tokens["boolean_or"] = True
        if " NOT " in query.upper():
            tokens["boolean_not"] = True

        # Check for range queries
        ranges = re.findall(r"(\w+):(\d+)\.\.(\d+)", query)
        if ranges:
            tokens["ranges"] = [
                {"field": field, "min": int(min_val), "max": int(max_val)}
                for field, min_val, max_val in ranges
            ]

        return tokens

    def remove_noise_words(self, query: str) -> str:
        """
        Remove noise words that don't contribute to search quality.

        Args:
            query: The search query string

        Returns:
            Query with noise words removed
        """
        words = query.split()
        filtered_words = [
            word for word in words if word.lower() not in self._stop_words
        ]
        if not filtered_words:
            # Don't return empty query, keep at least one word
            return words[0] if words else query
        return " ".join(filtered_words)

    def extract_keywords(self, query: str, top_k: int = 5) -> list[str]:
        """
        Extract the most important keywords from a query.

        Args:
            query: The search query string
            top_k: Maximum number of keywords to extract

        Returns:
            List of keywords
        """
        # Remove stop words first
        clean_query = self.remove_noise_words(query)
        words = clean_query.split()

        # For now, just return all words up to top_k
        # In a real implementation, you might use TF-IDF or other techniques
        return words[:top_k]
