"""
Action Type Classifier for Game Loop.

This module provides the main ActionTypeClassifier that combines rule-based
and LLM-based classification to determine the type and intent of player actions.
"""

import hashlib
import json
import logging
import time
from typing import Any

from game_loop.config.manager import ConfigManager
from game_loop.llm.nlp_processor import NLPProcessor
from game_loop.llm.ollama.client import OllamaClient

from ..models.system_models import SystemCommandClassification, SystemCommandType
from .exceptions import (
    ActionClassificationError,
    LLMClassificationError,
    PatternMatchError,
)
from .patterns import ActionPatternManager
from .types import ActionClassification, ActionType

logger = logging.getLogger(__name__)


class ActionClassificationCache:
    """Simple cache for action classifications to improve performance."""

    def __init__(self, max_size: int = 500, ttl_seconds: int = 300):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of entries to store
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[ActionClassification, float]] = {}
        self._access_order: list[str] = []

    def _generate_key(self, text: str, context_hash: str = "") -> str:
        """Generate cache key for input text and context."""
        combined = f"{text.lower().strip()}:{context_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def get(self, text: str, context_hash: str = "") -> ActionClassification | None:
        """Get cached classification if available and not expired."""
        key = self._generate_key(text, context_hash)

        if key in self._cache:
            classification, timestamp = self._cache[key]

            # Check if expired
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return None

            # Update access order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            return classification

        return None

    def set(
        self, text: str, classification: ActionClassification, context_hash: str = ""
    ) -> None:
        """Store classification in cache."""
        key = self._generate_key(text, context_hash)

        # Remove oldest entry if cache is full
        if len(self._cache) >= self.max_size and key not in self._cache:
            if self._access_order:
                oldest_key = self._access_order.pop(0)
                if oldest_key in self._cache:
                    del self._cache[oldest_key]

        # Store new entry
        self._cache[key] = (classification, time.time())

        # Update access order
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()


class ActionTypeClassifier:
    """
    Main classifier that combines rule-based and LLM classification.

    Uses a hybrid approach where rule-based patterns provide fast classification
    for common commands, while LLM processing handles complex or ambiguous inputs.
    """

    def __init__(
        self,
        config_manager: ConfigManager | None = None,
        ollama_client: OllamaClient | None = None,
        pattern_manager: ActionPatternManager | None = None,
        enable_cache: bool = True,
    ):
        """
        Initialize the action classifier.

        Args:
            config_manager: Configuration manager for LLM settings
            ollama_client: Client for Ollama API communication
            pattern_manager: Manager for rule-based patterns
            enable_cache: Whether to enable classification caching
        """
        self.config_manager = config_manager or ConfigManager()
        self.pattern_manager = pattern_manager or ActionPatternManager()

        # Initialize NLP processor for LLM fallback
        self.nlp_processor = NLPProcessor(
            config_manager=self.config_manager, ollama_client=ollama_client
        )

        # Get action classification configuration
        action_config = self.config_manager.get_config().action_classification

        # Classification cache
        self.cache = (
            ActionClassificationCache(
                max_size=action_config.cache_size,
                ttl_seconds=action_config.cache_ttl_seconds,
            )
            if action_config.enable_cache and enable_cache
            else None
        )

        # Configuration thresholds
        self.high_confidence_threshold = action_config.high_confidence_threshold
        self.rule_confidence_threshold = action_config.rule_confidence_threshold
        self.llm_fallback_threshold = action_config.llm_fallback_threshold

        # Statistics
        self._stats = {
            "total_classifications": 0,
            "rule_based_classifications": 0,
            "llm_classifications": 0,
            "hybrid_classifications": 0,
            "cache_hits": 0,
            "high_confidence_classifications": 0,
        }

    async def classify_action(
        self, user_input: str, game_context: dict[str, Any] | None = None
    ) -> ActionClassification:
        """
        Main method to classify a user action.

        Args:
            user_input: Raw user input text
            game_context: Current game state context

        Returns:
            ActionClassification with type, confidence, and extracted data
        """
        if not user_input or not user_input.strip():
            return ActionClassification(
                action_type=ActionType.UNKNOWN,
                confidence=0.0,
                raw_input=user_input,
                intent="Empty input",
            )

        normalized_input = self._normalize_input(user_input)
        context_hash = self._hash_context(game_context) if game_context else ""

        # Check cache first
        if self.cache:
            cached_result = self.cache.get(normalized_input, context_hash)
            if cached_result:
                self._stats["cache_hits"] += 1
                # Update raw_input to current input (might have different casing)
                cached_result.raw_input = user_input
                return cached_result

        self._stats["total_classifications"] += 1

        try:
            # Start with rule-based classification
            rule_results = self.classify_with_rules(normalized_input)

            # Determine if we need LLM assistance
            best_rule_confidence = rule_results.confidence if rule_results else 0.0

            if best_rule_confidence >= self.high_confidence_threshold:
                # High confidence rule match - use it directly
                classification = rule_results
                classification.raw_input = user_input
                self._stats["rule_based_classifications"] += 1

            elif best_rule_confidence >= self.rule_confidence_threshold:
                # Medium confidence - use hybrid approach
                classification = await self.hybrid_classification(
                    normalized_input, rule_results, game_context
                )
                classification.raw_input = user_input
                self._stats["hybrid_classifications"] += 1

            else:
                # Low confidence or no rule match - use LLM
                classification = await self.classify_with_llm(
                    normalized_input, game_context
                )
                classification.raw_input = user_input
                self._stats["llm_classifications"] += 1

            # Update statistics
            if classification.is_high_confidence:
                self._stats["high_confidence_classifications"] += 1

            # Cache the result
            if self.cache:
                self.cache.set(normalized_input, classification, context_hash)

            return classification

        except (PatternMatchError, LLMClassificationError) as e:
            logger.error(f"Specific error in action classification: {e}")
            # Return fallback classification with error context
            return ActionClassification(
                action_type=ActionType.UNKNOWN,
                confidence=0.0,
                raw_input=user_input,
                intent=f"Classification error: {str(e)}",
            )
        except Exception as e:
            logger.error(f"Unexpected error in action classification: {e}")
            # Wrap unknown errors in our base exception type
            raise ActionClassificationError(
                f"Unexpected classification error: {str(e)}", user_input
            ) from e

    def classify_with_rules(self, normalized_input: str) -> ActionClassification:
        """
        Classify input using rule-based patterns.

        Args:
            normalized_input: Normalized input text

        Returns:
            ActionClassification based on pattern matching
        """
        try:
            matches = self.pattern_manager.classify_input(normalized_input)

            if not matches:
                return ActionClassification(
                    action_type=ActionType.UNKNOWN,
                    confidence=0.0,
                    intent="No pattern matches",
                )

            # Get the best match
            best_match = matches[0]
            action_type, confidence, extracted_data = best_match

            # Extract components from the match
            target = extracted_data.get("target")
            secondary_target = extracted_data.get("secondary_target")

            # Create alternatives from lower-confidence matches
            alternatives = []
            for match in matches[1:4]:  # Up to 3 alternatives
                alt_type, alt_confidence, alt_data = match
                if alt_confidence > 0.3:  # Only include reasonable alternatives
                    alternatives.append(
                        ActionClassification(
                            action_type=alt_type,
                            confidence=alt_confidence,
                            target=alt_data.get("target"),
                            secondary_targets=(
                                [str(alt_data.get("secondary_target"))]
                                if alt_data.get("secondary_target") is not None
                                else []
                            ),
                            intent=f"Rule-based alternative: {alt_type.value}",
                        )
                    )

            return ActionClassification(
                action_type=action_type,
                confidence=confidence,
                target=target,
                secondary_targets=[secondary_target] if secondary_target else [],
                intent=f"Rule-based classification: {action_type.value}",
                alternatives=alternatives,
            )

        except Exception as e:
            logger.error(f"Error in rule-based classification: {e}")
            # Raise specific pattern match error
            raise PatternMatchError(
                f"Rule-based classification failed: {str(e)}",
                input_text=normalized_input,
            ) from e

    async def classify_with_llm(
        self, normalized_input: str, game_context: dict[str, Any] | None = None
    ) -> ActionClassification:
        """
        Classify input using LLM processing.

        Args:
            normalized_input: Normalized input text
            game_context: Current game state context

        Returns:
            ActionClassification based on LLM analysis
        """
        try:
            # Use the NLP processor to extract intent
            parsed_command = await self.nlp_processor.process_input(
                normalized_input, game_context
            )

            # Map command to action type
            action_type = self._map_command_to_action_type(parsed_command.command_type)

            # Extract confidence from parameters
            confidence = parsed_command.parameters.get("confidence", 0.5)

            return ActionClassification(
                action_type=action_type,
                confidence=confidence,
                target=parsed_command.target,
                intent=f"LLM classification: {parsed_command.action}",
                parameters={
                    "command_type": parsed_command.command_type.value,
                    "action": parsed_command.action,
                    "subject": parsed_command.subject,
                },
            )

        except Exception as e:
            logger.error(f"Error in LLM classification: {e}")
            # Raise specific LLM classification error
            raise LLMClassificationError(
                f"LLM classification failed: {str(e)}",
                input_text=normalized_input,
            ) from e

    async def hybrid_classification(
        self,
        normalized_input: str,
        rule_result: ActionClassification,
        game_context: dict[str, Any] | None = None,
    ) -> ActionClassification:
        """
        Combine rule-based and LLM classification for better accuracy.

        Args:
            normalized_input: Normalized input text
            rule_result: Result from rule-based classification
            game_context: Current game state context

        Returns:
            ActionClassification combining both approaches
        """
        try:
            # Get LLM classification
            llm_result = await self.classify_with_llm(normalized_input, game_context)

            # If both agree on action type, combine confidences
            if rule_result.action_type == llm_result.action_type:
                # Weighted average with rule bias for agreement
                combined_confidence = (
                    rule_result.confidence * 0.6 + llm_result.confidence * 0.4
                )
                combined_confidence = min(
                    1.0, combined_confidence + 0.1
                )  # Bonus for agreement

                return ActionClassification(
                    action_type=rule_result.action_type,
                    confidence=combined_confidence,
                    target=rule_result.target or llm_result.target,
                    secondary_targets=rule_result.secondary_targets or [],
                    intent=(
                        f"Hybrid classification (agreement): "
                        f"{rule_result.action_type.value}"
                    ),
                    parameters={
                        **(rule_result.parameters or {}),
                        **(llm_result.parameters or {}),
                        "classification_method": "hybrid_agreement",
                    },
                    alternatives=[llm_result] if llm_result.confidence > 0.3 else [],
                )

            # If they disagree, choose the higher confidence one but include alternative
            if rule_result.confidence > llm_result.confidence:
                primary = rule_result
                alternative = llm_result
                method = "hybrid_rule_preferred"
            else:
                primary = llm_result
                alternative = rule_result
                method = "hybrid_llm_preferred"

            primary.intent = (
                f"Hybrid classification ({method}): {primary.action_type.value}"
            )
            primary.parameters = primary.parameters or {}
            primary.parameters["classification_method"] = method

            if alternative.confidence > 0.3:
                primary.alternatives = [alternative]

            return primary

        except Exception as e:
            logger.error(f"Error in hybrid classification: {e}")
            # Fall back to rule result on error
            rule_result.intent = (
                f"Hybrid fallback to rules: {rule_result.action_type.value}"
            )
            return rule_result

    def extract_action_components(self, text: str) -> dict[str, Any]:
        """
        Extract action components from input text.

        Args:
            text: Input text to parse

        Returns:
            Dictionary with extracted components (action, target, etc.)
        """
        components: dict[str, Any] = {
            "raw_text": text,
            "normalized_text": self._normalize_input(text),
            "tokens": text.lower().split(),
            "action_verbs": [],
            "potential_targets": [],
        }

        # Simple verb extraction
        action_verbs = [
            "go",
            "move",
            "walk",
            "run",
            "take",
            "grab",
            "get",
            "drop",
            "put",
            "use",
            "open",
            "close",
            "look",
            "examine",
            "inspect",
            "talk",
            "say",
            "attack",
            "fight",
            "help",
            "quit",
            "save",
            "load",
        ]

        for token in components["tokens"]:
            if token in action_verbs:
                components["action_verbs"].append(token)

        # Extract potential targets (words after common prepositions and articles)
        target_indicators = ["at", "to", "with", "on", "in", "the"]
        tokens = components["tokens"]

        # Simple approach: look for nouns after verbs and prepositions
        for i, token in enumerate(tokens):
            if token in target_indicators and i + 1 < len(tokens):
                components["potential_targets"].append(tokens[i + 1])

        # Also extract words that come after action verbs (simple heuristic)
        for i, token in enumerate(tokens):
            if token in components["action_verbs"] and i + 1 < len(tokens):
                # Skip articles and add remaining words as potential targets
                for j in range(i + 1, len(tokens)):
                    if tokens[j] not in ["the", "a", "an"]:
                        components["potential_targets"].append(tokens[j])

        return components

    def calculate_confidence_scores(
        self,
        rule_confidence: float,
        llm_confidence: float,
        context_relevance: float = 0.5,
    ) -> dict[str, float]:
        """
        Calculate confidence scores for different classification methods.

        Args:
            rule_confidence: Confidence from rule-based classification
            llm_confidence: Confidence from LLM classification
            context_relevance: Relevance of game context (0.0 to 1.0)

        Returns:
            Dictionary with calculated confidence scores
        """
        # Base scores
        scores = {
            "rule_based": rule_confidence,
            "llm_based": llm_confidence,
            "hybrid_agreement": 0.0,
            "hybrid_weighted": 0.0,
        }

        # Agreement bonus
        if abs(rule_confidence - llm_confidence) < 0.2:
            agreement_bonus = 0.1 * (1.0 - abs(rule_confidence - llm_confidence) / 0.2)
            scores["hybrid_agreement"] = min(
                1.0, max(rule_confidence, llm_confidence) + agreement_bonus
            )

        # Weighted combination
        rule_weight = 0.6 + (0.2 if rule_confidence > 0.8 else 0.0)
        llm_weight = 1.0 - rule_weight
        scores["hybrid_weighted"] = (
            rule_confidence * rule_weight + llm_confidence * llm_weight
        )

        # Context adjustment
        if context_relevance > 0.5:
            scores["llm_based"] *= 1.0 + (context_relevance - 0.5) * 0.2
            scores["hybrid_weighted"] *= 1.0 + (context_relevance - 0.5) * 0.1

        return scores

    def get_classification_stats(self) -> dict[str, Any]:
        """
        Get classification statistics.

        Returns:
            Dictionary with classification performance statistics
        """
        total = self._stats["total_classifications"]

        stats: dict[str, Any] = dict(self._stats)

        if total > 0:
            stats["rule_based_percentage"] = (
                self._stats["rule_based_classifications"] / total
            ) * 100
            stats["llm_percentage"] = (self._stats["llm_classifications"] / total) * 100
            stats["hybrid_percentage"] = (
                self._stats["hybrid_classifications"] / total
            ) * 100
            stats["high_confidence_percentage"] = (
                self._stats["high_confidence_classifications"] / total
            ) * 100
            stats["cache_hit_rate"] = (
                (self._stats["cache_hits"] / total) * 100 if total > 0 else 0
            )

        # Add pattern manager stats
        stats["pattern_stats"] = self.pattern_manager.get_pattern_stats()

        return stats

    def clear_cache(self) -> None:
        """Clear the classification cache."""
        if self.cache:
            self.cache.clear()

    def _normalize_input(self, text: str) -> str:
        """Normalize input text for consistent processing."""
        return text.lower().strip()

    def _hash_context(self, context: dict[str, Any]) -> str:
        """Generate a hash for game context to use in caching."""
        try:
            # Create a simplified context hash based on key elements
            context_str = json.dumps(
                {
                    "location": context.get("current_location", {}).get("name", ""),
                    "objects": [
                        obj.get("name", "")
                        for obj in context.get("visible_objects", [])[:5]
                    ],
                    "npcs": [
                        npc.get("name", "") for npc in context.get("npcs", [])[:3]
                    ],
                },
                sort_keys=True,
            )
            return hashlib.sha256(context_str.encode()).hexdigest()[:8]
        except Exception:
            return ""

    def _map_command_to_action_type(self, command_type: Any) -> ActionType:
        """Map command type from NLP processor to ActionType."""
        # This is a simplified mapping - could be expanded based on command types
        type_mapping = {
            "MOVEMENT": ActionType.MOVEMENT,
            "LOOK": ActionType.OBSERVATION,
            "EXAMINE": ActionType.OBSERVATION,
            "TAKE": ActionType.OBJECT_INTERACTION,
            "USE": ActionType.OBJECT_INTERACTION,
            "INVENTORY": ActionType.QUERY,
            "HELP": ActionType.SYSTEM,
            "QUIT": ActionType.SYSTEM,
            "TALK": ActionType.CONVERSATION,
            "ATTACK": ActionType.PHYSICAL,
        }

        command_name = (
            command_type.name if hasattr(command_type, "name") else str(command_type)
        )
        return type_mapping.get(command_name, ActionType.UNKNOWN)

    async def classify_system_command(
        self, text: str
    ) -> SystemCommandClassification | None:
        """Classify system/meta-game commands."""
        try:
            text_lower = text.lower().strip()

            # Define system command patterns (order matters - more specific patterns first)
            system_patterns = {
                SystemCommandType.LIST_SAVES: [
                    r"^/?list\s+saves?$",
                    r"^/?show\s+saves?$",
                    r"^/?save\s+files?$",
                    r"^/?my\s+saves?$",
                ],
                SystemCommandType.LOAD_GAME: [
                    r"^/?load(?:\s+game)?(?:\s+.+)?$",
                    r"^/?restore(?:\s+save)?$",
                    r"^/?continue(?:\s+game)?$",
                    r"^/?quickload$",
                ],
                SystemCommandType.SAVE_GAME: [
                    r"^/?save(?:\s+game)?(?:\s+.+)?$",
                    r"^/?create\s+save\b",
                    r"^/?save\s+state\b",
                    r"^/?quicksave$",
                ],
                SystemCommandType.HELP: [
                    r"^/?help(?:\s+.+)?$",
                    r"^/?how\s+do\s+i\b",
                    r"^/?what\s+can\s+i\s+do$",
                    r"^/?commands?$",
                    r"^/?\?$",
                ],
                SystemCommandType.TUTORIAL: [
                    r"^/?tutorial$",
                    r"^/?guide$",
                    r"^/?show\s+me\s+how\b",
                    r"^/?teach\s+me$",
                    r"^/?learn$",
                ],
                SystemCommandType.SETTINGS: [
                    r"^/?settings?$",
                    r"^/?options?$",
                    r"^/?preferences?$",
                    r"^/?config$",
                    r"^/?set\s+\w+\b",
                ],
                SystemCommandType.QUIT_GAME: [
                    r"^/?(quit|exit)(?:\s+game)?$",
                    r"^/?leave$",
                    r"^/?stop$",
                    r"^/?end\s+game$",
                ],
            }

            import re

            # Check each pattern type
            for command_type, patterns in system_patterns.items():
                for pattern in patterns:
                    match = re.search(pattern, text_lower)
                    if match:
                        # Extract arguments from the match
                        args = self._extract_system_command_args(
                            text, command_type, match
                        )

                        return SystemCommandClassification(
                            command_type=command_type,
                            args=args,
                            confidence=0.9,  # High confidence for pattern matches
                            original_text=text,
                        )

            return None

        except Exception as e:
            logger.error(f"Error in system command classification: {e}")
            return None

    def _extract_system_command_args(
        self, text: str, command_type: SystemCommandType, match
    ) -> dict[str, Any]:
        """Extract arguments from system command match."""
        args = {}
        text_lower = text.lower().strip()

        if command_type == SystemCommandType.SAVE_GAME:
            # Extract save name if provided
            # Handle both "/save name" and "save as name" formats
            words = text.split()
            if "as" in text_lower:
                parts = text_lower.split("as", 1)
                if len(parts) > 1:
                    save_name = parts[1].strip()
                    args["save_name"] = save_name
            elif len(words) > 1:
                # Remove slash prefix if present
                first_word = words[0].lstrip("/")
                if first_word.lower() in ["save"]:
                    # Get everything after "save" (skip "game" if present)
                    save_parts = words[1:]
                    if save_parts and save_parts[0].lower() == "game":
                        save_parts = save_parts[1:]
                    if save_parts:
                        save_name = " ".join(save_parts).strip()
                        args["save_name"] = save_name

        elif command_type == SystemCommandType.LOAD_GAME:
            # Extract save name if provided
            words = text.split()
            if len(words) > 1:
                # Remove slash prefix if present
                first_word = words[0].lstrip("/")
                if first_word.lower() in ["load", "restore"]:
                    save_parts = words[1:]
                    if save_parts and save_parts[0].lower() == "game":
                        save_parts = save_parts[1:]
                    if save_parts:
                        save_name = " ".join(save_parts).strip()
                        args["save_name"] = save_name

        elif command_type == SystemCommandType.HELP:
            # Extract help topic if provided
            words = text.split()
            if len(words) > 1:
                topic = " ".join(words[1:]).strip()
                args["topic"] = topic

        elif command_type == SystemCommandType.SETTINGS:
            # Extract setting name and value if provided
            if "set" in text_lower:
                parts = text.split()
                if len(parts) >= 3 and parts[0].lower() == "set":
                    args["setting"] = parts[1]
                    if len(parts) > 2:
                        args["value"] = " ".join(parts[2:])

        return args
