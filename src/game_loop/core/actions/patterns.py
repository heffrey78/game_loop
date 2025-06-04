"""
Action pattern definitions for rule-based action classification.

This module provides patterns and rules for classifying player inputs
based on common command structures and keywords.
"""

import re
from dataclasses import dataclass
from functools import lru_cache
from re import Pattern
from typing import Any

from .types import ActionType


@lru_cache(maxsize=1000)
def _compile_pattern(pattern_str: str) -> Pattern[str]:
    """
    Compile and cache regex patterns for better performance.

    Args:
        pattern_str: Pattern string to compile

    Returns:
        Compiled regex pattern
    """
    try:
        return re.compile(pattern_str, re.IGNORECASE)
    except re.error:
        # Return a pattern that never matches for invalid regex
        return re.compile(r"(?!.*)")


@dataclass
class ActionPattern:
    """
    Pattern definition for matching action types.

    Contains regex patterns, keywords, and associated action types
    for rule-based classification of player inputs.
    """

    action_type: ActionType
    patterns: list[str]
    keywords: list[str]
    confidence: float = 0.8
    requires_target: bool = False

    def matches_input(self, text: str) -> tuple[bool, float, dict[str, Any]]:
        """
        Check if this pattern matches the input text.

        Args:
            text: Normalized input text to match against

        Returns:
            Tuple of (matches, confidence, extracted_data)
        """
        text_lower = text.lower().strip()
        extracted_data: dict[str, Any] = {}

        # Check keyword matches first
        keyword_matches = 0
        for keyword in self.keywords:
            if keyword.lower() in text_lower:
                keyword_matches += 1

        # Check regex patterns
        pattern_match = False
        for pattern_str in self.patterns:
            compiled_pattern = _compile_pattern(pattern_str)
            match = compiled_pattern.search(text_lower)
            if match:
                pattern_match = True
                # Extract groups from regex match
                groups = match.groups()
                if groups:
                    if len(groups) >= 1 and groups[0]:
                        extracted_data["target"] = groups[0].strip()
                    if len(groups) >= 2 and groups[1]:
                        extracted_data["secondary_target"] = groups[1].strip()
                break

        # Calculate match confidence
        if pattern_match or keyword_matches > 0:
            base_confidence = self.confidence

            # Boost confidence for multiple keyword matches
            if keyword_matches > 1:
                base_confidence = min(
                    1.0, base_confidence + (keyword_matches - 1) * 0.1
                )

            # Boost confidence for pattern matches
            if pattern_match:
                base_confidence = min(1.0, base_confidence + 0.1)

            return True, base_confidence, extracted_data

        return False, 0.0, {}


class ActionPatternManager:
    """
    Manages action patterns for rule-based classification.

    Provides methods to register patterns and classify input text
    based on predefined rules and patterns.
    """

    def __init__(self) -> None:
        """Initialize the pattern manager with default patterns."""
        self.patterns: list[ActionPattern] = []
        self._initialize_default_patterns()

    def _initialize_default_patterns(self) -> None:
        """Initialize default action patterns for common game commands."""

        # Movement patterns
        self.patterns.append(
            ActionPattern(
                action_type=ActionType.MOVEMENT,
                patterns=[
                    r"^(?:go|move|walk|run|head|travel)\s+(?:to\s+)?(.+)$",
                    r"^(north|south|east|west|up|down|n|s|e|w|u|d)$",
                    r"^(?:go\s+)?(north|south|east|west|up|down|n|s|e|w|u|d)$",
                ],
                keywords=[
                    "north",
                    "south",
                    "east",
                    "west",
                    "up",
                    "down",
                ],  # Removed generic verbs
                confidence=0.9,
            )
        )

        # Object interaction patterns
        self.patterns.append(
            ActionPattern(
                action_type=ActionType.OBJECT_INTERACTION,
                patterns=[
                    r"^(?:use|activate|operate|press|push|pull|turn)\s+(?:the\s+)?(.+?)(?:\s+(?:with|on)\s+(?:the\s+)?(.+))?$",
                    r"^(?:take|grab|get|pick\s+up)\s+(?:the\s+)?(.+)$",
                    r"^(?:drop|put\s+down)\s+(?:the\s+)?(.+)$",
                    r"^(?:open|close)\s+(?:the\s+)?(.+?)(?:\s+with\s+(?:the\s+)?(.+))?$",
                ],
                keywords=[
                    "use",
                    "take",
                    "grab",
                    "get",
                    "drop",
                    "open",
                    "close",
                    "push",
                    "pull",
                ],
                confidence=0.85,
                requires_target=True,
            )
        )

        # Observation patterns
        self.patterns.append(
            ActionPattern(
                action_type=ActionType.OBSERVATION,
                patterns=[
                    r"^(?:look|examine|inspect|check)\s+(?:at\s+)?(?:the\s+)?(.+)$",
                    r"^(?:look|l)(?:\s+around)?$",
                    r"^(?:describe|what\s+is)\s+(?:the\s+)?(.+)$",
                ],
                keywords=["look", "examine", "inspect", "check", "describe"],
                confidence=0.9,
            )
        )

        # Query patterns
        self.patterns.append(
            ActionPattern(
                action_type=ActionType.QUERY,
                patterns=[
                    r"^(?:inventory|i|items)$",
                    r"^(?:where\s+am\s+i|where|location)$",
                    r"^(?:what\s+(?:is|are)|who\s+(?:is|are))\s+(.+)$",
                    r"^(?:how\s+(?:do\s+i|to))\s+(.+)$",
                ],
                keywords=["inventory", "where", "what", "who", "how", "status"],
                confidence=0.9,
            )
        )

        # Conversation patterns
        self.patterns.append(
            ActionPattern(
                action_type=ActionType.CONVERSATION,
                patterns=[
                    r"^(?:talk\s+to|speak\s+with|ask|tell)\s+(?:the\s+)?(.+?)(?:\s+about\s+(.+))?$",
                    r"^(?:say|speak)\s+(.+)$",
                    r"^(?:greet|hello\s+to)\s+(?:the\s+)?(.+)$",
                ],
                keywords=["talk", "speak", "ask", "tell", "say", "greet", "hello"],
                confidence=0.85,
                requires_target=True,
            )
        )

        # System command patterns
        self.patterns.append(
            ActionPattern(
                action_type=ActionType.SYSTEM,
                patterns=[
                    r"^(?:help|h|\?)(?:\s+(.+))?$",
                    r"^(?:quit|exit|q|bye)$",
                    r"^(?:save|load)\s*(?:game)?(?:\s+(.+))?$",
                ],
                keywords=["help", "quit", "exit", "save", "load"],
                confidence=0.95,
            )
        )

        # Physical action patterns
        self.patterns.append(
            ActionPattern(
                action_type=ActionType.PHYSICAL,
                patterns=[
                    r"^(?:attack|fight|hit|strike)\s+(?:the\s+)?(.+?)(?:\s+with\s+(?:the\s+)?(.+))?$",
                    r"^(?:defend|block|dodge)(?:\s+(?:against|from)\s+(?:the\s+)?(.+))?$",
                ],
                keywords=["attack", "fight", "hit", "strike", "defend", "block"],
                confidence=0.8,
                requires_target=True,
            )
        )

    def add_pattern(self, pattern: ActionPattern) -> None:
        """
        Add a custom pattern to the manager.

        Args:
            pattern: ActionPattern to add
        """
        self.patterns.append(pattern)

    def classify_input(
        self, text: str
    ) -> list[tuple[ActionType, float, dict[str, Any]]]:
        """
        Classify input text using rule-based patterns.

        Args:
            text: Input text to classify

        Returns:
            List of (action_type, confidence, extracted_data) tuples,
            sorted by confidence in descending order
        """
        if not text.strip():
            return [(ActionType.UNKNOWN, 0.0, {})]

        matches: list[tuple[ActionType, float, dict[str, Any]]] = []

        for pattern in self.patterns:
            is_match, confidence, extracted_data = pattern.matches_input(text)
            if is_match:
                matches.append((pattern.action_type, confidence, extracted_data))

        # Sort by confidence (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)

        # Return top matches, or UNKNOWN if no matches
        if matches:
            return matches
        else:
            return [(ActionType.UNKNOWN, 0.0, {"raw_input": text})]

    def get_patterns_for_type(self, action_type: ActionType) -> list[ActionPattern]:
        """
        Get all patterns for a specific action type.

        Args:
            action_type: ActionType to get patterns for

        Returns:
            List of patterns matching the action type
        """
        return [p for p in self.patterns if p.action_type == action_type]

    def get_pattern_stats(self) -> dict[str, Any]:
        """
        Get statistics about registered patterns.

        Returns:
            Dictionary with pattern statistics
        """
        type_counts: dict[ActionType, int] = {}
        total_patterns = len(self.patterns)

        for pattern in self.patterns:
            type_counts[pattern.action_type] = (
                type_counts.get(pattern.action_type, 0) + 1
            )

        return {
            "total_patterns": total_patterns,
            "patterns_by_type": {t.value: count for t, count in type_counts.items()},
            "action_types_covered": len(type_counts),
        }
