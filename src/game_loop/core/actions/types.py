"""
Action type definitions for the game loop system.

This module defines the action types and classification structures used
throughout the game to categorize and process player commands.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ActionType(Enum):
    """
    Enumeration of all possible action types in the game.

    These types are used to classify player inputs and determine
    how they should be processed by the game system.
    """

    # Movement-related actions
    MOVEMENT = "movement"

    # Object and item interactions
    OBJECT_INTERACTION = "object_interaction"

    # Quest and story-related actions
    QUEST = "quest"

    # Dialogue and conversation
    CONVERSATION = "conversation"

    # Information queries and examination
    QUERY = "query"

    # System commands (save, load, help, etc.)
    SYSTEM = "system"

    # Physical actions (attack, defend, etc.)
    PHYSICAL = "physical"

    # Observation and examination
    OBSERVATION = "observation"

    # Unknown or unclassifiable actions
    UNKNOWN = "unknown"


@dataclass
class ActionClassification:
    """
    Data class representing the classification of a player action.

    This contains all the information needed to understand and process
    a player's input command.
    """

    # Primary action type
    action_type: ActionType

    # Confidence score (0.0 to 1.0) of the classification
    confidence: float

    # The primary target or object of the action
    target: str | None = None

    # Additional parameters or context for the action
    parameters: dict[str, Any] | None = None

    # Secondary targets or objects involved
    secondary_targets: list[str] | None = None

    # Extracted intent or purpose of the action
    intent: str | None = None

    # Raw input text that was classified
    raw_input: str | None = None

    # Alternative classifications with lower confidence
    alternatives: list["ActionClassification"] | None = None

    def __post_init__(self) -> None:
        """Initialize default values for mutable fields."""
        if self.parameters is None:
            self.parameters = {}
        if self.secondary_targets is None:
            self.secondary_targets = []
        if self.alternatives is None:
            self.alternatives = []

    @property
    def is_high_confidence(self) -> bool:
        """Check if this classification has high confidence (>= 0.8)."""
        return self.confidence >= 0.8

    @property
    def is_ambiguous(self) -> bool:
        """Check if this classification is ambiguous (has alternatives)."""
        return len(self.alternatives or []) > 0

    def to_dict(self) -> dict[str, Any]:
        """Convert the classification to a dictionary representation."""
        return {
            "action_type": self.action_type.value,
            "confidence": self.confidence,
            "target": self.target,
            "parameters": self.parameters or {},
            "secondary_targets": self.secondary_targets or [],
            "intent": self.intent,
            "raw_input": self.raw_input,
            "alternatives": [alt.to_dict() for alt in (self.alternatives or [])],
            "is_high_confidence": self.is_high_confidence,
            "is_ambiguous": self.is_ambiguous,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ActionClassification":
        """Create an ActionClassification from a dictionary representation."""
        alternatives = []
        if "alternatives" in data and data["alternatives"]:
            alternatives = [cls.from_dict(alt) for alt in data["alternatives"]]

        return cls(
            action_type=ActionType(data["action_type"]),
            confidence=data["confidence"],
            target=data.get("target"),
            parameters=data.get("parameters", {}),
            secondary_targets=data.get("secondary_targets", []),
            intent=data.get("intent"),
            raw_input=data.get("raw_input"),
            alternatives=alternatives,
        )
