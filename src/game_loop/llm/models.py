"""
Model definitions for LLM processing in Game Loop.
Contains Pydantic models for structured data used in NLP processing.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class CommandTypeStr(str, Enum):
    """String representation of CommandType for LLM processing."""

    MOVEMENT = "MOVEMENT"
    LOOK = "LOOK"
    INVENTORY = "INVENTORY"
    TAKE = "TAKE"
    DROP = "DROP"
    USE = "USE"
    EXAMINE = "EXAMINE"
    TALK = "TALK"
    HELP = "HELP"
    QUIT = "QUIT"
    UNKNOWN = "UNKNOWN"


class Intent(BaseModel):
    """Model for intent extraction from user input."""

    command_type: CommandTypeStr = Field(
        default=CommandTypeStr.UNKNOWN, description="The type of command being issued"
    )
    action: str = Field(
        default="unknown",
        description="The specific action verb (e.g. 'take', 'examine')",
    )
    subject: str | None = Field(
        default=None, description="The primary object of the action"
    )
    target: str | None = Field(default=None, description="The target of the action")
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score of this intent"
    )


class Disambiguation(BaseModel):
    """Model for disambiguation between multiple possible intents."""

    selected_interpretation: int = Field(
        default=0, description="Index of the selected interpretation"
    )
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence in this disambiguation"
    )
    explanation: str = Field(
        default="", description="Explanation for this interpretation"
    )


class GameLocation(BaseModel):
    """Model for location data in game context."""

    name: str
    description: str | None = None


class GameObject(BaseModel):
    """Model for game objects in context."""

    name: str
    description: str | None = None


class GameCharacter(BaseModel):
    """Model for NPCs in game context."""

    name: str
    description: str | None = None


class PlayerInventory(BaseModel):
    """Model for player inventory in game context."""

    items: list[GameObject | str] = Field(default_factory=list)


class GameContext(BaseModel):
    """Model for structured game context data."""

    current_location: GameLocation | None = None
    location: GameLocation | None = None  # Alternative field name
    visible_objects: list[GameObject | str] = Field(default_factory=list)
    npcs: list[GameCharacter | str] = Field(default_factory=list)
    player: dict[str, Any] | None = None
    inventory: list[GameObject | str] | None = None
