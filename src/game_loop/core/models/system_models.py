"""Data models for system command functionality."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class SystemCommandType(Enum):
    """Types of system commands."""

    SAVE_GAME = "save_game"
    LOAD_GAME = "load_game"
    HELP = "help"
    TUTORIAL = "tutorial"
    SETTINGS = "settings"
    QUIT_GAME = "quit_game"
    AUTO_SAVE = "auto_save"
    LIST_SAVES = "list_saves"


class TutorialType(Enum):
    """Types of tutorials."""

    BASIC_COMMANDS = "basic_commands"
    MOVEMENT = "movement"
    OBJECT_INTERACTION = "object_interaction"
    CONVERSATION = "conversation"
    INVENTORY = "inventory"
    QUESTS = "quests"


class PlayerSkillLevel(Enum):
    """Player skill assessment levels."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class SaveMetadata:
    """Metadata for a game save."""

    save_name: str
    description: str
    created_at: datetime
    file_size: int
    player_level: int
    location: str
    play_time: timedelta
    save_id: uuid.UUID = field(default_factory=uuid.uuid4)
    player_id: uuid.UUID | None = None
    file_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "save_id": str(self.save_id),
            "save_name": self.save_name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "file_size": self.file_size,
            "player_level": self.player_level,
            "location": self.location,
            "play_time": self.play_time.total_seconds(),
            "player_id": str(self.player_id) if self.player_id else None,
            "file_path": self.file_path,
        }


@dataclass
class SaveResult:
    """Result of a save operation."""

    success: bool
    save_name: str
    message: str
    save_metadata: SaveMetadata | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "save_name": self.save_name,
            "message": self.message,
            "save_metadata": (
                self.save_metadata.to_dict() if self.save_metadata else None
            ),
            "error": self.error,
        }


@dataclass
class LoadResult:
    """Result of a load operation."""

    success: bool
    save_name: str
    message: str
    game_state: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "save_name": self.save_name,
            "message": self.message,
            "game_state": self.game_state,
            "error": self.error,
        }


@dataclass
class HelpTopic:
    """Individual help topic."""

    topic_id: str
    title: str
    content: str
    category: str
    keywords: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    related_topics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "topic_id": self.topic_id,
            "title": self.title,
            "content": self.content,
            "category": self.category,
            "keywords": self.keywords,
            "examples": self.examples,
            "related_topics": self.related_topics,
        }


@dataclass
class HelpResponse:
    """Response to a help request."""

    topic: str
    content: str
    related_topics: list[str] = field(default_factory=list)
    contextual_suggestions: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    category: str = "general"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "topic": self.topic,
            "content": self.content,
            "related_topics": self.related_topics,
            "contextual_suggestions": self.contextual_suggestions,
            "examples": self.examples,
            "category": self.category,
        }


@dataclass
class TutorialHint:
    """A tutorial hint or guidance message."""

    hint_type: str
    message: str
    suggested_action: str
    priority: int = 1
    tutorial_type: TutorialType = TutorialType.BASIC_COMMANDS
    step_number: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "hint_type": self.hint_type,
            "message": self.message,
            "suggested_action": self.suggested_action,
            "priority": self.priority,
            "tutorial_type": self.tutorial_type.value,
            "step_number": self.step_number,
        }


@dataclass
class TutorialSession:
    """Active tutorial session."""

    tutorial_type: TutorialType
    player_id: uuid.UUID
    current_step: int = 0
    completed_steps: list[int] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    last_activity: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tutorial_type": self.tutorial_type.value,
            "player_id": str(self.player_id),
            "current_step": self.current_step,
            "completed_steps": self.completed_steps,
            "started_at": self.started_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
        }


@dataclass
class TutorialPrompt:
    """A prompt to start or continue a tutorial."""

    tutorial_type: TutorialType
    trigger_reason: str
    suggested_message: str
    priority: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tutorial_type": self.tutorial_type.value,
            "trigger_reason": self.trigger_reason,
            "suggested_message": self.suggested_message,
            "priority": self.priority,
        }


@dataclass
class SettingDefinition:
    """Definition of a configurable setting."""

    name: str
    description: str
    default_value: Any
    allowed_values: list[Any] | None = None
    value_type: str = "string"
    category: str = "general"
    validation_rules: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "default_value": self.default_value,
            "allowed_values": self.allowed_values,
            "value_type": self.value_type,
            "category": self.category,
            "validation_rules": self.validation_rules,
        }


@dataclass
class SettingInfo:
    """Information about a setting and its current value."""

    name: str
    description: str
    current_value: Any
    default_value: Any
    allowed_values: list[Any] | None = None
    category: str = "general"
    value_type: str = "string"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "current_value": self.current_value,
            "default_value": self.default_value,
            "allowed_values": self.allowed_values,
            "category": self.category,
            "value_type": self.value_type,
        }


@dataclass
class SettingResult:
    """Result of a setting operation."""

    success: bool
    setting_name: str
    old_value: Any
    new_value: Any
    message: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "setting_name": self.setting_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "message": self.message,
            "error": self.error,
        }


@dataclass
class HelpContext:
    """Context information for generating help."""

    current_location: str | None = None
    available_commands: list[str] = field(default_factory=list)
    nearby_objects: list[str] = field(default_factory=list)
    nearby_npcs: list[str] = field(default_factory=list)
    player_level: int = 1
    player_skill_level: PlayerSkillLevel = PlayerSkillLevel.BEGINNER
    recent_actions: list[str] = field(default_factory=list)
    current_quest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "current_location": self.current_location,
            "available_commands": self.available_commands,
            "nearby_objects": self.nearby_objects,
            "nearby_npcs": self.nearby_npcs,
            "player_level": self.player_level,
            "player_skill_level": self.player_skill_level.value,
            "recent_actions": self.recent_actions,
            "current_quest": self.current_quest,
        }


@dataclass
class SystemCommandClassification:
    """Classification result for system commands."""

    command_type: SystemCommandType
    args: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    original_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "command_type": self.command_type.value,
            "args": self.args,
            "confidence": self.confidence,
            "original_text": self.original_text,
        }
