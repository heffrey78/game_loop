"""Quest system data models."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QuestCategory(Enum):
    """Quest categories for organizing different types of quests."""

    DELIVERY = "delivery"
    EXPLORATION = "exploration"
    COMBAT = "combat"
    PUZZLE = "puzzle"
    SOCIAL = "social"
    CRAFTING = "crafting"
    COLLECTION = "collection"


class QuestDifficulty(Enum):
    """Quest difficulty levels."""

    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    LEGENDARY = "legendary"


class QuestStatus(Enum):
    """Quest progression status."""

    AVAILABLE = "available"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    ABANDONED = "abandoned"


class QuestInteractionType(Enum):
    """Types of quest interactions."""

    DISCOVER = "discover"
    ACCEPT = "accept"
    PROGRESS = "progress"
    COMPLETE = "complete"
    ABANDON = "abandon"
    QUERY = "query"


@dataclass
class QuestStep:
    """Individual step within a quest."""

    step_id: str
    description: str
    requirements: dict[str, Any]
    completion_conditions: list[str]
    rewards: dict[str, Any] = field(default_factory=dict)
    optional: bool = False

    def __post_init__(self) -> None:
        """Validate step data after initialization."""
        if not self.step_id.strip():
            raise ValueError("step_id cannot be empty")
        if not self.description.strip():
            raise ValueError("description cannot be empty")
        if not self.completion_conditions:
            raise ValueError("completion_conditions cannot be empty")


@dataclass
class Quest:
    """Complete quest definition."""

    quest_id: str
    title: str
    description: str
    category: QuestCategory
    difficulty: QuestDifficulty
    steps: list[QuestStep]
    prerequisites: list[str] = field(default_factory=list)
    rewards: dict[str, Any] = field(default_factory=dict)
    time_limit: float | None = None
    repeatable: bool = False

    def __post_init__(self) -> None:
        """Validate quest data after initialization."""
        if not self.quest_id.strip():
            raise ValueError("quest_id cannot be empty")
        if not self.title.strip():
            raise ValueError("title cannot be empty")
        if not self.description.strip():
            raise ValueError("description cannot be empty")
        if not self.steps:
            raise ValueError("Quest must have at least one step")

        # Validate all steps
        for step in self.steps:
            if not isinstance(step, QuestStep):
                raise ValueError("All steps must be QuestStep instances")

    @property
    def total_steps(self) -> int:
        """Get total number of steps in the quest."""
        return len(self.steps)

    @property
    def required_steps(self) -> list[QuestStep]:
        """Get list of required (non-optional) steps."""
        return [step for step in self.steps if not step.optional]

    @property
    def optional_steps(self) -> list[QuestStep]:
        """Get list of optional steps."""
        return [step for step in self.steps if step.optional]


@dataclass
class QuestProgress:
    """Player's progress on a specific quest."""

    quest_id: str
    player_id: str
    status: QuestStatus
    current_step: int = 0
    completed_steps: list[str] = field(default_factory=list)
    step_progress: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Validate progress data after initialization."""
        if not self.quest_id.strip():
            raise ValueError("quest_id cannot be empty")
        if not self.player_id.strip():
            raise ValueError("player_id cannot be empty")
        if self.current_step < 0:
            raise ValueError("current_step cannot be negative")

    def mark_step_complete(self, step_id: str) -> None:
        """Mark a quest step as completed."""
        if step_id not in self.completed_steps:
            self.completed_steps.append(step_id)
            self.updated_at = time.time()

    def update_step_progress(self, step_id: str, progress_data: dict[str, Any]) -> None:
        """Update progress data for a specific step."""
        self.step_progress[step_id] = progress_data
        self.updated_at = time.time()

    def advance_to_next_step(self) -> None:
        """Advance to the next quest step."""
        self.current_step += 1
        self.updated_at = time.time()

    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage based on completed steps."""
        if not self.completed_steps:
            return 0.0
        # Note: This is a simplified calculation
        # In practice, you might want to weight steps differently
        return (len(self.completed_steps) / max(1, self.current_step + 1)) * 100.0


@dataclass
class QuestInteractionResult:
    """Result of a quest interaction."""

    success: bool
    message: str
    quest_id: str | None = None
    updated_progress: QuestProgress | None = None
    rewards_granted: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class QuestUpdate:
    """Represents an update to quest progress."""

    quest_id: str
    player_id: str
    update_type: str
    update_data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class QuestCompletionResult:
    """Result of quest completion."""

    success: bool
    quest_id: str
    final_progress: QuestProgress | None
    rewards_granted: dict[str, Any] = field(default_factory=dict)
    completion_message: str = ""
    errors: list[str] = field(default_factory=list)
