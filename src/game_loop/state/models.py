"""Pydantic models for game state representation and persistence."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

# --- Player Related Models ---


class InventoryItem(BaseModel):
    """Represents an item held by the player."""

    item_id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    quantity: int = 1
    attributes: dict[str, Any] = Field(default_factory=dict)


class PlayerKnowledge(BaseModel):
    """Represents a piece of knowledge the player has acquired."""

    knowledge_id: UUID = Field(default_factory=uuid4)
    topic: str
    content: str
    discovered_at: datetime = Field(default_factory=datetime.now)
    source: str | None = None


class PlayerStats(BaseModel):
    """Represents the player's numerical statistics."""

    health: int = 100
    max_health: int = 100
    mana: int = 50
    max_mana: int = 50
    strength: int = 10
    dexterity: int = 10
    intelligence: int = 10


class PlayerProgress(BaseModel):
    """Tracks player progress, like quests or achievements."""

    active_quests: dict[UUID, dict[str, Any]] = Field(
        default_factory=dict
    )  # Quest ID -> Quest State
    completed_quests: list[UUID] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    flags: dict[str, bool] = Field(default_factory=dict)


class PlayerState(BaseModel):
    """Complete state of the player character."""

    player_id: UUID = Field(default_factory=uuid4)
    name: str = "Player"
    current_location_id: UUID | None = None
    inventory: list[InventoryItem] = Field(default_factory=list)
    knowledge: list[PlayerKnowledge] = Field(default_factory=list)
    stats: PlayerStats = Field(default_factory=PlayerStats)
    progress: PlayerProgress = Field(default_factory=PlayerProgress)
    visited_locations: list[UUID] = Field(default_factory=list)
    state_data_json: str | None = None


# --- World Related Models ---
class WorldObject(BaseModel):
    """Represents an object within a location."""

    object_id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    is_takeable: bool = False
    is_container: bool = False
    is_hidden: bool = False
    state: dict[str, Any] = Field(
        default_factory=dict
    )  # e.g., {"locked": True, "open": False}
    contained_items: list[UUID] = Field(
        default_factory=list
    )  # IDs of items inside, if container


class NonPlayerCharacter(BaseModel):
    """Represents an NPC within a location."""

    npc_id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    dialogue_state: str = "neutral"
    current_behavior: str = "idle"
    inventory: list[UUID] = Field(default_factory=list)
    knowledge: list[PlayerKnowledge] = Field(default_factory=list)
    state: dict[str, Any] = Field(default_factory=dict)


class Location(BaseModel):
    """Represents a single location in the game world."""

    location_id: UUID = Field(default_factory=uuid4)
    name: str
    description: str
    objects: dict[UUID, WorldObject] = Field(default_factory=dict)
    npcs: dict[UUID, NonPlayerCharacter] = Field(default_factory=dict)
    connections: dict[str, UUID] = Field(default_factory=dict)
    state_flags: dict[str, Any] = Field(default_factory=dict)
    first_visited: datetime | None = None
    last_visited: datetime | None = None


class WorldState(BaseModel):
    """Represents the entire state of the game world."""

    world_id: UUID = Field(default_factory=uuid4)
    locations: dict[UUID, Location] = Field(default_factory=dict)
    global_flags: dict[str, Any] = Field(default_factory=dict)
    current_time: datetime = Field(default_factory=datetime.now)
    evolution_queue: list[dict[str, Any]] = Field(default_factory=list)
    state_data_json: str | None = None


# --- Session and Action Models ---


class GameSession(BaseModel):
    """Metadata for a saved game session."""

    session_id: UUID = Field(default_factory=uuid4)
    player_state_id: UUID
    world_state_id: UUID
    save_name: str = "New Save"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    game_version: str = "0.1.0"


class ActionResult(BaseModel):
    """Result of executing a player action."""

    success: bool = True
    feedback_message: str = ""

    # State Change Flags/Data
    location_change: bool = False
    new_location_id: UUID | None = None

    inventory_changes: list[dict[str, Any]] | None = None
    knowledge_updates: list[PlayerKnowledge] | None = None
    stat_changes: dict[str, int | float] | None = None
    progress_updates: dict[str, Any] | None = None

    object_changes: list[dict[str, Any]] | None = None
    npc_changes: list[dict[str, Any]] | None = None
    location_state_changes: dict[str, Any] | None = None

    triggers_evolution: bool = False
    evolution_trigger: str | None = None
    evolution_data: dict[str, Any] | None = None
    priority: int = 5
    timestamp: datetime = Field(default_factory=datetime.now)

    # Optionally add the originating command/intent
    command: str | None = None
    processed_input: Any | None = None
