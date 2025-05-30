"""Repository modules for database access."""

from .base import BaseRepository
from .game_state import (
    EvolutionEventRepository,
    GameSessionRepository,
    WorldRuleRepository,
)
from .player import (
    PlayerHistoryRepository,
    PlayerInventoryRepository,
    PlayerKnowledgeRepository,
    PlayerRepository,
    PlayerSkillRepository,
)
from .world import (
    LocationConnectionRepository,
    LocationRepository,
    NPCRepository,
    ObjectRepository,
    QuestRepository,
    RegionRepository,
)

__all__ = [
    "BaseRepository",
    "PlayerRepository",
    "PlayerInventoryRepository",
    "PlayerKnowledgeRepository",
    "PlayerSkillRepository",
    "PlayerHistoryRepository",
    "RegionRepository",
    "LocationRepository",
    "ObjectRepository",
    "NPCRepository",
    "QuestRepository",
    "LocationConnectionRepository",
    "GameSessionRepository",
    "WorldRuleRepository",
    "EvolutionEventRepository",
]
