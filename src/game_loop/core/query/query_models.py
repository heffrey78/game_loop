"""Models for query processing system."""

import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any


class QueryType(Enum):
    """Types of queries that can be processed."""

    WORLD_INFO = "world_info"
    OBJECT_INFO = "object_info"
    NPC_INFO = "npc_info"
    LOCATION_INFO = "location_info"
    HELP = "help"
    STATUS = "status"
    INVENTORY = "inventory"
    QUEST_INFO = "quest_info"


@dataclass
class QueryRequest:
    """Represents a player information request."""

    query_id: str
    player_id: str
    query_text: str
    query_type: QueryType
    context: dict[str, Any]
    timestamp: float

    @classmethod
    def create(
        cls,
        player_id: str,
        query_text: str,
        query_type: QueryType,
        context: dict[str, Any] | None = None,
    ) -> "QueryRequest":
        """Create a new query request."""
        return cls(
            query_id=str(uuid.uuid4()),
            player_id=player_id,
            query_text=query_text,
            query_type=query_type,
            context=context or {},
            timestamp=time.time(),
        )


@dataclass
class QueryResponse:
    """Response to a player query."""

    success: bool
    response_text: str
    information_type: str
    sources: list[str]  # What sources provided the information
    related_queries: list[str]  # Suggested follow-up questions
    confidence: float  # How confident we are in the response
    errors: list[str] | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def success_response(
        cls,
        response_text: str,
        information_type: str,
        sources: list[str] | None = None,
        related_queries: list[str] | None = None,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> "QueryResponse":
        """Create a successful query response."""
        return cls(
            success=True,
            response_text=response_text,
            information_type=information_type,
            sources=sources or [],
            related_queries=related_queries or [],
            confidence=confidence,
            errors=None,
            metadata=metadata,
        )

    @classmethod
    def error_response(
        cls,
        error_message: str,
        errors: list[str] | None = None,
        information_type: str = "error",
    ) -> "QueryResponse":
        """Create an error query response."""
        return cls(
            success=False,
            response_text=error_message,
            information_type=information_type,
            sources=[],
            related_queries=[],
            confidence=0.0,
            errors=errors or [error_message],
        )


@dataclass
class InformationSource:
    """Represents a source of information for a query."""

    source_id: str
    source_type: str  # "entity", "location", "object", "quest", "database"
    source_name: str
    content: str
    relevance_score: float
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "content": self.content,
            "relevance_score": self.relevance_score,
            "metadata": self.metadata or {},
        }


@dataclass
class QueryContext:
    """Context information for processing queries."""

    player_id: str
    current_location_id: str | None = None
    recent_actions: list[str] | None = None
    active_quests: list[str] | None = None
    inventory_items: list[str] | None = None
    conversation_context: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for use in prompts and processing."""
        return {
            "player_id": self.player_id,
            "current_location": self.current_location_id,
            "recent_actions": self.recent_actions or [],
            "active_quests": self.active_quests or [],
            "inventory_items": self.inventory_items or [],
            "conversation_context": self.conversation_context or {},
            "metadata": self.metadata or {},
        }