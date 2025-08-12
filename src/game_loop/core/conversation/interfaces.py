"""Abstract interfaces for conversation system components."""

import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .memory_integration import MemoryContext, MemoryRetrievalResult
    from .conversation_models import ConversationContext, NPCPersonality


class ConversationMemoryInterface(ABC):
    """
    Abstract interface for memory integration in conversations.

    This interface follows the Interface Segregation Principle by only
    defining the methods actually needed by conversation flow management.
    """

    @abstractmethod
    async def extract_memory_context(
        self,
        conversation: "ConversationContext",
        player_message: str,
        npc_personality: "NPCPersonality",
    ) -> "MemoryContext":
        """
        Extract context for memory queries from current conversation state.

        Args:
            conversation: Current conversation context
            player_message: Latest player message
            npc_personality: NPC personality data

        Returns:
            MemoryContext: Extracted context for memory retrieval
        """
        pass

    @abstractmethod
    async def retrieve_relevant_memories(
        self,
        memory_context: "MemoryContext",
        npc_id: uuid.UUID,
        query_embedding: list[float] | None = None,
    ) -> "MemoryRetrievalResult":
        """
        Retrieve relevant memories based on context.

        Args:
            memory_context: Context for memory retrieval
            npc_id: NPC identifier
            query_embedding: Optional pre-computed embedding

        Returns:
            MemoryRetrievalResult: Retrieved memories and metadata
        """
        pass
