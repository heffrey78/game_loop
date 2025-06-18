"""
NPC Dialogue Enhancement System.

This package provides enhanced dialogue capabilities for NPCs including:
- Personality-driven responses based on NPC archetypes
- Conversation memory and relationship tracking
- Contextual knowledge based on NPC roles and locations
"""

from .knowledge_engine import NPCKnowledgeEngine
from .memory_manager import ConversationMemoryManager
from .personality_engine import NPCPersonalityEngine

__all__ = ["NPCPersonalityEngine", "ConversationMemoryManager", "NPCKnowledgeEngine"]
