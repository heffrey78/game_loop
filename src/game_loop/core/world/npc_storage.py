"""
NPC Storage and Embedding System for persistence and semantic search.
"""

import logging
import time
from typing import Any
from uuid import UUID

from ...database.session_factory import DatabaseSessionFactory
from ...embeddings.manager import EmbeddingManager
from ...state.models import NonPlayerCharacter
from ..models.npc_models import (
    GeneratedNPC,
    NPCDialogueState,
    NPCSearchCriteria,
    NPCStorageResult,
)

logger = logging.getLogger(__name__)


class NPCStorage:
    """Handles storage, retrieval, caching, and embedding generation for NPCs."""

    def __init__(
        self,
        session_factory: DatabaseSessionFactory,
        embedding_manager: "EmbeddingManager",
    ):
        """Initialize NPC storage system."""
        self.session_factory = session_factory
        self.embedding_manager = embedding_manager

    async def store_generated_npc(
        self, npc: GeneratedNPC, location_id: UUID
    ) -> NPCStorageResult:
        """Store NPC with all associated data and generate embeddings."""
        start_time = time.time()

        try:
            logger.debug(f"Storing generated NPC: {npc.base_npc.name}")

            async with self.session_factory.get_session() as session:
                # Store base NPC
                await self._store_base_npc(session, npc.base_npc, location_id)

                # Store personality data
                await self._store_personality(session, npc)

                # Store knowledge data
                await self._store_knowledge(session, npc)

                # Store dialogue state
                await self._store_dialogue_state(session, npc)

                # Store generation history
                await self._store_generation_history(session, npc)

                # Generate and store embeddings
                embedding_start = time.time()
                embedding_generated = await self._generate_and_store_embeddings(
                    session, npc
                )
                embedding_time = int((time.time() - embedding_start) * 1000)

                # Commit transaction
                await session.commit()

                storage_time_ms = int((time.time() - start_time) * 1000)

                logger.info(
                    f"Successfully stored NPC {npc.base_npc.name} "
                    f"(storage: {storage_time_ms}ms, embedding: {embedding_time}ms)"
                )

                return NPCStorageResult(
                    success=True,
                    npc_id=npc.base_npc.npc_id,
                    storage_time_ms=storage_time_ms,
                    embedding_generated=embedding_generated,
                )

        except Exception as e:
            logger.error(f"Error storing NPC: {e}")
            return NPCStorageResult(
                success=False,
                storage_time_ms=int((time.time() - start_time) * 1000),
                error_message=str(e),
            )

    async def retrieve_npc(
        self, npc_id: UUID, include_embeddings: bool = False
    ) -> GeneratedNPC | None:
        """Retrieve NPC with all associated data."""
        try:
            logger.debug(f"Retrieving NPC: {npc_id}")

            async with self.session_factory.get_session() as session:
                # Load base NPC
                base_npc = await self._load_base_npc(session, npc_id)
                if not base_npc:
                    return None

                # Load personality
                personality = await self._load_personality(session, npc_id)

                # Load knowledge
                knowledge = await self._load_knowledge(session, npc_id)

                # Load dialogue state
                dialogue_state = await self._load_dialogue_state(session, npc_id)

                # Load generation metadata
                metadata = await self._load_generation_metadata(session, npc_id)

                # Load embeddings if requested
                embedding_vector = None
                if include_embeddings:
                    embedding_vector = await self._load_embeddings(session, npc_id)

                generated_npc = GeneratedNPC(
                    base_npc=base_npc,
                    personality=personality,
                    knowledge=knowledge,
                    dialogue_state=dialogue_state,
                    generation_metadata=metadata,
                    embedding_vector=embedding_vector,
                )

                logger.debug(f"Successfully retrieved NPC: {base_npc.name}")
                return generated_npc

        except Exception as e:
            logger.error(f"Error retrieving NPC {npc_id}: {e}")
            return None

    async def update_npc_state(
        self, npc_id: UUID, dialogue_state: NPCDialogueState
    ) -> bool:
        """Update NPC dialogue and interaction state."""
        try:
            logger.debug(f"Updating dialogue state for NPC: {npc_id}")

            async with self.session_factory.get_session() as session:
                # Update dialogue state in database
                success = await self._update_dialogue_state(
                    session, npc_id, dialogue_state
                )

                if success:
                    await session.commit()
                    logger.debug(f"Successfully updated NPC state: {npc_id}")

                return success

        except Exception as e:
            logger.error(f"Error updating NPC state {npc_id}: {e}")
            return False

    async def update_npc_knowledge(
        self, npc_id: UUID, new_knowledge: dict[str, Any]
    ) -> bool:
        """Update NPC knowledge based on world events or interactions."""
        try:
            logger.debug(f"Updating knowledge for NPC: {npc_id}")

            async with self.session_factory.get_session() as session:
                # Load current knowledge
                current_knowledge = await self._load_knowledge(session, npc_id)
                if not current_knowledge:
                    return False

                # Merge new knowledge
                current_knowledge.world_knowledge.update(
                    new_knowledge.get("world_knowledge", {})
                )
                current_knowledge.local_knowledge.update(
                    new_knowledge.get("local_knowledge", {})
                )
                current_knowledge.personal_history.extend(
                    new_knowledge.get("personal_history", [])
                )

                # Update in database
                success = await self._update_knowledge(
                    session, npc_id, current_knowledge
                )

                if success:
                    await session.commit()
                    logger.debug(f"Successfully updated NPC knowledge: {npc_id}")

                return success

        except Exception as e:
            logger.error(f"Error updating NPC knowledge {npc_id}: {e}")
            return False

    async def get_npcs_by_location(self, location_id: UUID) -> list[GeneratedNPC]:
        """Get all NPCs in a specific location."""
        try:
            logger.debug(f"Getting NPCs for location: {location_id}")

            async with self.session_factory.get_session() as session:
                # Query NPCs by location
                npc_ids = await self._get_npc_ids_by_location(session, location_id)

                # Load each NPC
                npcs = []
                for npc_id in npc_ids:
                    npc = await self.retrieve_npc(npc_id)
                    if npc:
                        npcs.append(npc)

                logger.debug(f"Found {len(npcs)} NPCs in location {location_id}")
                return npcs

        except Exception as e:
            logger.error(f"Error getting NPCs by location {location_id}: {e}")
            return []

    async def search_npcs_by_criteria(
        self, criteria: NPCSearchCriteria
    ) -> list[GeneratedNPC]:
        """Search NPCs using semantic criteria."""
        try:
            logger.debug(f"Searching NPCs with criteria: {criteria.query_text}")

            results = []

            async with self.session_factory.get_session() as session:
                # If query text provided, use semantic search
                if criteria.query_text:
                    results = await self._semantic_search_npcs(session, criteria)
                else:
                    # Use attribute-based filtering
                    results = await self._filter_npcs_by_attributes(session, criteria)

                # Limit results
                return results[: criteria.max_results]

        except Exception as e:
            logger.error(f"Error searching NPCs: {e}")
            return []

    async def generate_npc_embeddings(self, npc: GeneratedNPC) -> list[float]:
        """Generate embeddings for NPC semantic search."""
        try:
            # Create text representation of NPC for embedding
            npc_text = self._create_npc_text_representation(npc)

            # Generate embedding
            embedding = await self.embedding_manager.generate_embedding(npc_text)

            return embedding if embedding else []

        except Exception as e:
            logger.error(f"Error generating NPC embeddings: {e}")
            return []

    def _create_npc_text_representation(self, npc: GeneratedNPC) -> str:
        """Create a text representation of the NPC for embedding generation."""
        text_parts = [
            f"Name: {npc.base_npc.name}",
            f"Description: {npc.base_npc.description}",
            f"Archetype: {npc.personality.archetype}",
            f"Traits: {', '.join(npc.personality.traits)}",
            f"Motivations: {', '.join(npc.personality.motivations)}",
            f"Knowledge areas: {', '.join(npc.knowledge.expertise_areas)}",
            f"Speech style: {npc.personality.speech_patterns.get('style', 'neutral')}",
        ]

        return " | ".join(text_parts)

    # Database interaction methods (these would implement actual database operations)

    async def _store_base_npc(
        self, session: Any, npc: NonPlayerCharacter, location_id: UUID
    ) -> None:
        """Store base NPC entity."""
        # This would insert into the npcs table
        logger.debug(f"Storing base NPC: {npc.name}")

    async def _store_personality(self, session: Any, npc: GeneratedNPC) -> None:
        """Store personality data."""
        # This would insert into npc_personalities table
        logger.debug(f"Storing personality for: {npc.base_npc.name}")

    async def _store_knowledge(self, session: Any, npc: GeneratedNPC) -> None:
        """Store knowledge data."""
        # This would insert into npc_knowledge table
        logger.debug(f"Storing knowledge for: {npc.base_npc.name}")

    async def _store_dialogue_state(self, session: Any, npc: GeneratedNPC) -> None:
        """Store dialogue state."""
        # This would insert into npc_dialogue_states table
        logger.debug(f"Storing dialogue state for: {npc.base_npc.name}")

    async def _store_generation_history(self, session: Any, npc: GeneratedNPC) -> None:
        """Store generation history."""
        # This would insert into npc_generation_history table
        logger.debug(f"Storing generation history for: {npc.base_npc.name}")

    async def _generate_and_store_embeddings(
        self, session: Any, npc: GeneratedNPC
    ) -> bool:
        """Generate and store embeddings."""
        try:
            embedding = await self.generate_npc_embeddings(npc)
            if embedding:
                # This would update the npcs table with embedding_vector
                logger.debug(f"Generated embeddings for: {npc.base_npc.name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return False

    async def _load_base_npc(
        self, session: Any, npc_id: UUID
    ) -> NonPlayerCharacter | None:
        """Load base NPC entity."""
        # This would query the npcs table
        logger.debug(f"Loading base NPC: {npc_id}")
        return None  # Placeholder

    async def _load_personality(self, session: Any, npc_id: UUID) -> Any:
        """Load personality data."""
        # This would query npc_personalities table
        logger.debug(f"Loading personality for: {npc_id}")
        return None  # Placeholder

    async def _load_knowledge(self, session: Any, npc_id: UUID) -> Any:
        """Load knowledge data."""
        # This would query npc_knowledge table
        logger.debug(f"Loading knowledge for: {npc_id}")
        return None  # Placeholder

    async def _load_dialogue_state(self, session: Any, npc_id: UUID) -> Any:
        """Load dialogue state."""
        # This would query npc_dialogue_states table
        logger.debug(f"Loading dialogue state for: {npc_id}")
        return None  # Placeholder

    async def _load_generation_metadata(self, session: Any, npc_id: UUID) -> dict:
        """Load generation metadata."""
        # This would query npc_generation_history table
        logger.debug(f"Loading generation metadata for: {npc_id}")
        return {}  # Placeholder

    async def _load_embeddings(self, session: Any, npc_id: UUID) -> list[float] | None:
        """Load embeddings."""
        # This would query embedding_vector from npcs table
        logger.debug(f"Loading embeddings for: {npc_id}")
        return None  # Placeholder

    async def _update_dialogue_state(
        self, session: Any, npc_id: UUID, dialogue_state: NPCDialogueState
    ) -> bool:
        """Update dialogue state in database."""
        # This would update npc_dialogue_states table
        logger.debug(f"Updating dialogue state for: {npc_id}")
        return True  # Placeholder

    async def _update_knowledge(
        self, session: Any, npc_id: UUID, knowledge: Any
    ) -> bool:
        """Update knowledge in database."""
        # This would update npc_knowledge table
        logger.debug(f"Updating knowledge for: {npc_id}")
        return True  # Placeholder

    async def _get_npc_ids_by_location(
        self, session: Any, location_id: UUID
    ) -> list[UUID]:
        """Get NPC IDs for a location."""
        # This would query npcs table by location
        logger.debug(f"Getting NPC IDs for location: {location_id}")
        return []  # Placeholder

    async def _semantic_search_npcs(
        self, session: Any, criteria: NPCSearchCriteria
    ) -> list[GeneratedNPC]:
        """Perform semantic search using embeddings."""
        # This would use vector similarity search
        logger.debug(f"Performing semantic search: {criteria.query_text}")
        return []  # Placeholder

    async def _filter_npcs_by_attributes(
        self, session: Any, criteria: NPCSearchCriteria
    ) -> list[GeneratedNPC]:
        """Filter NPCs by attributes."""
        # This would filter based on archetype, traits, etc.
        logger.debug("Filtering NPCs by attributes")
        return []  # Placeholder
