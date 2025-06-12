"""Conversation manager for handling NPC interactions and context tracking."""

import uuid
from typing import Any

from game_loop.database.repositories.conversation import ConversationRepositoryManager
from game_loop.database.session_factory import DatabaseSessionFactory
from game_loop.llm.ollama.client import OllamaClient
from game_loop.search.semantic_search import SemanticSearchService
from game_loop.state.manager import GameStateManager

from .conversation_models import (
    ConversationContext,
    ConversationExchange,
    ConversationResult,
    ConversationStatus,
    MessageType,
    NPCPersonality,
)
from .knowledge_extractor import KnowledgeExtractor


class ConversationManager:
    """Manages NPC conversations with context tracking and personality."""

    def __init__(
        self,
        llm_client: OllamaClient,
        game_state_manager: GameStateManager,
        semantic_search: SemanticSearchService,
        session_factory: DatabaseSessionFactory,
    ):
        self.llm_client = llm_client
        self.game_state_manager = game_state_manager
        self.semantic_search = semantic_search
        self.session_factory = session_factory
        self.knowledge_extractor = KnowledgeExtractor(llm_client, semantic_search)

        # Cache for frequently accessed personalities
        self._personality_cache: dict[uuid.UUID, NPCPersonality] = {}

    async def _ensure_default_personalities(self) -> None:
        """Ensure default NPC personalities exist in database."""
        async with self.session_factory.get_session() as session:
            repo_manager = ConversationRepositoryManager(session)

            # Check if guard captain exists
            guard_uuid = uuid.UUID("550e8400-e29b-41d4-a716-446655440001")
            guard_personality = await repo_manager.npc_personalities.get_by_npc_id(
                guard_uuid
            )
            if not guard_personality:
                await repo_manager.npc_personalities.create_personality(
                    npc_id=guard_uuid,
                    traits={
                        "authoritative": 0.9,
                        "helpful": 0.7,
                        "serious": 0.8,
                        "protective": 0.9,
                        "formal": 0.8,
                    },
                    knowledge_areas=[
                        "security",
                        "castle_layout",
                        "protocols",
                        "recent_events",
                    ],
                    speech_patterns={
                        "formality": "high",
                        "directness": "high",
                        "verbosity": "medium",
                    },
                    background_story="A veteran guard who has served the castle for many years. Known for being strict but fair.",
                    default_mood="professional",
                )

            # Check if village elder exists
            elder_uuid = uuid.UUID("550e8400-e29b-41d4-a716-446655440002")
            elder_personality = await repo_manager.npc_personalities.get_by_npc_id(
                elder_uuid
            )
            if not elder_personality:
                await repo_manager.npc_personalities.create_personality(
                    npc_id=elder_uuid,
                    traits={
                        "wise": 0.9,
                        "patient": 0.8,
                        "talkative": 0.7,
                        "helpful": 0.9,
                        "cryptic": 0.6,
                    },
                    knowledge_areas=[
                        "history",
                        "lore",
                        "village_matters",
                        "ancient_secrets",
                    ],
                    speech_patterns={
                        "formality": "medium",
                        "directness": "low",
                        "verbosity": "high",
                    },
                    background_story="The wise elder of the village, keeper of ancient knowledge and local history.",
                    default_mood="contemplative",
                )

    async def start_conversation(
        self,
        player_id: uuid.UUID,
        npc_id: uuid.UUID,
        context: dict[str, Any],
    ) -> ConversationContext:
        """Start a new conversation with an NPC."""
        async with self.session_factory.get_session() as session:
            repo_manager = ConversationRepositoryManager(session)

            # Ensure default personalities exist
            await self._ensure_default_personalities()

            # Get or create NPC personality
            npc_personality = await self._get_or_create_personality(
                npc_id, repo_manager
            )

            # Check for existing active conversation
            existing_conversation = await repo_manager.contexts.get_active_conversation(
                player_id, npc_id
            )
            if existing_conversation:
                # Convert to dataclass model
                return self._convert_db_to_dataclass_context(existing_conversation)

            # Generate initial greeting
            greeting = await self._generate_npc_greeting_for_new_conversation(
                npc_id, npc_personality, context
            )

            # Create new conversation with initial greeting
            db_conversation, db_exchange = (
                await repo_manager.create_complete_conversation(
                    player_id=player_id,
                    npc_id=npc_id,
                    initial_message=greeting,
                    message_type="greeting",
                    initial_mood=npc_personality.default_mood,
                )
            )

            # Convert to dataclass model for return
            return self._convert_db_to_dataclass_context(db_conversation)

    async def process_player_message(
        self,
        conversation_id: uuid.UUID,
        message: str,
        context: dict[str, Any],
    ) -> ConversationResult:
        """Process player message and generate NPC response."""
        try:
            async with self.session_factory.get_session() as session:
                repo_manager = ConversationRepositoryManager(session)

                # Get conversation context
                db_conversation = await repo_manager.contexts.get_with_exchanges(
                    conversation_id
                )
                if not db_conversation or db_conversation.status != "active":
                    return ConversationResult.error_result(
                        "Conversation not found or not active"
                    )

                # Get NPC personality
                npc_personality = await self._get_personality(
                    db_conversation.npc_id, repo_manager
                )
                if not npc_personality:
                    return ConversationResult.error_result("NPC personality not found")

                # Add player message to database
                player_exchange = await repo_manager.exchanges.create_exchange(
                    conversation_id=conversation_id,
                    speaker_id=db_conversation.player_id,
                    message_text=message,
                    message_type=self._classify_message_type_str(message),
                )

                # Convert to dataclass for processing
                conversation = self._convert_db_to_dataclass_context(db_conversation)

                # Generate NPC response
                npc_response_text = await self._generate_npc_response(
                    conversation, npc_personality, message, context
                )

                # Determine response emotion/mood
                response_emotion = await self._determine_response_emotion(
                    message, npc_response_text, conversation.mood, npc_personality
                )

                # Create NPC response exchange in database
                npc_exchange_db = await repo_manager.exchanges.create_exchange(
                    conversation_id=conversation_id,
                    speaker_id=db_conversation.npc_id,
                    message_text=npc_response_text,
                    message_type="statement",
                    emotion=response_emotion,
                )

                # Calculate relationship change
                relationship_change = await self._calculate_relationship_change(
                    message, npc_personality, conversation
                )

                # Update conversation mood and relationship
                if response_emotion != db_conversation.mood:
                    db_conversation.update_mood(response_emotion)
                if relationship_change != 0.0:
                    db_conversation.update_relationship(relationship_change)

                # Update NPC personality relationship
                if relationship_change != 0.0:
                    await repo_manager.npc_personalities.update_relationship(
                        db_conversation.npc_id,
                        db_conversation.player_id,
                        relationship_change,
                    )

                # Extract knowledge from the conversation
                knowledge_extracted = (
                    await self.knowledge_extractor.extract_information(conversation)
                )

                # Store extracted knowledge in database
                for knowledge in knowledge_extracted:
                    await repo_manager.knowledge.create_knowledge_entry(
                        conversation_id=conversation_id,
                        information_type=knowledge.get("type", "general"),
                        extracted_info=knowledge,
                        confidence_score=knowledge.get("confidence", 0.7),
                        source_exchange_id=npc_exchange_db.exchange_id,
                    )

                # Check if conversation should end
                should_end = await self._should_end_conversation(message, conversation)
                if should_end:
                    await repo_manager.contexts.end_conversation(
                        conversation_id, "natural_end"
                    )

                # Convert database exchange back to dataclass
                npc_exchange = ConversationExchange(
                    exchange_id=str(npc_exchange_db.exchange_id),
                    speaker_id=str(npc_exchange_db.speaker_id),
                    message_text=npc_exchange_db.message_text,
                    message_type=MessageType(npc_exchange_db.message_type),
                    emotion=npc_exchange_db.emotion,
                    timestamp=npc_exchange_db.timestamp.timestamp(),
                )

                return ConversationResult.success_result(
                    npc_response=npc_exchange,
                    relationship_change=relationship_change,
                    mood_change=(
                        response_emotion
                        if response_emotion != db_conversation.mood
                        else None
                    ),
                    knowledge_extracted=knowledge_extracted,
                )

        except Exception as e:
            return ConversationResult.error_result(
                f"Error processing message: {str(e)}"
            )

    async def end_conversation(
        self,
        conversation_id: uuid.UUID,
        reason: str = "player_ended",
    ) -> dict[str, Any]:
        """End conversation and extract learned information."""
        async with self.session_factory.get_session() as session:
            repo_manager = ConversationRepositoryManager(session)

            # Get conversation
            db_conversation = await repo_manager.contexts.get_with_exchanges(
                conversation_id
            )
            if not db_conversation:
                return {"error": "Conversation not found"}

            # Convert to dataclass for knowledge extraction
            conversation = self._convert_db_to_dataclass_context(db_conversation)

            # Extract final knowledge
            knowledge_extracted = await self.knowledge_extractor.extract_information(
                conversation
            )

            # Store final knowledge in database
            for knowledge in knowledge_extracted:
                await repo_manager.knowledge.create_knowledge_entry(
                    conversation_id=conversation_id,
                    information_type=knowledge.get("type", "general"),
                    extracted_info=knowledge,
                    confidence_score=knowledge.get("confidence", 0.7),
                )

            # End the conversation in database
            ended_conversation = await repo_manager.contexts.end_conversation(
                conversation_id, reason
            )

            return {
                "conversation_id": str(conversation_id),
                "ended_at": (
                    ended_conversation.ended_at.isoformat()
                    if ended_conversation and ended_conversation.ended_at
                    else None
                ),
                "reason": reason,
                "final_relationship": (
                    float(ended_conversation.relationship_level)
                    if ended_conversation
                    else 0.0
                ),
                "knowledge_extracted": knowledge_extracted,
                "exchange_count": (
                    ended_conversation.get_exchange_count() if ended_conversation else 0
                ),
            }

    async def _generate_npc_greeting_for_new_conversation(
        self,
        npc_id: uuid.UUID,
        personality: NPCPersonality,
        context: dict[str, Any],
    ) -> str:
        """Generate an initial greeting from the NPC."""
        greeting_prompt = f"""
        You are an NPC with the following personality:
        
        Traits: {personality.traits}
        Background: {personality.background_story}
        Default Mood: {personality.default_mood}
        Knowledge Areas: {personality.knowledge_areas}
        
        Generate a greeting for a player who just approached you. The greeting should:
        1. Reflect your personality and mood
        2. Be appropriate for the setting
        3. Be welcoming but stay in character
        4. Be 1-2 sentences long
        
        Current setting: {context.get('location', 'unknown location')}
        Time of day: {context.get('time_of_day', 'unknown')}
        
        Provide only the greeting dialogue, without quotes or character name.
        """

        try:
            response = await self.llm_client.generate_response(
                greeting_prompt, model="qwen2.5:3b"
            )
            return response.strip()
        except Exception:
            return "Hello there, traveler. How can I help you?"

    async def _generate_npc_response(
        self,
        conversation: ConversationContext,
        personality: NPCPersonality,
        player_message: str,
        context: dict[str, Any],
    ) -> str:
        """Generate contextual NPC response using LLM."""
        # Format conversation history for context
        history_text = ""
        recent_exchanges = conversation.get_recent_exchanges(5)
        for exchange in recent_exchanges[-5:]:  # Last 5 exchanges
            speaker = (
                "Player"
                if exchange.speaker_id == conversation.player_id
                else conversation.npc_id
            )
            history_text += f"{speaker}: {exchange.message_text}\n"

        response_prompt = f"""
        You are {conversation.npc_id}, an NPC in a text adventure game with the following personality:

        Personality Traits: {personality.traits}
        Background: {personality.background_story}
        Current Mood: {conversation.mood}
        Relationship with Player: {conversation.relationship_level:.2f} (-1.0 is hostile, 0 is neutral, 1.0 is friendly)
        Knowledge Areas: {personality.knowledge_areas}

        Recent Conversation History:
        {history_text}

        Current Location: {context.get('location', 'unknown')}
        
        The player just said: "{player_message}"

        Respond as this character would, considering:
        1. Your personality traits and current mood
        2. Your relationship with the player
        3. What you would realistically know based on your knowledge areas
        4. The conversation context and history
        5. Stay consistent with your established character

        Provide only your dialogue response, without quotes or character name.
        Keep responses conversational and engaging, typically 1-3 sentences.
        """

        try:
            response = await self.llm_client.generate_response(
                response_prompt, model="qwen2.5:3b"
            )
            return response.strip()
        except Exception:
            return "I'm not sure how to respond to that."

    async def _determine_response_emotion(
        self,
        player_message: str,
        npc_response: str,
        current_mood: str,
        personality: NPCPersonality,
    ) -> str:
        """Determine the emotion/mood for the NPC's response."""
        # Simple emotion determination based on message content and personality
        # In a more sophisticated system, this would use NLU

        message_lower = player_message.lower()
        response_lower = npc_response.lower()

        # Check for emotional keywords
        if any(word in message_lower for word in ["thank", "please", "help"]):
            if personality.get_trait_strength("helpful") > 0.7:
                return "pleased"

        if any(word in message_lower for word in ["threat", "attack", "kill"]):
            if personality.get_trait_strength("protective") > 0.7:
                return "alarmed"

        if any(word in response_lower for word in ["unfortunately", "sorry", "cannot"]):
            return "regretful"

        if any(word in response_lower for word in ["excellent", "wonderful", "great"]):
            return "pleased"

        # Default to current mood or neutral
        return current_mood if current_mood != "neutral" else "neutral"

    async def _calculate_relationship_change(
        self,
        player_message: str,
        personality: NPCPersonality,
        conversation: ConversationContext,
    ) -> float:
        """Calculate how the relationship should change based on the interaction."""
        message_lower = player_message.lower()
        change = 0.0

        # Positive interactions
        if any(word in message_lower for word in ["thank", "please", "help"]):
            if personality.get_trait_strength("helpful") > 0.5:
                change += 0.1

        if any(word in message_lower for word in ["compliment", "good", "excellent"]):
            change += 0.05

        # Negative interactions
        if any(word in message_lower for word in ["rude", "stupid", "idiot"]):
            change -= 0.2

        if any(word in message_lower for word in ["threat", "kill", "attack"]):
            change -= 0.5

        # Cap the change
        return max(-0.5, min(0.5, change))

    def _classify_message_type(self, message: str) -> MessageType:
        """Classify the type of player message."""
        message_lower = message.lower().strip()

        if any(
            word in message_lower
            for word in ["hello", "hi", "greetings", "good morning"]
        ):
            return MessageType.GREETING

        if any(
            word in message_lower for word in ["bye", "goodbye", "farewell", "see you"]
        ):
            return MessageType.FAREWELL

        if message_lower.endswith("?") or message_lower.startswith(
            ("what", "how", "why", "where", "when", "who")
        ):
            return MessageType.QUESTION

        return MessageType.STATEMENT

    def _classify_message_type_str(self, message: str) -> str:
        """Classify the type of player message and return as string."""
        return self._classify_message_type(message).value

    async def _should_end_conversation(
        self, player_message: str, conversation: ConversationContext
    ) -> bool:
        """Determine if the conversation should end naturally."""
        message_lower = player_message.lower()

        # Explicit farewell
        if any(
            word in message_lower for word in ["bye", "goodbye", "farewell", "leave"]
        ):
            return True

        # Very long conversation (more than 20 exchanges)
        if conversation.get_exchange_count() > 20:
            return True

        return False

    async def _get_or_create_personality(
        self, npc_id: uuid.UUID, repo_manager: ConversationRepositoryManager
    ) -> NPCPersonality:
        """Get NPC personality from database or create default."""
        # Check cache first
        if npc_id in self._personality_cache:
            return self._personality_cache[npc_id]

        # Get from database
        db_personality = await repo_manager.npc_personalities.get_by_npc_id(npc_id)
        if db_personality:
            personality = self._convert_db_to_dataclass_personality(db_personality)
            self._personality_cache[npc_id] = personality
            return personality

        # Create default personality
        db_personality = await repo_manager.npc_personalities.create_personality(
            npc_id=npc_id,
            traits={"helpful": 0.5, "talkative": 0.5, "friendly": 0.5},
            knowledge_areas=["general"],
            speech_patterns={"formality": "medium", "directness": "medium"},
            background_story="A resident of the area.",
        )

        personality = self._convert_db_to_dataclass_personality(db_personality)
        self._personality_cache[npc_id] = personality
        return personality

    async def _get_personality(
        self, npc_id: uuid.UUID, repo_manager: ConversationRepositoryManager
    ) -> NPCPersonality | None:
        """Get NPC personality from cache or database."""
        if npc_id in self._personality_cache:
            return self._personality_cache[npc_id]

        db_personality = await repo_manager.npc_personalities.get_by_npc_id(npc_id)
        if db_personality:
            personality = self._convert_db_to_dataclass_personality(db_personality)
            self._personality_cache[npc_id] = personality
            return personality

        return None

    def _convert_db_to_dataclass_personality(self, db_personality) -> NPCPersonality:
        """Convert database personality model to dataclass."""
        return NPCPersonality(
            npc_id=str(db_personality.npc_id),
            traits=db_personality.traits or {},
            knowledge_areas=db_personality.knowledge_areas or [],
            speech_patterns=db_personality.speech_patterns or {},
            relationships=db_personality.relationships or {},
            background_story=db_personality.background_story,
            default_mood=db_personality.default_mood,
        )

    def _convert_db_to_dataclass_context(self, db_context) -> ConversationContext:
        """Convert database conversation context to dataclass."""
        # Convert exchanges
        exchanges = []
        if hasattr(db_context, "exchanges") and db_context.exchanges:
            for db_exchange in db_context.exchanges:
                exchange = ConversationExchange(
                    exchange_id=str(db_exchange.exchange_id),
                    speaker_id=str(db_exchange.speaker_id),
                    message_text=db_exchange.message_text,
                    message_type=MessageType(db_exchange.message_type),
                    emotion=db_exchange.emotion,
                    timestamp=(
                        db_exchange.timestamp.timestamp()
                        if db_exchange.timestamp
                        else 0
                    ),
                    metadata=db_exchange.metadata or {},
                )
                exchanges.append(exchange)

        # Convert status
        status_map = {
            "active": ConversationStatus.ACTIVE,
            "ended": ConversationStatus.ENDED,
            "paused": ConversationStatus.PAUSED,
            "abandoned": ConversationStatus.ABANDONED,
        }

        return ConversationContext(
            conversation_id=str(db_context.conversation_id),
            player_id=str(db_context.player_id),
            npc_id=str(db_context.npc_id),
            topic=db_context.topic,
            mood=db_context.mood,
            relationship_level=float(db_context.relationship_level),
            conversation_history=exchanges,
            context_data=db_context.context_data or {},
            status=status_map.get(db_context.status, ConversationStatus.ACTIVE),
            started_at=(
                db_context.started_at.timestamp() if db_context.started_at else 0
            ),
            last_updated=(
                db_context.last_updated.timestamp() if db_context.last_updated else 0
            ),
            ended_at=db_context.ended_at.timestamp() if db_context.ended_at else None,
        )

    async def get_active_conversations(
        self, player_id: uuid.UUID
    ) -> list[ConversationContext]:
        """Get all active conversations for a player."""
        async with self.session_factory.get_session() as session:
            repo_manager = ConversationRepositoryManager(session)
            db_conversations = await repo_manager.contexts.get_player_conversations(
                player_id, status="active"
            )

            return [
                self._convert_db_to_dataclass_context(db_conv)
                for db_conv in db_conversations
            ]

    async def get_npc_personality(self, npc_id: uuid.UUID) -> NPCPersonality | None:
        """Get NPC personality by ID."""
        async with self.session_factory.get_session() as session:
            repo_manager = ConversationRepositoryManager(session)
            return await self._get_personality(npc_id, repo_manager)
