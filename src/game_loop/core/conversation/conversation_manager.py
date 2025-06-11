"""Conversation manager for handling NPC interactions and context tracking."""

from typing import Any

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
    ):
        self.llm_client = llm_client
        self.game_state_manager = game_state_manager
        self.semantic_search = semantic_search
        self.knowledge_extractor = KnowledgeExtractor(llm_client, semantic_search)

        # In-memory storage for active conversations and NPC personalities
        # In production, these would be backed by database
        self._active_conversations: dict[str, ConversationContext] = {}
        self._npc_personalities: dict[str, NPCPersonality] = {}

        # Initialize default personalities
        self._initialize_default_personalities()

    def _initialize_default_personalities(self) -> None:
        """Initialize some default NPC personalities for testing."""
        # Guard captain personality
        guard_personality = NPCPersonality(
            npc_id="guard_captain",
            traits={
                "authoritative": 0.9,
                "helpful": 0.7,
                "serious": 0.8,
                "protective": 0.9,
                "formal": 0.8,
            },
            knowledge_areas=["security", "castle_layout", "protocols", "recent_events"],
            speech_patterns={
                "formality": "high",
                "directness": "high",
                "verbosity": "medium",
            },
            relationships={},
            background_story="A veteran guard who has served the castle for many years. Known for being strict but fair.",
            default_mood="professional",
        )
        self._npc_personalities["guard_captain"] = guard_personality

        # Village elder personality
        elder_personality = NPCPersonality(
            npc_id="village_elder",
            traits={
                "wise": 0.9,
                "patient": 0.8,
                "talkative": 0.7,
                "helpful": 0.9,
                "cryptic": 0.6,
            },
            knowledge_areas=["history", "lore", "village_matters", "ancient_secrets"],
            speech_patterns={
                "formality": "medium",
                "directness": "low",
                "verbosity": "high",
            },
            relationships={},
            background_story="The wise elder of the village, keeper of ancient knowledge and local history.",
            default_mood="contemplative",
        )
        self._npc_personalities["village_elder"] = elder_personality

    async def start_conversation(
        self,
        player_id: str,
        npc_id: str,
        context: dict[str, Any],
    ) -> ConversationContext:
        """Start a new conversation with an NPC."""
        # Get or create NPC personality
        npc_personality = self._npc_personalities.get(npc_id)
        if not npc_personality:
            npc_personality = self._create_default_personality(npc_id)

        # Check for existing active conversation
        existing_conversation = self._find_active_conversation(player_id, npc_id)
        if existing_conversation:
            return existing_conversation

        # Create new conversation context
        conversation = ConversationContext.create(
            player_id=player_id,
            npc_id=npc_id,
            initial_mood=npc_personality.default_mood,
            relationship_level=npc_personality.get_relationship_level(player_id),
        )

        # Generate initial greeting
        greeting = await self._generate_npc_greeting(conversation, npc_personality, context)
        greeting_exchange = ConversationExchange.create_npc_message(
            npc_id=npc_id,
            message_text=greeting,
            message_type=MessageType.GREETING,
            emotion=conversation.mood,
        )
        conversation.add_exchange(greeting_exchange)

        # Store active conversation
        self._active_conversations[conversation.conversation_id] = conversation

        return conversation

    async def process_player_message(
        self,
        conversation_id: str,
        message: str,
        context: dict[str, Any],
    ) -> ConversationResult:
        """Process player message and generate NPC response."""
        try:
            # Get conversation context
            conversation = self._active_conversations.get(conversation_id)
            if not conversation:
                return ConversationResult.error_result("Conversation not found")

            if conversation.status != ConversationStatus.ACTIVE:
                return ConversationResult.error_result("Conversation is not active")

            # Get NPC personality
            npc_personality = self._npc_personalities.get(conversation.npc_id)
            if not npc_personality:
                return ConversationResult.error_result("NPC personality not found")

            # Add player message to conversation
            player_exchange = ConversationExchange.create_player_message(
                player_id=conversation.player_id,
                message_text=message,
                message_type=self._classify_message_type(message),
            )
            conversation.add_exchange(player_exchange)

            # Generate NPC response
            npc_response_text = await self._generate_npc_response(
                conversation, npc_personality, message, context
            )

            # Determine response emotion/mood
            response_emotion = await self._determine_response_emotion(
                message, npc_response_text, conversation.mood, npc_personality
            )

            # Create NPC response exchange
            npc_exchange = ConversationExchange.create_npc_message(
                npc_id=conversation.npc_id,
                message_text=npc_response_text,
                message_type=MessageType.STATEMENT,
                emotion=response_emotion,
            )
            conversation.add_exchange(npc_exchange)

            # Update relationship and mood
            relationship_change = await self._calculate_relationship_change(
                message, npc_personality, conversation
            )
            conversation.update_relationship(relationship_change)

            if response_emotion != conversation.mood:
                conversation.update_mood(response_emotion)

            # Extract knowledge from the conversation
            knowledge_extracted = await self.knowledge_extractor.extract_information(
                conversation
            )

            # Check if conversation should end
            should_end = await self._should_end_conversation(message, conversation)
            if should_end:
                conversation.end_conversation("natural_end")
                del self._active_conversations[conversation_id]

            return ConversationResult.success_result(
                npc_response=npc_exchange,
                relationship_change=relationship_change,
                mood_change=response_emotion if response_emotion != conversation.mood else None,
                knowledge_extracted=knowledge_extracted,
            )

        except Exception as e:
            return ConversationResult.error_result(f"Error processing message: {str(e)}")

    async def end_conversation(
        self,
        conversation_id: str,
        reason: str = "player_ended",
    ) -> dict[str, Any]:
        """End conversation and extract learned information."""
        conversation = self._active_conversations.get(conversation_id)
        if not conversation:
            return {"error": "Conversation not found"}

        # Extract final knowledge
        knowledge_extracted = await self.knowledge_extractor.extract_information(
            conversation
        )

        # Update NPC personality relationships
        npc_personality = self._npc_personalities.get(conversation.npc_id)
        if npc_personality:
            npc_personality.update_relationship(
                conversation.player_id, conversation.relationship_level
            )

        # End the conversation
        conversation.end_conversation(reason)

        # Remove from active conversations
        if conversation_id in self._active_conversations:
            del self._active_conversations[conversation_id]

        return {
            "conversation_id": conversation_id,
            "ended_at": conversation.ended_at,
            "reason": reason,
            "final_relationship": conversation.relationship_level,
            "knowledge_extracted": knowledge_extracted,
            "exchange_count": conversation.get_exchange_count(),
        }

    async def _generate_npc_greeting(
        self,
        conversation: ConversationContext,
        personality: NPCPersonality,
        context: dict[str, Any],
    ) -> str:
        """Generate an initial greeting from the NPC."""
        greeting_prompt = f"""
        You are {conversation.npc_id}, an NPC with the following personality:
        
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
            speaker = "Player" if exchange.speaker_id == conversation.player_id else conversation.npc_id
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

        if any(word in message_lower for word in ["hello", "hi", "greetings", "good morning"]):
            return MessageType.GREETING

        if any(word in message_lower for word in ["bye", "goodbye", "farewell", "see you"]):
            return MessageType.FAREWELL

        if message_lower.endswith("?") or message_lower.startswith(("what", "how", "why", "where", "when", "who")):
            return MessageType.QUESTION

        return MessageType.STATEMENT

    async def _should_end_conversation(
        self, player_message: str, conversation: ConversationContext
    ) -> bool:
        """Determine if the conversation should end naturally."""
        message_lower = player_message.lower()

        # Explicit farewell
        if any(word in message_lower for word in ["bye", "goodbye", "farewell", "leave"]):
            return True

        # Very long conversation (more than 20 exchanges)
        if conversation.get_exchange_count() > 20:
            return True

        return False

    def _find_active_conversation(
        self, player_id: str, npc_id: str
    ) -> ConversationContext | None:
        """Find existing active conversation between player and NPC."""
        for conversation in self._active_conversations.values():
            if (
                conversation.player_id == player_id
                and conversation.npc_id == npc_id
                and conversation.status == ConversationStatus.ACTIVE
            ):
                return conversation
        return None

    def _create_default_personality(self, npc_id: str) -> NPCPersonality:
        """Create a default personality for an unknown NPC."""
        personality = NPCPersonality(
            npc_id=npc_id,
            traits={
                "helpful": 0.5,
                "talkative": 0.5,
                "friendly": 0.5,
            },
            knowledge_areas=["general"],
            speech_patterns={"formality": "medium", "directness": "medium"},
            relationships={},
            background_story=f"A resident of the area known as {npc_id}.",
        )
        self._npc_personalities[npc_id] = personality
        return personality

    def get_active_conversations(self, player_id: str) -> list[ConversationContext]:
        """Get all active conversations for a player."""
        return [
            conv for conv in self._active_conversations.values()
            if conv.player_id == player_id and conv.status == ConversationStatus.ACTIVE
        ]

    def get_npc_personality(self, npc_id: str) -> NPCPersonality | None:
        """Get NPC personality by ID."""
        return self._npc_personalities.get(npc_id)
