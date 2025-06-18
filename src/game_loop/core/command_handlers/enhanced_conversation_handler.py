"""
Enhanced Conversation Command Handler with personality, memory, and knowledge integration.

This handler extends the base conversation functionality with:
- Personality-driven NPC responses
- Conversation memory and relationship tracking
- Contextual knowledge based on NPC roles
"""

import logging
from typing import TYPE_CHECKING, Any

from rich.console import Console

from game_loop.core.command_handlers.conversation_handler import (
    ConversationCommandHandler,
)
from game_loop.core.dialogue import (
    ConversationMemoryManager,
    NPCKnowledgeEngine,
    NPCPersonalityEngine,
)
from game_loop.core.input_processor import ParsedCommand
from game_loop.state.models import ActionResult

if TYPE_CHECKING:
    from game_loop.state.manager import GameStateManager

logger = logging.getLogger(__name__)


class EnhancedConversationCommandHandler(ConversationCommandHandler):
    """Enhanced conversation handler with personality, memory, and knowledge systems."""

    def __init__(self, console: Console, state_manager: "GameStateManager"):
        """Initialize the enhanced conversation handler."""
        super().__init__(console, state_manager)

        # Initialize enhancement components
        self.personality_engine = NPCPersonalityEngine()
        self.memory_manager = ConversationMemoryManager()
        self.knowledge_engine = NPCKnowledgeEngine()

        logger.info(
            "Enhanced conversation handler initialized with personality, memory, and knowledge systems"
        )

    async def _generate_npc_response(
        self, npc, context: dict[str, Any], topic: str | None = None
    ) -> str | None:
        """Generate enhanced NPC response using personality, memory, and knowledge."""
        try:
            # Extract NPC and player information
            npc_id = getattr(npc, "id", getattr(npc, "name", "unknown"))
            npc_name = getattr(npc, "name", "Someone")
            npc_archetype = getattr(npc, "archetype", getattr(npc, "type", "generic"))

            # Get player information from context
            player_id = context.get("player", {}).get("id", "player")
            player_name = context.get("player", {}).get("name")
            location_id = context.get("location", {}).get("id", "unknown")

            # Get conversation memory context
            memory_context = self.memory_manager.get_conversation_context(
                npc_id, player_id
            )

            # Get NPC knowledge for this context
            npc_knowledge = await self.knowledge_engine.get_npc_knowledge(
                npc_archetype, location_id, topic, context
            )

            # Build enhanced context for response generation
            enhanced_context = {
                **context,
                "npc_id": npc_id,
                "npc_name": npc_name,
                "npc_archetype": npc_archetype,
                "player_id": player_id,
                "player_name": player_name,
                "location_id": location_id,
                "time_of_day": self._determine_time_of_day(),
                "mood": getattr(npc, "mood", "neutral"),
                **memory_context,
                "knowledge": npc_knowledge,
            }

            # Generate personality-driven response
            if memory_context.get("is_first_meeting", True):
                response = await self._generate_first_meeting_response(
                    npc, enhanced_context, topic
                )
            else:
                response = await self._generate_returning_visitor_response(
                    npc, enhanced_context, topic
                )

            # Record the conversation for future reference
            await self.memory_manager.record_conversation(
                npc_id, player_id, topic or "general", response, enhanced_context
            )

            return response

        except Exception as e:
            logger.error(f"Error generating enhanced NPC response: {e}")
            # Fallback to parent implementation
            return await super()._generate_npc_response(npc, context, topic)

    async def _generate_first_meeting_response(
        self, npc, context: dict[str, Any], topic: str | None
    ) -> str:
        """Generate response for first meeting with player."""
        npc_archetype = context.get("npc_archetype", "generic")
        npc_name = context.get("npc_name", "Someone")

        # Use personality engine for first meeting
        base_response = self.personality_engine.generate_personality_response(
            npc_archetype, context, topic
        )

        # Add knowledge-based context if topic is provided
        if topic and context.get("knowledge"):
            knowledge_addition = await self._add_knowledge_context(
                npc_archetype, topic, context["knowledge"]
            )
            if knowledge_addition:
                base_response += f" {knowledge_addition}"

        # Add helpful suggestions based on NPC expertise
        suggestions = self._generate_conversation_suggestions(npc_archetype, context)
        if suggestions:
            base_response += f"\n\n*{npc_name} seems knowledgeable about: {', '.join(suggestions[:3])}*"

        return base_response

    async def _generate_returning_visitor_response(
        self, npc, context: dict[str, Any], topic: str | None
    ) -> str:
        """Generate response for returning visitor with memory references."""
        npc_name = context.get("npc_name", "Someone")
        player_name = context.get("player_name", "you")
        relationship_level = context.get("relationship_level", "neutral")
        previous_topics = context.get("previous_topics", [])
        time_since_last = context.get("time_since_last")

        # Generate greeting based on relationship and time since last meeting
        greeting = self._generate_memory_based_greeting(
            npc_name, player_name, relationship_level, time_since_last
        )

        # Add topic-specific response if provided
        if topic:
            topic_response = await self._generate_topic_response_with_memory(
                context, topic, previous_topics
            )
            if topic_response:
                greeting += f" {topic_response}"
        else:
            # Reference previous conversations
            if previous_topics:
                recent_topic = previous_topics[-1]
                if recent_topic != "general":
                    greeting += f" I remember we discussed {recent_topic} before."

        return greeting

    def _generate_memory_based_greeting(
        self, npc_name: str, player_name: str, relationship_level: str, time_since_last
    ) -> str:
        """Generate greeting based on relationship and time since last meeting."""
        # Use player name if known
        address = player_name if player_name else "you"

        # Time-based greeting modifiers
        time_modifier = ""
        if time_since_last:
            hours_since = time_since_last.total_seconds() / 3600
            if hours_since < 1:
                time_modifier = "again so soon"
            elif hours_since < 6:
                time_modifier = "again today"
            elif hours_since < 24:
                time_modifier = "again"
            elif hours_since < 168:  # within a week
                time_modifier = "back"
            else:
                time_modifier = "back after so long"

        # Relationship-based greetings
        relationship_greetings = {
            "trusted_friend": f"Hello, my dear friend {address}! Great to see you {time_modifier}.",
            "close_friend": f"Good to see you {time_modifier}, {address}!",
            "friend": f"Hello again, {address}! Welcome {time_modifier}.",
            "friendly_acquaintance": f"Nice to see you {time_modifier}, {address}.",
            "acquaintance": f"Hello {address}. You're {time_modifier}.",
            "neutral": f"Oh, it's {address} {time_modifier}.",
            "unfriendly": f"{address.title()}... you're {time_modifier}.",
            "hostile": "You again. What do you want this time?",
        }

        return relationship_greetings.get(relationship_level, f"Hello {address}.")

    async def _generate_topic_response_with_memory(
        self, context: dict[str, Any], topic: str, previous_topics: list
    ) -> str | None:
        """Generate topic response considering conversation history."""
        npc_archetype = context.get("npc_archetype", "generic")

        # Check if we've discussed this topic before
        if topic in previous_topics:
            return self._generate_repeat_topic_response(topic, npc_archetype)

        # Generate new topic response using personality and knowledge
        return self.personality_engine.generate_personality_response(
            npc_archetype, context, topic
        )

    def _generate_repeat_topic_response(self, topic: str, npc_archetype: str) -> str:
        """Generate response for topics discussed before."""
        repeat_responses = {
            "security_guard": f"As I mentioned before regarding {topic}, security protocols must be followed.",
            "scholar": f"We discussed {topic} previously - would you like me to elaborate further?",
            "administrator": f"Regarding {topic}, the procedures remain the same as I explained before.",
            "merchant": f"About {topic} - my position hasn't changed since we last talked.",
            "generic": f"We talked about {topic} before. Is there something specific you want to know?",
        }

        return repeat_responses.get(npc_archetype, repeat_responses["generic"])

    async def _add_knowledge_context(
        self, npc_archetype: str, topic: str, knowledge: dict[str, Any]
    ) -> str | None:
        """Add knowledge-based context to response."""
        # Check NPC's knowledge confidence about the topic
        confidence = knowledge.get("knowledge_confidence", 0.5)

        if confidence < 0.3:
            return "Though I'm not entirely certain about that topic."
        elif confidence > 0.8:
            return "I have extensive knowledge in that area."

        # Add role-specific knowledge hints
        if npc_archetype == "security_guard" and any(
            word in topic for word in ["access", "security", "area", "clearance"]
        ):
            return "That falls under my security responsibilities."
        elif npc_archetype == "scholar" and any(
            word in topic for word in ["research", "book", "study", "history"]
        ):
            return "That's within my area of academic expertise."
        elif npc_archetype == "administrator" and any(
            word in topic for word in ["procedure", "form", "regulation", "process"]
        ):
            return "I can help you with the official procedures for that."

        return None

    def _generate_conversation_suggestions(
        self, npc_archetype: str, context: dict[str, Any]
    ) -> list:
        """Generate conversation topic suggestions based on NPC expertise."""
        return self.personality_engine.suggest_conversation_topics(
            npc_archetype, context
        )

    def _determine_time_of_day(self) -> str:
        """Determine current time of day for context."""
        # This would normally check game time or system time
        # For now, return a default
        return "day"

    async def handle(self, command: ParsedCommand) -> ActionResult:
        """Handle conversation command with enhanced features."""
        try:
            # Get base conversation result
            result = await super().handle(command)

            # If successful, add relationship information
            if result.success and hasattr(result, "feedback_message"):
                # Extract NPC information from the original conversation flow
                target_name = self._extract_conversation_target(command)
                if target_name:
                    player_state, current_location, _ = await self.get_required_state()
                    if current_location and player_state:
                        npc = await self._find_conversation_target(
                            target_name, current_location
                        )
                        if npc:
                            # Add relationship summary to the response
                            npc_id = getattr(npc, "id", getattr(npc, "name", "unknown"))
                            player_id = getattr(player_state, "player_id", "player")

                            relationship_summary = (
                                self.memory_manager.get_relationship_summary(
                                    npc_id, player_id
                                )
                            )

                            # Append relationship info to response
                            if (
                                relationship_summary
                                and "first meeting" not in relationship_summary
                            ):
                                result.feedback_message += (
                                    f"\n\n*{relationship_summary}*"
                                )

            return result

        except Exception as e:
            logger.error(f"Error in enhanced conversation handling: {e}")
            # Fallback to parent implementation
            return await super().handle(command)

    def get_npc_knowledge_summary(self, npc_archetype: str, location_id: str) -> str:
        """Get summary of what an NPC knows."""
        return self.knowledge_engine.get_knowledge_summary(npc_archetype, location_id)

    def get_conversation_suggestions(self, npc_archetype: str) -> list:
        """Get suggested conversation topics for an NPC."""
        return self.knowledge_engine.suggest_knowledge_topics(npc_archetype)

    def clear_old_conversations(self, days_old: int = 30) -> int:
        """Clear old conversation data."""
        return self.memory_manager.clear_old_conversations(days_old)
