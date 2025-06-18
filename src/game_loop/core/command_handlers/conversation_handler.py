"""
Conversation command handler for the Game Loop.
Handles TALK commands for NPC dialogue and interaction.
"""

import logging
from typing import TYPE_CHECKING

from rich.console import Console

from game_loop.core.command_handlers.base_handler import CommandHandler
from game_loop.core.input_processor import ParsedCommand
from game_loop.state.models import ActionResult

if TYPE_CHECKING:
    from game_loop.state.manager import GameStateManager
    from game_loop.state.models import Location, PlayerState

logger = logging.getLogger(__name__)


class ConversationCommandHandler(CommandHandler):
    """
    Handler for conversation commands (TALK) in the Game Loop.

    Handles NPC dialogue with dynamic response generation,
    relationship tracking, and knowledge exchange.
    """

    def __init__(self, console: Console, state_manager: "GameStateManager"):
        """
        Initialize the conversation handler.

        Args:
            console: Rich console for output
            state_manager: Game state manager for accessing and updating game state
        """
        super().__init__(console, state_manager)

        # Conversation command mappings
        self.talk_commands = ["talk", "speak", "chat", "converse", "discuss"]
        self.greeting_commands = ["greet", "hello", "hi"]

        # Conversation context tracking
        self.conversation_history: dict[str, dict[str, list[dict]]] = {}

    async def handle(self, command: ParsedCommand) -> ActionResult:
        """
        Handle a conversation command and return the result.

        Args:
            command: The parsed conversation command to handle

        Returns:
            ActionResult describing the conversation outcome
        """
        # Get required game state
        player_state, current_location, world_state = await self.get_required_state()

        # Basic validation
        if not current_location:
            return ActionResult(
                success=False,
                feedback_message="Error: Cannot determine current location.",
            )

        if not player_state:
            return ActionResult(
                success=False, feedback_message="Error: Cannot access player state."
            )

        # Determine conversation target
        target_name = self._extract_conversation_target(command)
        if not target_name:
            return ActionResult(
                success=False, feedback_message="Who would you like to talk to?"
            )

        try:
            # Find the NPC to talk to
            npc = await self._find_conversation_target(target_name, current_location)

            if not npc:
                return ActionResult(
                    success=False,
                    feedback_message=f"You don't see anyone named '{target_name}' here to talk to.",
                )

            # Handle the conversation
            conversation_result = await self._handle_conversation(
                npc, command, current_location, player_state
            )

            return conversation_result

        except Exception as e:
            logger.error(f"Error handling conversation: {e}")
            return ActionResult(
                success=False,
                feedback_message="You have trouble starting a conversation.",
            )

    def _extract_conversation_target(self, command: ParsedCommand) -> str | None:
        """
        Extract the conversation target from the parsed command.

        Args:
            command: The parsed command

        Returns:
            The target name, or None if not found
        """
        # Check for "talk to [target]" pattern
        if command.action.lower() in self.talk_commands:
            if command.subject:
                # Remove common prepositions
                target = command.subject.lower()
                target = target.replace("to ", "").replace("with ", "").strip()
                return target
            elif command.target:
                return command.target.lower()

        # Check for direct greeting commands
        if command.action.lower() in self.greeting_commands:
            if command.subject:
                return command.subject.lower()

        return None

    async def _find_conversation_target(self, target_name: str, location: "Location"):
        """
        Find an NPC to talk to in the current location.

        Args:
            target_name: Name of NPC to find
            location: Current location

        Returns:
            The NPC object if found, None otherwise
        """
        try:
            if not hasattr(location, "npcs") or not location.npcs:
                return None

            normalized_target = self.normalize_name(target_name)

            # Search through NPCs
            for npc in location.npcs:
                npc_name = self.normalize_name(getattr(npc, "name", ""))

                # Check exact match first
                if npc_name == normalized_target:
                    return npc

                # Check partial match
                if npc_name and normalized_target in npc_name:
                    return npc

                # Check if target is in any NPC aliases or titles
                if hasattr(npc, "aliases") and npc.aliases:
                    for alias in npc.aliases:
                        if self.normalize_name(alias) == normalized_target:
                            return npc

            return None

        except Exception as e:
            logger.error(f"Error finding conversation target: {e}")
            return None

    async def _handle_conversation(
        self,
        npc,
        command: ParsedCommand,
        location: "Location",
        player_state: "PlayerState",
    ) -> ActionResult:
        """
        Handle the actual conversation with an NPC.

        Args:
            npc: The NPC to talk to
            command: The original command
            location: Current location
            player_state: Current player state

        Returns:
            ActionResult with conversation response
        """
        try:
            npc_name = getattr(npc, "name", "Someone")

            # Get conversation context
            conversation_context = await self._build_conversation_context(
                npc, location, player_state
            )

            # Check for specific dialogue topics in the command
            topic = self._extract_conversation_topic(command)

            # Generate NPC response
            response = await self._generate_npc_response(
                npc, conversation_context, topic
            )

            if not response:
                # Fallback to basic response
                response = await self._generate_fallback_response(npc)

            # Update conversation history
            self._update_conversation_history(npc, player_state, topic or "general")

            # Update NPC relationship if applicable
            await self._update_npc_relationship(npc, player_state, "talked")

            return ActionResult(success=True, feedback_message=response)

        except Exception as e:
            logger.error(f"Error handling conversation: {e}")
            npc_name = getattr(npc, "name", "Someone")
            return ActionResult(
                success=False,
                feedback_message=f"{npc_name} seems unable to talk right now.",
            )

    def _extract_conversation_topic(self, command: ParsedCommand) -> str | None:
        """
        Extract any specific topic from the conversation command.

        Args:
            command: The parsed command

        Returns:
            Topic string if found, None otherwise
        """
        # Look for "about" keyword
        full_text = (
            f"{command.action} {command.subject or ''} {command.target or ''}".strip()
        )

        if " about " in full_text.lower():
            topic_part = full_text.lower().split(" about ")[-1].strip()
            return topic_part if topic_part else None

        # Check for common topic keywords
        topic_keywords = [
            "quest",
            "help",
            "information",
            "direction",
            "location",
            "item",
        ]
        words = full_text.lower().split()

        for keyword in topic_keywords:
            if keyword in words:
                return keyword

        return None

    async def _build_conversation_context(
        self, npc, location: "Location", player_state: "PlayerState"
    ) -> dict:
        """
        Build context for conversation generation.

        Args:
            npc: The NPC being talked to
            location: Current location
            player_state: Current player state

        Returns:
            Context dictionary for conversation generation
        """
        try:
            context = {
                "npc": {
                    "name": getattr(npc, "name", "Unknown"),
                    "personality": getattr(npc, "personality", {}),
                    "knowledge": getattr(npc, "knowledge", {}),
                    "mood": getattr(npc, "mood", "neutral"),
                },
                "location": {
                    "name": getattr(location, "name", "Unknown Location"),
                    "type": getattr(location, "type", "generic"),
                    "description": getattr(location, "description", ""),
                },
                "player": {
                    "level": getattr(player_state, "level", 1),
                    "inventory_count": len(getattr(player_state, "inventory", {})),
                    "health": getattr(player_state, "health", 100),
                },
                "relationship": self._get_npc_relationship(npc, player_state),
                "conversation_history": self._get_conversation_history(
                    npc, player_state
                ),
            }

            return context

        except Exception as e:
            logger.error(f"Error building conversation context: {e}")
            return {}

    async def _generate_npc_response(
        self, npc, context: dict, topic: str | None = None
    ) -> str | None:
        """
        Generate an NPC response using available systems.

        Args:
            npc: The NPC
            context: Conversation context
            topic: Specific topic if any

        Returns:
            Generated response string, or None if generation fails
        """
        try:
            # Check if we have access to LLM generation through the state manager
            if hasattr(self.state_manager, "llm_client"):
                return await self._generate_llm_response(npc, context, topic)

            # Fallback to template-based response
            return await self._generate_template_response(npc, context, topic)

        except Exception as e:
            logger.error(f"Error generating NPC response: {e}")
            return None

    async def _generate_llm_response(
        self, npc, context: dict, topic: str | None = None
    ) -> str | None:
        """
        Generate response using LLM integration.

        Args:
            npc: The NPC
            context: Conversation context
            topic: Specific topic if any

        Returns:
            Generated response string, or None if generation fails
        """
        try:
            # This would integrate with the existing LLM dialogue generation
            # from the main game loop
            npc_name = getattr(npc, "name", "Someone")

            # Create a simple prompt for dialogue
            prompt = self._create_dialogue_prompt(npc, context, topic)

            # For now, return a placeholder that shows the integration point
            # In full implementation, this would call the actual LLM
            response = f"{npc_name} responds thoughtfully to your conversation."

            if topic:
                response += f" They seem knowledgeable about {topic}."

            return response

        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return None

    async def _generate_template_response(
        self, npc, context: dict, topic: str | None = None
    ) -> str | None:
        """
        Generate response using template system.

        Args:
            npc: The NPC
            context: Conversation context
            topic: Specific topic if any

        Returns:
            Generated response string, or None if generation fails
        """
        try:
            npc_name = getattr(npc, "name", "Someone")
            personality = getattr(npc, "personality", {})
            mood = getattr(npc, "mood", "neutral")

            # Generate personality-based response
            if isinstance(personality, dict):
                personality_type = personality.get("type", "neutral")
            else:
                personality_type = "neutral"

            # Topic-specific responses
            if topic:
                topic_responses = {
                    "quest": f"{npc_name} mentions they might have something for you to do.",
                    "help": f"{npc_name} offers what assistance they can.",
                    "direction": f"{npc_name} tries to help you with directions.",
                    "location": f"{npc_name} shares what they know about this area.",
                    "item": f"{npc_name} discusses items and equipment.",
                }

                if topic in topic_responses:
                    return topic_responses[topic]

            # Personality-based responses
            personality_responses = {
                "friendly": f"{npc_name} greets you warmly and seems happy to chat.",
                "serious": f"{npc_name} nods respectfully and speaks in measured tones.",
                "mysterious": f"{npc_name} regards you with enigmatic eyes before speaking.",
                "merchant": f"{npc_name} smiles and asks if you're interested in trade.",
                "guard": f"{npc_name} stands at attention and speaks formally.",
                "scholar": f"{npc_name} adjusts their glasses and speaks knowledgeably.",
            }

            response = personality_responses.get(
                personality_type,
                f"{npc_name} acknowledges your presence and prepares to speak.",
            )

            # Add mood modifier
            if mood == "happy":
                response += " They seem to be in good spirits."
            elif mood == "sad":
                response += " They seem somewhat melancholy."
            elif mood == "angry":
                response += " They appear somewhat irritated."
            elif mood == "worried":
                response += " They seem concerned about something."

            return response

        except Exception as e:
            logger.error(f"Error generating template response: {e}")
            return None

    async def _generate_fallback_response(self, npc) -> str:
        """
        Generate a basic fallback response.

        Args:
            npc: The NPC

        Returns:
            Basic response string
        """
        npc_name = getattr(npc, "name", "Someone")
        return f"{npc_name} looks at you and nods in acknowledgment."

    def _create_dialogue_prompt(
        self, npc, context: dict, topic: str | None = None
    ) -> str:
        """
        Create a prompt for LLM dialogue generation.

        Args:
            npc: The NPC
            context: Conversation context
            topic: Specific topic if any

        Returns:
            Dialogue generation prompt
        """
        npc_name = getattr(npc, "name", "Someone")
        location_name = context.get("location", {}).get("name", "unknown location")

        prompt = f"You are {npc_name} in {location_name}. "

        if topic:
            prompt += f"The player wants to talk about {topic}. "

        prompt += "Respond in character with 1-2 sentences."

        return prompt

    def _update_conversation_history(
        self, npc, player_state: "PlayerState", topic: str
    ):
        """
        Update conversation history for relationship tracking.

        Args:
            npc: The NPC
            player_state: Current player state
            topic: Conversation topic
        """
        try:
            npc_id = getattr(npc, "id", getattr(npc, "name", "unknown"))
            player_id = getattr(player_state, "player_id", "player")

            if player_id not in self.conversation_history:
                self.conversation_history[player_id] = {}

            if npc_id not in self.conversation_history[player_id]:
                self.conversation_history[player_id][npc_id] = []

            self.conversation_history[player_id][npc_id].append(
                {
                    "topic": topic,
                    "timestamp": "now",  # In full implementation, use proper timestamp
                }
            )

            # Keep only recent conversations
            if len(self.conversation_history[player_id][npc_id]) > 10:
                self.conversation_history[player_id][npc_id] = (
                    self.conversation_history[player_id][npc_id][-10:]
                )

        except Exception as e:
            logger.error(f"Error updating conversation history: {e}")

    def _get_conversation_history(self, npc, player_state: "PlayerState") -> list:
        """
        Get conversation history with an NPC.

        Args:
            npc: The NPC
            player_state: Current player state

        Returns:
            List of recent conversations
        """
        try:
            npc_id = getattr(npc, "id", getattr(npc, "name", "unknown"))
            player_id = getattr(player_state, "player_id", "player")

            return self.conversation_history.get(player_id, {}).get(npc_id, [])

        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []

    def _get_npc_relationship(self, npc, player_state: "PlayerState") -> str:
        """
        Get the relationship level with an NPC.

        Args:
            npc: The NPC
            player_state: Current player state

        Returns:
            Relationship level string
        """
        try:
            # This would integrate with a relationship tracking system
            # For now, return a default
            conversation_count = len(self._get_conversation_history(npc, player_state))

            if conversation_count == 0:
                return "stranger"
            elif conversation_count < 3:
                return "acquaintance"
            elif conversation_count < 10:
                return "friend"
            else:
                return "close_friend"

        except Exception as e:
            logger.error(f"Error getting NPC relationship: {e}")
            return "stranger"

    async def _update_npc_relationship(
        self, npc, player_state: "PlayerState", action: str
    ):
        """
        Update relationship with NPC based on interaction.

        Args:
            npc: The NPC
            player_state: Current player state
            action: Type of interaction
        """
        try:
            # This would update a persistent relationship system
            # For now, just log the interaction
            npc_name = getattr(npc, "name", "Someone")
            logger.info(f"Relationship update: {action} with {npc_name}")

        except Exception as e:
            logger.error(f"Error updating NPC relationship: {e}")
