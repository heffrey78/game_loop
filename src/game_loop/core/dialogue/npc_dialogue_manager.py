"""
NPC Dialogue Manager for dynamic dialogue generation and conversation state.
"""

import json
import logging
import re
import time
from typing import Any
from uuid import UUID

import ollama
from jinja2 import Environment, FileSystemLoader

from ...llm.config import LLMConfig
from ..models.npc_models import (
    DialogueContext,
    DialogueResponse,
    GeneratedNPC,
)
from ..world.npc_storage import NPCStorage

logger = logging.getLogger(__name__)


class NPCDialogueManager:
    """Manages dynamic dialogue generation and conversation state for NPCs."""

    def __init__(
        self,
        ollama_client: ollama.Client,
        npc_storage: NPCStorage,
        llm_config: LLMConfig | None = None,
    ):
        """Initialize dialogue management system."""
        self.ollama_client = ollama_client
        self.npc_storage = npc_storage
        self.llm_config = llm_config or LLMConfig()

        # Initialize Jinja2 environment for templates
        try:
            self.jinja_env: Environment | None = Environment(
                loader=FileSystemLoader("templates/npc_generation"),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        except Exception as e:
            logger.warning(f"Could not initialize template environment: {e}")
            self.jinja_env = None

    async def generate_dialogue_response(
        self, npc_id: UUID, player_input: str, context: dict[str, Any]
    ) -> str:
        """Generate contextual dialogue response."""
        try:
            logger.debug(f"Generating dialogue response for NPC {npc_id}")

            # Load NPC data
            npc = await self.npc_storage.retrieve_npc(npc_id)
            if not npc:
                return "I don't understand."

            # Create dialogue context
            dialogue_context = DialogueContext(
                npc=npc,
                player_input=player_input,
                conversation_history=npc.dialogue_state.conversation_history,
                current_location=context.get("location"),
                world_context=context.get("world_context", {}),
                interaction_type=context.get("interaction_type", "casual"),
            )

            # Generate response using LLM
            response = await self._generate_response_with_llm(dialogue_context)

            # Update conversation state
            await self._update_conversation_state(npc_id, dialogue_context, response)

            return response.response_text

        except Exception as e:
            logger.error(f"Error generating dialogue response: {e}")
            return "I'm not sure what to say."

    async def update_conversation_state(
        self, npc_id: UUID, interaction_data: dict[str, Any]
    ) -> None:
        """Update NPC conversation state after interaction."""
        try:
            logger.debug(f"Updating conversation state for NPC {npc_id}")

            # Load current NPC
            npc = await self.npc_storage.retrieve_npc(npc_id)
            if not npc:
                return

            # Update dialogue state
            dialogue_state = npc.dialogue_state

            # Add to conversation history
            dialogue_state.conversation_history.append(
                {
                    "timestamp": time.time(),
                    "player_input": interaction_data.get("player_input", ""),
                    "npc_response": interaction_data.get("npc_response", ""),
                    "interaction_type": interaction_data.get(
                        "interaction_type", "casual"
                    ),
                }
            )

            # Update interaction count
            dialogue_state.interaction_count += 1
            from datetime import datetime

            dialogue_state.last_interaction = datetime.now()

            # Update mood if specified
            if "mood_change" in interaction_data:
                dialogue_state.current_mood = interaction_data["mood_change"]

            # Update relationship level
            if "relationship_change" in interaction_data:
                change = interaction_data["relationship_change"]
                dialogue_state.relationship_level = max(
                    -1.0, min(1.0, dialogue_state.relationship_level + change)
                )

            # Update active topics
            if "new_topics" in interaction_data:
                for topic in interaction_data["new_topics"]:
                    if topic not in dialogue_state.active_topics:
                        dialogue_state.active_topics.append(topic)

            # Store updated state
            await self.npc_storage.update_npc_state(npc_id, dialogue_state)

        except Exception as e:
            logger.error(f"Error updating conversation state: {e}")

    async def get_available_topics(
        self, npc_id: UUID, player_context: dict[str, Any]
    ) -> list[str]:
        """Get topics the NPC can discuss based on current state."""
        try:
            # Load NPC
            npc = await self.npc_storage.retrieve_npc(npc_id)
            if not npc:
                return []

            available_topics = []

            # Always available basic topics
            available_topics.extend(["greeting", "location", "weather"])

            # Add archetype-specific topics
            archetype_topics = {
                "merchant": ["trade", "goods", "prices", "business"],
                "guard": ["security", "law", "patrol", "safety"],
                "scholar": ["knowledge", "books", "research", "history"],
                "hermit": ["solitude", "wisdom", "nature", "philosophy"],
                "innkeeper": ["travelers", "food", "lodging", "local_news"],
                "artisan": ["crafts", "skills", "tools", "creation"],
                "wanderer": ["travel", "roads", "adventure", "stories"],
            }

            archetype = npc.personality.archetype
            if archetype in archetype_topics:
                available_topics.extend(archetype_topics[archetype])

            # Add knowledge-based topics
            available_topics.extend(npc.knowledge.expertise_areas)

            # Add relationship-level topics
            relationship_level = npc.dialogue_state.relationship_level
            if relationship_level > 0.3:
                available_topics.extend(["personal_life", "opinions"])
            if relationship_level > 0.6:
                available_topics.extend(["secrets", "private_thoughts"])

            # Add quest topics if available
            if npc.dialogue_state.available_quests:
                available_topics.append("quests")

            # Remove duplicates and limit
            return list(set(available_topics))[:10]

        except Exception as e:
            logger.error(f"Error getting available topics: {e}")
            return ["greeting"]

    async def process_knowledge_sharing(
        self, npc_id: UUID, topic: str
    ) -> dict[str, Any]:
        """Process knowledge sharing for specific topics."""
        try:
            # Load NPC
            npc = await self.npc_storage.retrieve_npc(npc_id)
            if not npc:
                return {}

            knowledge_shared = {}

            # Check if NPC knows about the topic
            if topic in npc.knowledge.expertise_areas:
                # NPC has expertise in this area
                knowledge_shared = {
                    "topic": topic,
                    "expertise_level": "expert",
                    "information": f"I know quite a bit about {topic}.",
                    "can_teach": True,
                }
            elif topic in npc.knowledge.world_knowledge:
                # NPC has general knowledge
                knowledge_shared = {
                    "topic": topic,
                    "expertise_level": "general",
                    "information": npc.knowledge.world_knowledge[topic],
                    "can_teach": False,
                }
            elif topic in npc.knowledge.local_knowledge:
                # NPC has local knowledge
                knowledge_shared = {
                    "topic": topic,
                    "expertise_level": "local",
                    "information": npc.knowledge.local_knowledge[topic],
                    "can_teach": False,
                }
            else:
                # NPC doesn't know about the topic
                knowledge_shared = {
                    "topic": topic,
                    "expertise_level": "none",
                    "information": f"I don't know much about {topic}.",
                    "can_teach": False,
                }

            # Check relationship level for sharing secrets
            if (
                topic in npc.knowledge.secrets
                and npc.dialogue_state.relationship_level > 0.5
            ):
                knowledge_shared["secret_info"] = npc.knowledge.secrets[
                    npc.knowledge.secrets.index(topic)
                ]

            return knowledge_shared

        except Exception as e:
            logger.error(f"Error processing knowledge sharing: {e}")
            return {}

    async def update_relationship(
        self, npc_id: UUID, interaction_outcome: str
    ) -> float:
        """Update relationship level based on interaction."""
        try:
            # Load NPC
            npc = await self.npc_storage.retrieve_npc(npc_id)
            if not npc:
                return 0.0

            current_level = npc.dialogue_state.relationship_level

            # Determine relationship change based on outcome
            relationship_changes = {
                "positive": 0.1,
                "very_positive": 0.2,
                "negative": -0.1,
                "very_negative": -0.2,
                "neutral": 0.0,
                "helpful": 0.15,
                "rude": -0.15,
                "generous": 0.25,
                "threatening": -0.3,
            }

            change = relationship_changes.get(interaction_outcome, 0.0)

            # Apply personality modifiers
            personality_modifiers = {
                "friendly": 1.2,
                "suspicious": 0.8,
                "trusting": 1.5,
                "cautious": 0.7,
                "social": 1.3,
                "reclusive": 0.6,
            }

            for trait in npc.personality.traits:
                if trait in personality_modifiers:
                    change *= personality_modifiers[trait]

            # Calculate new level (clamped between -1.0 and 1.0)
            new_level = max(-1.0, min(1.0, current_level + change))

            # Update dialogue state
            npc.dialogue_state.relationship_level = new_level

            # Store updated state
            await self.npc_storage.update_npc_state(npc_id, npc.dialogue_state)

            logger.debug(
                f"Updated relationship for {npc_id}: {current_level:.2f} -> {new_level:.2f}"
            )

            return new_level

        except Exception as e:
            logger.error(f"Error updating relationship: {e}")
            return 0.0

    async def _generate_response_with_llm(
        self, context: DialogueContext
    ) -> DialogueResponse:
        """Generate dialogue response using LLM."""
        try:
            # Create prompt
            prompt = self._create_dialogue_prompt(context)

            # Call Ollama
            response_text = await self._call_ollama(prompt)

            # Parse response
            response_data = self._parse_dialogue_response(response_text)

            # Create structured response
            return DialogueResponse(
                response_text=response_data.get("dialogue", response_text),
                mood_change=response_data.get("mood"),
                relationship_change=response_data.get("relationship_change", 0.0),
                new_topics=response_data.get("new_topics", []),
                quest_offered=response_data.get("quest"),
                knowledge_shared=response_data.get("knowledge", {}),
                response_metadata={"generated_with_llm": True},
            )

        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return self._create_fallback_response(context)

    def _create_dialogue_prompt(self, context: DialogueContext) -> str:
        """Create the prompt for dialogue generation."""
        # Use template if available
        if self.jinja_env:
            try:
                template = self.jinja_env.get_template("dialogue_prompts.j2")
                return template.render(context=context)
            except Exception as e:
                logger.warning(f"Error using dialogue template: {e}")

        # Fallback to hardcoded prompt
        return self._create_fallback_prompt(context)

    def _create_fallback_prompt(self, context: DialogueContext) -> str:
        """Create a fallback dialogue prompt."""
        npc = context.npc
        recent_history = (
            context.conversation_history[-3:] if context.conversation_history else []
        )

        history_text = ""
        if recent_history:
            history_text = "Recent conversation:\n" + "\n".join(
                [
                    f"Player: {h.get('player_input', '')}\nNPC: {h.get('npc_response', '')}"
                    for h in recent_history
                ]
            )

        return f"""You are roleplaying as {npc.base_npc.name}, a {npc.personality.archetype}.

Character Details:
- Description: {npc.base_npc.description}
- Personality: {', '.join(npc.personality.traits)}
- Motivations: {', '.join(npc.personality.motivations)}
- Speech style: {npc.personality.speech_patterns.get('style', 'neutral')}
- Current mood: {npc.dialogue_state.current_mood}
- Relationship with player: {npc.dialogue_state.relationship_level:.1f} (-1 to 1)

Knowledge areas: {', '.join(npc.knowledge.expertise_areas)}

{history_text}

Player says: "{context.player_input}"

Respond as {npc.base_npc.name} would, staying in character. Keep the response to 1-3 sentences.
Consider your personality, current mood, and relationship level with the player.

Response:"""

    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API for dialogue generation."""
        try:
            model = self.llm_config.default_model if self.llm_config else "llama3.1:8b"

            response = self.ollama_client.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop": ["Player:", "---", "\n\n"],
                },
            )

            return str(response.get("response", "")).strip()

        except Exception as e:
            logger.error(f"Error calling Ollama for dialogue: {e}")
            raise

    def _parse_dialogue_response(self, response: str) -> dict[str, Any]:
        """Parse LLM dialogue response."""
        try:
            # Try to parse as JSON if it looks like JSON
            if response.strip().startswith("{"):
                result = json.loads(response)
                return (
                    dict(result) if isinstance(result, dict) else {"dialogue": response}
                )

            # Otherwise, treat as plain text
            return {"dialogue": response}

        except json.JSONDecodeError:
            return {"dialogue": response}

    def _create_fallback_response(self, context: DialogueContext) -> DialogueResponse:
        """Create a fallback response when LLM fails."""
        npc = context.npc

        # Create simple response based on archetype
        fallback_responses = {
            "merchant": "I'm always willing to make a deal!",
            "guard": "Stay safe out there, traveler.",
            "scholar": "Knowledge is the greatest treasure.",
            "hermit": "The world is full of mysteries.",
            "innkeeper": "Welcome! How can I help you?",
            "artisan": "I take pride in my work.",
            "wanderer": "The road calls to me.",
        }

        response_text = fallback_responses.get(
            npc.personality.archetype, "I understand."
        )

        return DialogueResponse(
            response_text=response_text,
            response_metadata={"fallback_response": True},
        )

    async def _update_conversation_state(
        self, npc_id: UUID, context: DialogueContext, response: DialogueResponse
    ) -> None:
        """Update conversation state after generating response."""
        try:
            interaction_data = {
                "player_input": context.player_input,
                "npc_response": response.response_text,
                "interaction_type": context.interaction_type,
                "mood_change": response.mood_change,
                "relationship_change": response.relationship_change,
                "new_topics": response.new_topics,
            }

            await self.update_conversation_state(npc_id, interaction_data)

        except Exception as e:
            logger.error(f"Error updating conversation state: {e}")
