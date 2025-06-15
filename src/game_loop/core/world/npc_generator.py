"""
NPC Generator Engine using LLM integration and contextual intelligence.
"""

import json
import logging
import re
import time
from typing import Any
from uuid import uuid4

import ollama
from jinja2 import Environment, FileSystemLoader

from ...llm.config import LLMConfig
from ...state.models import Location, NonPlayerCharacter, WorldState
from ..models.npc_models import (
    GeneratedNPC,
    NPCDialogueState,
    NPCGenerationContext,
    NPCGenerationMetrics,
    NPCKnowledge,
    NPCPersonality,
)
from .npc_context_collector import NPCContextCollector
from .npc_storage import NPCStorage
from .npc_theme_manager import NPCThemeManager

logger = logging.getLogger(__name__)


class NPCGenerator:
    """Main NPC generation engine using LLM integration and contextual intelligence."""

    def __init__(
        self,
        ollama_client: ollama.Client,
        world_state: WorldState,
        theme_manager: NPCThemeManager,
        context_collector: NPCContextCollector,
        npc_storage: NPCStorage,
        llm_config: LLMConfig | None = None,
    ):
        """Initialize NPC generation system."""
        self.ollama_client = ollama_client
        self.world_state = world_state
        self.theme_manager = theme_manager
        self.context_collector = context_collector
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

        self._generation_metrics: list[NPCGenerationMetrics] = []

    async def generate_npc(self, context: NPCGenerationContext) -> GeneratedNPC:
        """Generate a complete NPC based on context."""
        start_time = time.time()
        metrics = NPCGenerationMetrics()

        try:
            logger.info(
                f"Generating NPC for location {context.location.name} "
                f"with purpose {context.generation_purpose}"
            )

            # Determine appropriate archetype
            archetype_start = time.time()
            archetype = await self.theme_manager.determine_npc_archetype(context)
            metrics.context_collection_time_ms = int(
                (time.time() - archetype_start) * 1000
            )

            # Generate NPC characteristics using LLM
            llm_start = time.time()
            npc_data = await self._generate_with_llm(context, archetype)
            metrics.llm_response_time_ms = int((time.time() - llm_start) * 1000)

            # Create personality
            personality = await self._create_personality(
                npc_data, archetype, context.location
            )

            # Generate knowledge
            knowledge = await self._generate_knowledge(personality, context)

            # Create dialogue state
            dialogue_state = await self._create_dialogue_state(personality)

            # Create base NPC entity
            base_npc = NonPlayerCharacter(
                npc_id=uuid4(),
                name=npc_data.get("name", f"{archetype.title()} NPC"),
                description=npc_data.get("description", f"A {archetype} in the area"),
            )

            # Create complete generated NPC
            generated_npc = GeneratedNPC(
                base_npc=base_npc,
                personality=personality,
                knowledge=knowledge,
                dialogue_state=dialogue_state,
                generation_metadata={
                    "generation_timestamp": time.time(),
                    "generation_purpose": context.generation_purpose,
                    "location_theme": context.location_theme.name,
                    "archetype": archetype,
                    "llm_model": self.llm_config.default_model,
                },
            )

            # Validate generated NPC
            validation_start = time.time()
            is_valid = await self._validate_generated_npc(generated_npc, context)
            metrics.validation_time_ms = int((time.time() - validation_start) * 1000)

            if not is_valid:
                logger.warning(f"Generated NPC {base_npc.name} failed validation")
                # Add validation warning to metadata
                generated_npc.generation_metadata["validation_warning"] = True

            # Update metrics
            metrics.generation_time_ms = int((time.time() - start_time) * 1000)
            metrics.total_time_ms = (
                metrics.generation_time_ms
                + metrics.context_collection_time_ms
                + metrics.llm_response_time_ms
                + metrics.validation_time_ms
                + metrics.storage_time_ms
            )
            self._generation_metrics.append(metrics)

            logger.info(f"Successfully generated NPC: {base_npc.name} ({archetype})")
            return generated_npc

        except Exception as e:
            logger.error(f"Error generating NPC: {e}")
            metrics.generation_time_ms = int((time.time() - start_time) * 1000)
            metrics.total_time_ms = (
                metrics.generation_time_ms
                + metrics.context_collection_time_ms
                + metrics.llm_response_time_ms
                + metrics.validation_time_ms
                + metrics.storage_time_ms
            )
            self._generation_metrics.append(metrics)
            raise

    async def _generate_with_llm(
        self, context: NPCGenerationContext, archetype: str
    ) -> dict[str, Any]:
        """Use LLM to generate NPC characteristics."""
        try:
            # Create prompt
            prompt = self._create_generation_prompt(context, archetype)

            # Call Ollama
            response = await self._call_ollama(prompt)

            # Parse JSON response
            npc_data = self._parse_llm_response(response)

            return npc_data

        except Exception as e:
            logger.error(f"Error in LLM generation: {e}")
            # Return fallback NPC
            return self._create_fallback_npc(context, archetype)

    def _create_generation_prompt(
        self, context: NPCGenerationContext, archetype: str
    ) -> str:
        """Create the prompt for NPC generation."""
        # Use template if available
        if self.jinja_env:
            try:
                template = self.jinja_env.get_template("npc_prompts.j2")
                return template.render(
                    location=context.location,
                    location_theme=context.location_theme,
                    archetype=archetype,
                    nearby_npcs=context.nearby_npcs,
                    world_snapshot=context.world_state_snapshot,
                    generation_purpose=context.generation_purpose,
                    constraints=context.constraints,
                )
            except Exception as e:
                logger.warning(f"Error using template: {e}")

        # Fallback to hardcoded prompt
        return self._create_fallback_prompt(context, archetype)

    def _create_fallback_prompt(
        self, context: NPCGenerationContext, archetype: str
    ) -> str:
        """Create a fallback prompt when templates aren't available."""
        nearby_names = [npc.name for npc in context.nearby_npcs[:3]]
        nearby_desc = ", ".join(nearby_names) if nearby_names else "none"

        return f"""You are generating an NPC for a text adventure game.

Context:
- Location: {context.location.name} - {context.location.description}
- Location Theme: {context.location_theme.name}
- Archetype: {archetype}
- Nearby NPCs: {nearby_desc}
- Generation Purpose: {context.generation_purpose}
- Player Level: {context.player_level}

Generate an NPC that:
1. Fits the {archetype} archetype and {context.location_theme.name} theme
2. Has a distinct personality and background
3. Knows appropriate information about the area
4. Has realistic motivations and goals
5. Can engage in meaningful dialogue

Format your response as JSON with these fields:
- name: NPC name (2-3 words, appropriate to theme)
- description: Physical description (1-2 sentences)
- personality_traits: List of 3-4 personality traits
- motivations: List of 2-3 main motivations
- fears: List of 1-2 fears or concerns
- background: Brief background story (2-3 sentences)
- knowledge_areas: List of 3-4 things they know about
- speech_style: Description of how they speak
- initial_dialogue: Opening dialogue when first met
- special_abilities: Any special skills or knowledge (optional)

Respond only with the JSON, no additional text."""

    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API with the prompt."""
        try:
            model = self.llm_config.default_model if self.llm_config else "llama3.1:8b"

            response = self.ollama_client.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": 0.8,  # Higher creativity for NPCs
                    "top_p": 0.9,
                    "stop": ["</response>", "---"],
                },
            )

            return str(response.get("response", ""))

        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            raise

    def _parse_llm_response(self, response: str) -> dict[str, Any]:
        """Parse the LLM JSON response."""
        try:
            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                return dict(result) if isinstance(result, dict) else {}
            else:
                logger.warning("No JSON found in LLM response")
                return self._parse_fallback_response(response)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return self._parse_fallback_response(response)

    def _parse_fallback_response(self, response: str) -> dict[str, Any]:
        """Parse response when JSON parsing fails."""
        lines = response.split("\n")

        name = "Unknown NPC"
        description = "A mysterious figure."

        # Try to extract name from first line
        first_line = lines[0].strip()
        if len(first_line) < 50 and not first_line.startswith("{"):
            name = first_line

        return {
            "name": name,
            "description": description,
            "personality_traits": ["mysterious", "quiet"],
            "motivations": ["survival", "solitude"],
            "fears": ["strangers"],
            "background": "Little is known about this person.",
            "knowledge_areas": ["local_area"],
            "speech_style": "speaks briefly and carefully",
            "initial_dialogue": "Hello, traveler.",
        }

    def _create_fallback_npc(
        self, context: NPCGenerationContext, archetype: str
    ) -> dict[str, Any]:
        """Create a fallback NPC when LLM generation fails."""
        archetype_templates = {
            "merchant": {
                "name": "Local Trader",
                "description": "A seasoned merchant with goods to sell.",
                "traits": ["business-minded", "friendly", "practical"],
                "motivations": ["profit", "reputation"],
                "background": "Has been trading in this area for years.",
                "knowledge": ["trade_routes", "prices", "local_economy"],
                "dialogue": "Welcome! Care to see my wares?",
            },
            "guard": {
                "name": "Area Guardian",
                "description": "A vigilant protector of the area.",
                "traits": ["dutiful", "alert", "protective"],
                "motivations": ["duty", "safety"],
                "background": "Sworn to protect this location.",
                "knowledge": ["local_threats", "security", "law"],
                "dialogue": "State your business, traveler.",
            },
            "hermit": {
                "name": "Wise Hermit",
                "description": "A solitary figure who lives apart from others.",
                "traits": ["wise", "reclusive", "observant"],
                "motivations": ["wisdom", "solitude"],
                "background": "Has lived alone here for many years.",
                "knowledge": ["nature_lore", "ancient_knowledge", "philosophy"],
                "dialogue": "Few visitors come this way. What brings you here?",
            },
        }

        template = archetype_templates.get(archetype, archetype_templates["hermit"])

        return {
            "name": template["name"],
            "description": template["description"],
            "personality_traits": template["traits"],
            "motivations": template["motivations"],
            "fears": ["change"],
            "background": template["background"],
            "knowledge_areas": template["knowledge"],
            "speech_style": "speaks with experience",
            "initial_dialogue": template["dialogue"],
        }

    async def _create_personality(
        self, llm_data: dict, archetype: str, location: "Location"
    ) -> NPCPersonality:
        """Create structured personality from LLM output."""
        try:
            # Get base personality template
            base_personality = await self.theme_manager.get_personality_template(
                archetype, location.state_flags.get("theme", "generic")
            )

            # Override with LLM-generated data
            personality = NPCPersonality(
                name=llm_data.get("name", base_personality.name),
                archetype=archetype,
                traits=llm_data.get("personality_traits", base_personality.traits),
                motivations=llm_data.get("motivations", base_personality.motivations),
                fears=llm_data.get("fears", base_personality.fears),
                speech_patterns={
                    "style": llm_data.get("speech_style", "neutral"),
                    "formality": "casual",
                    "verbosity": "moderate",
                },
                relationship_tendencies=base_personality.relationship_tendencies,
            )

            # Apply cultural variations
            personality = await self.theme_manager.generate_cultural_variations(
                personality, location
            )

            return personality

        except Exception as e:
            logger.error(f"Error creating personality: {e}")
            return NPCPersonality(
                name=llm_data.get("name", "Unknown"),
                archetype=archetype,
                traits=["neutral"],
                motivations=["survival"],
            )

    async def _generate_knowledge(
        self, personality: NPCPersonality, context: NPCGenerationContext
    ) -> NPCKnowledge:
        """Generate appropriate knowledge for the NPC."""
        try:
            # Collect world knowledge for this location
            world_knowledge_data = await self.context_collector.collect_world_knowledge(
                context.location
            )

            # Create knowledge based on archetype and personality
            knowledge = NPCKnowledge(
                world_knowledge={
                    "general_area": context.location.name,
                    "local_theme": context.location_theme.name,
                    "connected_areas": list(context.location.connections.keys()),
                },
                local_knowledge={
                    "location_details": context.location.description,
                    "local_npcs": [npc.name for npc in context.nearby_npcs],
                    "notable_features": context.location.state_flags,
                },
                personal_history=[
                    f"Has lived in the {context.location_theme.name.lower()} area",
                    f"Works as a {personality.archetype}",
                ],
                relationships={
                    npc.name: {"relationship": "acquaintance", "trust": 0.5}
                    for npc in context.nearby_npcs
                },
                secrets=[],  # Will be populated based on archetype
                expertise_areas=self._determine_expertise_areas(personality.archetype),
            )

            # Add archetype-specific knowledge
            if personality.archetype == "merchant":
                knowledge.expertise_areas.extend(["trade", "economics", "negotiation"])
                knowledge.secrets.append("knows about hidden trade routes")
            elif personality.archetype == "scholar":
                knowledge.expertise_areas.extend(["history", "lore", "research"])
                knowledge.secrets.append("has access to rare knowledge")
            elif personality.archetype == "hermit":
                knowledge.expertise_areas.extend(["nature", "solitude", "wisdom"])
                knowledge.secrets.append("knows hidden places")

            return knowledge

        except Exception as e:
            logger.error(f"Error generating knowledge: {e}")
            return NPCKnowledge()

    async def _create_dialogue_state(
        self, personality: NPCPersonality
    ) -> NPCDialogueState:
        """Initialize dialogue state for the NPC."""
        return NPCDialogueState(
            current_mood="neutral",
            relationship_level=0.0,
            conversation_history=[],
            active_topics=["introduction", "local_area"],
            available_quests=[],
            interaction_count=0,
            last_interaction=None,
        )

    async def _validate_generated_npc(
        self, npc: GeneratedNPC, context: NPCGenerationContext
    ) -> bool:
        """Validate the generated NPC meets quality standards."""
        try:
            # Basic validation checks
            if not npc.base_npc.name or len(npc.base_npc.name.strip()) == 0:
                return False

            if not npc.base_npc.description or len(npc.base_npc.description) < 10:
                return False

            if not npc.personality.traits or len(npc.personality.traits) == 0:
                return False

            # Validate consistency with theme manager
            is_consistent = await self.theme_manager.validate_npc_consistency(
                npc, context.location
            )

            if not is_consistent:
                logger.warning(f"NPC {npc.base_npc.name} failed consistency check")

            return is_consistent

        except Exception as e:
            logger.error(f"Error validating NPC: {e}")
            return False

    def _determine_expertise_areas(self, archetype: str) -> list[str]:
        """Determine expertise areas based on archetype."""
        expertise_map = {
            "merchant": ["commerce", "appraisal", "persuasion"],
            "guard": ["combat", "law_enforcement", "protection"],
            "scholar": ["knowledge", "research", "teaching"],
            "hermit": ["survival", "nature_lore", "meditation"],
            "innkeeper": ["hospitality", "local_news", "cooking"],
            "artisan": ["crafting", "artistry", "materials"],
            "wanderer": ["travel", "geography", "survival"],
        }
        return expertise_map.get(archetype, ["general_knowledge"])

    def get_generation_metrics(self) -> list[NPCGenerationMetrics]:
        """Get performance metrics for NPC generation."""
        return self._generation_metrics.copy()

    def clear_metrics(self) -> None:
        """Clear collected metrics."""
        self._generation_metrics.clear()
        logger.debug("NPC generation metrics cleared")
