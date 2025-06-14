"""
NPC Theme and Archetype Manager for contextual NPC generation.
"""

import logging
from typing import Any
from uuid import uuid4

from ...database.session_factory import DatabaseSessionFactory
from ...state.models import Location, WorldState
from ..models.location_models import LocationTheme
from typing import TYPE_CHECKING

from ..models.npc_models import GeneratedNPC, NPCArchetype, NPCPersonality

if TYPE_CHECKING:
    from ..models.npc_models import NPCGenerationContext

logger = logging.getLogger(__name__)


class NPCThemeManager:
    """Manages NPC archetypes, themes, and ensures consistency with world and location themes."""

    def __init__(
        self, world_state: WorldState, session_factory: DatabaseSessionFactory
    ):
        """Initialize NPC theme management system."""
        self.world_state = world_state
        self.session_factory = session_factory

        # Define default archetypes
        self._default_archetypes = self._create_default_archetypes()

    def _create_default_archetypes(self) -> dict[str, NPCArchetype]:
        """Create default archetype definitions."""
        return {
            "merchant": NPCArchetype(
                name="merchant",
                description="A trader who buys and sells goods",
                typical_traits=["persuasive", "business-minded", "social"],
                typical_motivations=["profit", "reputation", "trade_routes"],
                speech_patterns={"formality": "polite", "verbosity": "moderate"},
                location_affinities={
                    "Village": 0.9,
                    "City": 0.8,
                    "Town": 0.7,
                    "Crossroads": 0.6,
                    "Forest": 0.2,
                },
                archetype_id=uuid4(),
            ),
            "guard": NPCArchetype(
                name="guard",
                description="A protector of people and places",
                typical_traits=["vigilant", "dutiful", "protective"],
                typical_motivations=["duty", "safety", "order"],
                speech_patterns={"formality": "formal", "verbosity": "concise"},
                location_affinities={
                    "City": 0.9,
                    "Town": 0.8,
                    "Village": 0.7,
                    "Castle": 0.9,
                    "Forest": 0.3,
                },
                archetype_id=uuid4(),
            ),
            "scholar": NPCArchetype(
                name="scholar",
                description="A learned person devoted to study and research",
                typical_traits=["knowledgeable", "curious", "analytical"],
                typical_motivations=["knowledge", "discovery", "teaching"],
                speech_patterns={"formality": "formal", "verbosity": "verbose"},
                location_affinities={
                    "Library": 0.9,
                    "Academy": 0.9,
                    "City": 0.6,
                    "Tower": 0.8,
                    "Forest": 0.4,
                },
                archetype_id=uuid4(),
            ),
            "hermit": NPCArchetype(
                name="hermit",
                description="A solitary person who lives apart from society",
                typical_traits=["wise", "reclusive", "self-sufficient"],
                typical_motivations=["solitude", "wisdom", "nature"],
                speech_patterns={"formality": "casual", "verbosity": "cryptic"},
                location_affinities={
                    "Forest": 0.9,
                    "Mountain": 0.8,
                    "Cave": 0.7,
                    "Wilderness": 0.9,
                    "City": 0.1,
                },
                archetype_id=uuid4(),
            ),
            "innkeeper": NPCArchetype(
                name="innkeeper",
                description="A host who provides food, drink, and lodging",
                typical_traits=["hospitable", "social", "practical"],
                typical_motivations=["hospitality", "community", "stories"],
                speech_patterns={"formality": "casual", "verbosity": "moderate"},
                location_affinities={
                    "Inn": 0.9,
                    "Tavern": 0.9,
                    "Village": 0.7,
                    "Town": 0.8,
                    "Forest": 0.2,
                },
                archetype_id=uuid4(),
            ),
            "artisan": NPCArchetype(
                name="artisan",
                description="A skilled craftsperson who creates goods",
                typical_traits=["skilled", "creative", "dedicated"],
                typical_motivations=["craftsmanship", "beauty", "utility"],
                speech_patterns={"formality": "casual", "verbosity": "moderate"},
                location_affinities={
                    "Workshop": 0.9,
                    "Village": 0.7,
                    "Town": 0.8,
                    "City": 0.6,
                    "Forest": 0.3,
                },
                archetype_id=uuid4(),
            ),
            "wanderer": NPCArchetype(
                name="wanderer",
                description="A traveler who roams from place to place",
                typical_traits=["adventurous", "experienced", "independent"],
                typical_motivations=["exploration", "freedom", "stories"],
                speech_patterns={"formality": "casual", "verbosity": "storytelling"},
                location_affinities={
                    "Crossroads": 0.8,
                    "Forest": 0.7,
                    "Mountain": 0.6,
                    "Path": 0.9,
                    "City": 0.4,
                },
                archetype_id=uuid4(),
            ),
        }

    async def get_available_archetypes(self, location_theme: str) -> list[str]:
        """Get NPC archetypes appropriate for a location theme."""
        try:
            logger.debug(f"Getting available archetypes for theme: {location_theme}")

            # Load custom archetypes from database
            custom_archetypes = await self._load_custom_archetypes()

            # Combine default and custom archetypes
            all_archetypes = {**self._default_archetypes, **custom_archetypes}

            # Filter by location affinity
            suitable_archetypes = []
            for archetype_name, archetype in all_archetypes.items():
                affinity = archetype.location_affinities.get(location_theme, 0.0)
                if affinity >= 0.3:  # Minimum affinity threshold
                    suitable_archetypes.append(archetype_name)

            # Always include at least wanderer as fallback
            if not suitable_archetypes:
                suitable_archetypes = ["wanderer"]

            logger.debug(
                f"Found {len(suitable_archetypes)} suitable archetypes for {location_theme}"
            )
            return suitable_archetypes

        except Exception as e:
            logger.error(f"Error getting available archetypes: {e}")
            return ["wanderer"]  # Fallback

    async def determine_npc_archetype(self, context: "NPCGenerationContext") -> str:
        """Determine the most appropriate archetype for the context."""
        try:
            location_theme = context.location_theme.name
            available_archetypes = await self.get_available_archetypes(location_theme)

            # Consider generation purpose
            purpose_preferences = {
                "populate_location": ["merchant", "guard", "artisan", "innkeeper"],
                "quest_related": ["scholar", "hermit", "wanderer"],
                "random_encounter": ["wanderer", "hermit"],
                "social": ["merchant", "innkeeper"],
            }

            preferred = purpose_preferences.get(context.generation_purpose, [])

            # Find intersection of available and preferred
            suitable = [arch for arch in preferred if arch in available_archetypes]

            if suitable:
                # Select based on location affinity
                best_archetype = suitable[0]
                best_affinity = 0.0

                for archetype_name in suitable:
                    if archetype_name in self._default_archetypes:
                        affinity = self._default_archetypes[
                            archetype_name
                        ].location_affinities.get(location_theme, 0.0)
                        if affinity > best_affinity:
                            best_affinity = affinity
                            best_archetype = archetype_name

                return best_archetype

            # Fallback to first available
            return available_archetypes[0] if available_archetypes else "wanderer"

        except Exception as e:
            logger.error(f"Error determining NPC archetype: {e}")
            return "wanderer"

    async def validate_npc_consistency(
        self, npc: GeneratedNPC, location: Location
    ) -> bool:
        """Validate that an NPC is consistent with their environment."""
        try:
            archetype_name = npc.personality.archetype
            location_theme = location.state_flags.get("theme", "generic")

            # Get archetype definition
            archetype = self._default_archetypes.get(archetype_name)
            if not archetype:
                return True  # Unknown archetype, assume valid

            # Check location affinity
            affinity = archetype.location_affinities.get(location_theme, 0.5)

            # Check trait consistency
            expected_traits = set(archetype.typical_traits)
            actual_traits = set(npc.personality.traits)
            trait_overlap = len(expected_traits.intersection(actual_traits))
            trait_consistency = (
                trait_overlap / len(expected_traits) if expected_traits else 1.0
            )

            # Overall consistency score
            consistency_score = (affinity * 0.6) + (trait_consistency * 0.4)

            logger.debug(
                f"NPC consistency: affinity={affinity:.2f}, traits={trait_consistency:.2f}, "
                f"overall={consistency_score:.2f}"
            )

            return consistency_score >= 0.4  # Minimum consistency threshold

        except Exception as e:
            logger.error(f"Error validating NPC consistency: {e}")
            return True  # Assume valid on error

    async def get_personality_template(
        self, archetype: str, location_theme: str
    ) -> NPCPersonality:
        """Get a personality template for the archetype and theme."""
        try:
            archetype_def = self._default_archetypes.get(archetype)
            if not archetype_def:
                # Create basic template
                return NPCPersonality(
                    name="Unknown",
                    archetype=archetype,
                    traits=["neutral"],
                    motivations=["survival"],
                    fears=["unknown"],
                    speech_patterns={"formality": "casual", "verbosity": "moderate"},
                    relationship_tendencies={"friendly": 0.5},
                )

            # Create template based on archetype
            template = NPCPersonality(
                name=archetype_def.name.title(),
                archetype=archetype,
                traits=archetype_def.typical_traits.copy(),
                motivations=archetype_def.typical_motivations.copy(),
                fears=self._generate_contextual_fears(archetype, location_theme),
                speech_patterns=archetype_def.speech_patterns.copy(),
                relationship_tendencies=self._generate_relationship_tendencies(
                    archetype
                ),
            )

            return template

        except Exception as e:
            logger.error(f"Error getting personality template: {e}")
            return NPCPersonality(
                name="Generic",
                archetype=archetype,
                traits=["neutral"],
                motivations=["survival"],
                fears=["unknown"],
            )

    async def generate_cultural_variations(
        self, base_personality: NPCPersonality, location: Location
    ) -> NPCPersonality:
        """Apply cultural variations based on location and world state."""
        try:
            # Create a copy to modify
            varied_personality = NPCPersonality(
                name=base_personality.name,
                archetype=base_personality.archetype,
                traits=base_personality.traits.copy(),
                motivations=base_personality.motivations.copy(),
                fears=base_personality.fears.copy(),
                speech_patterns=base_personality.speech_patterns.copy(),
                relationship_tendencies=base_personality.relationship_tendencies.copy(),
            )

            # Apply location-based variations
            location_theme = location.state_flags.get("theme", "generic")

            if location_theme == "Forest":
                varied_personality.traits.extend(["nature-loving", "observant"])
                varied_personality.motivations.append("environmental_protection")
            elif location_theme == "City":
                varied_personality.traits.extend(["streetwise", "networked"])
                varied_personality.motivations.append("social_advancement")
            elif location_theme == "Mountain":
                varied_personality.traits.extend(["hardy", "resilient"])
                varied_personality.motivations.append("endurance")

            # Remove duplicates
            varied_personality.traits = list(set(varied_personality.traits))
            varied_personality.motivations = list(set(varied_personality.motivations))

            return varied_personality

        except Exception as e:
            logger.error(f"Error generating cultural variations: {e}")
            return base_personality

    def _generate_contextual_fears(
        self, archetype: str, location_theme: str
    ) -> list[str]:
        """Generate contextual fears based on archetype and location."""
        base_fears = {
            "merchant": ["theft", "bad_reputation", "market_crash"],
            "guard": ["failure", "corruption", "lawlessness"],
            "scholar": ["ignorance", "lost_knowledge", "censorship"],
            "hermit": ["crowds", "civilization", "intrusion"],
            "innkeeper": ["empty_rooms", "bad_reviews", "violence"],
            "artisan": ["poor_craftsmanship", "lack_of_materials", "competition"],
            "wanderer": ["being_trapped", "boredom", "roots"],
        }

        location_fears = {
            "Forest": ["fire", "deforestation", "getting_lost"],
            "City": ["crime", "poverty", "isolation"],
            "Mountain": ["avalanche", "cold", "isolation"],
            "Cave": ["collapse", "darkness", "being_trapped"],
        }

        fears = base_fears.get(archetype, ["unknown"])
        fears.extend(location_fears.get(location_theme, []))

        return fears[:3]  # Limit to 3 fears

    def _generate_relationship_tendencies(self, archetype: str) -> dict[str, float]:
        """Generate relationship tendencies based on archetype."""
        base_tendencies = {
            "merchant": {"friendly": 0.8, "trusting": 0.6, "helpful": 0.7},
            "guard": {"protective": 0.9, "suspicious": 0.7, "dutiful": 0.8},
            "scholar": {"curious": 0.9, "patient": 0.8, "analytical": 0.7},
            "hermit": {"withdrawn": 0.8, "wise": 0.7, "mysterious": 0.9},
            "innkeeper": {"hospitable": 0.9, "social": 0.8, "caring": 0.7},
            "artisan": {"proud": 0.7, "perfectionist": 0.8, "focused": 0.6},
            "wanderer": {"independent": 0.9, "experienced": 0.8, "adaptable": 0.7},
        }

        return base_tendencies.get(archetype, {"neutral": 0.5})

    async def _load_custom_archetypes(self) -> dict[str, NPCArchetype]:
        """Load custom archetypes from database."""
        try:
            async with self.session_factory.get_session() as session:
                # This would load from npc_archetypes table
                # For now, return empty dict
                return {}
        except Exception as e:
            logger.error(f"Error loading custom archetypes: {e}")
            return {}

    def get_archetype_definition(self, archetype_name: str) -> NPCArchetype | None:
        """Get the definition for a specific archetype."""
        return self._default_archetypes.get(archetype_name)

    def get_all_archetypes(self) -> dict[str, NPCArchetype]:
        """Get all available archetype definitions."""
        return self._default_archetypes.copy()
