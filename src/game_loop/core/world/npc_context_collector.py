"""
NPC Context Collector for gathering comprehensive context for NPC generation.
"""

import logging
from typing import Any
from uuid import UUID

from ...database.session_factory import DatabaseSessionFactory
from ...state.models import Location, NonPlayerCharacter, WorldState
from ..models.location_models import LocationTheme
from ..models.npc_models import NPCGenerationContext

logger = logging.getLogger(__name__)


class NPCContextCollector:
    """Gathers comprehensive context for NPC generation from various sources."""

    def __init__(
        self, world_state: WorldState, session_factory: DatabaseSessionFactory
    ):
        """Initialize NPC context collection system."""
        self.world_state = world_state
        self.session_factory = session_factory

    async def collect_generation_context(
        self, location_id: UUID, purpose: str
    ) -> NPCGenerationContext:
        """Collect comprehensive context for NPC generation."""
        try:
            logger.debug(
                f"Collecting NPC generation context for location {location_id}"
            )

            # Get the target location
            location = self.world_state.locations.get(location_id)
            if not location:
                raise ValueError(f"Location {location_id} not found in world state")

            # Determine location theme
            location_theme = await self._determine_location_theme(location)

            # Gather nearby NPCs
            nearby_npcs = await self._get_nearby_npcs(location_id)

            # Collect world state snapshot
            world_snapshot = await self._create_world_snapshot()

            # Determine player level context
            player_level = await self._determine_player_level()

            # Apply generation constraints
            constraints = await self._gather_generation_constraints(
                location, purpose, nearby_npcs
            )

            context = NPCGenerationContext(
                location=location,
                location_theme=location_theme,
                nearby_npcs=nearby_npcs,
                world_state_snapshot=world_snapshot,
                player_level=player_level,
                generation_purpose=purpose,
                constraints=constraints,
            )

            logger.debug(
                f"Generated NPC context: theme={location_theme.name}, "
                f"nearby_npcs={len(nearby_npcs)}, purpose={purpose}"
            )

            return context

        except Exception as e:
            logger.error(f"Error collecting NPC generation context: {e}")
            raise

    async def analyze_location_needs(self, location: Location) -> dict[str, Any]:
        """Analyze what types of NPCs would enhance the location."""
        try:
            analysis = {
                "recommended_archetypes": [],
                "max_npcs": 3,
                "priority_roles": [],
                "social_needs": [],
                "functional_needs": [],
            }

            location_theme = location.state_flags.get("theme", "generic")
            location_type = location.state_flags.get("type", "generic")

            # Analyze based on location theme
            theme_recommendations = {
                "Village": {
                    "archetypes": ["merchant", "innkeeper", "artisan", "guard"],
                    "max_npcs": 4,
                    "priority": ["merchant", "innkeeper"],
                    "social": ["community_hub", "information_center"],
                    "functional": ["trade", "rest", "crafting"],
                },
                "Forest": {
                    "archetypes": ["hermit", "wanderer", "scholar"],
                    "max_npcs": 2,
                    "priority": ["hermit"],
                    "social": ["wisdom_sharing", "nature_lore"],
                    "functional": ["guidance", "natural_knowledge"],
                },
                "City": {
                    "archetypes": ["merchant", "guard", "scholar", "artisan"],
                    "max_npcs": 5,
                    "priority": ["guard", "merchant"],
                    "social": ["law_enforcement", "commerce"],
                    "functional": ["security", "trade", "information"],
                },
                "Crossroads": {
                    "archetypes": ["wanderer", "merchant", "guard"],
                    "max_npcs": 2,
                    "priority": ["wanderer"],
                    "social": ["travel_information", "news_sharing"],
                    "functional": ["directions", "travel_advice"],
                },
            }

            theme_info = theme_recommendations.get(
                location_theme,
                {
                    "archetypes": ["wanderer"],
                    "max_npcs": 1,
                    "priority": ["wanderer"],
                    "social": ["basic_interaction"],
                    "functional": ["basic_assistance"],
                },
            )

            analysis.update(
                {
                    "recommended_archetypes": theme_info["archetypes"],
                    "max_npcs": theme_info["max_npcs"],
                    "priority_roles": theme_info["priority"],
                    "social_needs": theme_info["social"],
                    "functional_needs": theme_info["functional"],
                }
            )

            # Adjust based on existing NPCs
            existing_npc_count = len(location.npcs)
            analysis["current_npc_count"] = existing_npc_count
            analysis["can_add_npcs"] = existing_npc_count < analysis["max_npcs"]

            # Analyze existing archetypes
            existing_archetypes = []
            for npc in location.npcs.values():
                # This would need to be enhanced when we have archetype data
                existing_archetypes.append("unknown")

            analysis["existing_archetypes"] = existing_archetypes
            analysis["missing_archetypes"] = [
                arch
                for arch in analysis["recommended_archetypes"]
                if arch not in existing_archetypes
            ]

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing location needs: {e}")
            return {"error": str(e)}

    async def gather_social_context(self, location_id: UUID) -> dict[str, Any]:
        """Gather information about existing social dynamics."""
        try:
            location = self.world_state.locations.get(location_id)
            if not location:
                return {}

            social_context = {
                "npc_relationships": {},
                "social_hierarchy": [],
                "conflict_potential": 0.0,
                "cooperation_level": 0.5,
                "community_mood": "neutral",
                "leadership": None,
                "factions": [],
            }

            # Analyze existing NPCs
            npcs = list(location.npcs.values())
            if not npcs:
                return social_context

            # Simple social analysis
            if len(npcs) >= 2:
                social_context["cooperation_level"] = 0.7
                social_context["community_mood"] = "collaborative"

            if len(npcs) >= 3:
                social_context["social_hierarchy"] = [npc.name for npc in npcs[:3]]

            # Look for potential leaders (guards, merchants, etc.)
            for npc in npcs:
                if "guard" in npc.name.lower() or "captain" in npc.name.lower():
                    social_context["leadership"] = npc.name
                    break
                elif "merchant" in npc.name.lower() or "trader" in npc.name.lower():
                    if not social_context["leadership"]:
                        social_context["leadership"] = npc.name

            return social_context

        except Exception as e:
            logger.error(f"Error gathering social context: {e}")
            return {}

    async def analyze_player_preferences(self, player_id: UUID) -> dict[str, Any]:
        """Analyze player interaction patterns with NPCs."""
        try:
            # This would analyze player interaction history
            # For now, return default preferences
            preferences = {
                "preferred_archetypes": ["merchant", "scholar"],
                "interaction_style": "friendly",
                "complexity_preference": "moderate",
                "dialogue_length": "medium",
                "quest_interest": 0.7,
                "lore_interest": 0.6,
                "trade_interest": 0.5,
            }

            # TODO: Implement actual player history analysis
            # This would query database for player interaction patterns

            return preferences

        except Exception as e:
            logger.error(f"Error analyzing player preferences: {e}")
            return {
                "preferred_archetypes": ["wanderer"],
                "interaction_style": "neutral",
                "complexity_preference": "simple",
            }

    async def collect_world_knowledge(self, location: Location) -> dict[str, Any]:
        """Collect relevant world knowledge for NPC background."""
        try:
            world_knowledge = {
                "current_events": [],
                "historical_events": [],
                "notable_locations": [],
                "important_figures": [],
                "cultural_context": {},
                "economic_situation": "stable",
                "political_climate": "peaceful",
            }

            # Gather information about connected locations
            connected_locations = []
            for direction, connected_id in location.connections.items():
                connected_location = self.world_state.locations.get(connected_id)
                if connected_location:
                    connected_locations.append(
                        {
                            "direction": direction,
                            "name": connected_location.name,
                            "theme": connected_location.state_flags.get(
                                "theme", "unknown"
                            ),
                        }
                    )

            world_knowledge["notable_locations"] = connected_locations

            # Analyze regional context
            location_theme = location.state_flags.get("theme", "generic")
            world_knowledge["cultural_context"] = {
                "primary_theme": location_theme,
                "regional_characteristics": self._get_regional_characteristics(
                    location_theme
                ),
                "common_knowledge": self._get_common_knowledge(location_theme),
            }

            # Add current world state information
            world_knowledge["world_size"] = len(self.world_state.locations)
            world_knowledge["exploration_level"] = self._calculate_exploration_level()

            return world_knowledge

        except Exception as e:
            logger.error(f"Error collecting world knowledge: {e}")
            return {}

    async def _determine_location_theme(self, location: Location) -> LocationTheme:
        """Determine the theme for a location."""
        theme_name = location.state_flags.get("theme", "Generic")

        # Create basic theme object
        return LocationTheme(
            name=theme_name,
            description=f"{theme_name} themed area",
            visual_elements=[theme_name.lower()],
            atmosphere="neutral",
            typical_objects=[],
            typical_npcs=[],
            generation_parameters={},
        )

    async def _get_nearby_npcs(self, location_id: UUID) -> list[NonPlayerCharacter]:
        """Get NPCs in the target location and adjacent locations."""
        nearby_npcs: list[NonPlayerCharacter] = []

        # Get NPCs from target location
        target_location = self.world_state.locations.get(location_id)
        if target_location:
            nearby_npcs.extend(target_location.npcs.values())

        # Get NPCs from adjacent locations
        if target_location:
            for connected_id in target_location.connections.values():
                connected_location = self.world_state.locations.get(connected_id)
                if connected_location:
                    nearby_npcs.extend(connected_location.npcs.values())

        return nearby_npcs

    async def _create_world_snapshot(self) -> dict[str, Any]:
        """Create a snapshot of relevant world state."""
        return {
            "total_locations": len(self.world_state.locations),
            "total_npcs": sum(
                len(loc.npcs) for loc in self.world_state.locations.values()
            ),
            "themes_present": list(
                set(
                    loc.state_flags.get("theme", "generic")
                    for loc in self.world_state.locations.values()
                )
            ),
            "world_complexity": "developing",
        }

    async def _determine_player_level(self) -> int:
        """Determine effective player level for content appropriateness."""
        # This would analyze player progression
        # For now, return a moderate level
        return 3

    async def _gather_generation_constraints(
        self, location: Location, purpose: str, nearby_npcs: list[NonPlayerCharacter]
    ) -> dict[str, Any]:
        """Gather constraints that should guide NPC generation."""
        constraints = {
            "max_npcs_per_location": 3,
            "avoid_duplicate_archetypes": True,
            "maintain_theme_consistency": True,
            "respect_location_capacity": True,
            "consider_existing_relationships": True,
        }

        # Adjust based on location characteristics
        location_type = location.state_flags.get("type", "generic")
        if location_type == "city":
            constraints["max_npcs_per_location"] = 5
        elif location_type == "village":
            constraints["max_npcs_per_location"] = 4
        elif location_type in ["cave", "clearing"]:
            constraints["max_npcs_per_location"] = 2

        # Add purpose-specific constraints
        if purpose == "quest_related":
            constraints["ensure_quest_capability"] = True
            constraints["knowledge_requirements"] = "specialized"

        # Consider existing NPC density
        current_npc_count = len(location.npcs)
        constraints["current_npc_count"] = current_npc_count
        constraints["can_add_npc"] = (
            current_npc_count < constraints["max_npcs_per_location"]
        )

        return constraints

    def _get_regional_characteristics(self, theme: str) -> list[str]:
        """Get characteristics typical of a regional theme."""
        characteristics = {
            "Forest": ["natural", "peaceful", "wild", "ancient"],
            "Village": ["communal", "agricultural", "traditional", "close-knit"],
            "City": ["bustling", "diverse", "commercial", "structured"],
            "Mountain": ["rugged", "challenging", "isolated", "hardy"],
            "Cave": ["mysterious", "hidden", "ancient", "dangerous"],
        }
        return characteristics.get(theme, ["neutral"])

    def _get_common_knowledge(self, theme: str) -> list[str]:
        """Get common knowledge for a regional theme."""
        knowledge = {
            "Forest": [
                "local_wildlife",
                "safe_paths",
                "seasonal_changes",
                "natural_resources",
            ],
            "Village": [
                "local_families",
                "crop_seasons",
                "market_days",
                "local_customs",
            ],
            "City": ["districts", "important_buildings", "trade_routes", "laws"],
            "Mountain": [
                "weather_patterns",
                "safe_passes",
                "local_dangers",
                "mineral_deposits",
            ],
        }
        return knowledge.get(theme, ["basic_survival"])

    def _calculate_exploration_level(self) -> float:
        """Calculate how much of the world has been explored."""
        # Simple calculation based on location count
        location_count = len(self.world_state.locations)
        if location_count <= 3:
            return 0.2
        elif location_count <= 6:
            return 0.4
        elif location_count <= 10:
            return 0.6
        else:
            return 0.8
