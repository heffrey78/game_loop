"""
Object Context Collector for gathering comprehensive context for object generation.

This module analyzes locations, existing objects, player patterns, and world state
to provide intelligent context for object generation decisions.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from game_loop.core.models.location_models import LocationTheme
from game_loop.core.models.object_models import ObjectGenerationContext
from game_loop.database.session_factory import DatabaseSessionFactory
from game_loop.state.models import Location, WorldState

logger = logging.getLogger(__name__)


class ObjectContextCollector:
    """Gathers comprehensive context for intelligent object generation."""

    def __init__(
        self, world_state: WorldState, session_factory: DatabaseSessionFactory
    ):
        self.world_state = world_state
        self.session_factory = session_factory

    async def collect_generation_context(
        self, location_id: UUID, purpose: str
    ) -> ObjectGenerationContext:
        """Collect comprehensive context for object generation."""
        try:
            # Get the location
            location = self.world_state.locations.get(location_id)
            if not location:
                raise ValueError(f"Location {location_id} not found in world state")

            # Determine location theme
            theme_name = location.state_flags.get("theme", "Unknown")
            location_theme = self._create_location_theme(theme_name, location)

            # Get existing objects in the location
            existing_objects = list(location.objects.values())

            # Analyze location needs
            location_analysis = await self.analyze_location_needs(location)

            # Gather object context
            object_context = await self.gather_object_context(location_id)

            # Collect world knowledge
            world_knowledge = await self.collect_world_knowledge(location)

            # Analyze player preferences (simplified for now)
            player_preferences = await self.analyze_player_preferences(
                UUID("00000000-0000-0000-0000-000000000000")
            )

            # Build constraints from analysis
            constraints = {
                "location_needs": location_analysis,
                "object_density": object_context.get("density", "normal"),
                "style_preferences": player_preferences.get("style", "balanced"),
                "avoid_duplicates": True,
                "theme_consistency": True,
            }

            # Create world state snapshot
            world_snapshot = {
                "location_type": location.state_flags.get("type", "unknown"),
                "population": location.state_flags.get("population", "unknown"),
                "danger_level": location.state_flags.get("danger_level", "unknown"),
                "economic_status": world_knowledge.get("economic_status", "normal"),
                "cultural_influences": world_knowledge.get("cultural_influences", []),
                "nearby_locations": self._get_nearby_location_info(location),
            }

            # Determine appropriate player level context
            player_level = self._estimate_appropriate_level(location, purpose)

            context = ObjectGenerationContext(
                location=location,
                location_theme=location_theme,
                generation_purpose=purpose,
                existing_objects=existing_objects,
                player_level=player_level,
                constraints=constraints,
                world_state_snapshot=world_snapshot,
            )

            logger.info(
                f"Collected generation context for {location.name}: {len(existing_objects)} existing objects"
            )
            return context

        except Exception as e:
            logger.error(f"Error collecting generation context for {location_id}: {e}")
            # Create minimal fallback context
            fallback_location = Location(
                location_id=location_id,
                name="Unknown Location",
                description="A mysterious place",
                connections={},
                objects={},
                npcs={},
                state_flags={"theme": "Unknown"},
            )

            fallback_theme = LocationTheme(
                name="Unknown",
                description="Unknown theme",
                visual_elements=[],
                atmosphere="unknown",
                typical_objects=[],
                typical_npcs=[],
                generation_parameters={},
            )

            return ObjectGenerationContext(
                location=fallback_location,
                location_theme=fallback_theme,
                generation_purpose=purpose,
                existing_objects=[],
                player_level=3,
                constraints={},
                world_state_snapshot={},
            )

    async def analyze_location_needs(self, location: Location) -> dict[str, Any]:
        """Analyze what types of objects the location needs."""
        try:
            analysis = {
                "missing_object_types": [],
                "density_level": "normal",
                "functional_gaps": [],
                "thematic_needs": [],
                "priority_objects": [],
            }

            # Count existing objects by type
            object_type_counts = {}
            for obj in location.objects.values():
                obj_type = getattr(obj, "object_type", "unknown")
                object_type_counts[obj_type] = object_type_counts.get(obj_type, 0) + 1

            # Determine density
            total_objects = len(location.objects)
            if total_objects < 3:
                analysis["density_level"] = "sparse"
            elif total_objects > 10:
                analysis["density_level"] = "crowded"
            else:
                analysis["density_level"] = "normal"

            # Analyze by location theme
            theme = location.state_flags.get("theme", "Unknown")
            location_type = location.state_flags.get("type", "unknown")

            if theme == "Village":
                analysis["thematic_needs"] = ["tool", "container", "furniture"]
                if "tool" not in object_type_counts:
                    analysis["missing_object_types"].append("tool")
                if "container" not in object_type_counts:
                    analysis["missing_object_types"].append("container")

            elif theme == "Forest":
                analysis["thematic_needs"] = ["natural", "survival_gear", "herb"]
                if "natural" not in object_type_counts:
                    analysis["missing_object_types"].append("natural")

            elif theme == "City":
                analysis["thematic_needs"] = ["luxury", "art", "tool", "container"]
                if "art" not in object_type_counts:
                    analysis["missing_object_types"].append("art")

            elif theme == "Dungeon":
                analysis["thematic_needs"] = ["treasure", "trap", "relic", "mystery"]
                if "treasure" not in object_type_counts:
                    analysis["missing_object_types"].append("treasure")

            # Functional analysis
            has_storage = any(
                "container" in getattr(obj, "object_type", "")
                for obj in location.objects.values()
            )
            has_light_source = any(
                "light" in getattr(obj, "special_properties", [])
                for obj in location.objects.values()
            )

            if not has_storage and theme in ["Village", "City", "Dungeon"]:
                analysis["functional_gaps"].append("storage")
                analysis["priority_objects"].append("container")

            if not has_light_source and theme in ["Dungeon", "Cave"]:
                analysis["functional_gaps"].append("illumination")
                analysis["priority_objects"].append("light_source")

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing location needs: {e}")
            return {
                "missing_object_types": ["container"],
                "density_level": "normal",
                "functional_gaps": [],
                "thematic_needs": ["container"],
                "priority_objects": ["container"],
            }

    async def gather_object_context(self, location_id: UUID) -> dict[str, Any]:
        """Gather context about existing objects and interactions."""
        try:
            location = self.world_state.locations.get(location_id)
            if not location:
                return {"density": "normal", "object_types": [], "total_value": 0}

            context = {
                "density": "normal",
                "object_types": [],
                "total_value": 0,
                "materials_present": [],
                "interaction_types": [],
                "rarity_distribution": {"common": 0, "uncommon": 0, "rare": 0},
            }

            total_objects = len(location.objects)
            total_value = 0
            materials = set()
            object_types = set()

            for obj in location.objects.values():
                # Basic properties
                obj_type = getattr(obj, "object_type", "unknown")
                object_types.add(obj_type)

                # Material analysis (if available)
                material = getattr(obj, "material", "unknown")
                if material != "unknown":
                    materials.add(material)

                # Value analysis (if available)
                value = getattr(obj, "value", 0)
                total_value += value

                # Rarity estimation based on value
                if value > 100:
                    context["rarity_distribution"]["rare"] += 1
                elif value > 50:
                    context["rarity_distribution"]["uncommon"] += 1
                else:
                    context["rarity_distribution"]["common"] += 1

            # Set density
            if total_objects < 2:
                context["density"] = "sparse"
            elif total_objects > 8:
                context["density"] = "crowded"
            else:
                context["density"] = "normal"

            context["object_types"] = list(object_types)
            context["total_value"] = total_value
            context["materials_present"] = list(materials)

            return context

        except Exception as e:
            logger.error(f"Error gathering object context: {e}")
            return {"density": "normal", "object_types": [], "total_value": 0}

    async def analyze_player_preferences(self, player_id: UUID) -> dict[str, Any]:
        """Analyze player interaction patterns for object generation."""
        try:
            # In a full implementation, this would analyze:
            # - Objects the player has interacted with
            # - Types of objects the player seeks out
            # - Player's preferred interaction styles
            # - Player's progression and interests

            # For now, return reasonable defaults
            preferences = {
                "preferred_object_types": ["tool", "container", "weapon"],
                "interaction_style": "explorer",  # "combat", "social", "explorer", "collector"
                "complexity_preference": "moderate",  # "simple", "moderate", "complex"
                "value_preference": "balanced",  # "practical", "valuable", "balanced"
                "style": "balanced",  # "rustic", "elegant", "practical", "balanced"
            }

            # Simple analysis based on player level (would be more sophisticated in real implementation)
            # For now, just provide defaults

            return preferences

        except Exception as e:
            logger.error(f"Error analyzing player preferences: {e}")
            return {
                "preferred_object_types": ["container", "tool"],
                "interaction_style": "explorer",
                "complexity_preference": "moderate",
                "value_preference": "balanced",
                "style": "balanced",
            }

    async def collect_world_knowledge(self, location: Location) -> dict[str, Any]:
        """Collect relevant world knowledge for object context."""
        try:
            knowledge = {
                "economic_status": "normal",
                "cultural_influences": [],
                "historical_period": "medieval",
                "technology_level": "pre_industrial",
                "magical_presence": "low",
                "trade_activity": "normal",
                "population_wealth": "mixed",
            }

            # Analyze based on location flags
            theme = location.state_flags.get("theme", "Unknown")
            location_type = location.state_flags.get("type", "unknown")

            # Economic analysis
            if theme == "City":
                knowledge["economic_status"] = "prosperous"
                knowledge["trade_activity"] = "high"
                knowledge["population_wealth"] = "wealthy"
            elif theme == "Village":
                knowledge["economic_status"] = "modest"
                knowledge["trade_activity"] = "low"
                knowledge["population_wealth"] = "poor_to_modest"
            elif theme == "Dungeon":
                knowledge["economic_status"] = "unknown"
                knowledge["trade_activity"] = "none"
                knowledge["population_wealth"] = "unknown"

            # Cultural influences based on theme
            if theme == "Forest":
                knowledge["cultural_influences"] = [
                    "nature_worship",
                    "druidic",
                    "primitive",
                ]
            elif theme == "City":
                knowledge["cultural_influences"] = [
                    "urban",
                    "mercantile",
                    "sophisticated",
                ]
            elif theme == "Village":
                knowledge["cultural_influences"] = [
                    "rural",
                    "agricultural",
                    "traditional",
                ]
            elif theme == "Dungeon":
                knowledge["cultural_influences"] = [
                    "ancient",
                    "mysterious",
                    "forgotten",
                ]

            # Technology and magic levels
            danger_level = location.state_flags.get("danger_level", "low")
            if danger_level in ["high", "extreme"]:
                knowledge["magical_presence"] = "high"
            elif danger_level == "medium":
                knowledge["magical_presence"] = "moderate"

            return knowledge

        except Exception as e:
            logger.error(f"Error collecting world knowledge: {e}")
            return {
                "economic_status": "normal",
                "cultural_influences": ["generic"],
                "historical_period": "medieval",
                "technology_level": "pre_industrial",
                "magical_presence": "low",
            }

    def _create_location_theme(
        self, theme_name: str, location: Location
    ) -> LocationTheme:
        """Create a LocationTheme object from theme name and location data."""
        try:
            # Define characteristics for different themes
            theme_characteristics = {
                "Village": ["rural", "peaceful", "agricultural", "community"],
                "Forest": ["natural", "wild", "organic", "secluded"],
                "City": ["urban", "bustling", "sophisticated", "diverse"],
                "Dungeon": ["dark", "mysterious", "dangerous", "ancient"],
                "Cave": ["dark", "damp", "echoing", "hidden"],
                "Mountain": ["high", "cold", "rugged", "remote"],
            }

            # Define object affinities
            object_affinities = {
                "Village": {
                    "tool": 0.8,
                    "container": 0.7,
                    "furniture": 0.6,
                    "weapon": 0.5,
                },
                "Forest": {
                    "natural": 0.9,
                    "herb": 0.8,
                    "survival_gear": 0.7,
                    "primitive_tool": 0.6,
                },
                "City": {"luxury": 0.8, "art": 0.7, "refined_tool": 0.6, "book": 0.6},
                "Dungeon": {"treasure": 0.9, "relic": 0.8, "trap": 0.7, "mystery": 0.8},
            }

            # Define atmosphere tags
            atmosphere_tags = {
                "Village": ["peaceful", "rustic", "homey", "safe"],
                "Forest": ["natural", "wild", "mysterious", "alive"],
                "City": ["busy", "sophisticated", "wealthy", "diverse"],
                "Dungeon": ["ominous", "ancient", "dangerous", "forgotten"],
            }

            characteristics = theme_characteristics.get(theme_name, ["generic"])
            affinities = object_affinities.get(theme_name, {})
            atmosphere = atmosphere_tags.get(theme_name, ["neutral"])

            return LocationTheme(
                name=theme_name,
                description=f"A {theme_name.lower()} environment",
                visual_elements=characteristics,
                atmosphere=atmosphere[0] if atmosphere else "neutral",
                typical_objects=list(affinities.keys()) if affinities else ["generic"],
                typical_npcs=["citizen", "traveler"],
                generation_parameters={"object_affinities": affinities},
            )

        except Exception as e:
            logger.error(f"Error creating location theme: {e}")
            return LocationTheme(
                name="Unknown",
                description="An unknown type of location",
                visual_elements=["generic"],
                atmosphere="neutral",
                typical_objects=["generic"],
                typical_npcs=["citizen"],
                generation_parameters={},
            )

    def _get_nearby_location_info(self, location: Location) -> dict[str, Any]:
        """Get information about nearby connected locations."""
        try:
            nearby_info = {
                "connected_themes": [],
                "connection_count": len(location.connections),
                "nearby_population": 0,
                "trade_routes": False,
            }

            for direction, connected_id in location.connections.items():
                connected_location = self.world_state.locations.get(connected_id)
                if connected_location:
                    connected_theme = connected_location.state_flags.get(
                        "theme", "Unknown"
                    )
                    if connected_theme not in nearby_info["connected_themes"]:
                        nearby_info["connected_themes"].append(connected_theme)

                    # Check if it's a trade route (connecting to cities or villages)
                    if connected_theme in ["City", "Village", "Town"]:
                        nearby_info["trade_routes"] = True

            return nearby_info

        except Exception as e:
            logger.error(f"Error getting nearby location info: {e}")
            return {
                "connected_themes": [],
                "connection_count": 0,
                "nearby_population": 0,
            }

    def _estimate_appropriate_level(self, location: Location, purpose: str) -> int:
        """Estimate appropriate player level for this context."""
        try:
            # Base level from location danger
            danger_level = location.state_flags.get("danger_level", "low")
            base_level = {
                "very_low": 1,
                "low": 2,
                "medium": 4,
                "high": 6,
                "very_high": 8,
                "extreme": 10,
            }.get(danger_level, 3)

            # Adjust for purpose
            purpose_modifiers = {
                "populate_location": 0,
                "quest_related": +1,
                "random_encounter": 0,
                "narrative_enhancement": +1,
                "environmental_storytelling": 0,
            }

            modifier = purpose_modifiers.get(purpose, 0)
            final_level = max(1, min(10, base_level + modifier))

            return final_level

        except Exception as e:
            logger.error(f"Error estimating appropriate level: {e}")
            return 3
