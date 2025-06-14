"""
Object Theme Manager for managing object archetypes and theme consistency.

This module provides theme-based object generation, cultural variations,
and consistency validation for dynamic object creation.
"""

from __future__ import annotations

import logging

from game_loop.core.models.object_models import (
    GeneratedObject,
    ObjectArchetype,
    ObjectGenerationContext,
    ObjectProperties,
    ObjectTheme,
)
from game_loop.database.session_factory import DatabaseSessionFactory
from game_loop.state.models import Location, WorldState

logger = logging.getLogger(__name__)


class ObjectThemeManager:
    """Manages object archetypes, themes, and cultural consistency."""

    def __init__(
        self, world_state: WorldState, session_factory: DatabaseSessionFactory
    ):
        self.world_state = world_state
        self.session_factory = session_factory
        self._archetype_cache: dict[str, ObjectArchetype] = {}
        self._theme_cache: dict[str, ObjectTheme] = {}
        self._initialize_default_archetypes()
        self._initialize_default_themes()

    def _initialize_default_archetypes(self):
        """Initialize default object archetypes with location affinities."""

        # Weapon archetypes
        self._archetype_cache["sword"] = ObjectArchetype(
            name="sword",
            description="A bladed weapon for combat and ceremony",
            typical_properties=ObjectProperties(
                name="iron sword",
                object_type="weapon",
                material="iron",
                size="medium",
                weight="heavy",
                durability="sturdy",
                value=50,
                special_properties=["sharp", "balanced", "weapon"],
                cultural_significance="martial",
            ),
            location_affinities={
                "Village": 0.6,
                "City": 0.8,
                "Castle": 0.9,
                "Battlefield": 0.95,
            },
            interaction_templates={
                "combat": ["attack", "parry", "flourish"],
                "utility": ["cut", "pry"],
            },
            rarity="uncommon",
        )

        # Tool archetypes
        self._archetype_cache["hammer"] = ObjectArchetype(
            name="hammer",
            description="A tool for building and crafting",
            typical_properties=ObjectProperties(
                name="smithing hammer",
                object_type="tool",
                material="iron_and_wood",
                size="medium",
                weight="heavy",
                durability="very_sturdy",
                value=25,
                special_properties=["blunt", "tool", "crafting"],
                cultural_significance="artisan",
            ),
            location_affinities={
                "Village": 0.8,
                "City": 0.7,
                "Forge": 0.95,
                "Workshop": 0.9,
            },
            interaction_templates={
                "crafting": ["hammer", "forge", "shape"],
                "combat": ["bludgeon"],
            },
            rarity="common",
        )

        # Container archetypes
        self._archetype_cache["chest"] = ObjectArchetype(
            name="chest",
            description="A storage container for valuables",
            typical_properties=ObjectProperties(
                name="wooden chest",
                object_type="container",
                material="wood",
                size="large",
                weight="heavy",
                durability="sturdy",
                value=30,
                special_properties=["storage", "lockable", "container"],
                cultural_significance="common",
            ),
            location_affinities={
                "Village": 0.7,
                "City": 0.6,
                "Dungeon": 0.8,
                "Treasure_Room": 0.95,
            },
            interaction_templates={
                "storage": ["open", "close", "lock", "unlock"],
                "utility": ["examine", "search"],
            },
            rarity="common",
        )

        # Book archetypes
        self._archetype_cache["book"] = ObjectArchetype(
            name="book",
            description="A collection of written knowledge",
            typical_properties=ObjectProperties(
                name="leather-bound tome",
                object_type="knowledge",
                material="parchment_and_leather",
                size="small",
                weight="light",
                durability="fragile",
                value=40,
                special_properties=["readable", "knowledge", "fragile"],
                cultural_significance="scholarly",
            ),
            location_affinities={
                "Library": 0.95,
                "Study": 0.9,
                "City": 0.7,
                "School": 0.85,
            },
            interaction_templates={
                "knowledge": ["read", "study", "research"],
                "utility": ["examine", "carry"],
            },
            rarity="uncommon",
        )

        # Natural archetypes
        self._archetype_cache["herb"] = ObjectArchetype(
            name="herb",
            description="A natural plant with medicinal or culinary properties",
            typical_properties=ObjectProperties(
                name="healing herb",
                object_type="natural",
                material="plant",
                size="tiny",
                weight="light",
                durability="fragile",
                value=15,
                special_properties=["medicinal", "consumable", "natural"],
                cultural_significance="healing",
            ),
            location_affinities={
                "Forest": 0.9,
                "Garden": 0.8,
                "Meadow": 0.85,
                "Wilderness": 0.7,
            },
            interaction_templates={
                "healing": ["consume", "brew", "apply"],
                "utility": ["gather", "examine"],
            },
            rarity="common",
        )

        # Gemstone archetypes
        self._archetype_cache["gem"] = ObjectArchetype(
            name="gem",
            description="A precious stone of value and beauty",
            typical_properties=ObjectProperties(
                name="sapphire gem",
                object_type="treasure",
                material="crystal",
                size="tiny",
                weight="light",
                durability="very_sturdy",
                value=200,
                special_properties=["precious", "beautiful", "magical_conduit"],
                cultural_significance="wealth",
            ),
            location_affinities={
                "Mine": 0.7,
                "Cave": 0.6,
                "Treasure_Room": 0.9,
                "Jewelry_Shop": 0.8,
            },
            interaction_templates={
                "treasure": ["appraise", "trade", "admire"],
                "magic": ["channel", "enhance"],
            },
            rarity="rare",
        )

    def _initialize_default_themes(self):
        """Initialize default object themes for different location types."""

        self._theme_cache["Village"] = ObjectTheme(
            name="Village",
            description="Simple, practical objects suited for rural life",
            typical_materials=["wood", "iron", "cloth", "leather", "ceramic"],
            common_object_types=[
                "tool",
                "furniture",
                "container",
                "weapon",
                "clothing",
            ],
            cultural_elements={
                "style": "rustic",
                "craftsmanship": "local",
                "decoration": "simple",
            },
            style_descriptors=[
                "weathered",
                "handmade",
                "practical",
                "sturdy",
                "well-used",
            ],
            forbidden_elements=["luxury", "ornate", "magical", "exotic"],
        )

        self._theme_cache["Forest"] = ObjectTheme(
            name="Forest",
            description="Natural objects and items suited for wilderness",
            typical_materials=["wood", "stone", "bone", "plant", "hide"],
            common_object_types=[
                "natural",
                "tool",
                "weapon",
                "container",
                "survival_gear",
            ],
            cultural_elements={
                "style": "natural",
                "craftsmanship": "primitive",
                "decoration": "carved",
            },
            style_descriptors=["organic", "rough", "primitive", "weathered", "natural"],
            forbidden_elements=["metal", "refined", "delicate", "manufactured"],
        )

        self._theme_cache["City"] = ObjectTheme(
            name="City",
            description="Refined objects reflecting urban sophistication",
            typical_materials=["steel", "brass", "silk", "marble", "glass"],
            common_object_types=["tool", "furniture", "art", "weapon", "luxury"],
            cultural_elements={
                "style": "refined",
                "craftsmanship": "professional",
                "decoration": "detailed",
            },
            style_descriptors=[
                "polished",
                "elegant",
                "sophisticated",
                "ornate",
                "quality",
            ],
            forbidden_elements=["crude", "primitive", "temporary", "makeshift"],
        )

        self._theme_cache["Dungeon"] = ObjectTheme(
            name="Dungeon",
            description="Ancient, mystical, or abandoned objects",
            typical_materials=["stone", "ancient_metal", "crystal", "bone", "unknown"],
            common_object_types=["treasure", "trap", "relic", "weapon", "mystery"],
            cultural_elements={
                "style": "ancient",
                "craftsmanship": "mysterious",
                "decoration": "runic",
            },
            style_descriptors=[
                "ancient",
                "mysterious",
                "magical",
                "ominous",
                "forgotten",
            ],
            forbidden_elements=["modern", "bright", "cheerful", "mundane"],
        )

    async def get_available_object_types(self, location_theme: str) -> list[str]:
        """Get object types suitable for the given location theme."""
        try:
            theme = self._theme_cache.get(location_theme)
            if not theme:
                logger.warning(f"Unknown location theme: {location_theme}")
                return ["container", "tool", "natural"]  # Default fallback

            # Get archetypes with good affinity for this theme
            suitable_types = []
            for archetype_name, archetype in self._archetype_cache.items():
                affinity = archetype.location_affinities.get(location_theme, 0.0)
                if affinity >= 0.5:  # Threshold for suitability
                    # Map archetype to object type
                    obj_type = archetype.typical_properties.object_type
                    if obj_type not in suitable_types:
                        suitable_types.append(obj_type)

            # Ensure we have at least some basic types
            if not suitable_types:
                suitable_types = theme.common_object_types[:3]

            return suitable_types

        except Exception as e:
            logger.error(
                f"Error getting available object types for {location_theme}: {e}"
            )
            return ["container", "tool", "natural"]

    async def determine_object_type(self, context: ObjectGenerationContext) -> str:
        """Determine most appropriate object type for generation context."""
        try:
            location_theme = context.location_theme.name
            purpose = context.generation_purpose

            # Get available types for this location
            available_types = await self.get_available_object_types(location_theme)

            # Purpose-based preferences
            purpose_preferences = {
                "populate_location": ["container", "tool", "furniture"],
                "quest_related": ["treasure", "key", "relic"],
                "random_encounter": ["weapon", "tool", "mystery"],
                "narrative_enhancement": ["art", "knowledge", "relic"],
                "environmental_storytelling": ["relic", "personal", "mystery"],
            }

            preferred = purpose_preferences.get(purpose, available_types)

            # Find intersection of preferred and available
            suitable = [t for t in preferred if t in available_types]

            if suitable:
                return suitable[0]  # Simple selection for now
            elif available_types:
                return available_types[0]
            else:
                return "container"  # Ultimate fallback

        except Exception as e:
            logger.error(f"Error determining object type: {e}")
            return "container"

    async def get_object_template(
        self, object_type: str, theme: str
    ) -> ObjectProperties:
        """Get base property template for object type and theme."""
        try:
            # Find archetype with matching object type and good theme affinity
            best_archetype = None
            best_affinity = 0.0

            for archetype in self._archetype_cache.values():
                if archetype.typical_properties.object_type == object_type:
                    affinity = archetype.location_affinities.get(theme, 0.0)
                    if affinity > best_affinity:
                        best_affinity = affinity
                        best_archetype = archetype

            if best_archetype:
                # Create a copy of the template properties
                template = ObjectProperties(
                    name=best_archetype.typical_properties.name,
                    object_type=best_archetype.typical_properties.object_type,
                    material=best_archetype.typical_properties.material,
                    size=best_archetype.typical_properties.size,
                    weight=best_archetype.typical_properties.weight,
                    durability=best_archetype.typical_properties.durability,
                    value=best_archetype.typical_properties.value,
                    special_properties=best_archetype.typical_properties.special_properties.copy(),
                    cultural_significance=best_archetype.typical_properties.cultural_significance,
                    description=best_archetype.typical_properties.description,
                )
                return template

            # Create generic template if no archetype found
            return ObjectProperties(
                name=f"{theme.lower()} {object_type}",
                object_type=object_type,
                material="unknown",
                description=f"A {object_type} suited for {theme} environments",
            )

        except Exception as e:
            logger.error(
                f"Error getting object template for {object_type}, {theme}: {e}"
            )
            return ObjectProperties(
                name=f"generic {object_type}", object_type=object_type
            )

    async def generate_cultural_variations(
        self, base_properties: ObjectProperties, location: Location
    ) -> ObjectProperties:
        """Apply cultural variations based on location context."""
        try:
            location_theme = location.state_flags.get("theme", "Unknown")
            theme = self._theme_cache.get(location_theme)

            if not theme:
                return base_properties  # No variations if theme unknown

            # Create modified copy
            varied_properties = ObjectProperties(
                name=base_properties.name,
                object_type=base_properties.object_type,
                material=base_properties.material,
                size=base_properties.size,
                weight=base_properties.weight,
                durability=base_properties.durability,
                value=base_properties.value,
                special_properties=base_properties.special_properties.copy(),
                cultural_significance=base_properties.cultural_significance,
                description=base_properties.description,
            )

            # Apply material variations
            if varied_properties.material in ["unknown", "generic"]:
                if theme.typical_materials:
                    varied_properties.material = theme.typical_materials[0]

            # Apply style descriptors
            if theme.style_descriptors:
                style_descriptor = theme.style_descriptors[0]  # Simple selection
                if style_descriptor not in varied_properties.special_properties:
                    varied_properties.special_properties.append(style_descriptor)

            # Apply cultural elements to description
            if theme.cultural_elements and varied_properties.description:
                style = theme.cultural_elements.get("style", "")
                if style:
                    varied_properties.description = (
                        f"{style.capitalize()} {varied_properties.description}"
                    )

            # Adjust value based on location type
            location_value_modifiers = {
                "Village": 0.8,  # Rural, simpler items
                "City": 1.2,  # Urban, refined items
                "Forest": 0.6,  # Natural, basic items
                "Dungeon": 1.5,  # Rare, ancient items
            }

            modifier = location_value_modifiers.get(location_theme, 1.0)
            varied_properties.value = int(varied_properties.value * modifier)

            return varied_properties

        except Exception as e:
            logger.error(f"Error generating cultural variations: {e}")
            return base_properties

    async def validate_object_consistency(
        self, generated_object: GeneratedObject, location: Location
    ) -> bool:
        """Validate that object fits thematically with location."""
        try:
            location_theme = location.state_flags.get("theme", "Unknown")
            theme = self._theme_cache.get(location_theme)

            if not theme:
                return True  # Can't validate unknown themes

            object_props = generated_object.properties

            # Check forbidden elements
            for forbidden in theme.forbidden_elements:
                if forbidden in object_props.special_properties:
                    logger.warning(
                        f"Object has forbidden element '{forbidden}' for theme {location_theme}"
                    )
                    return False
                if forbidden.lower() in object_props.description.lower():
                    logger.warning(
                        f"Object description contains forbidden element '{forbidden}'"
                    )
                    return False

            # Check material consistency
            if (
                object_props.material not in theme.typical_materials
                and object_props.material not in ["unknown", "mixed", "composite"]
            ):
                logger.info(
                    f"Object material '{object_props.material}' unusual for theme {location_theme}"
                )
                # This is a warning, not a failure

            # Check object type consistency
            if object_props.object_type not in theme.common_object_types:
                logger.info(
                    f"Object type '{object_props.object_type}' unusual for theme {location_theme}"
                )
                # This is also just informational

            return True

        except Exception as e:
            logger.error(f"Error validating object consistency: {e}")
            return True  # Default to valid on error

    def get_archetype_definition(self, archetype_name: str) -> ObjectArchetype | None:
        """Get archetype definition by name."""
        return self._archetype_cache.get(archetype_name)

    def get_theme_definition(self, theme_name: str) -> ObjectTheme | None:
        """Get theme definition by name."""
        return self._theme_cache.get(theme_name)

    async def get_archetype_for_object_type(
        self, object_type: str, location_theme: str
    ) -> str | None:
        """Get the best archetype name for a given object type and location theme."""
        try:
            best_archetype = None
            best_affinity = 0.0

            for archetype_name, archetype in self._archetype_cache.items():
                if archetype.typical_properties.object_type == object_type:
                    affinity = archetype.location_affinities.get(location_theme, 0.0)
                    if affinity > best_affinity:
                        best_affinity = affinity
                        best_archetype = archetype_name

            return best_archetype

        except Exception as e:
            logger.error(f"Error getting archetype for object type {object_type}: {e}")
            return None
