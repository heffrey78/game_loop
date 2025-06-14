"""
Object Placement Manager for intelligent object placement within locations.

This module handles spatial placement logic, density management, and thematic
consistency for object placement in game locations.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from game_loop.core.models.object_models import GeneratedObject, ObjectPlacement
from game_loop.database.session_factory import DatabaseSessionFactory
from game_loop.state.models import Location, WorldState

logger = logging.getLogger(__name__)


class ObjectPlacementManager:
    """Manages intelligent object placement with spatial and thematic considerations."""

    def __init__(
        self, world_state: WorldState, session_factory: DatabaseSessionFactory
    ):
        self.world_state = world_state
        self.session_factory = session_factory
        self._placement_rules = self._initialize_placement_rules()

    def _initialize_placement_rules(self) -> dict[str, Any]:
        """Initialize placement rules for different object types and locations."""
        return {
            "size_constraints": {
                "tiny": {
                    "max_per_location": 20,
                    "placement_types": ["shelf", "table", "floor", "container"],
                },
                "small": {
                    "max_per_location": 15,
                    "placement_types": ["shelf", "table", "floor"],
                },
                "medium": {
                    "max_per_location": 8,
                    "placement_types": ["floor", "table"],
                },
                "large": {"max_per_location": 4, "placement_types": ["floor"]},
                "huge": {
                    "max_per_location": 2,
                    "placement_types": ["floor", "embedded"],
                },
            },
            "theme_placements": {
                "Village": {
                    "preferred_types": ["table", "shelf", "floor"],
                    "avoid_types": ["embedded", "hidden"],
                    "typical_visibility": "visible",
                },
                "Forest": {
                    "preferred_types": ["floor", "embedded", "hidden"],
                    "avoid_types": ["table", "shelf"],
                    "typical_visibility": "partially_hidden",
                },
                "City": {
                    "preferred_types": ["shelf", "table", "container"],
                    "avoid_types": ["floor", "embedded"],
                    "typical_visibility": "visible",
                },
                "Dungeon": {
                    "preferred_types": ["floor", "hidden", "embedded"],
                    "avoid_types": ["shelf", "table"],
                    "typical_visibility": "hidden",
                },
            },
            "object_type_preferences": {
                "weapon": {
                    "placement_types": ["wall", "table", "floor"],
                    "visibility": "visible",
                },
                "tool": {
                    "placement_types": ["table", "shelf", "floor"],
                    "visibility": "visible",
                },
                "container": {
                    "placement_types": ["floor", "table"],
                    "visibility": "visible",
                },
                "treasure": {
                    "placement_types": ["hidden", "container", "floor"],
                    "visibility": "hidden",
                },
                "book": {
                    "placement_types": ["shelf", "table"],
                    "visibility": "visible",
                },
                "natural": {
                    "placement_types": ["floor", "embedded"],
                    "visibility": "partially_hidden",
                },
                "furniture": {"placement_types": ["floor"], "visibility": "visible"},
            },
        }

    async def determine_placement(
        self, generated_object: GeneratedObject, location: Location
    ) -> ObjectPlacement:
        """Determine optimal placement for object in location."""
        try:
            logger.debug(
                f"Determining placement for {generated_object.properties.name} in {location.name}"
            )

            # Get placement constraints
            size_constraints = self._get_size_constraints(
                generated_object.properties.size
            )
            theme_rules = self._get_theme_rules(location)
            object_type_rules = self._get_object_type_rules(
                generated_object.properties.object_type
            )

            # Determine placement type
            placement_type = self._select_placement_type(
                size_constraints, theme_rules, object_type_rules
            )

            # Determine visibility
            visibility = self._determine_visibility(
                generated_object, location, theme_rules, object_type_rules
            )

            # Determine accessibility
            accessibility = self._determine_accessibility(
                generated_object, placement_type
            )

            # Generate spatial description
            spatial_description = self._generate_spatial_description(
                generated_object, location, placement_type
            )

            # Calculate discovery difficulty
            discovery_difficulty = self._calculate_discovery_difficulty(
                visibility, placement_type, generated_object.properties.object_type
            )

            # Create placement metadata
            placement_metadata = {
                "placement_rules_used": {
                    "size_constraint": size_constraints,
                    "theme_preference": theme_rules,
                    "object_type_rule": object_type_rules,
                },
                "location_theme": location.state_flags.get("theme", "Unknown"),
                "object_size": generated_object.properties.size,
                "placement_timestamp": "now",
            }

            placement = ObjectPlacement(
                object_id=generated_object.base_object.object_id,
                location_id=location.location_id,
                placement_type=placement_type,
                visibility=visibility,
                accessibility=accessibility,
                spatial_description=spatial_description,
                discovery_difficulty=discovery_difficulty,
                placement_metadata=placement_metadata,
            )

            logger.info(
                f"Placed {generated_object.properties.name} as {placement_type} ({visibility})"
            )
            return placement

        except Exception as e:
            logger.error(f"Error determining placement: {e}")
            # Create fallback placement
            return ObjectPlacement(
                object_id=generated_object.base_object.object_id,
                location_id=location.location_id,
                placement_type="floor",
                visibility="visible",
                accessibility="accessible",
                spatial_description=f"The {generated_object.properties.name} rests on the ground.",
                discovery_difficulty=1,
                placement_metadata={"error": str(e), "fallback": True},
            )

    async def validate_placement(
        self, placement: ObjectPlacement, location: Location
    ) -> bool:
        """Validate that placement is spatially and thematically appropriate."""
        try:
            # Check basic placement type validity
            valid_types = [
                "floor",
                "table",
                "shelf",
                "wall",
                "ceiling",
                "hidden",
                "embedded",
                "container",
            ]
            if placement.placement_type not in valid_types:
                logger.warning(f"Invalid placement type: {placement.placement_type}")
                return False

            # Check discovery difficulty range
            if not (1 <= placement.discovery_difficulty <= 10):
                logger.warning(
                    f"Invalid discovery difficulty: {placement.discovery_difficulty}"
                )
                return False

            # Check theme consistency
            theme = location.state_flags.get("theme", "Unknown")
            theme_rules = self._placement_rules.get("theme_placements", {}).get(
                theme, {}
            )

            avoid_types = theme_rules.get("avoid_types", [])
            if placement.placement_type in avoid_types:
                logger.warning(
                    f"Placement type {placement.placement_type} not suitable for {theme} theme"
                )
                return False

            # Check density limits (simplified)
            density_info = await self.check_placement_density(location)
            if density_info.get("overcrowded", False):
                logger.warning(
                    f"Location {location.name} is overcrowded for new objects"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating placement: {e}")
            return False

    async def check_placement_density(self, location: Location) -> dict[str, Any]:
        """Check if location can accommodate additional objects."""
        try:
            current_count = len(location.objects)

            # Base limits by location theme
            theme = location.state_flags.get("theme", "Unknown")
            theme_limits = {
                "Village": 12,
                "Forest": 8,
                "City": 15,
                "Dungeon": 10,
                "Cave": 6,
            }

            max_objects = theme_limits.get(theme, 10)

            # Calculate density levels
            density_ratio = current_count / max_objects

            density_info = {
                "current_count": current_count,
                "max_recommended": max_objects,
                "density_ratio": density_ratio,
                "overcrowded": density_ratio > 1.0,
                "near_capacity": density_ratio > 0.8,
                "sparse": density_ratio < 0.3,
                "can_accommodate": density_ratio < 0.9,
            }

            # Analyze by object type
            type_counts = {}
            for obj in location.objects.values():
                obj_type = getattr(obj, "object_type", "unknown")
                type_counts[obj_type] = type_counts.get(obj_type, 0) + 1

            density_info["type_distribution"] = type_counts

            return density_info

        except Exception as e:
            logger.error(f"Error checking placement density: {e}")
            return {
                "current_count": 0,
                "max_recommended": 10,
                "density_ratio": 0.0,
                "overcrowded": False,
                "can_accommodate": True,
            }

    async def update_location_objects(
        self, location_id: UUID, placement: ObjectPlacement
    ) -> bool:
        """Update location with new object placement."""
        try:
            location = self.world_state.locations.get(location_id)
            if not location:
                logger.error(f"Location {location_id} not found")
                return False

            # This would typically update the database and location state
            # For now, we'll just log the update
            logger.info(
                f"Updated location {location.name} with object placement {placement.object_id}"
            )

            # In a full implementation, this would:
            # 1. Update the database with placement info
            # 2. Update the location's objects dict
            # 3. Update any spatial indexes
            # 4. Notify other systems of the change

            return True

        except Exception as e:
            logger.error(f"Error updating location objects: {e}")
            return False

    def _get_size_constraints(self, size: str) -> dict[str, Any]:
        """Get placement constraints based on object size."""
        return self._placement_rules["size_constraints"].get(
            size, self._placement_rules["size_constraints"]["medium"]
        )

    def _get_theme_rules(self, location: Location) -> dict[str, Any]:
        """Get placement rules based on location theme."""
        theme = location.state_flags.get("theme", "Unknown")
        return self._placement_rules["theme_placements"].get(
            theme,
            {
                "preferred_types": ["floor"],
                "avoid_types": [],
                "typical_visibility": "visible",
            },
        )

    def _get_object_type_rules(self, object_type: str) -> dict[str, Any]:
        """Get placement rules based on object type."""
        return self._placement_rules["object_type_preferences"].get(
            object_type, {"placement_types": ["floor"], "visibility": "visible"}
        )

    def _select_placement_type(
        self, size_constraints: dict, theme_rules: dict, object_type_rules: dict
    ) -> str:
        """Select the best placement type based on all constraints."""
        try:
            # Get allowed types from size constraints
            size_allowed = set(size_constraints.get("placement_types", ["floor"]))

            # Get preferred types from theme
            theme_preferred = set(theme_rules.get("preferred_types", ["floor"]))
            theme_avoided = set(theme_rules.get("avoid_types", []))

            # Get object type preferences
            object_preferred = set(object_type_rules.get("placement_types", ["floor"]))

            # Find intersection of allowed and preferred
            candidates = size_allowed & theme_preferred & object_preferred

            # Remove avoided types
            candidates = candidates - theme_avoided

            if candidates:
                # Simple selection - could be more sophisticated
                return list(candidates)[0]

            # Fallback to size constraints
            if size_allowed - theme_avoided:
                return list(size_allowed - theme_avoided)[0]

            # Ultimate fallback
            return "floor"

        except Exception as e:
            logger.error(f"Error selecting placement type: {e}")
            return "floor"

    def _determine_visibility(
        self,
        generated_object: GeneratedObject,
        location: Location,
        theme_rules: dict,
        object_type_rules: dict,
    ) -> str:
        """Determine object visibility based on context."""
        try:
            # Default from theme
            theme_visibility = theme_rules.get("typical_visibility", "visible")

            # Object type preference
            object_visibility = object_type_rules.get("visibility", "visible")

            # Special cases
            if generated_object.properties.object_type == "treasure":
                if generated_object.properties.value > 100:
                    return "hidden"
                else:
                    return "partially_hidden"

            if "hidden" in generated_object.properties.special_properties:
                return "hidden"

            if "conspicuous" in generated_object.properties.special_properties:
                return "visible"

            # Combine preferences
            if theme_visibility == "hidden" or object_visibility == "hidden":
                return "hidden"
            elif (
                theme_visibility == "partially_hidden"
                or object_visibility == "partially_hidden"
            ):
                return "partially_hidden"
            else:
                return "visible"

        except Exception as e:
            logger.error(f"Error determining visibility: {e}")
            return "visible"

    def _determine_accessibility(
        self, generated_object: GeneratedObject, placement_type: str
    ) -> str:
        """Determine object accessibility based on placement."""
        try:
            # Embedded or ceiling objects are typically harder to access
            if placement_type in ["embedded", "ceiling"]:
                return "requires_tool"

            # Hidden objects might be blocked
            if placement_type == "hidden":
                return "blocked"

            # Large objects might be hard to move
            if generated_object.properties.size in ["large", "huge"]:
                return "blocked"

            # Most objects are accessible
            return "accessible"

        except Exception as e:
            logger.error(f"Error determining accessibility: {e}")
            return "accessible"

    def _generate_spatial_description(
        self, generated_object: GeneratedObject, location: Location, placement_type: str
    ) -> str:
        """Generate descriptive text for object placement."""
        try:
            object_name = generated_object.properties.name

            descriptions = {
                "floor": f"The {object_name} rests on the ground",
                "table": f"The {object_name} sits on a table",
                "shelf": f"The {object_name} is placed on a shelf",
                "wall": f"The {object_name} hangs on the wall",
                "ceiling": f"The {object_name} is suspended from the ceiling",
                "hidden": f"The {object_name} is concealed from casual view",
                "embedded": f"The {object_name} is embedded in the surroundings",
                "container": f"The {object_name} is stored in a container",
            }

            base_description = descriptions.get(
                placement_type, f"The {object_name} is here"
            )

            # Add location context
            theme = location.state_flags.get("theme", "Unknown")
            if theme == "Forest":
                if placement_type == "floor":
                    base_description += " among the fallen leaves"
                elif placement_type == "embedded":
                    base_description += " within the bark of an ancient tree"
            elif theme == "Dungeon":
                if placement_type == "hidden":
                    base_description += " in a shadowy alcove"
                elif placement_type == "floor":
                    base_description += " on the cold stone floor"
            elif theme == "Village":
                if placement_type == "table":
                    base_description += " in someone's humble home"

            return base_description + "."

        except Exception as e:
            logger.error(f"Error generating spatial description: {e}")
            return f"The {generated_object.properties.name} is located here."

    def _calculate_discovery_difficulty(
        self, visibility: str, placement_type: str, object_type: str
    ) -> int:
        """Calculate how difficult the object is to discover (1-10 scale)."""
        try:
            base_difficulty = 1

            # Visibility modifier
            visibility_modifiers = {"visible": 0, "partially_hidden": 2, "hidden": 4}
            base_difficulty += visibility_modifiers.get(visibility, 0)

            # Placement modifier
            placement_modifiers = {
                "floor": 0,
                "table": 0,
                "shelf": 1,
                "wall": 1,
                "ceiling": 3,
                "hidden": 3,
                "embedded": 4,
                "container": 2,
            }
            base_difficulty += placement_modifiers.get(placement_type, 0)

            # Object type modifier
            if object_type == "treasure":
                base_difficulty += 2
            elif object_type == "trap":
                base_difficulty += 3
            elif object_type == "mystery":
                base_difficulty += 2

            # Ensure it's within valid range
            return max(1, min(10, base_difficulty))

        except Exception as e:
            logger.error(f"Error calculating discovery difficulty: {e}")
            return 1
