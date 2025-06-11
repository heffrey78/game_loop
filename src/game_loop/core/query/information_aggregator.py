"""Information aggregator for gathering data from multiple game systems."""

from typing import Any

from game_loop.search.semantic_search import SemanticSearchService
from game_loop.state.manager import GameStateManager


class InformationAggregator:
    """Aggregates information from multiple game systems for query processing."""

    def __init__(
        self,
        semantic_search: SemanticSearchService,
        game_state_manager: GameStateManager,
    ):
        self.semantic_search = semantic_search
        self.game_state_manager = game_state_manager

    async def gather_world_information(
        self, query: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Gather information about the game world."""
        world_info = {}

        try:
            # Search for world-related entities
            world_results = await self.semantic_search.search_entities(
                query, entity_types=["location", "world", "lore"], limit=5
            )

            # Aggregate world information
            locations = []
            lore_entries = []
            
            for result in world_results:
                entity_type = result.get("entity_type", "")
                if entity_type == "location":
                    locations.append({
                        "name": result.get("name", ""),
                        "description": result.get("description", ""),
                        "metadata": result.get("metadata", {}),
                    })
                elif entity_type in ["world", "lore"]:
                    lore_entries.append({
                        "name": result.get("name", ""),
                        "description": result.get("description", ""),
                        "metadata": result.get("metadata", {}),
                    })

            if locations:
                world_info["locations"] = self._format_location_info(locations)
            
            if lore_entries:
                world_info["lore"] = self._format_lore_info(lore_entries)

        except Exception as e:
            world_info["error"] = f"Could not gather world information: {str(e)}"

        return world_info

    async def gather_object_information(
        self, object_query: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Gather information about specific objects."""
        object_info = {}

        try:
            # Search for object-related entities
            object_results = await self.semantic_search.search_entities(
                object_query, entity_types=["object", "item"], limit=5
            )

            # Aggregate object information
            objects = []
            for result in object_results:
                objects.append({
                    "name": result.get("name", ""),
                    "description": result.get("description", ""),
                    "type": result.get("entity_type", ""),
                    "properties": result.get("metadata", {}),
                })

            if objects:
                object_info["objects"] = self._format_object_info(objects)
                
                # Add specific object details if we found a close match
                best_match = max(objects, key=lambda x: x.get("similarity", 0))
                if best_match:
                    object_info["primary_object"] = best_match

        except Exception as e:
            object_info["error"] = f"Could not gather object information: {str(e)}"

        return object_info

    async def gather_npc_information(
        self, npc_query: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Gather information about NPCs."""
        npc_info = {}

        try:
            # Search for NPC-related entities
            npc_results = await self.semantic_search.search_entities(
                npc_query, entity_types=["npc", "character"], limit=5
            )

            # Aggregate NPC information
            npcs = []
            for result in npc_results:
                npcs.append({
                    "name": result.get("name", ""),
                    "description": result.get("description", ""),
                    "type": result.get("entity_type", ""),
                    "background": result.get("metadata", {}).get("background", ""),
                    "role": result.get("metadata", {}).get("role", ""),
                })

            if npcs:
                npc_info["npcs"] = self._format_npc_info(npcs)
                
                # Add primary NPC if we found a good match
                best_match = max(npcs, key=lambda x: x.get("similarity", 0))
                if best_match:
                    npc_info["primary_npc"] = best_match

        except Exception as e:
            npc_info["error"] = f"Could not gather NPC information: {str(e)}"

        return npc_info

    async def gather_location_information(
        self, location_query: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Gather information about locations."""
        location_info = {}

        try:
            # Get current location if available
            current_location_id = context.get("current_location_id")
            if current_location_id:
                try:
                    location_state = await self.game_state_manager.get_location_state(
                        current_location_id
                    )
                    if location_state:
                        location_info["current_location"] = {
                            "id": current_location_id,
                            "description": getattr(location_state, "description", ""),
                            "details": location_state.to_dict(),
                        }
                except Exception:
                    pass

            # Search for location entities
            location_results = await self.semantic_search.search_entities(
                location_query, entity_types=["location"], limit=5
            )

            if location_results:
                locations = []
                for result in location_results:
                    locations.append({
                        "name": result.get("name", ""),
                        "description": result.get("description", ""),
                        "connections": result.get("metadata", {}).get("connections", []),
                        "features": result.get("metadata", {}).get("features", []),
                    })
                
                location_info["related_locations"] = self._format_location_info(locations)

        except Exception as e:
            location_info["error"] = f"Could not gather location information: {str(e)}"

        return location_info

    def _format_location_info(self, locations: list[dict[str, Any]]) -> str:
        """Format location information for display."""
        if not locations:
            return "No location information available."

        formatted = []
        for location in locations[:3]:  # Limit to top 3
            name = location.get("name", "Unknown Location")
            description = location.get("description", "No description available.")
            formatted.append(f"**{name}**: {description}")

        return "\n".join(formatted)

    def _format_object_info(self, objects: list[dict[str, Any]]) -> str:
        """Format object information for display."""
        if not objects:
            return "No object information available."

        formatted = []
        for obj in objects[:3]:  # Limit to top 3
            name = obj.get("name", "Unknown Object")
            description = obj.get("description", "No description available.")
            obj_type = obj.get("type", "")
            
            info = f"**{name}**"
            if obj_type:
                info += f" ({obj_type})"
            info += f": {description}"
            
            formatted.append(info)

        return "\n".join(formatted)

    def _format_npc_info(self, npcs: list[dict[str, Any]]) -> str:
        """Format NPC information for display."""
        if not npcs:
            return "No NPC information available."

        formatted = []
        for npc in npcs[:3]:  # Limit to top 3
            name = npc.get("name", "Unknown Character")
            description = npc.get("description", "No description available.")
            role = npc.get("role", "")
            
            info = f"**{name}**"
            if role:
                info += f" ({role})"
            info += f": {description}"
            
            formatted.append(info)

        return "\n".join(formatted)

    def _format_lore_info(self, lore_entries: list[dict[str, Any]]) -> str:
        """Format lore information for display."""
        if not lore_entries:
            return "No lore information available."

        formatted = []
        for entry in lore_entries[:3]:  # Limit to top 3
            name = entry.get("name", "Unknown")
            description = entry.get("description", "No information available.")
            formatted.append(f"**{name}**: {description}")

        return "\n".join(formatted)