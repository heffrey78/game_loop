"""
Contextual suggestion engine for providing helpful suggestions based on context and failed commands.

This module generates intelligent suggestions based on current game state,
available objects, and player intent analysis.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ContextualSuggestionEngine:
    """Generate helpful suggestions based on context and failed commands."""

    def __init__(self, semantic_search_service: Any | None = None):
        self.semantic_search = semantic_search_service

        # Tool categories for suggestions
        self.tool_categories = {
            "writing": ["pen", "pencil", "marker", "quill", "chalk", "stylus"],
            "cutting": ["knife", "blade", "scissors", "cutter", "razor"],
            "opening": ["key", "keycard", "crowbar", "lockpick"],
            "lighting": ["flashlight", "torch", "lamp", "candle", "lighter"],
            "container": ["bag", "box", "chest", "pouch", "satchel"],
        }

    async def generate_suggestions(
        self,
        failed_command: str,
        intent_analysis: dict[str, Any],
        context: dict[str, Any],
    ) -> list[str]:
        """Generate helpful suggestions for failed commands."""
        try:
            suggestion_type = intent_analysis.get("suggestion_type", "general")

            if suggestion_type == "object_modification":
                return await self._suggest_object_modifications(
                    intent_analysis, context
                )
            elif suggestion_type == "collection_interaction":
                return await self._suggest_collection_interactions(
                    intent_analysis, context
                )
            elif suggestion_type == "environmental_interaction":
                return await self._suggest_environmental_actions(
                    intent_analysis, context
                )
            elif suggestion_type == "detailed_exploration":
                return await self._suggest_exploration_alternatives(
                    intent_analysis, context
                )
            elif suggestion_type == "communication_attempt":
                return await self._suggest_communication_alternatives(
                    intent_analysis, context
                )
            elif suggestion_type == "landmark_navigation":
                return await self._suggest_navigation_alternatives(
                    intent_analysis, context
                )
            else:
                return await self._suggest_general_alternatives(failed_command, context)

        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return [
                "Try 'help' for available commands",
                "Look around to see what's available",
            ]

    async def _suggest_object_modifications(
        self, intent_analysis: dict[str, Any], context: dict[str, Any]
    ) -> list[str]:
        """Suggest object modification alternatives."""
        suggestions = []
        targets = intent_analysis.get("targets", [])
        verb = intent_analysis.get("verb", "use")

        # Check if player has required tools
        inventory = context.get("inventory_items", [])
        writing_tools = self._find_tools_in_inventory("writing", inventory)

        if verb in ["write", "inscribe"] and not writing_tools:
            suggestions.append(
                "You need something to write with. Look for a pen, pencil, or marker."
            )
            suggestions.append(
                "Try examining objects around you - there might be writing implements."
            )
        elif writing_tools:
            tool_name = writing_tools[0]
            if targets:
                target = targets[0]
                suggestions.append(f"Try 'use {tool_name} on {target}' to write on it.")
            else:
                suggestions.append(f"Try 'use {tool_name}' to write with it.")

        # Check for similar objects in location
        if targets:
            target = targets[0]
            similar_objects = await self._find_similar_objects(target, context)
            if similar_objects:
                suggestions.append(
                    f"Maybe try '{verb} on {similar_objects[0]}' instead?"
                )

        # Suggest examining objects first
        suggestions.append(
            "Try examining objects more closely to see what you can do with them."
        )

        return suggestions[:4]  # Return top 4 suggestions

    async def _suggest_collection_interactions(
        self, intent_analysis: dict[str, Any], context: dict[str, Any]
    ) -> list[str]:
        """Suggest alternatives for examining collections."""
        suggestions = []
        targets = intent_analysis.get("targets", [])

        if targets:
            target = targets[0]

            # Convert plural to singular for individual item suggestions
            singular_target = self._pluralize_to_singular(target)

            # Look for individual items of this type
            individual_items = await self._find_individual_items(target, context)
            if individual_items:
                suggestions.extend(
                    [
                        f"Try examining individual items: {', '.join(individual_items[:3])}",
                        f"Use 'examine {individual_items[0]}' for specific details.",
                    ]
                )
            else:
                # Suggest the singular form
                suggestions.append(
                    f"Try 'examine {singular_target}' for a specific item."
                )

            # Suggest more general examination
            suggestions.append(
                f"Use 'look around' to see what specific {target} are available."
            )
            suggestions.append(f"Try 'search {target}' to look through them.")

        return suggestions[:4]

    async def _suggest_environmental_actions(
        self, intent_analysis: dict[str, Any], context: dict[str, Any]
    ) -> list[str]:
        """Suggest environmental interaction alternatives."""
        suggestions = []
        targets = intent_analysis.get("targets", [])
        verb = intent_analysis.get("verb", "interact")

        if targets:
            target = targets[0]

            # Check if target is mentioned in location description
            location_description = context.get("location_description", "")
            if target.lower() in location_description.lower():
                suggestions.append(
                    f"The {target} is in this area. Try examining it first."
                )
                suggestions.append(
                    f"Use 'examine {target}' before trying to {verb} it."
                )
            else:
                suggestions.append(
                    f"I don't see '{target}' here. Try 'look around' to see what's available."
                )

            # Suggest tool requirements for certain actions
            if verb in ["open", "unlock"]:
                opening_tools = self._find_tools_in_inventory(
                    "opening", context.get("inventory_items", [])
                )
                if opening_tools:
                    suggestions.append(
                        f"Try using your {opening_tools[0]} to {verb} it."
                    )
                else:
                    suggestions.append(
                        f"You might need a key or tool to {verb} the {target}."
                    )

        # General environmental suggestions
        suggestions.append("Look around carefully - you might have missed something.")
        suggestions.append(
            "Try examining objects mentioned in the location description."
        )

        return suggestions[:4]

    async def _suggest_exploration_alternatives(
        self, intent_analysis: dict[str, Any], context: dict[str, Any]
    ) -> list[str]:
        """Suggest exploration alternatives."""
        suggestions = [
            "Try 'look around' to examine your surroundings in detail.",
            "Use 'examine [object]' to look at specific items you see.",
        ]

        # Add inventory suggestion if player has items
        inventory_items = context.get("inventory_items", [])
        if inventory_items:
            suggestions.append(
                "Check your 'inventory' to see what tools you have available."
            )

        # Add location-specific suggestions
        location_objects = context.get("location_objects", [])
        if location_objects:
            suggestions.append(f"You might examine: {', '.join(location_objects[:3])}")

        available_exits = context.get("available_exits", [])
        if available_exits:
            suggestions.append(f"You could explore: {', '.join(available_exits)}")

        return suggestions[:4]

    async def _suggest_communication_alternatives(
        self, intent_analysis: dict[str, Any], context: dict[str, Any]
    ) -> list[str]:
        """Suggest communication alternatives."""
        suggestions = []
        targets = intent_analysis.get("targets", [])

        if targets:
            target = targets[0]

            # Check if target is an NPC in the area
            npcs = context.get("npcs", [])
            npc_names = [
                npc.get("name", "").lower() for npc in npcs if isinstance(npc, dict)
            ]

            if any(target.lower() in name for name in npc_names):
                suggestions.extend(
                    [
                        f"Try 'talk to {target}' to start a conversation.",
                        f"Use 'greet {target}' for a simple greeting.",
                    ]
                )
            else:
                suggestions.append(f"I don't see '{target}' here to talk to.")
                if npc_names:
                    available_npcs = [name.title() for name in npc_names if name]
                    suggestions.append(
                        f"Available people to talk to: {', '.join(available_npcs[:3])}"
                    )

        suggestions.append("Look around to see who else is here.")
        suggestions.append("Some characters might be in other locations.")

        return suggestions[:4]

    async def _suggest_navigation_alternatives(
        self, intent_analysis: dict[str, Any], context: dict[str, Any]
    ) -> list[str]:
        """Suggest navigation alternatives."""
        suggestions = []
        targets = intent_analysis.get("targets", [])

        if targets:
            target = targets[0]
            suggestions.extend(
                [
                    f"Try 'go {target}' if it's a direction.",
                    f"Use 'find {target}' to search for that location.",
                    f"Ask an NPC about how to reach {target}.",
                ]
            )

        # Add general navigation help
        available_exits = context.get("available_exits", [])
        if available_exits:
            suggestions.append(f"Available directions: {', '.join(available_exits)}")

        suggestions.append("Try 'help navigation' for movement commands.")

        return suggestions[:4]

    async def _suggest_general_alternatives(
        self, failed_command: str, context: dict[str, Any]
    ) -> list[str]:
        """Suggest general alternatives for unrecognized commands."""
        suggestions = [
            "Try 'help' to see available commands.",
            "Use 'look around' to examine your surroundings.",
        ]

        # Extract potential objects from failed command for suggestions
        words = failed_command.lower().split()
        meaningful_words = [
            w for w in words if len(w) > 2 and w not in ["the", "a", "an"]
        ]

        if meaningful_words:
            potential_object = meaningful_words[-1]  # Often the last word is the object
            suggestions.append(f"Try 'examine {potential_object}' if you can see it.")

        suggestions.append("Check your 'inventory' to see what you're carrying.")

        return suggestions

    def _find_tools_in_inventory(
        self, tool_category: str, inventory: list[str]
    ) -> list[str]:
        """Find tools of a specific category in the inventory."""
        tools = self.tool_categories.get(tool_category, [])
        found_tools = []

        for item in inventory:
            item_lower = item.lower()
            for tool in tools:
                if tool in item_lower:
                    found_tools.append(item)
                    break

        return found_tools

    async def _find_similar_objects(
        self, target: str, context: dict[str, Any]
    ) -> list[str]:
        """Find objects similar to the target in the current location."""
        location_objects = context.get("location_objects", [])

        # Simple similarity check based on common words
        target_words = set(target.lower().split())
        similar_objects = []

        for obj in location_objects:
            obj_words = set(obj.lower().split())
            # Check for word overlap
            if target_words & obj_words:  # If there's any overlap
                similar_objects.append(obj)

        return similar_objects[:3]  # Return top 3 similar objects

    async def _find_individual_items(
        self, collection_name: str, context: dict[str, Any]
    ) -> list[str]:
        """Find individual items that might be part of a collection."""
        location_objects = context.get("location_objects", [])

        # Convert collection name to potential individual items
        singular = self._pluralize_to_singular(collection_name)

        individual_items = []
        for obj in location_objects:
            obj_lower = obj.lower()
            # Check if object name contains the singular form
            if singular in obj_lower or collection_name.rstrip("s") in obj_lower:
                individual_items.append(obj)

        return individual_items[:5]  # Return up to 5 individual items

    def _pluralize_to_singular(self, word: str) -> str:
        """Convert plural word to singular form (simple heuristic)."""
        if word.endswith("ies"):
            return word[:-3] + "y"
        elif word.endswith("es"):
            return word[:-2]
        elif word.endswith("s") and not word.endswith("ss"):
            return word[:-1]
        else:
            return word

    def get_contextual_hints(self, context: dict[str, Any]) -> list[str]:
        """Get contextual hints based on current game state."""
        hints = []

        # Inventory-based hints
        inventory = context.get("inventory_items", [])
        if inventory:
            writing_tools = self._find_tools_in_inventory("writing", inventory)
            if writing_tools:
                hints.append(
                    f"💡 You have {writing_tools[0]} - you can write on objects!"
                )

            opening_tools = self._find_tools_in_inventory("opening", inventory)
            if opening_tools:
                hints.append(f"💡 Your {opening_tools[0]} might open locked things!")

        # Location-based hints
        location_description = context.get("location_description", "")
        if "book" in location_description.lower():
            hints.append("💡 Try examining individual books rather than 'the books'")

        if any(
            word in location_description.lower() for word in ["door", "chest", "box"]
        ):
            hints.append("💡 Some objects might be openable or interactive")

        return hints[:3]  # Return top 3 hints
