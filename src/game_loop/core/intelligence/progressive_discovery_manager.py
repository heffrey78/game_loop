"""
Progressive discovery manager for gradually revealing interaction capabilities.

This module manages the progressive revelation of interaction possibilities as
players learn and explore, providing contextual hints and achievement-style discoveries.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


class ProgressiveDiscoveryManager:
    """Manage progressive revelation of interaction capabilities."""

    def __init__(self) -> None:
        self.discovered_interactions: dict[str, set[str]] = (
            {}
        )  # player_id -> set of discovered interaction types
        self.hint_cooldowns: dict[str, datetime] = {}  # hint_type -> last_shown_time
        self.discovery_achievements: dict[str, list[dict[str, Any]]] = (
            {}
        )  # player_id -> list of achievements

        # Define discovery categories and their unlock conditions
        self.discovery_categories = {
            "object_modification": {
                "unlock_conditions": ["has_writing_tool", "examined_writable_object"],
                "description": "Object Writing & Modification",
                "hint_message": "💡 Hint: You can write on objects using 'write [text] on [object]'",
                "achievement_message": "🎉 Discovery: You can now modify objects by writing on them!",
            },
            "collection_examination": {
                "unlock_conditions": [
                    "in_library_location",
                    "attempted_collection_examine",
                ],
                "description": "Individual Item Examination",
                "hint_message": "💡 Hint: Try examining specific items rather than collections",
                "achievement_message": "🎉 Discovery: You can examine individual items for detailed information!",
            },
            "environmental_interaction": {
                "unlock_conditions": [
                    "location_has_interactive_elements",
                    "attempted_environmental_action",
                ],
                "description": "Environmental Manipulation",
                "hint_message": "💡 Hint: You can interact with environmental objects like climbing or pushing them",
                "achievement_message": "🎉 Discovery: You can interact with environmental objects!",
            },
            "advanced_navigation": {
                "unlock_conditions": [
                    "visited_multiple_locations",
                    "attempted_landmark_navigation",
                ],
                "description": "Landmark Navigation",
                "hint_message": "💡 Hint: You can navigate using 'go to [landmark]' for known locations",
                "achievement_message": "🎉 Discovery: You can navigate using landmark names!",
            },
            "npc_conversation": {
                "unlock_conditions": ["met_npc", "attempted_conversation"],
                "description": "Advanced Dialogue",
                "hint_message": "💡 Hint: Try asking NPCs about specific topics or locations",
                "achievement_message": "🎉 Discovery: You can have detailed conversations with NPCs!",
            },
            "tool_usage": {
                "unlock_conditions": [
                    "has_multiple_tools",
                    "attempted_tool_combination",
                ],
                "description": "Advanced Tool Usage",
                "hint_message": "💡 Hint: Tools can be used in combination for complex tasks",
                "achievement_message": "🎉 Discovery: You can combine tools for advanced interactions!",
            },
        }

        # Hint cooldown period (minutes)
        self.hint_cooldown_minutes = 5

    async def check_for_discovery_opportunities(
        self, command_result: Any, context: dict[str, Any]
    ) -> str | None:
        """Check if this is a good time to hint at new interactions."""
        try:
            player_id = context.get("player_id", "default_player")

            # Track successful interactions
            if hasattr(command_result, "success") and command_result.success:
                await self._record_successful_interaction(
                    player_id, command_result, context
                )

            # Check for hint opportunities
            hints = await self._generate_discovery_hints(player_id, context)

            if hints:
                return self._format_discovery_hints(hints)

            return None

        except Exception as e:
            logger.error(f"Error checking for discovery opportunities: {e}")
            return None

    async def _record_successful_interaction(
        self, player_id: str, command_result: Any, context: dict[str, Any]
    ) -> None:
        """Record successful interaction for discovery tracking."""
        try:
            if player_id not in self.discovered_interactions:
                self.discovered_interactions[player_id] = set()

            # Determine interaction type from command result
            if hasattr(command_result, "metadata"):
                metadata = command_result.metadata or {}
                interaction_type = metadata.get("interaction_type", "unknown")

                if interaction_type != "unknown":
                    self.discovered_interactions[player_id].add(interaction_type)

            # Also track based on command context
            command_type = context.get("command_type", "")
            if command_type:
                self.discovered_interactions[player_id].add(command_type.lower())

        except Exception as e:
            logger.error(f"Error recording successful interaction: {e}")

    async def _generate_discovery_hints(
        self, player_id: str, context: dict[str, Any]
    ) -> list[str]:
        """Generate hints about undiscovered interactions."""
        discovered = self.discovered_interactions.get(player_id, set())
        hints = []

        for category, category_info in self.discovery_categories.items():
            # Skip if already discovered
            if category in discovered:
                continue

            # Check if hint is on cooldown
            if self._is_hint_on_cooldown(category):
                continue

            # Check unlock conditions
            if await self._check_unlock_conditions(
                category_info["unlock_conditions"], context
            ):
                hints.append(category)
                # Set cooldown for this hint
                self.hint_cooldowns[category] = datetime.now()

        return hints

    def _is_hint_on_cooldown(self, hint_type: str) -> bool:
        """Check if a hint type is on cooldown."""
        if hint_type not in self.hint_cooldowns:
            return False

        last_shown = self.hint_cooldowns[hint_type]
        cooldown_period = timedelta(minutes=self.hint_cooldown_minutes)

        return datetime.now() - last_shown < cooldown_period

    async def _check_unlock_conditions(
        self, conditions: list[str], context: dict[str, Any]
    ) -> bool:
        """Check if unlock conditions are met."""
        conditions_met = 0

        for condition in conditions:
            if await self._evaluate_condition(condition, context):
                conditions_met += 1

        # Require at least half of conditions to be met
        return conditions_met >= len(conditions) / 2

    async def _evaluate_condition(
        self, condition: str, context: dict[str, Any]
    ) -> bool:
        """Evaluate a specific unlock condition."""
        try:
            if condition == "has_writing_tool":
                inventory = context.get("inventory_items", [])
                writing_tools = ["pen", "pencil", "marker", "quill", "chalk"]
                return any(tool in str(inventory).lower() for tool in writing_tools)

            elif condition == "examined_writable_object":
                # This would be tracked through previous interactions
                # For now, assume true if player has examined objects
                return len(context.get("location_objects", [])) > 0

            elif condition == "in_library_location":
                location_name = context.get("location_name", "").lower()
                location_desc = context.get("location_description", "").lower()
                library_keywords = [
                    "library",
                    "archive",
                    "study",
                    "reading room",
                    "bookshelf",
                ]
                return any(
                    keyword in location_name or keyword in location_desc
                    for keyword in library_keywords
                )

            elif condition == "attempted_collection_examine":
                # This would be tracked from failed command attempts
                # For now, return True if in a location with collections
                location_desc = context.get("location_description", "").lower()
                return any(
                    word in location_desc for word in ["books", "shelves", "documents"]
                )

            elif condition == "location_has_interactive_elements":
                location_desc = context.get("location_description", "").lower()
                interactive_keywords = [
                    "bookcase",
                    "shelf",
                    "door",
                    "window",
                    "lever",
                    "switch",
                    "stairs",
                ]
                return any(keyword in location_desc for keyword in interactive_keywords)

            elif condition == "attempted_environmental_action":
                # This would be tracked from failed command attempts
                # For now, assume true if location has interactive elements
                return await self._evaluate_condition(
                    "location_has_interactive_elements", context
                )

            elif condition == "visited_multiple_locations":
                # This would be tracked through navigation history
                # For now, assume true (would need proper tracking)
                return True

            elif condition == "attempted_landmark_navigation":
                # This would be tracked from failed navigation attempts
                # For now, assume true if player has moved around
                return True

            elif condition == "met_npc":
                npcs = context.get("npcs", [])
                return len(npcs) > 0

            elif condition == "attempted_conversation":
                # This would be tracked from conversation attempts
                # For now, assume true if NPCs are present
                return await self._evaluate_condition("met_npc", context)

            elif condition == "has_multiple_tools":
                inventory = context.get("inventory_items", [])
                return len(inventory) >= 2

            elif condition == "attempted_tool_combination":
                # This would be tracked from complex use attempts
                # For now, assume true if player has multiple tools
                return await self._evaluate_condition("has_multiple_tools", context)

            else:
                logger.warning(f"Unknown unlock condition: {condition}")
                return False

        except Exception as e:
            logger.error(f"Error evaluating condition {condition}: {e}")
            return False

    def _format_discovery_hints(self, hints: list[str]) -> str:
        """Format discovery hints as helpful suggestions."""
        if not hints:
            return ""

        formatted_hints = []
        for hint_category in hints[:2]:  # Show max 2 hints at once
            category_info = self.discovery_categories.get(hint_category, {})
            hint_message = category_info.get(
                "hint_message", f"New interaction available: {hint_category}"
            )
            formatted_hints.append(hint_message)

        return "\n".join(formatted_hints)

    async def unlock_discovery(self, player_id: str, discovery_type: str) -> str | None:
        """Manually unlock a discovery and return achievement message."""
        try:
            if player_id not in self.discovered_interactions:
                self.discovered_interactions[player_id] = set()

            if discovery_type not in self.discovered_interactions[player_id]:
                self.discovered_interactions[player_id].add(discovery_type)

                # Add to achievements
                if player_id not in self.discovery_achievements:
                    self.discovery_achievements[player_id] = []

                category_info = self.discovery_categories.get(discovery_type, {})
                achievement_msg = category_info.get(
                    "achievement_message",
                    f"🎉 Discovery: {discovery_type.replace('_', ' ').title()}!",
                )

                achievement_data = {
                    "type": discovery_type,
                    "message": achievement_msg,
                    "timestamp": datetime.now(),
                }
                self.discovery_achievements[player_id].append(achievement_data)

                return achievement_msg

            return None

        except Exception as e:
            logger.error(f"Error unlocking discovery: {e}")
            return None

    def get_player_discoveries(self, player_id: str) -> dict[str, Any]:
        """Get all discoveries for a player."""
        return {
            "discovered_interactions": list(
                self.discovered_interactions.get(player_id, set())
            ),
            "achievements": self.discovery_achievements.get(player_id, []),
            "total_discoveries": len(
                self.discovered_interactions.get(player_id, set())
            ),
            "available_discoveries": len(self.discovery_categories),
        }

    def get_discovery_progress(self, player_id: str) -> str:
        """Get formatted discovery progress for player."""
        discoveries = self.get_player_discoveries(player_id)
        total = discoveries["total_discoveries"]
        available = discoveries["available_discoveries"]

        progress_bar = "█" * total + "░" * (available - total)

        return (
            f"**Discovery Progress:** {total}/{available}\n"
            f"`{progress_bar}`\n"
            f"*Type 'discoveries' to see your achievements.*"
        )

    async def process_failed_command_for_discovery(
        self,
        failed_command: str,
        intent_analysis: dict[str, Any],
        context: dict[str, Any],
    ) -> str | None:
        """Process failed command to potentially unlock new discoveries."""
        try:
            player_id = context.get("player_id", "default_player")
            intent_type = intent_analysis.get("intent_type", "unknown")

            # Map failed command intents to potential discoveries
            discovery_mapping = {
                "object_interaction": "object_modification",
                "collection_examination": "collection_examination",
                "environmental_action": "environmental_interaction",
                "navigation": "advanced_navigation",
                "communication": "npc_conversation",
            }

            potential_discovery = discovery_mapping.get(intent_type)
            if potential_discovery:
                # Check if conditions are met for unlocking this discovery
                category_info = self.discovery_categories.get(potential_discovery, {})
                conditions = category_info.get("unlock_conditions", [])

                if await self._check_unlock_conditions(conditions, context):
                    return await self.unlock_discovery(player_id, potential_discovery)

            return None

        except Exception as e:
            logger.error(f"Error processing failed command for discovery: {e}")
            return None
