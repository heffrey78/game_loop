"""
Enhanced command processor with intelligent error handling and progressive discovery.

This module integrates all command intelligence components to provide smart error responses,
contextual suggestions, and progressive discovery of game mechanics.
"""

import logging
from typing import Any

from game_loop.state.models import ActionResult

from .command_intent_analyzer import CommandIntentAnalyzer
from .contextual_suggestion_engine import ContextualSuggestionEngine
from .progressive_discovery_manager import ProgressiveDiscoveryManager
from .smart_error_response_generator import SmartErrorResponseGenerator

logger = logging.getLogger(__name__)


class EnhancedCommandProcessor:
    """Enhanced command processor with intelligent error handling and suggestions."""

    def __init__(
        self, existing_processor: Any, semantic_search_service: Any = None
    ) -> None:
        self.existing_processor = existing_processor

        # Initialize intelligence components
        self.intent_analyzer = CommandIntentAnalyzer()
        self.suggestion_engine = ContextualSuggestionEngine(semantic_search_service)
        self.discovery_manager = ProgressiveDiscoveryManager()
        self.error_response_generator = SmartErrorResponseGenerator(
            self.intent_analyzer, self.suggestion_engine
        )

        # Track failed attempts for progressive help
        self.failed_attempts: dict[str, list[str]] = (
            {}
        )  # player_id -> list of failed commands
        self.max_failed_attempts = 10

    async def process_command(
        self, command_text: str, context: dict[str, Any]
    ) -> ActionResult:
        """Process command with intelligent error handling and suggestions."""
        try:
            player_id = context.get("player_id", "default_player")

            # Try existing command processing first
            result = await self.existing_processor.process_command(
                command_text, context
            )

            # If command succeeded, check for discovery opportunities
            if result.success:
                discovery_message = (
                    await self.discovery_manager.check_for_discovery_opportunities(
                        result, context
                    )
                )
                if discovery_message:
                    # Append discovery message to feedback
                    result.feedback_message += f"\n\n{discovery_message}"

                # Clear failed attempts on success
                if player_id in self.failed_attempts:
                    del self.failed_attempts[player_id]

            else:
                # Command failed - provide intelligent error response
                result = await self._handle_command_failure(
                    command_text, result, context
                )

            return result

        except Exception as e:
            logger.error(f"Error in enhanced command processing: {e}")
            return ActionResult(
                success=False,
                feedback_message=f"An error occurred while processing your command: {str(e)}",
            )

    async def _handle_command_failure(
        self, command_text: str, original_result: ActionResult, context: dict[str, Any]
    ) -> ActionResult:
        """Handle command failure with intelligent error response."""
        try:
            player_id = context.get("player_id", "default_player")

            # Track failed attempt
            if player_id not in self.failed_attempts:
                self.failed_attempts[player_id] = []

            self.failed_attempts[player_id].append(command_text)

            # Keep only recent failed attempts
            if len(self.failed_attempts[player_id]) > self.max_failed_attempts:
                self.failed_attempts[player_id] = self.failed_attempts[player_id][
                    -self.max_failed_attempts :
                ]

            # Check if this qualifies for smart error handling
            if self._should_provide_smart_response(original_result):
                smart_response = await self._generate_smart_response(
                    command_text, original_result, context
                )

                # Create new result with smart response
                return ActionResult(
                    success=False,
                    feedback_message=smart_response,
                    metadata=original_result.metadata,
                )

            # Check for progressive help after multiple failures
            elif len(self.failed_attempts[player_id]) >= 3:
                progressive_help = (
                    await self.error_response_generator.generate_progressive_help(
                        self.failed_attempts[player_id], context
                    )
                )

                if progressive_help:
                    enhanced_message = (
                        f"{original_result.feedback_message}{progressive_help}"
                    )
                    return ActionResult(
                        success=False,
                        feedback_message=enhanced_message,
                        metadata=original_result.metadata,
                    )

            return original_result

        except Exception as e:
            logger.error(f"Error handling command failure: {e}")
            return original_result

    def _should_provide_smart_response(self, result: ActionResult) -> bool:
        """Determine if we should provide a smart error response."""
        # Provide smart response for generic error messages
        generic_errors = [
            "don't understand",
            "don't see any",
            "cannot go",
            "not found",
            "invalid command",
            "unknown command",
        ]

        message_lower = result.feedback_message.lower()
        return any(error in message_lower for error in generic_errors)

    async def _generate_smart_response(
        self, command_text: str, original_result: ActionResult, context: dict[str, Any]
    ) -> str:
        """Generate smart error response with discovery opportunities."""
        try:
            # Analyze command intent
            intent_analysis = self.intent_analyzer.analyze_failed_command(
                command_text, context
            )

            # Check for potential discovery unlocks
            discovery_message = (
                await self.discovery_manager.process_failed_command_for_discovery(
                    command_text, intent_analysis, context
                )
            )

            # Generate smart error response
            smart_response = (
                await self.error_response_generator.generate_smart_error_response(
                    command_text, context, original_result.feedback_message
                )
            )

            # Combine with discovery message if available
            if discovery_message:
                smart_response += f"\n\n{discovery_message}"

            return smart_response

        except Exception as e:
            logger.error(f"Error generating smart response: {e}")
            return original_result.feedback_message

    async def get_contextual_help(self, context: dict[str, Any]) -> str:
        """Get contextual help based on current game state."""
        try:
            help_parts = []

            # Get contextual hints
            hints = self.suggestion_engine.get_contextual_hints(context)
            if hints:
                help_parts.extend(hints)

            # Get available actions
            location_objects = context.get("location_objects", [])
            if location_objects:
                help_parts.append(f"📍 Objects here: {', '.join(location_objects[:3])}")

            available_exits = context.get("available_exits", [])
            if available_exits:
                help_parts.append(f"🚪 Exits: {', '.join(available_exits)}")

            inventory = context.get("inventory_items", [])
            if inventory:
                help_parts.append(f"🎒 In inventory: {', '.join(inventory[:3])}")

            # Get discovery progress
            player_id = context.get("player_id", "default_player")
            discovery_progress = self.discovery_manager.get_discovery_progress(
                player_id
            )
            help_parts.append(discovery_progress)

            return (
                "\n".join(help_parts)
                if help_parts
                else "Type 'help' for available commands."
            )

        except Exception as e:
            logger.error(f"Error getting contextual help: {e}")
            return "Type 'help' for available commands."

    async def analyze_command_patterns(self, player_id: str) -> dict[str, Any]:
        """Analyze command patterns for this player."""
        try:
            failed_attempts = self.failed_attempts.get(player_id, [])

            if not failed_attempts:
                return {"status": "no_failures", "message": "No recent failed commands"}

            # Analyze patterns in failed attempts
            patterns = {}
            for attempt in failed_attempts[-5:]:  # Last 5 attempts
                intent = self.intent_analyzer.analyze_failed_command(attempt, {})
                intent_type = intent.get("intent_type", "unknown")
                patterns[intent_type] = patterns.get(intent_type, 0) + 1

            # Get most common failure type
            most_common = (
                max(patterns.items(), key=lambda x: x[1])
                if patterns
                else ("unknown", 0)
            )

            # Generate analysis
            analysis = {
                "total_failed_attempts": len(failed_attempts),
                "recent_failed_attempts": len(failed_attempts[-5:]),
                "failure_patterns": patterns,
                "most_common_failure": most_common[0],
                "suggestions": [],
            }

            # Add targeted suggestions based on patterns
            if most_common[0] == "object_interaction":
                analysis["suggestions"].append(
                    "Focus on examining objects before trying to use them"
                )
            elif most_common[0] == "environmental_action":
                analysis["suggestions"].append(
                    "Try examining environmental elements before interacting"
                )
            elif most_common[0] == "navigation":
                analysis["suggestions"].append(
                    "Use cardinal directions or landmark names for movement"
                )

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing command patterns: {e}")
            return {"status": "error", "message": str(e)}

    def clear_player_data(self, player_id: str) -> None:
        """Clear all tracked data for a player (useful for new sessions)."""
        try:
            if player_id in self.failed_attempts:
                del self.failed_attempts[player_id]

            if player_id in self.discovery_manager.discovered_interactions:
                del self.discovery_manager.discovered_interactions[player_id]

            if player_id in self.discovery_manager.discovery_achievements:
                del self.discovery_manager.discovery_achievements[player_id]

            logger.info(f"Cleared intelligence data for player {player_id}")

        except Exception as e:
            logger.error(f"Error clearing player data: {e}")

    def get_intelligence_status(self, player_id: str) -> dict[str, Any]:
        """Get status of intelligence components for a player."""
        try:
            return {
                "failed_attempts_count": len(self.failed_attempts.get(player_id, [])),
                "discoveries": self.discovery_manager.get_player_discoveries(player_id),
                "hint_cooldowns": len(self.discovery_manager.hint_cooldowns),
                "intelligence_active": True,
            }

        except Exception as e:
            logger.error(f"Error getting intelligence status: {e}")
            return {"intelligence_active": False, "error": str(e)}
