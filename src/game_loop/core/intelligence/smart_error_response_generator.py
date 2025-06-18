"""
Smart error response generator for creating helpful error messages with actionable suggestions.

This module generates intelligent error responses that guide players toward successful
interactions instead of providing generic failure messages.
"""

import logging
from typing import Any

from .command_intent_analyzer import CommandIntentAnalyzer
from .contextual_suggestion_engine import ContextualSuggestionEngine

logger = logging.getLogger(__name__)


class SmartErrorResponseGenerator:
    """Generate intelligent error responses with helpful suggestions."""

    def __init__(
        self,
        intent_analyzer: CommandIntentAnalyzer,
        suggestion_engine: ContextualSuggestionEngine,
    ):
        self.intent_analyzer = intent_analyzer
        self.suggestion_engine = suggestion_engine

        # Response templates for different intent types
        self.acknowledgment_templates = {
            "object_interaction": "I can see you're trying to interact with an object.",
            "collection_examination": "You want to examine a collection of items.",
            "environmental_action": "You're trying to interact with something in the environment.",
            "exploration": "You want to explore and investigate the area.",
            "navigation": "You're trying to navigate or move somewhere.",
            "communication": "You want to communicate with someone.",
            "meta_game": "You're looking for help or information.",
            "unknown": "I'm not sure what you're trying to do.",
        }

    async def generate_smart_error_response(
        self, failed_command: str, context: dict[str, Any], original_error: str = None
    ) -> str:
        """Generate helpful error response instead of generic failure message."""
        try:
            # Analyze what the player was trying to do
            intent_analysis = self.intent_analyzer.analyze_failed_command(
                failed_command, context
            )

            # Generate contextual suggestions
            suggestions = await self.suggestion_engine.generate_suggestions(
                failed_command, intent_analysis, context
            )

            # Build helpful response
            response_parts = []

            # Acknowledge the attempt with understanding
            if intent_analysis.get("confidence", 0) > 0.3:
                acknowledgment = self._generate_intent_acknowledgment(intent_analysis)
                response_parts.append(acknowledgment)
            else:
                response_parts.append("I don't understand that command.")

            # Add specific issue if we can identify it
            issue_explanation = self._generate_issue_explanation(
                intent_analysis, context, original_error
            )
            if issue_explanation:
                response_parts.append(issue_explanation)

            # Provide actionable suggestions
            if suggestions:
                response_parts.append("\n**Here are some things you might try:**")
                for i, suggestion in enumerate(suggestions[:4], 1):
                    response_parts.append(f"  {i}. {suggestion}")

            # Add contextual hints
            hints = self.suggestion_engine.get_contextual_hints(context)
            if hints:
                response_parts.append(f"\n{hints[0]}")  # Add one relevant hint

            # Add discovery prompt
            response_parts.append(
                "\n*Type 'help' for commands or 'look around' to examine your surroundings.*"
            )

            return "\n".join(response_parts)

        except Exception as e:
            logger.error(f"Error generating smart error response: {e}")
            return self._generate_fallback_response(failed_command)

    def _generate_intent_acknowledgment(self, intent_analysis: dict[str, Any]) -> str:
        """Generate acknowledgment of what player was trying to do."""
        intent_type = intent_analysis.get("intent_type", "unknown")
        targets = intent_analysis.get("targets", [])
        verb = intent_analysis.get("verb", "")

        base_acknowledgment = self.acknowledgment_templates.get(
            intent_type, self.acknowledgment_templates["unknown"]
        )

        # Make it more specific if we have targets
        if targets and intent_type != "unknown":
            target = targets[0]
            if intent_type == "object_interaction":
                return f"I understand you want to {verb} something on '{target}'."
            elif intent_type == "environmental_action":
                return f"You're trying to {verb} the '{target}'."
            elif intent_type == "collection_examination":
                return f"You want to examine '{target}' in detail."
            elif intent_type == "navigation":
                return f"You're trying to go to '{target}'."
            elif intent_type == "communication":
                return f"You want to {verb} with '{target}'."

        return base_acknowledgment

    def _generate_issue_explanation(
        self,
        intent_analysis: dict[str, Any],
        context: dict[str, Any],
        original_error: str = None,
    ) -> str:
        """Generate explanation of why the command failed."""
        intent_type = intent_analysis.get("intent_type", "unknown")
        suggestion_type = intent_analysis.get("suggestion_type", "")
        targets = intent_analysis.get("targets", [])

        # Check for specific issues based on intent
        if suggestion_type == "object_modification":
            inventory = context.get("inventory_items", [])
            writing_tools = ["pen", "pencil", "marker", "quill"]
            has_writing_tool = any(
                tool in str(inventory).lower() for tool in writing_tools
            )

            if not has_writing_tool:
                return "However, you don't have anything to write with."
            elif targets:
                target = targets[0]
                location_objects = context.get("location_objects", [])
                if not any(target.lower() in obj.lower() for obj in location_objects):
                    return f"I don't see '{target}' in this location."

        elif suggestion_type == "collection_interaction":
            if targets:
                target = targets[0]
                return f"'{target}' refers to a collection - try examining specific items instead."

        elif suggestion_type == "environmental_interaction":
            if targets:
                target = targets[0]
                location_description = context.get("location_description", "")
                if target.lower() not in location_description.lower():
                    return f"I don't see '{target}' in this area."
                else:
                    return f"The '{target}' might not be interactive in the way you're trying."

        elif suggestion_type == "landmark_navigation":
            if targets:
                target = targets[0]
                return f"I don't recognize '{target}' as a known location."

        # Fall back to original error if we have it and it's informative
        if original_error and len(original_error) > 10:
            return f"The issue: {original_error.lower()}"

        return ""

    def _generate_fallback_response(self, failed_command: str) -> str:
        """Generate a fallback response when smart generation fails."""
        return (
            "I don't understand that command.\n\n"
            "**Try these options:**\n"
            "  1. Type 'help' to see available commands\n"
            "  2. Use 'look around' to see what's here\n"
            "  3. Check your 'inventory' for available items\n"
            "  4. Try simpler commands like 'examine [object]'\n\n"
            "*Remember: be specific about what you want to examine or use.*"
        )

    async def generate_progressive_help(
        self, failed_attempts: list[str], context: dict[str, Any]
    ) -> str:
        """Generate progressive help based on multiple failed attempts."""
        try:
            if len(failed_attempts) < 2:
                return ""

            response_parts = ["\n**I notice you're having trouble. Let me help:**"]

            # Analyze patterns in failed attempts
            common_intents = self._analyze_failure_patterns(failed_attempts, context)

            if "object_interaction" in common_intents:
                response_parts.append(
                    "• For object interactions: First 'examine [object]', then 'use [tool] on [object]'"
                )

            if "exploration" in common_intents:
                response_parts.append(
                    "• For exploration: Try 'look around', then 'examine' specific things you see"
                )

            if "navigation" in common_intents:
                response_parts.append(
                    "• For movement: Use cardinal directions (north, south, east, west) or 'go to [place]'"
                )

            # Add current context help
            available_actions = self._get_available_actions(context)
            if available_actions:
                response_parts.append(
                    f"• Available actions here: {', '.join(available_actions[:3])}"
                )

            response_parts.append(
                "\n*Type 'help [topic]' for specific guidance on commands.*"
            )

            return "\n".join(response_parts)

        except Exception as e:
            logger.error(f"Error generating progressive help: {e}")
            return ""

    def _analyze_failure_patterns(
        self, failed_attempts: list[str], context: dict[str, Any]
    ) -> list[str]:
        """Analyze patterns in failed command attempts."""
        patterns = []

        for attempt in failed_attempts[-3:]:  # Look at last 3 attempts
            intent = self.intent_analyzer.analyze_failed_command(attempt, context)
            intent_type = intent.get("intent_type", "unknown")
            if intent_type != "unknown":
                patterns.append(intent_type)

        # Return unique patterns
        return list(set(patterns))

    def _get_available_actions(self, context: dict[str, Any]) -> list[str]:
        """Get list of available actions based on current context."""
        actions = []

        # Always available
        actions.extend(["look around", "check inventory"])

        # Based on location objects
        location_objects = context.get("location_objects", [])
        if location_objects:
            actions.append(f"examine {location_objects[0]}")

        # Based on available exits
        available_exits = context.get("available_exits", [])
        if available_exits:
            actions.append(f"go {available_exits[0]}")

        # Based on NPCs
        npcs = context.get("npcs", [])
        if npcs:
            npc_name = (
                npcs[0].get("name", "person") if isinstance(npcs[0], dict) else "person"
            )
            actions.append(f"talk to {npc_name}")

        # Based on inventory
        inventory = context.get("inventory_items", [])
        if inventory:
            actions.append(f"use {inventory[0]}")

        return actions[:5]  # Return top 5 available actions

    def format_error_with_context(
        self,
        error_message: str,
        intent_summary: str = "",
        suggestions: list[str] = None,
    ) -> str:
        """Format error message with additional context and suggestions."""
        parts = [error_message]

        if intent_summary:
            parts.append(f"\n*Analysis: {intent_summary}*")

        if suggestions:
            parts.append("\n**Suggestions:**")
            for i, suggestion in enumerate(suggestions[:3], 1):
                parts.append(f"  {i}. {suggestion}")

        return "\n".join(parts)
