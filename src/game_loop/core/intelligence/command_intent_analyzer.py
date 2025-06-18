"""
Command intent analyzer for understanding failed commands and player intent.

This module analyzes failed commands to understand what players were trying to do
and provides contextual suggestions for successful interactions.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class CommandIntentAnalyzer:
    """Analyze failed commands to understand player intent."""

    def __init__(self) -> None:
        self.intent_patterns = {
            "object_interaction": {
                "verbs": [
                    "write",
                    "inscribe",
                    "carve",
                    "mark",
                    "sign",
                    "draw",
                    "scratch",
                    "engrave",
                ],
                "patterns": [
                    r"(write|inscribe|carve|mark|sign|draw)\s+(.+?)\s+on\s+(.+)",
                    r"(write|inscribe|carve|mark|sign|draw)\s+on\s+(.+?)\s+with\s+(.+)",
                    r"(write|inscribe|carve|mark|sign|draw)\s+(.+)",
                ],
                "suggestion_type": "object_modification",
            },
            "collection_examination": {
                "verbs": ["examine", "look at", "inspect", "study", "read", "check"],
                "patterns": [
                    r"examine\s+the\s+(\w+)",
                    r"look\s+at\s+the\s+(\w+)",
                    r"inspect\s+the\s+(\w+)",
                    r"read\s+the\s+(\w+)",
                    r"check\s+the\s+(\w+)",
                ],
                "suggestion_type": "collection_interaction",
            },
            "environmental_action": {
                "verbs": [
                    "climb",
                    "push",
                    "pull",
                    "move",
                    "lift",
                    "open",
                    "close",
                    "turn",
                ],
                "patterns": [
                    r"(climb|push|pull|move|lift|open|close|turn)\s+(.+)",
                    r"(climb|push|pull|move|lift|open|close|turn)\s+the\s+(.+)",
                ],
                "suggestion_type": "environmental_interaction",
            },
            "exploration": {
                "verbs": ["explore", "search", "investigate", "scan", "survey"],
                "patterns": [
                    r"explore\s+(.+)",
                    r"search\s+(.+)",
                    r"investigate\s+(.+)",
                    r"scan\s+(.+)",
                    r"survey\s+(.+)",
                ],
                "suggestion_type": "detailed_exploration",
            },
            "communication": {
                "verbs": ["ask", "tell", "speak", "say", "whisper", "shout"],
                "patterns": [
                    r"(ask|tell|speak|say)\s+(.+?)\s+about\s+(.+)",
                    r"(ask|tell|speak|say)\s+(.+)",
                    r"(whisper|shout)\s+(.+)",
                ],
                "suggestion_type": "communication_attempt",
            },
            "item_usage": {
                "verbs": ["use", "apply", "utilize", "employ", "activate"],
                "patterns": [
                    r"(use|apply|utilize|employ|activate)\s+(.+?)\s+on\s+(.+)",
                    r"(use|apply|utilize|employ|activate)\s+(.+?)\s+with\s+(.+)",
                    r"(use|apply|utilize|employ|activate)\s+(.+)",
                ],
                "suggestion_type": "item_usage_guidance",
            },
        }

    def analyze_failed_command(
        self, command_text: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Analyze a failed command to determine player intent."""
        try:
            command_lower = command_text.lower().strip()

            # Check each intent pattern
            for intent_type, intent_data in self.intent_patterns.items():
                # Check if command uses verbs associated with this intent
                if any(verb in command_lower for verb in intent_data["verbs"]):
                    # Try to match patterns to extract objects/targets
                    for pattern in intent_data["patterns"]:
                        match = re.search(pattern, command_lower)
                        if match:
                            return {
                                "intent_type": intent_type,
                                "suggestion_type": intent_data["suggestion_type"],
                                "verb": (
                                    match.group(1)
                                    if match.groups()
                                    else intent_data["verbs"][0]
                                ),
                                "targets": (
                                    list(match.groups()[1:])
                                    if len(match.groups()) > 1
                                    else []
                                ),
                                "confidence": self._calculate_confidence(
                                    intent_type, match, context
                                ),
                                "raw_match": match.groups(),
                            }

            # Check for common navigation attempts
            nav_result = self._analyze_navigation_intent(command_lower, context)
            if nav_result:
                return nav_result

            # Check for meta-game commands
            meta_result = self._analyze_meta_intent(command_lower, context)
            if meta_result:
                return meta_result

            return {
                "intent_type": "unknown",
                "confidence": 0.0,
                "suggestion_type": "general_help",
            }

        except Exception as e:
            logger.error(f"Error analyzing command intent: {e}")
            return {"intent_type": "error", "confidence": 0.0, "error": str(e)}

    def _calculate_confidence(
        self, intent_type: str, match: re.Match, context: dict[str, Any]
    ) -> float:
        """Calculate confidence score for intent analysis."""
        base_confidence = 0.7

        # Boost confidence if objects mentioned exist in context
        if match.groups() and len(match.groups()) > 1:
            targets = match.groups()[1:]
            location_objects = context.get("location_objects", [])
            inventory_items = context.get("inventory_items", [])

            for target in targets:
                if target:
                    # Check if target exists in current context
                    target_clean = target.replace("the ", "").strip()
                    if any(
                        target_clean in obj.lower()
                        for obj in location_objects + inventory_items
                    ):
                        base_confidence += 0.2
                        break

        # Boost confidence for common intent types
        if intent_type in ["object_interaction", "environmental_action"]:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _analyze_navigation_intent(
        self, command: str, context: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Analyze navigation-related command intents."""
        nav_patterns = {
            r"go\s+to\s+(.+)": "landmark_navigation",
            r"head\s+to\s+(.+)": "landmark_navigation",
            r"find\s+(.+)": "location_search",
            r"where\s+is\s+(.+)": "location_query",
            r"how\s+do\s+i\s+get\s+to\s+(.+)": "navigation_help",
            r"return\s+to\s+(.+)": "return_navigation",
            r"back\s+to\s+(.+)": "return_navigation",
        }

        for pattern, nav_type in nav_patterns.items():
            match = re.search(pattern, command)
            if match:
                return {
                    "intent_type": "navigation",
                    "suggestion_type": nav_type,
                    "verb": "navigate",
                    "targets": [match.group(1)] if match.groups() else [],
                    "confidence": 0.8,
                }

        return None

    def _analyze_meta_intent(
        self, command: str, context: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Analyze meta-game command intents."""
        meta_patterns = {
            r"what\s+can\s+i\s+do": "help_request",
            r"how\s+do\s+i\s+(.+)": "instruction_request",
            r"what\s+is\s+(.+)": "information_request",
            r"who\s+is\s+(.+)": "character_information",
            r"where\s+am\s+i": "location_information",
            r"(help|commands|instructions)": "general_help",
        }

        for pattern, meta_type in meta_patterns.items():
            match = re.search(pattern, command)
            if match:
                return {
                    "intent_type": "meta_game",
                    "suggestion_type": meta_type,
                    "verb": "query",
                    "targets": list(match.groups()) if match.groups() else [],
                    "confidence": 0.9,
                }

        return None

    def get_similar_commands(
        self, intent_analysis: dict[str, Any], context: dict[str, Any]
    ) -> list[str]:
        """Get similar commands that might work based on intent analysis."""
        intent_type = intent_analysis.get("intent_type", "unknown")
        suggestion_type = intent_analysis.get("suggestion_type", "general")
        targets = intent_analysis.get("targets", [])

        suggestions = []

        if suggestion_type == "object_modification":
            if targets:
                target = targets[0] if targets else "object"
                suggestions.extend(
                    [
                        f"use pen on {target}",
                        f"examine {target}",
                        f"look at {target}",
                    ]
                )

        elif suggestion_type == "collection_interaction":
            if targets:
                target = targets[0]
                # Suggest examining individual items instead of collections
                suggestions.extend(
                    [
                        f"examine a {target[:-1] if target.endswith('s') else target}",
                        f"look at {target} more closely",
                        f"search the {target}",
                    ]
                )

        elif suggestion_type == "environmental_interaction":
            if targets:
                target = targets[0]
                suggestions.extend(
                    [
                        f"examine {target}",
                        f"look at {target}",
                        f"use {target}",
                    ]
                )

        elif suggestion_type == "detailed_exploration":
            suggestions.extend(
                [
                    "look around",
                    "examine room",
                    "search area",
                    "check surroundings",
                ]
            )

        elif suggestion_type == "communication_attempt":
            if targets:
                target = targets[0]
                suggestions.extend(
                    [
                        f"talk to {target}",
                        f"speak with {target}",
                        f"greet {target}",
                    ]
                )

        elif suggestion_type == "landmark_navigation":
            if targets:
                target = targets[0]
                suggestions.extend(
                    [
                        f"go {target}",
                        f"head {target}",
                        f"walk to {target}",
                    ]
                )

        # Add general fallback suggestions
        if not suggestions:
            suggestions.extend(
                [
                    "look around",
                    "examine surroundings",
                    "check inventory",
                    "help",
                ]
            )

        return suggestions[:5]  # Return top 5 suggestions

    def extract_objects_from_command(self, command_text: str) -> list[str]:
        """Extract potential object names from command text."""
        # Remove common command words
        command_words = [
            "go",
            "walk",
            "move",
            "look",
            "examine",
            "use",
            "take",
            "drop",
            "open",
            "close",
        ]
        articles = ["the", "a", "an"]
        prepositions = ["on", "with", "to", "from", "in", "at", "by"]

        words = command_text.lower().split()

        # Filter out command words, articles, and prepositions
        potential_objects = []
        for word in words:
            if (
                word not in command_words
                and word not in articles
                and word not in prepositions
                and len(word) > 2
            ):
                potential_objects.append(word)

        return potential_objects

    def get_intent_summary(self, intent_analysis: dict[str, Any]) -> str:
        """Get a human-readable summary of the intent analysis."""
        intent_type = intent_analysis.get("intent_type", "unknown")
        confidence = intent_analysis.get("confidence", 0.0)
        targets = intent_analysis.get("targets", [])

        if intent_type == "object_interaction":
            if targets:
                return f"Trying to interact with '{targets[0]}' (confidence: {confidence:.1f})"
            return f"Attempting object interaction (confidence: {confidence:.1f})"

        elif intent_type == "environmental_action":
            if targets:
                return f"Trying to manipulate '{targets[0]}' (confidence: {confidence:.1f})"
            return f"Attempting environmental action (confidence: {confidence:.1f})"

        elif intent_type == "navigation":
            if targets:
                return f"Trying to navigate to '{targets[0]}' (confidence: {confidence:.1f})"
            return f"Attempting navigation (confidence: {confidence:.1f})"

        elif intent_type == "meta_game":
            return f"Requesting help or information (confidence: {confidence:.1f})"

        else:
            return f"Intent unclear (confidence: {confidence:.1f})"
