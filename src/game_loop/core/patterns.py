"""Advanced pattern management system for rule-based action classification.

This module provides a comprehensive pattern management system for classifying
user actions based on linguistic patterns, verbs, and synonyms.

NOTE: This is an advanced pattern system for complex use cases. The main
action classification system uses the simpler ActionPatternManager from
actions.patterns module. This system is maintained for advanced pattern
management and testing purposes.
"""

import logging
import re
from dataclasses import dataclass, field
from re import Pattern
from typing import Any

from .actions.types import ActionType

logger = logging.getLogger(__name__)


@dataclass
class ActionPattern:
    """Represents a pattern for matching user actions."""

    name: str
    action_type: ActionType
    verbs: set[str] = field(default_factory=set)
    synonyms: set[str] = field(default_factory=set)
    regex_patterns: list[Pattern] = field(default_factory=list)
    priority: int = 0
    context_modifiers: dict[str, float] = field(default_factory=dict)

    def matches(self, text: str) -> bool:
        """Check if the text matches this pattern."""
        text_lower = text.lower()

        # Check verb matches
        for verb in self.verbs:
            if verb in text_lower:
                return True

        # Check synonym matches
        for synonym in self.synonyms:
            if synonym in text_lower:
                return True

        # Check regex patterns
        for pattern in self.regex_patterns:
            if pattern.search(text_lower):
                return True

        return False

    def get_confidence(self, text: str, context: dict[str, Any] | None = None) -> float:
        """Calculate confidence score for this pattern match."""
        if not self.matches(text):
            return 0.0

        confidence = 1.0
        text_lower = text.lower()

        # Boost confidence for exact verb matches
        for verb in self.verbs:
            if text_lower.startswith(verb):
                confidence *= 1.5

        # Apply context modifiers
        if context and self.context_modifiers:
            for key, modifier in self.context_modifiers.items():
                if key in context:
                    confidence *= modifier

        return min(confidence, 2.0)  # Cap at 2.0


class ActionPatternManager:
    """Manages action patterns for classification."""

    def __init__(self) -> None:
        """Initialize the pattern manager with default patterns."""
        self.patterns: dict[str, ActionPattern] = {}
        self._initialize_default_patterns()

    def _initialize_default_patterns(self) -> None:
        """Initialize default patterns for each action type."""

        # Movement patterns
        self.register_pattern(
            ActionPattern(
                name="basic_movement",
                action_type=ActionType.MOVEMENT,
                verbs={"go", "walk", "run", "move", "travel", "head", "proceed"},
                synonyms={"journey", "venture", "wander", "stroll", "march", "stride"},
                regex_patterns=[
                    re.compile(r"\b(go|walk|run|move)\s+(to|towards?|into|through)\b"),
                    re.compile(r"\b(north|south|east|west|up|down|left|right)\b"),
                    re.compile(
                        r"\b(n|s|e|w|u|d)\b(?:\s|$)"
                    ),  # Single letter directions
                    re.compile(r"\benter\s+(\w+)\b"),
                    re.compile(r"\bleave\s+(\w+)?\b"),
                    re.compile(r"\bclimb\s+(up|down)\b"),
                ],
                priority=1,
                context_modifiers={"in_location": 1.2, "has_exits": 1.5},
            )
        )

        # Object interaction patterns
        self.register_pattern(
            ActionPattern(
                name="object_manipulation",
                action_type=ActionType.OBJECT_INTERACTION,
                verbs={
                    "take",
                    "get",
                    "pick",
                    "grab",
                    "drop",
                    "put",
                    "place",
                    "use",
                    "examine",
                    "inspect",
                },
                synonyms={
                    "acquire",
                    "obtain",
                    "collect",
                    "gather",
                    "release",
                    "deposit",
                    "utilize",
                    "employ",
                },
                regex_patterns=[
                    re.compile(r"\b(take|get|pick up|grab)\s+(\w+)\b"),
                    re.compile(r"\b(drop|put down|place)\s+(\w+)\b"),
                    re.compile(
                        r"\b(use|apply|activate)\s+(\w+)(?:\s+(?:on|with)\s+(\w+))?\b"
                    ),
                    re.compile(r"\b(examine|look at|inspect|check)\s+(\w+)\b"),
                    re.compile(r"\b(open|close|lock|unlock)\s+(\w+)\b"),
                    re.compile(r"\b(give|offer|hand)\s+(\w+)\s+to\s+(\w+)\b"),
                ],
                priority=2,
                context_modifiers={"has_inventory": 1.3, "near_items": 1.4},
            )
        )

        # Quest patterns
        self.register_pattern(
            ActionPattern(
                name="quest_management",
                action_type=ActionType.QUEST,
                verbs={
                    "complete",
                    "finish",
                    "accept",
                    "start",
                    "begin",
                    "check",
                    "abandon",
                },
                synonyms={
                    "accomplish",
                    "fulfill",
                    "undertake",
                    "commence",
                    "initiate",
                    "verify",
                    "quit",
                },
                regex_patterns=[
                    re.compile(r"\b(complete|finish|turn in)\s+(quest|mission|task)\b"),
                    re.compile(r"\b(accept|start|begin)\s+(quest|mission|task)\b"),
                    re.compile(
                        r"\b(check|view|show)\s+(quest|mission|task|progress)\b"
                    ),
                    re.compile(r"\b(abandon|quit|cancel)\s+(quest|mission|task)\b"),
                    re.compile(r"\bquest\s+(log|journal|status)\b"),
                ],
                priority=3,
                context_modifiers={"has_active_quest": 1.5, "near_quest_giver": 1.4},
            )
        )

        # Conversation patterns
        self.register_pattern(
            ActionPattern(
                name="dialogue",
                action_type=ActionType.CONVERSATION,
                verbs={"talk", "speak", "say", "tell", "ask", "greet", "chat"},
                synonyms={
                    "converse",
                    "communicate",
                    "discuss",
                    "inquire",
                    "address",
                    "hail",
                },
                regex_patterns=[
                    re.compile(r"\b(talk|speak|chat)\s+(?:to|with)\s+(\w+)\b"),
                    re.compile(r'\b(say|tell)\s+"([^"]+)"\s*(?:to\s+(\w+))?\b'),
                    re.compile(r"\b(ask|inquire)\s+(\w+)?\s*(?:about\s+(\w+))?\b"),
                    re.compile(r"\b(greet|hail|hello|hi|hey)\s*(\w+)?\b"),
                    re.compile(r'^"([^"]+)"$'),  # Direct quotes
                    re.compile(
                        r"\b(yes|no|maybe|okay|ok)\b(?:\s|$)"
                    ),  # Simple responses
                ],
                priority=4,
                context_modifiers={"near_npc": 1.6, "in_dialogue": 1.8},
            )
        )

        # Query patterns
        self.register_pattern(
            ActionPattern(
                name="information_query",
                action_type=ActionType.QUERY,
                verbs={"what", "where", "who", "when", "why", "how", "which"},
                synonyms={
                    "whereabouts",
                    "location",
                    "identity",
                    "reason",
                    "method",
                    "selection",
                },
                regex_patterns=[
                    re.compile(r"^(what|where|who|when|why|how|which)\s+"),
                    re.compile(
                        r"\b(what|where|who|when|why|how)\s+(is|are|was|were)\b"
                    ),
                    re.compile(
                        r"\b(tell me|show me|explain)\s+(about|how|what|where)\b"
                    ),
                    re.compile(r"\?$"),  # Questions ending with ?
                    re.compile(r"\b(help|hint|clue|tip)\b"),
                ],
                priority=5,
                context_modifiers={"needs_information": 1.5},
            )
        )

        # System patterns
        self.register_pattern(
            ActionPattern(
                name="system_command",
                action_type=ActionType.SYSTEM,
                verbs={"save", "load", "quit", "exit", "help", "settings", "options"},
                synonyms={
                    "store",
                    "restore",
                    "leave",
                    "configure",
                    "preferences",
                    "menu",
                },
                regex_patterns=[
                    re.compile(r"\b(save|load)\s*(game|progress)?\b"),
                    re.compile(r"\b(quit|exit|leave)\s*(game)?\b"),
                    re.compile(r"\b(help|tutorial|guide|manual)\b"),
                    re.compile(r"\b(settings|options|preferences|config)\b"),
                    re.compile(r"\b(pause|resume|restart)\b"),
                    re.compile(r"\binventory\b"),
                    re.compile(r"\b(map|journal|stats?)\b"),
                ],
                priority=6,
                context_modifiers={"in_menu": 1.7},
            )
        )

        # Physical action patterns
        self.register_pattern(
            ActionPattern(
                name="physical_action",
                action_type=ActionType.PHYSICAL,
                verbs={
                    "push",
                    "pull",
                    "climb",
                    "jump",
                    "attack",
                    "defend",
                    "fight",
                    "hit",
                },
                synonyms={
                    "shove",
                    "tug",
                    "scale",
                    "leap",
                    "strike",
                    "protect",
                    "battle",
                    "punch",
                },
                regex_patterns=[
                    re.compile(r"\b(push|pull|shove|tug)\s+(\w+)\b"),
                    re.compile(r"\b(climb|scale)\s+(up|down|over)?\s*(\w+)?\b"),
                    re.compile(r"\b(jump|leap|hop)\s*(over|across|to)?\s*(\w+)?\b"),
                    re.compile(r"\b(attack|hit|strike|fight)\s+(\w+)\b"),
                    re.compile(r"\b(defend|block|parry|dodge)\b"),
                    re.compile(r"\b(throw|toss|hurl)\s+(\w+)(?:\s+at\s+(\w+))?\b"),
                    re.compile(r"\b(break|smash|destroy)\s+(\w+)\b"),
                ],
                priority=7,
                context_modifiers={"in_combat": 1.8, "has_weapon": 1.3},
            )
        )

        # Observation patterns
        self.register_pattern(
            ActionPattern(
                name="observation",
                action_type=ActionType.OBSERVATION,
                verbs={
                    "look",
                    "examine",
                    "listen",
                    "smell",
                    "feel",
                    "touch",
                    "observe",
                    "search",
                },
                synonyms={
                    "gaze",
                    "peer",
                    "hear",
                    "sniff",
                    "sense",
                    "perceive",
                    "investigate",
                },
                regex_patterns=[
                    re.compile(
                        r"\b(look|gaze|peer)\s*(at|around|behind|under|over)?\s*(\w+)?\b"
                    ),
                    re.compile(r"\b(examine|inspect|study)\s+(\w+)\b"),
                    re.compile(r"\b(listen|hear)\s*(to|for)?\s*(\w+)?\b"),
                    re.compile(r"\b(smell|sniff)\s*(\w+)?\b"),
                    re.compile(r"\b(feel|touch)\s+(\w+)\b"),
                    re.compile(r"\b(search|investigate)\s*(for)?\s*(\w+)?\b"),
                    re.compile(r"^l$"),  # Single 'l' for look
                ],
                priority=8,
                context_modifiers={"dark_location": 0.7, "has_light": 1.2},
            )
        )

    def register_pattern(self, pattern: ActionPattern) -> None:
        """Register a new action pattern."""
        self.patterns[pattern.name] = pattern
        logger.debug(
            f"Registered pattern: {pattern.name} for {pattern.action_type.value}"
        )

    def unregister_pattern(self, pattern_name: str) -> bool:
        """Unregister an action pattern."""
        if pattern_name in self.patterns:
            del self.patterns[pattern_name]
            logger.debug(f"Unregistered pattern: {pattern_name}")
            return True
        return False

    def classify_action(
        self, text: str, context: dict[str, Any] | None = None
    ) -> tuple[ActionType, float]:
        """Classify an action based on the input text and context."""
        best_match = ActionType.UNKNOWN
        best_confidence = 0.0

        # Sort patterns by priority
        sorted_patterns = sorted(self.patterns.values(), key=lambda p: p.priority)

        for pattern in sorted_patterns:
            confidence = pattern.get_confidence(text, context)
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = pattern.action_type

        logger.debug(
            f"Classified '{text}' as {best_match.value} "
            f"with confidence {best_confidence}"
        )
        return best_match, best_confidence

    def get_patterns_by_type(self, action_type: ActionType) -> list[ActionPattern]:
        """Get all patterns for a specific action type."""
        return [p for p in self.patterns.values() if p.action_type == action_type]

    def add_verb_to_pattern(self, pattern_name: str, verb: str) -> bool:
        """Add a verb to an existing pattern."""
        if pattern_name in self.patterns:
            self.patterns[pattern_name].verbs.add(verb.lower())
            logger.debug(f"Added verb '{verb}' to pattern '{pattern_name}'")
            return True
        return False

    def add_synonym_to_pattern(self, pattern_name: str, synonym: str) -> bool:
        """Add a synonym to an existing pattern."""
        if pattern_name in self.patterns:
            self.patterns[pattern_name].synonyms.add(synonym.lower())
            logger.debug(f"Added synonym '{synonym}' to pattern '{pattern_name}'")
            return True
        return False

    def add_regex_to_pattern(self, pattern_name: str, regex_pattern: str) -> bool:
        """Add a regex pattern to an existing pattern."""
        if pattern_name in self.patterns:
            try:
                compiled = re.compile(regex_pattern, re.IGNORECASE)
                self.patterns[pattern_name].regex_patterns.append(compiled)
                logger.debug(
                    f"Added regex '{regex_pattern}' to pattern '{pattern_name}'"
                )
                return True
            except re.error as e:
                logger.error(f"Invalid regex pattern: {e}")
                return False
        return False

    def update_pattern_priority(self, pattern_name: str, priority: int) -> bool:
        """Update the priority of a pattern."""
        if pattern_name in self.patterns:
            self.patterns[pattern_name].priority = priority
            logger.debug(f"Updated priority of '{pattern_name}' to {priority}")
            return True
        return False

    def set_context_modifier(
        self, pattern_name: str, context_key: str, modifier: float
    ) -> bool:
        """Set a context modifier for a pattern."""
        if pattern_name in self.patterns:
            self.patterns[pattern_name].context_modifiers[context_key] = modifier
            logger.debug(
                f"Set context modifier '{context_key}' = {modifier} "
                f"for pattern '{pattern_name}'"
            )
            return True
        return False

    def validate_patterns(self) -> list[str]:
        """Validate all registered patterns and return any issues."""
        issues = []

        for name, pattern in self.patterns.items():
            # Check for empty pattern definitions
            if (
                not pattern.verbs
                and not pattern.synonyms
                and not pattern.regex_patterns
            ):
                issues.append(f"Pattern '{name}' has no matching criteria")

            # Check for invalid regex patterns
            for i, regex in enumerate(pattern.regex_patterns):
                try:
                    _ = regex.pattern  # Access pattern to ensure it's valid
                except Exception as e:
                    issues.append(
                        f"Pattern '{name}' has invalid regex at index {i}: {e}"
                    )

            # Check for duplicate verbs across patterns of same type
            same_type_patterns = self.get_patterns_by_type(pattern.action_type)
            for other in same_type_patterns:
                if other.name != name:
                    verb_overlap = pattern.verbs & other.verbs
                    if verb_overlap and pattern.priority == other.priority:
                        issues.append(
                            f"Patterns '{name}' and '{other.name}' "
                            f"share verbs {verb_overlap} with same priority"
                        )

        return issues

    def get_all_verbs(self) -> set[str]:
        """Get all registered verbs across all patterns."""
        all_verbs = set()
        for pattern in self.patterns.values():
            all_verbs.update(pattern.verbs)
        return all_verbs

    def get_pattern_info(self, pattern_name: str) -> dict[str, Any] | None:
        """Get detailed information about a specific pattern."""
        if pattern_name not in self.patterns:
            return None

        pattern = self.patterns[pattern_name]
        return {
            "name": pattern.name,
            "action_type": pattern.action_type.value,
            "verbs": list(pattern.verbs),
            "synonyms": list(pattern.synonyms),
            "regex_count": len(pattern.regex_patterns),
            "priority": pattern.priority,
            "context_modifiers": pattern.context_modifiers,
        }
