"""
Command Mapper for Game Loop.
Maps NLP intents to game commands and handles synonym resolution.
"""

from typing import Any

from game_loop.core.input_processor import CommandType, ParsedCommand


class CommandMapper:
    """
    Maps NLP intents to game commands and handles synonym resolution.
    """

    def __init__(self) -> None:
        """
        Initialize with action mappings and synonyms.
        """
        self._setup_action_mappings()

    def _setup_action_mappings(self) -> None:
        """Set up mappings from various action phrasings to canonical commands."""
        # Movement action mappings
        self.movement_actions = {
            "go": "go",
            "move": "go",
            "walk": "go",
            "run": "go",
            "travel": "go",
            "head": "go",
            "proceed": "go",
        }

        # Direction mappings
        self.direction_synonyms = {
            "north": ["n", "northern", "northward", "northwards"],
            "south": ["s", "southern", "southward", "southwards"],
            "east": ["e", "eastern", "eastward", "eastwards"],
            "west": ["w", "western", "westward", "westwards"],
            "up": ["u", "upward", "upwards", "upstairs", "above"],
            "down": ["d", "downward", "downwards", "downstairs", "below"],
        }

        # Look action mappings
        self.look_actions = {
            "look": "look",
            "see": "look",
            "observe": "look",
            "view": "look",
            "gaze": "look",
            "check": "look",
            "scan": "look",
            "examine": "look",
        }

        # Take action mappings
        self.take_actions = {
            "take": "take",
            "get": "take",
            "grab": "take",
            "pick": "take",
            "acquire": "take",
            "collect": "take",
            "retrieve": "take",
        }

        # Drop action mappings
        self.drop_actions = {
            "drop": "drop",
            "put": "drop",
            "discard": "drop",
            "throw": "drop",
            "release": "drop",
            "leave": "drop",
        }

        # Use action mappings
        self.use_actions = {
            "use": "use",
            "activate": "use",
            "operate": "use",
            "apply": "use",
            "employ": "use",
            "utilize": "use",
        }

        # Complex action mappings for multi-object interactions
        self.complex_actions = {
            "put": "put",
            "place": "put",
            "insert": "put",
            "stick": "put",
            "store": "put",
            "stash": "put",
            "hide": "put",
            "combine": "combine",
            "attach": "attach",
            "connect": "connect",
            "tie": "tie",
            "open": "open",
            "unlock": "unlock",
        }

        # Examine action mappings
        self.examine_actions = {
            "examine": "examine",
            "inspect": "examine",
            "scrutinize": "examine",
            "study": "examine",
            "check": "examine",
            "analyze": "examine",
            "investigate": "examine",
        }

        # Talk action mappings
        self.talk_actions = {
            "talk": "talk",
            "speak": "talk",
            "chat": "talk",
            "converse": "talk",
            "communicate": "talk",
            "discuss": "talk",
            "say": "talk",
            "ask": "talk",
        }

        # Inventory action mappings
        self.inventory_actions = {
            "inventory": "inventory",
            "i": "inventory",
            "items": "inventory",
            "belongings": "inventory",
            "possessions": "inventory",
            "carrying": "inventory",
        }

        # Help action mappings
        self.help_actions = {
            "help": "help",
            "h": "help",
            "?": "help",
            "commands": "help",
            "info": "help",
            "instructions": "help",
        }

        # Quit action mappings
        self.quit_actions = {
            "quit": "quit",
            "exit": "quit",
            "q": "quit",
            "leave": "quit",
            "end": "quit",
            "bye": "quit",
        }

        # Create a merged dictionary for easy lookup
        self.all_action_mappings = {}
        self.all_action_mappings.update(self.movement_actions)
        self.all_action_mappings.update(self.look_actions)
        self.all_action_mappings.update(self.take_actions)
        self.all_action_mappings.update(self.drop_actions)
        self.all_action_mappings.update(self.use_actions)
        self.all_action_mappings.update(self.complex_actions)
        self.all_action_mappings.update(self.examine_actions)
        self.all_action_mappings.update(self.talk_actions)
        self.all_action_mappings.update(self.inventory_actions)
        self.all_action_mappings.update(self.help_actions)
        self.all_action_mappings.update(self.quit_actions)

    def map_intent_to_command(self, intent_data: dict[str, Any]) -> ParsedCommand:
        """
        Convert NLP intent data to a ParsedCommand.

        Args:
            intent_data: Dictionary containing intent data from NLP processor

        Returns:
            ParsedCommand object representing the mapped command
        """
        # Extract the command type
        command_type_str = intent_data.get("command_type", "UNKNOWN")

        try:
            # Handle case sensitivity in enum matching
            command_type = next(
                (ct for ct in CommandType if ct.name == command_type_str.upper()),
                CommandType.UNKNOWN,
            )
        except Exception:
            command_type = CommandType.UNKNOWN

        # Get the action and normalize it
        action = intent_data.get("action", "")
        canonical_action = self.get_canonical_action(action)

        # Get subject and target
        subject = intent_data.get("subject")
        target = intent_data.get("target")

        # Handle special cases for complex commands
        if canonical_action in self.complex_actions.values() and target:
            # For complex actions like "put X in Y", we should use USE command type
            command_type = CommandType.USE

        # Handle direction normalization for movement commands
        if command_type == CommandType.MOVEMENT and subject:
            subject = self._normalize_direction(subject)

        # Create parameters dictionary with confidence if available
        parameters = {}
        if confidence := intent_data.get("confidence"):
            parameters["confidence"] = confidence

        return ParsedCommand(
            command_type=command_type,
            action=canonical_action,
            subject=subject,
            target=target,
            parameters=parameters,
        )

    def get_canonical_action(self, action_text: str) -> str:
        """
        Map various action phrasings to canonical commands.

        Args:
            action_text: The action text to normalize

        Returns:
            Canonical action string
        """
        if not action_text:
            return "unknown"

        # Normalize to lowercase
        normalized = action_text.lower().strip()

        # Check for exact matches first
        if normalized in self.all_action_mappings:
            return self.all_action_mappings[normalized]

        # Check for partial matches (e.g., "pick up" should match "pick")
        for action_phrase in self.all_action_mappings:
            if normalized.startswith(action_phrase):
                return self.all_action_mappings[action_phrase]

        return normalized

    def _normalize_direction(self, direction: str) -> str:
        """
        Normalize direction synonyms to canonical directions.

        Args:
            direction: Direction string to normalize

        Returns:
            Canonical direction string
        """
        if not direction:
            return ""

        normalized = direction.lower().strip()

        # Check direct matches first
        if normalized in self.direction_synonyms:
            return normalized

        # Check against all synonyms
        for canonical, synonyms in self.direction_synonyms.items():
            if normalized in synonyms:
                return canonical

        return normalized
