"""
Interactive examination handler for detailed object exploration and sub-object generation.

This handler enhances the examination system to generate interactive sub-objects
during detailed examination, particularly for collections and complex objects.
"""

import logging
import re
from typing import TYPE_CHECKING, Any

from rich.console import Console

from game_loop.core.command_handlers.base_handler import CommandHandler
from game_loop.core.input_processor import ParsedCommand
from game_loop.state.models import ActionResult

if TYPE_CHECKING:
    from game_loop.state.manager import GameStateManager
    from game_loop.state.models import Location, PlayerState

logger = logging.getLogger(__name__)


class InteractiveExaminationHandler(CommandHandler):
    """
    Enhanced examination handler that generates sub-objects and interactive elements.

    Integrates with world generation to create detailed sub-objects during examination,
    particularly for collections like books, shelves, and complex machinery.
    """

    def __init__(self, console: Console, state_manager: "GameStateManager"):
        super().__init__(console, state_manager)

        # Define collection types and their sub-object patterns
        self.collection_types = {
            "books": {
                "singular": "book",
                "sub_objects": ["journal", "tome", "manual", "diary", "notebook"],
                "examination_depth": 3,
                "interaction_verbs": ["read", "open", "flip through"],
            },
            "shelves": {
                "singular": "shelf",
                "sub_objects": ["books", "ornaments", "files", "equipment"],
                "examination_depth": 2,
                "interaction_verbs": ["search", "look behind", "check"],
            },
            "documents": {
                "singular": "document",
                "sub_objects": ["letter", "report", "memo", "form", "certificate"],
                "examination_depth": 2,
                "interaction_verbs": ["read", "examine", "unfold"],
            },
            "instruments": {
                "singular": "instrument",
                "sub_objects": ["gauge", "dial", "control", "meter", "display"],
                "examination_depth": 2,
                "interaction_verbs": ["adjust", "read", "operate"],
            },
            "files": {
                "singular": "file",
                "sub_objects": ["folder", "dossier", "report", "record", "case file"],
                "examination_depth": 2,
                "interaction_verbs": ["open", "read", "browse"],
            },
        }

        # Track generated sub-objects to maintain consistency
        self.generated_sub_objects: dict[str, list[dict[str, Any]]] = {}

    async def handle(self, command: ParsedCommand) -> ActionResult:
        """
        Handle interactive examination commands.

        Args:
            command: The parsed examination command

        Returns:
            ActionResult with examination details and any generated sub-objects
        """
        try:
            # Get required game state
            player_state, current_location, world_state = (
                await self.get_required_state()
            )

            if not current_location or not player_state:
                return ActionResult(
                    success=False,
                    feedback_message="Error: Cannot determine current location or player state.",
                )

            # Extract examination target
            target = self._extract_examination_target(command)
            if not target:
                return ActionResult(
                    success=False, feedback_message="What do you want to examine?"
                )

            # Check if target is a collection that can generate sub-objects
            collection_info = self._identify_collection(target)
            if collection_info:
                return await self._handle_collection_examination(
                    target, collection_info, current_location, player_state
                )
            else:
                return await self._handle_individual_examination(
                    target, current_location, player_state
                )

        except Exception as e:
            logger.error(f"Error in interactive examination: {e}")
            return ActionResult(
                success=False,
                feedback_message="You have trouble focusing on the details.",
            )

    def _extract_examination_target(self, command: ParsedCommand) -> str | None:
        """Extract the target object for examination."""
        # Handle different command patterns
        if command.subject:
            return command.subject.lower()

        # Handle "examine the books" pattern
        full_command = f"{command.action} {command.target or ''}".strip()

        # Look for "the [object]" pattern
        match = re.search(r"\bthe\s+(\w+)", full_command)
        if match:
            return match.group(1)

        # Look for any remaining object reference
        words = full_command.split()
        if len(words) > 1:
            return words[-1]  # Usually the last word is the object

        return None

    def _identify_collection(self, target: str) -> dict[str, Any] | None:
        """Identify if target is a collection type that can generate sub-objects."""
        target_lower = target.lower()

        for collection_name, collection_info in self.collection_types.items():
            if collection_name in target_lower or target_lower in collection_name:
                return {"type": collection_name, "info": collection_info}

        return None

    async def _handle_collection_examination(
        self,
        target: str,
        collection_info: dict[str, Any],
        current_location: "Location",
        player_state: "PlayerState",
    ) -> ActionResult:
        """Handle examination of collections that generate sub-objects."""
        collection_type = collection_info["type"]
        collection_data = collection_info["info"]

        # Check if target exists in location
        if not await self._target_exists_in_location(target, current_location):
            return ActionResult(
                success=False,
                feedback_message=f"I don't see '{target}' here to examine.",
            )

        # Generate or retrieve sub-objects for this collection
        location_id = getattr(current_location, "location_id", "unknown")
        cache_key = f"{location_id}_{collection_type}_{target}"

        if cache_key not in self.generated_sub_objects:
            sub_objects = await self._generate_sub_objects(
                collection_type, collection_data, current_location
            )
            self.generated_sub_objects[cache_key] = sub_objects
        else:
            sub_objects = self.generated_sub_objects[cache_key]

        # Generate examination response
        response = self._format_collection_examination_response(
            target, collection_type, sub_objects, collection_data
        )

        return ActionResult(
            success=True,
            feedback_message=response,
            object_changes=[
                {
                    "examination_type": "collection",
                    "collection_type": collection_type,
                    "generated_sub_objects": sub_objects,
                    "cache_key": cache_key,
                }
            ],
        )

    async def _generate_sub_objects(
        self,
        collection_type: str,
        collection_data: dict[str, Any],
        location: "Location",
    ) -> list[dict[str, Any]]:
        """Generate sub-objects for a collection."""
        sub_object_types = collection_data["sub_objects"]
        num_objects = min(len(sub_object_types), 4)  # Generate 2-4 sub-objects

        sub_objects = []
        location_theme = await self._get_location_theme(location)

        for i in range(num_objects):
            sub_object_type = sub_object_types[i % len(sub_object_types)]
            sub_object = await self._create_themed_sub_object(
                sub_object_type, location_theme, collection_type
            )
            sub_objects.append(sub_object)

        return sub_objects

    async def _get_location_theme(self, location: "Location") -> str:
        """Get the theme of the current location for contextual sub-object generation."""
        location_name = getattr(location, "name", "").lower()
        location_desc = getattr(location, "description", "").lower()

        # Determine theme based on location characteristics
        if any(
            word in location_name or word in location_desc
            for word in ["library", "archive", "study"]
        ):
            return "academic"
        elif any(
            word in location_name or word in location_desc
            for word in ["office", "corporate", "business"]
        ):
            return "corporate"
        elif any(
            word in location_name or word in location_desc
            for word in ["lab", "research", "science"]
        ):
            return "scientific"
        elif any(
            word in location_name or word in location_desc
            for word in ["security", "guard", "control"]
        ):
            return "security"
        else:
            return "general"

    async def _create_themed_sub_object(
        self, sub_object_type: str, theme: str, collection_type: str
    ) -> dict[str, Any]:
        """Create a themed sub-object based on location and collection context."""
        # Theme-specific naming patterns
        theme_patterns = {
            "academic": {
                "book": [
                    "ancient tome",
                    "research journal",
                    "scholarly text",
                    "reference manual",
                ],
                "document": [
                    "thesis paper",
                    "research note",
                    "academic report",
                    "study guide",
                ],
                "instrument": [
                    "microscope",
                    "measurement device",
                    "analysis tool",
                    "calibrator",
                ],
            },
            "corporate": {
                "book": [
                    "quarterly report",
                    "policy manual",
                    "business guide",
                    "procedure book",
                ],
                "document": [
                    "resignation letter",
                    "memo",
                    "employee file",
                    "meeting minutes",
                ],
                "instrument": [
                    "calculator",
                    "computer terminal",
                    "communication device",
                    "monitor",
                ],
            },
            "scientific": {
                "book": [
                    "lab manual",
                    "research log",
                    "technical specification",
                    "field guide",
                ],
                "document": [
                    "test results",
                    "experiment log",
                    "data sheet",
                    "safety protocol",
                ],
                "instrument": [
                    "spectrometer",
                    "analyzer",
                    "measurement gauge",
                    "control panel",
                ],
            },
            "security": {
                "book": [
                    "security log",
                    "protocol manual",
                    "incident report",
                    "access record",
                ],
                "document": [
                    "security briefing",
                    "alert notice",
                    "clearance form",
                    "patrol log",
                ],
                "instrument": [
                    "monitor",
                    "surveillance camera",
                    "access scanner",
                    "alarm panel",
                ],
            },
        }

        # Get themed names for this sub-object type
        themed_names = theme_patterns.get(theme, {}).get(
            sub_object_type, [f"{theme} {sub_object_type}"]
        )

        # Select a name (could be random in real implementation)
        name = themed_names[0] if themed_names else f"{theme} {sub_object_type}"

        # Generate description based on type and theme
        description = self._generate_sub_object_description(
            name, sub_object_type, theme
        )

        return {
            "name": name,
            "type": sub_object_type,
            "theme": theme,
            "description": description,
            "interactable": True,
            "parent_collection": collection_type,
        }

    def _generate_sub_object_description(
        self, name: str, object_type: str, theme: str
    ) -> str:
        """Generate contextual description for sub-object."""
        if object_type == "book" or object_type == "journal":
            return f"A {name} with detailed text and diagrams. It looks like it contains valuable information."
        elif object_type == "document":
            return (
                f"A {name} with official-looking text. The content appears important."
            )
        elif object_type == "instrument":
            return f"A {name} with various controls and displays. It seems to be functional."
        else:
            return f"A {name} that appears to be part of the {theme} collection here."

    def _format_collection_examination_response(
        self,
        target: str,
        collection_type: str,
        sub_objects: list[dict[str, Any]],
        collection_data: dict[str, Any],
    ) -> str:
        """Format the response for collection examination."""
        response_parts = [f"You examine the {target} closely."]

        if sub_objects:
            response_parts.append(
                "\nAs you look more carefully, you notice several individual items:"
            )

            for sub_object in sub_objects:
                name = sub_object["name"]
                description = sub_object["description"]
                response_parts.append(f"• **{name.title()}**: {description}")

            # Add interaction hints
            interaction_verbs = collection_data.get("interaction_verbs", ["examine"])
            first_object = sub_objects[0]["name"]
            hint_verb = interaction_verbs[0]

            response_parts.append(
                f"\n*You can now examine these individual items. Try '{hint_verb} {first_object}'*"
            )

        return "\n".join(response_parts)

    async def _handle_individual_examination(
        self, target: str, current_location: "Location", player_state: "PlayerState"
    ) -> ActionResult:
        """Handle examination of individual objects."""
        if not await self._target_exists_in_location(target, current_location):
            # Check if it's a generated sub-object
            sub_object = self._find_generated_sub_object(target, current_location)
            if sub_object:
                return self._examine_generated_sub_object(sub_object)
            else:
                return ActionResult(
                    success=False,
                    feedback_message=f"I don't see '{target}' here to examine.",
                )

        # Handle examination of existing location objects
        return await self._examine_location_object(target, current_location)

    def _find_generated_sub_object(
        self, target: str, location: "Location"
    ) -> dict[str, Any] | None:
        """Find a generated sub-object by name."""
        location_id = getattr(location, "location_id", "unknown")

        for cache_key, sub_objects in self.generated_sub_objects.items():
            if location_id in cache_key:
                for sub_object in sub_objects:
                    if target.lower() in sub_object["name"].lower():
                        return sub_object

        return None

    def _examine_generated_sub_object(self, sub_object: dict[str, Any]) -> ActionResult:
        """Examine a generated sub-object in detail."""
        name = sub_object["name"]
        description = sub_object["description"]
        object_type = sub_object["type"]

        # Generate detailed examination based on type
        if object_type in ["book", "journal"]:
            detailed_desc = (
                f"You examine the {name} carefully. {description}\n\n"
                f"The pages contain detailed information that would take time to read through. "
                f"You notice several interesting sections and diagrams."
            )
        elif object_type == "document":
            detailed_desc = (
                f"You examine the {name} closely. {description}\n\n"
                f"The document appears to contain official information and could be read for more details."
            )
        elif object_type == "instrument":
            detailed_desc = (
                f"You examine the {name} in detail. {description}\n\n"
                f"The instrument has various controls and displays that could potentially be operated."
            )
        else:
            detailed_desc = f"You examine the {name} thoroughly. {description}"

        return ActionResult(
            success=True,
            feedback_message=detailed_desc,
            object_changes=[
                {"examination_type": "sub_object", "sub_object": sub_object}
            ],
        )

    async def _examine_location_object(
        self, target: str, location: "Location"
    ) -> ActionResult:
        """Examine an existing location object."""
        # This would integrate with existing object examination
        return ActionResult(
            success=True,
            feedback_message=f"You examine the {target}. It appears to be a normal {target} with no special features.",
            object_changes=[{"examination_type": "standard_object", "target": target}],
        )

    async def _target_exists_in_location(
        self, target: str, location: "Location"
    ) -> bool:
        """Check if target exists in the current location."""
        # Check location description
        location_desc = getattr(location, "description", "").lower()
        if target in location_desc:
            return True

        # Check location objects
        objects = getattr(location, "objects", [])
        for obj in objects:
            obj_name = getattr(obj, "name", "").lower()
            if target in obj_name or obj_name in target:
                return True

        # Special handling for collection types mentioned in description
        collection_keywords = ["books", "shelves", "documents", "instruments", "files"]
        if target in collection_keywords:
            # Check if any collection keyword is mentioned in the location
            for keyword in collection_keywords:
                if (
                    keyword in location_desc or f"{keyword[:-1]}" in location_desc
                ):  # singular form
                    return True

        return False

    def can_handle_command(self, command: ParsedCommand) -> bool:
        """Check if this handler can process the given command."""
        action = command.action.lower()
        examination_verbs = ["examine", "inspect", "look at", "study", "check"]

        return action in examination_verbs

    def get_available_sub_objects(self, location_id: str) -> list[dict[str, Any]]:
        """Get all available sub-objects for a location."""
        available_objects = []

        for cache_key, sub_objects in self.generated_sub_objects.items():
            if location_id in cache_key:
                available_objects.extend(sub_objects)

        return available_objects
