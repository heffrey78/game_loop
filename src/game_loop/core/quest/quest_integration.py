"""Quest integration with object systems."""

from typing import Any

from game_loop.quests.models import Quest, QuestProgress, QuestStep, QuestUpdate
from game_loop.quests.quest_manager import QuestManager
from game_loop.state.models import ActionResult


class QuestObjectIntegration:
    """
    Integration layer between quest system and object systems.

    Handles quest triggers from object interactions, inventory changes,
    location visits, and other game events.
    """

    def __init__(self, quest_manager: QuestManager):
        self.quest_manager = quest_manager
        self._trigger_handlers = {
            "location_visit": self._handle_location_trigger,
            "object_interaction": self._handle_object_trigger,
            "inventory_change": self._handle_inventory_trigger,
            "combat_action": self._handle_combat_trigger,
            "crafting_action": self._handle_crafting_trigger,
            "social_interaction": self._handle_social_trigger,
        }

    async def process_action_for_quests(
        self, player_id: str, action_result: ActionResult, context: dict[str, Any]
    ) -> list[QuestUpdate]:
        """Process an action result for all active quests."""
        updates = []

        # Get all active quests for the player
        active_quests = await self.quest_manager.get_player_active_quests(player_id)

        # Process each active quest
        for progress in active_quests:
            quest_updated = await self.quest_manager.update_quest_progress(
                player_id, progress.quest_id, action_result
            )

            if quest_updated:
                update = QuestUpdate(
                    quest_id=progress.quest_id,
                    player_id=player_id,
                    update_type="action_processed",
                    update_data={
                        "action_type": action_result.command,
                        "action_success": action_result.success,
                        "target_object": getattr(action_result, "target_object", None),
                        "context": context,
                    },
                )
                updates.append(update)

        return updates

    async def check_quest_triggers(
        self, player_id: str, trigger_type: str, trigger_data: dict[str, Any]
    ) -> list[Quest]:
        """Check if any new quests should be triggered by an event."""
        triggered_quests: list[Quest] = []

        # Get trigger handler
        handler = self._trigger_handlers.get(trigger_type)
        if not handler:
            return triggered_quests

        # Get available quests that could be triggered
        location_id = trigger_data.get("location_id")
        available_quests = (
            await self.quest_manager.quest_repository.get_available_quests(
                player_id, location_id
            )
        )

        # Check each quest for triggers
        for quest in available_quests:
            if await handler(quest, trigger_data, player_id):
                triggered_quests.append(quest)

        return triggered_quests

    async def update_quest_objectives(
        self, player_id: str, objective_type: str, objective_data: dict[str, Any]
    ) -> list[QuestProgress]:
        """Update quest objectives based on game events."""
        updated_progress = []

        # Get active quests
        active_quests = await self.quest_manager.get_player_active_quests(player_id)

        for progress in active_quests:
            quest = await self.quest_manager.get_quest_by_id(progress.quest_id)
            if not quest:
                continue

            # Check if current step objectives match this event
            if progress.current_step < len(quest.steps):
                current_step = quest.steps[progress.current_step]

                if await self._check_objective_match(
                    current_step, objective_type, objective_data
                ):
                    # Update step progress
                    progress.update_step_progress(
                        current_step.step_id,
                        {
                            "objective_type": objective_type,
                            "objective_data": objective_data,
                            "updated_at": objective_data.get("timestamp"),
                        },
                    )

                    # Save progress
                    await self.quest_manager.quest_repository.update_progress(progress)
                    updated_progress.append(progress)

        return updated_progress

    async def _handle_location_trigger(
        self, quest: Quest, trigger_data: dict[str, Any], player_id: str
    ) -> bool:
        """Check if quest should be triggered by location visit."""
        location_id = trigger_data.get("location_id")

        # Check quest metadata for location triggers
        for step in quest.steps:
            requirements = step.requirements
            if requirements.get("trigger_location") == location_id:
                return True

        # Check if this is an exploration quest for this location
        if quest.category.value == "exploration" and any(
            "location" in step.requirements for step in quest.steps
        ):
            return True

        return False

    async def _handle_object_trigger(
        self, quest: Quest, trigger_data: dict[str, Any], player_id: str
    ) -> bool:
        """Check if quest should be triggered by object interaction."""
        object_id = trigger_data.get("object_id")
        interaction_type = trigger_data.get("interaction_type")

        # Check quest steps for object triggers
        for step in quest.steps:
            requirements = step.requirements
            if (
                requirements.get("trigger_object") == object_id
                and requirements.get("trigger_action") == interaction_type
            ):
                return True

        return False

    async def _handle_inventory_trigger(
        self, quest: Quest, trigger_data: dict[str, Any], player_id: str
    ) -> bool:
        """Check if quest should be triggered by inventory changes."""
        item_id = trigger_data.get("item_id")
        change_type = trigger_data.get("change_type")  # "added", "removed", "used"

        # Check for collection quests triggered by obtaining items
        if (
            quest.category.value == "collection"
            and change_type == "added"
            and any(
                item_id in step.requirements.get("items", []) for step in quest.steps
            )
        ):
            return True

        return False

    async def _handle_combat_trigger(
        self, quest: Quest, trigger_data: dict[str, Any], player_id: str
    ) -> bool:
        """Check if quest should be triggered by combat actions."""
        enemy_type = trigger_data.get("enemy_type")
        combat_result = trigger_data.get("result")  # "victory", "defeat", "fled"

        # Check for combat quests
        if (
            quest.category.value == "combat"
            and combat_result == "victory"
            and any(
                enemy_type in step.requirements.get("enemies", [])
                for step in quest.steps
            )
        ):
            return True

        return False

    async def _handle_crafting_trigger(
        self, quest: Quest, trigger_data: dict[str, Any], player_id: str
    ) -> bool:
        """Check if quest should be triggered by crafting actions."""
        crafted_item = trigger_data.get("crafted_item")
        crafting_success = trigger_data.get("success", False)

        # Check for crafting quests
        if (
            quest.category.value == "crafting"
            and crafting_success
            and any(
                crafted_item in step.requirements.get("craft_items", [])
                for step in quest.steps
            )
        ):
            return True

        return False

    async def _handle_social_trigger(
        self, quest: Quest, trigger_data: dict[str, Any], player_id: str
    ) -> bool:
        """Check if quest should be triggered by social interactions."""
        npc_id = trigger_data.get("npc_id")
        interaction_type = trigger_data.get(
            "interaction_type"
        )  # "talk", "trade", "help"

        # Check for social quests
        if quest.category.value == "social" and any(
            npc_id == step.requirements.get("npc_id")
            and interaction_type in step.requirements.get("interactions", [])
            for step in quest.steps
        ):
            return True

        return False

    async def _check_objective_match(
        self, quest_step: QuestStep, objective_type: str, objective_data: dict[str, Any]
    ) -> bool:
        """Check if an objective matches a quest step."""
        step_requirements = quest_step.requirements

        if objective_type == "location_visit":
            required_location = step_requirements.get("location")
            visited_location = objective_data.get("location_id")
            return required_location == visited_location

        elif objective_type == "item_collection":
            required_items = step_requirements.get("items", [])
            collected_item = objective_data.get("item_id")
            return collected_item in required_items

        elif objective_type == "object_interaction":
            # Check both possible field names for compatibility
            required_object = step_requirements.get(
                "target_object"
            ) or step_requirements.get("target")
            required_action = step_requirements.get(
                "action_type"
            ) or step_requirements.get("action")
            interacted_object = objective_data.get("object_id")
            action_type = objective_data.get("action_type")
            return (
                required_object == interacted_object and required_action == action_type
            )

        elif objective_type == "combat_victory":
            required_enemies = step_requirements.get("enemies", [])
            defeated_enemy = objective_data.get("enemy_type")
            return defeated_enemy in required_enemies

        elif objective_type == "crafting_completion":
            required_items = step_requirements.get("craft_items", [])
            crafted_item = objective_data.get("item_id")
            return crafted_item in required_items

        elif objective_type == "social_interaction":
            required_npc = step_requirements.get("npc_id")
            required_interaction = step_requirements.get("interaction_type")
            target_npc = objective_data.get("npc_id")
            interaction_type = objective_data.get("interaction_type")
            return (
                required_npc == target_npc and required_interaction == interaction_type
            )

        return False

    async def get_quest_hints_for_location(
        self, player_id: str, location_id: str
    ) -> list[dict[str, Any]]:
        """Get quest hints and objectives for the current location."""
        hints = []

        # Get active quests
        active_quests = await self.quest_manager.get_player_active_quests(player_id)

        for progress in active_quests:
            quest = await self.quest_manager.get_quest_by_id(progress.quest_id)
            if not quest or progress.current_step >= len(quest.steps):
                continue

            current_step = quest.steps[progress.current_step]
            step_requirements = current_step.requirements

            # Check if current step involves this location
            if step_requirements.get("location") == location_id:
                hint = {
                    "quest_id": quest.quest_id,
                    "quest_title": quest.title,
                    "step_description": current_step.description,
                    "objective_type": "location_objective",
                    "hint_text": f"This location is relevant to the quest '{quest.title}'",
                }
                hints.append(hint)

        return hints

    async def get_available_quest_actions(
        self, player_id: str, context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Get available quest-related actions in the current context."""
        actions = []

        # Get active quests
        active_quests = await self.quest_manager.get_player_active_quests(player_id)

        for progress in active_quests:
            quest = await self.quest_manager.get_quest_by_id(progress.quest_id)
            if not quest or progress.current_step >= len(quest.steps):
                continue

            current_step = quest.steps[progress.current_step]

            # Generate context-appropriate actions
            for condition in current_step.completion_conditions:
                if ":" in condition:
                    condition_type, condition_value = condition.split(":", 1)

                    if condition_type == "action_type":
                        action = {
                            "quest_id": quest.quest_id,
                            "action_type": condition_value,
                            "step_id": current_step.step_id,
                            "description": f"Perform {condition_value} action for quest '{quest.title}'",
                        }
                        actions.append(action)

        return actions
