"""Quest manager for coordinating quest system operations."""

import time
import uuid
from typing import Any

from game_loop.database.repositories.quest import QuestRepository
from game_loop.quests.models import (
    Quest,
    QuestCategory,
    QuestDifficulty,
    QuestProgress,
    QuestStatus,
    QuestStep,
)
from game_loop.state.models import ActionResult


class QuestManager:
    """
    Central manager for quest system operations.

    Coordinates quest data, progress tracking, and integration
    with other game systems.
    """

    def __init__(self, quest_repository: QuestRepository):
        self.quest_repository = quest_repository
        self._cache: dict[str, Quest] = {}
        self._progress_cache: dict[str, QuestProgress] = {}

    async def get_player_active_quests(self, player_id: str) -> list[QuestProgress]:
        """Get all active quests for a player."""
        return await self.quest_repository.get_active_quests(player_id)

    async def get_quest_by_id(self, quest_id: str) -> Quest | None:
        """Get a quest by its ID, using cache when possible."""
        # Check cache first
        if quest_id in self._cache:
            return self._cache[quest_id]

        # Load from database
        quest = await self.quest_repository.get_quest(quest_id)
        if quest:
            self._cache[quest_id] = quest

        return quest

    async def validate_quest_prerequisites(
        self, player_id: str, quest_id: str, game_state: dict[str, Any] | None = None
    ) -> tuple[bool, list[str]]:
        """Validate if player meets quest prerequisites."""
        errors = []

        quest = await self.get_quest_by_id(quest_id)
        if not quest:
            errors.append(f"Quest {quest_id} not found")
            return False, errors

        # Check prerequisite quests
        for prereq_quest_id in quest.prerequisites:
            progress = await self.quest_repository.get_player_progress(
                player_id, prereq_quest_id
            )

            if not progress or progress.status != QuestStatus.COMPLETED:
                errors.append(f"Must complete quest '{prereq_quest_id}' first")

        # Check if quest is already active or completed
        current_progress = await self.quest_repository.get_player_progress(
            player_id, quest_id
        )

        if current_progress:
            if current_progress.status == QuestStatus.ACTIVE:
                errors.append("Quest is already active")
            elif (
                current_progress.status == QuestStatus.COMPLETED
                and not quest.repeatable
            ):
                errors.append("Quest is already completed and not repeatable")
            # For repeatable quests that are completed, we allow re-acceptance

        # Additional validation based on game state could go here
        # For example: player level, location, inventory items, etc.

        return len(errors) == 0, errors

    async def check_step_completion_conditions(
        self, player_id: str, quest_id: str, step_id: str, action_result: ActionResult
    ) -> bool:
        """Check if a quest step's completion conditions are met."""
        quest = await self.get_quest_by_id(quest_id)
        if not quest:
            return False

        # Find the specific step
        target_step = None
        for step in quest.steps:
            if step.step_id == step_id:
                target_step = step
                break

        if not target_step:
            return False

        # Check each completion condition
        for condition in target_step.completion_conditions:
            if not await self._evaluate_condition(
                condition, action_result, player_id, quest_id
            ):
                return False

        return True

    async def grant_quest_rewards(
        self, player_id: str, rewards: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Grant quest rewards to the player."""
        granted_rewards = {}

        for reward_type, reward_value in rewards.items():
            if reward_type == "experience":
                # Integration point: would call player state manager
                granted_rewards["experience"] = reward_value

            elif reward_type == "gold":
                # Integration point: would call inventory/currency manager
                granted_rewards["gold"] = reward_value

            elif reward_type == "items":
                # Integration point: would call inventory manager
                granted_rewards["items"] = reward_value

            elif reward_type == "skills":
                # Integration point: would call skill manager
                granted_rewards["skills"] = reward_value

            else:
                # Custom reward handling
                granted_rewards[reward_type] = reward_value

        # Log the reward granting
        await self.quest_repository.log_interaction(
            context.get("quest_id", ""),
            player_id,
            "reward_grant",
            {"rewards": granted_rewards, "context": context},
        )

        return granted_rewards

    async def generate_dynamic_quest(
        self, player_id: str, context: dict[str, Any], quest_type: str
    ) -> Quest | None:
        """Generate a dynamic quest based on context."""
        # This would typically use templates and LLM generation
        # For now, return a simple template-based quest

        quest_templates = {
            "delivery": self._create_delivery_quest,
            "exploration": self._create_exploration_quest,
            "collection": self._create_collection_quest,
        }

        if quest_type not in quest_templates:
            return None

        return await quest_templates[quest_type](player_id, context)

    async def start_quest(
        self, player_id: str, quest_id: str, game_state: dict[str, Any] | None = None
    ) -> tuple[bool, list[str]]:
        """Start a quest for a player."""
        # Validate prerequisites
        is_valid, errors = await self.validate_quest_prerequisites(
            player_id, quest_id, game_state
        )

        if not is_valid:
            return False, errors

        # Check if this is a restart of a repeatable quest
        existing_progress = await self.quest_repository.get_player_progress(
            player_id, quest_id
        )

        if existing_progress and existing_progress.status == QuestStatus.COMPLETED:
            quest = await self.get_quest_by_id(quest_id)
            if quest and quest.repeatable:
                # Reset progress for repeatable quest
                existing_progress.status = QuestStatus.ACTIVE
                existing_progress.current_step = 0
                existing_progress.completed_steps = []
                existing_progress.step_progress = {}
                existing_progress.started_at = time.time()
                existing_progress.updated_at = time.time()
                progress = existing_progress
            else:
                return False, ["Quest is not repeatable"]
        else:
            # Create initial progress
            progress = QuestProgress(
                quest_id=quest_id,
                player_id=player_id,
                status=QuestStatus.ACTIVE,
                current_step=0,
                completed_steps=[],
                step_progress={},
                started_at=time.time(),
                updated_at=time.time(),
            )

        # Save progress
        success = await self.quest_repository.update_progress(progress)
        if not success:
            return False, ["Failed to save quest progress"]

        # Log interaction
        await self.quest_repository.log_interaction(
            quest_id, player_id, "accept", {"timestamp": time.time()}
        )

        return True, []

    async def update_quest_progress(
        self, player_id: str, quest_id: str, action_result: ActionResult
    ) -> bool:
        """Update quest progress based on an action result."""
        progress = await self.quest_repository.get_player_progress(player_id, quest_id)

        if not progress or progress.status != QuestStatus.ACTIVE:
            return False

        quest = await self.get_quest_by_id(quest_id)
        if not quest or progress.current_step >= len(quest.steps):
            return False

        current_step = quest.steps[progress.current_step]

        # Check if current step is completed
        if await self.check_step_completion_conditions(
            player_id, quest_id, current_step.step_id, action_result
        ):
            # Mark step as completed
            progress.mark_step_complete(current_step.step_id)

            # Grant step rewards if any
            if current_step.rewards:
                await self.grant_quest_rewards(
                    player_id,
                    current_step.rewards,
                    {"quest_id": quest_id, "step_id": current_step.step_id},
                )

            # Check if this was the final step
            if progress.current_step >= len(quest.steps) - 1:
                # Quest completed
                progress.status = QuestStatus.COMPLETED

                # Grant final quest rewards
                if quest.rewards:
                    await self.grant_quest_rewards(
                        player_id, quest.rewards, {"quest_id": quest_id}
                    )

                # Log completion
                await self.quest_repository.log_interaction(
                    quest_id, player_id, "complete", {"timestamp": time.time()}
                )
            else:
                # Advance to next step
                progress.advance_to_next_step()

            # Save updated progress
            success = await self.quest_repository.update_progress(progress)
            return success

        return False

    async def abandon_quest(self, player_id: str, quest_id: str) -> bool:
        """Abandon an active quest."""
        progress = await self.quest_repository.get_player_progress(player_id, quest_id)

        if not progress or progress.status != QuestStatus.ACTIVE:
            return False

        progress.status = QuestStatus.ABANDONED
        progress.updated_at = time.time()

        # Log abandonment
        await self.quest_repository.log_interaction(
            quest_id, player_id, "abandon", {"timestamp": time.time()}
        )

        return await self.quest_repository.update_progress(progress)

    async def _evaluate_condition(
        self, condition: str, action_result: ActionResult, player_id: str, quest_id: str
    ) -> bool:
        """Evaluate a completion condition."""
        # Simple condition evaluation - would be more sophisticated in practice
        # Format: "action_type:target_object" or "location:location_id" etc.

        if ":" not in condition:
            return False

        condition_type, condition_value = condition.split(":", 1)

        if condition_type == "action_type":
            # Since ActionResult doesn't have action_type, we'll use command field
            return (
                action_result.command == condition_value
                if action_result.command
                else False
            )
        elif condition_type == "target_object":
            # Check if the target object matches from processed_input
            if (
                action_result.processed_input
                and "target" in action_result.processed_input
            ):
                return action_result.processed_input["target"] == condition_value
            return False
        elif condition_type == "location":
            # Would check current player location from new_location_id or context
            return (
                action_result.new_location_id is not None if condition_value else True
            )
        elif condition_type == "item_obtained":
            # Check inventory changes for item acquisition
            if action_result.inventory_changes:
                for change in action_result.inventory_changes:
                    if (
                        change.get("action") == "add"
                        and change.get("item") == condition_value
                    ):
                        return True
            return False

        return False

    async def _create_delivery_quest(
        self, player_id: str, context: dict[str, Any]
    ) -> Quest:
        """Create a delivery quest template."""
        quest_id = f"delivery_{uuid.uuid4().hex[:8]}"

        steps = [
            QuestStep(
                step_id="pickup",
                description="Pick up the package",
                requirements={
                    "location": context.get("pickup_location", "town_center")
                },
                completion_conditions=["action_type:take"],
                rewards={},
            ),
            QuestStep(
                step_id="delivery",
                description="Deliver the package",
                requirements={"location": context.get("delivery_location", "castle")},
                completion_conditions=["action_type:give"],
                rewards={},
            ),
        ]

        return Quest(
            quest_id=quest_id,
            title="Package Delivery",
            description="Deliver a package to its destination",
            category=QuestCategory.DELIVERY,
            difficulty=QuestDifficulty.EASY,
            steps=steps,
            rewards={"experience": 100, "gold": 50},
        )

    async def _create_exploration_quest(
        self, player_id: str, context: dict[str, Any]
    ) -> Quest:
        """Create an exploration quest template."""
        quest_id = f"explore_{uuid.uuid4().hex[:8]}"

        steps = [
            QuestStep(
                step_id="visit_location",
                description="Visit the mysterious location",
                requirements={"location": context.get("target_location", "forest")},
                completion_conditions=["action_type:look"],
                rewards={},
            )
        ]

        return Quest(
            quest_id=quest_id,
            title="Exploration Mission",
            description="Explore a new location",
            category=QuestCategory.EXPLORATION,
            difficulty=QuestDifficulty.MEDIUM,
            steps=steps,
            rewards={"experience": 150},
        )

    async def _create_collection_quest(
        self, player_id: str, context: dict[str, Any]
    ) -> Quest:
        """Create a collection quest template."""
        quest_id = f"collect_{uuid.uuid4().hex[:8]}"

        steps = [
            QuestStep(
                step_id="collect_items",
                description="Collect the required items",
                requirements={"items": context.get("required_items", ["herb"])},
                completion_conditions=["item_obtained:herb"],
                rewards={},
            )
        ]

        return Quest(
            quest_id=quest_id,
            title="Collection Task",
            description="Collect specific items",
            category=QuestCategory.COLLECTION,
            difficulty=QuestDifficulty.EASY,
            steps=steps,
            rewards={"experience": 75, "gold": 25},
        )
