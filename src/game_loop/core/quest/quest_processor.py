"""Quest interaction processor for handling quest-related actions."""

import time
from typing import Any

from game_loop.quests.models import (
    Quest,
    QuestCompletionResult,
    QuestInteractionResult,
    QuestInteractionType,
    QuestStatus,
    QuestUpdate,
)
from game_loop.quests.quest_manager import QuestManager
from game_loop.state.models import ActionResult


class QuestInteractionProcessor:
    """
    Process quest-related interactions and manage quest progression.

    Handles quest discovery, acceptance, progress tracking, and completion.
    Integrates with the existing object and action systems.
    """

    def __init__(self, quest_manager: QuestManager):
        self.quest_manager = quest_manager

    async def process_quest_interaction(
        self,
        interaction_type: QuestInteractionType,
        player_id: str,
        quest_context: dict[str, Any],
        game_state: dict[str, Any],
    ) -> QuestInteractionResult:
        """Process a quest interaction and return the result."""

        try:
            if interaction_type == QuestInteractionType.DISCOVER:
                return await self._process_discover_interaction(
                    player_id, quest_context, game_state
                )
            elif interaction_type == QuestInteractionType.ACCEPT:
                return await self._process_accept_interaction(
                    player_id, quest_context, game_state
                )
            elif interaction_type == QuestInteractionType.PROGRESS:
                return await self._process_progress_interaction(
                    player_id, quest_context, game_state
                )
            elif interaction_type == QuestInteractionType.COMPLETE:
                return await self._process_complete_interaction(
                    player_id, quest_context, game_state
                )
            elif interaction_type == QuestInteractionType.ABANDON:
                return await self._process_abandon_interaction(
                    player_id, quest_context, game_state
                )
            elif interaction_type == QuestInteractionType.QUERY:
                return await self._process_query_interaction(
                    player_id, quest_context, game_state
                )
            else:
                return QuestInteractionResult(
                    success=False,
                    message=f"Unknown interaction type: {interaction_type}",
                    errors=[f"Unsupported interaction type: {interaction_type.value}"],
                )

        except Exception as e:
            return QuestInteractionResult(
                success=False,
                message="An error occurred while processing quest interaction",
                errors=[str(e)],
            )

    async def discover_available_quests(
        self, player_id: str, location_id: str, context: dict[str, Any]
    ) -> list[Quest]:
        """Discover available quests for a player in a specific location."""
        # Get all available quests from repository
        available_quests = (
            await self.quest_manager.quest_repository.get_available_quests(
                player_id, location_id
            )
        )

        # Filter based on context (location, prerequisites, etc.)
        filtered_quests = []
        for quest in available_quests:
            # Check if player meets prerequisites
            is_valid, _ = await self.quest_manager.validate_quest_prerequisites(
                player_id, quest.quest_id, context.get("game_state")
            )

            if is_valid:
                filtered_quests.append(quest)

        return filtered_quests

    async def accept_quest(
        self, player_id: str, quest_id: str, context: dict[str, Any]
    ) -> tuple[bool, dict[str, Any]]:
        """Accept a quest and start tracking progress."""
        game_state = context.get("game_state")

        # Validate and start quest
        success, errors = await self.quest_manager.start_quest(
            player_id, quest_id, game_state
        )

        result_data = {
            "quest_id": quest_id,
            "accepted_at": time.time(),
            "errors": errors,
        }

        if success:
            # Get the quest details for response
            quest = await self.quest_manager.get_quest_by_id(quest_id)
            if quest:
                result_data["quest_title"] = quest.title
                result_data["quest_description"] = quest.description
                result_data["total_steps"] = quest.total_steps

        return success, result_data

    async def update_quest_progress(
        self,
        player_id: str,
        quest_id: str,
        action_result: ActionResult,
        context: dict[str, Any],
    ) -> list[QuestUpdate]:
        """Update quest progress based on an action result."""
        updates = []

        # Update the specific quest
        success = await self.quest_manager.update_quest_progress(
            player_id, quest_id, action_result
        )

        if success:
            # Get updated progress
            progress = await self.quest_manager.quest_repository.get_player_progress(
                player_id, quest_id
            )

            if progress:
                update = QuestUpdate(
                    quest_id=quest_id,
                    player_id=player_id,
                    update_type="progress_update",
                    update_data={
                        "current_step": progress.current_step,
                        "completed_steps": progress.completed_steps,
                        "status": progress.status.value,
                        "action_processed": {
                            "action_type": action_result.command,
                            "success": action_result.success,
                        },
                    },
                )
                updates.append(update)

        return updates

    async def complete_quest_step(
        self,
        player_id: str,
        quest_id: str,
        step_id: str,
        completion_data: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        """Complete a specific quest step."""
        # Get current progress
        progress = await self.quest_manager.quest_repository.get_player_progress(
            player_id, quest_id
        )

        if not progress or progress.status != QuestStatus.ACTIVE:
            return False, {"error": "Quest is not active"}

        # Mark step as completed
        progress.mark_step_complete(step_id)
        progress.update_step_progress(step_id, completion_data)

        # Save progress
        success = await self.quest_manager.quest_repository.update_progress(progress)

        result_data = {
            "step_id": step_id,
            "completed_at": time.time(),
            "completion_data": completion_data,
        }

        if success:
            # Check if quest is now complete
            quest = await self.quest_manager.get_quest_by_id(quest_id)
            if quest and len(progress.completed_steps) >= len(quest.required_steps):
                progress.status = QuestStatus.COMPLETED
                await self.quest_manager.quest_repository.update_progress(progress)
                result_data["quest_completed"] = True

                # Grant quest rewards
                if quest.rewards:
                    rewards = await self.quest_manager.grant_quest_rewards(
                        player_id, quest.rewards, {"quest_id": quest_id}
                    )
                    result_data["rewards"] = rewards

        return success, result_data

    async def complete_quest(
        self, player_id: str, quest_id: str, context: dict[str, Any]
    ) -> QuestCompletionResult:
        """Complete a quest and grant final rewards."""
        # Get quest and progress
        quest = await self.quest_manager.get_quest_by_id(quest_id)
        progress = await self.quest_manager.quest_repository.get_player_progress(
            player_id, quest_id
        )

        if not quest:
            return QuestCompletionResult(
                success=False,
                quest_id=quest_id,
                final_progress=None,
                errors=["Quest not found"],
            )

        if not progress or progress.status != QuestStatus.ACTIVE:
            return QuestCompletionResult(
                success=False,
                quest_id=quest_id,
                final_progress=progress,
                errors=["Quest is not active"],
            )

        # Check if all required steps are completed
        required_steps = quest.required_steps
        completed_required = [
            step.step_id
            for step in required_steps
            if step.step_id in progress.completed_steps
        ]

        if len(completed_required) < len(required_steps):
            return QuestCompletionResult(
                success=False,
                quest_id=quest_id,
                final_progress=progress,
                errors=[
                    f"Not all required steps completed: {len(completed_required)}/{len(required_steps)}"
                ],
            )

        # Complete the quest
        progress.status = QuestStatus.COMPLETED
        progress.updated_at = time.time()

        # Grant final rewards
        rewards_granted = {}
        if quest.rewards:
            rewards_granted = await self.quest_manager.grant_quest_rewards(
                player_id, quest.rewards, context
            )

        # Save final progress
        success = await self.quest_manager.quest_repository.update_progress(progress)

        completion_message = f"Quest '{quest.title}' completed!"
        if rewards_granted:
            reward_text = ", ".join([f"{k}: {v}" for k, v in rewards_granted.items()])
            completion_message += f" Rewards: {reward_text}"

        return QuestCompletionResult(
            success=success,
            quest_id=quest_id,
            final_progress=progress,
            rewards_granted=rewards_granted,
            completion_message=completion_message,
            errors=[] if success else ["Failed to save quest completion"],
        )

    async def _process_discover_interaction(
        self, player_id: str, quest_context: dict[str, Any], game_state: dict[str, Any]
    ) -> QuestInteractionResult:
        """Process quest discovery interaction."""
        location_id = quest_context.get("location_id", "")

        available_quests = await self.discover_available_quests(
            player_id, location_id, {"game_state": game_state}
        )

        if available_quests:
            quest_summaries = [
                {
                    "quest_id": q.quest_id,
                    "title": q.title,
                    "description": q.description,
                    "difficulty": q.difficulty.value,
                    "category": q.category.value,
                }
                for q in available_quests
            ]

            message = f"Discovered {len(available_quests)} available quest(s)"
            return QuestInteractionResult(
                success=True,
                message=message,
                quest_id=None,
                updated_progress=None,
                rewards_granted={"discovered_quests": quest_summaries},
            )
        else:
            return QuestInteractionResult(
                success=True,
                message="No available quests found in this location",
                quest_id=None,
            )

    async def _process_accept_interaction(
        self, player_id: str, quest_context: dict[str, Any], game_state: dict[str, Any]
    ) -> QuestInteractionResult:
        """Process quest acceptance interaction."""
        quest_id = quest_context.get("quest_id")
        if not quest_id:
            return QuestInteractionResult(
                success=False,
                message="Quest ID required for acceptance",
                errors=["No quest_id provided"],
            )

        success, result_data = await self.accept_quest(
            player_id, quest_id, {"game_state": game_state}
        )

        if success:
            message = f"Quest '{result_data.get('quest_title', quest_id)}' accepted!"
            return QuestInteractionResult(
                success=True,
                message=message,
                quest_id=quest_id,
                rewards_granted={"quest_data": result_data},
            )
        else:
            return QuestInteractionResult(
                success=False,
                message="Failed to accept quest",
                quest_id=quest_id,
                errors=result_data.get("errors", []),
            )

    async def _process_progress_interaction(
        self, player_id: str, quest_context: dict[str, Any], game_state: dict[str, Any]
    ) -> QuestInteractionResult:
        """Process quest progress update interaction."""
        quest_id = quest_context.get("quest_id")
        action_result = quest_context.get("action_result")

        if not quest_id or not action_result:
            return QuestInteractionResult(
                success=False,
                message="Quest ID and action result required for progress update",
                errors=["Missing quest_id or action_result"],
            )

        updates = await self.update_quest_progress(
            player_id, quest_id, action_result, quest_context
        )

        if updates:
            return QuestInteractionResult(
                success=True,
                message="Quest progress updated",
                quest_id=quest_id,
                rewards_granted={"updates": [u.__dict__ for u in updates]},
            )
        else:
            return QuestInteractionResult(
                success=False, message="No progress updates occurred", quest_id=quest_id
            )

    async def _process_complete_interaction(
        self, player_id: str, quest_context: dict[str, Any], game_state: dict[str, Any]
    ) -> QuestInteractionResult:
        """Process quest completion interaction."""
        quest_id = quest_context.get("quest_id")
        if not quest_id:
            return QuestInteractionResult(
                success=False,
                message="Quest ID required for completion",
                errors=["No quest_id provided"],
            )

        completion_result = await self.complete_quest(
            player_id, quest_id, quest_context
        )

        return QuestInteractionResult(
            success=completion_result.success,
            message=completion_result.completion_message
            or "Quest completion processed",
            quest_id=quest_id,
            updated_progress=completion_result.final_progress,
            rewards_granted=completion_result.rewards_granted,
            errors=completion_result.errors,
        )

    async def _process_abandon_interaction(
        self, player_id: str, quest_context: dict[str, Any], game_state: dict[str, Any]
    ) -> QuestInteractionResult:
        """Process quest abandonment interaction."""
        quest_id = quest_context.get("quest_id")
        if not quest_id:
            return QuestInteractionResult(
                success=False,
                message="Quest ID required for abandonment",
                errors=["No quest_id provided"],
            )

        success = await self.quest_manager.abandon_quest(player_id, quest_id)

        if success:
            return QuestInteractionResult(
                success=True, message=f"Quest {quest_id} abandoned", quest_id=quest_id
            )
        else:
            return QuestInteractionResult(
                success=False,
                message="Failed to abandon quest",
                quest_id=quest_id,
                errors=["Quest not found or not active"],
            )

    async def _process_query_interaction(
        self, player_id: str, quest_context: dict[str, Any], game_state: dict[str, Any]
    ) -> QuestInteractionResult:
        """Process quest query interaction."""
        quest_id = quest_context.get("quest_id")

        if quest_id:
            # Query specific quest
            quest = await self.quest_manager.get_quest_by_id(quest_id)
            progress = await self.quest_manager.quest_repository.get_player_progress(
                player_id, quest_id
            )

            if quest:
                quest_data = {
                    "quest": {
                        "quest_id": quest.quest_id,
                        "title": quest.title,
                        "description": quest.description,
                        "difficulty": quest.difficulty.value,
                        "category": quest.category.value,
                        "total_steps": quest.total_steps,
                    },
                    "progress": (
                        {
                            "status": (
                                progress.status.value if progress else "not_started"
                            ),
                            "current_step": progress.current_step if progress else 0,
                            "completed_steps": (
                                progress.completed_steps if progress else []
                            ),
                            "completion_percentage": (
                                progress.completion_percentage if progress else 0.0
                            ),
                        }
                        if progress
                        else None
                    ),
                }

                return QuestInteractionResult(
                    success=True,
                    message=f"Quest information for {quest.title}",
                    quest_id=quest_id,
                    rewards_granted={"quest_info": quest_data},
                )
            else:
                return QuestInteractionResult(
                    success=False,
                    message="Quest not found",
                    quest_id=quest_id,
                    errors=["Quest not found"],
                )
        else:
            # Query all active quests
            active_quests = await self.quest_manager.get_player_active_quests(player_id)

            if active_quests:
                quest_summaries = []
                for progress in active_quests:
                    quest = await self.quest_manager.get_quest_by_id(progress.quest_id)
                    if quest:
                        quest_summaries.append(
                            {
                                "quest_id": quest.quest_id,
                                "title": quest.title,
                                "status": progress.status.value,
                                "current_step": progress.current_step,
                                "total_steps": quest.total_steps,
                                "completion_percentage": progress.completion_percentage,
                            }
                        )

                return QuestInteractionResult(
                    success=True,
                    message=f"You have {len(active_quests)} active quest(s)",
                    rewards_granted={"active_quests": quest_summaries},
                )
            else:
                return QuestInteractionResult(
                    success=True,
                    message="You have no active quests",
                    rewards_granted={"active_quests": []},
                )
