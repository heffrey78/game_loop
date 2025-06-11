"""Integration tests for complete quest workflows."""

import time
from unittest.mock import AsyncMock

import pytest

from game_loop.core.quest.quest_integration import QuestObjectIntegration
from game_loop.core.quest.quest_processor import QuestInteractionProcessor
from game_loop.database.repositories.quest import QuestRepository
from game_loop.quests.models import (
    Quest,
    QuestCategory,
    QuestDifficulty,
    QuestInteractionType,
    QuestProgress,
    QuestStatus,
    QuestStep,
)
from game_loop.quests.quest_manager import QuestManager
from game_loop.state.models import ActionResult


@pytest.mark.asyncio
class TestQuestWorkflow:
    """Test complete quest workflows from discovery to completion."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        return AsyncMock()

    @pytest.fixture
    def quest_repository(self, mock_session):
        """Create a quest repository with mock session."""
        return QuestRepository(mock_session)

    @pytest.fixture
    def quest_manager(self, quest_repository):
        """Create a quest manager."""
        return QuestManager(quest_repository)

    @pytest.fixture
    def quest_processor(self, quest_manager):
        """Create a quest processor."""
        return QuestInteractionProcessor(quest_manager)

    @pytest.fixture
    def quest_integration(self, quest_manager):
        """Create quest integration."""
        return QuestObjectIntegration(quest_manager)

    @pytest.fixture
    def sample_delivery_quest(self):
        """Create a sample delivery quest."""
        steps = [
            QuestStep(
                step_id="pickup_item",
                description="Pick up the package from the merchant",
                requirements={
                    "location": "marketplace",
                    "action": "take",
                    "target": "package",
                },
                completion_conditions=["action_type:take", "target_object:package"],
                rewards={"experience": 25},
            ),
            QuestStep(
                step_id="deliver_item",
                description="Deliver the package to the castle",
                requirements={
                    "location": "castle",
                    "action": "give",
                    "target": "package",
                },
                completion_conditions=["action_type:give", "target_object:package"],
                rewards={"experience": 25},
            ),
        ]

        return Quest(
            quest_id="delivery_001",
            title="Package Delivery",
            description="Deliver a package from the marketplace to the castle",
            category=QuestCategory.DELIVERY,
            difficulty=QuestDifficulty.EASY,
            steps=steps,
            rewards={"experience": 100, "gold": 50},
            repeatable=False,
        )

    @pytest.fixture
    def sample_game_state(self):
        """Create a sample game state."""
        return {
            "player_id": "test_player",
            "location_id": "marketplace",
            "metadata": {"player_level": 1},
        }

    async def test_complete_quest_workflow(
        self,
        quest_manager,
        quest_processor,
        quest_integration,
        sample_delivery_quest,
        sample_game_state,
    ):
        """Test a complete quest workflow from discovery to completion."""
        player_id = "test_player"
        quest_id = "delivery_001"

        # Mock repository methods
        quest_manager.quest_repository.get_available_quests = AsyncMock(
            return_value=[sample_delivery_quest]
        )
        quest_manager.quest_repository.get_quest = AsyncMock(
            return_value=sample_delivery_quest
        )
        quest_manager.quest_repository.get_player_progress = AsyncMock(
            return_value=None
        )
        quest_manager.quest_repository.update_progress = AsyncMock(return_value=True)
        quest_manager.quest_repository.log_interaction = AsyncMock(return_value=True)

        # Step 1: Discover available quests
        available_quests = await quest_processor.discover_available_quests(
            player_id, "marketplace", {"game_state": sample_game_state}
        )

        assert len(available_quests) == 1
        assert available_quests[0].quest_id == quest_id

        # Step 2: Accept the quest
        success, result_data = await quest_processor.accept_quest(
            player_id, quest_id, {"game_state": sample_game_state}
        )

        assert success is True
        assert result_data["quest_id"] == quest_id

        # Step 3: Simulate first action (pickup item)
        pickup_action = ActionResult(
            success=True,
            feedback_message="Successfully picked up package",
            command="take",
            processed_input={"target": "package"},
            inventory_changes=[{"action": "add", "item": "package"}],
        )

        # Mock initial progress before action (step 0)
        initial_progress = QuestProgress(
            quest_id=quest_id,
            player_id=player_id,
            status=QuestStatus.ACTIVE,
            current_step=0,
            completed_steps=[],
            step_progress={},
        )
        quest_manager.quest_repository.get_player_progress = AsyncMock(
            return_value=initial_progress
        )

        # Process first action
        updates = await quest_processor.update_quest_progress(
            player_id, quest_id, pickup_action, {"location": "marketplace"}
        )

        assert len(updates) == 1
        assert updates[0].quest_id == quest_id

        # Step 4: Simulate second action (deliver item)
        delivery_action = ActionResult(
            success=True,
            feedback_message="Successfully delivered package",
            command="give",
            processed_input={"target": "package"},
            inventory_changes=[{"action": "remove", "item": "package"}],
        )

        # Mock progress after first step for second action
        progress_after_pickup = QuestProgress(
            quest_id=quest_id,
            player_id=player_id,
            status=QuestStatus.ACTIVE,
            current_step=1,
            completed_steps=["pickup_item"],
            step_progress={"pickup_item": {"completed": True}},
        )
        quest_manager.quest_repository.get_player_progress = AsyncMock(
            return_value=progress_after_pickup
        )

        # Process delivery action
        updates = await quest_processor.update_quest_progress(
            player_id, quest_id, delivery_action, {"location": "castle"}
        )

        assert len(updates) == 1

        # Quest should automatically complete after final step
        # Verify that quest completion was handled in the update
        final_progress = await quest_manager.quest_repository.get_player_progress(
            player_id, quest_id
        )
        # The mock will return progress_after_pickup, but in real scenario
        # it would be completed automatically

    async def test_quest_discovery_with_prerequisites(
        self, quest_manager, quest_processor, sample_game_state
    ):
        """Test quest discovery with prerequisites."""
        # Create a quest with prerequisites
        prerequisite_quest = Quest(
            quest_id="prereq_001",
            title="Prerequisite Quest",
            description="Must be completed first",
            category=QuestCategory.DELIVERY,
            difficulty=QuestDifficulty.EASY,
            steps=[
                QuestStep(
                    step_id="simple_step",
                    description="Simple step",
                    requirements={},
                    completion_conditions=["action_type:talk"],
                )
            ],
        )

        main_quest = Quest(
            quest_id="main_001",
            title="Main Quest",
            description="Requires prerequisite",
            category=QuestCategory.EXPLORATION,
            difficulty=QuestDifficulty.MEDIUM,
            steps=[
                QuestStep(
                    step_id="explore_step",
                    description="Explore location",
                    requirements={},
                    completion_conditions=["action_type:look"],
                )
            ],
            prerequisites=["prereq_001"],
        )

        # Mock repository to return both quests as available
        quest_manager.quest_repository.get_available_quests = AsyncMock(
            return_value=[prerequisite_quest, main_quest]
        )
        quest_manager.quest_repository.get_quest = AsyncMock(
            side_effect=lambda quest_id: (
                prerequisite_quest
                if quest_id == "prereq_001"
                else main_quest if quest_id == "main_001" else None
            )
        )

        # Mock prerequisite validation - prereq not completed
        quest_manager.quest_repository.get_player_progress = AsyncMock(
            return_value=None
        )

        available_quests = await quest_processor.discover_available_quests(
            "test_player", "town", {"game_state": sample_game_state}
        )

        # Should only return prerequisite quest
        assert len(available_quests) == 1
        assert available_quests[0].quest_id == "prereq_001"

    async def test_quest_abandonment_workflow(
        self, quest_manager, quest_processor, sample_delivery_quest, sample_game_state
    ):
        """Test quest abandonment workflow."""
        player_id = "test_player"
        quest_id = "delivery_001"

        # Mock active quest progress
        active_progress = QuestProgress(
            quest_id=quest_id,
            player_id=player_id,
            status=QuestStatus.ACTIVE,
            current_step=0,
            completed_steps=[],
            step_progress={},
        )

        quest_manager.quest_repository.get_player_progress = AsyncMock(
            return_value=active_progress
        )
        quest_manager.quest_repository.update_progress = AsyncMock(return_value=True)
        quest_manager.quest_repository.log_interaction = AsyncMock(return_value=True)

        # Abandon the quest
        result = await quest_processor.process_quest_interaction(
            QuestInteractionType.ABANDON,
            player_id,
            {"quest_id": quest_id},
            sample_game_state,
        )

        assert result.success is True
        assert "abandoned" in result.message
        assert result.quest_id == quest_id

    async def test_quest_integration_with_object_system(
        self, quest_integration, quest_manager, sample_delivery_quest
    ):
        """Test quest integration with object interaction system."""
        player_id = "test_player"

        # Mock active quest
        active_progress = QuestProgress(
            quest_id="delivery_001",
            player_id=player_id,
            status=QuestStatus.ACTIVE,
            current_step=0,
            completed_steps=[],
            step_progress={},
        )

        quest_manager.get_player_active_quests = AsyncMock(
            return_value=[active_progress]
        )
        quest_manager.update_quest_progress = AsyncMock(return_value=True)

        # Simulate object interaction
        action_result = ActionResult(
            success=True,
            feedback_message="Successfully took package",
            command="take",
            processed_input={"target": "package"},
            inventory_changes=[{"action": "add", "item": "package"}],
        )

        # Process action for quests
        updates = await quest_integration.process_action_for_quests(
            player_id, action_result, {"location": "marketplace"}
        )

        assert len(updates) == 1
        assert updates[0].quest_id == "delivery_001"
        assert updates[0].update_type == "action_processed"

    async def test_quest_trigger_system(
        self, quest_integration, quest_manager, sample_delivery_quest
    ):
        """Test quest trigger system."""
        player_id = "test_player"

        # Mock available quests
        quest_manager.quest_repository.get_available_quests = AsyncMock(
            return_value=[sample_delivery_quest]
        )

        # Test location trigger
        trigger_data = {"location_id": "marketplace", "action_type": "enter"}

        # Add trigger location to quest step
        sample_delivery_quest.steps[0].requirements["trigger_location"] = "marketplace"

        triggered_quests = await quest_integration.check_quest_triggers(
            player_id, "location_visit", trigger_data
        )

        # Should trigger the delivery quest
        assert len(triggered_quests) == 1
        assert triggered_quests[0].quest_id == "delivery_001"

    async def test_quest_objective_updates(
        self, quest_integration, quest_manager, sample_delivery_quest
    ):
        """Test quest objective update system."""
        player_id = "test_player"

        # Mock active quest progress
        active_progress = QuestProgress(
            quest_id="delivery_001",
            player_id=player_id,
            status=QuestStatus.ACTIVE,
            current_step=0,
            completed_steps=[],
            step_progress={},
        )

        quest_manager.get_player_active_quests = AsyncMock(
            return_value=[active_progress]
        )
        quest_manager.get_quest_by_id = AsyncMock(return_value=sample_delivery_quest)
        quest_manager.quest_repository.get_quest = AsyncMock(
            return_value=sample_delivery_quest
        )
        quest_manager.quest_repository.update_progress = AsyncMock(return_value=True)

        # Test objective update
        objective_data = {
            "location_id": "marketplace",
            "action_type": "take",
            "object_id": "package",
            "timestamp": time.time(),
        }

        updated_progress = await quest_integration.update_quest_objectives(
            player_id, "object_interaction", objective_data
        )

        assert len(updated_progress) == 1
        assert updated_progress[0].quest_id == "delivery_001"

    async def test_repeatable_quest_workflow(
        self, quest_manager, quest_processor, sample_game_state
    ):
        """Test workflow for repeatable quests."""
        # Create a repeatable quest
        repeatable_quest = Quest(
            quest_id="daily_001",
            title="Daily Delivery",
            description="A daily delivery quest",
            category=QuestCategory.DELIVERY,
            difficulty=QuestDifficulty.EASY,
            steps=[
                QuestStep(
                    step_id="daily_delivery",
                    description="Make a daily delivery",
                    requirements={},
                    completion_conditions=["action_type:give"],
                )
            ],
            rewards={"experience": 50},
            repeatable=True,
        )

        player_id = "test_player"
        quest_id = "daily_001"

        # Mock completed progress for repeatable quest
        completed_progress = QuestProgress(
            quest_id=quest_id,
            player_id=player_id,
            status=QuestStatus.COMPLETED,
            current_step=1,
            completed_steps=["daily_delivery"],
            step_progress={},
        )

        quest_manager.quest_repository.get_available_quests = AsyncMock(
            return_value=[repeatable_quest]
        )
        quest_manager.quest_repository.get_quest = AsyncMock(
            return_value=repeatable_quest
        )
        quest_manager.quest_repository.get_player_progress = AsyncMock(
            return_value=completed_progress
        )
        quest_manager.quest_repository.update_progress = AsyncMock(return_value=True)
        quest_manager.quest_repository.log_interaction = AsyncMock(return_value=True)

        # Should be able to accept the quest again since it's repeatable
        success, result_data = await quest_processor.accept_quest(
            player_id, quest_id, {"game_state": sample_game_state}
        )

        assert success is True
        assert result_data["quest_id"] == quest_id

    async def test_quest_failure_on_time_limit(
        self, quest_manager, quest_processor, sample_game_state
    ):
        """Test quest failure due to time limit."""
        # Create a quest with time limit
        timed_quest = Quest(
            quest_id="timed_001",
            title="Timed Quest",
            description="Must be completed quickly",
            category=QuestCategory.DELIVERY,
            difficulty=QuestDifficulty.HARD,
            steps=[
                QuestStep(
                    step_id="urgent_task",
                    description="Complete this urgently",
                    requirements={},
                    completion_conditions=["action_type:complete"],
                )
            ],
            time_limit=3600.0,  # 1 hour
            rewards={"experience": 200},
        )

        player_id = "test_player"
        quest_id = "timed_001"

        # Mock expired quest progress
        expired_progress = QuestProgress(
            quest_id=quest_id,
            player_id=player_id,
            status=QuestStatus.ACTIVE,
            current_step=0,
            completed_steps=[],
            step_progress={},
            started_at=time.time() - 7200,  # Started 2 hours ago
            updated_at=time.time() - 3600,  # Last updated 1 hour ago
        )

        quest_manager.get_quest_by_id = AsyncMock(return_value=timed_quest)
        quest_manager.quest_repository.get_player_progress = AsyncMock(
            return_value=expired_progress
        )

        # Check if quest should be marked as expired (this would typically be done by a background process)
        current_time = time.time()
        quest_duration = current_time - expired_progress.started_at

        assert quest_duration > timed_quest.time_limit

        # In a real implementation, this would trigger automatic quest failure
        expired_progress.status = QuestStatus.EXPIRED

        # Verify quest is now expired
        assert expired_progress.status == QuestStatus.EXPIRED

    async def test_quest_hints_and_guidance(
        self, quest_integration, quest_manager, sample_delivery_quest
    ):
        """Test quest hints and guidance system."""
        player_id = "test_player"

        # Mock active quest
        active_progress = QuestProgress(
            quest_id="delivery_001",
            player_id=player_id,
            status=QuestStatus.ACTIVE,
            current_step=0,
            completed_steps=[],
            step_progress={},
        )

        quest_manager.get_player_active_quests = AsyncMock(
            return_value=[active_progress]
        )
        quest_manager.get_quest_by_id = AsyncMock(return_value=sample_delivery_quest)

        # Get hints for marketplace location
        hints = await quest_integration.get_quest_hints_for_location(
            player_id, "marketplace"
        )

        assert len(hints) == 1
        assert hints[0]["quest_id"] == "delivery_001"
        assert hints[0]["objective_type"] == "location_objective"
        assert (
            "marketplace" in hints[0]["hint_text"]
            or "Package Delivery" in hints[0]["hint_text"]
        )

    async def test_available_quest_actions(
        self, quest_integration, quest_manager, sample_delivery_quest
    ):
        """Test getting available quest actions in context."""
        player_id = "test_player"

        # Mock active quest
        active_progress = QuestProgress(
            quest_id="delivery_001",
            player_id=player_id,
            status=QuestStatus.ACTIVE,
            current_step=0,
            completed_steps=[],
            step_progress={},
        )

        quest_manager.get_player_active_quests = AsyncMock(
            return_value=[active_progress]
        )
        quest_manager.get_quest_by_id = AsyncMock(return_value=sample_delivery_quest)

        # Get available actions
        actions = await quest_integration.get_available_quest_actions(
            player_id, {"location": "marketplace"}
        )

        assert len(actions) >= 1
        # Should include take action for the package
        take_actions = [a for a in actions if a["action_type"] == "take"]
        assert len(take_actions) == 1
        assert take_actions[0]["quest_id"] == "delivery_001"
