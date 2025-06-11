"""Unit tests for quest interaction processor."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from game_loop.core.quest.quest_processor import QuestInteractionProcessor
from game_loop.quests.models import (
    Quest,
    QuestCategory,
    QuestDifficulty,
    QuestInteractionType,
    QuestProgress,
    QuestStatus,
    QuestStep,
)
from game_loop.state.models import ActionResult


@pytest.mark.asyncio
class TestQuestInteractionProcessor:
    """Test QuestInteractionProcessor functionality."""

    @pytest.fixture
    def mock_quest_manager(self):
        """Create a mock quest manager."""
        return AsyncMock()

    @pytest.fixture
    def quest_processor(self, mock_quest_manager):
        """Create a quest processor with mock manager."""
        return QuestInteractionProcessor(mock_quest_manager)

    @pytest.fixture
    def sample_quest(self):
        """Create a sample quest for testing."""
        step = QuestStep(
            step_id="step_1",
            description="Talk to the NPC",
            requirements={"location": "town"},
            completion_conditions=["action_type:talk"],
            rewards={"experience": 50},
        )

        return Quest(
            quest_id="quest_001",
            title="Sample Quest",
            description="A sample quest for testing",
            category=QuestCategory.DELIVERY,
            difficulty=QuestDifficulty.EASY,
            steps=[step],
            rewards={"experience": 100, "gold": 50},
        )

    @pytest.fixture
    def sample_progress(self):
        """Create sample quest progress."""
        return QuestProgress(
            quest_id="quest_001",
            player_id="player_001",
            status=QuestStatus.ACTIVE,
            current_step=0,
            completed_steps=[],
            step_progress={},
        )

    @pytest.fixture
    def sample_game_state(self):
        """Create sample game state."""
        return {"player_id": "player_001", "location_id": "town", "metadata": {}}

    @pytest.fixture
    def sample_action_result(self):
        """Create sample action result."""
        return ActionResult(
            success=True,
            feedback_message="Successfully talked to NPC",
            command="talk",
            processed_input={"target": "npc_001"},
        )

    async def test_process_quest_interaction_discover(
        self, quest_processor, mock_quest_manager, sample_game_state, sample_quest
    ):
        """Test processing discover interaction."""
        quest_context = {"location_id": "town"}

        # Mock quest discovery
        with patch.object(
            quest_processor, "discover_available_quests", return_value=[sample_quest]
        ):
            result = await quest_processor.process_quest_interaction(
                QuestInteractionType.DISCOVER,
                "player_001",
                quest_context,
                sample_game_state,
            )

        assert result.success is True
        assert "1 available quest" in result.message
        assert "discovered_quests" in result.rewards_granted

    async def test_process_quest_interaction_accept(
        self, quest_processor, mock_quest_manager, sample_game_state
    ):
        """Test processing accept interaction."""
        quest_context = {"quest_id": "quest_001"}

        # Mock quest acceptance
        with patch.object(
            quest_processor,
            "accept_quest",
            return_value=(True, {"quest_title": "Sample Quest"}),
        ):
            result = await quest_processor.process_quest_interaction(
                QuestInteractionType.ACCEPT,
                "player_001",
                quest_context,
                sample_game_state,
            )

        assert result.success is True
        assert "accepted" in result.message
        assert result.quest_id == "quest_001"

    async def test_process_quest_interaction_progress(
        self,
        quest_processor,
        mock_quest_manager,
        sample_game_state,
        sample_action_result,
    ):
        """Test processing progress interaction."""
        quest_context = {"quest_id": "quest_001", "action_result": sample_action_result}

        # Mock progress update
        mock_update = Mock()
        mock_update.__dict__ = {"quest_id": "quest_001", "update_type": "progress"}

        with patch.object(
            quest_processor, "update_quest_progress", return_value=[mock_update]
        ):
            result = await quest_processor.process_quest_interaction(
                QuestInteractionType.PROGRESS,
                "player_001",
                quest_context,
                sample_game_state,
            )

        assert result.success is True
        assert "progress updated" in result.message

    async def test_process_quest_interaction_complete(
        self, quest_processor, mock_quest_manager, sample_game_state
    ):
        """Test processing complete interaction."""
        quest_context = {"quest_id": "quest_001"}

        # Mock quest completion
        completion_result = Mock()
        completion_result.success = True
        completion_result.completion_message = "Quest completed!"
        completion_result.final_progress = None
        completion_result.rewards_granted = {"experience": 100}
        completion_result.errors = []

        with patch.object(
            quest_processor, "complete_quest", return_value=completion_result
        ):
            result = await quest_processor.process_quest_interaction(
                QuestInteractionType.COMPLETE,
                "player_001",
                quest_context,
                sample_game_state,
            )

        assert result.success is True
        assert result.message == "Quest completed!"
        assert result.rewards_granted == {"experience": 100}

    async def test_process_quest_interaction_abandon(
        self, quest_processor, mock_quest_manager, sample_game_state
    ):
        """Test processing abandon interaction."""
        quest_context = {"quest_id": "quest_001"}

        mock_quest_manager.abandon_quest.return_value = True

        result = await quest_processor.process_quest_interaction(
            QuestInteractionType.ABANDON, "player_001", quest_context, sample_game_state
        )

        assert result.success is True
        assert "abandoned" in result.message

    async def test_process_quest_interaction_query_specific(
        self,
        quest_processor,
        mock_quest_manager,
        sample_game_state,
        sample_quest,
        sample_progress,
    ):
        """Test processing query interaction for specific quest."""
        quest_context = {"quest_id": "quest_001"}

        mock_quest_manager.get_quest_by_id.return_value = sample_quest
        mock_quest_manager.quest_repository.get_player_progress.return_value = (
            sample_progress
        )

        result = await quest_processor.process_quest_interaction(
            QuestInteractionType.QUERY, "player_001", quest_context, sample_game_state
        )

        assert result.success is True
        assert "quest_info" in result.rewards_granted

    async def test_process_quest_interaction_query_all_active(
        self,
        quest_processor,
        mock_quest_manager,
        sample_game_state,
        sample_quest,
        sample_progress,
    ):
        """Test processing query interaction for all active quests."""
        quest_context = {}

        mock_quest_manager.get_player_active_quests.return_value = [sample_progress]
        mock_quest_manager.get_quest_by_id.return_value = sample_quest

        result = await quest_processor.process_quest_interaction(
            QuestInteractionType.QUERY, "player_001", quest_context, sample_game_state
        )

        assert result.success is True
        assert "1 active quest" in result.message
        assert "active_quests" in result.rewards_granted

    async def test_process_quest_interaction_unknown_type(
        self, quest_processor, mock_quest_manager, sample_game_state
    ):
        """Test processing unknown interaction type."""
        # Create a mock enum value that doesn't exist
        unknown_type = Mock()
        unknown_type.value = "unknown"

        result = await quest_processor.process_quest_interaction(
            unknown_type, "player_001", {}, sample_game_state
        )

        assert result.success is False
        assert "Unknown interaction type" in result.message

    async def test_process_quest_interaction_exception(
        self, quest_processor, mock_quest_manager, sample_game_state
    ):
        """Test processing interaction with exception."""
        # Mock an exception during processing
        mock_quest_manager.abandon_quest.side_effect = Exception("Test error")

        quest_context = {"quest_id": "quest_001"}

        result = await quest_processor.process_quest_interaction(
            QuestInteractionType.ABANDON, "player_001", quest_context, sample_game_state
        )

        assert result.success is False
        assert "error occurred" in result.message
        assert "Test error" in result.errors

    async def test_discover_available_quests(
        self, quest_processor, mock_quest_manager, sample_quest
    ):
        """Test discovering available quests."""
        mock_quest_manager.quest_repository.get_available_quests.return_value = [
            sample_quest
        ]
        mock_quest_manager.validate_quest_prerequisites.return_value = (True, [])

        context = {"game_state": Mock()}

        result = await quest_processor.discover_available_quests(
            "player_001", "town", context
        )

        assert len(result) == 1
        assert result[0] == sample_quest

    async def test_discover_available_quests_invalid_prerequisites(
        self, quest_processor, mock_quest_manager, sample_quest
    ):
        """Test discovering quests with invalid prerequisites."""
        mock_quest_manager.quest_repository.get_available_quests.return_value = [
            sample_quest
        ]
        mock_quest_manager.validate_quest_prerequisites.return_value = (
            False,
            ["Missing prerequisite"],
        )

        context = {"game_state": Mock()}

        result = await quest_processor.discover_available_quests(
            "player_001", "town", context
        )

        assert len(result) == 0

    async def test_accept_quest_success(
        self, quest_processor, mock_quest_manager, sample_quest
    ):
        """Test successful quest acceptance."""
        mock_quest_manager.start_quest.return_value = (True, [])
        mock_quest_manager.get_quest_by_id.return_value = sample_quest

        context = {"game_state": Mock()}

        success, result_data = await quest_processor.accept_quest(
            "player_001", "quest_001", context
        )

        assert success is True
        assert result_data["quest_id"] == "quest_001"
        assert result_data["quest_title"] == sample_quest.title

    async def test_accept_quest_failure(self, quest_processor, mock_quest_manager):
        """Test failed quest acceptance."""
        mock_quest_manager.start_quest.return_value = (False, ["Validation failed"])

        context = {"game_state": Mock()}

        success, result_data = await quest_processor.accept_quest(
            "player_001", "quest_001", context
        )

        assert success is False
        assert "Validation failed" in result_data["errors"]

    async def test_update_quest_progress_success(
        self, quest_processor, mock_quest_manager, sample_progress, sample_action_result
    ):
        """Test successful quest progress update."""
        mock_quest_manager.update_quest_progress.return_value = True
        mock_quest_manager.quest_repository.get_player_progress.return_value = (
            sample_progress
        )

        result = await quest_processor.update_quest_progress(
            "player_001", "quest_001", sample_action_result, {}
        )

        assert len(result) == 1
        assert result[0].quest_id == "quest_001"
        assert result[0].update_type == "progress_update"

    async def test_update_quest_progress_failure(
        self, quest_processor, mock_quest_manager, sample_action_result
    ):
        """Test failed quest progress update."""
        mock_quest_manager.update_quest_progress.return_value = False

        result = await quest_processor.update_quest_progress(
            "player_001", "quest_001", sample_action_result, {}
        )

        assert len(result) == 0

    async def test_complete_quest_step_success(
        self, quest_processor, mock_quest_manager, sample_progress, sample_quest
    ):
        """Test successful quest step completion."""
        sample_progress.status = QuestStatus.ACTIVE
        mock_quest_manager.quest_repository.get_player_progress.return_value = (
            sample_progress
        )
        mock_quest_manager.quest_repository.update_progress.return_value = True
        mock_quest_manager.get_quest_by_id.return_value = sample_quest
        mock_quest_manager.grant_quest_rewards.return_value = {"experience": 100}

        completion_data = {"notes": "Step completed successfully"}

        success, result_data = await quest_processor.complete_quest_step(
            "player_001", "quest_001", "step_1", completion_data
        )

        assert success is True
        assert result_data["step_id"] == "step_1"
        assert result_data["completion_data"] == completion_data

    async def test_complete_quest_step_quest_not_active(
        self, quest_processor, mock_quest_manager, sample_progress
    ):
        """Test quest step completion when quest is not active."""
        sample_progress.status = QuestStatus.COMPLETED
        mock_quest_manager.quest_repository.get_player_progress.return_value = (
            sample_progress
        )

        success, result_data = await quest_processor.complete_quest_step(
            "player_001", "quest_001", "step_1", {}
        )

        assert success is False
        assert "not active" in result_data["error"]

    async def test_complete_quest_success(
        self, quest_processor, mock_quest_manager, sample_quest, sample_progress
    ):
        """Test successful quest completion."""
        # Setup quest with required steps
        sample_quest.steps[0].optional = False
        sample_progress.completed_steps = ["step_1"]  # All required steps completed

        mock_quest_manager.get_quest_by_id.return_value = sample_quest
        mock_quest_manager.quest_repository.get_player_progress.return_value = (
            sample_progress
        )
        mock_quest_manager.quest_repository.update_progress.return_value = True
        mock_quest_manager.grant_quest_rewards.return_value = {
            "experience": 100,
            "gold": 50,
        }

        result = await quest_processor.complete_quest("player_001", "quest_001", {})

        assert result.success is True
        assert result.quest_id == "quest_001"
        assert result.rewards_granted == {"experience": 100, "gold": 50}
        assert "completed" in result.completion_message

    async def test_complete_quest_not_found(self, quest_processor, mock_quest_manager):
        """Test quest completion when quest doesn't exist."""
        mock_quest_manager.get_quest_by_id.return_value = None

        result = await quest_processor.complete_quest("player_001", "nonexistent", {})

        assert result.success is False
        assert "Quest not found" in result.errors

    async def test_complete_quest_not_active(
        self, quest_processor, mock_quest_manager, sample_quest, sample_progress
    ):
        """Test quest completion when quest is not active."""
        sample_progress.status = QuestStatus.COMPLETED

        mock_quest_manager.get_quest_by_id.return_value = sample_quest
        mock_quest_manager.quest_repository.get_player_progress.return_value = (
            sample_progress
        )

        result = await quest_processor.complete_quest("player_001", "quest_001", {})

        assert result.success is False
        assert "Quest is not active" in result.errors

    async def test_complete_quest_steps_not_completed(
        self, quest_processor, mock_quest_manager, sample_quest, sample_progress
    ):
        """Test quest completion when not all required steps are completed."""
        # Quest has required steps but none are completed
        sample_quest.steps[0].optional = False
        sample_progress.completed_steps = []

        mock_quest_manager.get_quest_by_id.return_value = sample_quest
        mock_quest_manager.quest_repository.get_player_progress.return_value = (
            sample_progress
        )

        result = await quest_processor.complete_quest("player_001", "quest_001", {})

        assert result.success is False
        assert "Not all required steps completed:" in result.errors[0]

    async def test_process_discover_interaction_no_quests(
        self, quest_processor, mock_quest_manager, sample_game_state
    ):
        """Test discover interaction when no quests are available."""
        quest_context = {"location_id": "town"}

        with patch.object(
            quest_processor, "discover_available_quests", return_value=[]
        ):
            result = await quest_processor._process_discover_interaction(
                "player_001", quest_context, sample_game_state
            )

        assert result.success is True
        assert "No available quests found" in result.message

    async def test_process_accept_interaction_no_quest_id(
        self, quest_processor, mock_quest_manager, sample_game_state
    ):
        """Test accept interaction without quest ID."""
        quest_context = {}

        result = await quest_processor._process_accept_interaction(
            "player_001", quest_context, sample_game_state
        )

        assert result.success is False
        assert "Quest ID required" in result.message

    async def test_process_progress_interaction_missing_data(
        self, quest_processor, mock_quest_manager, sample_game_state
    ):
        """Test progress interaction with missing data."""
        quest_context = {"quest_id": "quest_001"}  # Missing action_result

        result = await quest_processor._process_progress_interaction(
            "player_001", quest_context, sample_game_state
        )

        assert result.success is False
        assert "action result required" in result.message

    async def test_process_query_interaction_quest_not_found(
        self, quest_processor, mock_quest_manager, sample_game_state
    ):
        """Test query interaction for non-existent quest."""
        quest_context = {"quest_id": "nonexistent"}

        mock_quest_manager.get_quest_by_id.return_value = None

        result = await quest_processor._process_query_interaction(
            "player_001", quest_context, sample_game_state
        )

        assert result.success is False
        assert "Quest not found" in result.message

    async def test_process_query_interaction_no_active_quests(
        self, quest_processor, mock_quest_manager, sample_game_state
    ):
        """Test query interaction when player has no active quests."""
        quest_context = {}

        mock_quest_manager.get_player_active_quests.return_value = []

        result = await quest_processor._process_query_interaction(
            "player_001", quest_context, sample_game_state
        )

        assert result.success is True
        assert "no active quests" in result.message
        assert result.rewards_granted["active_quests"] == []
