"""Unit tests for quest manager."""

from unittest.mock import AsyncMock, patch

import pytest

from game_loop.quests.models import (
    Quest,
    QuestCategory,
    QuestDifficulty,
    QuestProgress,
    QuestStatus,
    QuestStep,
)
from game_loop.quests.quest_manager import QuestManager
from game_loop.state.models import ActionResult


@pytest.mark.asyncio
class TestQuestManager:
    """Test QuestManager functionality."""

    @pytest.fixture
    def mock_repository(self):
        """Create a mock quest repository."""
        return AsyncMock()

    @pytest.fixture
    def quest_manager(self, mock_repository):
        """Create a quest manager with mock repository."""
        return QuestManager(mock_repository)

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
    def sample_action_result(self):
        """Create sample action result."""
        return ActionResult(
            success=True,
            feedback_message="Successfully talked to NPC",
            command="talk",
            processed_input={"target": "npc_001"},
        )

    @pytest.fixture
    def sample_game_state(self):
        """Create sample game state."""
        return {"player_id": "player_001", "location_id": "town", "metadata": {}}

    async def test_get_player_active_quests(
        self, quest_manager, mock_repository, sample_progress
    ):
        """Test getting player's active quests."""
        mock_repository.get_active_quests.return_value = [sample_progress]

        result = await quest_manager.get_player_active_quests("player_001")

        assert len(result) == 1
        assert result[0] == sample_progress
        mock_repository.get_active_quests.assert_called_once_with("player_001")

    async def test_get_quest_by_id_from_cache(
        self, quest_manager, mock_repository, sample_quest
    ):
        """Test getting quest by ID from cache."""
        # Add quest to cache
        quest_manager._cache["quest_001"] = sample_quest

        result = await quest_manager.get_quest_by_id("quest_001")

        assert result == sample_quest
        # Repository should not be called when quest is in cache
        mock_repository.get_quest.assert_not_called()

    async def test_get_quest_by_id_from_repository(
        self, quest_manager, mock_repository, sample_quest
    ):
        """Test getting quest by ID from repository."""
        mock_repository.get_quest.return_value = sample_quest

        result = await quest_manager.get_quest_by_id("quest_001")

        assert result == sample_quest
        assert quest_manager._cache["quest_001"] == sample_quest
        mock_repository.get_quest.assert_called_once_with("quest_001")

    async def test_get_quest_by_id_not_found(self, quest_manager, mock_repository):
        """Test getting quest by ID when quest doesn't exist."""
        mock_repository.get_quest.return_value = None

        result = await quest_manager.get_quest_by_id("nonexistent")

        assert result is None
        assert "nonexistent" not in quest_manager._cache

    async def test_validate_quest_prerequisites_success(
        self, quest_manager, mock_repository, sample_quest, sample_game_state
    ):
        """Test successful quest prerequisite validation."""
        # Mock quest retrieval
        quest_manager._cache["quest_001"] = sample_quest

        # Mock no existing progress (quest not started)
        mock_repository.get_player_progress.return_value = None

        # Quest has no prerequisites
        sample_quest.prerequisites = []

        is_valid, errors = await quest_manager.validate_quest_prerequisites(
            "player_001", "quest_001", sample_game_state
        )

        assert is_valid is True
        assert errors == []

    async def test_validate_quest_prerequisites_quest_not_found(
        self, quest_manager, mock_repository, sample_game_state
    ):
        """Test quest prerequisite validation when quest doesn't exist."""
        mock_repository.get_quest.return_value = None

        is_valid, errors = await quest_manager.validate_quest_prerequisites(
            "player_001", "nonexistent", sample_game_state
        )

        assert is_valid is False
        assert "Quest nonexistent not found" in errors

    async def test_validate_quest_prerequisites_already_active(
        self,
        quest_manager,
        mock_repository,
        sample_quest,
        sample_progress,
        sample_game_state,
    ):
        """Test quest prerequisite validation when quest is already active."""
        quest_manager._cache["quest_001"] = sample_quest
        sample_quest.prerequisites = []

        # Mock existing active progress
        sample_progress.status = QuestStatus.ACTIVE
        mock_repository.get_player_progress.return_value = sample_progress

        is_valid, errors = await quest_manager.validate_quest_prerequisites(
            "player_001", "quest_001", sample_game_state
        )

        assert is_valid is False
        assert "Quest is already active" in errors

    async def test_validate_quest_prerequisites_already_completed_non_repeatable(
        self,
        quest_manager,
        mock_repository,
        sample_quest,
        sample_progress,
        sample_game_state,
    ):
        """Test quest prerequisite validation when quest is completed and not repeatable."""
        quest_manager._cache["quest_001"] = sample_quest
        sample_quest.prerequisites = []
        sample_quest.repeatable = False

        # Mock existing completed progress
        sample_progress.status = QuestStatus.COMPLETED
        mock_repository.get_player_progress.return_value = sample_progress

        is_valid, errors = await quest_manager.validate_quest_prerequisites(
            "player_001", "quest_001", sample_game_state
        )

        assert is_valid is False
        assert "Quest is already completed and not repeatable" in errors

    async def test_validate_quest_prerequisites_missing_prerequisite(
        self, quest_manager, mock_repository, sample_quest, sample_game_state
    ):
        """Test quest prerequisite validation when prerequisite is not completed."""
        quest_manager._cache["quest_001"] = sample_quest
        sample_quest.prerequisites = ["quest_000"]

        # Mock no progress on prerequisite quest
        mock_repository.get_player_progress.return_value = None

        is_valid, errors = await quest_manager.validate_quest_prerequisites(
            "player_001", "quest_001", sample_game_state
        )

        assert is_valid is False
        assert "Must complete quest 'quest_000' first" in errors

    async def test_check_step_completion_conditions_success(
        self, quest_manager, mock_repository, sample_quest, sample_action_result
    ):
        """Test successful step completion condition check."""
        quest_manager._cache["quest_001"] = sample_quest

        # Mock condition evaluation
        with patch.object(quest_manager, "_evaluate_condition", return_value=True):
            result = await quest_manager.check_step_completion_conditions(
                "player_001", "quest_001", "step_1", sample_action_result
            )

        assert result is True

    async def test_check_step_completion_conditions_quest_not_found(
        self, quest_manager, mock_repository, sample_action_result
    ):
        """Test step completion condition check when quest doesn't exist."""
        mock_repository.get_quest.return_value = None

        result = await quest_manager.check_step_completion_conditions(
            "player_001", "nonexistent", "step_1", sample_action_result
        )

        assert result is False

    async def test_check_step_completion_conditions_step_not_found(
        self, quest_manager, mock_repository, sample_quest, sample_action_result
    ):
        """Test step completion condition check when step doesn't exist."""
        quest_manager._cache["quest_001"] = sample_quest

        result = await quest_manager.check_step_completion_conditions(
            "player_001", "quest_001", "nonexistent_step", sample_action_result
        )

        assert result is False

    async def test_grant_quest_rewards(self, quest_manager, mock_repository):
        """Test granting quest rewards."""
        rewards = {
            "experience": 100,
            "gold": 50,
            "items": ["sword"],
            "skills": {"combat": 5},
        }
        context = {"quest_id": "quest_001"}

        mock_repository.log_interaction.return_value = True

        granted_rewards = await quest_manager.grant_quest_rewards(
            "player_001", rewards, context
        )

        assert granted_rewards == rewards
        mock_repository.log_interaction.assert_called_once_with(
            "quest_001",
            "player_001",
            "reward_grant",
            {"rewards": rewards, "context": context},
        )

    async def test_start_quest_success(
        self, quest_manager, mock_repository, sample_quest, sample_game_state
    ):
        """Test successful quest start."""
        quest_manager._cache["quest_001"] = sample_quest
        sample_quest.prerequisites = []

        # Mock no existing progress
        mock_repository.get_player_progress.return_value = None
        mock_repository.update_progress.return_value = True
        mock_repository.log_interaction.return_value = True

        success, errors = await quest_manager.start_quest(
            "player_001", "quest_001", sample_game_state
        )

        assert success is True
        assert errors == []
        mock_repository.update_progress.assert_called_once()
        mock_repository.log_interaction.assert_called_once()

    async def test_start_quest_validation_failure(
        self, quest_manager, mock_repository, sample_quest, sample_game_state
    ):
        """Test quest start with validation failure."""
        quest_manager._cache["quest_001"] = sample_quest
        sample_quest.prerequisites = ["quest_000"]

        # Mock no prerequisite completion
        mock_repository.get_player_progress.return_value = None

        success, errors = await quest_manager.start_quest(
            "player_001", "quest_001", sample_game_state
        )

        assert success is False
        assert len(errors) > 0
        mock_repository.update_progress.assert_not_called()

    async def test_update_quest_progress_success(
        self,
        quest_manager,
        mock_repository,
        sample_quest,
        sample_progress,
        sample_action_result,
    ):
        """Test successful quest progress update."""
        quest_manager._cache["quest_001"] = sample_quest
        mock_repository.get_player_progress.return_value = sample_progress
        mock_repository.update_progress.return_value = True
        mock_repository.log_interaction.return_value = True

        # Mock successful step completion
        with patch.object(
            quest_manager, "check_step_completion_conditions", return_value=True
        ):
            with patch.object(
                quest_manager, "grant_quest_rewards", return_value={"experience": 50}
            ):
                result = await quest_manager.update_quest_progress(
                    "player_001", "quest_001", sample_action_result
                )

        assert result is True
        mock_repository.update_progress.assert_called_once()

    async def test_update_quest_progress_no_progress(
        self, quest_manager, mock_repository, sample_action_result
    ):
        """Test quest progress update when no progress exists."""
        mock_repository.get_player_progress.return_value = None

        result = await quest_manager.update_quest_progress(
            "player_001", "quest_001", sample_action_result
        )

        assert result is False

    async def test_abandon_quest_success(
        self, quest_manager, mock_repository, sample_progress
    ):
        """Test successful quest abandonment."""
        sample_progress.status = QuestStatus.ACTIVE
        mock_repository.get_player_progress.return_value = sample_progress
        mock_repository.update_progress.return_value = True
        mock_repository.log_interaction.return_value = True

        result = await quest_manager.abandon_quest("player_001", "quest_001")

        assert result is True
        assert sample_progress.status == QuestStatus.ABANDONED
        mock_repository.update_progress.assert_called_once()
        mock_repository.log_interaction.assert_called_once()

    async def test_abandon_quest_not_active(
        self, quest_manager, mock_repository, sample_progress
    ):
        """Test quest abandonment when quest is not active."""
        sample_progress.status = QuestStatus.COMPLETED
        mock_repository.get_player_progress.return_value = sample_progress

        result = await quest_manager.abandon_quest("player_001", "quest_001")

        assert result is False
        mock_repository.update_progress.assert_not_called()

    async def test_generate_dynamic_quest_delivery(
        self, quest_manager, mock_repository
    ):
        """Test generating a delivery quest."""
        context = {"pickup_location": "tavern", "delivery_location": "castle"}

        quest = await quest_manager.generate_dynamic_quest(
            "player_001", context, "delivery"
        )

        assert quest is not None
        assert quest.category == QuestCategory.DELIVERY
        assert len(quest.steps) == 2
        assert "pickup" in quest.steps[0].step_id
        assert "delivery" in quest.steps[1].step_id

    async def test_generate_dynamic_quest_exploration(
        self, quest_manager, mock_repository
    ):
        """Test generating an exploration quest."""
        context = {"target_location": "ancient_ruins"}

        quest = await quest_manager.generate_dynamic_quest(
            "player_001", context, "exploration"
        )

        assert quest is not None
        assert quest.category == QuestCategory.EXPLORATION
        assert len(quest.steps) == 1

    async def test_generate_dynamic_quest_invalid_type(
        self, quest_manager, mock_repository
    ):
        """Test generating quest with invalid type."""
        quest = await quest_manager.generate_dynamic_quest(
            "player_001", {}, "invalid_type"
        )

        assert quest is None

    async def test_evaluate_condition_action_type(
        self, quest_manager, sample_action_result
    ):
        """Test evaluating action type condition."""
        condition = "action_type:talk"

        result = await quest_manager._evaluate_condition(
            condition, sample_action_result, "player_001", "quest_001"
        )

        assert result is True

    async def test_evaluate_condition_target_object(
        self, quest_manager, sample_action_result
    ):
        """Test evaluating target object condition."""
        condition = "target_object:npc_001"

        result = await quest_manager._evaluate_condition(
            condition, sample_action_result, "player_001", "quest_001"
        )

        assert result is True  # Returns True for placeholder implementation

    async def test_evaluate_condition_invalid_format(
        self, quest_manager, sample_action_result
    ):
        """Test evaluating condition with invalid format."""
        condition = "invalid_condition"

        result = await quest_manager._evaluate_condition(
            condition, sample_action_result, "player_001", "quest_001"
        )

        assert result is False

    async def test_evaluate_condition_mismatch(
        self, quest_manager, sample_action_result
    ):
        """Test evaluating condition that doesn't match."""
        condition = "action_type:fight"  # action_result has "talk"

        result = await quest_manager._evaluate_condition(
            condition, sample_action_result, "player_001", "quest_001"
        )

        assert result is False
