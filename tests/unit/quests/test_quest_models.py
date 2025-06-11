"""Unit tests for quest models."""

import time

import pytest

from game_loop.quests.models import (
    Quest,
    QuestCategory,
    QuestCompletionResult,
    QuestDifficulty,
    QuestInteractionResult,
    QuestProgress,
    QuestStatus,
    QuestStep,
    QuestUpdate,
)


class TestQuestStep:
    """Test QuestStep model."""

    def test_quest_step_creation(self):
        """Test creating a valid quest step."""
        step = QuestStep(
            step_id="step_1",
            description="Do something important",
            requirements={"location": "town"},
            completion_conditions=["action_type:talk"],
            rewards={"experience": 50},
            optional=False,
        )

        assert step.step_id == "step_1"
        assert step.description == "Do something important"
        assert step.requirements == {"location": "town"}
        assert step.completion_conditions == ["action_type:talk"]
        assert step.rewards == {"experience": 50}
        assert step.optional is False

    def test_quest_step_validation_empty_step_id(self):
        """Test validation fails with empty step_id."""
        with pytest.raises(ValueError, match="step_id cannot be empty"):
            QuestStep(
                step_id="",
                description="Test description",
                requirements={},
                completion_conditions=["test"],
            )

    def test_quest_step_validation_empty_description(self):
        """Test validation fails with empty description."""
        with pytest.raises(ValueError, match="description cannot be empty"):
            QuestStep(
                step_id="step_1",
                description="",
                requirements={},
                completion_conditions=["test"],
            )

    def test_quest_step_validation_empty_conditions(self):
        """Test validation fails with empty completion conditions."""
        with pytest.raises(ValueError, match="completion_conditions cannot be empty"):
            QuestStep(
                step_id="step_1",
                description="Test description",
                requirements={},
                completion_conditions=[],
            )

    def test_quest_step_defaults(self):
        """Test quest step default values."""
        step = QuestStep(
            step_id="step_1",
            description="Test description",
            requirements={},
            completion_conditions=["test"],
        )

        assert step.rewards == {}
        assert step.optional is False


class TestQuest:
    """Test Quest model."""

    def create_sample_step(self, step_id="step_1"):
        """Create a sample quest step."""
        return QuestStep(
            step_id=step_id,
            description="Sample step",
            requirements={},
            completion_conditions=["test"],
        )

    def test_quest_creation(self):
        """Test creating a valid quest."""
        steps = [self.create_sample_step()]

        quest = Quest(
            quest_id="quest_001",
            title="Test Quest",
            description="A test quest",
            category=QuestCategory.DELIVERY,
            difficulty=QuestDifficulty.EASY,
            steps=steps,
            prerequisites=["quest_000"],
            rewards={"gold": 100},
            time_limit=3600.0,
            repeatable=True,
        )

        assert quest.quest_id == "quest_001"
        assert quest.title == "Test Quest"
        assert quest.description == "A test quest"
        assert quest.category == QuestCategory.DELIVERY
        assert quest.difficulty == QuestDifficulty.EASY
        assert len(quest.steps) == 1
        assert quest.prerequisites == ["quest_000"]
        assert quest.rewards == {"gold": 100}
        assert quest.time_limit == 3600.0
        assert quest.repeatable is True

    def test_quest_validation_empty_quest_id(self):
        """Test validation fails with empty quest_id."""
        steps = [self.create_sample_step()]

        with pytest.raises(ValueError, match="quest_id cannot be empty"):
            Quest(
                quest_id="",
                title="Test Quest",
                description="A test quest",
                category=QuestCategory.DELIVERY,
                difficulty=QuestDifficulty.EASY,
                steps=steps,
            )

    def test_quest_validation_empty_title(self):
        """Test validation fails with empty title."""
        steps = [self.create_sample_step()]

        with pytest.raises(ValueError, match="title cannot be empty"):
            Quest(
                quest_id="quest_001",
                title="",
                description="A test quest",
                category=QuestCategory.DELIVERY,
                difficulty=QuestDifficulty.EASY,
                steps=steps,
            )

    def test_quest_validation_empty_steps(self):
        """Test validation fails with empty steps."""
        with pytest.raises(ValueError, match="Quest must have at least one step"):
            Quest(
                quest_id="quest_001",
                title="Test Quest",
                description="A test quest",
                category=QuestCategory.DELIVERY,
                difficulty=QuestDifficulty.EASY,
                steps=[],
            )

    def test_quest_validation_invalid_step_type(self):
        """Test validation fails with invalid step type."""
        with pytest.raises(ValueError, match="All steps must be QuestStep instances"):
            Quest(
                quest_id="quest_001",
                title="Test Quest",
                description="A test quest",
                category=QuestCategory.DELIVERY,
                difficulty=QuestDifficulty.EASY,
                steps=["invalid_step"],
            )

    def test_quest_properties(self):
        """Test quest properties."""
        required_step = self.create_sample_step("required")
        optional_step = QuestStep(
            step_id="optional",
            description="Optional step",
            requirements={},
            completion_conditions=["test"],
            optional=True,
        )

        quest = Quest(
            quest_id="quest_001",
            title="Test Quest",
            description="A test quest",
            category=QuestCategory.DELIVERY,
            difficulty=QuestDifficulty.EASY,
            steps=[required_step, optional_step],
        )

        assert quest.total_steps == 2
        assert len(quest.required_steps) == 1
        assert len(quest.optional_steps) == 1
        assert quest.required_steps[0].step_id == "required"
        assert quest.optional_steps[0].step_id == "optional"

    def test_quest_defaults(self):
        """Test quest default values."""
        steps = [self.create_sample_step()]

        quest = Quest(
            quest_id="quest_001",
            title="Test Quest",
            description="A test quest",
            category=QuestCategory.DELIVERY,
            difficulty=QuestDifficulty.EASY,
            steps=steps,
        )

        assert quest.prerequisites == []
        assert quest.rewards == {}
        assert quest.time_limit is None
        assert quest.repeatable is False


class TestQuestProgress:
    """Test QuestProgress model."""

    def test_quest_progress_creation(self):
        """Test creating quest progress."""
        start_time = time.time()

        progress = QuestProgress(
            quest_id="quest_001",
            player_id="player_001",
            status=QuestStatus.ACTIVE,
            current_step=1,
            completed_steps=["step_1"],
            step_progress={"step_1": {"progress": 100}},
            started_at=start_time,
            updated_at=start_time,
        )

        assert progress.quest_id == "quest_001"
        assert progress.player_id == "player_001"
        assert progress.status == QuestStatus.ACTIVE
        assert progress.current_step == 1
        assert progress.completed_steps == ["step_1"]
        assert progress.step_progress == {"step_1": {"progress": 100}}
        assert progress.started_at == start_time
        assert progress.updated_at == start_time

    def test_quest_progress_validation_empty_quest_id(self):
        """Test validation fails with empty quest_id."""
        with pytest.raises(ValueError, match="quest_id cannot be empty"):
            QuestProgress(
                quest_id="", player_id="player_001", status=QuestStatus.ACTIVE
            )

    def test_quest_progress_validation_empty_player_id(self):
        """Test validation fails with empty player_id."""
        with pytest.raises(ValueError, match="player_id cannot be empty"):
            QuestProgress(quest_id="quest_001", player_id="", status=QuestStatus.ACTIVE)

    def test_quest_progress_validation_negative_step(self):
        """Test validation fails with negative current_step."""
        with pytest.raises(ValueError, match="current_step cannot be negative"):
            QuestProgress(
                quest_id="quest_001",
                player_id="player_001",
                status=QuestStatus.ACTIVE,
                current_step=-1,
            )

    def test_quest_progress_defaults(self):
        """Test quest progress default values."""
        progress = QuestProgress(
            quest_id="quest_001", player_id="player_001", status=QuestStatus.ACTIVE
        )

        assert progress.current_step == 0
        assert progress.completed_steps == []
        assert progress.step_progress == {}
        assert isinstance(progress.started_at, float)
        assert isinstance(progress.updated_at, float)

    def test_mark_step_complete(self):
        """Test marking a step as complete."""
        progress = QuestProgress(
            quest_id="quest_001", player_id="player_001", status=QuestStatus.ACTIVE
        )

        original_update_time = progress.updated_at
        time.sleep(0.01)  # Ensure time difference

        progress.mark_step_complete("step_1")

        assert "step_1" in progress.completed_steps
        assert progress.updated_at > original_update_time

        # Marking the same step again shouldn't duplicate
        progress.mark_step_complete("step_1")
        assert progress.completed_steps.count("step_1") == 1

    def test_update_step_progress(self):
        """Test updating step progress."""
        progress = QuestProgress(
            quest_id="quest_001", player_id="player_001", status=QuestStatus.ACTIVE
        )

        original_update_time = progress.updated_at
        time.sleep(0.01)  # Ensure time difference

        progress_data = {"progress": 50, "notes": "halfway done"}
        progress.update_step_progress("step_1", progress_data)

        assert progress.step_progress["step_1"] == progress_data
        assert progress.updated_at > original_update_time

    def test_advance_to_next_step(self):
        """Test advancing to next step."""
        progress = QuestProgress(
            quest_id="quest_001",
            player_id="player_001",
            status=QuestStatus.ACTIVE,
            current_step=0,
        )

        original_update_time = progress.updated_at
        time.sleep(0.01)  # Ensure time difference

        progress.advance_to_next_step()

        assert progress.current_step == 1
        assert progress.updated_at > original_update_time

    def test_completion_percentage(self):
        """Test completion percentage calculation."""
        progress = QuestProgress(
            quest_id="quest_001",
            player_id="player_001",
            status=QuestStatus.ACTIVE,
            current_step=2,
            completed_steps=["step_1", "step_2"],
        )

        # 2 completed out of 3 total (current_step + 1)
        expected_percentage = (2 / 3) * 100
        assert progress.completion_percentage == expected_percentage

        # Test with no completed steps
        progress.completed_steps = []
        assert progress.completion_percentage == 0.0


class TestQuestInteractionResult:
    """Test QuestInteractionResult model."""

    def test_quest_interaction_result_creation(self):
        """Test creating quest interaction result."""
        progress = QuestProgress(
            quest_id="quest_001", player_id="player_001", status=QuestStatus.ACTIVE
        )

        result = QuestInteractionResult(
            success=True,
            message="Quest accepted successfully",
            quest_id="quest_001",
            updated_progress=progress,
            rewards_granted={"experience": 100},
            errors=[],
        )

        assert result.success is True
        assert result.message == "Quest accepted successfully"
        assert result.quest_id == "quest_001"
        assert result.updated_progress == progress
        assert result.rewards_granted == {"experience": 100}
        assert result.errors == []

    def test_quest_interaction_result_defaults(self):
        """Test quest interaction result default values."""
        result = QuestInteractionResult(
            success=False, message="Failed to process quest"
        )

        assert result.quest_id is None
        assert result.updated_progress is None
        assert result.rewards_granted == {}
        assert result.errors == []


class TestQuestUpdate:
    """Test QuestUpdate model."""

    def test_quest_update_creation(self):
        """Test creating quest update."""
        update_time = time.time()

        update = QuestUpdate(
            quest_id="quest_001",
            player_id="player_001",
            update_type="step_completed",
            update_data={"step_id": "step_1", "progress": 100},
            timestamp=update_time,
        )

        assert update.quest_id == "quest_001"
        assert update.player_id == "player_001"
        assert update.update_type == "step_completed"
        assert update.update_data == {"step_id": "step_1", "progress": 100}
        assert update.timestamp == update_time

    def test_quest_update_default_timestamp(self):
        """Test quest update with default timestamp."""
        update = QuestUpdate(
            quest_id="quest_001",
            player_id="player_001",
            update_type="step_completed",
            update_data={},
        )

        assert isinstance(update.timestamp, float)
        assert update.timestamp > 0


class TestQuestCompletionResult:
    """Test QuestCompletionResult model."""

    def test_quest_completion_result_creation(self):
        """Test creating quest completion result."""
        progress = QuestProgress(
            quest_id="quest_001", player_id="player_001", status=QuestStatus.COMPLETED
        )

        result = QuestCompletionResult(
            success=True,
            quest_id="quest_001",
            final_progress=progress,
            rewards_granted={"experience": 200, "gold": 100},
            completion_message="Quest completed successfully!",
            errors=[],
        )

        assert result.success is True
        assert result.quest_id == "quest_001"
        assert result.final_progress == progress
        assert result.rewards_granted == {"experience": 200, "gold": 100}
        assert result.completion_message == "Quest completed successfully!"
        assert result.errors == []

    def test_quest_completion_result_defaults(self):
        """Test quest completion result default values."""
        result = QuestCompletionResult(
            success=False, quest_id="quest_001", final_progress=None
        )

        assert result.rewards_granted == {}
        assert result.completion_message == ""
        assert result.errors == []
