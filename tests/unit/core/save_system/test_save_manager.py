"""Tests for SaveManager."""

import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from game_loop.core.models.system_models import SaveMetadata
from game_loop.core.save_system.save_manager import SaveManager


@pytest.fixture
def mock_session_factory():
    """Mock session factory."""
    from unittest.mock import MagicMock

    factory = MagicMock()
    session = AsyncMock()

    # Create a proper async context manager mock
    async def mock_get_session():
        return session

    # Mock the context manager behavior
    async_context = AsyncMock()
    async_context.__aenter__ = AsyncMock(return_value=session)
    async_context.__aexit__ = AsyncMock(return_value=None)

    factory.get_session.return_value = async_context

    return factory


@pytest.fixture
def mock_game_state_manager():
    """Mock game state manager."""
    manager = AsyncMock()

    # Mock current player state
    player_state = MagicMock()
    player_state.player_id = uuid.uuid4()
    manager.get_current_player_state.return_value = player_state

    # Mock game state data
    manager.get_player_state.return_value = {
        "player_level": 5,
        "current_location": "Forest",
        "play_time": 3600,
        "inventory": ["sword", "potion"],
    }
    manager.get_world_state.return_value = {
        "time_of_day": "morning",
        "weather": "sunny",
    }

    return manager


@pytest.fixture
def save_manager(mock_session_factory, mock_game_state_manager):
    """Create SaveManager with mocked dependencies."""
    return SaveManager(mock_session_factory, mock_game_state_manager)


class TestSaveManager:
    """Test cases for SaveManager."""

    @pytest.mark.asyncio
    async def test_create_save_success(
        self, save_manager, mock_session_factory, mock_game_state_manager
    ):
        """Test successful save creation."""
        save_name = "test_save"
        description = "Test save description"

        # Mock session operations
        session = mock_session_factory.get_session.return_value.__aenter__.return_value
        session.execute = AsyncMock()
        session.commit = AsyncMock()

        # Mock file operations
        with (
            patch("builtins.open", mock_open()) as mock_file,
            patch("pathlib.Path.stat") as mock_stat,
            patch("pathlib.Path.mkdir"),
        ):

            mock_stat.return_value.st_size = 1024

            result = await save_manager.create_save(save_name, description)

            assert result.success is True
            assert result.save_name == save_name
            assert "saved as" in result.message.lower()
            assert result.save_metadata is not None
            assert result.save_metadata.save_name == save_name

    @pytest.mark.asyncio
    async def test_create_save_no_player(self, save_manager, mock_game_state_manager):
        """Test save creation when no player is available."""
        # Mock no current player
        mock_game_state_manager.get_current_player_state.return_value = None

        result = await save_manager.create_save("test_save")

        assert result.success is False
        assert "no active player" in result.message.lower()

    @pytest.mark.asyncio
    async def test_auto_generate_save_name(self, save_manager, mock_session_factory):
        """Test automatic save name generation."""
        # Mock session operations
        session = mock_session_factory.get_session.return_value.__aenter__.return_value
        session.execute = AsyncMock()
        session.commit = AsyncMock()

        # Mock file operations
        with (
            patch("builtins.open", mock_open()) as mock_file,
            patch("pathlib.Path.stat") as mock_stat,
            patch("pathlib.Path.mkdir"),
        ):

            mock_stat.return_value.st_size = 1024

            result = await save_manager.create_save()

            assert result.success is True
            assert result.save_name.startswith("autosave_")

    @pytest.mark.asyncio
    async def test_load_save_success(self, save_manager, mock_session_factory):
        """Test successful save loading."""
        save_name = "test_save"
        player_id = uuid.uuid4()

        # Mock save metadata
        save_metadata = SaveMetadata(
            save_id=uuid.uuid4(),
            save_name=save_name,
            description="Test save",
            created_at=datetime.now(),
            file_size=1024,
            player_level=5,
            location="Forest",
            play_time=timedelta(hours=1),
            player_id=player_id,
            file_path="saves/test_save.json",
        )

        # Mock database query for _get_save_metadata
        session = mock_session_factory.get_session.return_value.__aenter__.return_value
        mock_row = [
            save_metadata.save_id,
            save_metadata.save_name,
            save_metadata.player_id,
            save_metadata.description,
            save_metadata.file_path,
            save_metadata.file_size,
            save_metadata.player_level,
            save_metadata.location,
            save_metadata.play_time,
            save_metadata.created_at,
        ]

        # Mock the fetchone result for the database query
        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_row
        session.execute.return_value = mock_result

        # Mock file content
        save_data = {
            "player_id": str(player_id),
            "save_name": save_name,
            "created_at": datetime.now().isoformat(),
            "game_state": {"test": "data"},
        }

        with (
            patch("builtins.open", mock_open(read_data=json.dumps(save_data))),
            patch("pathlib.Path.exists", return_value=True),
        ):

            result = await save_manager.load_save(save_name)

            assert result.success is True
            assert result.save_name == save_name
            assert "loaded from" in result.message.lower()

    @pytest.mark.asyncio
    async def test_load_save_not_found(self, save_manager, mock_session_factory):
        """Test loading non-existent save."""
        # Mock no save found
        session = mock_session_factory.get_session.return_value.__aenter__.return_value
        session.execute.return_value.fetchone.return_value = None

        result = await save_manager.load_save("nonexistent_save")

        assert result.success is False
        assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_list_saves(self, save_manager, mock_session_factory):
        """Test listing saves."""
        player_id = uuid.uuid4()

        # Mock database results - using raw SQL format (tuples)
        session = mock_session_factory.get_session.return_value.__aenter__.return_value

        save1_id = uuid.uuid4()
        save2_id = uuid.uuid4()
        now = datetime.now()

        mock_rows = [
            # save_id, save_name, player_id, description, file_path, file_size, player_level, location, play_time, created_at
            [
                save1_id,
                "save1",
                player_id,
                "First save",
                "saves/save1.json",
                1024,
                5,
                "Forest",
                timedelta(hours=1),
                now,
            ],
            [
                save2_id,
                "save2",
                player_id,
                "Second save",
                "saves/save2.json",
                2048,
                10,
                "Castle",
                timedelta(hours=2),
                now,
            ],
        ]

        # Mock the query result properly - session.execute should return an iterable
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter(mock_rows)
        session.execute.return_value = mock_result

        saves = await save_manager.list_saves()

        assert len(saves) == 2
        assert saves[0].save_name == "save1"
        assert saves[1].save_name == "save2"

    @pytest.mark.asyncio
    async def test_delete_save(self, save_manager, mock_session_factory):
        """Test save deletion."""
        save_name = "test_save"
        player_id = uuid.uuid4()

        # Mock save metadata
        session = mock_session_factory.get_session.return_value.__aenter__.return_value
        mock_row = [
            uuid.uuid4(),
            save_name,
            player_id,
            "Test save",
            "saves/test_save.json",
            1024,
            5,
            "Forest",
            timedelta(hours=1),
            datetime.now(),
        ]

        # Set up multiple return values for session.execute
        # First call: _get_save_metadata (returns result with fetchone)
        mock_metadata_result = MagicMock()
        mock_metadata_result.fetchone.return_value = mock_row

        # Second call: delete operation (returns result with rowcount)
        mock_delete_result = MagicMock()
        mock_delete_result.rowcount = 1

        # Configure session.execute to return different results for different calls
        session.execute.side_effect = [mock_metadata_result, mock_delete_result]

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.unlink") as mock_unlink,
        ):

            result = await save_manager.delete_save(save_name)

            assert result is True
            mock_unlink.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_save_not_found(self, save_manager, mock_session_factory):
        """Test deleting non-existent save."""
        # Mock no save found
        session = mock_session_factory.get_session.return_value.__aenter__.return_value
        session.execute.return_value.fetchone.return_value = None

        result = await save_manager.delete_save("nonexistent_save")

        assert result is False

    @pytest.mark.asyncio
    async def test_auto_save(self, save_manager, mock_session_factory):
        """Test automatic save."""
        # Mock session operations
        session = mock_session_factory.get_session.return_value.__aenter__.return_value
        session.execute = AsyncMock()
        session.commit = AsyncMock()

        # Mock file operations
        with (
            patch("builtins.open", mock_open()) as mock_file,
            patch("pathlib.Path.stat") as mock_stat,
            patch("pathlib.Path.mkdir"),
        ):

            mock_stat.return_value.st_size = 1024

            result = await save_manager.auto_save()

            assert result.success is True
            assert result.save_name.startswith("autosave_")
            assert result.save_metadata.description == "Automatic save"

    @pytest.mark.asyncio
    async def test_cleanup_old_saves(self, save_manager, mock_session_factory):
        """Test cleanup of old saves."""
        player_id = uuid.uuid4()

        # Mock old saves to delete
        session = mock_session_factory.get_session.return_value.__aenter__.return_value
        mock_old_saves = [
            ("old_save_1", "saves/old_save_1.json"),
            ("old_save_2", "saves/old_save_2.json"),
        ]
        session.execute.return_value = mock_old_saves

        # Mock successful deletion for each save
        mock_result = MagicMock()
        mock_result.rowcount = 1
        session.execute.return_value = mock_result

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.unlink"),
        ):

            # Should not raise any exception
            await save_manager.cleanup_old_saves(player_id, max_saves=5)

    def test_validate_save_data_success(self, save_manager):
        """Test successful save data validation."""
        valid_data = {
            "player_id": str(uuid.uuid4()),
            "save_name": "test_save",
            "created_at": datetime.now().isoformat(),
            "game_state": {"test": "data"},
        }

        assert save_manager.validate_save_data(valid_data) is True

    def test_validate_save_data_invalid(self, save_manager):
        """Test save data validation failure."""
        # Test with None data
        assert save_manager.validate_save_data(None) is False

        # Test with missing fields
        invalid_data = {
            "player_id": str(uuid.uuid4()),
            "save_name": "test_save",
            # Missing created_at and game_state
        }

        assert save_manager.validate_save_data(invalid_data) is False

        # Test with non-dict data
        assert save_manager.validate_save_data("not a dict") is False

    @pytest.mark.asyncio
    async def test_error_handling(
        self, save_manager, mock_session_factory, mock_game_state_manager
    ):
        """Test error handling in save operations."""
        # Mock database error
        mock_game_state_manager.get_current_player_state.side_effect = Exception(
            "Database error"
        )

        result = await save_manager.create_save("test_save")

        assert result.success is False
        assert "failed to save game" in result.message.lower()

    def test_generate_save_summary(self, save_manager):
        """Test save summary generation."""
        game_state = {
            "current_location": "Enchanted Forest",
            "player_level": 15,
            "play_time": 7200,  # 2 hours
        }

        summary = save_manager.generate_save_summary(game_state)

        assert "Level 15" in summary
        assert "Enchanted Forest" in summary
        assert "2h 0m" in summary
