"""
Integration tests for embedding service integration (Phase 3).

These tests verify that the consolidated ConfigManager, dependency injection,
and CLI integration work correctly together for embedding functionality.
"""

from unittest.mock import Mock, patch

from game_loop.config.cli import ConfigCLI
from game_loop.config.manager import ConfigManager
from game_loop.state.manager import GameStateManager


class TestConfigManagerEmbeddingIntegration:
    """Test ConfigManager direct embedding integration."""

    def test_embedding_service_creation_with_valid_config(self):
        """Test that embedding service can be created with valid config."""
        config_manager = ConfigManager()

        # Verify the method exists
        assert hasattr(config_manager, "create_embedding_service")
        assert callable(config_manager.create_embedding_service)

    def test_is_embedding_enabled_method_exists(self):
        """Test that is_embedding_enabled method exists."""
        config_manager = ConfigManager()
        assert hasattr(config_manager, "is_embedding_enabled")
        assert callable(config_manager.is_embedding_enabled)

    def test_is_embedding_enabled_default_false(self):
        """Test that embedding is disabled by default."""
        config_manager = ConfigManager()

        # Should be False by default
        assert not config_manager.is_embedding_enabled()

    def test_is_embedding_enabled_with_feature_flag(self):
        """Test embedding enabled when feature flag is set."""
        config_manager = ConfigManager()
        # Enable the feature flag
        config_manager.config.features.use_embedding_search = True

        assert config_manager.is_embedding_enabled()

    @patch("game_loop.embeddings.service.EmbeddingService")
    def test_create_embedding_service_success(self, mock_embedding_service):
        """Test successful embedding service creation."""
        mock_service = Mock()
        mock_embedding_service.return_value = mock_service

        config_manager = ConfigManager()
        service = config_manager.create_embedding_service()

        # Verify service was created correctly
        assert service is mock_service
        mock_embedding_service.assert_called_once_with(config_manager=config_manager)

    @patch("game_loop.embeddings.service.EmbeddingService")
    def test_create_embedding_service_import_error(self, mock_embedding_service):
        """Test handling of import errors during service creation."""
        mock_embedding_service.side_effect = ImportError("Missing dependencies")

        config_manager = ConfigManager()

        try:
            config_manager.create_embedding_service()
            raise AssertionError("Expected ImportError to be raised")
        except ImportError as e:
            assert "Missing dependencies" in str(e)


class TestGameStateManagerEmbeddingIntegration:
    """Test GameStateManager embedding integration."""

    def test_optional_embedding_service_injection(self):
        """Test that embedding service can be injected as dependency."""
        config_manager = ConfigManager()
        db_pool = Mock()
        mock_embedding_service = Mock()

        # Test with injected service
        manager = GameStateManager(
            config_manager=config_manager,
            db_pool=db_pool,
            embedding_service=mock_embedding_service,
        )

        assert manager._embedding_service is mock_embedding_service

    def test_embedding_service_property_when_disabled(self):
        """Test embedding_service property when feature is disabled."""
        config_manager = ConfigManager()
        # Ensure feature is disabled (default)
        config_manager.config.features.use_embedding_search = False

        db_pool = Mock()
        manager = GameStateManager(config_manager=config_manager, db_pool=db_pool)

        # Should return None when disabled
        assert manager.embedding_service is None

    def test_embedding_service_property_when_enabled(self):
        """Test embedding_service property when feature is enabled."""
        # Setup mocks
        mock_config_manager = Mock()
        mock_config_manager.is_embedding_enabled.return_value = True
        mock_service = Mock()
        mock_config_manager.create_embedding_service.return_value = mock_service

        db_pool = Mock()
        manager = GameStateManager(config_manager=mock_config_manager, db_pool=db_pool)

        # Should create and return service
        service = manager.embedding_service
        assert service is mock_service

        # Second call should return cached service
        service2 = manager.embedding_service
        assert service2 is mock_service
        mock_config_manager.create_embedding_service.assert_called_once()

    @patch("game_loop.state.manager.logger")
    def test_embedding_service_creation_failure_handling(self, mock_logger):
        """Test graceful handling of embedding service creation failures."""
        config_manager = ConfigManager()
        config_manager.config.features.use_embedding_search = True

        # Mock create_embedding_service to raise an exception
        def mock_create_embedding_service():
            raise ImportError("Missing embedding dependencies")

        config_manager.create_embedding_service = mock_create_embedding_service

        db_pool = Mock()
        manager = GameStateManager(config_manager=config_manager, db_pool=db_pool)

        # Should return None and log warning
        service = manager.embedding_service
        assert service is None
        mock_logger.warning.assert_called_once()


class TestCLIEmbeddingIntegration:
    """Test CLI embedding feature integration."""

    def test_cli_embedding_feature_flag_enable(self):
        """Test CLI flag to enable embedding search."""
        cli = ConfigCLI()
        args_dict = cli.parse_args(["--features.use-embedding-search"])

        assert "features.use_embedding_search" in args_dict
        assert args_dict["features.use_embedding_search"] is True

    def test_cli_embedding_feature_flag_disable(self):
        """Test CLI flag to disable embedding search."""
        cli = ConfigCLI()
        args_dict = cli.parse_args(["--features.no-embedding-search"])

        assert "features.use_embedding_search" in args_dict
        assert args_dict["features.use_embedding_search"] is False

    def test_cli_to_config_integration(self):
        """Test that CLI args properly configure embedding feature."""
        cli = ConfigCLI()
        config_manager = cli.init_config(["--features.use-embedding-search"])

        # Feature should be enabled
        assert config_manager.config.features.use_embedding_search is True
        assert config_manager.is_embedding_enabled() is True

    def test_cli_feature_group_exists(self):
        """Test that feature argument group exists in CLI."""
        cli = ConfigCLI()

        # Check that the parser has the feature group
        group_names = [group.title for group in cli.parser._action_groups]
        assert "Feature options" in group_names


class TestEndToEndIntegration:
    """Test end-to-end integration flow."""

    @patch("game_loop.embeddings.service.EmbeddingService")
    def test_cli_to_embedding_service_flow(self, mock_embedding_service):
        """Test complete flow: CLI → Config → Service."""
        # Setup mocks
        mock_service = Mock()
        mock_embedding_service.return_value = mock_service

        # CLI configuration
        cli = ConfigCLI()
        config_manager = cli.init_config(["--features.use-embedding-search"])

        # Create GameStateManager
        db_pool = Mock()
        manager = GameStateManager(config_manager=config_manager, db_pool=db_pool)

        # Verify embedding service is created
        service = manager.embedding_service
        assert service is mock_service

        # Verify configuration flow
        assert config_manager.config.features.use_embedding_search is True
        mock_embedding_service.assert_called_once_with(config_manager=config_manager)

    def test_feature_disabled_by_default_flow(self):
        """Test that embedding is disabled by default in complete flow."""
        # CLI configuration without feature flag
        cli = ConfigCLI()
        config_manager = cli.init_config([])

        # Create GameStateManager
        db_pool = Mock()
        manager = GameStateManager(config_manager=config_manager, db_pool=db_pool)

        # Verify embedding service is None
        assert manager.embedding_service is None
        assert not config_manager.is_embedding_enabled()
