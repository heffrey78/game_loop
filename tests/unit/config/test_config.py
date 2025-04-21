"""
Tests for the configuration system.
"""

import os
import tempfile
from unittest import mock

import pytest
import yaml

from game_loop.config import (
    ConfigCLI,
    ConfigManager,
    DatabaseConfig,
    GameConfig,
    LLMConfig,
    get_config_manager,
)


class TestConfigModels:
    """Tests for the configuration models."""

    def test_default_game_config(self):
        """Test that the default game configuration has expected values."""
        config = GameConfig()
        assert config.llm.provider == "ollama"
        assert config.llm.base_url == "http://localhost:11434"
        assert config.database.host == "localhost"
        assert config.database.port == 5432
        assert config.ollama.completion_params["temperature"] == 0.7
        assert config.database.embedding_dimensions == 384
        assert config.llm.embedding_dimensions is None  # Should be None by default

    def test_embedding_dimensions_validator(self):
        """Test that the embedding dimensions validator works."""
        config = GameConfig()
        # When accessing llm, the validator should set embedding_dimensions
        # from database
        assert config.llm.embedding_dimensions is None

        # Set the database embedding dimensions
        config.database.embedding_dimensions = 384
        config.sync_embedding_dimensions()

        # After validation, the llm.embedding_dimensions should match
        # database.embedding_dimensions
        config_dict = config.model_dump()
        assert config_dict["llm"]["embedding_dimensions"] == 384


class TestConfigManager:
    """Tests for the configuration manager."""

    def test_init_with_defaults(self):
        """Test initializing the configuration manager with defaults."""
        manager = ConfigManager()
        config = manager.get_config()
        assert isinstance(config, GameConfig)
        assert config.llm.provider == "ollama"
        assert config.database.host == "localhost"

    def test_load_from_file(self):
        """Test loading configuration from a file."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as temp:
            config_data = {
                "llm": {
                    "provider": "test-provider",
                    "default_model": "test-model",
                },
                "database": {
                    "host": "test-host",
                    "port": 1234,
                },
                "ollama": {
                    "completion_params": {
                        "temperature": 0.5,
                    },
                },
            }
            yaml.dump(config_data, temp)
            temp_path = temp.name

        try:
            # Load configuration from the file
            manager = ConfigManager(config_file=temp_path)
            config = manager.get_config()

            # Check that values were loaded correctly
            assert config.llm.provider == "test-provider"
            assert config.llm.default_model == "test-model"
            assert config.database.host == "test-host"
            assert config.database.port == 1234
            assert config.ollama.completion_params["temperature"] == 0.5
            # Check that other values still have defaults
            assert config.database.username == "postgres"
            assert config.ollama.completion_params["top_p"] == 0.9
        finally:
            # Clean up temporary file
            os.unlink(temp_path)

    def test_get_section(self):
        """Test getting a configuration section."""
        manager = ConfigManager()

        # Get a section with the correct type
        llm_config = manager.get_section("llm", LLMConfig)
        assert isinstance(llm_config, LLMConfig)
        assert llm_config.provider == "ollama"

        # Get a section with the base type
        db_config = manager.get_section("database")
        assert isinstance(db_config, DatabaseConfig)
        assert db_config.host == "localhost"

        # Try getting a non-existent section
        with pytest.raises(ValueError):
            manager.get_section("nonexistent")

    @mock.patch.dict(
        os.environ,
        {
            "GAMELOOP_LLM_PROVIDER": "env-provider",
            "GAMELOOP_DATABASE_HOST": "env-host",
            "GAMELOOP_OLLAMA_TEMPERATURE": "0.3",
            "GAMELOOP_DATABASE_PORT": "4321",
        },
    )
    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        manager = ConfigManager()
        config = manager.get_config()

        # Check that values were loaded from environment variables
        assert config.llm.provider == "env-provider"
        assert config.database.host == "env-host"
        assert config.database.port == 4321
        assert config.ollama.completion_params["temperature"] == 0.3

    def test_load_from_cli(self):
        """Test loading configuration from CLI arguments."""
        cli_args = {
            "llm.provider": "cli-provider",
            "database.host": "cli-host",
            "ollama.completion_params.temperature": 0.1,
            "database.port": 5678,
        }

        manager = ConfigManager(cli_args=cli_args)
        config = manager.get_config()

        # Check that values were loaded from CLI arguments
        assert config.llm.provider == "cli-provider"
        assert config.database.host == "cli-host"
        assert config.database.port == 5678
        assert config.ollama.completion_params["temperature"] == 0.1

    @mock.patch.dict(
        os.environ,
        {
            "GAMELOOP_LLM_PROVIDER": "env-provider",
            "GAMELOOP_DATABASE_HOST": "env-host",
        },
    )
    def test_precedence(self):
        """Test that configuration sources have the correct precedence."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as temp:
            config_data = {
                "llm": {
                    "provider": "file-provider",
                    "default_model": "file-model",
                },
                "database": {
                    "host": "file-host",
                    "port": 1234,
                },
            }
            yaml.dump(config_data, temp)
            temp_path = temp.name

        try:
            # CLI args should override environment variables and file
            cli_args = {
                "llm.provider": "cli-provider",
                "database.port": 5678,
            }

            manager = ConfigManager(config_file=temp_path, cli_args=cli_args)
            config = manager.get_config()

            # Check precedence: CLI > Environment > File > Default
            assert config.llm.provider == "cli-provider"  # From CLI
            assert config.database.host == "env-host"  # From environment
            assert config.database.port == 5678  # From CLI
            assert config.llm.default_model == "file-model"  # From file
            assert config.database.username == "postgres"  # Default
        finally:
            # Clean up temporary file
            os.unlink(temp_path)

    def test_resolve_path(self):
        """Test resolving paths relative to the base directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ConfigManager(config_dir=temp_dir)

            # Test relative path
            rel_path = "relative/path"
            abs_path = manager.resolve_path(rel_path)
            expected_path = os.path.normpath(os.path.join(temp_dir, rel_path))
            assert abs_path == expected_path

            # Test absolute path
            abs_path_input = "/absolute/path"
            abs_path_output = manager.resolve_path(abs_path_input)
            assert abs_path_output == abs_path_input


class TestConfigCLI:
    """Tests for the configuration CLI."""

    def test_parse_args(self):
        """Test parsing command line arguments."""
        cli = ConfigCLI()
        args = cli.parse_args(
            [
                "--config",
                "test-config.yaml",
                "--llm.provider",
                "cli-provider",
                "--database.host",
                "cli-host",
                "--database.port",
                "9876",
                "--ollama.temperature",
                "0.2",
                "--ollama.system-prompt",
                "Test prompt",
                "--verbose",
            ]
        )

        # Check that arguments were parsed correctly
        assert args["config"] == "test-config.yaml"
        assert args["llm.provider"] == "cli-provider"
        assert args["database.host"] == "cli-host"
        assert args["database.port"] == 9876
        assert args["ollama.completion_params.temperature"] == 0.2
        assert args["ollama.system_prompt"] == "Test prompt"
        assert "logging.level" in args

    def test_verbosity(self):
        """Test that verbosity is converted to logging level."""
        cli = ConfigCLI()

        # No verbosity
        args = cli.parse_args([])
        assert "logging.level" not in args

        # Single verbosity
        args = cli.parse_args(["-v"])
        assert args["logging.level"] == "INFO"

        # Double verbosity
        args = cli.parse_args(["-vv"])
        assert args["logging.level"] == "DEBUG"

    @mock.patch("game_loop.config.cli.ConfigManager")
    def test_init_config(self, mock_config_manager):
        """Test initializing configuration from command line arguments."""
        cli = ConfigCLI()
        cli.init_config(
            ["--config", "test-config.yaml", "--llm.provider", "test-provider"]
        )

        # Check that ConfigManager was called with the right arguments
        mock_config_manager.assert_called_once()
        args = mock_config_manager.call_args[1]
        assert args["config_file"] == "test-config.yaml"
        assert "llm.provider" in args["cli_args"]
        assert args["cli_args"]["llm.provider"] == "test-provider"

    @mock.patch("game_loop.config.cli.ConfigCLI")
    def test_get_config_manager(self, mock_config_cli):
        """Test the get_config_manager helper function."""
        mock_instance = mock_config_cli.return_value
        get_config_manager(["--config", "test-config.yaml"])

        # Check that ConfigCLI was initialized and init_config was called
        mock_config_cli.assert_called_once()
        mock_instance.init_config.assert_called_once_with(
            ["--config", "test-config.yaml"]
        )
