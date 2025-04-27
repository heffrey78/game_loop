"""
Unit tests for the LLM configuration manager.
"""

import os
import tempfile
from pathlib import Path

import yaml

from game_loop.llm.config import (
    ConfigManager,
)


class TestConfigManager:
    """Tests for the ConfigManager class."""

    def test_default_config(self):
        """Test default configuration."""
        config = ConfigManager()
        assert config.llm_config.provider == "ollama"
        assert config.llm_config.default_model == "qwen2.5:3b"
        assert config.ollama_config.completion_params["temperature"] == 0.7

    def test_load_config_from_yaml(self):
        """Test loading configuration from a YAML file."""
        # Create a temporary config file
        config_data = {
            "llm": {
                "provider": "test-provider",
                "base_url": "http://test-url:11434",
                "timeout": 30.0,
                "default_model": "test-model",
                "embedding_model": "embed-model",
            },
            "ollama": {
                "completion_params": {
                    "temperature": 0.5,
                    "top_p": 0.8,
                    "max_tokens": 500,
                },
                "system_prompt": "This is a test system prompt.",
            },
            "prompts": {
                "template_dir": "/test/templates",
                "default_template": "test.txt",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name

        try:
            # Load the config file
            config = ConfigManager(config_file=config_file)

            # Check LLM config
            assert config.llm_config.provider == "test-provider"
            assert config.llm_config.base_url == "http://test-url:11434"
            assert config.llm_config.timeout == 30.0
            assert config.llm_config.default_model == "test-model"
            assert config.llm_config.embedding_model == "embed-model"

            # Check Ollama config
            assert config.ollama_config.completion_params["temperature"] == 0.5
            assert config.ollama_config.completion_params["top_p"] == 0.8
            assert config.ollama_config.completion_params["max_tokens"] == 500
            assert config.ollama_config.system_prompt == "This is a test system prompt."

            # Check prompt config
            assert config.prompt_config.template_dir == "/test/templates"
            assert config.prompt_config.default_template == "test.txt"
        finally:
            # Clean up the temporary file
            os.unlink(config_file)

    def test_merge_with_env(self, monkeypatch):
        """Test merging configuration with environment variables."""
        # Set environment variables
        monkeypatch.setenv("GAMELOOP_LLM_PROVIDER", "env-provider")
        monkeypatch.setenv("GAMELOOP_LLM_BASE_URL", "http://env-url:11434")
        monkeypatch.setenv("GAMELOOP_LLM_DEFAULT_MODEL", "env-model")
        monkeypatch.setenv("GAMELOOP_OLLAMA_TEMPERATURE", "0.3")

        # Create config manager and merge with env
        config = ConfigManager()
        config.merge_with_env()

        # Check that environment variables took precedence
        assert config.llm_config.provider == "env-provider"
        assert config.llm_config.base_url == "http://env-url:11434"
        assert config.llm_config.default_model == "env-model"
        assert config.ollama_config.completion_params["temperature"] == 0.3

    def test_prompt_template(self):
        """Test prompt template loading and formatting."""
        # Create a temporary template file
        template_content = "Hello, {name}! Welcome to {game}."

        with tempfile.TemporaryDirectory() as temp_dir:
            template_dir = Path(temp_dir)
            template_file = template_dir / "test_template.txt"

            with open(template_file, "w") as f:
                f.write(template_content)

            # Create config with the template directory
            config = ConfigManager()
            config.prompt_config.template_dir = str(temp_dir)

            # Test getting the template
            template = config.get_prompt_template("test_template")
            assert template == template_content

            # Test formatting the template
            formatted = config.format_prompt(
                "test_template", name="Player", game="Adventure"
            )
            assert formatted == "Hello, Player! Welcome to Adventure."
