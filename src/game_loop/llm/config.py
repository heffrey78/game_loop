"""
Configuration manager for LLM settings.
Provides a centralized way to manage and validate LLM configuration.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Main configuration for LLM services."""

    provider: str = Field(default="ollama", description="LLM provider (e.g., 'ollama')")
    base_url: str = Field(
        default="http://localhost:11434", description="Base URL for LLM API"
    )
    timeout: float = Field(default=60.0, description="API timeout in seconds")
    default_model: str = Field(
        default="qwen3:1.7b", description="Default model for completions"
    )
    embedding_model: str = Field(
        default="nomic-embed-text", description="Default model for embeddings"
    )
    embedding_dimensions: int | None = Field(
        default=None, description="Dimensions for embeddings"
    )


class OllamaConfig(BaseModel):
    """Ollama-specific configuration."""

    completion_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": 1024,
        },
        description="Default parameters for completions",
    )
    system_prompt: str | None = Field(default=None, description="Default system prompt")


class PromptTemplateConfig(BaseModel):
    """Configuration for prompt templates."""

    template_dir: str = Field(
        default="prompts", description="Directory containing prompt templates"
    )
    default_template: str = Field(
        default="default.txt", description="Default template to use"
    )


class ConfigManager:
    """Manager for LLM configuration and prompt templates."""

    def __init__(
        self, config_file: str | None = None, config_dir: str | None = None
    ) -> None:
        """
        Initialize the configuration manager.

        Args:
            config_file: Path to the configuration file
            config_dir: Directory containing configuration files
        """
        self.config_file = config_file
        self.config_dir = config_dir or os.getenv(
            "GAMELOOP_CONFIG_DIR", str(Path.home() / ".game_loop" / "config")
        )

        # Store the base directory where config_file is located (if provided)
        self.base_dir = None
        if config_file:
            self.base_dir = str(Path(config_file).parent)

        self.llm_config = LLMConfig()
        self.ollama_config = OllamaConfig()
        self.prompt_config = PromptTemplateConfig()
        self._prompt_templates: dict[str, str] = {}

        # Load configuration if a file is specified
        if config_file:
            self.load_config(config_file)

    def load_config(self, config_file: str) -> None:
        """
        Load configuration from a YAML file.

        Args:
            config_file: Path to the configuration file
        """
        try:
            with open(config_file) as f:
                config_data = yaml.safe_load(f)

            if not config_data:
                return

            # Update LLM configuration
            if "llm" in config_data:
                self.llm_config = LLMConfig(**config_data["llm"])

            # Update Ollama configuration
            if "ollama" in config_data:
                self.ollama_config = OllamaConfig(**config_data["ollama"])

            # Update prompt configuration
            if "prompts" in config_data:
                self.prompt_config = PromptTemplateConfig(**config_data["prompts"])

        except (OSError, yaml.YAMLError) as e:
            raise ValueError(
                f"Failed to load configuration from {config_file}: {e}"
            ) from e

    def get_prompt_template(self, template_name: str) -> str:
        """
        Get a prompt template by name.

        Args:
            template_name: Name of the template to load

        Returns:
            The prompt template text
        """
        # If template is already loaded, return it
        if template_name in self._prompt_templates:
            return self._prompt_templates[template_name]

        # Otherwise, try to load it
        template_path = self._get_template_path(template_name)
        try:
            with open(template_path) as f:
                template_content = f.read()
            self._prompt_templates[template_name] = template_content
            return template_content
        except OSError as e:
            raise ValueError(f"Failed to load template {template_name}: {e}") from e

    def _get_template_path(self, template_name: str) -> str:
        """
        Get the full path for a template.

        Args:
            template_name: Name of the template

        Returns:
            Full path to the template file
        """
        # Get template directory from configuration
        template_dir_path = Path(self.prompt_config.template_dir)

        # Handle relative paths based on config file location if available
        if not template_dir_path.is_absolute():
            if self.base_dir:
                # If we have a base_dir (from config file), use it directly
                template_dir_path = Path(self.base_dir) / template_dir_path
            else:
                # Otherwise use the default config directory
                # Ensure config_dir is not None before passing to Path
                config_dir = self.config_dir if self.config_dir is not None else ""
                template_dir_path = Path(config_dir) / template_dir_path

        # Create full path to the template
        template_path = template_dir_path / template_name

        # Add .txt extension if not present
        if not template_path.suffix:
            template_path = template_path.with_suffix(".txt")

        return str(template_path)

    def format_prompt(self, template_name: str, **kwargs: Any) -> str:
        """
        Load and format a prompt template with the provided variables.

        Args:
            template_name: Name of the template to use
            **kwargs: Variables to substitute in the template

        Returns:
            Formatted prompt text
        """
        try:
            template = self.get_prompt_template(template_name)
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(
                f"Missing required variable in template '{template_name}': {e}"
            ) from e
        except Exception as e:
            raise ValueError(f"Error formatting template '{template_name}': {e}") from e

    def merge_with_env(self) -> None:
        """
        Merge configuration with environment variables.

        Environment variables take precedence over file configuration.
        """
        # LLM config from environment variables
        if provider := os.getenv("GAMELOOP_LLM_PROVIDER"):
            self.llm_config.provider = provider

        if base_url := os.getenv("GAMELOOP_LLM_BASE_URL"):
            self.llm_config.base_url = base_url

        if timeout := os.getenv("GAMELOOP_LLM_TIMEOUT"):
            self.llm_config.timeout = float(timeout)

        if default_model := os.getenv("GAMELOOP_LLM_DEFAULT_MODEL"):
            self.llm_config.default_model = default_model

        if embedding_model := os.getenv("GAMELOOP_LLM_EMBEDDING_MODEL"):
            self.llm_config.embedding_model = embedding_model

        # Ollama config from environment variables
        if system_prompt := os.getenv("GAMELOOP_OLLAMA_SYSTEM_PROMPT"):
            self.ollama_config.system_prompt = system_prompt

        if temperature := os.getenv("GAMELOOP_OLLAMA_TEMPERATURE"):
            self.ollama_config.completion_params["temperature"] = float(temperature)

        if max_tokens := os.getenv("GAMELOOP_OLLAMA_MAX_TOKENS"):
            self.ollama_config.completion_params["max_tokens"] = int(max_tokens)
