"""
Configuration manager for the Game Loop application.
Handles loading, merging, and accessing configuration from multiple sources.
"""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast, get_type_hints

import yaml
from pydantic import BaseModel

from game_loop.config.models import (
    GameConfig,
    OllamaConfig,
)

if TYPE_CHECKING:
    from game_loop.embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class ConfigManager:
    """
    Configuration manager for the Game Loop application.

    Handles loading configuration from:
    - Default values
    - Configuration files
    - Environment variables
    - Command line arguments

    Configuration sources are applied in order of increasing precedence.
    """

    ENV_PREFIX = "GAMELOOP_"
    DEFAULT_CONFIG_DIR = "~/.game_loop/config"
    DEFAULT_CONFIG_FILE = "config.yaml"

    def __init__(
        self,
        config_file: str | None = None,
        config_dir: str | None = None,
        cli_args: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the configuration manager.

        Args:
            config_file: Path to the configuration file
            config_dir: Directory containing configuration files
            cli_args: Command line arguments
        """
        self.config_file = config_file
        self.config_dir = config_dir or os.getenv(
            f"{self.ENV_PREFIX}CONFIG_DIR", os.path.expanduser(self.DEFAULT_CONFIG_DIR)
        )
        self.cli_args = cli_args or {}

        # Base directory for resolving relative paths
        self.base_dir = ""
        if config_file:
            self.base_dir = str(Path(config_file).parent.absolute())
        elif config_dir:
            self.base_dir = str(config_dir)  # Ensure str type
        else:
            self.base_dir = os.path.expanduser(self.DEFAULT_CONFIG_DIR)

        # Initialize with default configuration
        self.config = GameConfig()

        # Apply configuration sources in order of increasing precedence
        self._load_from_file()
        self._load_from_env()
        self._load_from_cli()

    def _convert_value(self, field_type: Any, raw_value: Any) -> Any:
        """
        Convert a value to the appropriate type based on field_type.

        Args:
            field_type: The target type to convert to
            raw_value: The raw value to convert

        Returns:
            The converted value or None if conversion fails
        """
        try:
            # Handle string conversion first as it's a common case
            if isinstance(raw_value, str):
                if field_type is bool or field_type is bool | None:
                    return raw_value.lower() in ("1", "true", "yes", "y", "on")
                elif field_type is int or field_type is int | None:
                    return int(raw_value)
                elif field_type is float or field_type is float | None:
                    return float(raw_value)
                elif field_type is str or field_type is str | None:
                    return raw_value
                else:
                    return raw_value  # Fallback
            else:
                # For non-string values, apply appropriate conversions
                if field_type is bool or field_type is bool | None:
                    return bool(raw_value)
                elif field_type is int or field_type is int | None:
                    return int(raw_value)
                elif field_type is float or field_type is float | None:
                    return float(raw_value)
                elif field_type is str or field_type is str | None:
                    return str(raw_value)
                else:
                    return raw_value  # Fallback
        except Exception as e:
            logger.warning(f"Failed to convert '{raw_value}' to {field_type}: {e}")
            return None

    def _load_from_file(self) -> None:
        """Load configuration from a YAML file."""
        if not self.config_file:
            # Ensure self.config_dir is a string
            config_dir = str(self.config_dir) if self.config_dir is not None else ""
            default_path = os.path.join(config_dir, self.DEFAULT_CONFIG_FILE)
            if os.path.exists(default_path):
                self.config_file = default_path
            else:
                logger.debug("No configuration file specified and default not found")
                return

        try:
            with open(self.config_file) as f:
                config_data = yaml.safe_load(f)

            if not config_data:
                logger.debug("Configuration file is empty")
                return

            # Special handling for completion_params to preserve defaults
            if "ollama" in config_data and "completion_params" in config_data["ollama"]:
                # Create a copy of the default completion params
                default_params = OllamaConfig().completion_params.copy()
                # Update with values from file
                default_params.update(config_data["ollama"]["completion_params"])
                # Set back to config data
                config_data["ollama"]["completion_params"] = default_params

            # Use the model_validate method to validate and load from dict
            if config_data:
                self.config = GameConfig.model_validate(config_data)
                logger.info(f"Loaded configuration from {self.config_file}")

        except (OSError, yaml.YAMLError) as e:
            logger.warning(f"Failed to load configuration from {self.config_file}: {e}")

    def _load_from_env(self) -> None:
        """
        Load configuration from environment variables.

        Environment variables should be in the format:
        GAMELOOP_SECTION_KEY=value

        For example:
        GAMELOOP_DATABASE_HOST=localhost
        GAMELOOP_LLM_DEFAULT_MODEL=qwen3:1.7b
        """
        env_vars = {
            k: v for k, v in os.environ.items() if k.startswith(self.ENV_PREFIX)
        }

        # Process each environment variable
        for env_name, env_value in env_vars.items():
            # Remove prefix and split into parts
            key_parts = env_name[len(self.ENV_PREFIX) :].lower().split("_")

            if len(key_parts) < 2:
                continue

            section, *subparts = key_parts
            key = "_".join(subparts)

            # Special handling for Ollama completion parameters
            if section == "ollama" and key == "temperature":
                try:
                    # Parse temperature value as float
                    temp_value = float(env_value)
                    self.config.ollama.completion_params["temperature"] = temp_value
                    logger.debug(
                        f"Set ollama.completion_params.temperature from "
                        f"environment variable {env_name}"
                    )
                    continue
                except (ValueError, AttributeError) as e:
                    logger.warning(
                        f"Failed to set ollama.completion_params.temperature from "
                        f"environment variable {env_name}: {e}"
                    )
                    continue

            # Find the appropriate section in the config
            section_config = getattr(self.config, section, None)
            if not section_config or not isinstance(section_config, BaseModel):
                continue

            # Try to set the value on the section
            try:
                # Get the field info to determine the type
                model_fields = getattr(section_config.__class__, "model_fields", {})
                field_info = model_fields.get(key)

                if not field_info:
                    logger.warning(f"Field '{key}' not found in section '{section}'")
                    continue

                # Get the annotation from type hints to determine the type
                annotations = get_type_hints(section_config.__class__)
                if key not in annotations:
                    logger.warning(
                        f"Type hint for '{key}' not found in section '{section}'"
                    )
                    continue

                field_type = annotations[key]

                # Convert the value to the appropriate type
                converted_value = self._convert_value(field_type, env_value)
                if converted_value is not None:
                    setattr(section_config, key, converted_value)
                    logger.debug(
                        f"Set {section}.{key} from environment variable {env_name}"
                    )

            except (ValueError, AttributeError) as e:
                logger.warning(
                    f"Failed to set {section}.{key} from environment variable "
                    f"{env_name}: {e}"
                )

    def _load_from_cli(self) -> None:
        """
        Load configuration from command line arguments.

        Command line arguments should be in the format:
        --section.key=value

        For example:
        --database.host=localhost
        --llm.default-model=qwen3:1.7b
        """
        if not self.cli_args:
            return

        for arg_name, arg_value in self.cli_args.items():
            if not arg_name or arg_value is None:
                continue

            # Split into section and key
            if "." not in arg_name:
                continue

            section, key = arg_name.split(".", 1)
            key = key.replace("-", "_")  # Convert kebab-case to snake_case

            # Special handling for Ollama completion parameters
            if section == "ollama" and key.startswith("completion_params."):
                param_key = key[len("completion_params.") :]
                try:
                    # Ensure the completion_params dict exists
                    if not hasattr(self.config.ollama, "completion_params"):
                        self.config.ollama.completion_params = {}

                    # Parse the value based on likely type
                    if param_key in ["temperature", "top_p", "top_k"]:
                        self.config.ollama.completion_params[param_key] = float(
                            arg_value
                        )
                    elif param_key in ["max_tokens"]:
                        self.config.ollama.completion_params[param_key] = int(arg_value)
                    else:
                        self.config.ollama.completion_params[param_key] = arg_value

                    logger.debug(
                        f"Set ollama.completion_params.{param_key} from "
                        f"CLI argument {arg_name}"
                    )
                    continue
                except (ValueError, AttributeError) as e:
                    logger.warning(
                        f"Failed to set ollama.completion_params.{param_key} from "
                        f"CLI argument {arg_name}: {e}"
                    )
                    continue

            # Find the appropriate section in the config
            section_config = getattr(self.config, section, None)
            if not section_config or not isinstance(section_config, BaseModel):
                logger.warning(
                    f"Section '{section}' not found for CLI argument '{arg_name}'"
                )
                continue

            # Try to set the value on the section
            try:
                # Get the field info to determine the type
                model_fields = getattr(section_config.__class__, "model_fields", {})
                field_info = model_fields.get(key)

                if not field_info:
                    logger.warning(
                        f"Field '{key}' not found in section '{section}' for "
                        f"CLI argument '{arg_name}'"
                    )
                    continue

                # Get the annotation from type hints to determine the type
                annotations = get_type_hints(section_config.__class__)
                if key not in annotations:
                    logger.warning(
                        f"Type hint for '{key}' not found in section '{section}' for "
                        f"CLI argument '{arg_name}'"
                    )
                    continue

                field_type = annotations[key]

                # Convert the value using our helper method
                converted_value = self._convert_value(field_type, arg_value)
                if converted_value is not None:
                    setattr(section_config, key, converted_value)
                    logger.debug(f"Set {section}.{key} from CLI argument {arg_name}")

            except (ValueError, AttributeError) as e:
                logger.warning(
                    f"Failed to set {section}.{key} from CLI argument {arg_name}: {e}"
                )

    def get_config(self) -> GameConfig:
        """Get the complete configuration."""
        return self.config

    def get_section(self, section_name: str, model_class: type[T] | None = None) -> Any:
        """
        Get a specific configuration section.

        Args:
            section_name: Name of the section to get
            model_class: Expected model class for the section (optional)

        Returns:
            The configuration section, cast to the specified model class if provided
        """
        section = getattr(self.config, section_name, None)
        if section is None:
            raise ValueError(f"Configuration section '{section_name}' not found")

        # Return with the correct cast if model_class is provided
        if model_class is not None:
            return cast(T, section)

        # Otherwise, return the section as is
        return section

    def resolve_path(self, path: str) -> str:
        """
        Resolve a path relative to the base directory.

        Args:
            path: Path to resolve

        Returns:
            Absolute path
        """
        # If it's already absolute, use as-is
        if os.path.isabs(path):
            return path

        # If we have a config file, use its directory as base
        if self.config_file:
            config_dir = os.path.dirname(self.config_file)
            return os.path.abspath(os.path.join(config_dir, path))

        # Otherwise, use the config directory
        if self.config_dir is None:
            # Fallback or raise error if config_dir is essential and None
            raise ValueError("Config directory is not set, cannot resolve path.")
        return os.path.abspath(os.path.join(self.config_dir, path))

    def create_embedding_service(self) -> "EmbeddingService":
        """
        Create embedding service with current configuration.

        Returns:
            Configured EmbeddingService instance

        Raises:
            ImportError: If embedding service dependencies are not available
            ValueError: If configuration is invalid for embedding service
        """
        try:
            from game_loop.embeddings.service import EmbeddingService

            # Pass this ConfigManager directly - no bridge needed
            return EmbeddingService(config_manager=self)
        except ImportError as e:
            raise ImportError(
                f"Failed to import EmbeddingService: {e}. "
                "Ensure embedding dependencies are installed."
            ) from e
        except Exception as e:
            raise ValueError(f"Failed to create EmbeddingService: {e}") from e

    def is_embedding_enabled(self) -> bool:
        """
        Check if embedding functionality is enabled via feature flags.

        Returns:
            True if embedding search feature is enabled, False otherwise
        """
        try:
            return self.config.features.use_embedding_search
        except AttributeError:
            return False

    def get_prompt_template(self, template_name: str) -> str:
        """
        Load and cache a prompt template.

        Args:
            template_name: Name of the template to load

        Returns:
            Template content as string

        Raises:
            ValueError: If template cannot be loaded
        """
        if not hasattr(self, "_prompt_templates"):
            self._prompt_templates: dict[str, str] = {}

        # If template is already loaded, return it
        if template_name in self._prompt_templates:
            return self._prompt_templates[template_name]

        # Build template path
        template_dir = self.config.prompts.template_dir
        if not os.path.isabs(template_dir):
            template_dir = self.resolve_path(template_dir)

        template_path = os.path.join(template_dir, template_name)
        if not template_path.endswith(".txt"):
            template_path += ".txt"

        # Load and cache template
        try:
            with open(template_path) as f:
                template_content = f.read()
            self._prompt_templates[template_name] = template_content
            return template_content
        except OSError as e:
            raise ValueError(f"Failed to load template {template_name}: {e}") from e

    def format_prompt(self, template_name: str, **kwargs: Any) -> str:
        """
        Load and format a prompt template with variables.

        Args:
            template_name: Name of the template to use
            **kwargs: Variables to substitute in the template

        Returns:
            Formatted prompt text

        Raises:
            ValueError: If template loading or formatting fails
        """
        template = self.get_prompt_template(template_name)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(
                f"Missing required variable in template '{template_name}': {e}"
            ) from e

    def to_dict(self) -> dict[str, Any]:
        """Convert the configuration to a dictionary."""
        # Explicitly cast the result to dict[str, Any]
        return dict(self.config.model_dump())

    def to_yaml(self) -> str:
        """Convert the configuration to YAML."""
        return yaml.dump(self.to_dict())
