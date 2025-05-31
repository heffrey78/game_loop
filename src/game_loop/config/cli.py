"""
Command-line interface for the Game Loop application.
Provides CLI argument parsing and configuration initialization.
"""

import argparse
import logging
from typing import Any

from game_loop.config.manager import ConfigManager

logger = logging.getLogger(__name__)


class ConfigCLI:
    """Command-line interface for the Game Loop application."""

    def __init__(self) -> None:
        """Initialize the CLI parser."""
        self.parser = argparse.ArgumentParser(
            description="Game Loop text adventure system",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self._setup_parser()

    def _setup_parser(self) -> None:
        """Set up the command-line argument parser."""
        # General options
        self.parser.add_argument(
            "--config",
            "-c",
            help="Path to configuration file",
            default=None,
        )
        self.parser.add_argument(
            "--config-dir",
            help="Directory containing configuration files",
            default=None,
        )
        self.parser.add_argument(
            "--verbose",
            "-v",
            action="count",
            default=0,
            help="Increase verbosity (can be used multiple times)",
        )

        # Database options
        db_group = self.parser.add_argument_group("Database options")
        db_group.add_argument(
            "--database.host",
            help="Database host",
            default=None,
        )
        db_group.add_argument(
            "--database.port",
            help="Database port",
            type=int,
            default=None,
        )
        db_group.add_argument(
            "--database.username",
            help="Database username",
            default=None,
        )
        db_group.add_argument(
            "--database.password",
            help="Database password",
            default=None,
        )
        db_group.add_argument(
            "--database.name",
            help="Database name",
            default=None,
            dest="database.database",
        )
        db_group.add_argument(
            "--database.embedding-dimensions",
            help="Vector embedding dimensions",
            type=int,
            default=None,
            dest="database.embedding_dimensions",
        )

        # LLM options
        llm_group = self.parser.add_argument_group("LLM options")
        llm_group.add_argument(
            "--llm.provider",
            help="LLM provider",
            default=None,
        )
        llm_group.add_argument(
            "--llm.base-url",
            help="LLM base URL",
            default=None,
            dest="llm.base_url",
        )
        llm_group.add_argument(
            "--llm.default-model",
            help="Default LLM model for completions",
            default=None,
            dest="llm.default_model",
        )
        llm_group.add_argument(
            "--llm.embedding-model",
            help="Default LLM model for embeddings",
            default=None,
            dest="llm.embedding_model",
        )
        llm_group.add_argument(
            "--llm.timeout",
            help="LLM API timeout in seconds",
            type=float,
            default=None,
        )

        # Ollama options
        ollama_group = self.parser.add_argument_group("Ollama options")
        ollama_group.add_argument(
            "--ollama.temperature",
            help="Ollama temperature",
            type=float,
            default=None,
            # Direct mapping to nested param
            dest="ollama.completion_params.temperature",
        )
        ollama_group.add_argument(
            "--ollama.top-p",
            help="Ollama top-p",
            type=float,
            default=None,
            dest="ollama.completion_params.top_p",  # Direct mapping to nested param
        )
        ollama_group.add_argument(
            "--ollama.max-tokens",
            help="Ollama max tokens",
            type=int,
            default=None,
            # Direct mapping to nested param
            dest="ollama.completion_params.max_tokens",
        )
        ollama_group.add_argument(
            "--ollama.system-prompt",
            help="Ollama system prompt",
            default=None,
            dest="ollama.system_prompt",
        )
        ollama_group.add_argument(
            "--ollama.context-window",
            help="Ollama context window size",
            type=int,
            default=None,
            dest="ollama.context_window",
        )
        ollama_group.add_argument(
            "--ollama.use-gpu",
            help="Use GPU acceleration for Ollama",
            action="store_true",
            default=None,
            dest="ollama.use_gpu",
        )
        ollama_group.add_argument(
            "--ollama.no-gpu",
            help="Disable GPU acceleration for Ollama",
            action="store_false",
            dest="ollama.use_gpu",
        )

        # Prompt template options
        prompt_group = self.parser.add_argument_group("Prompt template options")
        prompt_group.add_argument(
            "--prompts.template-dir",
            help="Directory containing prompt templates",
            default=None,
            dest="prompts.template_dir",
        )
        prompt_group.add_argument(
            "--prompts.default-template",
            help="Default prompt template",
            default=None,
            dest="prompts.default_template",
        )

        # Feature options
        feature_group = self.parser.add_argument_group("Feature options")
        feature_group.add_argument(
            "--features.use-embedding-search",
            help="Enable embedding-based semantic search",
            action="store_true",
            default=None,
            dest="features.use_embedding_search",
        )
        feature_group.add_argument(
            "--features.no-embedding-search",
            help="Disable embedding-based semantic search",
            action="store_false",
            dest="features.use_embedding_search",
        )

    def parse_args(self, args: list[str] | None = None) -> dict[str, Any]:
        """
        Parse command line arguments.

        Args:
            args: Command line arguments to parse, defaults to sys.argv[1:]

        Returns:
            Dictionary of parsed arguments
        """
        parsed_args = self.parser.parse_args(args)

        # Convert to dictionary
        args_dict = vars(parsed_args)

        # Filter out None values
        args_dict = {k: v for k, v in args_dict.items() if v is not None}

        # Special handling for verbosity
        if "verbose" in args_dict:
            verbosity = args_dict.pop("verbose")
            if verbosity > 0:
                log_level = max(logging.DEBUG, logging.WARNING - verbosity * 10)
                args_dict["logging.level"] = logging.getLevelName(log_level)

        return args_dict

    def init_config(self, args: list[str] | None = None) -> ConfigManager:
        """
        Initialize and load configuration from command line arguments.

        Args:
            args: Command line arguments to parse, defaults to sys.argv[1:]

        Returns:
            Initialized ConfigManager
        """
        args_dict = self.parse_args(args)

        # Extract config file and directory
        config_file = args_dict.pop("config", None)
        config_dir = args_dict.pop("config_dir", None)

        # Initialize configuration manager
        return ConfigManager(
            config_file=config_file, config_dir=config_dir, cli_args=args_dict
        )


def get_config_manager(args: list[str] | None = None) -> ConfigManager:
    """
    Helper function to get a configuration manager from command line arguments.

    Args:
        args: Command line arguments to parse, defaults to sys.argv[1:]

    Returns:
        Initialized ConfigManager
    """
    cli = ConfigCLI()
    return cli.init_config(args)
