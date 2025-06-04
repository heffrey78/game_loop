"""
Configuration models for the Game Loop application.
These models define the configuration schema and validation rules.
"""

from typing import Any

from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    """Configuration for database connection and settings."""

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    username: str = Field(default="postgres", description="Database username")
    password: str = Field(default="postgres", description="Database password")
    database: str = Field(default="game_loop", description="Database name")
    db_schema: str = Field(default="public", description="Database schema")
    pool_size: int = Field(default=5, description="Connection pool size")
    max_overflow: int = Field(default=10, description="Maximum connection overflow")
    pool_timeout: int = Field(default=30, description="Connection pool timeout")
    pool_recycle: int = Field(default=1800, description="Connection pool recycle time")
    echo: bool = Field(default=False, description="Echo SQL statements")
    embedding_dimensions: int = Field(
        default=384, description="Vector embedding dimensions"
    )


class LLMConfig(BaseModel):
    """Configuration for LLM services."""

    provider: str = Field(default="ollama", description="LLM provider (e.g., 'ollama')")
    base_url: str = Field(
        default="http://localhost:11434", description="Base URL for LLM API"
    )
    timeout: float = Field(default=60.0, description="API timeout in seconds")
    default_model: str = Field(
        default="qwen2.5:3b", description="Default model for completions"
    )
    embedding_model: str = Field(
        default="nomic-embed-text", description="Default model for embeddings"
    )
    embedding_dimensions: int | None = Field(
        default=None, description="Dimensions for embeddings"
    )
    max_retries: int = Field(
        default=3, description="Maximum number of retries for failed requests"
    )
    retry_delay: int = Field(default=2, description="Delay between retries in seconds")
    enable_streaming: bool = Field(
        default=True, description="Enable streaming responses"
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
    context_window: int = Field(default=8192, description="Context window size")
    use_gpu: bool = Field(default=True, description="Use GPU acceleration if available")


class PromptTemplateConfig(BaseModel):
    """Configuration for prompt templates."""

    template_dir: str = Field(
        default="prompts", description="Directory containing prompt templates"
    )
    default_template: str = Field(
        default="default.txt", description="Default template to use"
    )


class GameRulesConfig(BaseModel):
    """Configuration for game rules."""

    rules_file: str = Field(default="rules.yaml", description="Path to the rules file")
    enable_dynamic_rules: bool = Field(
        default=True, description="Enable dynamic rule generation"
    )
    max_rules_per_context: int = Field(
        default=5, description="Maximum number of rules to apply per context"
    )
    rule_priority_levels: int = Field(
        default=3, description="Number of priority levels for rules"
    )


class GameFlowConfig(BaseModel):
    """Configuration for game flow and mechanics."""

    enable_evolution: bool = Field(default=True, description="Enable world evolution")
    evolution_interval_minutes: int = Field(
        default=60, description="Interval for evolution events in minutes"
    )
    max_locations_per_region: int = Field(
        default=20, description="Maximum locations per region"
    )
    max_npcs_per_location: int = Field(
        default=5, description="Maximum NPCs per location"
    )
    max_objects_per_location: int = Field(
        default=10, description="Maximum objects per location"
    )


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: str = Field(default="INFO", description="Logging level")
    file: str | None = Field(default=None, description="Log file path")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string",
    )
    enable_console: bool = Field(default=True, description="Enable console logging")


class ActionClassificationConfig(BaseModel):
    """Configuration for action classification system."""

    high_confidence_threshold: float = Field(
        default=0.8, description="Threshold for high confidence classifications"
    )
    rule_confidence_threshold: float = Field(
        default=0.7, description="Threshold for rule-based classification confidence"
    )
    llm_fallback_threshold: float = Field(
        default=0.6, description="Threshold for triggering LLM fallback"
    )
    enable_cache: bool = Field(
        default=True, description="Enable classification result caching"
    )
    cache_size: int = Field(
        default=500, description="Maximum cache size for classifications"
    )
    cache_ttl_seconds: int = Field(
        default=300, description="Cache time-to-live in seconds"
    )


class FeaturesConfig(BaseModel):
    """Configuration for game features."""

    use_nlp: bool = Field(
        default=True, description="Use Natural Language Processing for input"
    )
    use_embedding_search: bool = Field(
        default=False, description="Use embedding-based semantic search"
    )
    enable_conversation_memory: bool = Field(
        default=True, description="Enable conversation history tracking"
    )
    max_conversation_history: int = Field(
        default=5, description="Maximum conversation exchanges to remember"
    )


class GameConfig(BaseModel):
    """Main configuration container for the Game Loop application."""

    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig, description="Database configuration"
    )
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
    ollama: OllamaConfig = Field(
        default_factory=OllamaConfig, description="Ollama configuration"
    )
    prompts: PromptTemplateConfig = Field(
        default_factory=PromptTemplateConfig,
        description="Prompt template configuration",
    )
    game_rules: GameRulesConfig = Field(
        default_factory=GameRulesConfig, description="Game rules configuration"
    )
    game_flow: GameFlowConfig = Field(
        default_factory=GameFlowConfig, description="Game flow configuration"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )
    features: FeaturesConfig = Field(
        default_factory=FeaturesConfig,
        description="Feature flags configuration",
    )
    action_classification: ActionClassificationConfig = Field(
        default_factory=ActionClassificationConfig,
        description="Action classification configuration",
    )

    def sync_embedding_dimensions(self) -> "GameConfig":
        """Set embedding dimensions from database if not in LLM config."""
        if self.llm.embedding_dimensions is None and hasattr(self, "database"):
            self.llm.embedding_dimensions = self.database.embedding_dimensions
        return self
