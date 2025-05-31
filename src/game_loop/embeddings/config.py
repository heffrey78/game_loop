"""
Configuration models for the embedding service.
"""

from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Configuration for embedding service parameters and model settings."""

    model_name: str = Field(
        default="nomic-embed-text",
        description="The model to use for embedding generation",
    )
    max_text_length: int = Field(
        default=512, description="Maximum text length for embedding generation"
    )
    batch_size: int = Field(
        default=10, description="Number of texts to process in a batch"
    )
    cache_enabled: bool = Field(
        default=True, description="Whether to enable embedding caching"
    )
    cache_size: int = Field(
        default=1000,
        description="Maximum number of embeddings to cache in memory",
    )
    retry_attempts: int = Field(
        default=3, description="Number of retry attempts for failed requests"
    )
    retry_delay: float = Field(
        default=1.0, description="Delay between retry attempts in seconds"
    )
    preprocessing_enabled: bool = Field(
        default=True, description="Whether to enable text preprocessing"
    )
    disk_cache_enabled: bool = Field(
        default=True, description="Whether to enable persistent disk caching"
    )
    disk_cache_dir: str | None = Field(
        default=None, description="Directory for disk cache storage"
    )
