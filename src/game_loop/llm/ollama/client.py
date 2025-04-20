"""
Ollama API client for the game-loop project.
Provides communication with the Ollama API for text generation and embeddings.
"""

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class OllamaModelParameters(BaseModel):
    """Configuration parameters for Ollama API calls."""

    model: str = Field(default="llama3", description="The model to use for generation")
    temperature: float = Field(
        default=0.7, description="Sampling temperature between 0 and 1"
    )
    top_p: float = Field(
        default=0.9,
        description="Limit to the highest probability tokens with combined "
        "probability of top_p",
    )
    top_k: int = Field(
        default=40, description="Limit to the top k highest probability tokens"
    )
    max_tokens: int = Field(default=1024, description="Maximum tokens to generate")
    system_prompt: str | None = Field(
        default=None, description="System prompt for setting context"
    )
    stream: bool = Field(default=False, description="Whether to stream the response")
    num_ctx: int | None = Field(
        default=None, description="Context window size in tokens"
    )


class OllamaEmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""

    model: str = Field(
        default="nomic-embed-text", description="The model to use for embeddings"
    )
    dimensions: int | None = Field(
        default=None, description="Dimensions of the embeddings"
    )


class OllamaClient:
    """Client for interacting with the Ollama API."""

    def __init__(
        self, base_url: str = "http://localhost:11434", timeout: float = 60.0
    ) -> None:
        """
        Initialize the Ollama API client.

        Args:
            base_url: Base URL for the Ollama API
            timeout: Timeout for API requests in seconds
        """
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def __aenter__(self) -> "OllamaClient":
        """Async context manager entry."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Async context manager exit."""
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def list_models(self) -> list[dict[str, Any]]:
        """
        List available models.

        Returns:
            List of model information dictionaries
        """
        url = f"{self.base_url}/api/tags"
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
            # Explicitly ensure we're returning list[dict[str, Any]]
            return [dict(model) for model in models]
        except httpx.HTTPError as e:
            logger.error(f"Failed to list models: {e}")
            raise

    async def check_model_availability(self, model_name: str) -> bool:
        """
        Check if a model is available.

        Args:
            model_name: Name of the model to check

        Returns:
            True if the model is available, False otherwise
        """
        try:
            models = await self.list_models()
            return any(model.get("name") == model_name for model in models)
        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")
            return False

    async def generate_completion(
        self,
        prompt: str,
        params: OllamaModelParameters | None = None,
        raw_response: bool = False,
    ) -> dict[str, Any]:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The prompt to generate from
            params: Generation parameters
            raw_response: If True, return the raw response

        Returns:
            Response from Ollama API
        """
        params = params or OllamaModelParameters()
        url = f"{self.base_url}/api/generate"

        payload: dict[str, Any] = {
            "model": params.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "top_k": params.top_k,
                "num_predict": params.max_tokens,
            },
        }

        if params.system_prompt:
            payload["system"] = params.system_prompt

        if params.num_ctx:
            payload["options"]["num_ctx"] = params.num_ctx

        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()

            if raw_response:
                # Ensure we return a dict[str, Any] even for raw responses
                return dict(result)

            return {
                "text": result.get("response", ""),
                "model": result.get("model", params.model),
                "total_duration": result.get("total_duration", 0),
                "load_duration": result.get("load_duration", 0),
                "eval_count": result.get("eval_count", 0),
                "eval_duration": result.get("eval_duration", 0),
            }
        except httpx.HTTPError as e:
            logger.error(f"Failed to generate completion: {e}")
            raise

    async def stream_completion(
        self,
        prompt: str,
        params: OllamaModelParameters | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Stream a completion for the given prompt.

        Args:
            prompt: The prompt to generate from
            params: Generation parameters

        Yields:
            Chunks of the generated text
        """
        params = params or OllamaModelParameters(stream=True)
        url = f"{self.base_url}/api/generate"

        payload: dict[str, Any] = {
            "model": params.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "top_k": params.top_k,
                "num_predict": params.max_tokens,
            },
        }

        if params.system_prompt:
            payload["system"] = params.system_prompt

        if params.num_ctx:
            payload["options"]["num_ctx"] = params.num_ctx

        try:
            async with self.client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            chunk = json.loads(line)
                            yield chunk
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse streaming response: {line}")
                            continue
        except httpx.HTTPError as e:
            logger.error(f"Failed to stream completion: {e}")
            raise

    async def generate_embeddings(
        self, text: str, config: OllamaEmbeddingConfig | None = None
    ) -> list[float]:
        """
        Generate embeddings for the given text.

        Args:
            text: The text to generate embeddings for
            config: Embedding configuration

        Returns:
            List of embedding values
        """
        config = config or OllamaEmbeddingConfig()
        url = f"{self.base_url}/api/embed"

        payload: dict[str, Any] = {
            "model": config.model,
            "input": text,
        }

        if config.dimensions:
            payload["dimensions"] = config.dimensions

        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()

            # Handle the response correctly - embeddings is a nested array in the
            # response
            if (
                "embeddings" in result
                and isinstance(result["embeddings"], list)
                and result["embeddings"]
            ):
                return list(result["embeddings"][0])  # Ensure list[float] type
            elif "embedding" in result:  # Fall back to the old format just in case
                return list(result.get("embedding", []))  # Ensure list[float] type
            else:
                logger.warning("No embeddings found in the response")
                return []
        except httpx.HTTPError as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

    async def health_check(self) -> bool:
        """
        Check if the Ollama API is up and running.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Just try to list models as a health check
            await self.list_models()
            return True
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
