"""
Integration tests for the Ollama client.
These tests require a running Ollama server.
"""

import os

import pytest
import pytest_asyncio

from game_loop.llm.ollama.client import (
    OllamaClient,
    OllamaEmbeddingConfig,
    OllamaModelParameters,
)

# Skip tests if SKIP_OLLAMA_TESTS environment variable is set
skip_ollama_tests = os.getenv("SKIP_OLLAMA_TESTS", "").lower() in ("1", "true", "yes")
skip_reason = "Ollama integration tests skipped (set SKIP_OLLAMA_TESTS=0 to run)"


@pytest.mark.skipif(skip_ollama_tests, reason=skip_reason)
@pytest.mark.integration
class TestOllamaClient:
    """Integration tests for OllamaClient."""

    @pytest_asyncio.fixture
    async def client(self):
        """Create an OllamaClient instance for testing."""
        async with OllamaClient(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ) as client:
            yield client

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test health check functionality."""
        is_healthy = await client.health_check()
        assert is_healthy, "Ollama server should be healthy"

    @pytest.mark.asyncio
    async def test_list_models(self, client):
        """Test listing available models."""
        models = await client.list_models()
        assert isinstance(models, list), "Should return a list of models"
        print(f"Available models: {[model.get('name') for model in models]}")

    @pytest.mark.asyncio
    async def test_check_model_availability(self, client):
        """Test checking model availability."""
        # Test with a model that should exist
        model_name = os.getenv("TEST_OLLAMA_MODEL", "qwen3:1.7b")
        is_available = await client.check_model_availability(model_name)

        # If the model isn't available, print a message but don't fail the test
        if not is_available:
            pytest.skip(f"Model {model_name} is not available. Test skipped.")
        else:
            assert is_available, f"Model {model_name} should be available"

        # Test with a model that should not exist
        fake_model = "non_existent_model_123456789"
        is_available = await client.check_model_availability(fake_model)
        assert not is_available, f"Model {fake_model} should not be available"

    @pytest.mark.asyncio
    async def test_generate_completion(self, client):
        """Test generating completions."""
        # Skip if no model is available
        model_name = os.getenv("TEST_OLLAMA_MODEL", "qwen3:1.7b")
        if not await client.check_model_availability(model_name):
            pytest.skip(f"Model {model_name} is not available. Test skipped.")

        # Generate a completion
        params = OllamaModelParameters(model=model_name, temperature=0.7, max_tokens=50)

        response = await client.generate_completion(
            "Write a short poem about a dragon.", params
        )

        assert "text" in response, "Response should contain generated text"
        assert response["text"], "Generated text should not be empty"
        print(f"Generated text: {response['text']}")

    @pytest.mark.asyncio
    async def test_generate_embeddings(self, client):
        """Test generating embeddings."""
        # Skip if no model is available
        model_name = os.getenv("TEST_OLLAMA_MODEL", "qwen3:1.7b")
        if not await client.check_model_availability(model_name):
            pytest.skip(f"Model {model_name} is not available. Test skipped.")

        # Generate embeddings
        config = OllamaEmbeddingConfig(model=model_name)

        embedding = await client.generate_embeddings(
            "This is a test sentence for embedding generation.", config
        )

        assert isinstance(embedding, list), "Embedding should be a list"
        assert len(embedding) > 0, "Embedding should not be empty"
        assert all(
            isinstance(x, float) for x in embedding
        ), "All values should be floats"

        print(f"Generated embedding with {len(embedding)} dimensions")
