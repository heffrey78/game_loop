#!/usr/bin/env python3
"""
Test script to demonstrate the updated embedding functionality with dedicated
embedding models. This script generates embeddings for several sentences and
compares their similarities.
"""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from game_loop.llm.ollama.client import OllamaClient, OllamaEmbeddingConfig

# Add project root to Python path to allow imports
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Cosine similarity score (between -1 and 1)
    """
    v1_array = np.array(v1)
    v2_array = np.array(v2)

    dot_product = np.dot(v1_array, v2_array)
    norm_v1 = np.linalg.norm(v1_array)
    norm_v2 = np.linalg.norm(v2_array)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    return dot_product / (norm_v1 * norm_v2)


async def _direct_api_call(client, text):
    """Test direct API call to diagnose any issues."""
    print("\n=== Testing direct API call ===")
    url = f"{client.base_url}/api/embed"
    payload = {"model": "nomic-embed-text", "input": text}

    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = await client.client.post(url, json=payload)
        response_json = response.json()
        print(f"Status code: {response.status_code}")
        print(
            f"Response: {json.dumps(response_json, indent=2)[:1000]}..."
        )  # Truncate if too large

        # Check for embeddings in the correct format
        if (
            "embeddings" in response_json
            and isinstance(response_json["embeddings"], list)
            and response_json["embeddings"]
        ):
            return response_json["embeddings"][0]  # Return the first embedding array
        elif "embedding" in response_json:
            return response_json.get("embedding", [])
        else:
            print("Warning: No embeddings found in the response")
            return []
    except Exception as e:
        print(f"Error during direct API call: {str(e)}")
        return []


@pytest.mark.asyncio
async def test_embeddings():
    """Test embedding generation and similarity comparison."""
    print("\n==== Ollama Embedding Model Demonstration ====\n")

    # Create client
    async with OllamaClient() as client:

        # First check if the embedding model is available
        embedding_model = "nomic-embed-text"
        is_available = await client.check_model_availability(embedding_model)

        if not is_available:
            print(f"Model '{embedding_model}' is not available. You can pull it using:")
            print(f"  ollama pull {embedding_model}")

            # Try to find a different embedding model
            models = await client.list_models()
            model_names = [model.get("name") for model in models]

            for model_name in model_names:
                if any(
                    name in model_name.lower()
                    for name in ["embed", "nomic", "minilm", "mxbai"]
                ):
                    embedding_model = model_name
                    print(f"Using alternative embedding model: {embedding_model}")
                    break
            else:
                print("No suitable embedding model found. Please install one.")
                return

        # Print available models
        models = await client.list_models()
        print("\nAvailable models:")
        for model in models:
            print(f"- {model.get('name')}")

        # Test sentences to compare
        sentences = [
            "The player walks through the dark forest.",
            "A character moves through a shadowy woodland.",
            "The hero traverses the gloomy woods.",
            "The adventurer examines the ancient artifact.",
            "The dragon flies over the mountain.",
        ]

        # Make a direct API call to debug
        direct_embedding = await _direct_api_call(client, sentences[0])
        if not direct_embedding:
            print(
                "\nDirect API call returned empty embeddings. Check Ollama server "
                "and model."
            )
            return

        print(
            f"\nDirect API call successful! Got embedding with "
            f"{len(direct_embedding)} dimensions."
        )

        print(f"\nGenerating embeddings using the '{embedding_model}' model...")

        # Generate embeddings for all sentences
        config = OllamaEmbeddingConfig(model=embedding_model)
        embeddings = []

        for sentence in sentences:
            print(f'• Generating embedding for: "{sentence}"')
            embedding = await client.generate_embeddings(sentence, config)
            if not embedding:
                print(f'  Warning: Empty embedding returned for: "{sentence}"')
            embeddings.append(embedding)

        # Print embedding dimension information
        if embeddings and embeddings[0]:
            print(f"\nEmbedding dimensions: {len(embeddings[0])}")
        else:
            print("\nError: No valid embeddings generated!")
            return

        # Compare similarities between sentences
        print("\nSimilarity comparison between sentences:")
        print("----------------------------------------")

        # Calculate all pairwise similarities
        pairs = []
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                similarity = cosine_similarity(embeddings[i], embeddings[j])
                pairs.append((i, j, similarity))

        # Sort by similarity (highest first)
        pairs.sort(key=lambda x: x[2], reverse=True)

        # Print the results
        for i, j, similarity in pairs:
            print(
                f'Similarity: {similarity:.4f} | "{sentences[i]}" and "{sentences[j]}"'
            )


if __name__ == "__main__":
    asyncio.run(test_embeddings())
