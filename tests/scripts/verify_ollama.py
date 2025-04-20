"""
Test script to verify Ollama integration functionality.
"""

import asyncio
import sys
from pathlib import Path

from game_loop.llm.config import ConfigManager
from game_loop.llm.ollama.client import OllamaClient, OllamaEmbeddingConfig

# Add project root to Python path to allow imports
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))


async def verify_ollama_integration():
    """Run verification steps for Ollama integration."""
    print("\n==== Ollama Integration Verification ====\n")

    # Load test configuration
    config_file = Path(project_root) / "tests" / "configs" / "test_ollama_config.yaml"
    config_mgr = ConfigManager(config_file=str(config_file))

    # Step 1: Connect to Ollama API and check health
    print("Step 1: Connecting to Ollama API...")
    client = OllamaClient(
        base_url=config_mgr.llm_config.base_url, timeout=config_mgr.llm_config.timeout
    )

    try:
        # Check API health
        healthy = await client.health_check()
        print(f"✓ Ollama API health check: {'Healthy' if healthy else 'Not healthy'}")

        if not healthy:
            print(
                "✗ Ollama API is not healthy. Please check if Ollama server is running."
            )
            return False

        # Step 2: Verify model availability
        print("\nStep 2: Checking model availability...")
        models = await client.list_models()
        model_names = [model.get("name") for model in models]
        print(f"✓ Available models: {model_names}")

        if not models:
            print("✗ No models available. Please install at least one model in Ollama.")
            return False

        test_model = (
            model_names[0] if model_names else config_mgr.llm_config.default_model
        )
        model_available = await client.check_model_availability(test_model)
        print(
            f"✓ Model '{test_model}' availability: "
            f"{'Available' if model_available else 'Not available'}"
        )

        # Step 3: Generate a test embedding
        if model_available:
            print("\nStep 3: Generating test embedding...")
            config = OllamaEmbeddingConfig(model=test_model)
            test_text = "This is a test sentence for embedding generation."

            try:
                embedding = await client.generate_embeddings(test_text, config)
                print(
                    f"✓ Successfully generated embedding with "
                    f"{len(embedding)} dimensions"
                )
            except Exception as e:
                print(f"✗ Failed to generate embedding: {e}")
                return False

        # Step 4: Load sample prompt templates
        print("\nStep 4: Loading prompt templates...")

        try:
            default_template = config_mgr.get_prompt_template("default")
            print(f"✓ Successfully loaded default template: {default_template}")

            embedding_template = config_mgr.get_prompt_template("embedding")
            print(f"✓ Successfully loaded {embedding_template}")

            # Format a prompt
            formatted = config_mgr.format_prompt(
                "default", input="Hello, AI assistant!"
            )
            print(f"✓ Successfully formatted: {formatted}")

        except Exception as e:
            print(f"✗ Failed to load templates: {e}")
            return False

        print("\n✅ All verification steps completed successfully!")
        return True

    except Exception as e:
        print(f"✗ Error during verification: {e}")
        return False
    finally:
        await client.close()


if __name__ == "__main__":
    result = asyncio.run(verify_ollama_integration())
    sys.exit(0 if result else 1)
