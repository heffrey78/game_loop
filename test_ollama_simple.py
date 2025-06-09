#!/usr/bin/env python3
"""
Simple test to debug Ollama connectivity with the game's setup.
"""

import asyncio

import ollama
from rich.console import Console

console = Console()


async def test_direct_ollama():
    """Test direct ollama package usage."""
    console.print("[bold blue]Testing Direct Ollama Package[/bold blue]\n")

    # Test 1: Check if we can list models
    console.print("1. Listing models...")
    try:
        models = ollama.list()
        console.print(f"[green]✓ Found {len(models.get('models', []))} models[/green]")
        for model in models.get("models", []):
            console.print(f"  - {model['name']}")
    except Exception as e:
        console.print(f"[red]✗ Error listing models: {e}[/red]")

    console.print()

    # Test 2: Test generation (the way NLPProcessor does it)
    console.print("2. Testing generation like NLPProcessor...")
    prompt = "Complete this sentence: The quick brown fox"

    try:
        # This is how the NLPProcessor calls it
        response = ollama.generate(
            model="qwen2.5:3b",
            prompt=prompt,
            options={"temperature": 0.7, "top_p": 0.9, "top_k": 40, "num_predict": 100},
        )
        console.print("[green]✓ Generation successful[/green]")
        console.print(f"  Response: {response.get('response', 'No response')[:100]}...")
    except Exception as e:
        console.print(f"[red]✗ Generation error: {e}[/red]")
        console.print(f"  Error type: {type(e).__name__}")

    console.print()

    # Test 3: Test with the exact config from the game
    console.print("3. Testing with game configuration...")
    try:
        from game_loop.config.manager import ConfigManager

        config_manager = ConfigManager()

        console.print(f"  Config attributes: {dir(config_manager)}")

        # Check what the config manager actually has
        if hasattr(config_manager, "config"):
            console.print("  Has 'config' attribute")
            if hasattr(config_manager.config, "llm"):
                console.print(f"  Config LLM: {config_manager.config.llm}")
        elif hasattr(config_manager, "llm_config"):
            console.print("  Has 'llm_config' attribute")
            console.print(f"  LLM Config: {config_manager.llm_config}")
        else:
            console.print("[yellow]  No llm config found[/yellow]")

    except Exception as e:
        console.print(f"[red]✗ Config error: {e}[/red]")


async def test_nlp_processor():
    """Test the actual NLP processor."""
    console.print("\n[bold blue]Testing NLP Processor[/bold blue]\n")

    try:
        from game_loop.config.manager import ConfigManager
        from game_loop.llm.nlp_processor import NLPProcessor

        config_manager = ConfigManager()
        nlp = NLPProcessor(config_manager=config_manager)

        console.print("NLP Processor initialized:")
        console.print(f"  Host: {nlp.host}")
        console.print(f"  Model: {nlp.model}")
        console.print(f"  Client type: {type(nlp.client)}")

        # Test processing
        test_input = "look around"
        console.print(f"\nTesting input: '{test_input}'")

        try:
            result = await nlp.process_input(test_input)
            console.print("[green]✓ Processing successful[/green]")
            console.print(f"  Command type: {result.command_type}")
            console.print(f"  Action: {result.action}")
        except Exception as e:
            console.print(f"[red]✗ Processing failed: {e}[/red]")

            # Try extracting intent directly
            console.print("\nTrying extract_intent directly...")
            try:
                intent = await nlp.extract_intent(test_input)
                if intent:
                    console.print("[green]✓ Intent extracted[/green]")
                    console.print(f"  Intent: {intent}")
                else:
                    console.print("[yellow]No intent extracted[/yellow]")
            except Exception as e2:
                console.print(f"[red]✗ Intent extraction failed: {e2}[/red]")

    except Exception as e:
        console.print(f"[red]✗ NLP Processor error: {e}[/red]")


async def main():
    """Run all tests."""
    await test_direct_ollama()
    await test_nlp_processor()


if __name__ == "__main__":
    asyncio.run(main())
