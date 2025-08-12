#!/usr/bin/env python3
"""
Test script to debug Ollama connectivity issues.
Run with: poetry run python test_ollama.py
"""

import asyncio

import httpx
from rich.console import Console

console = Console()


async def test_ollama_connection():
    """Test basic connectivity to Ollama."""
    console.print("[bold blue]Testing Ollama Connection[/bold blue]\n")

    # Test 1: Check if Ollama is running
    console.print("1. Testing basic connectivity...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json()["models"]
                console.print("[green]✓ Ollama is running[/green]")
                console.print(f"  Found {len(models)} models:")
                for model in models:
                    console.print(f"  - {model['name']}")
            else:
                console.print(
                    f"[red]✗ Ollama returned status {response.status_code}[/red]"
                )
    except Exception as e:
        console.print(f"[red]✗ Cannot connect to Ollama: {e}[/red]")
        return

    console.print()

    # Test 2: Try each model with a simple completion
    console.print("2. Testing model completions...")

    test_prompt = "Complete this sentence: The quick brown fox"

    for model_name in ["qwen3:1.7b", "qwen3:4b", "deepseek-r1:1.5b"]:
        console.print(f"\n  Testing model: [cyan]{model_name}[/cyan]")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": model_name,
                        "prompt": test_prompt,
                        "stream": False,
                        "options": {"temperature": 0.7, "max_tokens": 50},
                    },
                )

                if response.status_code == 200:
                    result = response.json()
                    console.print("  [green]✓ Success![/green]")
                    console.print(
                        f"  Response: {result.get('response', 'No response')[:100]}..."
                    )
                else:
                    console.print(f"  [red]✗ Status {response.status_code}[/red]")
                    console.print(f"  Error: {response.text}")

        except httpx.TimeoutException:
            console.print("  [red]✗ Timeout - model may be downloading[/red]")
        except Exception as e:
            console.print(f"  [red]✗ Error: {e}[/red]")

    console.print()

    # Test 3: Test embedding model
    console.print("3. Testing embedding model...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "http://localhost:11434/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": "Hello world"},
            )

            if response.status_code == 200:
                result = response.json()
                embedding = result.get("embedding", [])
                console.print("[green]✓ Embedding model works[/green]")
                console.print(f"  Embedding dimensions: {len(embedding)}")
            else:
                console.print(f"[red]✗ Embedding failed: {response.status_code}[/red]")

    except Exception as e:
        console.print(f"[red]✗ Embedding error: {e}[/red]")

    console.print()

    # Test 4: Test with exact game configuration
    console.print("4. Testing with game configuration...")
    try:
        from game_loop.llm.config import LLMConfig
        from game_loop.llm.ollama.client import OllamaClient

        config = LLMConfig()
        console.print(f"  Default model: [cyan]{config.default_model}[/cyan]")
        console.print(f"  Base URL: [cyan]{config.base_url}[/cyan]")
        console.print(f"  Timeout: [cyan]{config.timeout}s[/cyan]")

        client = OllamaClient(config)

        # Test completion
        console.print("\n  Testing completion with game client...")
        try:
            result = await client.complete("Hello, how are you?")
            console.print("  [green]✓ Completion successful[/green]")
            console.print(f"  Response: {result[:100]}...")
        except Exception as e:
            console.print(f"  [red]✗ Completion failed: {e}[/red]")

        # Test embedding
        console.print("\n  Testing embedding with game client...")
        try:
            embedding = await client.get_embedding("test text")
            console.print("  [green]✓ Embedding successful[/green]")
            console.print(f"  Dimensions: {len(embedding)}")
        except Exception as e:
            console.print(f"  [red]✗ Embedding failed: {e}[/red]")

    except ImportError as e:
        console.print(f"[yellow]⚠ Cannot import game modules: {e}[/yellow]")
    except Exception as e:
        console.print(f"[red]✗ Game client error: {e}[/red]")


async def test_nlp_processor():
    """Test the NLP processor specifically."""
    console.print("\n[bold blue]Testing NLP Processor[/bold blue]\n")

    try:
        from game_loop.llm.config import LLMConfig
        from game_loop.llm.nlp_processor import NLPProcessor
        from game_loop.llm.ollama.client import OllamaClient

        config = LLMConfig()
        ollama_client = OllamaClient(config)
        nlp = NLPProcessor(ollama_client)

        test_inputs = [
            "I want to look around",
            "pick up the key",
            "talk to the guard",
            "what's in my inventory?",
        ]

        for test_input in test_inputs:
            console.print(f"\nTesting: '[yellow]{test_input}[/yellow]'")
            try:
                result = await nlp.process_input(test_input)
                console.print("[green]✓ Processed successfully[/green]")
                console.print(f"  Intent: {result.get('intent', 'unknown')}")
                console.print(f"  Action: {result.get('action', 'unknown')}")
                if "target" in result:
                    console.print(f"  Target: {result['target']}")
            except Exception as e:
                console.print(f"[red]✗ Processing failed: {e}[/red]")

    except Exception as e:
        console.print(f"[red]✗ Cannot test NLP processor: {e}[/red]")


async def main():
    """Run all tests."""
    await test_ollama_connection()
    await test_nlp_processor()

    console.print("\n[bold]Debugging Summary:[/bold]")
    console.print("1. If Ollama is not running: [yellow]ollama serve[/yellow]")
    console.print("2. If models are missing: [yellow]ollama pull qwen3:1.7b[/yellow]")
    console.print(
        "3. Check logs for more details: [yellow]journalctl -u ollama -f[/yellow]"
    )


if __name__ == "__main__":
    asyncio.run(main())
