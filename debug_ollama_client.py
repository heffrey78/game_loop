#!/usr/bin/env python3
"""
Debug the Ollama client specifically.
"""

import asyncio

import ollama
from rich.console import Console

console = Console()


async def test_ollama_client():
    """Test the exact way the NLP processor uses ollama."""
    console.print("[bold blue]Testing Ollama Client Details[/bold blue]\n")

    # Test the exact method the NLP processor uses
    console.print("1. Testing ollama module attributes...")
    console.print(f"  Has 'generate' method: {hasattr(ollama, 'generate')}")
    console.print(
        f"  Is 'generate' callable: {callable(getattr(ollama, 'generate', None))}"
    )

    # Test the exact call pattern from _generate_completion_async
    console.print("\n2. Testing the exact call pattern...")

    try:
        # Prepare options like NLPProcessor does
        options = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "num_predict": 1024,
        }

        system_prompt = "You must respond with valid JSON only, with no explanations or additional text."
        prompt = "Test prompt: analyze the input 'look around'"
        model = "qwen2.5:3b"

        console.print(f"  Model: {model}")
        console.print(f"  Prompt: {prompt[:50]}...")
        console.print(f"  Options: {options}")
        console.print(f"  System: {system_prompt[:50]}...")

        # Test synchronous call first
        console.print("\n3. Testing synchronous call...")
        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                system=system_prompt,
                options=options,
            )
            console.print("  [green]✓ Sync call successful[/green]")
            console.print(f"  Response type: {type(response)}")
            console.print(
                f"  Response keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}"
            )
            if isinstance(response, dict) and "response" in response:
                console.print(f"  Response content: {response['response'][:100]}...")
        except Exception as e:
            console.print(f"  [red]✗ Sync call failed: {e}[/red]")
            console.print(f"  Error type: {type(e).__name__}")

        # Test async wrapper like NLPProcessor does
        console.print("\n4. Testing async wrapper...")
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: ollama.generate(
                    model=model,
                    prompt=prompt,
                    system=system_prompt,
                    options=options,
                ),
            )
            console.print("  [green]✓ Async wrapper successful[/green]")
            console.print(f"  Response type: {type(response)}")
            if isinstance(response, dict):
                console.print(f"  Response keys: {list(response.keys())}")
                if "response" in response:
                    console.print(
                        f"  Response content: {response['response'][:100]}..."
                    )
            else:
                console.print(f"  Response: {response}")
        except Exception as e:
            console.print(f"  [red]✗ Async wrapper failed: {e}[/red]")
            console.print(f"  Error type: {type(e).__name__}")

    except Exception as e:
        console.print(f"[red]✗ Setup error: {e}[/red]")


async def main():
    """Run tests."""
    await test_ollama_client()


if __name__ == "__main__":
    asyncio.run(main())
