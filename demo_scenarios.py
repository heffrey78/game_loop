#!/usr/bin/env python3
"""
Game Loop Technology Demonstration Scenarios

This script demonstrates the sophisticated technology built into the Game Loop system.
Run after setting up the demo world with: poetry run python demo_world_setup.py

Usage: poetry run python demo_scenarios.py
"""

import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from game_loop.config.models import DatabaseConfig
from game_loop.core.actions.action_classifier import ActionTypeClassifier
from game_loop.core.enhanced_input_processor import EnhancedInputProcessor
from game_loop.database.session_factory import DatabaseSessionFactory
from game_loop.embeddings.service import EmbeddingService
from game_loop.llm.config import LLMConfig
from game_loop.llm.nlp_processor import NLPProcessor
from game_loop.llm.ollama.client import OllamaClient
from game_loop.search.semantic_search import SemanticSearchService

console = Console()


async def demo_nlp_processing():
    """Demonstrate natural language processing capabilities."""
    console.print(Panel.fit("🧠 Natural Language Processing Demo", style="bold blue"))

    # Initialize components
    llm_config = LLMConfig()
    ollama_client = OllamaClient(llm_config)
    nlp_processor = NLPProcessor(ollama_client)
    input_processor = EnhancedInputProcessor(nlp_processor, None)

    test_inputs = [
        "I want to go north and then pick up the crystal",
        "Can you show me what's in my backpack?",
        "I'd like to examine the mysterious glowing orb very carefully",
        "Put the philosopher's stone into the ancient chest",
        "Ask the wizard about the forbidden knowledge",
    ]

    for test_input in test_inputs:
        console.print(f"\n[yellow]Input:[/yellow] '{test_input}'")

        # Process with NLP
        result = await input_processor.process(
            test_input, {"current_location": "wizard_tower"}
        )

        # Display results
        table = Table(title="Processing Results")
        table.add_column("Component", style="cyan")
        table.add_column("Result", style="green")

        table.add_row("Command Type", result.get("command_type", "unknown"))
        table.add_row("Primary Action", result.get("action", "none"))

        if "target" in result:
            table.add_row("Target", result["target"])
        if "modifiers" in result:
            table.add_row("Modifiers", ", ".join(result.get("modifiers", [])))
        if "intent" in result:
            table.add_row("Intent", result["intent"])

        console.print(table)


async def demo_action_classification():
    """Demonstrate the action classification system."""
    console.print(Panel.fit("🎯 Action Classification Demo", style="bold green"))

    # Initialize classifier
    llm_config = LLMConfig()
    ollama_client = OllamaClient(llm_config)
    nlp_processor = NLPProcessor(ollama_client)
    classifier = ActionTypeClassifier(nlp_processor, ollama_client)

    test_actions = [
        "walk through the enchanted garden",
        "pick up the glowing crystal and examine it closely",
        "ask Pip about alchemy experiments",
        "unlock the chest with the silver key",
        "save my game progress",
        "what items are in this room?",
        "cast a spell of protection",
    ]

    for action in test_actions:
        console.print(f"\n[yellow]Action:[/yellow] '{action}'")

        # Classify the action
        classification = await classifier.classify_action(action)

        # Display classification
        console.print(f"[green]Type:[/green] {classification.action_type.value}")
        console.print(f"[green]Confidence:[/green] {classification.confidence:.2%}")
        console.print(f"[green]Method:[/green] {classification.classification_method}")

        if classification.primary_verb:
            console.print(f"[green]Verb:[/green] {classification.primary_verb}")
        if classification.target_entity:
            console.print(f"[green]Target:[/green] {classification.target_entity}")


async def demo_semantic_search():
    """Demonstrate semantic search capabilities."""
    console.print(Panel.fit("🔍 Semantic Search Demo", style="bold magenta"))

    # Initialize database session factory
    db_config = DatabaseConfig()
    session_factory = DatabaseSessionFactory(db_config)
    await session_factory.initialize()

    async with session_factory.get_session() as session:
        # Initialize search service
        llm_config = LLMConfig()
        ollama_client = OllamaClient(llm_config)
        embedding_service = EmbeddingService(ollama_client)
        search_service = SemanticSearchService(session, embedding_service)

        queries = [
            ("something that glows", "item"),
            ("a place to do experiments", "location"),
            ("someone who can teach me magic", "npc"),
            ("illumination source", "item"),
            ("ancient wisdom", "any"),
        ]

        for query, search_type in queries:
            console.print(
                f"\n[yellow]Query:[/yellow] '{query}' (searching for: {search_type})"
            )

            # Perform semantic search
            if search_type == "any":
                results = await search_service.search_all_entities(query, limit=3)
            else:
                results = await search_service.search_entities(
                    query, search_type, limit=3
                )

            # Display results
            for i, result in enumerate(results, 1):
                console.print(
                    f"[green]{i}. {result['name']}[/green] (similarity: {result['similarity']:.2%})"
                )
                console.print(f"   Type: {result['entity_type']}")
                console.print(f"   Preview: {result['description'][:100]}...")


async def demo_context_awareness():
    """Demonstrate context-aware processing."""
    console.print(Panel.fit("🎭 Context-Aware Processing Demo", style="bold cyan"))

    # This would typically use the full game state
    contexts = [
        {
            "location": "wizard_tower",
            "inventory": ["silver key", "torch"],
            "nearby_items": ["Crystal of Eternal Light", "Tome of Forbidden Knowledge"],
            "nearby_npcs": ["Archmagus Aldric"],
            "description": "Player is in the wizard's tower with a key",
        },
        {
            "location": "alchemy_lab",
            "inventory": ["Philosopher's Stone"],
            "nearby_items": ["Vial of Liquid Moonlight"],
            "nearby_npcs": ["Pip the Apprentice"],
            "description": "Player is in the lab holding the philosopher's stone",
        },
    ]

    llm_config = LLMConfig()
    ollama_client = OllamaClient(llm_config)
    nlp_processor = NLPProcessor(ollama_client)
    input_processor = EnhancedInputProcessor(nlp_processor, None)

    # Test ambiguous commands in different contexts
    ambiguous_commands = ["use it", "talk to them", "take the glowing thing"]

    for context in contexts:
        console.print(f"\n[blue]Context:[/blue] {context['description']}")

        for command in ambiguous_commands:
            console.print(f"\n[yellow]Command:[/yellow] '{command}'")

            # Process with context
            result = await input_processor.process(command, context)

            # Show how context affects interpretation
            if "disambiguation" in result:
                console.print("[red]Needs clarification:[/red]")
                for option in result["disambiguation"]:
                    console.print(f"  - {option}")
            else:
                console.print(
                    f"[green]Interpreted as:[/green] {result.get('action', 'unknown')} {result.get('target', '')}"
                )


async def demo_rich_output():
    """Demonstrate rich output generation."""
    console.print(Panel.fit("✨ Rich Output Generation Demo", style="bold white"))

    # Show different types of rich output
    from rich.markdown import Markdown

    # Location description with rich formatting
    location_text = """
## Archmage's Tower

You stand in the **grand library** of the Archmage's Tower. Ancient tomes line the walls,
their spines _glowing with ethereal light_. A massive crystalline orb hovers in the center 
of the room, pulsing with arcane energy.

### Available Exits:
- **North**: A spiral staircase ascends into darkness
- **South**: Back to the tower entrance

### Items Here:
1. Crystal of Eternal Light ✨
2. Tome of Forbidden Knowledge 📖
3. Mirror of True Sight 🔮
    """

    console.print(Markdown(location_text))

    # Action result with formatting
    action_result = Panel(
        "[green]Success![/green] You carefully pick up the [bold cyan]Crystal of Eternal Light[/bold cyan].\n\n"
        "The crystal feels warm in your hands, and its gentle glow illuminates your surroundings "
        "with a comforting radiance. You notice ancient runes becoming visible on nearby surfaces.",
        title="Action Result",
        border_style="green",
    )
    console.print(action_result)

    # Error message with helpful suggestions
    error_panel = Panel(
        "[red]I don't understand that command.[/red]\n\n"
        "Did you mean one of these?\n"
        "• [yellow]take crystal[/yellow] - Pick up the Crystal of Eternal Light\n"
        "• [yellow]go north[/yellow] - Climb the spiral staircase\n"
        "• [yellow]examine tome[/yellow] - Look at the Tome of Forbidden Knowledge\n"
        "• [yellow]talk to aldric[/yellow] - Speak with the Archmagus",
        title="Command Not Recognized",
        border_style="red",
    )
    console.print(error_panel)


async def main():
    """Run all demonstrations."""
    console.print(
        Panel.fit(
            "🎮 Game Loop Technology Demonstration\n"
            "Showcasing NLP, Semantic Search, and Rich Interactions",
            style="bold white on blue",
        )
    )

    demos = [
        ("NLP Processing", demo_nlp_processing),
        ("Action Classification", demo_action_classification),
        ("Semantic Search", demo_semantic_search),
        ("Context Awareness", demo_context_awareness),
        ("Rich Output", demo_rich_output),
    ]

    for name, demo_func in demos:
        console.print(f"\n{'='*60}\n")
        try:
            await demo_func()
        except Exception as e:
            console.print(f"[red]Error in {name}: {e}[/red]")

        # Pause between demos
        console.print("\n[dim]Press Enter to continue to next demo...[/dim]")
        input()

    console.print(
        Panel.fit(
            "✅ Demonstration Complete!\n\n"
            "These examples show just a fraction of what the Game Loop engine can do.\n"
            "The combination of NLP, semantic search, and rich output creates\n"
            "an immersive and intelligent text adventure experience.",
            style="bold green",
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
