#!/usr/bin/env python3
"""
Interactive Game Loop Technology Demo

This script provides an interactive demonstration of specific Game Loop features.
Choose which technology to demonstrate and interact with it directly.

Usage: poetry run python interactive_demo.py
"""

import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
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


async def interactive_nlp_demo():
    """Interactive NLP processing demonstration."""
    console.print(Panel.fit("🧠 Interactive NLP Processing Demo", style="bold blue"))
    console.print(
        "Type natural language commands and see how the system interprets them!"
    )
    console.print("Type 'exit' to return to main menu.\n")

    # Initialize components
    llm_config = LLMConfig()
    ollama_client = OllamaClient(llm_config)
    nlp_processor = NLPProcessor(ollama_client)
    input_processor = EnhancedInputProcessor(nlp_processor, None)

    context = {
        "current_location": "wizard_tower",
        "inventory": ["silver key", "torch"],
        "nearby_items": ["Crystal of Eternal Light", "Tome of Forbidden Knowledge"],
        "nearby_npcs": ["Archmagus Aldric"],
    }

    while True:
        user_input = Prompt.ask("[yellow]Enter command")

        if user_input.lower() == "exit":
            break

        console.print(f"\n[cyan]Processing:[/cyan] '{user_input}'")

        try:
            result = await input_processor.process(user_input, context)

            # Display results in a nice table
            table = Table(title="NLP Analysis Results")
            table.add_column("Component", style="cyan", width=20)
            table.add_column("Result", style="green")

            for key, value in result.items():
                if isinstance(value, list):
                    value = ", ".join(map(str, value))
                elif isinstance(value, dict):
                    value = str(value)
                table.add_row(key.replace("_", " ").title(), str(value))

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")

        console.print()


async def interactive_search_demo():
    """Interactive semantic search demonstration."""
    console.print(
        Panel.fit("🔍 Interactive Semantic Search Demo", style="bold magenta")
    )
    console.print("Search for items, locations, or NPCs using natural language!")
    console.print("Type 'exit' to return to main menu.\n")

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

        while True:
            query = Prompt.ask("[yellow]Enter search query")

            if query.lower() == "exit":
                break

            search_type = Prompt.ask(
                "[cyan]Search type",
                choices=["item", "location", "npc", "all"],
                default="all",
            )

            console.print(f"\n[cyan]Searching for:[/cyan] '{query}' in {search_type}")

            try:
                if search_type == "all":
                    results = await search_service.search_all_entities(query, limit=5)
                else:
                    results = await search_service.search_entities(
                        query, search_type, limit=5
                    )

                if results:
                    table = Table(title="Search Results")
                    table.add_column("Rank", width=6)
                    table.add_column("Name", style="bold")
                    table.add_column("Type", style="cyan")
                    table.add_column("Similarity", style="green")
                    table.add_column("Description", style="dim")

                    for i, result in enumerate(results, 1):
                        table.add_row(
                            str(i),
                            result["name"],
                            result["entity_type"],
                            f"{result['similarity']:.1%}",
                            (
                                result["description"][:50] + "..."
                                if len(result["description"]) > 50
                                else result["description"]
                            ),
                        )

                    console.print(table)
                else:
                    console.print("[yellow]No results found.[/yellow]")

            except Exception as e:
                console.print(f"[red]Error:[/red] {e}")

            console.print()


async def interactive_classification_demo():
    """Interactive action classification demonstration."""
    console.print(
        Panel.fit("🎯 Interactive Action Classification Demo", style="bold green")
    )
    console.print("Enter actions and see how the system classifies them!")
    console.print("Type 'exit' to return to main menu.\n")

    # Initialize classifier
    llm_config = LLMConfig()
    ollama_client = OllamaClient(llm_config)
    nlp_processor = NLPProcessor(ollama_client)
    classifier = ActionTypeClassifier(nlp_processor, ollama_client)

    while True:
        action = Prompt.ask("[yellow]Enter action")

        if action.lower() == "exit":
            break

        console.print(f"\n[cyan]Classifying:[/cyan] '{action}'")

        try:
            classification = await classifier.classify_action(action)

            # Create a detailed breakdown
            console.print(
                f"[bold green]Action Type:[/bold green] {classification.action_type.value.upper()}"
            )
            console.print(
                f"[bold green]Confidence:[/bold green] {classification.confidence:.1%}"
            )
            console.print(
                f"[bold green]Method:[/bold green] {classification.classification_method}"
            )

            if classification.primary_verb:
                console.print(
                    f"[green]Primary Verb:[/green] {classification.primary_verb}"
                )
            if classification.target_entity:
                console.print(f"[green]Target:[/green] {classification.target_entity}")
            if classification.modifiers:
                console.print(
                    f"[green]Modifiers:[/green] {', '.join(classification.modifiers)}"
                )

            # Show confidence level interpretation
            if classification.confidence >= 0.8:
                confidence_desc = (
                    "[bold green]High - Very confident in classification[/bold green]"
                )
            elif classification.confidence >= 0.6:
                confidence_desc = "[yellow]Medium - Reasonably confident[/yellow]"
            else:
                confidence_desc = "[red]Low - May need clarification[/red]"

            console.print(f"[green]Confidence Level:[/green] {confidence_desc}")

        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")

        console.print()


def show_tech_showcase():
    """Show what technologies are available to demonstrate."""
    console.print(
        Panel.fit(
            "🎮 Game Loop Technology Showcase\n\n"
            "Available demonstrations:\n\n"
            "🧠 [bold cyan]Natural Language Processing[/bold cyan]\n"
            "   • Parse complex natural language commands\n"
            "   • Extract intent, actions, and targets\n"
            "   • Handle ambiguous inputs with context\n\n"
            "🔍 [bold magenta]Semantic Search[/bold magenta]\n"
            "   • Search game entities by meaning, not just keywords\n"
            "   • Find items like 'something that glows' or 'illumination source'\n"
            "   • Vector embeddings for intelligent matching\n\n"
            "🎯 [bold green]Action Classification[/bold green]\n"
            "   • Automatically categorize player actions\n"
            "   • Rule-based patterns + LLM fallbacks\n"
            "   • Extract verbs, targets, and modifiers\n\n"
            "💾 [bold yellow]Rich Database Integration[/bold yellow]\n"
            "   • PostgreSQL with pgvector for embeddings\n"
            "   • Async SQLAlchemy ORM\n"
            "   • Efficient vector similarity search\n\n"
            "🎨 [bold white]Rich Output Generation[/bold white]\n"
            "   • Beautiful console formatting\n"
            "   • Template-based responses\n"
            "   • Context-aware descriptions",
            style="bold white on blue",
            title="Technology Overview",
        )
    )


async def main():
    """Main interactive demo menu."""
    show_tech_showcase()

    while True:
        console.print("\n" + "=" * 60)
        choice = Prompt.ask(
            "\n[bold cyan]Choose a demo[/bold cyan]",
            choices=["nlp", "search", "classification", "overview", "quit"],
            default="quit",
        )

        if choice == "quit":
            console.print("\n[green]Thanks for exploring Game Loop technology![/green]")
            break
        elif choice == "overview":
            show_tech_showcase()
        elif choice == "nlp":
            await interactive_nlp_demo()
        elif choice == "search":
            await interactive_search_demo()
        elif choice == "classification":
            await interactive_classification_demo()

        console.print("\n[dim]Returning to main menu...[/dim]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Demo interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Demo error: {e}[/red]")
