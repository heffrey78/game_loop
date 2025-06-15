#!/usr/bin/env python3
"""
Demo script for World Connection Management System.

This script demonstrates the complete functionality of the world connection
generation system, including:
- Connection generation between different location themes
- Connection validation and quality assessment
- Connection storage and retrieval
- Graph-based connectivity analysis
- Embedding-based similarity search

Usage: python scripts/demo_world_connections.py
"""

import asyncio
import json
import sys
from pathlib import Path
from uuid import uuid4

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.tree import Tree

from game_loop.core.models.connection_models import (
    GeneratedConnection,
)
from game_loop.core.world.connection_context_collector import ConnectionContextCollector
from game_loop.core.world.connection_storage import ConnectionStorage
from game_loop.core.world.connection_theme_manager import ConnectionThemeManager
from game_loop.core.world.world_connection_manager import WorldConnectionManager
from game_loop.embeddings.connection_embedding_manager import ConnectionEmbeddingManager
from game_loop.state.models import Location, WorldState


class WorldConnectionDemo:
    """Demo class for world connection system."""

    def __init__(self):
        self.console = Console()
        self.world_state = WorldState()
        self.locations: dict[str, Location] = {}

        # Mock dependencies for demo
        self.session_factory = None  # Would be database session factory in real use
        self.llm_client = None  # Would be real LLM client
        self.template_env = None  # Would be Jinja2 environment
        self.embedding_manager = None  # Would be real embedding manager

        # Connection system components
        self.context_collector: ConnectionContextCollector | None = None
        self.theme_manager: ConnectionThemeManager | None = None
        self.connection_manager: WorldConnectionManager | None = None
        self.storage: ConnectionStorage | None = None
        self.embedding_mgr: ConnectionEmbeddingManager | None = None

        # Generated connections for demo
        self.connections: list[GeneratedConnection] = []

    async def setup_demo_environment(self):
        """Set up the demo environment with sample locations and mock services."""
        self.console.print(
            "[bold blue]Setting up World Connection Demo Environment[/bold blue]"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:

            # Create sample locations
            task = progress.add_task("Creating sample locations...", total=None)
            self._create_sample_locations()
            progress.advance(task)

            # Setup mock services
            task = progress.add_task("Initializing mock services...", total=None)
            self._setup_mock_services()
            progress.advance(task)

            # Initialize connection system
            task = progress.add_task("Initializing connection system...", total=None)
            self._initialize_connection_system()
            progress.advance(task)

        self.console.print("[green]✓ Demo environment ready![/green]\n")

    def _create_sample_locations(self):
        """Create a diverse set of sample locations for testing."""
        location_configs = [
            {
                "key": "elderwood_forest",
                "name": "Elderwood Forest",
                "description": "An ancient forest where towering oaks and mysterious shadows dance in eternal twilight. Moss-covered stones hint at forgotten secrets.",
                "theme": "Forest",
                "type": "wilderness",
                "features": [
                    "ancient_trees",
                    "moss_covered_stones",
                    "twilight_atmosphere",
                ],
            },
            {
                "key": "millbrook_village",
                "name": "Millbrook Village",
                "description": "A peaceful farming village with cobblestone streets, thatched cottages, and the gentle sound of a waterwheel turning.",
                "theme": "Village",
                "type": "settlement",
                "features": ["cobblestone_streets", "thatched_cottages", "waterwheel"],
            },
            {
                "key": "stormwind_peaks",
                "name": "Stormwind Peaks",
                "description": "Towering mountain peaks shrouded in mist, where eagles soar between jagged cliffs and ancient watchtowers stand sentinel.",
                "theme": "Mountain",
                "type": "wilderness",
                "features": ["jagged_cliffs", "watchtowers", "mist_shrouded_peaks"],
            },
            {
                "key": "goldenharbor_city",
                "name": "Goldenharbor City",
                "description": "A bustling port city with grand architecture, marble columns, and ships from distant lands crowding the harbor.",
                "theme": "City",
                "type": "urban",
                "features": [
                    "marble_architecture",
                    "busy_harbor",
                    "international_trade",
                ],
            },
            {
                "key": "whispering_caves",
                "name": "Whispering Caves",
                "description": "A network of limestone caves where underground rivers echo through crystalline chambers and ancient drawings tell forgotten stories.",
                "theme": "Cave",
                "type": "underground",
                "features": [
                    "limestone_formations",
                    "underground_rivers",
                    "ancient_drawings",
                ],
            },
            {
                "key": "mystic_ruins",
                "name": "Mystic Ruins",
                "description": "Crumbling stone temples overgrown with luminescent vines, where magical energies still pulse through carved archways.",
                "theme": "Ruins",
                "type": "magical",
                "features": ["stone_temples", "luminescent_vines", "magical_energies"],
            },
        ]

        for config in location_configs:
            location = Location(
                location_id=uuid4(),
                name=config["name"],
                description=config["description"],
                connections={},
                objects={},
                npcs={},
                state_flags={
                    "theme": config["theme"],
                    "type": config["type"],
                    "features": config["features"],
                },
            )
            self.locations[config["key"]] = location
            self.world_state.locations[location.location_id] = location

    def _setup_mock_services(self):
        """Set up mock services for the demo."""

        # Mock LLM responses for different connection types
        class MockLLMClient:
            def __init__(self):
                self.responses = {
                    "bridge": {
                        "description": "A graceful stone bridge arches across the divide, its weathered stones telling tales of countless travelers who have crossed before. Moss clings to the ancient mortar, and small wildflowers bloom between the gaps.",
                        "travel_time": 45,
                        "difficulty": 3,
                        "requirements": [],
                        "special_features": ["scenic_overlook", "ancient_construction"],
                        "atmosphere": "majestic and timeless",
                    },
                    "path": {
                        "description": "A winding forest path meanders between ancient trees, dappled with sunlight that filters through the canopy. Fallen leaves carpet the way, whispering stories with each step.",
                        "travel_time": 90,
                        "difficulty": 2,
                        "requirements": [],
                        "special_features": ["natural_beauty", "wildlife_sounds"],
                        "atmosphere": "peaceful and natural",
                    },
                    "tunnel": {
                        "description": "A carved tunnel cuts through solid rock, its walls smooth from centuries of passage. Crystalline formations catch torchlight, creating dancing shadows on the stone floor.",
                        "travel_time": 120,
                        "difficulty": 4,
                        "requirements": ["torch_or_light"],
                        "special_features": ["crystal_formations", "echo_chamber"],
                        "atmosphere": "mysterious and enclosed",
                    },
                    "road": {
                        "description": "A well-maintained cobblestone road stretches between destinations, marked by stone milestones and bordered by wildflower meadows that sway in the gentle breeze.",
                        "travel_time": 30,
                        "difficulty": 1,
                        "requirements": [],
                        "special_features": ["milestone_markers", "merchant_friendly"],
                        "atmosphere": "civilized and safe",
                    },
                }

            async def generate_response(self, **kwargs):
                prompt = kwargs.get("prompt", "").lower()
                for conn_type, response in self.responses.items():
                    if conn_type in prompt:
                        return json.dumps(response)
                return json.dumps(self.responses["path"])  # Default fallback

        # Mock template environment
        class MockTemplate:
            def render(self, **kwargs):
                return f"Generate a connection of type {kwargs.get('connection_type', 'passage')} between locations..."

        class MockTemplateEnv:
            def get_template(self, template_name):
                return MockTemplate()

        # Mock embedding manager
        class MockEmbeddingManager:
            async def generate_embedding(self, text: str) -> list[float]:
                # Generate a simple mock embedding based on text hash
                import hashlib

                hash_obj = hashlib.sha256(text.encode())
                # Create a 1536-dimension embedding (OpenAI standard)
                seed = int(hash_obj.hexdigest()[:8], 16)
                import random

                random.seed(seed)
                return [random.random() * 2 - 1 for _ in range(1536)]

        self.llm_client = MockLLMClient()
        self.template_env = MockTemplateEnv()
        self.embedding_manager = MockEmbeddingManager()

    def _initialize_connection_system(self):
        """Initialize the connection system components."""
        self.context_collector = ConnectionContextCollector(
            self.world_state, self.session_factory
        )
        self.theme_manager = ConnectionThemeManager(
            self.world_state, self.session_factory
        )
        self.connection_manager = WorldConnectionManager(
            self.world_state, self.session_factory, self.llm_client, self.template_env
        )
        self.storage = ConnectionStorage(self.session_factory)
        self.embedding_mgr = ConnectionEmbeddingManager(
            self.embedding_manager, self.session_factory
        )

    async def demo_connection_generation(self):
        """Demonstrate connection generation between different location types."""
        self.console.print("[bold green]🔗 Connection Generation Demo[/bold green]")

        # Define interesting connection pairs
        connection_pairs = [
            (
                "elderwood_forest",
                "millbrook_village",
                "A forest path leading to civilization",
            ),
            (
                "millbrook_village",
                "goldenharbor_city",
                "The main trade route between settlements",
            ),
            (
                "stormwind_peaks",
                "whispering_caves",
                "A mountain passage to hidden depths",
            ),
            (
                "mystic_ruins",
                "elderwood_forest",
                "An ancient path through mystical lands",
            ),
            (
                "goldenharbor_city",
                "stormwind_peaks",
                "The mountain road from the coast",
            ),
        ]

        for source_key, target_key, description in connection_pairs:
            source_location = self.locations[source_key]
            target_location = self.locations[target_key]

            self.console.print(
                f"\n[yellow]Generating connection: {source_location.name} → {target_location.name}[/yellow]"
            )
            self.console.print(f"[dim]{description}[/dim]")

            # Generate connection
            connection = await self.connection_manager.generate_connection(
                source_location_id=source_location.location_id,
                target_location_id=target_location.location_id,
                purpose="expand_world",
            )

            # Generate embedding
            embedding = await self.embedding_mgr.generate_connection_embedding(
                connection
            )
            connection.embedding_vector = embedding

            # Store connection (mock storage)
            self.connections.append(connection)

            # Display connection details
            self._display_connection_details(connection)

    def _display_connection_details(self, connection: GeneratedConnection):
        """Display detailed information about a generated connection."""
        source = self.world_state.locations[connection.source_location_id]
        target = self.world_state.locations[connection.target_location_id]

        # Create a table for connection details
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Type", connection.properties.connection_type)
        table.add_row("Difficulty", f"{connection.properties.difficulty}/10")
        table.add_row("Travel Time", f"{connection.properties.travel_time}s")
        table.add_row("Visibility", connection.properties.visibility)
        table.add_row("Reversible", "Yes" if connection.properties.reversible else "No")

        if connection.properties.requirements:
            table.add_row("Requirements", ", ".join(connection.properties.requirements))

        if connection.properties.special_features:
            table.add_row("Features", ", ".join(connection.properties.special_features))

        self.console.print(table)

        # Display description in a panel
        description_panel = Panel(
            connection.properties.description,
            title="[bold]Connection Description[/bold]",
            border_style="green",
        )
        self.console.print(description_panel)

    async def demo_validation_system(self):
        """Demonstrate the connection validation system."""
        self.console.print("\n[bold blue]🔍 Connection Validation Demo[/bold blue]")

        if not self.connections:
            self.console.print(
                "[red]No connections available for validation demo[/red]"
            )
            return

        # Validate a few connections
        for i, connection in enumerate(self.connections[:3]):
            source = self.world_state.locations[connection.source_location_id]
            target = self.world_state.locations[connection.target_location_id]

            self.console.print(
                f"\n[yellow]Validating: {source.name} → {target.name}[/yellow]"
            )

            # Get generation context
            context = await self.context_collector.collect_generation_context(
                connection.source_location_id,
                connection.target_location_id,
                "expand_world",
            )

            # Validate connection
            validation_result = await self.connection_manager.validate_connection(
                connection, context
            )

            # Display validation results
            self._display_validation_results(validation_result)

    def _display_validation_results(self, validation_result):
        """Display connection validation results."""
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Validation Metric", style="cyan")
        table.add_column("Score", style="white")
        table.add_column("Status", style="white")

        def get_status_style(score: float) -> str:
            if score >= 0.8:
                return "[green]Excellent[/green]"
            elif score >= 0.6:
                return "[yellow]Good[/yellow]"
            elif score >= 0.4:
                return "[orange1]Fair[/orange1]"
            else:
                return "[red]Poor[/red]"

        table.add_row(
            "Overall Valid",
            "Yes" if validation_result.is_valid else "No",
            "[green]✓[/green]" if validation_result.is_valid else "[red]✗[/red]",
        )
        table.add_row(
            "Consistency Score",
            f"{validation_result.consistency_score:.2f}",
            get_status_style(validation_result.consistency_score),
        )
        table.add_row(
            "Logical Soundness",
            f"{validation_result.logical_soundness:.2f}",
            get_status_style(validation_result.logical_soundness),
        )
        table.add_row(
            "Terrain Compatibility",
            f"{validation_result.terrain_compatibility:.2f}",
            get_status_style(validation_result.terrain_compatibility),
        )

        self.console.print(table)

        if validation_result.validation_errors:
            error_panel = Panel(
                "\n".join(
                    f"• {error}" for error in validation_result.validation_errors
                ),
                title="[bold red]Validation Errors[/bold red]",
                border_style="red",
            )
            self.console.print(error_panel)

        if validation_result.warnings:
            warning_panel = Panel(
                "\n".join(f"• {warning}" for warning in validation_result.warnings),
                title="[bold yellow]Warnings[/bold yellow]",
                border_style="yellow",
            )
            self.console.print(warning_panel)

    async def demo_connectivity_analysis(self):
        """Demonstrate world connectivity analysis."""
        self.console.print(
            "\n[bold magenta]🌐 World Connectivity Analysis[/bold magenta]"
        )

        # Build connectivity graph
        graph = self.connection_manager.connectivity_graph
        for connection in self.connections:
            graph.add_connection(connection)

        # Display connectivity information
        self._display_connectivity_tree()

        # Analyze connection opportunities
        await self._analyze_connection_opportunities()

    def _display_connectivity_tree(self):
        """Display the world connectivity as a tree structure."""
        tree = Tree("[bold]World Connectivity Graph[/bold]")

        # Create nodes for each location
        location_nodes = {}
        for key, location in self.locations.items():
            location_node = tree.add(
                f"[cyan]{location.name}[/cyan] ([dim]{location.state_flags.get('theme', 'Unknown')}[/dim])"
            )
            location_nodes[location.location_id] = location_node

        # Add connections
        for connection in self.connections:
            source = self.world_state.locations[connection.source_location_id]
            target = self.world_state.locations[connection.target_location_id]

            if connection.source_location_id in location_nodes:
                connection_info = f"[green]→[/green] {target.name} [dim]({connection.properties.connection_type})[/dim]"
                location_nodes[connection.source_location_id].add(connection_info)

        self.console.print(tree)

    async def _analyze_connection_opportunities(self):
        """Analyze and display connection opportunities."""
        self.console.print(
            "\n[bold cyan]🔍 Connection Opportunities Analysis[/bold cyan]"
        )

        # Analyze opportunities for each location
        for key, location in list(self.locations.items())[
            :3
        ]:  # Limit to first 3 for demo
            opportunities = await self.connection_manager.find_connection_opportunities(
                location.location_id
            )

            if opportunities:
                self.console.print(
                    f"\n[yellow]Opportunities from {location.name}:[/yellow]"
                )

                table = Table(show_header=True, header_style="bold green")
                table.add_column("Target Location", style="cyan")
                table.add_column("Opportunity Score", style="white")
                table.add_column("Themes", style="dim")

                for target_id, score in opportunities[:3]:  # Show top 3
                    target_location = self.world_state.locations.get(target_id)
                    if target_location:
                        table.add_row(
                            target_location.name,
                            f"{score:.2f}",
                            f"{location.state_flags.get('theme', 'Unknown')} → {target_location.state_flags.get('theme', 'Unknown')}",
                        )

                self.console.print(table)

    async def demo_similarity_search(self):
        """Demonstrate embedding-based similarity search."""
        self.console.print("\n[bold yellow]🔍 Similarity Search Demo[/bold yellow]")

        if len(self.connections) < 2:
            self.console.print(
                "[red]Need at least 2 connections for similarity search demo[/red]"
            )
            return

        # Use first connection as query
        query_connection = self.connections[0]
        source = self.world_state.locations[query_connection.source_location_id]
        target = self.world_state.locations[query_connection.target_location_id]

        self.console.print(
            f"[cyan]Finding connections similar to: {source.name} → {target.name}[/cyan]"
        )
        self.console.print(
            f"[dim]Type: {query_connection.properties.connection_type}, Difficulty: {query_connection.properties.difficulty}[/dim]\n"
        )

        # Mock similarity search (in real implementation, this would use vector similarity)
        similar_connections = []
        for connection in self.connections[1:]:
            # Simple similarity based on connection type and difficulty
            type_match = (
                1.0
                if connection.properties.connection_type
                == query_connection.properties.connection_type
                else 0.3
            )
            difficulty_similarity = (
                1.0
                - abs(
                    connection.properties.difficulty
                    - query_connection.properties.difficulty
                )
                / 10.0
            )
            similarity_score = (type_match + difficulty_similarity) / 2.0
            similar_connections.append((connection, similarity_score))

        # Sort by similarity
        similar_connections.sort(key=lambda x: x[1], reverse=True)

        # Display results
        table = Table(show_header=True, header_style="bold yellow")
        table.add_column("Similar Connection", style="cyan")
        table.add_column("Type", style="white")
        table.add_column("Difficulty", style="white")
        table.add_column("Similarity", style="green")

        for connection, similarity in similar_connections[:3]:
            source = self.world_state.locations[connection.source_location_id]
            target = self.world_state.locations[connection.target_location_id]
            table.add_row(
                f"{source.name} → {target.name}",
                connection.properties.connection_type,
                f"{connection.properties.difficulty}/10",
                f"{similarity:.2f}",
            )

        self.console.print(table)

    def display_summary(self):
        """Display a summary of the demo results."""
        self.console.print("\n[bold blue]📊 Demo Summary[/bold blue]")

        summary_table = Table(show_header=True, header_style="bold blue")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="white")

        summary_table.add_row("Total Locations", str(len(self.locations)))
        summary_table.add_row("Generated Connections", str(len(self.connections)))
        summary_table.add_row(
            "Location Themes",
            str(
                len(
                    set(
                        loc.state_flags.get("theme", "Unknown")
                        for loc in self.locations.values()
                    )
                )
            ),
        )

        connection_types = [
            conn.properties.connection_type for conn in self.connections
        ]
        summary_table.add_row("Connection Types", str(len(set(connection_types))))

        avg_difficulty = (
            sum(conn.properties.difficulty for conn in self.connections)
            / len(self.connections)
            if self.connections
            else 0
        )
        summary_table.add_row("Average Difficulty", f"{avg_difficulty:.1f}/10")

        self.console.print(summary_table)

        # Connection type distribution
        if self.connections:
            self.console.print("\n[bold]Connection Type Distribution:[/bold]")
            type_counts = {}
            for conn in self.connections:
                conn_type = conn.properties.connection_type
                type_counts[conn_type] = type_counts.get(conn_type, 0) + 1

            for conn_type, count in sorted(type_counts.items()):
                self.console.print(f"  • {conn_type}: {count}")

    async def run_demo(self):
        """Run the complete demo."""
        self.console.print(
            Panel.fit(
                "[bold blue]World Connection Management System Demo[/bold blue]\n"
                "[dim]Demonstrating intelligent connection generation, validation, and analysis[/dim]",
                border_style="blue",
            )
        )

        try:
            # Setup
            await self.setup_demo_environment()

            # Run demo sections
            await self.demo_connection_generation()
            await self.demo_validation_system()
            await self.demo_connectivity_analysis()
            await self.demo_similarity_search()

            # Summary
            self.display_summary()

            self.console.print(
                "\n[bold green]✨ Demo completed successfully![/bold green]"
            )
            self.console.print(
                "[dim]The World Connection Management system is ready for integration into the game loop.[/dim]"
            )

        except Exception as e:
            self.console.print(f"[bold red]Demo failed with error: {e}[/bold red]")
            raise


async def main():
    """Main entry point for the demo."""
    demo = WorldConnectionDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
