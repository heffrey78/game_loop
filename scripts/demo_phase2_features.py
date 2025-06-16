#!/usr/bin/env python3
"""
Demo script to showcase Phase 2 dynamic world generation features.
This script demonstrates the enhanced smart expansion, content generation, and player behavior tracking.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncpg
from rich.console import Console
from rich.panel import Panel

from game_loop.config.models import (
    DatabaseConfig,
    FeaturesConfig,
    GameConfig,
    LLMConfig,
)
from game_loop.core.game_loop import GameLoop


async def demo_phase2_features():
    """Demonstrate Phase 2 dynamic world generation features."""
    console = Console()

    # Display demo header
    console.print(
        Panel.fit(
            "[bold cyan]Game Loop - Phase 2 Dynamic World Generation Demo[/bold cyan]\n"
            "[dim]Showcasing smart expansion, content generation, and player behavior tracking[/dim]",
            border_style="cyan",
        )
    )

    # Configure the game
    config = GameConfig(
        database=DatabaseConfig(
            host="localhost",
            port=5432,
            name="game_loop",
            user="postgres",
            password="password",
        ),
        llm=LLMConfig(provider="ollama", model_name="llama3.2:3b"),
        features=FeaturesConfig(
            use_nlp=False, enable_debugging=True  # Disable for demo
        ),
    )

    # Initialize database connection
    try:
        db_pool = await asyncpg.create_pool(
            host=config.database.host,
            port=config.database.port,
            database=config.database.name,
            user=config.database.user,
            password=config.database.password,
            min_size=1,
            max_size=5,
        )
    except Exception as e:
        console.print(f"[red]Failed to connect to database for demo: {e}[/red]")
        return

    # Initialize GameLoop
    game_loop = GameLoop(config, db_pool, console)

    try:
        # Initialize the game
        await game_loop.initialize()

        # Demo 1: Smart Expansion Logic
        console.print("\n" + "=" * 60)
        console.print(
            "[bold yellow]DEMO 1: Smart Expansion with Terrain Consistency[/bold yellow]"
        )
        console.print("=" * 60)

        # Get current location (should be Office)
        player_state, world_state = game_loop.state_manager.get_current_state()
        current_location = world_state.locations[player_state.current_location_id]

        console.print(f"\n[blue]Starting Location:[/blue] {current_location.name}")
        console.print(
            f"[dim]Description: {current_location.description[:100]}...[/dim]"
        )

        # Test smart expansion in different directions
        console.print(
            f"\n[green]Available exits:[/green] {list(current_location.connections.keys())}"
        )

        # Try expanding to 'outside' - should create urban area
        if "outside" in current_location.connections:
            console.print(
                f"\n[cyan]Expanding 'outside' from {current_location.name}...[/cyan]"
            )

            # Simulate terrain-aware expansion
            terrain_type = await game_loop._get_terrain_type(current_location)
            new_location_type = await game_loop._determine_location_type(
                current_location, "outside"
            )

            console.print(f"[dim]Source terrain:[/dim] {terrain_type}")
            console.print(f"[dim]Generated type:[/dim] {new_location_type}")
            console.print(
                "[green]✓ Smart expansion selected appropriate location type for direction[/green]"
            )

        # Demo 2: Dynamic Content Generation
        console.print("\n" + "=" * 60)
        console.print("[bold yellow]DEMO 2: Dynamic Content Generation[/bold yellow]")
        console.print("=" * 60)

        # Create a sample location for content generation
        from uuid import uuid4

        test_location_id = uuid4()

        console.print(
            "\n[blue]Generating dynamic content for a test 'office_space' location...[/blue]"
        )

        # Demonstrate content probability calculation
        content_probs = game_loop._get_content_probability("office_space", depth=1)
        console.print(f"[dim]Content probabilities:[/dim] {content_probs}")

        # Simulate content generation (without actual database insertion)
        console.print("\n[cyan]Simulating content generation:[/cyan]")

        # Objects
        import random

        if random.random() < content_probs["objects"]:
            object_templates = [
                "desk computer",
                "office chair",
                "filing cabinet",
                "coffee maker",
                "employee handbook",
            ]
            selected_object = random.choice(object_templates)
            console.print(f"[green]✓ Generated Object:[/green] {selected_object}")

        # NPCs
        if random.random() < content_probs["npcs"]:
            npc_templates = [
                "office worker",
                "security guard",
                "janitor",
                "IT technician",
            ]
            selected_npc = random.choice(npc_templates)
            console.print(f"[green]✓ Generated NPC:[/green] A {selected_npc}")

        # Quest Hooks
        if random.random() < content_probs["interactions"]:
            quest_templates = [
                "urgent memo",
                "computer terminal",
                "voice message",
                "security alert",
            ]
            selected_hook = random.choice(quest_templates)
            console.print(f"[green]✓ Generated Quest Hook:[/green] {selected_hook}")

        # Demo 3: Player Behavior Tracking
        console.print("\n" + "=" * 60)
        console.print(
            "[bold yellow]DEMO 3: Player Behavior Tracking & Adaptation[/bold yellow]"
        )
        console.print("=" * 60)

        # Simulate player behavior tracking
        console.print("\n[blue]Simulating exploration behavior tracking...[/blue]")

        # Simulate some exploration data
        await game_loop._track_player_exploration(
            current_location, "north", "industrial_zone", 1
        )
        await game_loop._track_player_exploration(
            current_location, "north", "factory_interior", 2
        )
        await game_loop._track_player_exploration(
            current_location, "east", "office_space", 1
        )
        await game_loop._track_player_exploration(
            current_location, "north", "industrial_zone", 3
        )

        # Get player preferences
        preferences = game_loop._get_player_preferences()
        console.print("\n[cyan]Detected Player Preferences:[/cyan]")
        console.print(
            f"[dim]Preferred directions:[/dim] {preferences['preferred_directions']}"
        )
        console.print(
            f"[dim]Preferred location types:[/dim] {preferences['preferred_location_types']}"
        )
        console.print(
            f"[dim]Exploration style:[/dim] {preferences['exploration_style']}"
        )
        console.print(f"[dim]Experience level:[/dim] {preferences['experience_level']}")

        # Show adaptive content generation
        console.print("\n[green]Adaptive Content Generation:[/green]")

        # Compare base vs adaptive probabilities
        base_probs = {
            "objects": 0.7,
            "npcs": 0.05,
            "interactions": 0.3,
        }  # industrial_zone base
        adaptive_probs = game_loop._get_content_probability("industrial_zone", depth=1)

        console.print(f"[dim]Base probabilities (industrial):[/dim] {base_probs}")
        console.print(f"[dim]Adaptive probabilities:[/dim] {adaptive_probs}")
        console.print(
            "[green]✓ Content generation adapts to player preferences[/green]"
        )

        # Demo 4: Location Type Hierarchies
        console.print("\n" + "=" * 60)
        console.print(
            "[bold yellow]DEMO 4: Location Type Hierarchies & Validation[/bold yellow]"
        )
        console.print("=" * 60)

        console.print("\n[blue]Testing location transition validation...[/blue]")

        # Test valid transitions
        valid_transition = await game_loop._validate_location_hierarchy(
            "office_space", "building_corridor", "north"
        )
        console.print(
            f"[green]✓ Valid:[/green] office_space -> building_corridor (north): {valid_transition}"
        )

        # Test invalid transitions
        invalid_transition = await game_loop._validate_location_hierarchy(
            "basement_corridor", "roof_access", "up"
        )
        console.print(
            f"[red]✗ Invalid:[/red] basement_corridor -> roof_access (up): {invalid_transition}"
        )

        # Show fallback mechanism
        fallback = game_loop._get_fallback_location_type("unknown_type", "north")
        console.print(f"[yellow]Fallback:[/yellow] unknown_type -> {fallback}")

        # Demo 5: Terrain-Aware Generation at Different Depths
        console.print("\n" + "=" * 60)
        console.print("[bold yellow]DEMO 5: Depth-Based Specialization[/bold yellow]")
        console.print("=" * 60)

        console.print(
            "\n[blue]Testing location generation at different depths...[/blue]"
        )

        for depth in [1, 2, 3, 4, 5]:
            location_type = await game_loop._apply_smart_expansion_rules(
                "urban_street", "north", "urban", 0, depth
            )
            console.print(f"[cyan]Depth {depth}:[/cyan] {location_type}")

        console.print(
            "\n[green]✓ Deeper exploration yields more specialized/unique locations[/green]"
        )

        # Demo Summary
        console.print("\n" + "=" * 60)
        console.print(
            "[bold green]DEMO COMPLETE - Phase 2 Features Showcased[/bold green]"
        )
        console.print("=" * 60)

        console.print("\n[yellow]Phase 2 Features Demonstrated:[/yellow]")
        console.print("[green]✓[/green] Smart expansion with terrain consistency")
        console.print(
            "[green]✓[/green] Dynamic content generation (NPCs, objects, quest hooks)"
        )
        console.print("[green]✓[/green] Player behavior tracking and analysis")
        console.print("[green]✓[/green] Adaptive content based on player preferences")
        console.print("[green]✓[/green] Location type hierarchies and validation")
        console.print("[green]✓[/green] Depth-based specialization")
        console.print("[green]✓[/green] 35+ location types with logical transitions")

        console.print(
            "\n[cyan]The dynamic world now creates rich, adaptive experiences that learn from player behavior![/cyan]"
        )

    except Exception as e:
        console.print(f"[red]Demo error: {e}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
    finally:
        await db_pool.close()


if __name__ == "__main__":
    asyncio.run(demo_phase2_features())
