#!/usr/bin/env python3
"""
Simple demo of Phase 2 dynamic world generation features.
This demonstrates the logic without requiring full database setup.
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import random

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


# Mock classes to demonstrate the logic
class MockLocation:
    def __init__(self, name, location_type="unknown", depth=0):
        self.name = name
        self.state_flags = {
            "location_type": location_type,
            "expansion_depth": depth,
            "elevation": 0,
        }


class MockGameLoop:
    """Mock game loop to demonstrate Phase 2 features."""

    def __init__(self):
        self.behavior_stats = {
            "preferred_directions": {"north": 5, "east": 3, "south": 1},
            "preferred_location_types": {
                "industrial_zone": 4,
                "office_space": 3,
                "urban_street": 2,
            },
            "exploration_depth": 3,
            "total_expansions": 12,
        }

    async def _get_terrain_type(self, location):
        """Determine the terrain type of a location for consistency checking."""
        location_type = location.state_flags.get("location_type", "unknown")

        terrain_mapping = {
            "urban_street": "urban",
            "industrial_district": "industrial",
            "industrial_zone": "industrial",
            "factory_interior": "industrial",
            "basement_access": "underground",
            "basement_corridor": "underground",
            "sublevel": "underground",
            "utility_tunnels": "underground",
            "building_interior": "building",
            "office_space": "building",
            "upper_floor": "building",
            "mechanical_room": "building",
            "loading_dock": "industrial",
            "roof_access": "building",
        }

        return terrain_mapping.get(location_type, "unknown")

    async def _get_elevation_level(self, source_location, direction):
        """Calculate elevation level based on direction and source location."""
        current_elevation = source_location.state_flags.get("elevation", 0)

        elevation_changes = {"up": 1, "down": -1, "stairs": 1, "elevator": 1}

        return current_elevation + elevation_changes.get(direction, 0)

    def _classify_direction(self, direction, elevation):
        """Classify direction into movement type for terrain transitions."""
        if direction in ["up", "stairs up", "climb"]:
            return "up"
        elif direction in ["down", "stairs down", "descend"]:
            return "down"
        elif direction in ["inside", "enter", "through"]:
            return "inside"
        else:
            return "horizontal"

    async def _apply_smart_expansion_rules(
        self, source_type, direction, terrain_type, elevation, depth
    ):
        """Apply smart expansion rules considering all context factors."""
        # Terrain-consistent transitions
        terrain_transitions = {
            "urban": {
                "horizontal": [
                    "urban_street",
                    "commercial_district",
                    "residential_area",
                ],
                "inside": ["building_interior", "office_building", "retail_space"],
                "down": ["basement_access", "subway_station", "underground_passage"],
            },
            "industrial": {
                "horizontal": [
                    "industrial_zone",
                    "factory_complex",
                    "warehouse_district",
                ],
                "inside": [
                    "factory_interior",
                    "manufacturing_floor",
                    "storage_facility",
                ],
                "down": ["industrial_sublevel", "utility_tunnels", "maintenance_areas"],
            },
            "underground": {
                "horizontal": [
                    "basement_corridor",
                    "tunnel_system",
                    "underground_passage",
                ],
                "down": ["sublevel", "deep_tunnels", "underground_complex"],
                "up": ["ground_access", "stairwell_exit", "elevator_shaft"],
            },
            "building": {
                "horizontal": ["office_space", "building_corridor", "conference_area"],
                "up": ["upper_floor", "executive_level", "roof_access"],
                "down": ["lower_floor", "basement_access", "service_level"],
            },
        }

        # Determine movement type
        movement_type = self._classify_direction(direction, elevation)

        # Get possible transitions for this terrain
        transitions = terrain_transitions.get(terrain_type, {})
        candidates = transitions.get(movement_type, [])

        if not candidates:
            return "unknown_area"

        # Apply depth-based variation
        if depth >= 3:
            specialized_types = {
                "urban": ["abandoned_district", "ruined_quarter", "ghost_town"],
                "industrial": [
                    "derelict_factory",
                    "toxic_wasteland",
                    "abandoned_complex",
                ],
                "underground": [
                    "forgotten_tunnels",
                    "ancient_catacombs",
                    "deep_chambers",
                ],
                "building": ["hidden_floors", "secret_chambers", "abandoned_wings"],
            }
            depth_candidates = specialized_types.get(terrain_type, candidates)
            candidates.extend(depth_candidates)

        return random.choice(candidates)

    def _get_player_preferences(self):
        """Get player preferences based on tracked behavior."""
        stats = self.behavior_stats

        # Analyze direction preferences
        direction_counts = stats.get("preferred_directions", {})
        preferred_directions = sorted(
            direction_counts.keys(), key=lambda k: direction_counts[k], reverse=True
        )[:3]

        # Analyze location type preferences
        type_counts = stats.get("preferred_location_types", {})
        preferred_types = sorted(
            type_counts.keys(), key=lambda k: type_counts[k], reverse=True
        )[:3]

        # Determine exploration style
        total_expansions = stats.get("total_expansions", 0)
        max_depth = stats.get("exploration_depth", 0)

        if max_depth >= 4:
            exploration_style = "deep_explorer"
        elif total_expansions >= 10:
            exploration_style = "broad_explorer"
        elif len(preferred_directions) <= 2:
            exploration_style = "focused_explorer"
        else:
            exploration_style = "casual_explorer"

        # Determine experience level
        if total_expansions >= 20:
            experience_level = "expert"
        elif total_expansions >= 10:
            experience_level = "experienced"
        elif total_expansions >= 5:
            experience_level = "intermediate"
        else:
            experience_level = "beginner"

        return {
            "preferred_directions": preferred_directions,
            "preferred_location_types": preferred_types,
            "exploration_style": exploration_style,
            "experience_level": experience_level,
            "total_expansions": total_expansions,
            "max_depth": max_depth,
        }

    def _get_content_probability(self, location_type, depth):
        """Get adaptive content generation probabilities."""
        base_probabilities = {
            "urban_street": {"objects": 0.4, "npcs": 0.1, "interactions": 0.3},
            "industrial_zone": {"objects": 0.7, "npcs": 0.05, "interactions": 0.3},
            "office_space": {"objects": 0.5, "npcs": 0.12, "interactions": 0.35},
            "basement_corridor": {"objects": 0.3, "npcs": 0.08, "interactions": 0.4},
        }

        probabilities = base_probabilities.get(
            location_type, {"objects": 0.3, "npcs": 0.1, "interactions": 0.2}
        )

        # Apply player preferences for adaptive generation
        preferences = self._get_player_preferences()

        # Boost content generation for preferred location types
        if location_type in preferences["preferred_location_types"]:
            type_boost = 1.2
            probabilities = {
                key: min(value * type_boost, 0.9)
                for key, value in probabilities.items()
            }

        # Adjust based on exploration style
        exploration_style = preferences["exploration_style"]
        if exploration_style == "deep_explorer":
            probabilities["interactions"] = min(
                probabilities["interactions"] * 1.3, 0.8
            )
            probabilities["npcs"] = min(probabilities["npcs"] * 1.2, 0.6)
        elif exploration_style == "broad_explorer":
            probabilities["objects"] = min(probabilities["objects"] * 1.3, 0.9)

        # Increase complexity with depth
        depth_multiplier = 1.0 + (depth * 0.1)
        return {
            key: min(value * depth_multiplier, 0.9)
            for key, value in probabilities.items()
        }


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

    game_loop = MockGameLoop()

    # Demo 1: Smart Expansion Logic
    console.print("\n" + "=" * 70)
    console.print(
        "[bold yellow]DEMO 1: Smart Expansion with Terrain Consistency[/bold yellow]"
    )
    console.print("=" * 70)

    test_locations = [
        MockLocation("Office Building", "office_space", 0),
        MockLocation("Factory Floor", "factory_interior", 1),
        MockLocation("Underground Tunnel", "basement_corridor", 2),
        MockLocation("City Street", "urban_street", 0),
    ]

    directions = ["north", "up", "down", "inside"]

    table = Table(title="Smart Expansion Examples")
    table.add_column("Source Location", style="cyan")
    table.add_column("Direction", style="yellow")
    table.add_column("Terrain Type", style="green")
    table.add_column("Generated Type", style="magenta")

    for location in test_locations:
        for direction in directions[:2]:  # Limit for display
            terrain = await game_loop._get_terrain_type(location)
            elevation = await game_loop._get_elevation_level(location, direction)
            new_type = await game_loop._apply_smart_expansion_rules(
                location.state_flags["location_type"],
                direction,
                terrain,
                elevation,
                location.state_flags["expansion_depth"],
            )
            table.add_row(location.name, direction, terrain, new_type)

    console.print(table)
    console.print(
        "[green]✓ Locations are generated based on terrain consistency and direction logic[/green]"
    )

    # Demo 2: Player Behavior Analysis
    console.print("\n" + "=" * 70)
    console.print(
        "[bold yellow]DEMO 2: Player Behavior Tracking & Analysis[/bold yellow]"
    )
    console.print("=" * 70)

    preferences = game_loop._get_player_preferences()

    behavior_table = Table(title="Player Behavior Analysis")
    behavior_table.add_column("Metric", style="cyan")
    behavior_table.add_column("Value", style="yellow")
    behavior_table.add_column("Impact", style="green")

    behavior_table.add_row(
        "Preferred Directions",
        str(preferences["preferred_directions"]),
        "Influences exit generation",
    )
    behavior_table.add_row(
        "Preferred Location Types",
        str(preferences["preferred_location_types"]),
        "Boosts content in these areas",
    )
    behavior_table.add_row(
        "Exploration Style", preferences["exploration_style"], "Affects content mix"
    )
    behavior_table.add_row(
        "Experience Level",
        preferences["experience_level"],
        "Influences content complexity",
    )
    behavior_table.add_row(
        "Total Expansions",
        str(preferences["total_expansions"]),
        "Determines experience level",
    )
    behavior_table.add_row(
        "Max Depth", str(preferences["max_depth"]), "Unlocks specialized content"
    )

    console.print(behavior_table)
    console.print(
        "[green]✓ System tracks player behavior and adapts generation accordingly[/green]"
    )

    # Demo 3: Adaptive Content Generation
    console.print("\n" + "=" * 70)
    console.print("[bold yellow]DEMO 3: Adaptive Content Generation[/bold yellow]")
    console.print("=" * 70)

    location_types = [
        "urban_street",
        "industrial_zone",
        "office_space",
        "basement_corridor",
    ]

    content_table = Table(title="Content Generation Probabilities")
    content_table.add_column("Location Type", style="cyan")
    content_table.add_column("Objects", style="yellow")
    content_table.add_column("NPCs", style="green")
    content_table.add_column("Interactions", style="magenta")
    content_table.add_column("Adaptation", style="blue")

    for loc_type in location_types:
        probs = game_loop._get_content_probability(loc_type, depth=1)
        adaptation = (
            "Boosted" if loc_type in preferences["preferred_location_types"] else "Base"
        )
        content_table.add_row(
            loc_type,
            f"{probs['objects']:.2f}",
            f"{probs['npcs']:.2f}",
            f"{probs['interactions']:.2f}",
            adaptation,
        )

    console.print(content_table)
    console.print(
        "[green]✓ Content generation adapts to player preferences and exploration style[/green]"
    )

    # Demo 4: Depth-Based Specialization
    console.print("\n" + "=" * 70)
    console.print("[bold yellow]DEMO 4: Depth-Based Specialization[/bold yellow]")
    console.print("=" * 70)

    console.print("[blue]Urban area expansion at different depths:[/blue]")

    urban_location = MockLocation("City Street", "urban_street", 0)
    terrain = await game_loop._get_terrain_type(urban_location)

    depth_table = Table(title="Depth-Based Location Generation")
    depth_table.add_column("Depth", style="cyan")
    depth_table.add_column("Generated Location Type", style="yellow")
    depth_table.add_column("Specialization Level", style="green")

    for depth in range(1, 6):
        location_type = await game_loop._apply_smart_expansion_rules(
            "urban_street", "north", terrain, 0, depth
        )
        specialization = "Basic" if depth < 3 else "Specialized/Unique"
        depth_table.add_row(str(depth), location_type, specialization)

    console.print(depth_table)
    console.print(
        "[green]✓ Deeper exploration yields more specialized and unique locations[/green]"
    )

    # Demo 5: Content Generation Example
    console.print("\n" + "=" * 70)
    console.print(
        "[bold yellow]DEMO 5: Dynamic Content Generation Example[/bold yellow]"
    )
    console.print("=" * 70)

    console.print(
        "[blue]Generating content for an Industrial Zone (player's preferred type):[/blue]"
    )

    probs = game_loop._get_content_probability("industrial_zone", depth=2)

    # Simulate content generation
    objects = [
        "machinery part",
        "tool box",
        "safety equipment",
        "control panel",
        "warning sign",
    ]
    npcs = ["factory worker", "supervisor", "maintenance tech", "safety inspector"]
    quest_hooks = [
        "warning notice",
        "maintenance request",
        "emergency protocol",
        "system alert",
    ]

    generated_content = []

    if random.random() < probs["objects"]:
        obj = random.choice(objects)
        generated_content.append(f"[yellow]Object:[/yellow] {obj}")

    if random.random() < probs["npcs"]:
        npc = random.choice(npcs)
        generated_content.append(f"[green]NPC:[/green] A {npc}")

    if random.random() < probs["interactions"]:
        hook = random.choice(quest_hooks)
        generated_content.append(f"[cyan]Quest Hook:[/cyan] {hook}")

    if generated_content:
        console.print("\n[bold]Generated Content:[/bold]")
        for content in generated_content:
            console.print(f"  • {content}")
    else:
        console.print(
            "\n[dim]No content generated this time (based on probabilities)[/dim]"
        )

    console.print(
        f"\n[dim]Generation probabilities: Objects={probs['objects']:.2f}, NPCs={probs['npcs']:.2f}, Interactions={probs['interactions']:.2f}[/dim]"
    )
    console.print(
        "[green]✓ Content is generated contextually and adapts to player preferences[/green]"
    )

    # Demo Summary
    console.print("\n" + "=" * 70)
    console.print("[bold green]PHASE 2 DEMO COMPLETE[/bold green]")
    console.print("=" * 70)

    summary_table = Table(title="Phase 2 Features Summary")
    summary_table.add_column("Feature", style="cyan")
    summary_table.add_column("Implementation", style="yellow")
    summary_table.add_column("Benefit", style="green")

    summary_table.add_row(
        "Smart Expansion",
        "Terrain-aware generation with directional logic",
        "Logical, consistent world building",
    )
    summary_table.add_row(
        "Dynamic Content",
        "NPCs, objects, quest hooks generated contextually",
        "Rich, varied location experiences",
    )
    summary_table.add_row(
        "Behavior Tracking",
        "Player preference analysis and classification",
        "Personalized game experience",
    )
    summary_table.add_row(
        "Adaptive Generation",
        "Content adapts to player style and preferences",
        "Engaging, tailored exploration",
    )
    summary_table.add_row(
        "Depth Specialization",
        "Unique content at deeper exploration levels",
        "Rewards for thorough exploration",
    )
    summary_table.add_row(
        "35+ Location Types",
        "Comprehensive terrain-based location catalog",
        "Diverse world variety",
    )

    console.print(summary_table)

    console.print(
        "\n[bold cyan]The dynamic world generation system now creates rich, adaptive experiences that learn from and respond to player behavior![/bold cyan]"
    )


if __name__ == "__main__":
    asyncio.run(demo_phase2_features())
