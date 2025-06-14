#!/usr/bin/env python3
"""
Interactive Location Generation System Demonstration

This script provides an interactive demonstration of the location generation system
for validation and testing purposes. It simulates the key components and workflows
without requiring full implementation.
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.tree import Tree


# Mock data models for demonstration
@dataclass
class Location:
    location_id: str
    name: str
    description: str
    theme: str
    connections: dict[str, str]
    objects: list[str]
    metadata: dict[str, Any]


@dataclass
class ExpansionPoint:
    location_id: str
    direction: str
    priority: float
    context: dict[str, Any]


@dataclass
class LocationTheme:
    name: str
    description: str
    visual_elements: list[str]
    atmosphere: str
    typical_objects: list[str]
    compatibility_themes: list[str]


@dataclass
class PlayerPreferences:
    environments: list[str]
    interaction_style: str
    complexity_level: str
    visited_locations: int


@dataclass
class GeneratedLocation:
    name: str
    description: str
    theme: str
    location_type: str
    objects: list[str]
    connections: dict[str, str]
    special_features: list[str]
    validation_score: float


class LocationGenerationDemo:
    """Interactive demonstration of the location generation system."""

    def __init__(self):
        self.console = Console()
        self.current_world = self._initialize_sample_world()
        self.themes = self._initialize_themes()
        self.player_prefs = self._initialize_player_preferences()

    def _initialize_sample_world(self) -> dict[str, Location]:
        """Initialize a sample world with existing locations."""
        return {
            "forest_grove": Location(
                location_id="forest_grove",
                name="Enchanted Grove",
                description="A peaceful clearing surrounded by ancient oak trees. Sunlight filters through the canopy, creating dancing patterns on the moss-covered ground.",
                theme="forest",
                connections={"north": "ungenerated", "east": "meadow"},
                objects=["ancient_oak", "crystal_spring", "moss_covered_stones"],
                metadata={"danger_level": 1, "exploration_reward": 3},
            ),
            "meadow": Location(
                location_id="meadow",
                name="Wildflower Meadow",
                description="A vast expanse of colorful wildflowers stretches before you. Butterflies dance among the blooms while a gentle breeze carries the sweet scent of nectar.",
                theme="plains",
                connections={"west": "forest_grove", "north": "ungenerated"},
                objects=["wildflowers", "butterfly_swarm", "old_scarecrow"],
                metadata={"danger_level": 0, "exploration_reward": 2},
            ),
            "mountain_base": Location(
                location_id="mountain_base",
                name="Mountain Base Camp",
                description="The foot of towering peaks, where hardy travelers prepare for mountain ascents. Weathered equipment and supply caches hint at many expeditions.",
                theme="mountain",
                connections={"south": "ungenerated", "up": "ungenerated"},
                objects=["supply_cache", "climbing_gear", "weather_station"],
                metadata={"danger_level": 2, "exploration_reward": 4},
            ),
        }

    def _initialize_themes(self) -> dict[str, LocationTheme]:
        """Initialize available location themes."""
        return {
            "forest": LocationTheme(
                name="Forest",
                description="Dense woodland areas with trees, wildlife, and natural features",
                visual_elements=["trees", "undergrowth", "wildlife", "natural_light"],
                atmosphere="peaceful, mysterious, natural",
                typical_objects=["trees", "streams", "rocks", "wildlife", "mushrooms"],
                compatibility_themes=["plains", "mountain", "swamp"],
            ),
            "plains": LocationTheme(
                name="Plains",
                description="Open grasslands and meadows with wide visibility",
                visual_elements=["grass", "flowers", "open_sky", "distant_horizons"],
                atmosphere="open, peaceful, windy",
                typical_objects=["grass", "flowers", "stones", "paths", "wildlife"],
                compatibility_themes=["forest", "hills", "river"],
            ),
            "mountain": LocationTheme(
                name="Mountain",
                description="Rocky, elevated terrain with challenging navigation",
                visual_elements=["rock", "steep_slopes", "thin_air", "snow"],
                atmosphere="challenging, majestic, cold",
                typical_objects=["rocks", "cliffs", "caves", "ice", "equipment"],
                compatibility_themes=["forest", "cave", "plateau"],
            ),
            "cave": LocationTheme(
                name="Cave",
                description="Underground chambers and tunnels",
                visual_elements=["stone", "darkness", "echoes", "mineral_formations"],
                atmosphere="dark, mysterious, enclosed",
                typical_objects=["stalactites", "pools", "crystals", "bones", "echoes"],
                compatibility_themes=["mountain", "underground", "dungeon"],
            ),
            "ruins": LocationTheme(
                name="Ancient Ruins",
                description="Remnants of old civilizations and structures",
                visual_elements=[
                    "crumbling_stone",
                    "overgrown_vegetation",
                    "artifacts",
                ],
                atmosphere="mysterious, historical, haunting",
                typical_objects=[
                    "pillars",
                    "statues",
                    "inscriptions",
                    "artifacts",
                    "rubble",
                ],
                compatibility_themes=["forest", "desert", "swamp"],
            ),
        }

    def _initialize_player_preferences(self) -> PlayerPreferences:
        """Initialize sample player preferences."""
        return PlayerPreferences(
            environments=["forest", "mountain"],
            interaction_style="exploratory",
            complexity_level="moderate",
            visited_locations=12,
        )

    async def run_demo(self):
        """Run the interactive demonstration."""
        self.console.print(
            Panel.fit(
                "[bold blue]Location Generation System Demo[/bold blue]\n"
                "This interactive demo showcases the location generation system\n"
                "including theme management, context collection, and validation.",
                title="🌍 Game Loop Location Generation",
            )
        )

        while True:
            choice = self._show_main_menu()

            if choice == "1":
                await self._demo_world_overview()
            elif choice == "2":
                await self._demo_expansion_points()
            elif choice == "3":
                await self._demo_context_collection()
            elif choice == "4":
                await self._demo_location_generation()
            elif choice == "5":
                await self._demo_theme_validation()
            elif choice == "6":
                await self._demo_integration_workflow()
            elif choice == "7":
                self._demo_player_preferences()
            elif choice == "8":
                self._demo_performance_metrics()
            elif choice == "9":
                break

            self.console.print("\n" + "=" * 60 + "\n")

    def _show_main_menu(self) -> str:
        """Display the main demo menu."""
        self.console.print("\n[bold cyan]Demo Options:[/bold cyan]")
        options = [
            "1. View Current World State",
            "2. Identify Expansion Points",
            "3. Demonstrate Context Collection",
            "4. Generate New Location",
            "5. Validate Theme Consistency",
            "6. Full Integration Workflow",
            "7. Player Preference Analysis",
            "8. Performance Metrics Simulation",
            "9. Exit Demo",
        ]

        for option in options:
            self.console.print(f"  {option}")

        return Prompt.ask("\nSelect option", choices=[str(i) for i in range(1, 10)])

    async def _demo_world_overview(self):
        """Demonstrate current world state visualization."""
        self.console.print(Panel("[bold green]Current World State[/bold green]"))

        # Create world map visualization
        tree = Tree("🌍 Game World")

        for loc_id, location in self.current_world.items():
            loc_branch = tree.add(f"📍 {location.name} ({location.theme})")
            loc_branch.add(f"Description: {location.description[:60]}...")

            connections = loc_branch.add("🔗 Connections:")
            for direction, target in location.connections.items():
                if target == "ungenerated":
                    connections.add(f"[red]{direction}: UNEXPLORED[/red]")
                else:
                    connections.add(f"[green]{direction}: {target}[/green]")

            objects = loc_branch.add("📦 Objects:")
            for obj in location.objects:
                objects.add(f"• {obj}")

        self.console.print(tree)

        # Show theme distribution
        theme_table = Table(title="Theme Distribution")
        theme_table.add_column("Theme")
        theme_table.add_column("Count")
        theme_table.add_column("Locations")

        theme_counts = {}
        for location in self.current_world.values():
            theme = location.theme
            if theme not in theme_counts:
                theme_counts[theme] = []
            theme_counts[theme].append(location.name)

        for theme, locations in theme_counts.items():
            theme_table.add_row(theme, str(len(locations)), ", ".join(locations))

        self.console.print(theme_table)

    async def _demo_expansion_points(self):
        """Demonstrate expansion point identification."""
        self.console.print(Panel("[bold green]Expansion Point Analysis[/bold green]"))

        expansion_points = []

        # Simulate boundary detection
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Analyzing world boundaries...", total=None)
            await asyncio.sleep(1)  # Simulate processing

        # Identify expansion points
        for loc_id, location in self.current_world.items():
            for direction, target in location.connections.items():
                if target == "ungenerated":
                    # Calculate priority based on location properties
                    priority = self._calculate_expansion_priority(location, direction)

                    expansion_points.append(
                        ExpansionPoint(
                            location_id=loc_id,
                            direction=direction,
                            priority=priority,
                            context={
                                "source_theme": location.theme,
                                "source_name": location.name,
                                "danger_level": location.metadata.get(
                                    "danger_level", 1
                                ),
                            },
                        )
                    )

        # Sort by priority
        expansion_points.sort(key=lambda x: x.priority, reverse=True)

        # Display expansion points
        exp_table = Table(title="Identified Expansion Points")
        exp_table.add_column("Priority", style="bold")
        exp_table.add_column("Location")
        exp_table.add_column("Direction")
        exp_table.add_column("Source Theme")
        exp_table.add_column("Context")

        for point in expansion_points:
            location = self.current_world[point.location_id]
            exp_table.add_row(
                f"{point.priority:.2f}",
                location.name,
                point.direction,
                point.context["source_theme"],
                f"Danger: {point.context['danger_level']}",
            )

        self.console.print(exp_table)

        # Allow user to select an expansion point
        if expansion_points and Confirm.ask(
            "Would you like to explore an expansion point?"
        ):
            choice = Prompt.ask(
                "Select expansion point by index",
                choices=[str(i) for i in range(len(expansion_points))],
                default="0",
            )
            selected_point = expansion_points[int(choice)]

            self.console.print(
                Panel(
                    f"[bold]Selected Expansion Point:[/bold]\n"
                    f"Location: {self.current_world[selected_point.location_id].name}\n"
                    f"Direction: {selected_point.direction}\n"
                    f"Priority: {selected_point.priority:.2f}\n"
                    f"Context: {json.dumps(selected_point.context, indent=2)}",
                    title="🎯 Expansion Point Details",
                )
            )

            return selected_point

        return None

    def _calculate_expansion_priority(
        self, location: Location, direction: str
    ) -> float:
        """Calculate expansion priority for a given direction."""
        base_priority = 5.0

        # Higher priority for directions from interesting locations
        exploration_reward = location.metadata.get("exploration_reward", 2)
        base_priority += exploration_reward * 0.5

        # Adjust based on danger level (moderate danger = higher priority)
        danger_level = location.metadata.get("danger_level", 1)
        if danger_level == 1:  # Sweet spot
            base_priority += 1.0
        elif danger_level == 0:  # Too easy
            base_priority += 0.5
        elif danger_level >= 3:  # Too dangerous
            base_priority -= 1.0

        # Cardinal directions get slight preference
        if direction in ["north", "south", "east", "west"]:
            base_priority += 0.5

        # Add some randomness for variety
        import random

        base_priority += random.uniform(-0.5, 0.5)

        return max(0.0, base_priority)

    async def _demo_context_collection(self):
        """Demonstrate context collection for location generation."""
        self.console.print(Panel("[bold green]Context Collection Demo[/bold green]"))

        # Let user select a location to expand from
        location_choices = list(self.current_world.keys())
        choice = Prompt.ask(
            "Select location to expand from",
            choices=location_choices,
            default=location_choices[0],
        )

        selected_location = self.current_world[choice]

        # Simulate context collection process
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:

            task1 = progress.add_task("Analyzing player preferences...", total=None)
            await asyncio.sleep(0.8)
            progress.update(task1, description="✓ Player preferences analyzed")

            task2 = progress.add_task(
                "Gathering adjacent location context...", total=None
            )
            await asyncio.sleep(0.6)
            progress.update(task2, description="✓ Adjacent context collected")

            task3 = progress.add_task("Extracting narrative context...", total=None)
            await asyncio.sleep(0.7)
            progress.update(task3, description="✓ Narrative context extracted")

            task4 = progress.add_task("Calculating theme compatibility...", total=None)
            await asyncio.sleep(0.5)
            progress.update(task4, description="✓ Theme compatibility calculated")

        # Display collected context
        context = {
            "source_location": {
                "name": selected_location.name,
                "theme": selected_location.theme,
                "atmosphere": self.themes[selected_location.theme].atmosphere,
                "objects": selected_location.objects,
            },
            "player_preferences": {
                "preferred_environments": self.player_prefs.environments,
                "interaction_style": self.player_prefs.interaction_style,
                "experience_level": self.player_prefs.complexity_level,
                "exploration_history": f"{self.player_prefs.visited_locations} locations visited",
            },
            "adjacent_themes": self._get_adjacent_themes(selected_location),
            "narrative_context": {
                "suggested_progression": "gradual difficulty increase",
                "thematic_continuity": "maintain natural flow",
                "discovery_potential": "introduce new elements",
            },
            "generation_parameters": {
                "complexity_target": self.player_prefs.complexity_level,
                "danger_level_range": [0, 3],
                "required_connections": ["back_to_source"],
                "optional_connections": ["forward_exploration"],
            },
        }

        self.console.print(
            Panel(JSON.from_data(context), title="📋 Collected Generation Context")
        )

        return context

    def _get_adjacent_themes(self, location: Location) -> list[str]:
        """Get themes of adjacent locations."""
        adjacent_themes = []
        for direction, target_id in location.connections.items():
            if target_id != "ungenerated" and target_id in self.current_world:
                target_location = self.current_world[target_id]
                adjacent_themes.append(target_location.theme)
        return list(set(adjacent_themes))

    async def _demo_location_generation(self):
        """Demonstrate the location generation process."""
        self.console.print(Panel("[bold green]Location Generation Demo[/bold green]"))

        # Get context first
        context = await self._demo_context_collection()

        # Let user choose direction
        direction = Prompt.ask(
            "Direction for new location",
            choices=["north", "south", "east", "west", "up", "down"],
            default="north",
        )

        # Simulate LLM generation process
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:

            task1 = progress.add_task("Preparing generation prompts...", total=None)
            await asyncio.sleep(0.8)

            task2 = progress.add_task("Sending request to LLM...", total=None)
            await asyncio.sleep(2.0)  # Simulate LLM processing time

            task3 = progress.add_task("Processing LLM response...", total=None)
            await asyncio.sleep(0.5)

            task4 = progress.add_task("Validating generated content...", total=None)
            await asyncio.sleep(0.7)

        # Generate location based on context
        generated_location = self._generate_mock_location(context, direction)

        # Display generated location
        self.console.print(
            Panel(
                f"[bold cyan]Generated Location:[/bold cyan]\n\n"
                f"[bold]Name:[/bold] {generated_location.name}\n"
                f"[bold]Theme:[/bold] {generated_location.theme}\n"
                f"[bold]Type:[/bold] {generated_location.location_type}\n\n"
                f"[bold]Description:[/bold]\n{generated_location.description}\n\n"
                f"[bold]Objects:[/bold]\n"
                + "\n".join(f"• {obj}" for obj in generated_location.objects)
                + "\n\n"
                "[bold]Special Features:[/bold]\n"
                + "\n".join(
                    f"• {feature}" for feature in generated_location.special_features
                )
                + "\n\n"
                "[bold]Connections:[/bold]\n"
                + "\n".join(
                    f"• {dir}: {target}"
                    for dir, target in generated_location.connections.items()
                )
                + "\n\n"
                f"[bold]Validation Score:[/bold] {generated_location.validation_score:.2f}/10.0",
                title="🎨 Generated Location",
                width=80,
            )
        )

        # Ask if user wants to add to world
        if Confirm.ask("Add this location to the world?"):
            new_location = Location(
                location_id=f"generated_{len(self.current_world)}",
                name=generated_location.name,
                description=generated_location.description,
                theme=generated_location.theme,
                connections={direction.replace("to_", ""): "back"},  # Simplified
                objects=generated_location.objects,
                metadata={
                    "generated": True,
                    "validation_score": generated_location.validation_score,
                },
            )

            self.current_world[new_location.location_id] = new_location
            self.console.print("[green]✓ Location added to world![/green]")

        return generated_location

    def _generate_mock_location(
        self, context: dict, direction: str
    ) -> GeneratedLocation:
        """Generate a mock location based on context."""
        import random

        source_theme = context["source_location"]["theme"]
        compatible_themes = self.themes[source_theme].compatibility_themes

        # Choose theme based on compatibility and player preferences
        possible_themes = list(
            set(compatible_themes) & set(self.player_prefs.environments)
        )
        if not possible_themes:
            possible_themes = compatible_themes

        chosen_theme = random.choice(possible_themes)
        theme_data = self.themes[chosen_theme]

        # Generate location components
        location_types = {
            "forest": ["clearing", "thicket", "grove", "woodland_path"],
            "mountain": ["peak", "ridge", "cliff", "cave_entrance"],
            "plains": ["meadow", "grassland", "hill", "valley"],
            "cave": ["chamber", "tunnel", "grotto", "cavern"],
            "ruins": ["temple", "tower", "courtyard", "tomb"],
        }

        location_type = random.choice(location_types.get(chosen_theme, ["area"]))

        # Generate name
        name_prefixes = {
            "forest": ["Whispering", "Ancient", "Shadowed", "Moonlit"],
            "mountain": ["Windswept", "Towering", "Jagged", "Frozen"],
            "plains": ["Rolling", "Golden", "Endless", "Peaceful"],
            "cave": ["Hidden", "Crystal", "Echoing", "Deep"],
            "ruins": ["Forgotten", "Crumbling", "Lost", "Sacred"],
        }

        prefix = random.choice(name_prefixes.get(chosen_theme, ["Mysterious"]))
        name = f"{prefix} {location_type.replace('_', ' ').title()}"

        # Generate description
        descriptions = {
            "forest": [
                "Tall trees create a natural cathedral overhead, their branches intertwining to filter the light into dappled patterns.",
                "The air is rich with the scent of earth and growing things, while bird songs echo through the canopy.",
                "Moss-covered stones and fallen logs create natural seating areas throughout this woodland sanctuary.",
            ],
            "mountain": [
                "The rocky terrain challenges every step, with loose stones and steep inclines testing your resolve.",
                "Wind howls across the exposed stone faces, carrying the crisp scent of high altitude air.",
                "The view from here reveals vast distances, with lower peaks stretching to the horizon.",
            ],
            "plains": [
                "Grass waves in the breeze like a green ocean, stretching endlessly in all directions.",
                "Wildflowers dot the landscape with splashes of color, attracting butterflies and bees.",
                "The openness here brings a sense of freedom, with nothing to obstruct the vast sky above.",
            ],
            "cave": [
                "Darkness presses in from all sides, broken only by the glimmer of mineral formations.",
                "The air is cool and still, carrying the sound of distant water dripping.",
                "Ancient rock formations create natural sculptures in this underground chamber.",
            ],
            "ruins": [
                "Crumbling stone speaks of a civilization long past, with intricate carvings still visible.",
                "Vines and moss have claimed much of the structure, nature reclaiming its domain.",
                "The silence here feels heavy with history and forgotten stories.",
            ],
        }

        description_parts = descriptions.get(
            chosen_theme, ["An interesting location awaits exploration."]
        )
        description = " ".join(
            random.sample(description_parts, min(len(description_parts), 2))
        )

        # Generate objects
        objects = random.sample(
            theme_data.typical_objects, min(len(theme_data.typical_objects), 3)
        )

        # Generate special features
        special_features = [
            f"Interactive {random.choice(objects)}",
            f"Hidden passage to {random.choice(['north', 'east', 'secret chamber'])}",
            f"Atmospheric effect: {theme_data.atmosphere}",
        ]

        # Generate connections
        connections = {f"to_{direction}": "source_location"}

        # Add forward connection if appropriate
        if random.random() > 0.3:  # 70% chance of forward connection
            forward_dirs = ["north", "east", "south", "west"]
            if direction in forward_dirs:
                forward_dirs.remove(direction)
            connections[random.choice(forward_dirs)] = "unexplored"

        # Calculate validation score
        validation_score = self._calculate_validation_score(
            chosen_theme, source_theme, context
        )

        return GeneratedLocation(
            name=name,
            description=description,
            theme=chosen_theme,
            location_type=location_type,
            objects=objects,
            connections=connections,
            special_features=special_features[:2],  # Limit to 2 features
            validation_score=validation_score,
        )

    def _calculate_validation_score(
        self, new_theme: str, source_theme: str, context: dict
    ) -> float:
        """Calculate a validation score for the generated location."""
        score = 5.0  # Base score

        # Theme compatibility
        if new_theme in self.themes[source_theme].compatibility_themes:
            score += 2.0
        else:
            score -= 1.0

        # Player preference alignment
        if new_theme in self.player_prefs.environments:
            score += 1.5

        # Complexity matching
        if self.player_prefs.complexity_level == "moderate":
            score += 1.0

        # Random variation for realism
        import random

        score += random.uniform(-0.5, 0.5)

        return max(1.0, min(10.0, score))

    async def _demo_theme_validation(self):
        """Demonstrate theme consistency validation."""
        self.console.print(Panel("[bold green]Theme Validation Demo[/bold green]"))

        # Show theme compatibility matrix
        self.console.print("[bold]Theme Compatibility Matrix:[/bold]")

        theme_table = Table()
        theme_table.add_column("Theme")
        for theme_name in self.themes.keys():
            theme_table.add_column(theme_name[:8])

        for theme_name, theme_data in self.themes.items():
            row = [theme_name]
            for other_theme in self.themes.keys():
                if other_theme in theme_data.compatibility_themes:
                    row.append("[green]✓[/green]")
                elif other_theme == theme_name:
                    row.append("[blue]●[/blue]")
                else:
                    row.append("[red]✗[/red]")
            theme_table.add_row(*row)

        self.console.print(theme_table)

        # Demonstrate validation process
        self.console.print("\n[bold]Validation Process Demo:[/bold]")

        # Create a test scenario
        test_scenarios = [
            {
                "source": "forest",
                "target": "plains",
                "description": "Transitioning from dense forest to open meadow",
                "expected": "VALID",
            },
            {
                "source": "mountain",
                "target": "cave",
                "description": "Mountain cave entrance",
                "expected": "VALID",
            },
            {
                "source": "plains",
                "target": "cave",
                "description": "Cave opening in flat grassland",
                "expected": "QUESTIONABLE",
            },
            {
                "source": "forest",
                "target": "ruins",
                "description": "Ancient ruins hidden in the woods",
                "expected": "VALID",
            },
        ]

        for scenario in test_scenarios:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
            ) as progress:
                task = progress.add_task(
                    f"Validating {scenario['source']} → {scenario['target']}...",
                    total=None,
                )
                await asyncio.sleep(0.8)

            # Calculate validation
            is_compatible = (
                scenario["target"]
                in self.themes[scenario["source"]].compatibility_themes
            )

            if is_compatible:
                status_color = "green"
                status = "✓ VALID"
            else:
                status_color = (
                    "yellow" if scenario["expected"] == "QUESTIONABLE" else "red"
                )
                status = (
                    "⚠ QUESTIONABLE"
                    if scenario["expected"] == "QUESTIONABLE"
                    else "✗ INVALID"
                )

            self.console.print(
                f"[{status_color}]{status}[/{status_color}] "
                f"{scenario['source'].title()} → {scenario['target'].title()}: "
                f"{scenario['description']}"
            )

        # Interactive validation
        if Confirm.ask("\nTry interactive theme validation?"):
            source_theme = Prompt.ask(
                "Source theme", choices=list(self.themes.keys()), default="forest"
            )

            target_theme = Prompt.ask(
                "Target theme", choices=list(self.themes.keys()), default="plains"
            )

            is_compatible = (
                target_theme in self.themes[source_theme].compatibility_themes
            )
            compatibility_score = 8.5 if is_compatible else 3.2

            validation_result = {
                "themes": f"{source_theme} → {target_theme}",
                "compatible": is_compatible,
                "compatibility_score": compatibility_score,
                "reasoning": [
                    f"Theme transition from {source_theme} to {target_theme}",
                    f"Compatibility: {'Yes' if is_compatible else 'No'}",
                    f"Atmospheric match: {'Good' if is_compatible else 'Poor'}",
                    f"Visual coherence: {'Maintained' if is_compatible else 'Disrupted'}",
                ],
                "recommendations": [
                    (
                        "Add transitional elements"
                        if not is_compatible
                        else "Direct transition acceptable"
                    ),
                    (
                        "Consider gradual theme shift"
                        if not is_compatible
                        else "Maintain theme consistency"
                    ),
                ],
            }

            self.console.print(
                Panel(JSON.from_data(validation_result), title="🔍 Validation Result")
            )

    async def _demo_integration_workflow(self):
        """Demonstrate the full integration workflow."""
        self.console.print(
            Panel("[bold green]Full Integration Workflow Demo[/bold green]")
        )

        self.console.print("This demo shows the complete location generation pipeline:")

        # Step 1: Boundary Detection
        self.console.print("\n[bold cyan]Step 1: Boundary Detection[/bold cyan]")
        expansion_point = await self._demo_expansion_points()

        if not expansion_point:
            self.console.print(
                "[red]No expansion point selected. Workflow terminated.[/red]"
            )
            return

        # Step 2: Context Collection
        self.console.print("\n[bold cyan]Step 2: Context Collection[/bold cyan]")
        await asyncio.sleep(0.5)  # Brief pause
        context = await self._demo_context_collection()

        # Step 3: Location Generation
        self.console.print("\n[bold cyan]Step 3: Location Generation[/bold cyan]")
        await asyncio.sleep(0.5)
        generated_location = await self._demo_location_generation()

        # Step 4: Validation
        self.console.print("\n[bold cyan]Step 4: Theme Validation[/bold cyan]")
        await asyncio.sleep(0.5)

        source_theme = context["source_location"]["theme"]
        target_theme = generated_location.theme
        is_valid = target_theme in self.themes[source_theme].compatibility_themes

        self.console.print(
            f"Validation Result: [{'green' if is_valid else 'red'}]{'VALID' if is_valid else 'INVALID'}[/]"
        )

        # Step 5: Integration
        self.console.print("\n[bold cyan]Step 5: World Integration[/bold cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:

            task1 = progress.add_task("Storing location in database...", total=None)
            await asyncio.sleep(0.8)

            task2 = progress.add_task("Generating embeddings...", total=None)
            await asyncio.sleep(1.2)

            task3 = progress.add_task("Updating connection graph...", total=None)
            await asyncio.sleep(0.6)

            task4 = progress.add_task("Updating world state...", total=None)
            await asyncio.sleep(0.4)

        # Show final result
        workflow_result = {
            "status": "completed",
            "location_added": generated_location.name,
            "theme": generated_location.theme,
            "validation_score": generated_location.validation_score,
            "integration_time": "3.2 seconds",
            "database_stored": True,
            "embeddings_generated": True,
            "connections_updated": True,
            "world_state_updated": True,
        }

        self.console.print(
            Panel(JSON.from_data(workflow_result), title="✅ Integration Complete")
        )

    def _demo_player_preferences(self):
        """Demonstrate player preference analysis."""
        self.console.print(Panel("[bold green]Player Preference Analysis[/bold green]"))

        # Show current preferences
        prefs_table = Table(title="Current Player Preferences")
        prefs_table.add_column("Aspect")
        prefs_table.add_column("Value")
        prefs_table.add_column("Influence on Generation")

        prefs_table.add_row(
            "Preferred Environments",
            ", ".join(self.player_prefs.environments),
            "Higher priority for these themes",
        )
        prefs_table.add_row(
            "Interaction Style",
            self.player_prefs.interaction_style,
            "Affects object and feature generation",
        )
        prefs_table.add_row(
            "Complexity Level",
            self.player_prefs.complexity_level,
            "Influences puzzle and challenge difficulty",
        )
        prefs_table.add_row(
            "Experience",
            f"{self.player_prefs.visited_locations} locations visited",
            "Higher experience = more complex content",
        )

        self.console.print(prefs_table)

        # Show preference learning simulation
        self.console.print("\n[bold]Preference Learning Simulation:[/bold]")

        learning_examples = [
            "Player spends 10+ minutes examining forest objects → Increase exploration preference",
            "Player avoids combat-heavy mountain areas → Decrease danger tolerance",
            "Player seeks out ruins repeatedly → Add 'historical' to preferred themes",
            "Player uses complex puzzle solutions → Increase complexity preference",
        ]

        for example in learning_examples:
            self.console.print(f"• {example}")

        # Allow preference modification
        if Confirm.ask("\nModify player preferences for testing?"):
            new_env = Prompt.ask(
                "Add preferred environment",
                choices=list(self.themes.keys()),
                default="cave",
            )

            if new_env not in self.player_prefs.environments:
                self.player_prefs.environments.append(new_env)
                self.console.print(
                    f"[green]✓ Added {new_env} to preferred environments[/green]"
                )

    def _demo_performance_metrics(self):
        """Demonstrate performance metrics simulation."""
        self.console.print(
            Panel("[bold green]Performance Metrics Simulation[/bold green]")
        )

        import random

        # Simulate performance metrics
        metrics = {
            "Location Generation": {
                "avg_time": f"{random.uniform(2.5, 4.2):.2f}s",
                "success_rate": f"{random.uniform(92, 98):.1f}%",
                "cache_hit_rate": f"{random.uniform(35, 55):.1f}%",
                "llm_calls": random.randint(1, 3),
            },
            "Context Collection": {
                "avg_time": f"{random.uniform(0.8, 1.5):.2f}s",
                "data_sources": random.randint(4, 7),
                "cache_hit_rate": f"{random.uniform(60, 85):.1f}%",
            },
            "Theme Validation": {
                "avg_time": f"{random.uniform(0.3, 0.8):.2f}s",
                "validation_accuracy": f"{random.uniform(88, 96):.1f}%",
                "false_positives": f"{random.uniform(2, 8):.1f}%",
            },
            "Database Operations": {
                "storage_time": f"{random.uniform(200, 500):.0f}ms",
                "retrieval_time": f"{random.uniform(50, 150):.0f}ms",
                "embedding_time": f"{random.uniform(800, 1500):.0f}ms",
            },
        }

        for category, category_metrics in metrics.items():
            table = Table(title=category)
            table.add_column("Metric")
            table.add_column("Value")

            for metric, value in category_metrics.items():
                table.add_row(metric.replace("_", " ").title(), str(value))

            self.console.print(table)
            self.console.print()

        # Show performance recommendations
        recommendations = [
            "✓ Cache hit rates are within target ranges",
            "⚠ Consider pre-generating common location types",
            "✓ LLM response times are acceptable",
            "💡 Embedding generation could benefit from batching",
            "✓ Database performance is optimal",
        ]

        self.console.print(
            Panel("\n".join(recommendations), title="🚀 Performance Recommendations")
        )


async def main():
    """Main entry point for the demo."""
    demo = LocationGenerationDemo()
    await demo.run_demo()

    demo.console.print(
        Panel(
            "[bold green]Demo Complete![/bold green]\n\n"
            "This demonstration showcased:\n"
            "• World boundary detection and expansion points\n"
            "• Context collection from multiple sources\n"
            "• LLM-powered location generation\n"
            "• Theme consistency validation\n"
            "• Full integration workflow\n"
            "• Player preference analysis\n"
            "• Performance monitoring\n\n"
            "The actual implementation will provide these capabilities\n"
            "with real LLM integration, database persistence, and\n"
            "embedding generation for semantic search.",
            title="🎉 Location Generation System Demo",
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
