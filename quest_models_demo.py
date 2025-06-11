#!/usr/bin/env python3
"""
Quest Models Demo

Demonstrates the quest data models, validation, and type safety features.
"""

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from src.game_loop.quests.models import (
    Quest,
    QuestCategory,
    QuestCompletionResult,
    QuestDifficulty,
    QuestInteractionResult,
    QuestProgress,
    QuestStatus,
    QuestStep,
)

console = Console()


def demo_quest_enums():
    """Demonstrate quest enumerations."""
    console.print("📋 [bold yellow]Quest Enumerations Demo[/bold yellow]")

    # Categories
    categories_table = Table(title="Quest Categories")
    categories_table.add_column("Category", style="cyan")
    categories_table.add_column("Value", style="white")

    for category in QuestCategory:
        categories_table.add_row(category.name, category.value)

    console.print(categories_table)

    # Difficulties
    difficulties_table = Table(title="Quest Difficulties")
    difficulties_table.add_column("Difficulty", style="magenta")
    difficulties_table.add_column("Value", style="white")

    for difficulty in QuestDifficulty:
        difficulties_table.add_row(difficulty.name, difficulty.value)

    console.print(difficulties_table)

    # Status
    status_table = Table(title="Quest Status")
    status_table.add_column("Status", style="green")
    status_table.add_column("Value", style="white")

    for status in QuestStatus:
        status_table.add_row(status.name, status.value)

    console.print(status_table)


def demo_quest_step_validation():
    """Demonstrate QuestStep validation features."""
    console.print("\n" + "=" * 50)
    console.print("✅ [bold yellow]QuestStep Validation Demo[/bold yellow]")
    console.print("=" * 50)

    # Valid quest step
    try:
        valid_step = QuestStep(
            step_id="find_item",
            description="Find the magical sword",
            requirements={"location": "dungeon", "item": "magical_sword"},
            completion_conditions=["action_type:take", "item_obtained:magical_sword"],
        )
        console.print("✅ [green]Valid QuestStep created successfully[/green]")

        step_panel = Panel(
            f"[bold]Step ID:[/bold] {valid_step.step_id}\n"
            f"[bold]Description:[/bold] {valid_step.description}\n"
            f"[bold]Requirements:[/bold] {valid_step.requirements}\n"
            f"[bold]Conditions:[/bold] {valid_step.completion_conditions}\n"
            f"[bold]Optional:[/bold] {valid_step.optional}",
            title="Valid Quest Step",
            border_style="green",
        )
        console.print(step_panel)

    except ValueError as e:
        console.print(f"❌ [red]Validation failed: {e}[/red]")

    # Invalid quest steps (empty values)
    console.print("\n[cyan]Testing validation errors...[/cyan]")

    validation_tests = [
        (
            "Empty step_id",
            {
                "step_id": "",
                "description": "Valid",
                "requirements": {},
                "completion_conditions": ["test"],
            },
        ),
        (
            "Empty description",
            {
                "step_id": "valid",
                "description": "",
                "requirements": {},
                "completion_conditions": ["test"],
            },
        ),
        (
            "Empty conditions",
            {
                "step_id": "valid",
                "description": "Valid",
                "requirements": {},
                "completion_conditions": [],
            },
        ),
    ]

    for test_name, step_data in validation_tests:
        try:
            QuestStep(**step_data)
            console.print(f"❌ [red]{test_name}: Should have failed but didn't[/red]")
        except ValueError as e:
            console.print(f"✅ [green]{test_name}: Correctly caught - {e}[/green]")


def demo_quest_creation():
    """Demonstrate Quest creation and properties."""
    console.print("\n" + "=" * 50)
    console.print("🏗️ [bold yellow]Quest Creation Demo[/bold yellow]")
    console.print("=" * 50)

    # Create a complex quest
    quest_steps = [
        QuestStep(
            step_id="gather_intel",
            description="Gather information about the dragon's lair",
            requirements={"location": "tavern", "npc": "old_sage"},
            completion_conditions=["action_type:talk"],
        ),
        QuestStep(
            step_id="find_lair",
            description="Locate the dragon's lair in the mountains",
            requirements={"location": "mountain_path"},
            completion_conditions=["location:dragon_lair"],
        ),
        QuestStep(
            step_id="defeat_dragon",
            description="Defeat the ancient dragon",
            requirements={"enemy": "ancient_dragon"},
            completion_conditions=["combat_victory:ancient_dragon"],
        ),
        QuestStep(
            step_id="optional_treasure",
            description="Search for additional treasure",
            requirements={"location": "dragon_hoard"},
            completion_conditions=["action_type:search"],
            optional=True,
        ),
    ]

    epic_quest = Quest(
        quest_id="epic_dragon_quest",
        title="The Dragon's Bane",
        description="Defeat the ancient dragon terrorizing the kingdom",
        category=QuestCategory.COMBAT,
        difficulty=QuestDifficulty.LEGENDARY,
        steps=quest_steps,
        prerequisites=["basic_combat_training", "mountain_access"],
        rewards={
            "experience": 1000,
            "gold": 500,
            "items": ["dragon_scale_armor", "flame_sword"],
            "title": "Dragon Slayer",
        },
        time_limit=7200.0,  # 2 hours
        repeatable=False,
    )

    console.print("✅ [green]Epic quest created![/green]")

    # Display quest properties
    properties_table = Table(title="Quest Properties")
    properties_table.add_column("Property", style="cyan")
    properties_table.add_column("Value", style="white")

    properties_table.add_row("Quest ID", epic_quest.quest_id)
    properties_table.add_row("Title", epic_quest.title)
    properties_table.add_row("Category", epic_quest.category.value.title())
    properties_table.add_row("Difficulty", epic_quest.difficulty.value.title())
    properties_table.add_row("Total Steps", str(epic_quest.total_steps))
    properties_table.add_row("Required Steps", str(len(epic_quest.required_steps)))
    properties_table.add_row("Optional Steps", str(len(epic_quest.optional_steps)))
    properties_table.add_row("Prerequisites", str(len(epic_quest.prerequisites)))
    properties_table.add_row(
        "Time Limit",
        f"{epic_quest.time_limit/3600:.1f} hours" if epic_quest.time_limit else "None",
    )
    properties_table.add_row("Repeatable", "Yes" if epic_quest.repeatable else "No")

    console.print(properties_table)

    # Show quest step breakdown
    steps_table = Table(title="Quest Steps")
    steps_table.add_column("Step", style="yellow")
    steps_table.add_column("Description", style="white")
    steps_table.add_column("Type", style="magenta")

    for i, step in enumerate(epic_quest.steps, 1):
        step_type = "Optional" if step.optional else "Required"
        steps_table.add_row(f"Step {i}", step.description, step_type)

    console.print(steps_table)


def demo_quest_progress():
    """Demonstrate QuestProgress functionality."""
    console.print("\n" + "=" * 50)
    console.print("📈 [bold yellow]Quest Progress Demo[/bold yellow]")
    console.print("=" * 50)

    # Create quest progress
    progress = QuestProgress(
        quest_id="epic_dragon_quest",
        player_id="hero_player",
        status=QuestStatus.ACTIVE,
        current_step=1,
        completed_steps=["gather_intel"],
        step_progress={
            "gather_intel": {"talked_to_sage": True, "info_gathered": "lair_location"}
        },
    )

    console.print("✅ [green]Quest progress created![/green]")

    # Display initial state
    progress_panel = Panel(
        f"[bold]Quest:[/bold] {progress.quest_id}\n"
        f"[bold]Player:[/bold] {progress.player_id}\n"
        f"[bold]Status:[/bold] {progress.status.value.title()}\n"
        f"[bold]Current Step:[/bold] {progress.current_step}\n"
        f"[bold]Completed Steps:[/bold] {len(progress.completed_steps)}\n"
        f"[bold]Completion:[/bold] {progress.completion_percentage:.1f}%",
        title="Quest Progress State",
        border_style="blue",
    )
    console.print(progress_panel)

    # Demonstrate progress methods
    console.print("\n[cyan]Testing progress methods...[/cyan]")

    # Mark another step complete
    progress.mark_step_complete("find_lair")
    console.print("✅ [green]Marked 'find_lair' as complete[/green]")

    # Update step progress
    progress.update_step_progress(
        "defeat_dragon", {"dragon_health": 75, "battle_started": True}
    )
    console.print("✅ [green]Updated dragon battle progress[/green]")

    # Advance to next step
    progress.advance_to_next_step()
    console.print("✅ [green]Advanced to next step[/green]")

    # Show updated state
    updated_panel = Panel(
        f"[bold]Completed Steps:[/bold] {progress.completed_steps}\n"
        f"[bold]Current Step:[/bold] {progress.current_step}\n"
        f"[bold]Step Progress:[/bold] {progress.step_progress}\n"
        f"[bold]Completion:[/bold] {progress.completion_percentage:.1f}%",
        title="Updated Progress State",
        border_style="green",
    )
    console.print(updated_panel)


def demo_result_models():
    """Demonstrate quest result models."""
    console.print("\n" + "=" * 50)
    console.print("📤 [bold yellow]Quest Result Models Demo[/bold yellow]")
    console.print("=" * 50)

    # Quest interaction result
    interaction_result = QuestInteractionResult(
        success=True,
        message="Quest accepted successfully!",
        quest_id="epic_dragon_quest",
        updated_progress=None,
        rewards_granted={"experience": 50},
        errors=[],
    )

    console.print("✅ [green]Quest interaction result created[/green]")

    # Quest completion result
    completion_result = QuestCompletionResult(
        success=True,
        quest_id="epic_dragon_quest",
        final_progress=None,
        rewards_granted={
            "experience": 1000,
            "gold": 500,
            "items": ["dragon_scale_armor", "flame_sword"],
            "title": "Dragon Slayer",
        },
        completion_message="Congratulations! You have slain the ancient dragon and saved the kingdom!",
        errors=[],
    )

    console.print("✅ [green]Quest completion result created[/green]")

    # Display results
    results_table = Table(title="Result Models")
    results_table.add_column("Result Type", style="cyan")
    results_table.add_column("Success", style="green")
    results_table.add_column("Message/Rewards", style="white")

    results_table.add_row(
        "Interaction Result",
        "✅" if interaction_result.success else "❌",
        interaction_result.message,
    )

    rewards_str = ", ".join(
        [f"{k}: {v}" for k, v in completion_result.rewards_granted.items()]
    )
    results_table.add_row(
        "Completion Result",
        "✅" if completion_result.success else "❌",
        rewards_str[:50] + "..." if len(rewards_str) > 50 else rewards_str,
    )

    console.print(results_table)


def demo_code_examples():
    """Show code examples of quest system usage."""
    console.print("\n" + "=" * 50)
    console.print("💻 [bold yellow]Code Examples[/bold yellow]")
    console.print("=" * 50)

    # Quest creation example
    quest_code = """
# Creating a quest with validation
quest = Quest(
    quest_id="my_quest",
    title="Epic Adventure",
    description="A thrilling quest",
    category=QuestCategory.EXPLORATION,
    difficulty=QuestDifficulty.MEDIUM,
    steps=[
        QuestStep(
            step_id="step_1",
            description="First step",
            requirements={"location": "start"},
            completion_conditions=["action_type:begin"]
        )
    ],
    rewards={"xp": 100, "gold": 50}
)

# Accessing quest properties
print(f"Total steps: {quest.total_steps}")
print(f"Required steps: {len(quest.required_steps)}")
print(f"Optional steps: {len(quest.optional_steps)}")
"""

    console.print(Syntax(quest_code, "python", theme="monokai", line_numbers=True))

    # Progress tracking example
    progress_code = """
# Creating and managing quest progress
progress = QuestProgress(
    quest_id="my_quest",
    player_id="player_123", 
    status=QuestStatus.ACTIVE
)

# Updating progress
progress.mark_step_complete("step_1")
progress.update_step_progress("step_2", {"items_found": 3})
progress.advance_to_next_step()

# Check completion percentage
completion = progress.completion_percentage
print(f"Quest {completion:.1f}% complete")
"""

    console.print(Syntax(progress_code, "python", theme="monokai", line_numbers=True))


def main():
    """Run the quest models demo."""
    console.print(
        Panel.fit(
            "[bold blue]Quest Data Models Demo[/bold blue]\n\n"
            "Showcasing quest system data models:\n"
            "• Type-safe enumerations\n"
            "• Data validation\n"
            "• Quest creation and properties\n"
            "• Progress tracking\n"
            "• Result handling",
            border_style="bright_blue",
        )
    )

    demo_quest_enums()
    demo_quest_step_validation()
    demo_quest_creation()
    demo_quest_progress()
    demo_result_models()
    demo_code_examples()

    # Final summary
    console.print("\n" + "=" * 50)
    console.print("🎯 [bold green]Models Demo Complete![/bold green]")
    console.print("=" * 50)

    summary_panel = Panel(
        "[bold]Quest Models Features:[/bold]\n\n"
        "[green]✅ Data Integrity:[/green]\n"
        "• Strong type annotations\n"
        "• Comprehensive validation\n"
        "• Enum-based categorization\n"
        "• Required field enforcement\n\n"
        "[yellow]📋 Flexibility:[/yellow]\n"
        "• Optional quest steps\n"
        "• Dynamic progress tracking\n"
        "• Extensible reward system\n"
        "• Rich metadata support\n\n"
        "[blue]🔧 Developer Experience:[/blue]\n"
        "• Clear error messages\n"
        "• IDE autocompletion\n"
        "• Self-documenting code\n"
        "• Type safety guarantees",
        title="Quest Models Summary",
        border_style="bright_green",
    )
    console.print(summary_panel)


if __name__ == "__main__":
    main()
