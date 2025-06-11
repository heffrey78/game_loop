#!/usr/bin/env python3
"""
Simple Quest System Demo

A focused demonstration of the Quest Interaction System core functionality.
"""

import asyncio
import uuid
from unittest.mock import AsyncMock

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Import quest system components
from src.game_loop.quests.models import (
    Quest,
    QuestCategory,
    QuestDifficulty,
    QuestProgress,
    QuestStatus,
    QuestStep,
)
from src.game_loop.quests.quest_manager import QuestManager
from src.game_loop.state.models import ActionResult

console = Console()


async def create_simple_demo():
    """Create a simple working demo of the quest system."""

    console.print(
        Panel.fit(
            "[bold blue]Quest System Core Functionality Demo[/bold blue]\n\n"
            "Demonstrating:\n"
            "• Quest creation and validation\n"
            "• Quest management operations\n"
            "• Progress tracking\n"
            "• Reward system",
            border_style="bright_blue",
        )
    )

    # Create mock repository
    mock_repo = AsyncMock()
    quest_manager = QuestManager(mock_repo)

    # === Demo 1: Quest Creation ===
    console.print("\n" + "=" * 50)
    console.print("🏗️  [bold yellow]Quest Creation Demo[/bold yellow]")
    console.print("=" * 50)

    # Create a sample quest
    sample_quest = Quest(
        quest_id="demo_quest_001",
        title="Find the Lost Artifact",
        description="Recover the ancient crystal from the forgotten temple",
        category=QuestCategory.EXPLORATION,
        difficulty=QuestDifficulty.MEDIUM,
        steps=[
            QuestStep(
                step_id="enter_temple",
                description="Enter the forgotten temple",
                requirements={"location": "temple_entrance"},
                completion_conditions=["location:temple_interior"],
            ),
            QuestStep(
                step_id="find_crystal",
                description="Locate the ancient crystal",
                requirements={"target_object": "ancient_crystal"},
                completion_conditions=["action_type:take"],
            ),
        ],
        rewards={"experience": 250, "gold": 100, "items": ["ancient_crystal"]},
    )

    console.print("✅ [green]Created quest successfully![/green]")

    # Display quest info
    quest_table = Table(title="Quest Details")
    quest_table.add_column("Property", style="cyan")
    quest_table.add_column("Value", style="white")

    quest_table.add_row("Title", sample_quest.title)
    quest_table.add_row("Category", sample_quest.category.value.title())
    quest_table.add_row("Difficulty", sample_quest.difficulty.value.title())
    quest_table.add_row("Steps", str(sample_quest.total_steps))
    quest_table.add_row("Required Steps", str(len(sample_quest.required_steps)))
    quest_table.add_row("Repeatable", "Yes" if sample_quest.repeatable else "No")

    console.print(quest_table)

    # === Demo 2: Quest Validation ===
    console.print("\n" + "=" * 50)
    console.print("✅ [bold yellow]Quest Validation Demo[/bold yellow]")
    console.print("=" * 50)

    player_id = "demo_player"

    # Mock repository responses
    mock_repo.get_quest.return_value = sample_quest
    mock_repo.get_player_progress.return_value = None

    # Test prerequisite validation
    is_valid, errors = await quest_manager.validate_quest_prerequisites(
        player_id, "demo_quest_001"
    )

    if is_valid:
        console.print("✅ [green]Quest prerequisites validated successfully![/green]")
        console.print("🎯 [blue]Player can accept this quest[/blue]")
    else:
        console.print("❌ [red]Quest validation failed:[/red]")
        for error in errors:
            console.print(f"   • {error}")

    # === Demo 3: Quest Start ===
    console.print("\n" + "=" * 50)
    console.print("🚀 [bold yellow]Quest Start Demo[/bold yellow]")
    console.print("=" * 50)

    mock_repo.update_progress.return_value = True
    mock_repo.log_interaction.return_value = True

    # Start the quest
    success, start_errors = await quest_manager.start_quest(player_id, "demo_quest_001")

    if success:
        console.print("✅ [green]Quest started successfully![/green]")

        start_panel = Panel(
            f"[bold]{sample_quest.title}[/bold]\n\n"
            f"{sample_quest.description}\n\n"
            "[yellow]Current Objective:[/yellow]\n"
            f"• {sample_quest.steps[0].description}\n\n"
            "[yellow]Rewards on Completion:[/yellow]\n"
            "• 250 Experience Points\n"
            "• 100 Gold\n"
            "• Ancient Crystal (Item)",
            title="Quest Active",
            border_style="green",
        )
        console.print(start_panel)
    else:
        console.print("❌ [red]Failed to start quest:[/red]")
        for error in start_errors:
            console.print(f"   • {error}")

    # === Demo 4: Progress Tracking ===
    console.print("\n" + "=" * 50)
    console.print("📈 [bold yellow]Progress Tracking Demo[/bold yellow]")
    console.print("=" * 50)

    # Create mock progress
    active_progress = QuestProgress(
        quest_id="demo_quest_001",
        player_id=player_id,
        status=QuestStatus.ACTIVE,
        current_step=0,
        completed_steps=[],
        step_progress={},
    )

    mock_repo.get_player_progress.return_value = active_progress

    # Simulate completing first step
    console.print("[cyan]Player enters the temple...[/cyan]")

    action_result = ActionResult(
        success=True,
        feedback_message="You step into the ancient temple",
        command="enter",
        location_change=True,
        new_location_id=uuid.uuid4(),
    )

    # Check step completion
    step_completed = await quest_manager.check_step_completion_conditions(
        player_id, "demo_quest_001", "enter_temple", action_result
    )

    if step_completed:
        console.print("✅ [green]Step completed: Enter the temple[/green]")

        # Update progress
        success = await quest_manager.update_quest_progress(
            player_id, "demo_quest_001", action_result
        )

        if success:
            console.print("📊 [blue]Quest progress updated![/blue]")

            progress_panel = Panel(
                "[green]✅ Step 1 Complete:[/green] Enter the forgotten temple\n"
                "[yellow]📍 Next Step:[/yellow] Locate the ancient crystal\n\n"
                "[dim]Progress: 1/2 steps completed (50%)[/dim]",
                title="Quest Progress",
                border_style="yellow",
            )
            console.print(progress_panel)

    # === Demo 5: Reward System ===
    console.print("\n" + "=" * 50)
    console.print("🎁 [bold yellow]Reward System Demo[/bold yellow]")
    console.print("=" * 50)

    # Simulate granting rewards
    console.print("[cyan]Simulating quest completion rewards...[/cyan]")

    rewards = await quest_manager.grant_quest_rewards(
        player_id, sample_quest.rewards, {"quest_id": "demo_quest_001"}
    )

    console.print("🎉 [green]Rewards granted successfully![/green]")

    # Display rewards
    rewards_table = Table(title="Rewards Granted")
    rewards_table.add_column("Type", style="yellow")
    rewards_table.add_column("Amount", style="green")

    for reward_type, reward_value in rewards.items():
        if isinstance(reward_value, list):
            value_str = ", ".join(str(item) for item in reward_value)
        else:
            value_str = str(reward_value)
        rewards_table.add_row(reward_type.title(), value_str)

    console.print(rewards_table)

    # === Demo 6: Dynamic Quest Generation ===
    console.print("\n" + "=" * 50)
    console.print("🎲 [bold yellow]Dynamic Quest Generation Demo[/bold yellow]")
    console.print("=" * 50)

    # Generate a dynamic delivery quest
    console.print("[cyan]Generating a dynamic delivery quest...[/cyan]")

    dynamic_quest = await quest_manager.generate_dynamic_quest(
        player_id,
        context={"pickup_location": "marketplace", "delivery_location": "castle"},
        quest_type="delivery",
    )

    if dynamic_quest:
        console.print("✅ [green]Dynamic quest generated![/green]")

        dynamic_panel = Panel(
            f"[bold]{dynamic_quest.title}[/bold]\n\n"
            f"{dynamic_quest.description}\n\n"
            f"[yellow]Category:[/yellow] {dynamic_quest.category.value.title()}\n"
            f"[yellow]Difficulty:[/yellow] {dynamic_quest.difficulty.value.title()}\n"
            f"[yellow]Steps:[/yellow] {len(dynamic_quest.steps)}\n\n"
            "[yellow]Quest Steps:[/yellow]\n"
            + "\n".join(f"• {step.description}" for step in dynamic_quest.steps),
            title="Generated Quest",
            border_style="magenta",
        )
        console.print(dynamic_panel)

    # === Summary ===
    console.print("\n" + "=" * 50)
    console.print("🎯 [bold green]Demo Complete![/bold green]")
    console.print("=" * 50)

    summary_panel = Panel(
        "[bold]Quest System Features Demonstrated:[/bold]\n\n"
        "[green]✅ Core Functionality:[/green]\n"
        "• Quest model creation and validation\n"
        "• Prerequisite checking system\n"
        "• Quest lifecycle management\n"
        "• Progress tracking and updates\n"
        "• Step completion detection\n"
        "• Reward distribution system\n"
        "• Dynamic quest generation\n\n"
        "[yellow]📋 System Capabilities:[/yellow]\n"
        "• Type-safe quest operations\n"
        "• Async/await architecture\n"
        "• Rich data validation\n"
        "• Flexible quest templates\n"
        "• Integration-ready design",
        title="Quest System Demo Results",
        border_style="bright_green",
    )
    console.print(summary_panel)


if __name__ == "__main__":
    asyncio.run(create_simple_demo())
