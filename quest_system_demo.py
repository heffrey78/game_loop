#!/usr/bin/env python3
"""
Quest Interaction System Demo

This demo showcases the Quest system implementation with realistic scenarios.
"""

import asyncio
import time
import uuid

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.game_loop.core.quest.quest_processor import QuestInteractionProcessor

# Import our quest system components
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


class MockQuestRepository:
    """Mock repository for demo purposes."""

    def __init__(self):
        self.quests = {}
        self.progress = {}
        self.interactions = []

    async def get_quest(self, quest_id: str):
        return self.quests.get(quest_id)

    async def create_quest(self, quest: Quest) -> bool:
        self.quests[quest.quest_id] = quest
        return True

    async def get_player_progress(self, player_id: str, quest_id: str):
        return self.progress.get(f"{player_id}:{quest_id}")

    async def update_progress(self, progress: QuestProgress) -> bool:
        key = f"{progress.player_id}:{progress.quest_id}"
        self.progress[key] = progress
        return True

    async def get_available_quests(self, player_id: str, location_id: str = None):
        available = []
        for quest in self.quests.values():
            progress_key = f"{player_id}:{quest.quest_id}"
            if progress_key not in self.progress:
                available.append(quest)
        return available

    async def get_active_quests(self, player_id: str):
        active = []
        for key, progress in self.progress.items():
            if (
                progress.player_id == player_id
                and progress.status == QuestStatus.ACTIVE
            ):
                active.append(progress)
        return active

    async def log_interaction(
        self,
        quest_id: str,
        player_id: str,
        interaction_type: str,
        interaction_data: dict,
    ):
        self.interactions.append(
            {
                "quest_id": quest_id,
                "player_id": player_id,
                "type": interaction_type,
                "data": interaction_data,
                "timestamp": time.time(),
            }
        )
        return True


def create_sample_quests():
    """Create sample quests for the demo."""

    # 1. Simple delivery quest
    delivery_quest = Quest(
        quest_id="delivery_001",
        title="The Merchant's Package",
        description="Help Marcus the merchant by delivering a package to the castle",
        category=QuestCategory.DELIVERY,
        difficulty=QuestDifficulty.EASY,
        steps=[
            QuestStep(
                step_id="pickup_package",
                description="Pick up the package from Marcus",
                requirements={"location": "marketplace", "npc": "marcus"},
                completion_conditions=["action_type:take"],
            ),
            QuestStep(
                step_id="deliver_package",
                description="Deliver the package to Captain Elena at the castle",
                requirements={"location": "castle", "npc": "elena"},
                completion_conditions=["action_type:give"],
            ),
        ],
        rewards={"experience": 100, "gold": 50},
        time_limit=None,
        repeatable=False,
    )

    # 2. Exploration quest
    exploration_quest = Quest(
        quest_id="explore_001",
        title="Discover the Ancient Ruins",
        description="Explore the mysterious ruins discovered north of town",
        category=QuestCategory.EXPLORATION,
        difficulty=QuestDifficulty.MEDIUM,
        steps=[
            QuestStep(
                step_id="find_ruins",
                description="Locate the ancient ruins",
                requirements={"location": "northern_forest"},
                completion_conditions=["location:ancient_ruins"],
            ),
            QuestStep(
                step_id="investigate_altar",
                description="Examine the mysterious altar",
                requirements={"target_object": "ancient_altar"},
                completion_conditions=["action_type:examine"],
            ),
        ],
        rewards={"experience": 200, "items": ["ancient_map"]},
        time_limit=None,
        repeatable=False,
    )

    # 3. Collection quest with prerequisites
    collection_quest = Quest(
        quest_id="collect_001",
        title="Herbalist's Request",
        description="Gather rare herbs for the village herbalist",
        category=QuestCategory.COLLECTION,
        difficulty=QuestDifficulty.EASY,
        steps=[
            QuestStep(
                step_id="collect_moonflower",
                description="Find 3 moonflowers in the forest",
                requirements={"items": ["moonflower"], "quantity": 3},
                completion_conditions=["item_obtained:moonflower"],
            ),
            QuestStep(
                step_id="collect_crystalroot",
                description="Harvest 2 crystal roots from the cave",
                requirements={"items": ["crystalroot"], "quantity": 2},
                completion_conditions=["item_obtained:crystalroot"],
            ),
        ],
        prerequisites=["delivery_001"],  # Must complete delivery quest first
        rewards={"experience": 150, "gold": 75, "items": ["healing_potion"]},
        repeatable=True,
    )

    return [delivery_quest, exploration_quest, collection_quest]


async def demo_quest_discovery(
    quest_processor: QuestInteractionProcessor, player_id: str
):
    """Demo quest discovery functionality."""
    console.print("\n" + "=" * 60)
    console.print("🔍 [bold yellow]QUEST DISCOVERY DEMO[/bold yellow]")
    console.print("=" * 60)

    # Discover available quests
    console.print("\n[cyan]Player arrives at the marketplace...[/cyan]")

    result = await quest_processor.process_quest_interaction(
        interaction_type="discover",
        player_id=player_id,
        quest_context={"location_id": "marketplace"},
        game_state={"player_id": player_id, "location_id": "marketplace"},
    )

    if result.success:
        console.print(f"\n✅ [green]{result.message}[/green]")

        # Display available quests in a table
        available_quests = await quest_processor.discover_available_quests(
            player_id, "marketplace", {"game_state": {"location_id": "marketplace"}}
        )

        if available_quests:
            table = Table(title="Available Quests")
            table.add_column("Quest", style="cyan", no_wrap=True)
            table.add_column("Difficulty", style="yellow")
            table.add_column("Category", style="green")
            table.add_column("Rewards", style="magenta")

            for quest in available_quests:
                rewards_str = ", ".join([f"{k}: {v}" for k, v in quest.rewards.items()])
                table.add_row(
                    quest.title,
                    quest.difficulty.value.title(),
                    quest.category.value.title(),
                    rewards_str,
                )

            console.print(table)
    else:
        console.print(f"❌ [red]{result.message}[/red]")


async def demo_quest_acceptance(
    quest_processor: QuestInteractionProcessor, player_id: str
):
    """Demo quest acceptance."""
    console.print("\n" + "=" * 60)
    console.print("✋ [bold yellow]QUEST ACCEPTANCE DEMO[/bold yellow]")
    console.print("=" * 60)

    # Accept the delivery quest
    console.print(
        "\n[cyan]Player decides to accept 'The Merchant's Package' quest...[/cyan]"
    )

    result = await quest_processor.process_quest_interaction(
        interaction_type="accept",
        player_id=player_id,
        quest_context={"quest_id": "delivery_001"},
        game_state={"player_id": player_id, "location_id": "marketplace"},
    )

    if result.success:
        console.print(f"\n✅ [green]{result.message}[/green]")

        # Show quest details
        quest_panel = Panel(
            "[bold]The Merchant's Package[/bold]\n\n"
            "Help Marcus the merchant by delivering a package to the castle\n\n"
            "[yellow]Current Step:[/yellow] Pick up the package from Marcus\n"
            "[yellow]Rewards:[/yellow] 100 XP, 50 Gold",
            title="Quest Accepted",
            border_style="green",
        )
        console.print(quest_panel)
    else:
        console.print(f"❌ [red]{result.message}[/red]")


async def demo_quest_progression(
    quest_processor: QuestInteractionProcessor, player_id: str
):
    """Demo quest progression through actions."""
    console.print("\n" + "=" * 60)
    console.print("⚡ [bold yellow]QUEST PROGRESSION DEMO[/bold yellow]")
    console.print("=" * 60)

    # Simulate picking up the package
    console.print("\n[cyan]Player talks to Marcus and picks up the package...[/cyan]")

    # Create an action result for taking the package
    action_result = ActionResult(
        success=True,
        feedback_message="You carefully take the sealed package from Marcus",
        command="take",
        processed_input={"target": "package", "npc": "marcus"},
        inventory_changes=[{"action": "add", "item": "package", "quantity": 1}],
    )

    # Update quest progress
    updates = await quest_processor.update_quest_progress(
        player_id, "delivery_001", action_result, {"location_id": "marketplace"}
    )

    if updates:
        console.print("✅ [green]Quest step completed![/green]")
        console.print("📦 [blue]Package acquired![/blue]")

        # Show progress update
        progress_panel = Panel(
            "[yellow]Step 1 Complete:[/yellow] ✅ Pick up the package from Marcus\n"
            "[yellow]Next Step:[/yellow] 🏰 Deliver the package to Captain Elena at the castle\n\n"
            "[dim]Progress: 1/2 steps completed[/dim]",
            title="Quest Progress Updated",
            border_style="yellow",
        )
        console.print(progress_panel)

    # Simulate traveling to castle and delivering
    console.print(
        "\n[cyan]Player travels to the castle and finds Captain Elena...[/cyan]"
    )

    delivery_action = ActionResult(
        success=True,
        feedback_message="Captain Elena accepts the package with gratitude",
        command="give",
        processed_input={"target": "package", "npc": "elena"},
        inventory_changes=[{"action": "remove", "item": "package", "quantity": 1}],
        location_change=True,
        new_location_id=uuid.uuid4(),
    )

    # Complete the quest
    updates = await quest_processor.update_quest_progress(
        player_id, "delivery_001", delivery_action, {"location_id": "castle"}
    )

    if updates:
        console.print("🎉 [bold green]Quest Completed![/bold green]")

        completion_panel = Panel(
            "[bold green]The Merchant's Package - COMPLETED![/bold green]\n\n"
            "✅ Package delivered successfully to Captain Elena\n\n"
            "[yellow]Rewards Earned:[/yellow]\n"
            "• 💰 50 Gold\n"
            "• ⭐ 100 Experience Points\n\n"
            "[dim]This quest unlocks new opportunities...[/dim]",
            title="Quest Complete!",
            border_style="bright_green",
        )
        console.print(completion_panel)


async def demo_quest_queries(
    quest_processor: QuestInteractionProcessor, player_id: str
):
    """Demo quest query functionality."""
    console.print("\n" + "=" * 60)
    console.print("❓ [bold yellow]QUEST QUERY DEMO[/bold yellow]")
    console.print("=" * 60)

    console.print("\n[cyan]Player checks their active quests...[/cyan]")

    # Query all active quests
    result = await quest_processor.process_quest_interaction(
        interaction_type="query",
        player_id=player_id,
        quest_context={},
        game_state={"player_id": player_id},
    )

    if result.success:
        console.print(f"\n✅ [green]{result.message}[/green]")

        # Show quest log
        quest_log = Panel(
            "[bold]Quest Log[/bold]\n\n"
            "[green]✅ Completed Quests:[/green]\n"
            "• The Merchant's Package (Completed)\n\n"
            "[yellow]📋 Available Quests:[/yellow]\n"
            "• Discover the Ancient Ruins (Medium Difficulty)\n"
            "• Herbalist's Request (Easy - Requires: Merchant's Package)",
            title="Player Quest Log",
            border_style="blue",
        )
        console.print(quest_log)


async def demo_prerequisite_system(
    quest_processor: QuestInteractionProcessor, player_id: str
):
    """Demo quest prerequisite system."""
    console.print("\n" + "=" * 60)
    console.print("🔗 [bold yellow]QUEST PREREQUISITE DEMO[/bold yellow]")
    console.print("=" * 60)

    console.print(
        "\n[cyan]Player tries to accept the Herbalist's Request quest...[/cyan]"
    )

    # Try to accept collection quest (should work now since delivery is complete)
    result = await quest_processor.process_quest_interaction(
        interaction_type="accept",
        player_id=player_id,
        quest_context={"quest_id": "collect_001"},
        game_state={"player_id": player_id, "location_id": "village"},
    )

    if result.success:
        console.print(f"\n✅ [green]{result.message}[/green]")

        prereq_panel = Panel(
            "[bold]Herbalist's Request[/bold]\n\n"
            "Gather rare herbs for the village herbalist\n\n"
            "[green]✅ Prerequisites Met:[/green]\n"
            "• The Merchant's Package (Completed)\n\n"
            "[yellow]Quest Steps:[/yellow]\n"
            "1. Find 3 moonflowers in the forest\n"
            "2. Harvest 2 crystal roots from the cave\n\n"
            "[yellow]Rewards:[/yellow] 150 XP, 75 Gold, Healing Potion",
            title="Quest Accepted - Prerequisites Satisfied",
            border_style="green",
        )
        console.print(prereq_panel)
    else:
        console.print(f"❌ [red]{result.message}[/red]")


async def main():
    """Run the quest system demo."""
    console.print(
        Panel.fit(
            "[bold blue]Quest Interaction System Demo[/bold blue]\n\n"
            "This demo showcases the quest system features:\n"
            "• Quest Discovery & Availability\n"
            "• Quest Acceptance & Validation\n"
            "• Dynamic Quest Progression\n"
            "• Prerequisite System\n"
            "• Quest Completion & Rewards",
            border_style="bright_blue",
        )
    )

    # Set up the quest system
    mock_repo = MockQuestRepository()
    quest_manager = QuestManager(mock_repo)
    quest_processor = QuestInteractionProcessor(quest_manager)

    # Create and store sample quests
    sample_quests = create_sample_quests()
    for quest in sample_quests:
        await mock_repo.create_quest(quest)

    player_id = "demo_player"

    try:
        # Run the demo sequence
        await demo_quest_discovery(quest_processor, player_id)
        await asyncio.sleep(1)

        await demo_quest_acceptance(quest_processor, player_id)
        await asyncio.sleep(1)

        await demo_quest_progression(quest_processor, player_id)
        await asyncio.sleep(1)

        await demo_quest_queries(quest_processor, player_id)
        await asyncio.sleep(1)

        await demo_prerequisite_system(quest_processor, player_id)

        # Final summary
        console.print("\n" + "=" * 60)
        console.print("🎯 [bold green]DEMO COMPLETE![/bold green]")
        console.print("=" * 60)

        summary_panel = Panel(
            "[bold]Quest System Demo Summary[/bold]\n\n"
            "[green]✅ Successfully demonstrated:[/green]\n"
            "• Quest discovery and filtering\n"
            "• Quest acceptance with validation\n"
            "• Real-time quest progression tracking\n"
            "• Prerequisite requirement system\n"
            "• Reward distribution\n"
            "• Quest completion workflow\n\n"
            "[yellow]Key Features Shown:[/yellow]\n"
            "• Event-driven progression\n"
            "• Database persistence simulation\n"
            "• Rich user interface integration\n"
            "• Type-safe quest operations",
            title="Demo Results",
            border_style="bright_green",
        )
        console.print(summary_panel)

    except Exception as e:
        console.print(f"\n❌ [red]Demo error: {str(e)}[/red]")
        raise


if __name__ == "__main__":
    asyncio.run(main())
