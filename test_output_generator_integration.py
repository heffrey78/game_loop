#!/usr/bin/env python3
"""
Simple integration test for OutputGenerator system.
Tests basic functionality without the full game loop.
"""

from uuid import uuid4

from rich.console import Console

from game_loop.core.output_generator import OutputGenerator
from game_loop.state.models import ActionResult, Location


def test_output_generator_basic() -> None:
    """Test basic OutputGenerator functionality."""
    console = Console()
    output_gen = OutputGenerator(console)

    # Test action result generation
    action_result = ActionResult(
        success=True,
        feedback_message="You picked up the sword.",
        inventory_changes=[{"action": "add", "item": "sword"}],
    )

    print("Testing ActionResult generation...")
    output_gen.generate_response(action_result, {})

    # Test location display
    location = Location(
        location_id=uuid4(),
        name="Forest Clearing",
        description="A peaceful clearing surrounded by tall trees.",
        objects={},
        npcs={},
        connections={"north": uuid4(), "south": uuid4()},
        state_flags={},
    )

    print("\nTesting Location display...")
    output_gen.display_location(location)

    # Test error message
    print("\nTesting Error message...")
    output_gen.display_error("Something went wrong!")

    # Test system message
    print("\nTesting System message...")
    output_gen.display_system_message("Game saved successfully.")

    print("\nOutputGenerator test completed!")


if __name__ == "__main__":
    test_output_generator_basic()
