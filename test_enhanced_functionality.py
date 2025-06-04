#!/usr/bin/env python3
"""
Test script to verify the enhanced InputProcessor functionality.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from rich.console import Console

from game_loop.core.input_processor import CommandType, InputProcessor


async def test_enhanced_input_processor():
    """Test the enhanced InputProcessor with context integration."""
    print("🔍 Testing Enhanced InputProcessor Integration...")

    # Create mock GameStateManager
    mock_game_state_manager = AsyncMock()

    # Mock location details
    mock_location = MagicMock()
    mock_location.location_id = "room_001"
    mock_location.name = "Testing Chamber"
    mock_location.description = "A chamber for testing enhanced input processing"
    mock_location.connections = {"north": "room_002", "south": "room_000"}
    mock_location.objects = {}
    mock_location.npcs = {}

    mock_game_state_manager.get_current_location_details.return_value = mock_location

    # Mock player state
    mock_player_state = MagicMock()
    mock_player_state.name = "Test Player"
    mock_player_state.current_location_id = "room_001"
    mock_player_state.inventory = {}

    mock_game_state_manager.get_current_state.return_value = (mock_player_state, None)

    # Create InputProcessor with enhanced capabilities
    console = Console()
    processor = InputProcessor(
        console=console, game_state_manager=mock_game_state_manager
    )

    print("✅ InputProcessor created with GameStateManager integration")

    # Test 1: Basic synchronous processing (backward compatibility)
    print("\n📝 Testing backward compatibility...")
    result = processor.process_input("look around")
    assert result.command_type == CommandType.LOOK
    assert result.action == "look"
    print(f"✅ Synchronous processing: {result.action} command recognized")

    # Test 2: Enhanced async processing with context
    print("\n🚀 Testing enhanced async processing...")
    result = await processor.process("take sword")
    assert result.command_type == CommandType.TAKE
    assert result.action == "take"
    assert result.subject == "sword"
    print(f"✅ Async processing: {result.action} {result.subject}")

    # Test 3: Context retrieval
    print("\n🌍 Testing context retrieval...")
    context = await processor._get_current_context()
    assert "current_location" in context
    assert context["current_location"]["name"] == "Testing Chamber"
    assert "player" in context
    assert context["player"]["name"] == "Test Player"
    print("✅ Context retrieval working correctly")
    print(f"   Location: {context['current_location']['name']}")
    print(f"   Player: {context['player']['name']}")

    # Test 4: Custom context processing
    print("\n⚙️ Testing custom context processing...")
    custom_context = {
        "current_location": {
            "name": "Custom Room",
            "objects": ["magic sword", "ancient tome"],
        },
        "player": {"name": "Custom Player"},
    }
    result = await processor.process("examine tome", custom_context)
    assert result.command_type == CommandType.EXAMINE
    assert result.subject == "tome"
    print(f"✅ Custom context processing: {result.action} {result.subject}")

    # Test 5: Process with context method
    print("\n🎯 Testing _process_with_context...")
    result = await processor._process_with_context("go north", context)
    assert result.command_type == CommandType.MOVEMENT
    assert result.subject == "north"
    print(f"✅ Context-aware processing: {result.action} {result.subject}")

    print("\n🎉 All enhanced functionality tests passed!")
    print("📊 Summary:")
    print("   ✅ Backward compatibility maintained")
    print("   ✅ Enhanced async processing working")
    print("   ✅ GameStateManager integration active")
    print("   ✅ Context retrieval functional")
    print("   ✅ Custom context processing working")
    print("   ✅ Enhanced methods operational")


if __name__ == "__main__":
    asyncio.run(test_enhanced_input_processor())
