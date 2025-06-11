"""
Unit tests for CraftingManager.

Tests crafting session management, recipe validation, and success probability calculation.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from src.game_loop.core.crafting.crafting_manager import (
    CraftingManager,
    CraftingRecipe,
    CraftingComplexity
)


class TestCraftingManager:
    """Test cases for CraftingManager functionality."""

    @pytest.fixture
    def crafting_manager(self):
        """Create crafting manager for testing."""
        object_manager = Mock()
        
        inventory_manager = Mock()
        inventory_manager.remove_item = AsyncMock(return_value=(True, {"item_id": "test_item"}))
        inventory_manager.add_item = AsyncMock(return_value=(True, {"item_id": "test_item"}))
        
        skill_manager = Mock()
        skill_manager.get_skills = AsyncMock(return_value={"smithing": 5, "crafting": 3})
        
        physics_engine = Mock()
        
        manager = CraftingManager(object_manager, inventory_manager, skill_manager, physics_engine)
        return manager

    @pytest.fixture
    def basic_recipe(self):
        """Create basic crafting recipe for testing."""
        return CraftingRecipe(
            recipe_id="basic_sword",
            name="Basic Iron Sword",
            description="A simple iron sword",
            required_components={"iron_ingot": 2, "wood_handle": 1},
            optional_components={"leather_wrap": 1},
            required_tools=["hammer", "anvil"],
            required_skills={"smithing": 3},
            crafting_stations=["forge"],
            complexity=CraftingComplexity.SIMPLE,
            base_success_chance=0.8,
            crafting_time=1800.0,
            energy_cost=25.0,
            products={"iron_sword": 1},
            byproducts={"metal_shavings": 1},
            skill_experience={"smithing": 10.0}
        )

    @pytest.mark.asyncio
    async def test_start_crafting_session_success(self, crafting_manager, basic_recipe):
        """Test successful crafting session start."""
        # Register the recipe
        crafting_manager.register_recipe(basic_recipe)
        
        component_sources = {
            "iron_ingot": "player_inventory",
            "wood_handle": "player_inventory"
        }
        
        success, session_data = await crafting_manager.start_crafting_session(
            "player1",
            "basic_sword",
            component_sources
        )
        
        assert success is True
        assert "session_id" in session_data
        assert session_data["recipe_name"] == "Basic Iron Sword"
        assert "estimated_time" in session_data

    @pytest.mark.asyncio
    async def test_start_crafting_session_invalid_recipe(self, crafting_manager):
        """Test crafting session start with invalid recipe."""
        success, session_data = await crafting_manager.start_crafting_session(
            "player1",
            "nonexistent_recipe",
            {}
        )
        
        assert success is False
        assert "not found" in session_data["error"]

    @pytest.mark.asyncio
    async def test_start_crafting_session_missing_components(self, crafting_manager, basic_recipe):
        """Test crafting session start with missing components."""
        crafting_manager.register_recipe(basic_recipe)
        
        # No component sources provided
        success, session_data = await crafting_manager.start_crafting_session(
            "player1",
            "basic_sword",
            {}
        )
        
        assert success is False
        assert "No source specified" in session_data["error"]

    @pytest.mark.asyncio
    async def test_process_crafting_step(self, crafting_manager, basic_recipe):
        """Test processing a crafting step."""
        crafting_manager.register_recipe(basic_recipe)
        
        # Start session first
        component_sources = {"iron_ingot": "inv1", "wood_handle": "inv1"}
        success, session_data = await crafting_manager.start_crafting_session(
            "player1", "basic_sword", component_sources
        )
        
        session_id = session_data["session_id"]
        
        # Process a step
        step_result = await crafting_manager.process_crafting_step(
            session_id,
            {"difficulty": 1.0, "tool_quality": 0.8}
        )
        
        assert "step_number" in step_result
        assert "step_successful" in step_result
        assert "progress" in step_result
        assert isinstance(step_result["step_successful"], bool)

    @pytest.mark.asyncio
    async def test_process_crafting_step_invalid_session(self, crafting_manager):
        """Test processing step with invalid session ID."""
        step_result = await crafting_manager.process_crafting_step(
            "invalid_session",
            {"difficulty": 1.0}
        )
        
        assert "error" in step_result
        assert "not found" in step_result["error"]

    @pytest.mark.asyncio
    async def test_complete_crafting_session_success(self, crafting_manager, basic_recipe):
        """Test successful crafting session completion."""
        crafting_manager.register_recipe(basic_recipe)
        
        # Start session
        component_sources = {"iron_ingot": "inv1", "wood_handle": "inv1"}
        success, session_data = await crafting_manager.start_crafting_session(
            "player1", "basic_sword", component_sources
        )
        
        session_id = session_data["session_id"]
        
        # Manually set session as completed
        session = crafting_manager._active_crafting_sessions[session_id]
        session["status"] = "completed"
        session["progress"] = 1.0
        
        # Complete session
        success, completion_result = await crafting_manager.complete_crafting_session(session_id)
        
        assert isinstance(success, bool)
        assert "session_id" in completion_result
        assert "recipe_name" in completion_result
        assert "success" in completion_result

    @pytest.mark.asyncio
    async def test_complete_crafting_session_not_ready(self, crafting_manager, basic_recipe):
        """Test completing session that's not ready."""
        crafting_manager.register_recipe(basic_recipe)
        
        # Start session
        component_sources = {"iron_ingot": "inv1", "wood_handle": "inv1"}
        success, session_data = await crafting_manager.start_crafting_session(
            "player1", "basic_sword", component_sources
        )
        
        session_id = session_data["session_id"]
        # Don't set status to completed
        
        success, completion_result = await crafting_manager.complete_crafting_session(session_id)
        
        assert success is False
        assert "not ready" in completion_result["error"]

    @pytest.mark.asyncio
    async def test_cancel_crafting_session(self, crafting_manager, basic_recipe):
        """Test canceling a crafting session."""
        crafting_manager.register_recipe(basic_recipe)
        
        # Start session
        component_sources = {"iron_ingot": "inv1", "wood_handle": "inv1"}
        success, session_data = await crafting_manager.start_crafting_session(
            "player1", "basic_sword", component_sources
        )
        
        session_id = session_data["session_id"]
        
        # Cancel session
        cancellation_result = await crafting_manager.cancel_crafting_session(
            session_id,
            recovery_percentage=0.8
        )
        
        assert "session_id" in cancellation_result
        assert "components_recovered" in cancellation_result
        assert "recovery_percentage" in cancellation_result
        assert session_id not in crafting_manager._active_crafting_sessions

    @pytest.mark.asyncio
    async def test_discover_recipe(self, crafting_manager):
        """Test recipe discovery from components."""
        components = ["iron_ingot", "wood_handle"]
        context = {"location": "forge", "tools_available": ["hammer"]}
        
        discovered_recipe = await crafting_manager.discover_recipe(components, context)
        
        # Might be None if no discovery patterns match
        if discovered_recipe:
            assert hasattr(discovered_recipe, 'recipe_id')
            assert hasattr(discovered_recipe, 'name')
            assert discovered_recipe.recipe_id in crafting_manager._recipe_registry

    @pytest.mark.asyncio
    async def test_validate_crafting_requirements_success(self, crafting_manager, basic_recipe):
        """Test successful crafting requirements validation."""
        crafting_manager.register_recipe(basic_recipe)
        
        # Mock crafter with sufficient skills
        crafting_manager._get_crafter_skills = AsyncMock(return_value={"smithing": 5})
        crafting_manager._get_available_tools = AsyncMock(return_value=["hammer", "anvil"])
        crafting_manager._get_available_crafting_stations = AsyncMock(return_value=["forge"])
        crafting_manager._get_crafter_state = AsyncMock(return_value={"energy": 100})
        
        is_valid, errors = await crafting_manager.validate_crafting_requirements(
            "basic_sword",
            "player1",
            {"iron_ingot": 2, "wood_handle": 1}
        )
        
        assert is_valid is True
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_validate_crafting_requirements_insufficient_skill(self, crafting_manager, basic_recipe):
        """Test validation failure due to insufficient skill."""
        crafting_manager.register_recipe(basic_recipe)
        
        # Mock crafter with insufficient skills
        crafting_manager._get_crafter_skills = AsyncMock(return_value={"smithing": 1})  # Need 3
        crafting_manager._get_available_tools = AsyncMock(return_value=["hammer", "anvil"])
        crafting_manager._get_available_crafting_stations = AsyncMock(return_value=["forge"])
        crafting_manager._get_crafter_state = AsyncMock(return_value={"energy": 100})
        
        is_valid, errors = await crafting_manager.validate_crafting_requirements(
            "basic_sword",
            "player1",
            {"iron_ingot": 2, "wood_handle": 1}
        )
        
        assert is_valid is False
        assert any("smithing" in error for error in errors)

    @pytest.mark.asyncio
    async def test_validate_crafting_requirements_missing_tools(self, crafting_manager, basic_recipe):
        """Test validation failure due to missing tools."""
        crafting_manager.register_recipe(basic_recipe)
        
        crafting_manager._get_crafter_skills = AsyncMock(return_value={"smithing": 5})
        crafting_manager._get_available_tools = AsyncMock(return_value=["hammer"])  # Missing anvil
        crafting_manager._get_available_crafting_stations = AsyncMock(return_value=["forge"])
        crafting_manager._get_crafter_state = AsyncMock(return_value={"energy": 100})
        
        is_valid, errors = await crafting_manager.validate_crafting_requirements(
            "basic_sword",
            "player1",
            {"iron_ingot": 2, "wood_handle": 1}
        )
        
        assert is_valid is False
        assert any("anvil" in error for error in errors)

    @pytest.mark.asyncio
    async def test_calculate_crafting_success_probability(self, crafting_manager, basic_recipe):
        """Test crafting success probability calculation."""
        crafting_manager.register_recipe(basic_recipe)
        
        crafter_skills = {"smithing": 5}  # Higher than required (3)
        component_quality = {"iron_ingot": 0.8, "wood_handle": 0.7}
        
        probability = await crafting_manager.calculate_crafting_success_probability(
            "basic_sword",
            crafter_skills,
            component_quality
        )
        
        assert 0.0 <= probability <= 1.0
        # Should be higher than base due to good skills
        assert probability >= basic_recipe.base_success_chance

    @pytest.mark.asyncio
    async def test_calculate_crafting_success_probability_low_skill(self, crafting_manager, basic_recipe):
        """Test success probability with low skills."""
        crafting_manager.register_recipe(basic_recipe)
        
        crafter_skills = {"smithing": 1}  # Lower than required (3)
        component_quality = {"iron_ingot": 0.5, "wood_handle": 0.5}
        
        probability = await crafting_manager.calculate_crafting_success_probability(
            "basic_sword",
            crafter_skills,
            component_quality
        )
        
        # Should be lower than base due to poor skills
        assert probability < basic_recipe.base_success_chance

    @pytest.mark.asyncio
    async def test_get_available_recipes(self, crafting_manager, basic_recipe):
        """Test getting available recipes for a crafter."""
        crafting_manager.register_recipe(basic_recipe)
        
        crafting_manager._get_crafter_skills = AsyncMock(return_value={"smithing": 5})
        crafting_manager._get_available_tools = AsyncMock(return_value=["hammer", "anvil"])
        crafting_manager._get_available_crafting_stations = AsyncMock(return_value=["forge"])
        
        available_components = ["iron_ingot", "wood_handle", "leather_wrap"]
        
        recipes = await crafting_manager.get_available_recipes("player1", available_components)
        
        assert len(recipes) >= 1
        assert any(recipe.recipe_id == "basic_sword" for recipe in recipes)

    @pytest.mark.asyncio
    async def test_get_available_recipes_insufficient_requirements(self, crafting_manager, basic_recipe):
        """Test getting available recipes with insufficient requirements."""
        crafting_manager.register_recipe(basic_recipe)
        
        crafting_manager._get_crafter_skills = AsyncMock(return_value={"smithing": 1})  # Too low
        crafting_manager._get_available_tools = AsyncMock(return_value=[])  # No tools
        crafting_manager._get_available_crafting_stations = AsyncMock(return_value=[])  # No stations
        
        available_components = ["iron_ingot"]  # Missing components
        
        recipes = await crafting_manager.get_available_recipes("player1", available_components)
        
        # Should not include basic_sword due to insufficient requirements
        assert not any(recipe.recipe_id == "basic_sword" for recipe in recipes)

    @pytest.mark.asyncio
    async def test_enhance_crafting_with_modifiers(self, crafting_manager, basic_recipe):
        """Test enhancing crafting session with modifiers."""
        crafting_manager.register_recipe(basic_recipe)
        
        # Start session
        component_sources = {"iron_ingot": "inv1", "wood_handle": "inv1"}
        success, session_data = await crafting_manager.start_crafting_session(
            "player1", "basic_sword", component_sources
        )
        
        session_id = session_data["session_id"]
        
        # Apply modifiers
        modifiers = {
            "skill_bonus": 0.2,
            "tool_quality_bonus": 0.1,
            "environmental_bonus": 0.05
        }
        
        result = await crafting_manager.enhance_crafting_with_modifiers(session_id, modifiers)
        
        assert result["session_id"] == session_id
        assert "modifiers_applied" in result
        assert result["modifiers_applied"] == modifiers

    @pytest.mark.asyncio
    async def test_analyze_component_compatibility(self, crafting_manager):
        """Test component compatibility analysis."""
        components = ["iron_ingot", "steel_ingot", "wood_handle"]
        
        analysis = await crafting_manager.analyze_component_compatibility(components)
        
        assert "components" in analysis
        assert "compatibility_matrix" in analysis
        assert "overall_compatibility" in analysis
        assert "compatibility_rating" in analysis
        assert analysis["components"] == components

    @pytest.mark.asyncio
    async def test_register_recipe(self, crafting_manager, basic_recipe):
        """Test registering a new recipe."""
        crafting_manager.register_recipe(basic_recipe)
        
        assert "basic_sword" in crafting_manager._recipe_registry
        assert crafting_manager._recipe_registry["basic_sword"] == basic_recipe

    @pytest.mark.asyncio
    async def test_register_crafting_station(self, crafting_manager):
        """Test registering a new crafting station."""
        station_capabilities = {
            "success_bonus": 0.15,
            "quality_bonus": 0.2,
            "time_reduction": 0.25
        }
        
        crafting_manager.register_crafting_station("master_forge", station_capabilities)
        
        assert "master_forge" in crafting_manager._crafting_stations
        assert crafting_manager._crafting_stations["master_forge"] == station_capabilities

    @pytest.mark.asyncio
    async def test_crafting_step_calculation(self, crafting_manager):
        """Test crafting step calculation based on complexity."""
        simple_recipe = CraftingRecipe(
            recipe_id="simple_item",
            name="Simple Item",
            description="A simple item",
            required_components={},
            optional_components={},
            required_tools=[],
            required_skills={},
            crafting_stations=[],
            complexity=CraftingComplexity.SIMPLE,
            base_success_chance=0.9,
            crafting_time=300.0,
            energy_cost=10.0,
            products={},
            byproducts={},
            skill_experience={}
        )
        
        master_recipe = CraftingRecipe(
            recipe_id="master_item",
            name="Master Item", 
            description="A master-level item",
            required_components={},
            optional_components={},
            required_tools=[],
            required_skills={},
            crafting_stations=[],
            complexity=CraftingComplexity.MASTER,
            base_success_chance=0.3,
            crafting_time=3600.0,
            energy_cost=50.0,
            products={},
            byproducts={},
            skill_experience={}
        )
        
        simple_steps = crafting_manager._calculate_crafting_steps(simple_recipe)
        master_steps = crafting_manager._calculate_crafting_steps(master_recipe)
        
        assert simple_steps < master_steps
        assert simple_steps >= 1
        assert master_steps >= 1

    @pytest.mark.asyncio
    async def test_session_cleanup_on_completion(self, crafting_manager, basic_recipe):
        """Test that sessions are cleaned up after completion."""
        crafting_manager.register_recipe(basic_recipe)
        
        # Start session
        component_sources = {"iron_ingot": "inv1", "wood_handle": "inv1"}
        success, session_data = await crafting_manager.start_crafting_session(
            "player1", "basic_sword", component_sources
        )
        
        session_id = session_data["session_id"]
        
        # Manually complete session
        session = crafting_manager._active_crafting_sessions[session_id]
        session["status"] = "completed"
        
        await crafting_manager.complete_crafting_session(session_id)
        
        # Session should be cleaned up
        assert session_id not in crafting_manager._active_crafting_sessions

    @pytest.mark.asyncio
    async def test_error_handling_invalid_session_operations(self, crafting_manager):
        """Test error handling for operations on invalid sessions."""
        # Try to process step on nonexistent session
        step_result = await crafting_manager.process_crafting_step("invalid", {})
        assert "error" in step_result
        
        # Try to complete nonexistent session
        success, result = await crafting_manager.complete_crafting_session("invalid")
        assert success is False
        assert "error" in result
        
        # Try to cancel nonexistent session
        cancel_result = await crafting_manager.cancel_crafting_session("invalid")
        assert "error" in cancel_result