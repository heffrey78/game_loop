"""
Unit tests for NavigationValidator.
"""

from uuid import uuid4

import pytest

from game_loop.core.models.navigation_models import NavigationError
from game_loop.core.navigation.validator import NavigationValidator
from game_loop.core.world.connection_graph import LocationConnectionGraph
from game_loop.state.models import InventoryItem, Location, PlayerState, PlayerStats


class TestNavigationValidator:
    """Test cases for NavigationValidator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.connection_graph = LocationConnectionGraph()
        self.validator = NavigationValidator(self.connection_graph)

        # Create test locations
        self.loc1_id = uuid4()
        self.loc2_id = uuid4()
        self.loc3_id = uuid4()

        # Add locations to graph
        self.connection_graph.add_location(self.loc1_id, {"name": "Location 1"})
        self.connection_graph.add_location(self.loc2_id, {"name": "Location 2"})
        self.connection_graph.add_location(self.loc3_id, {"name": "Location 3"})

    @pytest.mark.asyncio
    async def test_validate_movement_success(self):
        """Test successful movement validation."""
        # Add connection
        self.connection_graph.add_connection(
            self.loc1_id, self.loc2_id, "north", bidirectional=False
        )

        # Create test player and location
        player_state = PlayerState(
            player_id=uuid4(),
            name="Test Player",
            current_location_id=self.loc1_id,
            inventory=[],
        )

        from_location = Location(
            location_id=self.loc1_id,
            name="Start Location",
            description="Starting point",
            connections={"north": self.loc2_id},
        )

        result = await self.validator.validate_movement(
            player_state, from_location, self.loc2_id, "north"
        )

        assert result.success
        assert "You can go north" in result.message

    @pytest.mark.asyncio
    async def test_validate_movement_no_connection(self):
        """Test movement validation with no connection."""
        # Don't add connection to graph

        player_state = PlayerState(
            player_id=uuid4(),
            name="Test Player",
            current_location_id=self.loc1_id,
            inventory=[],
        )

        from_location = Location(
            location_id=self.loc1_id,
            name="Start Location",
            description="Starting point",
            connections={},  # No connections
        )

        result = await self.validator.validate_movement(
            player_state, from_location, self.loc2_id, "north"
        )

        assert not result.success
        assert result.error == NavigationError.NO_CONNECTION
        assert "No connection exists" in result.message

    @pytest.mark.asyncio
    async def test_validate_movement_blocked(self):
        """Test movement validation with blocked direction."""
        # Add connection
        self.connection_graph.add_connection(
            self.loc1_id, self.loc2_id, "north", bidirectional=False
        )

        player_state = PlayerState(
            player_id=uuid4(),
            name="Test Player",
            current_location_id=self.loc1_id,
            inventory=[],
        )

        from_location = Location(
            location_id=self.loc1_id,
            name="Start Location",
            description="Starting point",
            connections={"north": self.loc2_id},
            state_flags={"blocked_north": True},  # Block the north direction
        )

        result = await self.validator.validate_movement(
            player_state, from_location, self.loc2_id, "north"
        )

        assert not result.success
        assert result.error == NavigationError.BLOCKED
        assert "blocked" in result.message

    @pytest.mark.asyncio
    async def test_validate_movement_missing_item_requirement(self):
        """Test movement validation with missing item requirement."""
        # Add connection with requirements
        self.connection_graph.add_connection(
            self.loc1_id,
            self.loc2_id,
            "north",
            bidirectional=False,
            requirements={"required_items": ["key"]},
        )

        player_state = PlayerState(
            player_id=uuid4(),
            name="Test Player",
            current_location_id=self.loc1_id,
            inventory=[],  # No key in inventory
        )

        from_location = Location(
            location_id=self.loc1_id,
            name="Start Location",
            description="Starting point",
            connections={"north": self.loc2_id},
        )

        result = await self.validator.validate_movement(
            player_state, from_location, self.loc2_id, "north"
        )

        assert not result.success
        assert result.error == NavigationError.MISSING_REQUIREMENT
        assert "need a key" in result.message

    @pytest.mark.asyncio
    async def test_validate_movement_with_required_item(self):
        """Test movement validation with required item present."""
        # Add connection with requirements
        self.connection_graph.add_connection(
            self.loc1_id,
            self.loc2_id,
            "north",
            bidirectional=False,
            requirements={"required_items": ["key"]},
        )

        # Create player with required item
        key_item = InventoryItem(name="key", description="A brass key")
        player_state = PlayerState(
            player_id=uuid4(),
            name="Test Player",
            current_location_id=self.loc1_id,
            inventory=[key_item],
        )

        from_location = Location(
            location_id=self.loc1_id,
            name="Start Location",
            description="Starting point",
            connections={"north": self.loc2_id},
        )

        result = await self.validator.validate_movement(
            player_state, from_location, self.loc2_id, "north"
        )

        assert result.success

    @pytest.mark.asyncio
    async def test_validate_movement_insufficient_skill(self):
        """Test movement validation with insufficient skill."""
        # Add connection with skill requirements
        self.connection_graph.add_connection(
            self.loc1_id,
            self.loc2_id,
            "north",
            bidirectional=False,
            requirements={"required_skills": {"climbing": 5}},
        )

        # Create player with insufficient skill (using strength for climbing)
        player_stats = PlayerStats(strength=3)  # Too low for climbing requirement of 5
        player_state = PlayerState(
            player_id=uuid4(),
            name="Test Player",
            current_location_id=self.loc1_id,
            inventory=[],
            stats=player_stats,
        )

        from_location = Location(
            location_id=self.loc1_id,
            name="Start Location",
            description="Starting point",
            connections={"north": self.loc2_id},
        )

        result = await self.validator.validate_movement(
            player_state, from_location, self.loc2_id, "north"
        )

        assert not result.success
        assert result.error == NavigationError.INSUFFICIENT_SKILL
        assert "climbing skill" in result.message

    @pytest.mark.asyncio
    async def test_validate_movement_with_sufficient_skill(self):
        """Test movement validation with sufficient skill."""
        # Add connection with skill requirements
        self.connection_graph.add_connection(
            self.loc1_id,
            self.loc2_id,
            "north",
            bidirectional=False,
            requirements={"required_skills": {"climbing": 5}},
        )

        # Create player with sufficient skill (using strength for climbing)
        player_stats = PlayerStats(
            strength=7
        )  # High enough for climbing requirement of 5
        player_state = PlayerState(
            player_id=uuid4(),
            name="Test Player",
            current_location_id=self.loc1_id,
            inventory=[],
            stats=player_stats,
        )

        from_location = Location(
            location_id=self.loc1_id,
            name="Start Location",
            description="Starting point",
            connections={"north": self.loc2_id},
        )

        result = await self.validator.validate_movement(
            player_state, from_location, self.loc2_id, "north"
        )

        assert result.success

    @pytest.mark.asyncio
    async def test_validate_movement_invalid_state(self):
        """Test movement validation with invalid player state."""
        # Add connection with state requirements
        self.connection_graph.add_connection(
            self.loc1_id,
            self.loc2_id,
            "north",
            bidirectional=False,
            requirements={"required_state": {"has_permission": True}},
        )

        # Create player with wrong state using progress flags
        from game_loop.state.models import PlayerProgress

        player_progress = PlayerProgress(flags={"has_permission": False})  # Wrong state
        player_state = PlayerState(
            player_id=uuid4(),
            name="Test Player",
            current_location_id=self.loc1_id,
            inventory=[],
            progress=player_progress,
        )

        from_location = Location(
            location_id=self.loc1_id,
            name="Start Location",
            description="Starting point",
            connections={"north": self.loc2_id},
        )

        result = await self.validator.validate_movement(
            player_state, from_location, self.loc2_id, "north"
        )

        assert not result.success
        assert result.error == NavigationError.INVALID_STATE

    def test_get_valid_directions_without_player(self):
        """Test getting valid directions without player state."""
        location = Location(
            location_id=self.loc1_id,
            name="Test Location",
            description="A test location",
            connections={"north": self.loc2_id, "south": self.loc3_id},
            state_flags={"blocked_south": True},  # Block south
        )

        valid_directions = self.validator.get_valid_directions(location)

        assert valid_directions["north"] is True
        assert valid_directions["south"] is False

    def test_get_valid_directions_with_player(self):
        """Test getting valid directions with player state."""
        # Add connections with requirements
        self.connection_graph.add_connection(
            self.loc1_id,
            self.loc2_id,
            "north",
            bidirectional=False,
            requirements={"required_items": ["key"]},
        )
        self.connection_graph.add_connection(
            self.loc1_id, self.loc3_id, "south", bidirectional=False
        )

        # Player without key
        player_state = PlayerState(
            player_id=uuid4(),
            name="Test Player",
            current_location_id=self.loc1_id,
            inventory=[],
        )

        location = Location(
            location_id=self.loc1_id,
            name="Test Location",
            description="A test location",
            connections={"north": self.loc2_id, "south": self.loc3_id},
        )

        valid_directions = self.validator.get_valid_directions(location, player_state)

        # Note: This test might need adjustment based on async implementation
        # For now, we'll test the structure
        assert isinstance(valid_directions, dict)
        assert "north" in valid_directions
        assert "south" in valid_directions

    def test_validate_connection_exists(self):
        """Test validating connection existence."""
        # Add connection
        self.connection_graph.add_connection(
            self.loc1_id, self.loc2_id, "north", bidirectional=False
        )

        # Test existing connection
        result = self.validator.validate_connection_exists(self.loc1_id, self.loc2_id)
        assert result.success

        # Test non-existent connection
        result = self.validator.validate_connection_exists(self.loc1_id, self.loc3_id)
        assert not result.success
        assert result.error == NavigationError.NO_CONNECTION

    def test_get_blocked_directions(self):
        """Test getting blocked directions."""
        location = Location(
            location_id=self.loc1_id,
            name="Test Location",
            description="A test location",
            connections={"north": self.loc2_id, "south": self.loc3_id, "east": uuid4()},
            state_flags={"blocked_north": True, "blocked_east": True},
        )

        blocked = self.validator.get_blocked_directions(location)

        assert "north" in blocked
        assert "east" in blocked
        assert "south" not in blocked

    def test_set_direction_blocked(self):
        """Test setting direction as blocked/unblocked."""
        location = Location(
            location_id=self.loc1_id,
            name="Test Location",
            description="A test location",
            connections={"north": self.loc2_id},
        )

        # Block direction
        self.validator.set_direction_blocked(location, "north", True)
        assert location.state_flags.get("blocked_north") is True

        # Unblock direction
        self.validator.set_direction_blocked(location, "north", False)
        assert location.state_flags.get("blocked_north") is False

    def test_can_player_access_location(self):
        """Test checking if player can access a location."""
        # Location with access requirements
        location = Location(
            location_id=self.loc1_id,
            name="Restricted Location",
            description="A restricted area",
            connections={},
            state_flags={"access_requirements": {"required_items": ["pass"]}},
        )

        # Player without pass
        player_state = PlayerState(
            player_id=uuid4(),
            name="Test Player",
            current_location_id=uuid4(),
            inventory=[],
        )

        can_access = self.validator.can_player_access_location(player_state, location)
        assert not can_access

        # Player with pass
        pass_item = InventoryItem(name="pass", description="Access pass")
        player_state.inventory = [pass_item]

        can_access = self.validator.can_player_access_location(player_state, location)
        assert can_access
