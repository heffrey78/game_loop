"""Tests for query system models."""

import pytest
from game_loop.core.query.query_models import (
    QueryRequest,
    QueryResponse,
    QueryType,
    InformationSource,
    QueryContext,
)


class TestQueryRequest:
    """Test QueryRequest model."""

    def test_create_query_request(self):
        """Test creating a query request."""
        request = QueryRequest.create(
            player_id="test_player",
            query_text="What is the ancient temple?",
            query_type=QueryType.WORLD_INFO,
            context={"location": "town_square"},
        )

        assert request.player_id == "test_player"
        assert request.query_text == "What is the ancient temple?"
        assert request.query_type == QueryType.WORLD_INFO
        assert request.context == {"location": "town_square"}
        assert request.query_id is not None
        assert request.timestamp is not None

    def test_query_request_with_minimal_data(self):
        """Test creating query request with minimal data."""
        request = QueryRequest.create(
            player_id="test_player",
            query_text="help",
            query_type=QueryType.HELP,
        )

        assert request.player_id == "test_player"
        assert request.query_text == "help"
        assert request.query_type == QueryType.HELP
        assert request.context == {}


class TestQueryResponse:
    """Test QueryResponse model."""

    def test_success_response(self):
        """Test creating a successful response."""
        response = QueryResponse.success_response(
            response_text="The ancient temple is a mysterious place...",
            information_type="world_info",
            sources=["ancient_scroll", "village_elder"],
            related_queries=["Who built the temple?", "What's inside the temple?"],
            confidence=0.8,
        )

        assert response.success is True
        assert response.response_text == "The ancient temple is a mysterious place..."
        assert response.information_type == "world_info"
        assert response.sources == ["ancient_scroll", "village_elder"]
        assert response.related_queries == ["Who built the temple?", "What's inside the temple?"]
        assert response.confidence == 0.8
        assert response.errors is None

    def test_error_response(self):
        """Test creating an error response."""
        response = QueryResponse.error_response(
            error_message="Could not find information",
            errors=["No matching entities", "Search failed"],
        )

        assert response.success is False
        assert response.response_text == "Could not find information"
        assert response.information_type == "error"
        assert response.sources == []
        assert response.related_queries == []
        assert response.confidence == 0.0
        assert response.errors == ["No matching entities", "Search failed"]


class TestInformationSource:
    """Test InformationSource model."""

    def test_information_source_creation(self):
        """Test creating an information source."""
        source = InformationSource(
            source_id="temple_001",
            source_type="entity",
            source_name="Ancient Temple",
            content="A mysterious temple built by ancient civilization...",
            relevance_score=0.9,
            metadata={"type": "location", "era": "ancient"},
        )

        assert source.source_id == "temple_001"
        assert source.source_type == "entity"
        assert source.source_name == "Ancient Temple"
        assert source.content == "A mysterious temple built by ancient civilization..."
        assert source.relevance_score == 0.9
        assert source.metadata == {"type": "location", "era": "ancient"}

    def test_information_source_to_dict(self):
        """Test converting information source to dictionary."""
        source = InformationSource(
            source_id="temple_001",
            source_type="entity",
            source_name="Ancient Temple",
            content="A mysterious temple...",
            relevance_score=0.9,
        )

        source_dict = source.to_dict()

        assert source_dict["source_id"] == "temple_001"
        assert source_dict["source_type"] == "entity"
        assert source_dict["source_name"] == "Ancient Temple"
        assert source_dict["content"] == "A mysterious temple..."
        assert source_dict["relevance_score"] == 0.9
        assert source_dict["metadata"] == {}


class TestQueryContext:
    """Test QueryContext model."""

    def test_query_context_creation(self):
        """Test creating query context."""
        context = QueryContext(
            player_id="test_player",
            current_location_id="town_square",
            recent_actions=["look around", "examine statue"],
            active_quests=["find_artifact", "talk_to_mayor"],
            inventory_items=["sword", "potion"],
        )

        assert context.player_id == "test_player"
        assert context.current_location_id == "town_square"
        assert context.recent_actions == ["look around", "examine statue"]
        assert context.active_quests == ["find_artifact", "talk_to_mayor"]
        assert context.inventory_items == ["sword", "potion"]

    def test_query_context_to_dict(self):
        """Test converting query context to dictionary."""
        context = QueryContext(
            player_id="test_player",
            current_location_id="town_square",
            recent_actions=["look around"],
            active_quests=["find_artifact"],
        )

        context_dict = context.to_dict()

        assert context_dict["player_id"] == "test_player"
        assert context_dict["current_location"] == "town_square"
        assert context_dict["recent_actions"] == ["look around"]
        assert context_dict["active_quests"] == ["find_artifact"]
        assert context_dict["inventory_items"] == []
        assert context_dict["conversation_context"] == {}


class TestQueryType:
    """Test QueryType enum."""

    def test_all_query_types_exist(self):
        """Test that all expected query types exist."""
        expected_types = [
            "WORLD_INFO",
            "OBJECT_INFO", 
            "NPC_INFO",
            "LOCATION_INFO",
            "HELP",
            "STATUS",
            "INVENTORY",
            "QUEST_INFO",
        ]

        for type_name in expected_types:
            assert hasattr(QueryType, type_name)

    def test_query_type_values(self):
        """Test query type enum values."""
        assert QueryType.WORLD_INFO.value == "world_info"
        assert QueryType.OBJECT_INFO.value == "object_info"
        assert QueryType.NPC_INFO.value == "npc_info"
        assert QueryType.LOCATION_INFO.value == "location_info"
        assert QueryType.HELP.value == "help"
        assert QueryType.STATUS.value == "status"
        assert QueryType.INVENTORY.value == "inventory"
        assert QueryType.QUEST_INFO.value == "quest_info"