"""
Unit tests for the NLP processor.
"""

import json
from unittest.mock import MagicMock

import pytest

from game_loop.core.input_processor import CommandType
from game_loop.llm.nlp_processor import NLPProcessor


@pytest.fixture
def mock_config_manager():
    """Create a mock config manager for testing."""
    mock_config = MagicMock()
    mock_config.format_prompt.return_value = "Test prompt"
    mock_config.llm_config = MagicMock()
    mock_config.llm_config.base_url = "http://localhost:11434"
    mock_config.llm_config.default_model = "mistral"
    mock_config.ollama_config = MagicMock()
    mock_config.ollama_config.completion_params = {"temperature": 0.7, "top_p": 0.9}
    mock_config.ollama_config.system_prompt = "You are a helpful assistant"
    return mock_config


@pytest.fixture
def mock_ollama_client():
    """Create a mock Ollama client for testing."""
    mock_client = MagicMock()
    # Mock the generate method to return a pre-defined response
    mock_client.generate.return_value = {
        "response": json.dumps(
            {
                "command_type": "LOOK",
                "action": "look",
                "subject": "room",
                "target": None,
                "confidence": 0.9,
            }
        )
    }
    return mock_client


@pytest.fixture
def nlp_processor(mock_config_manager, mock_ollama_client):
    """Create an NLP processor with mock dependencies for testing."""
    return NLPProcessor(
        config_manager=mock_config_manager, ollama_client=mock_ollama_client
    )


class TestNLPProcessor:
    """Tests for the NLPProcessor class."""

    @pytest.mark.asyncio
    async def test_process_input(self, nlp_processor):
        """Test processing input with the NLP processor."""
        result = await nlp_processor.process_input("look around the room")

        assert result.command_type == CommandType.LOOK
        assert result.action == "look"
        assert result.subject == "room"
        assert result.target is None
        assert result.parameters.get("confidence") == 0.9

    @pytest.mark.asyncio
    async def test_extract_intent(self, nlp_processor, mock_ollama_client):
        """Test intent extraction."""
        result = await nlp_processor.extract_intent("look around the room")

        # Verify the LLM was called correctly
        mock_ollama_client.generate.assert_called_once()

        # Verify the result is parsed correctly
        assert result.get("command_type") == "LOOK"
        assert result.get("action") == "look"
        assert result.get("subject") == "room"
        assert result.get("confidence") == 0.9

    @pytest.mark.asyncio
    async def test_extract_intent_error_handling(
        self, nlp_processor, mock_ollama_client
    ):
        """Test error handling during intent extraction."""
        # Make the mock client return an invalid JSON response
        mock_ollama_client.generate.return_value = {"response": "Not a valid JSON"}

        # The method should handle the error and return an empty dict
        result = await nlp_processor.extract_intent("look around the room")
        assert result == {}

    @pytest.mark.asyncio
    async def test_disambiguate_input(self, nlp_processor, mock_ollama_client):
        """Test disambiguating between multiple possible interpretations."""
        # Set up mock response
        mock_ollama_client.generate.return_value = {
            "response": json.dumps(
                {
                    "selected_interpretation": 1,
                    "confidence": 0.8,
                    "explanation": "The second interpretation makes more sense.",
                }
            )
        }

        interpretations = [
            {
                "command_type": "LOOK",
                "action": "look",
                "subject": "room",
                "confidence": 0.6,
            },
            {
                "command_type": "EXAMINE",
                "action": "examine",
                "subject": "room",
                "confidence": 0.5,
            },
        ]

        result = await nlp_processor.disambiguate_input(
            "check out the room", interpretations, "You are in a dimly lit room."
        )

        assert result.get("command_type") == "EXAMINE"
        assert result.get("action") == "examine"
        assert result.get("subject") == "room"
        assert result.get("confidence") == 0.8

    def test_normalize_input(self, nlp_processor):
        """Test input normalization."""
        result = nlp_processor._normalize_input("  LOOK AROUND  ")
        assert result == "look around"

    def test_format_context(self, nlp_processor):
        """Test game context formatting."""
        game_context = {
            "location": {
                "name": "Dusty Library",
                "description": "A room filled with old books.",
            },
            "visible_objects": [
                {"name": "book", "description": "An ancient tome."},
                {"name": "desk", "description": "A wooden desk."},
            ],
            "npcs": [{"name": "Librarian", "description": "An elderly man."}],
            "inventory": [{"name": "key", "description": "A brass key."}],
        }

        result = nlp_processor._format_context(game_context)

        assert "You are in: Dusty Library" in result
        assert "Location description: A room filled with old books." in result
        assert "You can see:" in result
        assert "- book" in result
        assert "- desk" in result
        assert "Characters present:" in result
        assert "- Librarian" in result
        assert "You are carrying:" in result
        assert "- key" in result

    @pytest.mark.asyncio
    async def test_generate_semantic_query(self, nlp_processor):
        """Test generating a semantic search query from intent data."""
        intent_data = {"action": "use", "subject": "key", "target": "door"}

        query = await nlp_processor.generate_semantic_query(intent_data)
        assert query == "use key door"
