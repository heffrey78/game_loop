"""
Unit tests for the NLP processor.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from game_loop.core.input_processor import CommandType
from game_loop.llm.models import CommandTypeStr, Intent
from game_loop.llm.nlp_processor import NLPProcessor


@pytest.fixture
def mock_config_manager() -> MagicMock:
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
def mock_ollama_client() -> MagicMock:
    """Create a mock Ollama client for testing."""
    mock_client = MagicMock()

    # Create a response dict that matches what the updated code expects
    # The response should be a dictionary that can be parsed directly
    mock_result = {
        "command_type": "LOOK",
        "action": "look",
        "subject": "room",
        "target": None,
        "confidence": 0.9,
    }

    # Mock the generate method to return a dict that json.loads can handle
    mock_client.generate.return_value = mock_result
    return mock_client


@pytest.fixture
def nlp_processor(
    mock_config_manager: MagicMock, mock_ollama_client: MagicMock
) -> NLPProcessor:
    """Create an NLP processor with mock dependencies for testing."""
    return NLPProcessor(
        config_manager=mock_config_manager, ollama_client=mock_ollama_client
    )


class TestNLPProcessor:
    """Tests for the NLPProcessor class."""

    @pytest.mark.asyncio
    async def test_process_input(self, nlp_processor: NLPProcessor) -> None:
        """Test processing input with the NLP processor."""
        # Set up a properly formatted mock response for process_input
        mock_intent = Intent(
            command_type=CommandTypeStr.LOOK,
            action="look",
            subject="room",
            target=None,
            confidence=0.9,
        )

        # Use patch to mock the extract_intent method properly
        with patch.object(
            nlp_processor, "extract_intent", return_value=mock_intent
        ) as mock_extract_intent:
            # Call the method
            result = await nlp_processor.process_input("look around the room")

            # Verify the mock was called
            mock_extract_intent.assert_called_once()

            # Assertions on the result
            assert result.command_type == CommandType.LOOK
            assert result.action == "look"
            assert result.subject == "room"
            assert result.target is None
            assert result.parameters.get("confidence") == 0.9

    @pytest.mark.asyncio
    async def test_extract_intent(self, nlp_processor: NLPProcessor) -> None:
        """Test intent extraction."""
        # Create a response that matches the structure expected by the code
        # The nlp_processor.extract_intent tries to directly validate the response
        mock_response = {
            "command_type": "LOOK",
            "action": "look",
            "subject": "room",
            "target": None,
            "confidence": 0.9,
        }

        # Use patch to mock the _generate_completion_async method
        with patch.object(
            nlp_processor, "_generate_completion_async", return_value=mock_response
        ) as mock_generate:
            # Call the method
            result = await nlp_processor.extract_intent("look around the room")

            # Verify the mock was called
            mock_generate.assert_called_once()

            # Verify the result is parsed correctly
            assert result is not None
            assert result.command_type == "LOOK"
            assert result.action == "look"
            assert result.subject == "room"
            assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_extract_intent_error_handling(
        self, nlp_processor: NLPProcessor
    ) -> None:
        """Test error handling during intent extraction."""
        # Use patch to mock the _generate_completion_async method to simulate an error
        with patch.object(
            nlp_processor,
            "_generate_completion_async",
            side_effect=Exception("Test exception"),
        ) as mock_generate_error:
            # The method should handle the error and return None
            result = await nlp_processor.extract_intent("look around the room")

            # Verify the mock was called
            mock_generate_error.assert_called_once()

            # Verify the correct error handling
            assert result is None

    @pytest.mark.asyncio
    async def test_disambiguate_input(self, nlp_processor: NLPProcessor) -> None:
        """Test disambiguating between multiple possible interpretations."""

        mock_response = {
            "response": json.dumps(
                {
                    "selected_interpretation": 1,
                    "confidence": 0.8,
                    "explanation": "The second interpretation makes more sense.",
                }
            )
        }

        # Use patch to mock the _generate_completion_async method
        with patch.object(
            nlp_processor, "_generate_completion_async", return_value=mock_response
        ) as mock_disambiguation:
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

            # Verify the mock was called
            mock_disambiguation.assert_called_once()

            # Verify the result
            assert result.get("command_type") == "EXAMINE"
            assert result.get("confidence") == 0.8
            assert (
                result.get("explanation")
                == "The second interpretation makes more sense."
            )  # noqa: E501

    def test_normalize_input(self, nlp_processor: NLPProcessor) -> None:
        """Test input normalization."""
        result = nlp_processor._normalize_input("  LOOK AROUND  ")
        assert result == "look around"

    def test_format_context(self, nlp_processor: NLPProcessor) -> None:
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
    async def test_generate_semantic_query(self, nlp_processor: NLPProcessor) -> None:
        """Test generating a semantic search query from intent data."""
        intent_data = {"action": "use", "subject": "key", "target": "door"}

        query = await nlp_processor.generate_semantic_query(intent_data)
        assert query == "use key door"
