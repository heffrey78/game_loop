"""
NLP Processor for Game Loop.
Processes natural language input using LLM to extract intent and entities.
"""

import json
import logging
import re
from typing import Any

import ollama
from pydantic import BaseModel

from game_loop.core.input_processor import CommandType, ParsedCommand
from game_loop.llm.config import ConfigManager

logger = logging.getLogger(__name__)


class NLPProcessor:
    """
    Processes natural language input using LLM to extract intent and entities.
    Works alongside InputProcessor to handle complex language structures.
    """

    def __init__(
        self, config_manager: ConfigManager | None = None, ollama_client: Any = None
    ):
        """
        Initialize the NLP processor with config and client.

        Args:
            config_manager: Configuration manager for LLM settings
            ollama_client: Client for Ollama API communication
        """
        self.config_manager = config_manager or ConfigManager()

        # Use the official Ollama Python client
        self.host = self.config_manager.llm_config.base_url
        self.model = self.config_manager.llm_config.default_model
        self.client = ollama_client or ollama

        # Load parameters from config
        self.temperature = self.config_manager.ollama_config.completion_params.get(
            "temperature", 0.7
        )
        self.top_p = self.config_manager.ollama_config.completion_params.get(
            "top_p", 0.9
        )
        self.top_k = self.config_manager.ollama_config.completion_params.get(
            "top_k", 40
        )
        self.max_tokens = self.config_manager.ollama_config.completion_params.get(
            "max_tokens", 1024
        )
        self.system_prompt = self.config_manager.ollama_config.system_prompt

    async def process_input(
        self, user_input: str, game_context: dict[str, Any] | None = None
    ) -> ParsedCommand:
        """
        Process natural language input and extract structured command.

        Args:
            user_input: Raw user input text
            game_context: Current game state context for better understanding

        Returns:
            Parsed command representing the user's intent
        """
        normalized_input = self._normalize_input(user_input)

        # Extract context string from game_context if provided
        context_str = (
            self._format_context(game_context)
            if game_context
            else "No additional context available."
        )

        # Extract intent using LLM
        intent_data = await self.extract_intent(normalized_input, context_str)

        if not intent_data:
            # Fallback if intent extraction fails
            return ParsedCommand(
                command_type=CommandType.UNKNOWN,
                action="unknown",
                subject=normalized_input,
            )

        try:
            # Map the LLM intent data to a ParsedCommand
            command_type_str = intent_data.get("command_type", "UNKNOWN")
            # Handle case sensitivity in enum matching
            command_type = next(
                (ct for ct in CommandType if ct.name == command_type_str),
                CommandType.UNKNOWN,
            )

            return ParsedCommand(
                command_type=command_type,
                action=intent_data.get("action", "unknown"),
                subject=intent_data.get("subject"),
                target=intent_data.get("target"),
                parameters={"confidence": intent_data.get("confidence", 0.0)},
            )
        except Exception as e:
            logger.error(f"Error converting intent data to ParsedCommand: {e}")
            return ParsedCommand(
                command_type=CommandType.UNKNOWN,
                action="unknown",
                subject=normalized_input,
            )

    async def extract_intent(
        self, normalized_input: str, game_context: str = "No additional context."
    ) -> dict[str, Any]:
        """
        Use LLM to recognize user intent from natural language.

        Args:
            normalized_input: Normalized user input
            game_context: Formatted game context string

        Returns:
            Dictionary containing extracted intent data
        """
        try:

            class Intent(BaseModel):
                command_type: str
                action: str
                subject: str
                target: str
                confidence: float

            self.format = Intent.model_json_schema()

            # Format the prompt using template
            prompt = self.config_manager.format_prompt(
                "intent_recognition", input=normalized_input, context=game_context
            )

            # Use the official Ollama Python client to generate completion
            try:
                # Add timeout to avoid hanging
                import asyncio

                # Call the Ollama API with format=json for structured output
                response = await asyncio.wait_for(
                    self._generate_completion_async(prompt),
                    timeout=10.0,  # 10-second timeout
                )

                # Extract and process the response text
                if response and isinstance(response, dict) and "response" in response:
                    response_text = response["response"].strip()
                    logger.debug(f"Raw LLM response: {response_text}")

                    # Try to parse the JSON response
                    intent_data = self._extract_json_from_text(response_text)

                    if intent_data:
                        logger.debug(
                            f"Successfully extracted intent data: {intent_data}"
                        )
                        return (
                            dict(intent_data) if isinstance(intent_data, dict) else {}
                        )

                # If JSON parsing failed or response is empty, return empty dict
                logger.warning("Invalid JSON response received from LLM")
                return {}

            except (asyncio.TimeoutError, Exception) as e:
                logger.error(f"Error calling Ollama API: {e}")
                return {}

        except Exception as e:
            logger.error(f"Error in intent extraction: {e}")
            return {}

    async def _generate_completion_async(self, prompt: str) -> dict[str, Any]:
        """
        Generate completion using Ollama API asynchronously.

        Args:
            prompt: The prompt to send to Ollama

        Returns:
            Dictionary containing the Ollama response
        """
        import asyncio

        # Create a thread-safe event loop
        loop = asyncio.get_running_loop()

        # Run the synchronous client in a thread pool
        response = await loop.run_in_executor(
            None,
            lambda: self.client.generate(
                model=self.model,
                prompt=prompt,
                system=self.system_prompt,
                options={
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                    "num_predict": self.max_tokens,
                    "format": self.format,
                },
            ),
        )
        if not isinstance(response, dict):
            return {"response": str(response)}
        return response

    def _try_fallback_parsing(
        self, response_text: str, normalized_input: str
    ) -> dict[str, Any]:
        """
        Attempt to extract JSON from potentially malformed response.

        Args:
            response_text: Text response from LLM
            normalized_input: Original normalized input for fallback

        Returns:
            Extracted intent data or fallback
        """
        # Try different cleaning approaches
        cleaned_attempts = [
            response_text.strip(),  # Basic strip
            response_text.strip().strip('"').strip(),  # Remove quotes
            response_text.strip().replace("\n", "").strip(),  # Remove newlines
            response_text.replace("```json", "")
            .replace("```", "")
            .strip(),  # Remove markdown
        ]

        # Try each cleaned version
        for attempt in cleaned_attempts:
            try:
                intent_data = json.loads(attempt)
                logger.debug("Successfully parsed JSON after cleaning")
                if intent_data:
                    logger.debug(f"Successfully extracted intent data: {intent_data}")
                    return dict(intent_data) if isinstance(intent_data, dict) else {}

            except json.JSONDecodeError:
                pass  # Try next cleaning approach

        # If cleaning didn't work, try to extract JSON object
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            logger.debug(f"Extracted potential JSON: {json_str}")
            try:
                intent_data = json.loads(json_str)
                logger.debug("Successfully parsed extracted JSON")

                if intent_data:
                    logger.debug(f"Successfully extracted intent data: {intent_data}")
                    return dict(intent_data) if isinstance(intent_data, dict) else {}

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse extracted JSON: {e}")

                # One more attempt with extra cleaning on the extracted JSON
                try:
                    cleaned_json = json_str.replace("\n", " ").replace("\\", "").strip()
                    intent_data = json.loads(cleaned_json)
                    logger.debug("Successfully parsed JSON after extra cleaning")

                    if intent_data:
                        logger.debug(
                            f"Successfully extracted intent data: {intent_data}"
                        )
                        return (
                            dict(intent_data) if isinstance(intent_data, dict) else {}
                        )
                except json.JSONDecodeError:
                    pass

        # Pattern matching for complex commands

        # Check for "put X in/into Y" pattern
        put_in_match = re.search(
            r"put\s+(?:the\s+)?([a-zA-Z0-9_\s]+)\s+(?:in|into)\s+(?:the\s+)?([a-zA-Z0-9_\s]+)",
            normalized_input,
        )
        if put_in_match:
            subject = put_in_match.group(1).strip()
            target = put_in_match.group(2).strip()
            return {
                "command_type": "USE",
                "action": "put",
                "subject": subject,
                "target": target,
                "confidence": 0.85,
            }

        # Check for "place X on/onto Y" pattern
        place_on_match = re.search(
            r"place\s+(?:the\s+)?([a-zA-Z0-9_\s]+)\s+(?:on|onto)\s+(?:the\s+)?([a-zA-Z0-9_\s]+)",
            normalized_input,
        )
        if place_on_match:
            subject = place_on_match.group(1).strip()
            target = place_on_match.group(2).strip()
            return {
                "command_type": "USE",
                "action": "place",
                "subject": subject,
                "target": target,
                "confidence": 0.85,
            }

        # Check for "use X with/on Y" pattern
        use_with_match = re.search(
            r"use\s+(?:the\s+)?([a-zA-Z0-9_\s]+)\s+(?:with|on)\s+(?:the\s+)?([a-zA-Z0-9_\s]+)",
            normalized_input,
        )
        if use_with_match:
            subject = use_with_match.group(1).strip()
            target = use_with_match.group(2).strip()
            return {
                "command_type": "USE",
                "action": "use",
                "subject": subject,
                "target": target,
                "confidence": 0.9,
            }

        # Check for "open X with Y" pattern
        open_with_match = re.search(
            r"open\s+(?:the\s+)?([a-zA-Z0-9_\s]+)\s+(?:with)\s+(?:the\s+)?([a-zA-Z0-9_\s]+)",
            normalized_input,
        )
        if open_with_match:
            subject = open_with_match.group(1).strip()
            target = open_with_match.group(2).strip()
            return {
                "command_type": "USE",
                "action": "open",
                "subject": subject,
                "target": target,
                "confidence": 0.85,
            }

        # Check for "take" action with specific patterns
        if (
            "pick up" in normalized_input
            or "take" in normalized_input
            or "grab" in normalized_input
        ):
            words = normalized_input.split()
            item_words = []
            recording = False

            for word in words:
                if word in ["pick", "take", "grab", "get"]:
                    recording = True
                    continue
                if recording and word not in ["up", "the"]:
                    item_words.append(word)

            if item_words:
                item_name = " ".join(item_words)
                return {
                    "command_type": "TAKE",
                    "action": "take",
                    "subject": item_name,
                    "target": None,
                    "confidence": 0.8,
                }

        # Check for "examine" action with specific patterns
        if (
            "look at" in normalized_input
            or "examine" in normalized_input
            or "check" in normalized_input
        ):
            words = normalized_input.split()
            item_words = []
            recording = False

            for i, word in enumerate(words):
                if (
                    word == "look" and i + 1 < len(words) and words[i + 1] == "at"
                ) or word in ["examine", "check", "inspect"]:
                    recording = True
                    if word == "look":  # Skip the "at" in "look at"
                        continue
                    continue
                if recording and word not in ["at", "the"]:
                    item_words.append(word)

            if item_words:
                item_name = " ".join(item_words)
                return {
                    "command_type": "EXAMINE",
                    "action": "examine",
                    "subject": item_name,
                    "target": None,
                    "confidence": 0.8,
                }

        # If all parsing attempts fail, use fallback
        return self._create_fallback_intent(normalized_input)

    async def disambiguate_input(
        self,
        normalized_input: str,
        possible_interpretations: list[dict[str, Any]],
        game_context: str = "No additional context.",
    ) -> dict[str, Any]:
        """
        Resolve ambiguous commands when multiple interpretations are possible.

        Args:
            normalized_input: Normalized user input
            possible_interpretations: List of possible interpretations
            game_context: Current game context

        Returns:
            Selected interpretation with confidence score
        """
        try:

            class Disambiguation(BaseModel):
                selected_interpretation: int
                confidence: float
                explanation: str

            self.format = Disambiguation.model_json_schema()

            interpretations_str = json.dumps(possible_interpretations, indent=2)

            # Format the prompt using template
            prompt = self.config_manager.format_prompt(
                "disambiguation",
                input=normalized_input,
                context=game_context,
                interpretations=interpretations_str,
            )

            response = await self._generate_completion_async(prompt)

            # Extract and parse the response
            if response and "response" in response:
                response_text = response["response"].strip()
                logger.debug(f"Raw disambiguation response: {response_text}")

                try:
                    disambiguation_data = json.loads(response_text)

                    # Get the selected interpretation
                    selected_index = disambiguation_data.get(
                        "selected_interpretation", 0
                    )
                    if 0 <= selected_index < len(possible_interpretations):
                        return {
                            **possible_interpretations[selected_index],
                            "confidence": disambiguation_data.get("confidence", 0.5),
                            "explanation": disambiguation_data.get("explanation", ""),
                        }
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse JSON from disambiguation response: {e}"
                    )

            logger.warning("Invalid disambiguation response or index out of range")
            # Return default dict instead of possibly returning None
            return (
                possible_interpretations[0]
                if possible_interpretations
                else {"command_type": "UNKNOWN", "action": "unknown", "confidence": 0.0}
            )
        except Exception as e:
            logger.error(f"Error in disambiguation: {e}")
            return (
                possible_interpretations[0]
                if possible_interpretations
                else dict[str, Any]()
            )

    async def generate_semantic_query(self, intent_data: dict[str, Any]) -> str:
        """
        Generate semantic search query based on extracted intent.

        Args:
            intent_data: Dictionary containing intent data

        Returns:
            Query string for semantic search
        """
        query_parts = []

        # Add action to query
        if action := intent_data.get("action"):
            query_parts.append(action)

        # Add subject to query if available
        if subject := intent_data.get("subject"):
            query_parts.append(subject)

        # Add target to query if available
        if target := intent_data.get("target"):
            query_parts.append(target)

        # Combine parts into a query string
        query = " ".join(query_parts)

        # If query is empty, return a default
        return query if query else "game object"

    def _normalize_input(self, user_input: str) -> str:
        """
        Prepare input for NLP processing.

        Args:
            user_input: Raw user input

        Returns:
            Normalized input string
        """
        return user_input.lower().strip()

    def _format_context(self, game_context: dict[str, Any]) -> str:
        """
        Format game context dictionary as a string for LLM.

        Args:
            game_context: Dictionary containing game context

        Returns:
            Formatted context string
        """
        context_parts = []

        # Add location description if available
        if location := game_context.get("current_location"):
            context_parts.append(
                f"You are in: {location.get('name', 'unknown location')}"
            )
            if description := location.get("description"):
                context_parts.append(f"Location description: {description}")
        elif location := game_context.get(
            "location"
        ):  # Support both context structures
            context_parts.append(
                f"You are in: {location.get('name', 'unknown location')}"
            )
            if description := location.get("description"):
                context_parts.append(f"Location description: {description}")

        # Add visible objects if available
        if objects := game_context.get("visible_objects"):
            if isinstance(objects, list) and objects:
                context_parts.append("You can see:")
                for obj in objects:
                    if isinstance(obj, dict):
                        context_parts.append(f"- {obj.get('name', 'unknown object')}")
                    else:
                        context_parts.append(f"- {obj}")

        # Add NPCs if available
        if npcs := game_context.get("npcs"):
            if isinstance(npcs, list) and npcs:
                context_parts.append("Characters present:")
                for npc in npcs:
                    if isinstance(npc, dict):
                        context_parts.append(
                            f"- {npc.get('name', 'unknown character')}"
                        )
                    else:
                        context_parts.append(f"- {npc}")

        # Add inventory if available
        if player := game_context.get("player"):
            inventory = player.get("inventory", [])
            if isinstance(inventory, list) and inventory:
                context_parts.append("You are carrying:")
                for item in inventory:
                    if isinstance(item, dict):
                        context_parts.append(f"- {item.get('name', 'unknown item')}")
                    else:
                        context_parts.append(f"- {item}")
        elif inventory := game_context.get(
            "inventory"
        ):  # Support both context structures
            if isinstance(inventory, list) and inventory:
                context_parts.append("You are carrying:")
                for item in inventory:
                    if isinstance(item, dict):
                        context_parts.append(f"- {item.get('name', 'unknown item')}")
                    else:
                        context_parts.append(f"- {item}")

        # Combine all parts with line breaks
        return (
            "\n".join(context_parts)
            if context_parts
            else "No specific context available."
        )

    def _create_fallback_intent(self, input_text: str) -> dict[str, Any]:
        """
        Create a fallback intent for common commands when LLM parsing fails.

        Args:
            input_text: The normalized user input

        Returns:
            A fallback intent dictionary, or empty dict if no fallback applies
        """
        # Simple word-based detection for common movement commands
        movement_words = ["go", "move", "walk", "run", "head", "travel"]
        directions = [
            "north",
            "south",
            "east",
            "west",
            "up",
            "down",
            "n",
            "s",
            "e",
            "w",
            "u",
            "d",
        ]

        words = input_text.split()

        # Check for movement pattern
        for word in words:
            # Direct direction ("north")
            if word in directions:
                return {
                    "command_type": "MOVEMENT",
                    "action": "go",
                    "subject": word,
                    "target": None,
                    "confidence": 0.8,
                }

        # Check for verb+direction pattern ("go north")
        if len(words) >= 2:
            for i, word in enumerate(words[:-1]):
                if word in movement_words and words[i + 1] in directions:
                    return {
                        "command_type": "MOVEMENT",
                        "action": "go",
                        "subject": words[i + 1],
                        "target": None,
                        "confidence": 0.9,
                    }

        # Check for other common commands
        if input_text in ["look", "l", "look around"]:
            return {
                "command_type": "LOOK",
                "action": "look",
                "subject": None,
                "target": None,
                "confidence": 0.9,
            }

        if input_text in ["inventory", "i", "items"]:
            return {
                "command_type": "INVENTORY",
                "action": "inventory",
                "subject": None,
                "target": None,
                "confidence": 0.9,
            }

        if input_text in ["help", "h", "?"]:
            return {
                "command_type": "HELP",
                "action": "help",
                "subject": None,
                "target": None,
                "confidence": 0.9,
            }

        if input_text in ["quit", "exit", "q", "bye"]:
            return {
                "command_type": "QUIT",
                "action": "quit",
                "subject": None,
                "target": None,
                "confidence": 0.9,
            }

        # No applicable fallback found, return an empty dict with proper structure
        return {"command_type": "UNKNOWN", "action": "unknown", "confidence": 0.0}

    def _extract_json_from_text(self, text: str) -> dict[str, Any]:
        """
        Extract a valid JSON object from potentially malformed text.
        Uses multiple strategies to find and parse JSON.

        Args:
            text: The text that may contain a JSON object

        Returns:
            Dictionary from parsed JSON or empty dict if parsing fails
        """
        import re

        if not text:
            return {}

        # Strategy 1: Try direct parsing
        try:
            json_result = json.loads(text)
            # Ensure we return a dictionary
            if isinstance(json_result, dict):
                return json_result
            else:
                logger.debug(
                    f"JSON parsed successfully but result is not a dict: "
                    f"{type(json_result)}"
                )
                return {}
        except json.JSONDecodeError:
            logger.debug("Direct JSON parsing failed, trying alternative methods")

        # Strategy 2: Try various cleaning approaches
        cleaned_versions = [
            text.strip(),
            text.strip().strip('"').strip("'"),
            text.replace("\n", "").strip(),
            text.replace("```json", "").replace("```", "").strip(),
            "".join(c for c in text if c not in "\n\r\t"),
            re.sub(r"[^\x20-\x7E]", "", text),
        ]

        for cleaned in cleaned_versions:
            try:
                json_result = json.loads(cleaned)
                # Ensure we return a dictionary
                if isinstance(json_result, dict):
                    return json_result
                else:
                    logger.debug(f"Result is not a dict: {type(json_result)}")
                    continue
            except json.JSONDecodeError:
                pass

        # Strategy 3: Extract JSON using regex pattern
        try:
            json_pattern = r"\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}"
            matches = re.findall(json_pattern, text)
            if matches:
                for match in matches:
                    try:
                        json_result = json.loads(match)
                        # Ensure we return a dictionary
                        if isinstance(json_result, dict):
                            return json_result
                        else:
                            logger.debug(f"Result is not a dict: {type(json_result)}")
                            continue
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.debug(f"Regex extraction failed: {e}")

        # Strategy 4: Find JSON object using brace matching
        try:
            json_start = text.find("{")
            if json_start >= 0:
                # Find matching closing brace
                brace_count = 0
                for i in range(json_start, len(text)):
                    if text[i] == "{":
                        brace_count += 1
                    elif text[i] == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            # Found complete JSON object
                            json_str = text[json_start : i + 1]
                            try:
                                json_result = json.loads(json_str)
                                # Ensure we return a dictionary
                                if isinstance(json_result, dict):
                                    return json_result
                                else:
                                    logger.debug(
                                        f"Result is not a dict: " f"{type(json_result)}"
                                    )

                                    # Try one more cleaning pass
                                    clean_json = json_str.replace("\n", " ").replace(
                                        "\\", ""
                                    )
                                    try:
                                        json_result = json.loads(clean_json)
                                        if isinstance(json_result, dict):
                                            return json_result
                                        else:
                                            logger.debug(
                                                f"Result is not a dict: "
                                                f"{type(json_result)}"
                                            )
                                    except json.JSONDecodeError:
                                        pass
                                    break
                            except json.JSONDecodeError:
                                # Try one more cleaning pass
                                clean_json = json_str.replace("\n", " ").replace(
                                    "\\", ""
                                )
                                try:
                                    json_result = json.loads(clean_json)
                                    if isinstance(json_result, dict):
                                        return json_result
                                    else:
                                        logger.debug(
                                            f"Result is not a dict: "
                                            f"{type(json_result)}"
                                        )
                                except json.JSONDecodeError:
                                    break
        except Exception as e:
            logger.debug(f"Brace matching failed: {e}")

        # If all extraction methods fail, return empty dict
        logger.warning("All JSON extraction methods failed")
        return {}
