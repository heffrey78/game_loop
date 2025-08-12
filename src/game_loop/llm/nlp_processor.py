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

from game_loop.config.manager import ConfigManager
from game_loop.core.input_processor import CommandType, ParsedCommand
from game_loop.llm.models import (
    CommandTypeStr,
    Disambiguation,
    GameCharacter,
    GameContext,
    GameObject,
    Intent,
)

logger = logging.getLogger(__name__)


class NLPProcessor:
    """
    Processes natural language input using LLM to extract intent and entities.
    Works alongside InputProcessor to handle complex language structures.
    """

    @staticmethod
    def _strip_thinking_tags(text: str) -> str:
        """
        Remove thinking tags and their content from text.

        Args:
            text: Input text that may contain <think>...</think> tags

        Returns:
            Text with thinking tags and their content removed
        """
        # Use regex to remove <think>...</think> blocks (including multiline)
        pattern = r"<think>.*?</think>"
        cleaned_text = re.sub(pattern, "", text, flags=re.DOTALL)
        # Clean up extra whitespace that might be left behind
        cleaned_text = re.sub(r"\n\s*\n", "\n", cleaned_text)
        return cleaned_text.strip()

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

        # Override prompt template directory to point to the correct location
        if hasattr(self.config_manager, "config") and hasattr(
            self.config_manager.config, "prompts"
        ):
            from pathlib import Path

            # Set the prompts directory to the actual location in the source code
            prompts_dir = Path(__file__).parent / "prompts"
            self.config_manager.config.prompts.template_dir = str(prompts_dir)

        # Use the official Ollama Python client
        if hasattr(self.config_manager, "llm_config"):
            # New ConfigManager structure
            self.host = self.config_manager.llm_config.base_url
            self.model = self.config_manager.llm_config.default_model
        else:
            # Fallback to default values from LLMConfig
            from .config import LLMConfig

            default_config = LLMConfig()
            self.host = default_config.base_url
            self.model = default_config.default_model

        self.client = ollama_client or ollama

        # Load parameters from config
        if hasattr(self.config_manager, "ollama_config"):
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
        else:
            # Fallback to default values from OllamaConfig
            from .config import OllamaConfig

            default_ollama_config = OllamaConfig()
            self.temperature = default_ollama_config.completion_params.get(
                "temperature", 0.7
            )
            self.top_p = default_ollama_config.completion_params.get("top_p", 0.9)
            self.top_k = default_ollama_config.completion_params.get("top_k", 40)
            self.max_tokens = default_ollama_config.completion_params.get(
                "max_tokens", 1024
            )
            self.system_prompt = default_ollama_config.system_prompt

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
        intent = await self.extract_intent(normalized_input, context_str)

        if not intent:
            # Fallback if intent extraction fails
            return ParsedCommand(
                command_type=CommandType.UNKNOWN,
                action="unknown",
                subject=normalized_input,
            )

        try:
            # Map the Intent to a ParsedCommand
            command_type = next(
                (ct for ct in CommandType if ct.name == intent.command_type),
                CommandType.UNKNOWN,
            )

            return ParsedCommand(
                command_type=command_type,
                action=intent.action,
                subject=intent.subject,
                target=intent.target,
                parameters={"confidence": intent.confidence},
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
    ) -> Intent | None:
        """
        Use LLM to recognize user intent from natural language.

        Args:
            normalized_input: Normalized user input
            game_context: Formatted game context string

        Returns:
            Intent model containing parsed intent data, or None if extraction fails
        """
        try:
            # Format the prompt using template
            prompt = self.config_manager.format_prompt(
                "intent_recognition", input=normalized_input, context=game_context
            )

            # Use the official Ollama Python client to generate completion
            try:
                # Add timeout to avoid hanging
                import asyncio

                # Call the Ollama API with format parameter for structured output
                response = await asyncio.wait_for(
                    self._generate_completion_async(prompt, Intent),
                    timeout=30.0,  # Increased timeout for LLM processing
                )

                json_string = json.dumps(response)

                if response and Intent.model_validate_json(json_string):
                    try:
                        intent: Intent = Intent.model_validate_json(json_string)
                        logger.debug(f"Successfully parsed intent: {intent}")
                        return intent
                    except Exception as e:
                        logger.debug(f"Direct model validation failed: {e}")
                elif response:
                    fallback_intent = self._try_fallback_parsing(
                        json_string, normalized_input
                    )
                    if fallback_intent:
                        try:
                            result_intent: Intent = Intent.model_validate(
                                fallback_intent
                            )  # noqa: E501
                            return result_intent
                        except Exception as e:
                            logger.debug(f"Failed to validate fallback " f"intent: {e}")
                            return None

                    # If all parsing attempts fail, try direct fallback
                    fallback = self._create_fallback_intent(normalized_input)
                    if fallback:
                        try:
                            final_intent: Intent = Intent.model_validate(fallback)
                            return final_intent
                        except Exception as e:
                            logger.debug(f"Failed to validate fallback intent: {e}")
                            return None
                else:
                    logger.warning("No response from LLM, returning None")
                    return None
            except (asyncio.TimeoutError, Exception) as e:
                logger.error(f"Error calling Ollama API: {e}")
                return None

        except Exception as e:
            logger.error(f"Error in intent extraction: {e}")
            return None

        return None

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
            interpretations_str = json.dumps(possible_interpretations, indent=2)

            # Format the prompt using template
            prompt = self.config_manager.format_prompt(
                "disambiguation",
                input=normalized_input,
                context=game_context,
                interpretations=interpretations_str,
            )

            # Call the Ollama API with format parameter for structured output
            response = await self._generate_completion_async(prompt, Disambiguation)

            # Extract and process the response text
            if response and isinstance(response, dict) and "response" in response:
                response_text = response["response"].strip()
                logger.debug(f"Raw disambiguation response: {response_text}")

                try:
                    # Try direct model validation first
                    disambiguation = Disambiguation.model_validate_json(response_text)
                    logger.debug(
                        f"Successfully parsed disambiguation: " f"{disambiguation}"
                    )

                    # Get the selected interpretation
                    selected_index = disambiguation.selected_interpretation
                    if 0 <= selected_index < len(possible_interpretations):
                        return {
                            **possible_interpretations[selected_index],
                            "confidence": disambiguation.confidence,
                            "explanation": disambiguation.explanation,
                        }
                except Exception as e:
                    logger.error(f"Failed to parse disambiguation response: {e}")
                    # Try to extract JSON without model validation
                    try:
                        disambiguation_data = json.loads(response_text)
                        selected_index = disambiguation_data.get(
                            "selected_interpretation", 0
                        )
                        if 0 <= selected_index < len(possible_interpretations):
                            return {
                                **possible_interpretations[selected_index],
                                "confidence": disambiguation_data.get(
                                    "confidence", 0.5
                                ),
                                "explanation": disambiguation_data.get(
                                    "explanation", ""
                                ),
                            }
                    except json.JSONDecodeError:
                        logger.error("Failed to parse JSON from response")

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
                else {"command_type": "UNKNOWN", "action": "unknown", "confidence": 0.0}
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

    def _format_context(self, game_context: dict[str, Any] | GameContext | None) -> str:
        """
        Format game context as a string for LLM.

        Args:
            game_context: Game context as dictionary or GameContext model

        Returns:
            Formatted context string
        """
        if game_context is None:
            return "No specific context available."

        context_parts = []

        # Handle both dict and GameContext
        if isinstance(game_context, dict):
            # Handle dictionary input for backwards compatibility
            # Add location description if available
            if location := game_context.get("current_location"):
                context_parts.append(
                    f"You are in: {location.get('name', 'unknown location')}"
                )
                if description := location.get("description"):
                    context_parts.append(f"Location description: {description}")
            elif location := game_context.get("location"):
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
                            context_parts.append(
                                f"- {obj.get('name', 'unknown object')}"
                            )
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
                            context_parts.append(
                                f"- {item.get('name', 'unknown item')}"
                            )
                        else:
                            context_parts.append(f"- {item}")
            elif inventory := game_context.get("inventory"):
                if isinstance(inventory, list) and inventory:
                    context_parts.append("You are carrying:")
                    for item in inventory:
                        if isinstance(item, dict):
                            context_parts.append(
                                f"- {item.get('name', 'unknown item')}"
                            )
                        else:
                            context_parts.append(f"- {item}")

        else:  # Handle GameContext model
            # Add location description if available
            if location := getattr(game_context, "current_location", None):
                context_parts.append(f"You are in: {location.name}")
                if description := location.description:
                    context_parts.append(f"Location description: {description}")
            elif location := getattr(game_context, "location", None):
                context_parts.append(f"You are in: {location.name}")
                if description := location.description:
                    context_parts.append(f"Location description: {description}")

            # Add visible objects if available
            if (
                hasattr(game_context, "visible_objects")
                and game_context.visible_objects
            ):
                context_parts.append("You can see:")
                for obj in game_context.visible_objects:
                    if isinstance(obj, GameObject):
                        context_parts.append(f"- {obj.name}")
                    else:
                        context_parts.append(f"- {obj}")

            # Add NPCs if available
            if hasattr(game_context, "npcs") and game_context.npcs:
                context_parts.append("Characters present:")
                for npc in game_context.npcs:
                    if isinstance(npc, GameCharacter):
                        context_parts.append(f"- {npc.name}")
                    else:
                        context_parts.append(f"- {npc}")

            # Add inventory if available
            if hasattr(game_context, "inventory") and game_context.inventory:
                context_parts.append("You are carrying:")
                for item in game_context.inventory:
                    if isinstance(item, GameObject):
                        context_parts.append(f"- {item.name}")
                    else:
                        context_parts.append(f"- {item}")

        # Combine all parts with line breaks
        return (
            "\n".join(context_parts)
            if context_parts
            else ("No specific context available.")
        )

    def _create_fallback_intent(self, input_text: str) -> dict[str, Any]:
        """
        Create a fallback intent for common commands when LLM parsing fails.

        Args:
            input_text: The normalized user input

        Returns:
            Intent data as dictionary that can be used to create an Intent model
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
                    "command_type": CommandTypeStr.MOVEMENT,
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
                        "command_type": CommandTypeStr.MOVEMENT,
                        "action": "go",
                        "subject": words[i + 1],
                        "target": None,
                        "confidence": 0.9,
                    }

        # Check for other common commands
        if input_text in ["look", "l", "look around"]:
            return {
                "command_type": CommandTypeStr.LOOK,
                "action": "look",
                "subject": None,
                "target": None,
                "confidence": 0.9,
            }

        if input_text in ["inventory", "i", "items"]:
            return {
                "command_type": CommandTypeStr.INVENTORY,
                "action": "inventory",
                "subject": None,
                "target": None,
                "confidence": 0.9,
            }

        if input_text in ["help", "h", "?"]:
            return {
                "command_type": CommandTypeStr.HELP,
                "action": "help",
                "subject": None,
                "target": None,
                "confidence": 0.9,
            }

        if input_text in ["quit", "exit", "q", "bye"]:
            return {
                "command_type": CommandTypeStr.QUIT,
                "action": "quit",
                "subject": None,
                "target": None,
                "confidence": 0.9,
            }

        # No applicable fallback found, return an empty dict with proper structure
        return {
            "command_type": CommandTypeStr.UNKNOWN,
            "action": "unknown",
            "subject": input_text if input_text else None,
            "target": None,
            "confidence": 0.0,
        }

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
                if intent_data and isinstance(intent_data, dict):
                    logger.debug(f"Successfully extracted intent data: {intent_data}")
                    return dict(intent_data)
                else:
                    logger.debug(f"Parsed JSON is not a dict: {type(intent_data)}")
                    return {}

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

                if intent_data and isinstance(intent_data, dict):
                    logger.debug(f"Successfully extracted intent data: {intent_data}")
                    return dict(intent_data)
                else:
                    logger.debug(f"Parsed JSON is not a dict: {type(intent_data)}")
                    return {}

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse extracted JSON: {e}")

                # One more attempt with extra cleaning on the extracted JSON
                try:
                    cleaned_json = json_str.replace("\n", " ").replace("\\", "").strip()
                    intent_data = json.loads(cleaned_json)
                    logger.debug("Successfully parsed JSON after extra cleaning")

                    if intent_data and isinstance(intent_data, dict):
                        logger.debug(
                            f"Successfully extracted intent data: {intent_data}"
                        )
                        return dict(intent_data)
                    else:
                        logger.debug(f"Parsed JSON is not a dict: {type(intent_data)}")
                        return {}
                except json.JSONDecodeError:
                    pass

        # Pattern matching for "look at X" commands
        look_at_match = re.search(
            r"look\s+at\s+(?:the\s+)?([a-zA-Z0-9_\s]+)",
            normalized_input,
        )
        if look_at_match:
            item_name = look_at_match.group(1).strip()
            return {
                "command_type": "EXAMINE",
                "action": "examine",
                "subject": item_name,
                "target": None,
                "confidence": 0.95,  # High confidence for explicit look at pattern
            }

        # Pattern matching for "examine X" commands
        examine_match = re.search(
            r"(?:examine|inspect|check)\s+(?:the\s+)?([a-zA-Z0-9_\s]+)",
            normalized_input,
        )
        if examine_match:
            item_name = examine_match.group(1).strip()
            return {
                "command_type": "EXAMINE",
                "action": "examine",
                "subject": item_name,
                "target": None,
                "confidence": 0.95,  # High confidence for explicit examine pattern
            }

        # Continue with other pattern matching for complex commands
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
            or "get" in normalized_input
        ):
            # Special handling for "pick up" as a phrase
            pick_up_match = re.search(
                r"pick\s+up\s+(?:the\s+)?([a-zA-Z0-9_\s]+)",
                normalized_input,
            )
            if pick_up_match:
                item_name = pick_up_match.group(1).strip()
                return {
                    "command_type": CommandTypeStr.TAKE,
                    "action": "take",
                    "subject": item_name,
                    "target": None,
                    "confidence": 0.9,
                }

            # General handling for other take commands
            take_match = re.search(
                r"(?:take|grab|get)\s+(?:the\s+)?([a-zA-Z0-9_\s]+)",
                normalized_input,
            )
            if take_match:
                item_name = take_match.group(1).strip()
                return {
                    "command_type": CommandTypeStr.TAKE,
                    "action": "take",
                    "subject": item_name,
                    "target": None,
                    "confidence": 0.9,
                }

            # Fallback to simple word parsing if regex doesn't match
            words = normalized_input.split()
            item_words = []
            recording = False
            skip_next = False

            for i, word in enumerate(words):
                if skip_next:
                    skip_next = False
                    continue

                if word == "pick" and i + 1 < len(words) and words[i + 1] == "up":
                    recording = True
                    skip_next = True  # Skip the "up" in the next iteration
                    continue
                elif word in ["take", "grab", "get"]:
                    recording = True
                    continue

                if recording:
                    if word == "the" and i < len(words) - 1:
                        continue  # Skip "the" but only if it's not the last word
                    item_words.append(word)

            if item_words:
                item_name = " ".join(item_words)
                return {
                    "command_type": CommandTypeStr.TAKE,
                    "action": "take",
                    "subject": item_name,
                    "target": None,
                    "confidence": 0.8,
                }

        # If all parsing attempts fail, use fallback
        return self._create_fallback_intent(normalized_input)

    async def _generate_completion_async(
        self, prompt: str, model_class: type[BaseModel] | None = None
    ) -> dict[str, Any]:
        """
        Generate completion using Ollama API asynchronously.

        Args:
            prompt: The prompt to send to Ollama
            model_class: Optional Pydantic model class to use for structured output

        Returns:
            Dictionary containing the Ollama response
        """
        import asyncio

        # Create a thread-safe event loop
        loop = asyncio.get_running_loop()

        # Prepare basic options
        options = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_predict": self.max_tokens,
        }

        # Add format information for JSON output
        if model_class and issubclass(model_class, BaseModel):
            # According to Ollama docs, we should use the model's JSON schema directly
            options["format"] = model_class.model_json_schema()

        # Use enhanced system prompt that reinforces JSON output requirement
        system_prompt = (
            f"{self.system_prompt}\nYou must respond with valid JSON only, "
            "with no explanations or additional text."
        )

        try:
            # For unit tests, we allow a direct mock response
            if hasattr(self.client, "generate") and callable(self.client.generate):
                ollama_response = await loop.run_in_executor(
                    None,
                    lambda: self.client.generate(
                        model=self.model,
                        prompt=prompt,
                        system=system_prompt,
                        options=options,
                    ),
                )

                if ollama_response:
                    # Handle ollama._types.GenerateResponse object
                    if hasattr(ollama_response, "response"):
                        response_text = ollama_response.response
                    elif isinstance(ollama_response, dict):
                        response_text = ollama_response.get("response", "")
                    else:
                        response_text = str(ollama_response)

                    # Strip thinking tags before parsing JSON
                    response_text = self._strip_thinking_tags(response_text)

                    try:
                        response_json = json.loads(response_text)
                        if isinstance(response_json, dict):
                            return response_json
                        else:
                            logger.error(
                                f"Unexpected Ollama response format: "
                                f"{type(response_json)}"
                            )
                            return {"response": str(ollama_response)}
                    except json.JSONDecodeError:
                        logger.error(
                            f"Failed to parse response as JSON: " f"{response_text}"
                        )
                        return {"response": response_text}
                return {"response": "{}"}
            else:
                logger.error("Invalid client configuration")
                return {"response": "{}"}

        except Exception as e:
            logger.error(f"Error in _generate_completion_async: {e}")
            return {"response": "{}"}
