"""
Location generator that uses LLM to dynamically create new locations.
"""

import json
import logging
import re
import time
from typing import Any
from uuid import UUID

import ollama
from jinja2 import Environment, FileSystemLoader

from ...llm.config import LLMConfig
from ...state.models import WorldState
from ..models.location_models import (
    EnrichedContext,
    GeneratedLocation,
    GenerationMetrics,
    LocationConnection,
    LocationGenerationContext,
    LocationTheme,
    ValidationResult,
)
from ..models.navigation_models import ExpansionPoint
from .context_collector import LocationContextCollector
from .location_storage import LocationStorage
from .theme_manager import LocationThemeManager

logger = logging.getLogger(__name__)


class LocationGenerator:
    """Generates new locations using LLM with context awareness and theme consistency."""

    def __init__(
        self,
        ollama_client: ollama.Client,
        world_state: WorldState,
        theme_manager: LocationThemeManager,
        context_collector: LocationContextCollector,
        location_storage: LocationStorage,
        llm_config: LLMConfig | None = None,
    ):
        self.ollama_client = ollama_client
        self.world_state = world_state
        self.theme_manager = theme_manager
        self.context_collector = context_collector
        self.location_storage = location_storage
        self.llm_config = llm_config or LLMConfig()

        # Initialize Jinja2 environment for templates
        try:
            self.jinja_env = Environment(
                loader=FileSystemLoader("templates/location_generation"),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        except Exception as e:
            logger.warning(f"Could not initialize template environment: {e}")
            self.jinja_env = None

        self._generation_metrics: list[GenerationMetrics] = []

    async def generate_location(
        self, context: LocationGenerationContext
    ) -> GeneratedLocation:
        """Generate a new location based on provided context."""
        start_time = time.time()
        metrics = GenerationMetrics(
            generation_time_ms=0,
            context_collection_time_ms=0,
            llm_response_time_ms=0,
            validation_time_ms=0,
            storage_time_ms=0,
        )

        try:
            logger.info(
                f"Generating location for expansion at {context.expansion_point.direction}"
            )

            # Check cache first
            cache_start = time.time()
            context_data = self._serialize_context_for_cache(context)
            context_hash = self.location_storage.generate_context_hash(context_data)
            cached_result = await self.location_storage.get_cached_generation(
                context_hash
            )

            if cached_result:
                logger.debug("Using cached location generation result")
                metrics.cache_hit = True
                metrics.generation_time_ms = int((time.time() - start_time) * 1000)
                self._generation_metrics.append(metrics)
                return cached_result.generated_location

            # Enrich context with additional analysis
            context_start = time.time()
            enriched_context = await self.context_collector.enrich_context(context)
            metrics.context_collection_time_ms = int(
                (time.time() - context_start) * 1000
            )

            # Determine appropriate theme
            selected_theme = await self.theme_manager.determine_location_theme(context)

            # Generate location using LLM
            llm_start = time.time()
            location_data = await self._generate_with_llm(
                enriched_context, selected_theme
            )
            metrics.llm_response_time_ms = int((time.time() - llm_start) * 1000)

            # Create GeneratedLocation object
            generated_location = GeneratedLocation(
                name=location_data["name"],
                description=location_data["description"],
                theme=selected_theme,
                location_type=location_data.get("location_type", "generic"),
                objects=location_data.get("objects", []),
                npcs=location_data.get("potential_npcs", []),
                connections=location_data.get("connections", {}),
                metadata={
                    "atmosphere": location_data.get("atmosphere", ""),
                    "special_features": location_data.get("special_features", []),
                    "generation_timestamp": time.time(),
                    "generation_context_hash": context_hash,
                },
                short_description=location_data.get("short_description", ""),
                atmosphere=location_data.get("atmosphere", ""),
                special_features=location_data.get("special_features", []),
                generation_context=context,
            )

            # Validate location consistency
            validation_start = time.time()
            validation_result = await self.validate_location_consistency(
                generated_location,
                [
                    LocationTheme(
                        name=adj.theme,
                        description="",
                        visual_elements=[],
                        atmosphere="",
                        typical_objects=[],
                        typical_npcs=[],
                        generation_parameters={},
                    )
                    for adj in context.adjacent_locations
                ],
            )
            metrics.validation_time_ms = int((time.time() - validation_start) * 1000)

            if not validation_result.is_valid:
                logger.warning(
                    f"Generated location failed validation: {validation_result.issues}"
                )
                # For now, we'll still return the location but with a warning
                generated_location.metadata["validation_warnings"] = (
                    validation_result.issues
                )

            # Cache the result
            try:
                from datetime import timedelta

                await self.location_storage.cache_generation_result(
                    context_hash, generated_location, timedelta(hours=1)
                )
            except Exception as e:
                logger.warning(f"Failed to cache generation result: {e}")

            # Update metrics
            metrics.generation_time_ms = int((time.time() - start_time) * 1000)
            self._generation_metrics.append(metrics)

            logger.info(f"Successfully generated location: {generated_location.name}")
            return generated_location

        except Exception as e:
            logger.error(f"Error generating location: {e}")
            metrics.generation_time_ms = int((time.time() - start_time) * 1000)
            self._generation_metrics.append(metrics)
            raise

    def _serialize_context_for_cache(
        self, context: LocationGenerationContext
    ) -> dict[str, Any]:
        """Serialize context for cache key generation."""
        return {
            "expansion_point": {
                "location_id": context.expansion_point.location_id,
                "direction": context.expansion_point.direction,
                "priority": context.expansion_point.priority,
            },
            "adjacent_locations": [
                {"name": adj.name, "theme": adj.theme, "direction": adj.direction}
                for adj in context.adjacent_locations
            ],
            "player_preferences": {
                "environments": context.player_preferences.environments,
                "complexity_level": context.player_preferences.complexity_level,
                "preferred_themes": context.player_preferences.preferred_themes,
            },
            "world_themes": [theme.name for theme in context.world_themes],
        }

    async def _generate_with_llm(
        self, enriched_context: EnrichedContext, selected_theme: LocationTheme
    ) -> dict[str, Any]:
        """Generate location using LLM."""

        # Create prompt
        prompt = self._create_generation_prompt(enriched_context, selected_theme)

        try:
            # Call Ollama
            response = await self._call_ollama(prompt)

            # Parse JSON response
            location_data = self._parse_llm_response(response)

            return location_data

        except Exception as e:
            logger.error(f"Error in LLM generation: {e}")
            # Return fallback location
            return self._create_fallback_location(enriched_context, selected_theme)

    def _create_generation_prompt(
        self, enriched_context: EnrichedContext, selected_theme: LocationTheme
    ) -> str:
        """Create the prompt for location generation."""

        context = enriched_context.base_context

        # Use template if available
        if self.jinja_env:
            try:
                template = self.jinja_env.get_template("location_prompts.j2")
                return template.render(
                    expansion_point=context.expansion_point,
                    current_location=self._get_current_location_info(context),
                    adjacent_locations=context.adjacent_locations,
                    player_preferences=context.player_preferences,
                    world_theme=selected_theme.name,
                    desired_atmosphere=selected_theme.atmosphere,
                    generation_hints=enriched_context.generation_hints,
                    priority_elements=enriched_context.priority_elements,
                )
            except Exception as e:
                logger.warning(f"Error using template: {e}")

        # Fallback to hardcoded prompt
        return self._create_fallback_prompt(enriched_context, selected_theme)

    def _get_current_location_info(
        self, context: LocationGenerationContext
    ) -> dict[str, Any]:
        """Get information about the current location."""
        location_id = context.expansion_point.location_id
        if location_id in self.world_state.locations:
            location = self.world_state.locations[location_id]
            return {"name": location.name, "description": location.description}
        return {"name": "Unknown Location", "description": "A mysterious place"}

    def _create_fallback_prompt(
        self, enriched_context: EnrichedContext, selected_theme: LocationTheme
    ) -> str:
        """Create a fallback prompt when templates aren't available."""
        context = enriched_context.base_context

        adjacent_desc = "\n".join(
            [
                f"- {adj.direction}: {adj.name} ({adj.theme}) - {adj.short_description}"
                for adj in context.adjacent_locations
            ]
        )

        return f"""You are generating a new location for a text adventure game.

Context:
- Direction from current location: {context.expansion_point.direction}
- Current location: {context.expansion_point.context.get('location_name', 'Unknown')}
- World theme: {selected_theme.name}
- Desired atmosphere: {selected_theme.atmosphere}

Adjacent locations:
{adjacent_desc}

Player preferences:
- Preferred environments: {', '.join(context.player_preferences.environments)}
- Complexity preference: {context.player_preferences.complexity_level}

Generation hints:
{chr(10).join(enriched_context.generation_hints)}

Generate a new location that:
1. Fits naturally with the surrounding areas
2. Matches the {selected_theme.name} theme and {selected_theme.atmosphere} atmosphere
3. Provides appropriate content for the player's experience level
4. Includes 2-3 interesting features or objects
5. Has a compelling reason for existing in this location

Format your response as JSON with these fields:
- name: Location name (2-4 words)
- description: Full location description (2-3 paragraphs)  
- short_description: Brief description for maps/travel
- location_type: Type of location (clearing, crossroads, cave, etc.)
- atmosphere: Emotional tone/feeling
- objects: List of 2-3 notable objects
- potential_npcs: List of 1-2 potential NPCs that would fit
- connections: Suggested connections to other directions
- special_features: Any unique interactive elements

Respond only with the JSON, no additional text."""

    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API with the prompt."""
        try:
            model = self.llm_config.default_model if self.llm_config else "llama3.1:8b"

            response = self.ollama_client.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop": ["</response>", "---"],
                },
            )

            return response.get("response", "")

        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            raise

    def _parse_llm_response(self, response: str) -> dict[str, Any]:
        """Parse the LLM JSON response."""
        try:
            # Extract JSON from response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in LLM response")
                return self._parse_fallback_response(response)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            return self._parse_fallback_response(response)

    def _parse_fallback_response(self, response: str) -> dict[str, Any]:
        """Parse response when JSON parsing fails."""
        # Extract key information from text response
        lines = response.split("\n")

        name = "Generated Location"
        description = response[:200] + "..." if len(response) > 200 else response

        # Try to extract name from first line
        first_line = lines[0].strip()
        if len(first_line) < 50 and not first_line.startswith("{"):
            name = first_line

        return {
            "name": name,
            "description": description,
            "short_description": name,
            "location_type": "generic",
            "atmosphere": "mysterious",
            "objects": ["unknown object", "mysterious item"],
            "potential_npcs": ["stranger"],
            "connections": {},
            "special_features": ["unexplored area"],
        }

    def _create_fallback_location(
        self, enriched_context: EnrichedContext, selected_theme: LocationTheme
    ) -> dict[str, Any]:
        """Create a fallback location when LLM generation fails."""
        context = enriched_context.base_context

        direction = context.expansion_point.direction
        theme_name = selected_theme.name.lower()

        name = f"{selected_theme.name} {direction.title()}"

        return {
            "name": name,
            "description": f"A {theme_name} area to the {direction}. {selected_theme.description}",
            "short_description": f"A {theme_name} area",
            "location_type": "generic",
            "atmosphere": selected_theme.atmosphere,
            "objects": selected_theme.typical_objects[:2],
            "potential_npcs": selected_theme.typical_npcs[:1],
            "connections": {},
            "special_features": ["newly discovered area"],
        }

    async def validate_location_consistency(
        self, location: GeneratedLocation, adjacent_themes: list[LocationTheme]
    ) -> ValidationResult:
        """Validate location consistency with adjacent areas."""
        start_time = time.time()

        try:
            logger.debug(f"Validating consistency for location: {location.name}")

            issues = []
            suggestions = []
            scores = {
                "thematic": 0.0,
                "logical": 0.0,
                "complexity": 0.0,
                "uniqueness": 0.0,
                "overall": 0.0,
            }

            # Theme consistency check
            theme_consistent = await self.theme_manager.validate_theme_consistency(
                location.theme, adjacent_themes
            )

            if theme_consistent:
                scores["thematic"] = 8.0
            else:
                scores["thematic"] = 4.0
                issues.append("Theme may not be consistent with adjacent areas")
                suggestions.append(
                    "Consider adjusting theme or adding transitional elements"
                )

            # Logical placement check
            if location.name and len(location.name.strip()) > 0:
                scores["logical"] = 7.0
            else:
                scores["logical"] = 3.0
                issues.append("Location name is missing or invalid")

            if location.description and len(location.description) >= 50:
                scores["logical"] += 1.0
            else:
                issues.append("Location description is too short or missing")
                suggestions.append("Add more descriptive detail to the location")

            # Complexity check
            object_count = len(location.objects)
            npc_count = len(location.npcs)

            if 1 <= object_count <= 5 and 0 <= npc_count <= 3:
                scores["complexity"] = 7.0
            else:
                scores["complexity"] = 5.0
                if object_count > 5:
                    issues.append("Too many objects for a single location")
                if npc_count > 3:
                    issues.append("Too many NPCs for a single location")

            # Uniqueness check (basic implementation)
            if location.special_features:
                scores["uniqueness"] = 7.0
            else:
                scores["uniqueness"] = 5.0
                suggestions.append(
                    "Consider adding unique features to make location memorable"
                )

            # Calculate overall score
            scores["overall"] = sum(scores.values()) / len(scores)

            # Determine if valid
            is_valid = scores["overall"] >= 6.0 and len(issues) <= 2
            approval = scores["overall"] >= 7.0

            validation_time_ms = int((time.time() - start_time) * 1000)

            return ValidationResult(
                is_valid=is_valid,
                issues=issues,
                suggestions=suggestions,
                confidence_score=min(scores["overall"] / 10.0, 1.0),
                overall_score=scores["overall"],
                thematic_score=scores["thematic"],
                logical_score=scores["logical"],
                complexity_score=scores["complexity"],
                uniqueness_score=scores["uniqueness"],
                approval=approval,
            )

        except Exception as e:
            logger.error(f"Error validating location consistency: {e}")
            return ValidationResult(
                is_valid=False,
                issues=[f"Validation error: {str(e)}"],
                suggestions=["Manual review recommended"],
                confidence_score=0.0,
            )

    async def enrich_location_context(
        self, base_context: LocationGenerationContext
    ) -> EnrichedContext:
        """Enhance context with player history and world themes."""
        return await self.context_collector.enrich_context(base_context)

    async def generate_location_connections(
        self, location: GeneratedLocation, boundary_point: ExpansionPoint
    ) -> list[LocationConnection]:
        """Generate appropriate connections for the new location."""
        connections = []

        try:
            # Create the primary connection back to the expansion point
            primary_connection = LocationConnection(
                from_location_id=UUID(str(boundary_point.location_id)),
                to_location_id=UUID(
                    "00000000-0000-0000-0000-000000000000"
                ),  # Will be set when location is stored
                direction=boundary_point.direction,
                connection_type="normal",
                description=f"Path {boundary_point.direction} to {location.name}",
                is_bidirectional=True,
            )
            connections.append(primary_connection)

            # Generate additional connections based on location type
            additional_connections = self._generate_additional_connections(location)
            connections.extend(additional_connections)

            logger.debug(
                f"Generated {len(connections)} connections for location {location.name}"
            )
            return connections

        except Exception as e:
            logger.error(f"Error generating location connections: {e}")
            return []

    def _generate_additional_connections(
        self, location: GeneratedLocation
    ) -> list[LocationConnection]:
        """Generate additional connections based on location characteristics."""
        additional = []

        # Based on location type, suggest additional connection possibilities
        if location.location_type == "crossroads":
            # Crossroads might have multiple paths
            suggested_directions = ["north", "south", "east", "west"]
        elif location.location_type == "clearing":
            # Clearings might have 2-3 paths
            suggested_directions = ["north", "east"]
        elif location.location_type == "cave":
            # Caves might have deeper passages
            suggested_directions = ["down", "in"]
        else:
            # Generic locations might have 1-2 additional paths
            suggested_directions = ["north"]

        # Note: These would be expansion points, not actual connections
        # The actual connection creation would happen when these areas are generated

        return additional

    def get_generation_metrics(self) -> list[GenerationMetrics]:
        """Get generation performance metrics."""
        return self._generation_metrics.copy()

    def clear_metrics(self) -> None:
        """Clear collected metrics."""
        self._generation_metrics.clear()
        logger.debug("Generation metrics cleared")
