"""
World Connection Manager for World Connection Management System.

This module serves as the main orchestrator for connection generation, validation,
and graph management with LLM integration.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from jinja2 import Environment, TemplateNotFound

from game_loop.core.models.connection_models import (
    ConnectionGenerationContext,
    ConnectionProperties,
    ConnectionValidationResult,
    GeneratedConnection,
    WorldConnectivityGraph,
)
from game_loop.core.world.connection_context_collector import ConnectionContextCollector
from game_loop.core.world.connection_theme_manager import ConnectionThemeManager
from game_loop.database.session_factory import DatabaseSessionFactory
from game_loop.llm.ollama.client import OllamaClient
from game_loop.state.models import WorldState

logger = logging.getLogger(__name__)


class WorldConnectionManager:
    """Main orchestrator for connection generation, validation, and graph management."""

    def __init__(
        self,
        world_state: WorldState,
        session_factory: DatabaseSessionFactory,
        llm_client: OllamaClient,
        template_env: Environment,
    ):
        """Initialize connection manager with dependencies."""
        self.world_state = world_state
        self.session_factory = session_factory
        self.llm_client = llm_client
        self.template_env = template_env

        # Initialize component managers
        self.theme_manager = ConnectionThemeManager(world_state, session_factory)
        self.context_collector = ConnectionContextCollector(
            world_state, session_factory
        )

        # Initialize connectivity graph
        self.connectivity_graph = WorldConnectivityGraph(nodes={}, edges={})

        # Generation cache
        self._generation_cache: dict[tuple[UUID, UUID], GeneratedConnection] = {}

    async def generate_connection(
        self,
        source_location_id: UUID,
        target_location_id: UUID,
        purpose: str = "expand_world",
    ) -> GeneratedConnection:
        """Generate intelligent connection between locations."""
        start_time = datetime.now()

        try:
            logger.info(
                f"Generating connection: {source_location_id} -> {target_location_id} ({purpose})"
            )

            # Check cache first
            cache_key = (source_location_id, target_location_id)
            if cache_key in self._generation_cache:
                logger.info("Returning cached connection")
                return self._generation_cache[cache_key]

            # Collect generation context
            context = await self.context_collector.collect_generation_context(
                source_location_id, target_location_id, purpose
            )

            # Determine appropriate connection type
            connection_type = await self.theme_manager.determine_connection_type(
                context
            )

            # Create connection properties
            properties = await self.create_connection_properties(
                connection_type, context
            )

            # Generate detailed description using LLM
            enhanced_description = await self.generate_connection_description(
                properties, context
            )
            properties.description = enhanced_description

            # Create the generated connection
            generated_connection = GeneratedConnection(
                source_location_id=source_location_id,
                target_location_id=target_location_id,
                properties=properties,
                metadata={
                    "generation_purpose": purpose,
                    "connection_type": connection_type,
                    "generation_context": self._serialize_context(context),
                },
                generation_timestamp=start_time,
            )

            # Validate the connection
            validation_result = await self.validate_connection(
                generated_connection, context
            )

            # Add validation results to metadata
            generated_connection.metadata["validation_result"] = {
                "is_valid": validation_result.is_valid,
                "consistency_score": validation_result.consistency_score,
                "warnings": validation_result.warnings,
            }

            # Cache the connection
            self._generation_cache[cache_key] = generated_connection

            generation_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.info(
                f"Generated connection in {generation_time:.1f}ms: {properties.connection_type}"
            )

            return generated_connection

        except Exception as e:
            logger.error(f"Error generating connection: {e}")
            # Return a basic fallback connection
            return self._create_fallback_connection(
                source_location_id, target_location_id, purpose
            )

    async def create_connection_properties(
        self, connection_type: str, context: ConnectionGenerationContext
    ) -> ConnectionProperties:
        """Create detailed connection properties."""
        try:
            # Get archetype as base
            archetype = self.theme_manager.get_connection_archetype(connection_type)
            if archetype:
                base_properties = archetype.typical_properties
            else:
                # Create basic properties if no archetype found
                base_properties = ConnectionProperties(
                    connection_type=connection_type,
                    difficulty=2,
                    travel_time=60,
                    description=f"A {connection_type} connecting the areas",
                    visibility="visible",
                    requirements=[],
                )

            # Enhance properties based on context
            enhanced_properties = self._enhance_properties_with_context(
                base_properties, context
            )

            return enhanced_properties

        except Exception as e:
            logger.error(f"Error creating connection properties: {e}")
            return ConnectionProperties(
                connection_type=connection_type,
                difficulty=2,
                travel_time=60,
                description=f"A {connection_type} between the locations",
                visibility="visible",
                requirements=[],
            )

    async def validate_connection(
        self, connection: GeneratedConnection, context: ConnectionGenerationContext
    ) -> ConnectionValidationResult:
        """Validate connection for consistency and logic."""
        try:
            errors = []
            warnings = []

            # Basic validation
            if not connection.properties.description:
                errors.append("Connection description is empty")

            if connection.properties.travel_time <= 0:
                errors.append("Travel time must be positive")

            if not (1 <= connection.properties.difficulty <= 10):
                errors.append("Difficulty must be between 1 and 10")

            # Theme consistency validation
            consistency_score = await self._validate_theme_consistency(
                connection, context
            )
            if consistency_score < 0.5:
                warnings.append("Connection may not be thematically consistent")

            # Logical soundness validation
            logical_score = await self._validate_logical_soundness(connection, context)
            if logical_score < 0.6:
                warnings.append("Connection logic may be questionable")

            # Terrain compatibility validation
            terrain_score = await self._validate_terrain_compatibility(
                connection, context
            )
            if terrain_score < 0.4:
                warnings.append("Connection may not suit the terrain")

            # Special requirements validation
            if connection.properties.requirements:
                req_validation = self._validate_requirements(
                    connection.properties.requirements
                )
                if not req_validation:
                    warnings.append("Some connection requirements may be problematic")

            # Create validation result
            is_valid = len(errors) == 0
            result = ConnectionValidationResult(
                is_valid=is_valid,
                validation_errors=errors,
                warnings=warnings,
                consistency_score=consistency_score,
                logical_soundness=logical_score,
                terrain_compatibility=terrain_score,
            )

            return result

        except Exception as e:
            logger.error(f"Error validating connection: {e}")
            return ConnectionValidationResult(
                is_valid=False,
                validation_errors=[f"Validation error: {e}"],
                consistency_score=0.5,
                logical_soundness=0.5,
                terrain_compatibility=0.5,
            )

    async def update_world_graph(self, connection: GeneratedConnection) -> bool:
        """Update world connectivity graph with new connection."""
        try:
            # Add connection to the graph
            self.connectivity_graph.add_connection(connection)

            # Update world state if needed
            source_location = self.world_state.locations.get(
                connection.source_location_id
            )
            target_location = self.world_state.locations.get(
                connection.target_location_id
            )

            if source_location and target_location:
                # For now, just ensure the locations exist in the graph nodes
                self.connectivity_graph.nodes[connection.source_location_id] = {
                    "name": source_location.name,
                    "theme": source_location.state_flags.get("theme", "Unknown"),
                }
                self.connectivity_graph.nodes[connection.target_location_id] = {
                    "name": target_location.name,
                    "theme": target_location.state_flags.get("theme", "Unknown"),
                }

            logger.info(
                f"Updated world graph with connection: {connection.connection_id}"
            )
            return True

        except Exception as e:
            logger.error(f"Error updating world graph: {e}")
            return False

    async def find_connection_opportunities(
        self, location_id: UUID
    ) -> list[tuple[UUID, float]]:
        """Find potential connection targets with suitability scores."""
        try:
            opportunities: list[tuple[UUID, float]] = []
            source_location = self.world_state.locations.get(location_id)

            if not source_location:
                return opportunities

            # Check all other locations as potential targets
            for target_id, target_location in self.world_state.locations.items():
                if target_id == location_id:
                    continue

                # Check if connection already exists
                existing_connections = self.connectivity_graph.get_connections_from(
                    location_id
                )
                if any(
                    conn.target_location_id == target_id
                    for conn in existing_connections
                ):
                    continue

                # Calculate suitability score
                suitability = await self._calculate_connection_suitability(
                    source_location, target_location
                )

                if suitability > 0.3:  # Minimum threshold
                    opportunities.append((target_id, suitability))

            # Sort by suitability score (highest first)
            opportunities.sort(key=lambda x: x[1], reverse=True)

            return opportunities[:10]  # Return top 10 opportunities

        except Exception as e:
            logger.error(f"Error finding connection opportunities: {e}")
            return []

    async def generate_connection_description(
        self, properties: ConnectionProperties, context: ConnectionGenerationContext
    ) -> str:
        """Generate detailed connection description using LLM."""
        try:
            # Try to load and render the template
            try:
                template = self.template_env.get_template(
                    "connection_generation/connection_prompts.j2"
                )
                prompt = template.render(
                    context=context,
                    connection_type=properties.connection_type,
                    properties=properties,
                )
            except TemplateNotFound:
                # Fallback to built-in prompt
                prompt = self._create_fallback_prompt(properties, context)

            # Generate description with LLM
            try:
                response = await self.llm_client.generate_completion(
                    prompt=prompt,
                    model="qwen2.5:3b",
                    temperature=0.7,
                    max_tokens=200,
                )

                # Parse response (should be a dict from generate_completion)
                if isinstance(response, dict):
                    # Try to extract from response content
                    response_text = response.get("response", "")
                    if response_text:
                        try:
                            # Try to parse as JSON if it looks like JSON
                            if response_text.startswith("{"):
                                parsed_response = json.loads(response_text)
                                if "description" in parsed_response:
                                    return str(parsed_response["description"])
                            return str(response_text)[
                                :200
                            ]  # Limit length and ensure string
                        except json.JSONDecodeError:
                            return str(response_text)[:200]

                return str(response)[:200]

            except Exception as llm_error:
                logger.warning(f"LLM description generation failed: {llm_error}")
                # Fall back to theme manager
                return await self.theme_manager.generate_theme_appropriate_description(
                    properties.connection_type, context
                )

        except Exception as e:
            logger.error(f"Error generating connection description: {e}")
            return f"A {properties.connection_type} connecting {context.source_location.name} and {context.target_location.name}."

    def _enhance_properties_with_context(
        self,
        base_properties: ConnectionProperties,
        context: ConnectionGenerationContext,
    ) -> ConnectionProperties:
        """Enhance connection properties based on generation context."""
        try:
            enhanced = ConnectionProperties(
                connection_type=base_properties.connection_type,
                difficulty=base_properties.difficulty,
                travel_time=base_properties.travel_time,
                description=base_properties.description,
                visibility=base_properties.visibility,
                requirements=base_properties.requirements.copy(),
                reversible=base_properties.reversible,
                condition_flags=base_properties.condition_flags.copy(),
                special_features=base_properties.special_features.copy(),
            )

            # Adjust difficulty based on context
            narrative_context = context.narrative_context
            difficulty_pref = narrative_context.get("difficulty_preference", "medium")

            if difficulty_pref == "high":
                enhanced.difficulty = min(enhanced.difficulty + 2, 10)
            elif difficulty_pref == "low":
                enhanced.difficulty = max(enhanced.difficulty - 1, 1)

            # Adjust travel time based on distance preference
            if context.distance_preference == "short":
                enhanced.travel_time = max(enhanced.travel_time // 2, 10)
            elif context.distance_preference == "long":
                enhanced.travel_time = min(enhanced.travel_time * 2, 300)

            # Add special requirements based on narrative context
            special_reqs = narrative_context.get("special_requirements", [])
            for req in special_reqs:
                if req not in enhanced.requirements:
                    enhanced.requirements.append(req)

            # Adjust visibility based on purpose
            if context.generation_purpose == "exploration":
                if enhanced.visibility == "visible":
                    enhanced.visibility = "partially_hidden"

            return enhanced

        except Exception as e:
            logger.error(f"Error enhancing properties: {e}")
            return base_properties

    async def _validate_theme_consistency(
        self, connection: GeneratedConnection, context: ConnectionGenerationContext
    ) -> float:
        """Validate theme consistency of the connection."""
        try:
            source_theme = context.source_location.state_flags.get("theme", "Unknown")
            target_theme = context.target_location.state_flags.get("theme", "Unknown")

            # Get terrain compatibility
            terrain_compat = await self.theme_manager.get_terrain_compatibility(
                source_theme.lower(), target_theme.lower()
            )

            # Check if connection type is appropriate for themes
            available_types = await self.theme_manager.get_available_connection_types(
                source_theme, target_theme
            )

            type_score = (
                1.0 if connection.properties.connection_type in available_types else 0.5
            )

            # Combine scores
            consistency_score = (terrain_compat + type_score) / 2

            return min(consistency_score, 1.0)

        except Exception as e:
            logger.error(f"Error validating theme consistency: {e}")
            return 0.5

    async def _validate_logical_soundness(
        self, connection: GeneratedConnection, context: ConnectionGenerationContext
    ) -> float:
        """Validate logical soundness of the connection."""
        try:
            score = 1.0

            # Check travel time vs difficulty relationship
            if (
                connection.properties.difficulty > 5
                and connection.properties.travel_time < 30
            ):
                score -= 0.2  # High difficulty should take more time

            # Check requirements make sense
            if "magical_attunement" in connection.properties.requirements:
                if connection.properties.connection_type not in [
                    "portal",
                    "teleporter",
                ]:
                    score -= 0.3  # Magical requirements for non-magical connections

            # Check visibility vs difficulty
            if (
                connection.properties.visibility == "hidden"
                and connection.properties.difficulty < 3
            ):
                score -= 0.1  # Hidden connections should be somewhat difficult

            return max(score, 0.0)

        except Exception as e:
            logger.error(f"Error validating logical soundness: {e}")
            return 0.5

    async def _validate_terrain_compatibility(
        self, connection: GeneratedConnection, context: ConnectionGenerationContext
    ) -> float:
        """Validate terrain compatibility of the connection."""
        try:
            # Get terrain constraints from context
            terrain_constraints = context.terrain_constraints
            source_terrain = terrain_constraints.get("source_terrain", "generic")
            target_terrain = terrain_constraints.get("target_terrain", "generic")

            # Calculate base compatibility
            compatibility = await self.theme_manager.get_terrain_compatibility(
                source_terrain, target_terrain
            )

            # Check if connection type suits the terrain
            connection_type = connection.properties.connection_type
            archetype = self.theme_manager.get_connection_archetype(connection_type)

            if archetype:
                source_affinity = archetype.terrain_affinities.get(source_terrain, 0.5)
                target_affinity = archetype.terrain_affinities.get(target_terrain, 0.5)
                type_compatibility = (source_affinity + target_affinity) / 2
            else:
                type_compatibility = 0.5

            # Combine compatibility scores
            overall_score = (compatibility + type_compatibility) / 2

            return min(overall_score, 1.0)

        except Exception as e:
            logger.error(f"Error validating terrain compatibility: {e}")
            return 0.5

    def _validate_requirements(self, requirements: list[str]) -> bool:
        """Validate that requirements are reasonable."""
        # Check for conflicting requirements
        if "magical_attunement" in requirements and "no_magic" in requirements:
            return False

        # Check for overly restrictive requirements
        if len(requirements) > 3:
            return False

        return True

    async def _calculate_connection_suitability(
        self, source_location, target_location
    ) -> float:
        """Calculate suitability score for connecting two locations."""
        try:
            score = 0.0

            # Theme compatibility
            source_theme = source_location.state_flags.get("theme", "Unknown")
            target_theme = target_location.state_flags.get("theme", "Unknown")

            terrain_compat = await self.theme_manager.get_terrain_compatibility(
                source_theme.lower(), target_theme.lower()
            )
            score += terrain_compat * 0.4

            # Check if both locations need more connections
            source_connections = len(source_location.connections)
            target_connections = len(target_location.connections)

            if source_connections < 3:
                score += 0.2
            if target_connections < 3:
                score += 0.2

            # Prefer connecting different themes for variety
            if source_theme != target_theme:
                score += 0.2

            return min(score, 1.0)

        except Exception as e:
            logger.error(f"Error calculating connection suitability: {e}")
            return 0.3

    def _create_fallback_connection(
        self, source_location_id: UUID, target_location_id: UUID, purpose: str
    ) -> GeneratedConnection:
        """Create a basic fallback connection."""
        properties = ConnectionProperties(
            connection_type="passage",
            difficulty=2,
            travel_time=60,
            description="A simple passage connects the two areas.",
            visibility="visible",
            requirements=[],
        )

        return GeneratedConnection(
            source_location_id=source_location_id,
            target_location_id=target_location_id,
            properties=properties,
            metadata={
                "generation_purpose": purpose,
                "fallback": True,
            },
        )

    def _create_fallback_prompt(
        self, properties: ConnectionProperties, context: ConnectionGenerationContext
    ) -> str:
        """Create a fallback prompt when template is not available."""
        return f"""Generate a detailed description for a {properties.connection_type} connecting two locations.

Source Location: {context.source_location.name}
Description: {context.source_location.description}

Target Location: {context.target_location.name}
Description: {context.target_location.description}

Create a vivid, immersive description that captures the essence of this {properties.connection_type}.
Focus on atmospheric details and how it feels to traverse this connection.

Return only the description text, no JSON or additional formatting."""

    def _serialize_context(
        self, context: ConnectionGenerationContext
    ) -> dict[str, Any]:
        """Serialize context for metadata storage."""
        return {
            "source_location_name": context.source_location.name,
            "target_location_name": context.target_location.name,
            "generation_purpose": context.generation_purpose,
            "distance_preference": context.distance_preference,
            "player_level": context.player_level,
        }
