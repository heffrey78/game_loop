"""
World Generation Pipeline for Dynamic World Integration.

Coordinates multiple content generators to create cohesive, interconnected
content that maintains world consistency.
"""

import asyncio
import logging
from typing import Any
from uuid import UUID

from game_loop.core.world.location_generator import LocationGenerator
from game_loop.core.world.npc_generator import NPCGenerator
from game_loop.core.world.object_generator import ObjectGenerator
from game_loop.core.world.world_connection_manager import WorldConnectionManager
from game_loop.state.models import (
    ConsistencyReport,
    ContentCluster,
    GeneratedContent,
    GenerationPipelineResult,
    GenerationPlan,
    GenerationRecovery,
    GenerationRequest,
    Location,
    LocationExpansion,
    LocationWithContent,
)

logger = logging.getLogger(__name__)


class WorldGenerationPipeline:
    """
    Coordinates multiple content generators for cohesive world creation.

    This class orchestrates:
    - Multi-generator coordination for consistent content creation
    - Content dependency management and ordering
    - Consistency validation across generated content
    - Error recovery and fallback strategies
    - Performance optimization for generation workflows
    """

    def __init__(
        self,
        location_generator: LocationGenerator,
        npc_generator: NPCGenerator,
        object_generator: ObjectGenerator,
        connection_manager: WorldConnectionManager,
        session_factory,
    ):
        """Initialize pipeline with all generators."""
        self.location_generator = location_generator
        self.npc_generator = npc_generator
        self.object_generator = object_generator
        self.connection_manager = connection_manager
        self.session_factory = session_factory

        # Pipeline configuration
        self.max_concurrent_generations = 3
        self.generation_timeout = 30.0  # seconds
        self.consistency_threshold = 0.7

        # Generation coordination state
        self.active_generations = {}
        self.generation_queue = []
        self.dependency_graph = {}

    async def execute_generation_plan(
        self, plan: GenerationPlan
    ) -> GenerationPipelineResult:
        """
        Execute coordinated generation plan.

        Args:
            plan: Generation plan with coordinated requests

        Returns:
            GenerationPipelineResult with results and metrics
        """
        try:
            result = GenerationPipelineResult()
            start_time = asyncio.get_event_loop().time()

            logger.info(
                f"Executing generation plan {plan.plan_id} with {len(plan.generation_requests)} requests"
            )

            # Optimize generation order
            optimized_requests = await self.optimize_generation_order(
                plan.generation_requests
            )

            # Execute based on coordination strategy
            if plan.coordination_strategy == "sequential":
                generated_content = await self._execute_sequential(optimized_requests)
            elif plan.coordination_strategy == "parallel":
                generated_content = await self._execute_parallel(optimized_requests)
            else:  # mixed strategy
                generated_content = await self._execute_mixed(optimized_requests)

            result.generated_content = generated_content
            result.success = len(generated_content) > 0

            # Validate consistency
            consistency_report = await self.validate_content_consistency(
                [GeneratedContent(**content) for content in generated_content]
            )
            result.consistency_score = consistency_report.overall_consistency

            # Calculate coordination quality
            result.coordination_quality = await self._calculate_coordination_quality(
                plan, generated_content
            )

            # Record timing
            end_time = asyncio.get_event_loop().time()
            result.pipeline_time = end_time - start_time

            logger.info(
                f"Generation plan completed in {result.pipeline_time:.2f}s with consistency {result.consistency_score:.2f}"
            )
            return result

        except Exception as e:
            logger.error(f"Error executing generation plan: {e}")
            return GenerationPipelineResult(
                success=False,
                error_messages=[str(e)],
            )

    async def generate_location_with_content(
        self, context: dict[str, Any]
    ) -> LocationWithContent:
        """
        Generate location with appropriate NPCs, objects, and connections.

        Args:
            context: Generation context for the location

        Returns:
            LocationWithContent with location and associated content
        """
        try:
            location_with_content = LocationWithContent(
                location=Location(name="Placeholder", description="Placeholder"),
            )

            # Generate the main location
            location_context = context.get("location_context", {})
            generated_location = await self.location_generator.generate_location(
                **location_context
            )
            location_with_content.location = generated_location

            # Generate NPCs for the location
            npc_count = await self._determine_npc_count(generated_location, context)
            for i in range(npc_count):
                npc_context = await self._create_npc_context(
                    generated_location, context, i
                )
                npc = await self.npc_generator.generate_npc(**npc_context)
                location_with_content.generated_npcs.append(npc)

            # Generate objects for the location
            object_count = await self._determine_object_count(
                generated_location, context
            )
            for i in range(object_count):
                object_context = await self._create_object_context(
                    generated_location, context, i
                )
                obj = await self.object_generator.generate_object(**object_context)
                location_with_content.generated_objects.append(obj)

            # Generate connections if requested
            if context.get("generate_connections", True):
                connection_count = await self._determine_connection_count(
                    generated_location, context
                )
                for i in range(connection_count):
                    connection_context = await self._create_connection_context(
                        generated_location, context, i
                    )
                    connection = await self.connection_manager.generate_connection(
                        **connection_context
                    )
                    location_with_content.generated_connections.append(
                        {
                            "connection": connection,
                            "context": connection_context,
                        }
                    )

            # Add generation metadata
            location_with_content.generation_metadata = {
                "generation_strategy": "coordinated",
                "npc_count": len(location_with_content.generated_npcs),
                "object_count": len(location_with_content.generated_objects),
                "connection_count": len(location_with_content.generated_connections),
                "theme": generated_location.state_flags.get("theme"),
            }

            logger.info(
                f"Generated location {generated_location.name} with full content"
            )
            return location_with_content

        except Exception as e:
            logger.error(f"Error generating location with content: {e}")
            # Return minimal viable location
            return LocationWithContent(
                location=Location(
                    name="Generated Location", description="A basic location"
                ),
            )

    async def expand_existing_location(
        self, location_id: UUID, expansion_type: str
    ) -> LocationExpansion:
        """
        Add content to existing location.

        Args:
            location_id: ID of the location to expand
            expansion_type: Type of expansion ("npc_addition", "object_addition", etc.)

        Returns:
            LocationExpansion with added content
        """
        try:
            expansion = LocationExpansion(
                location_id=location_id,
                expansion_type=expansion_type,
            )

            # Get current location state
            location = await self._get_location(location_id)
            if not location:
                logger.warning(f"Location {location_id} not found for expansion")
                return expansion

            # Generate content based on expansion type
            if expansion_type == "npc_addition":
                npc_context = await self._create_expansion_npc_context(location)
                npc = await self.npc_generator.generate_npc(**npc_context)
                expansion.added_content.append(
                    {
                        "type": "npc",
                        "content": npc,
                    }
                )

            elif expansion_type == "object_addition":
                object_context = await self._create_expansion_object_context(location)
                obj = await self.object_generator.generate_object(**object_context)
                expansion.added_content.append(
                    {
                        "type": "object",
                        "content": obj,
                    }
                )

            elif expansion_type == "connection_addition":
                connection_context = await self._create_expansion_connection_context(
                    location
                )
                connection = await self.connection_manager.generate_connection(
                    **connection_context
                )
                expansion.added_content.append(
                    {
                        "type": "connection",
                        "content": connection,
                    }
                )

            # Assess integration quality
            expansion.integration_quality = await self._assess_integration_quality(
                location, expansion.added_content
            )

            logger.info(f"Expanded location {location_id} with {expansion_type}")
            return expansion

        except Exception as e:
            logger.error(f"Error expanding location: {e}")
            return LocationExpansion(
                location_id=location_id,
                expansion_type=expansion_type,
                integration_quality=0.0,
            )

    async def create_content_cluster(
        self, anchor_location_id: UUID, cluster_theme: str
    ) -> ContentCluster:
        """
        Create thematically connected group of content.

        Args:
            anchor_location_id: ID of the anchor location
            cluster_theme: Theme for the content cluster

        Returns:
            ContentCluster with thematically connected content
        """
        try:
            cluster = ContentCluster(
                theme=cluster_theme,
                anchor_location_id=anchor_location_id,
            )

            # Generate cluster content based on theme
            cluster_content = []

            # Generate related locations
            location_count = await self._determine_cluster_location_count(cluster_theme)
            for i in range(location_count):
                location_context = await self._create_cluster_location_context(
                    anchor_location_id, cluster_theme, i
                )
                location = await self.location_generator.generate_location(
                    **location_context
                )
                cluster_content.append(
                    {
                        "type": "location",
                        "content": location,
                        "relationship": "thematic_sibling",
                    }
                )

            # Generate connecting NPCs
            npc_count = await self._determine_cluster_npc_count(cluster_theme)
            for i in range(npc_count):
                npc_context = await self._create_cluster_npc_context(
                    anchor_location_id, cluster_theme, i
                )
                npc = await self.npc_generator.generate_npc(**npc_context)
                cluster_content.append(
                    {
                        "type": "npc",
                        "content": npc,
                        "relationship": "theme_representative",
                    }
                )

            # Generate thematic objects
            object_count = await self._determine_cluster_object_count(cluster_theme)
            for i in range(object_count):
                object_context = await self._create_cluster_object_context(
                    anchor_location_id, cluster_theme, i
                )
                obj = await self.object_generator.generate_object(**object_context)
                cluster_content.append(
                    {
                        "type": "object",
                        "content": obj,
                        "relationship": "thematic_artifact",
                    }
                )

            cluster.cluster_content = cluster_content

            # Calculate coherence score
            cluster.coherence_score = await self._calculate_cluster_coherence(
                cluster_theme, cluster_content
            )

            logger.info(
                f"Created content cluster '{cluster_theme}' with {len(cluster_content)} items"
            )
            return cluster

        except Exception as e:
            logger.error(f"Error creating content cluster: {e}")
            return ContentCluster(
                theme=cluster_theme,
                anchor_location_id=anchor_location_id,
                coherence_score=0.0,
            )

    async def validate_content_consistency(
        self, generated_content: list[GeneratedContent]
    ) -> ConsistencyReport:
        """
        Validate that generated content is internally consistent.

        Args:
            generated_content: List of generated content to validate

        Returns:
            ConsistencyReport with consistency analysis
        """
        try:
            report = ConsistencyReport()

            if not generated_content:
                return report

            # Check theme consistency
            themes = [
                content.generation_metadata.get("theme")
                for content in generated_content
                if content.generation_metadata.get("theme")
            ]

            if themes:
                theme_variety = len(set(themes)) / len(themes)
                report.theme_consistency = 1.0 - min(0.5, theme_variety)

            # Check narrative consistency
            narrative_elements = []
            for content in generated_content:
                if "narrative_elements" in content.generation_metadata:
                    narrative_elements.extend(
                        content.generation_metadata["narrative_elements"]
                    )

            if narrative_elements:
                # Check for conflicting narrative elements
                conflicts = await self._detect_narrative_conflicts(narrative_elements)
                report.narrative_consistency = 1.0 - min(1.0, len(conflicts) * 0.2)
                if conflicts:
                    report.inconsistencies_found.extend(conflicts)

            # Check mechanical consistency
            mechanical_consistency = await self._check_mechanical_consistency(
                generated_content
            )
            report.mechanical_consistency = mechanical_consistency

            # Calculate overall consistency
            scores = [
                report.theme_consistency,
                report.narrative_consistency,
                report.mechanical_consistency,
            ]
            report.overall_consistency = sum(scores) / len(scores)

            logger.info(
                f"Validated consistency for {len(generated_content)} items: {report.overall_consistency:.2f}"
            )
            return report

        except Exception as e:
            logger.error(f"Error validating content consistency: {e}")
            return ConsistencyReport()

    async def optimize_generation_order(
        self, generation_requests: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Optimize order of generation for efficiency and consistency.

        Args:
            generation_requests: List of generation requests

        Returns:
            Optimized list of generation requests
        """
        try:
            # Convert to GenerationRequest objects for easier handling
            requests = [GenerationRequest(**req) for req in generation_requests]

            # Build dependency graph
            dependency_graph = await self._build_dependency_graph(requests)

            # Topological sort for dependency ordering
            ordered_requests = await self._topological_sort(requests, dependency_graph)

            # Further optimize for parallel execution opportunities
            optimized_requests = await self._optimize_for_parallelism(ordered_requests)

            # Convert back to dict format
            result = [
                {
                    "request_id": req.request_id,
                    "content_type": req.content_type,
                    "generation_context": req.generation_context,
                    "priority": req.priority,
                    "dependencies": req.dependencies,
                }
                for req in optimized_requests
            ]

            logger.info(f"Optimized {len(generation_requests)} generation requests")
            return result

        except Exception as e:
            logger.error(f"Error optimizing generation order: {e}")
            return generation_requests  # Return original order on error

    async def handle_generation_failure(
        self, failed_request: dict[str, Any], error: Exception
    ) -> GenerationRecovery:
        """
        Handle failures in generation pipeline.

        Args:
            failed_request: The generation request that failed
            error: The exception that occurred

        Returns:
            GenerationRecovery with recovery strategy
        """
        try:
            recovery = GenerationRecovery(recovery_type="retry")

            error_type = type(error).__name__

            # Determine recovery strategy based on error type
            if "timeout" in str(error).lower():
                recovery.recovery_type = "retry"
                recovery.retry_parameters = {"timeout": 60, "max_retries": 2}

            elif "validation" in str(error).lower():
                recovery.recovery_type = "fallback"
                recovery.fallback_content = await self._create_fallback_content(
                    failed_request["content_type"]
                )

            elif "resource" in str(error).lower():
                recovery.recovery_type = "skip"

            else:
                # Default to retry with modified parameters
                recovery.recovery_type = "retry"
                recovery.retry_parameters = {"simplified": True}

            logger.info(
                f"Generated recovery strategy '{recovery.recovery_type}' for failed request"
            )
            return recovery

        except Exception as e:
            logger.error(f"Error handling generation failure: {e}")
            return GenerationRecovery(recovery_type="skip")

    # Private helper methods

    async def _execute_sequential(
        self, requests: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Execute generation requests sequentially."""
        generated_content = []

        for request in requests:
            try:
                content = await self._execute_single_request(request)
                if content:
                    generated_content.append(content)
            except Exception as e:
                logger.warning(
                    f"Failed to execute request {request.get('request_id')}: {e}"
                )
                # Handle failure
                recovery = await self.handle_generation_failure(request, e)
                if recovery.recovery_type == "fallback" and recovery.fallback_content:
                    generated_content.append(recovery.fallback_content)

        return generated_content

    async def _execute_parallel(
        self, requests: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Execute generation requests in parallel."""
        tasks = [self._execute_single_request(request) for request in requests]

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            generated_content = []

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(
                        f"Parallel execution failed for request {i}: {result}"
                    )
                    # Handle failure
                    recovery = await self.handle_generation_failure(requests[i], result)
                    if (
                        recovery.recovery_type == "fallback"
                        and recovery.fallback_content
                    ):
                        generated_content.append(recovery.fallback_content)
                elif result:
                    generated_content.append(result)

            return generated_content

        except Exception as e:
            logger.error(f"Error in parallel execution: {e}")
            return []

    async def _execute_mixed(
        self, requests: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Execute generation requests with mixed strategy."""
        # Group requests by dependency level
        grouped_requests = await self._group_by_dependency_level(requests)

        generated_content = []

        # Execute each level, parallel within level, sequential between levels
        for level_requests in grouped_requests:
            level_content = await self._execute_parallel(level_requests)
            generated_content.extend(level_content)

        return generated_content

    async def _execute_single_request(
        self, request: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Execute a single generation request."""
        content_type = request["content_type"]
        context = request["generation_context"]

        try:
            if content_type == "location":
                content = await self.location_generator.generate_location(**context)
                return {"type": "location", "content": content}

            elif content_type == "npc":
                content = await self.npc_generator.generate_npc(**context)
                return {"type": "npc", "content": content}

            elif content_type == "object":
                content = await self.object_generator.generate_object(**context)
                return {"type": "object", "content": content}

            elif content_type == "connection":
                content = await self.connection_manager.generate_connection(**context)
                return {"type": "connection", "content": content}

            else:
                logger.warning(f"Unknown content type: {content_type}")
                return None

        except Exception as e:
            logger.error(f"Error executing {content_type} generation: {e}")
            raise

    async def _calculate_coordination_quality(
        self, plan: GenerationPlan, generated_content: list[dict[str, Any]]
    ) -> float:
        """Calculate quality of coordination in generation."""
        if not generated_content:
            return 0.0

        # Check if all requested content was generated
        completion_rate = len(generated_content) / len(plan.generation_requests)

        # Check thematic consistency
        themes = [
            content.get("content", {}).get("theme")
            for content in generated_content
            if content.get("content", {}).get("theme")
        ]

        theme_consistency = 1.0
        if themes:
            unique_themes = len(set(themes))
            theme_consistency = 1.0 - min(0.5, (unique_themes - 1) / len(themes))

        # Combine metrics
        coordination_quality = (completion_rate + theme_consistency) / 2
        return min(1.0, max(0.0, coordination_quality))

    async def _determine_npc_count(
        self, location: Location, context: dict[str, Any]
    ) -> int:
        """Determine appropriate number of NPCs for location."""
        theme = location.state_flags.get("theme", "")

        npc_counts = {
            "City": 3,
            "Village": 2,
            "Forest": 1,
            "Mountain": 1,
            "Dungeon": 2,
        }

        base_count = npc_counts.get(theme, 1)

        # Adjust based on context
        if context.get("high_activity", False):
            base_count += 1

        return min(5, max(0, base_count))

    async def _determine_object_count(
        self, location: Location, context: dict[str, Any]
    ) -> int:
        """Determine appropriate number of objects for location."""
        theme = location.state_flags.get("theme", "")

        object_counts = {
            "City": 4,
            "Village": 3,
            "Forest": 2,
            "Mountain": 2,
            "Dungeon": 3,
        }

        return object_counts.get(theme, 2)

    async def _determine_connection_count(
        self, location: Location, context: dict[str, Any]
    ) -> int:
        """Determine appropriate number of connections for location."""
        # Most locations should have 2-4 connections
        return 2  # Default moderate connectivity

    async def _create_npc_context(
        self, location: Location, context: dict[str, Any], index: int
    ) -> dict[str, Any]:
        """Create context for NPC generation."""
        return {
            "location_id": location.location_id,
            "location_theme": location.state_flags.get("theme"),
            "generation_purpose": f"populate_location_{index}",
            "narrative_context": context.get("narrative_context", {}),
        }

    async def _create_object_context(
        self, location: Location, context: dict[str, Any], index: int
    ) -> dict[str, Any]:
        """Create context for object generation."""
        return {
            "location_id": location.location_id,
            "location_theme": location.state_flags.get("theme"),
            "generation_purpose": f"furnish_location_{index}",
            "object_type": "furniture" if index == 0 else "decoration",
        }

    async def _create_connection_context(
        self, location: Location, context: dict[str, Any], index: int
    ) -> dict[str, Any]:
        """Create context for connection generation."""
        return {
            "source_location_id": location.location_id,
            "target_location_id": None,  # Will be determined by connection manager
            "purpose": "world_expansion",
        }

    # Mock implementations for methods that would require full system integration

    async def _get_location(self, location_id: UUID) -> Location | None:
        """Get location by ID."""
        # Mock implementation
        return None

    async def _create_expansion_npc_context(self, location: Location) -> dict[str, Any]:
        """Create context for expanding location with NPC."""
        return {"location_id": location.location_id, "expansion": True}

    async def _create_expansion_object_context(
        self, location: Location
    ) -> dict[str, Any]:
        """Create context for expanding location with object."""
        return {"location_id": location.location_id, "expansion": True}

    async def _create_expansion_connection_context(
        self, location: Location
    ) -> dict[str, Any]:
        """Create context for expanding location with connection."""
        return {"source_location_id": location.location_id, "expansion": True}

    async def _assess_integration_quality(
        self, location: Location, added_content: list[dict[str, Any]]
    ) -> float:
        """Assess how well new content integrates with existing location."""
        return 0.8  # Mock good integration

    async def _determine_cluster_location_count(self, theme: str) -> int:
        """Determine number of locations for cluster."""
        return 2  # Default cluster size

    async def _determine_cluster_npc_count(self, theme: str) -> int:
        """Determine number of NPCs for cluster."""
        return 1

    async def _determine_cluster_object_count(self, theme: str) -> int:
        """Determine number of objects for cluster."""
        return 2

    async def _create_cluster_location_context(
        self, anchor_id: UUID, theme: str, index: int
    ) -> dict[str, Any]:
        """Create context for cluster location generation."""
        return {"theme": theme, "cluster_anchor": anchor_id}

    async def _create_cluster_npc_context(
        self, anchor_id: UUID, theme: str, index: int
    ) -> dict[str, Any]:
        """Create context for cluster NPC generation."""
        return {"theme": theme, "cluster_role": "theme_representative"}

    async def _create_cluster_object_context(
        self, anchor_id: UUID, theme: str, index: int
    ) -> dict[str, Any]:
        """Create context for cluster object generation."""
        return {"theme": theme, "cluster_role": "thematic_artifact"}

    async def _calculate_cluster_coherence(
        self, theme: str, content: list[dict[str, Any]]
    ) -> float:
        """Calculate coherence score for content cluster."""
        return 0.8  # Mock good coherence

    async def _detect_narrative_conflicts(
        self, narrative_elements: list[str]
    ) -> list[str]:
        """Detect conflicts in narrative elements."""
        return []  # Mock no conflicts

    async def _check_mechanical_consistency(
        self, generated_content: list[GeneratedContent]
    ) -> float:
        """Check mechanical consistency of content."""
        return 0.9  # Mock good mechanical consistency

    async def _build_dependency_graph(
        self, requests: list[GenerationRequest]
    ) -> dict[UUID, list[UUID]]:
        """Build dependency graph for requests."""
        graph = {}
        for request in requests:
            graph[request.request_id] = request.dependencies
        return graph

    async def _topological_sort(
        self,
        requests: list[GenerationRequest],
        dependency_graph: dict[UUID, list[UUID]],
    ) -> list[GenerationRequest]:
        """Perform topological sort on requests."""
        # Simple implementation - in practice would use proper topological sort
        return sorted(requests, key=lambda r: len(r.dependencies))

    async def _optimize_for_parallelism(
        self, ordered_requests: list[GenerationRequest]
    ) -> list[GenerationRequest]:
        """Optimize ordering for parallel execution opportunities."""
        return ordered_requests  # Mock - return as-is

    async def _group_by_dependency_level(
        self, requests: list[dict[str, Any]]
    ) -> list[list[dict[str, Any]]]:
        """Group requests by dependency level."""
        # Mock implementation - single level
        return [requests]

    async def _create_fallback_content(self, content_type: str) -> dict[str, Any]:
        """Create fallback content for failed generation."""
        return {
            "type": content_type,
            "content": {
                "name": f"Fallback {content_type}",
                "description": "Basic fallback content",
            },
            "fallback": True,
        }
