"""
Dynamic World Coordinator for Game Loop Integration.

Main orchestrator that coordinates all dynamic generation systems and 
integrates them with the game loop.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from game_loop.core.world.content_discovery_tracker import ContentDiscoveryTracker
from game_loop.core.world.generation_quality_monitor import GenerationQualityMonitor
from game_loop.core.world.generation_trigger_manager import GenerationTriggerManager
from game_loop.core.world.player_history_analyzer import PlayerHistoryAnalyzer
from game_loop.core.world.world_generation_pipeline import WorldGenerationPipeline
from game_loop.state.models import (
    ActionResult,
    GeneratedContent,
    GenerationContext,
    GenerationOpportunity,
    GenerationPipelineResult,
    GenerationPlan,
    GenerationTrigger,
    Location,
    PlayerState,
    QualityFeedback,
    QualityReport,
    WorldGenerationResponse,
    WorldGenerationStatus,
    WorldState,
)

logger = logging.getLogger(__name__)


class DynamicWorldCoordinator:
    """
    Main orchestrator for dynamic world generation integration.
    
    This class coordinates:
    - All dynamic generation systems (locations, NPCs, objects, connections)
    - Player action analysis and response
    - Generation trigger management and prioritization
    - Quality monitoring and improvement
    - Resource management and optimization
    """

    def __init__(self, world_state: WorldState, session_factory: Any, llm_client: Any, template_env: Any):
        """Initialize coordinator with all generation systems."""
        self.world_state = world_state
        self.session_factory = session_factory
        self.llm_client = llm_client
        self.template_env = template_env
        
        # Initialize subsystems
        self.trigger_manager = GenerationTriggerManager(world_state, session_factory)
        self.history_analyzer = PlayerHistoryAnalyzer(session_factory)
        self.discovery_tracker = ContentDiscoveryTracker(session_factory)
        self.quality_monitor = GenerationQualityMonitor(session_factory)
        
        # Initialize pipeline (will be set by game loop)
        self.generation_pipeline: WorldGenerationPipeline | None = None
        
        # Coordinator state
        self.generation_queue: list[GenerationOpportunity] = []
        self.active_generations: dict[str, Any] = {}
        self.generation_statistics: dict[str, Any] = {
            "total_generated": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "avg_generation_time": 0.0,
            "last_generation": None,
        }
        
        # Configuration
        self.max_concurrent_generations = 2
        self.generation_cooldown = timedelta(minutes=5)
        self.quality_threshold = 0.6
        self.player_satisfaction_threshold = 3  # Out of 5

    def set_generation_pipeline(self, pipeline: WorldGenerationPipeline) -> None:
        """Set the generation pipeline (called by game loop during initialization)."""
        self.generation_pipeline = pipeline

    async def process_player_action(
        self, action_result: ActionResult, player_state: PlayerState
    ) -> WorldGenerationResponse:
        """
        Process player action and trigger appropriate world generation.
        
        Args:
            action_result: Result of the player's action
            player_state: Current player state
            
        Returns:
            WorldGenerationResponse with any generated content
        """
        try:
            response = WorldGenerationResponse()
            start_time = datetime.now()
            
            logger.info(f"Processing player action for dynamic world generation")
            
            # Analyze action for generation triggers
            triggers = await self.trigger_manager.analyze_action_for_triggers(
                action_result, player_state
            )
            
            if not triggers:
                logger.debug("No generation triggers identified from player action")
                return response
            
            # Evaluate generation opportunities
            current_location = None
            if player_state.current_location_id is not None:
                current_location = self.world_state.locations.get(player_state.current_location_id)
            
            context = GenerationContext(
                player_state=player_state,
                current_location=current_location,
                recent_actions=[action_result.command] if action_result.command else [],
            )
            
            opportunities = await self.evaluate_generation_opportunities(context)
            
            # Filter opportunities based on triggers
            relevant_opportunities = [
                opp for opp in opportunities 
                if any(self._opportunity_matches_trigger(opp, trigger) for trigger in triggers)
            ]
            
            if not relevant_opportunities:
                logger.debug("No relevant generation opportunities found")
                return response
            
            # Check if we should proceed with generation
            if not await self._should_generate_now(player_state, relevant_opportunities):
                logger.debug("Generation conditions not met, queuing for later")
                self.generation_queue.extend(relevant_opportunities)
                return response
            
            # Execute coordinated generation
            pipeline_result = await self.coordinate_generation_pipeline(relevant_opportunities)
            
            if pipeline_result.success:
                response.has_new_content = True
                response.generated_content = pipeline_result.generated_content
                response.generation_time = pipeline_result.pipeline_time
                response.quality_scores = {
                    "consistency": pipeline_result.consistency_score,
                    "coordination": pipeline_result.coordination_quality,
                }
                response.integration_required = True
                
                # Update statistics
                self.generation_statistics["total_generated"] += len(pipeline_result.generated_content)
                self.generation_statistics["successful_generations"] += 1
                self.generation_statistics["last_generation"] = datetime.now()
            else:
                self.generation_statistics["failed_generations"] += 1
                logger.warning("Generation pipeline failed")
            
            # Record timing
            total_time = (datetime.now() - start_time).total_seconds()
            response.generation_time = total_time
            
            logger.info(f"Dynamic world generation completed in {total_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing player action for world generation: {e}")
            return WorldGenerationResponse()

    async def evaluate_generation_opportunities(
        self, context: GenerationContext
    ) -> list[GenerationOpportunity]:
        """
        Evaluate what content should be generated based on current context.
        
        Args:
            context: Current generation context
            
        Returns:
            List of generation opportunities sorted by priority
        """
        try:
            opportunities = []
            
            # Get player preferences to inform generation
            player_preferences = await self.history_analyzer.analyze_player_preferences(
                context.player_state.player_id
            )
            
            # Predict player interests
            interest_predictions = await self.history_analyzer.predict_player_interests(
                context.player_state, {"current_location": context.current_location}
            )
            
            # Evaluate world gaps
            if context.current_location:
                world_gaps = await self.trigger_manager.evaluate_world_gaps(
                    context.current_location.location_id
                )
                
                # Convert gaps to opportunities
                for gap in world_gaps:
                    opportunity = GenerationOpportunity(
                        content_type=self._gap_to_content_type(gap.gap_type),
                        opportunity_score=gap.player_impact,
                        generation_context={
                            "gap_type": gap.gap_type,
                            "location_id": gap.location_id,
                            "suggested_content": gap.suggested_content,
                            "severity": gap.severity,
                        },
                    )
                    opportunities.append(opportunity)
            
            # Add opportunities based on player interests
            for prediction in interest_predictions:
                if prediction.interest_score > 0.6:  # High interest threshold
                    opportunity = GenerationOpportunity(
                        content_type=prediction.content_type,
                        opportunity_score=prediction.interest_score * prediction.confidence,
                        generation_context={
                            "player_preference": True,
                            "interest_score": prediction.interest_score,
                            "reasoning": prediction.reasoning,
                        },
                    )
                    opportunities.append(opportunity)
            
            # Evaluate exploration opportunities
            if context.current_location:
                exploration_opportunities = await self._evaluate_exploration_opportunities(
                    context.current_location, context.player_state
                )
                opportunities.extend(exploration_opportunities)
            
            # Sort by opportunity score
            opportunities.sort(key=lambda x: x.opportunity_score, reverse=True)
            
            # Limit to reasonable number
            opportunities = opportunities[:5]
            
            logger.info(f"Evaluated {len(opportunities)} generation opportunities")
            return opportunities
            
        except Exception as e:
            logger.error(f"Error evaluating generation opportunities: {e}")
            return []

    async def coordinate_generation_pipeline(
        self, opportunities: list[GenerationOpportunity]
    ) -> GenerationPipelineResult:
        """
        Coordinate multiple generators to create cohesive content.
        
        Args:
            opportunities: List of generation opportunities
            
        Returns:
            GenerationPipelineResult with coordinated content
        """
        try:
            if not self.generation_pipeline:
                logger.error("Generation pipeline not initialized")
                return GenerationPipelineResult(success=False, error_messages=["Pipeline not initialized"])
            
            # Convert opportunities to generation plan
            generation_plan = await self._create_generation_plan(opportunities)
            
            # Execute the plan
            pipeline_result = await self.generation_pipeline.execute_generation_plan(generation_plan)
            
            # Monitor quality of generated content
            if pipeline_result.success and pipeline_result.generated_content:
                await self._monitor_generated_content_quality(pipeline_result.generated_content)
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Error coordinating generation pipeline: {e}")
            return GenerationPipelineResult(
                success=False,
                error_messages=[str(e)],
            )

    async def monitor_generation_quality(
        self, generated_content: list[GeneratedContent]
    ) -> QualityReport:
        """
        Monitor and assess quality of generated content.
        
        Args:
            generated_content: List of generated content to monitor
            
        Returns:
            QualityReport with quality assessment
        """
        try:
            # Assess quality of each piece of content
            quality_assessments = []
            for content in generated_content:
                assessment = await self.quality_monitor.assess_content_quality(
                    content, {"generation_context": content.generation_metadata}
                )
                quality_assessments.append(assessment)
            
            # Detect quality issues
            quality_issues = await self.quality_monitor.detect_quality_issues(generated_content)
            
            # Generate quality report
            quality_report = await self.quality_monitor.generate_quality_report()
            
            # Update generation statistics
            avg_quality = sum(a.overall_quality_score for a in quality_assessments) / len(quality_assessments)
            self._update_quality_statistics(avg_quality, quality_issues)
            
            logger.info(f"Monitored quality for {len(generated_content)} items, avg quality: {avg_quality:.2f}")
            return quality_report
            
        except Exception as e:
            logger.error(f"Error monitoring generation quality: {e}")
            return QualityReport(
                report_period=(datetime.now(), datetime.now()),
                overall_quality_score=0.5,
            )

    async def update_generation_preferences(self, quality_feedback: QualityFeedback) -> bool:
        """
        Update generation parameters based on quality feedback.
        
        Args:
            quality_feedback: Feedback on generation quality
            
        Returns:
            True if preferences were updated successfully
        """
        try:
            # Update player preference model
            if quality_feedback.player_id:
                from game_loop.state.models import PlayerFeedback
                player_feedback = PlayerFeedback(
                    player_id=quality_feedback.player_id,
                    content_id=quality_feedback.content_id,
                    feedback_type="quality_rating",
                    feedback_data={"rating": quality_feedback.rating},
                )
                
                await self.history_analyzer.update_preference_model(
                    quality_feedback.player_id, player_feedback
                )
            
            # Update quality thresholds if needed
            if hasattr(quality_feedback, 'rating') and quality_feedback.rating is not None and quality_feedback.rating < 3:
                await self._adjust_quality_thresholds(quality_feedback)
            
            logger.info("Updated generation preferences based on quality feedback")
            return True
            
        except Exception as e:
            logger.error(f"Error updating generation preferences: {e}")
            return False

    async def get_world_generation_status(self) -> WorldGenerationStatus:
        """
        Get current status of world generation systems.
        
        Returns:
            WorldGenerationStatus with system health and metrics
        """
        try:
            status = WorldGenerationStatus()
            
            # Check which generators are active
            active_generators = []
            if self.generation_pipeline:
                active_generators.extend([
                    "location_generator",
                    "npc_generator", 
                    "object_generator",
                    "connection_manager",
                ])
            
            status.active_generators = active_generators
            status.generation_queue_size = len(self.generation_queue)
            
            # Calculate average generation time
            successful_gens = self.generation_statistics.get("successful_generations", 0)
            if successful_gens > 0:
                avg_time = self.generation_statistics.get("avg_generation_time", 0.0)
                status.average_generation_time = avg_time if isinstance(avg_time, (int, float)) else 0.0
            
            # Get recent quality scores
            status.recent_quality_scores = await self._get_recent_quality_scores()
            
            # Determine system health
            failed_gens = self.generation_statistics.get("failed_generations", 0)
            total_gens = self.generation_statistics.get("total_generated", 0)
            if failed_gens == 0:
                status.system_health = "healthy"
            elif (failed_gens / max(1, total_gens)) < 0.1:
                status.system_health = "healthy"
            else:
                status.system_health = "degraded"
            
            status.last_maintenance = datetime.now()  # Mock maintenance time
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting world generation status: {e}")
            return WorldGenerationStatus(system_health="error")

    # Private helper methods

    def _opportunity_matches_trigger(
        self, opportunity: GenerationOpportunity, trigger: "GenerationTrigger"
    ) -> bool:
        """Check if opportunity matches trigger."""
        # Simple matching based on content type and context
        trigger_content_mapping = {
            "location_boundary": "location",
            "exploration": "connection",
            "quest_need": "npc",
            "content_gap": opportunity.content_type,  # Flexible matching
        }
        
        expected_content_type = trigger_content_mapping.get(trigger.trigger_type)
        return expected_content_type == opportunity.content_type

    async def _should_generate_now(
        self, player_state: PlayerState, opportunities: list[GenerationOpportunity]
    ) -> bool:
        """Determine if generation should proceed now."""
        # Check cooldown
        last_generation = self.generation_statistics.get("last_generation")
        if (last_generation and isinstance(last_generation, datetime) and
            datetime.now() - last_generation < self.generation_cooldown):
            return False
        
        # Check if opportunities meet minimum threshold
        if not opportunities or opportunities[0].opportunity_score < 0.5:
            return False
        
        # Check concurrent generation limit
        if len(self.active_generations) >= self.max_concurrent_generations:
            return False
        
        return True

    def _gap_to_content_type(self, gap_type: str) -> str:
        """Convert gap type to content type."""
        gap_mappings = {
            "missing_connection": "connection",
            "empty_location": "object",
            "no_npcs": "npc",
            "no_objects": "object",
        }
        return gap_mappings.get(gap_type, "object")

    async def _evaluate_exploration_opportunities(
        self, location: "Location", player_state: PlayerState
    ) -> list[GenerationOpportunity]:
        """Evaluate opportunities based on exploration patterns."""
        opportunities = []
        
        # Check if location needs more connections for exploration
        if len(location.connections) < 2:
            opportunity = GenerationOpportunity(
                content_type="connection",
                opportunity_score=0.7,
                generation_context={
                    "exploration_opportunity": True,
                    "location_id": location.location_id,
                    "current_connections": len(location.connections),
                },
            )
            opportunities.append(opportunity)
        
        return opportunities

    async def _create_generation_plan(
        self, opportunities: list[GenerationOpportunity]
    ) -> "GenerationPlan":
        """Create generation plan from opportunities."""
        from game_loop.state.models import GenerationPlan
        
        generation_requests = []
        for i, opportunity in enumerate(opportunities):
            request = {
                "request_id": f"req_{i}",
                "content_type": opportunity.content_type,
                "generation_context": opportunity.generation_context,
                "priority": opportunity.opportunity_score,
                "quality_requirements": {"minimum_quality": self.quality_threshold},
                "dependencies": [],
            }
            generation_requests.append(request)
        
        plan = GenerationPlan(
            generation_requests=generation_requests,
            coordination_strategy="mixed",  # Use mixed strategy for flexibility
            estimated_time=len(opportunities) * 2.0,  # Estimate 2 seconds per item
            quality_targets={"overall": self.quality_threshold},
        )
        
        return plan

    async def _monitor_generated_content_quality(self, generated_content: list[dict[str, Any]]) -> None:
        """Monitor quality of generated content."""
        for content_data in generated_content:
            # Convert to GeneratedContent model
            content = GeneratedContent(
                content_type=content_data.get("type", "unknown"),
                content_data=content_data.get("content", {}),
                generation_metadata=content_data.get("metadata", {}),
            )
            
            # Assess quality
            await self.quality_monitor.assess_content_quality(
                content, {"generation_context": content.generation_metadata}
            )

    def _update_quality_statistics(self, avg_quality: float, quality_issues: list) -> None:
        """Update internal quality statistics."""
        # Update average generation time (mock calculation)
        self.generation_statistics["avg_generation_time"] = (
            self.generation_statistics.get("avg_generation_time", 0) * 0.9 + 
            2.0 * 0.1  # Mock 2 second generation time
        )

    async def _adjust_quality_thresholds(self, quality_feedback: "QualityFeedback") -> None:
        """Adjust quality thresholds based on feedback."""
        if hasattr(quality_feedback, 'rating') and quality_feedback.rating is not None and quality_feedback.rating < 3:
            # Lower threshold temporarily to generate more content
            self.quality_threshold = max(0.4, self.quality_threshold - 0.1)
            logger.info(f"Adjusted quality threshold to {self.quality_threshold}")

    async def _get_recent_quality_scores(self) -> list[float]:
        """Get recent quality scores."""
        # Mock implementation - would query actual quality data
        return [0.7, 0.8, 0.75, 0.82, 0.78]