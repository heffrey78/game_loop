"""
Content Discovery Tracker for Dynamic World Integration.

Tracks how players discover and interact with generated content to improve 
future generation.
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from game_loop.state.models import (
    ContentEffectiveness,
    DiscoveryAnalytics,
    DiscoveryEvent,
    DiscoveryPatterns,
    InteractionEvent,
    UndiscoveredContent,
)

logger = logging.getLogger(__name__)


class ContentDiscoveryTracker:
    """
    Tracks content discovery and interaction patterns.
    
    This class monitors:
    - When and how players discover generated content
    - Player interaction patterns with discovered content
    - Content effectiveness and engagement metrics
    - Discovery difficulty and optimization opportunities
    """

    def __init__(self, session_factory):
        """Initialize content discovery tracking system."""
        self.session_factory = session_factory
        self.discovery_cache = {}
        self.interaction_cache = {}
        
        # Discovery method weights for analytics
        self.discovery_method_weights = {
            "exploration": 1.0,
            "quest": 0.9,
            "hint": 0.7,
            "accident": 0.6,
            "guidance": 0.5,
            "search": 0.8,
        }
        
        # Satisfaction thresholds
        self.satisfaction_thresholds = {
            "high": 4,      # Rating 4-5
            "medium": 3,    # Rating 3
            "low": 2,       # Rating 1-2
        }

    async def track_content_discovery(self, discovery_event: DiscoveryEvent) -> bool:
        """
        Track when player discovers generated content.
        
        Args:
            discovery_event: Event describing the discovery
            
        Returns:
            True if tracking was successful
        """
        try:
            # Validate the discovery event
            if not await self._validate_discovery_event(discovery_event):
                logger.warning(f"Invalid discovery event: {discovery_event.discovery_id}")
                return False
            
            # Store discovery event in database
            await self._store_discovery_event(discovery_event)
            
            # Update discovery cache
            cache_key = f"{discovery_event.player_id}_{discovery_event.content_id}"
            self.discovery_cache[cache_key] = discovery_event
            
            # Calculate discovery difficulty if not provided
            if discovery_event.time_to_discovery_seconds is not None:
                difficulty = await self._calculate_discovery_difficulty_from_time(
                    discovery_event.time_to_discovery_seconds,
                    discovery_event.content_type,
                    discovery_event.discovery_method
                )
                await self._update_discovery_difficulty(discovery_event.discovery_id, difficulty)
            
            # Update content effectiveness metrics
            await self._update_content_effectiveness(discovery_event.content_id)
            
            logger.info(f"Tracked discovery of {discovery_event.content_type} content by player {discovery_event.player_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error tracking content discovery: {e}")
            return False

    async def track_content_interaction(self, interaction_event: InteractionEvent) -> bool:
        """
        Track how player interacts with discovered content.
        
        Args:
            interaction_event: Event describing the interaction
            
        Returns:
            True if tracking was successful
        """
        try:
            # Validate the interaction event
            if not await self._validate_interaction_event(interaction_event):
                logger.warning(f"Invalid interaction event: {interaction_event.interaction_id}")
                return False
            
            # Store interaction event in database
            await self._store_interaction_event(interaction_event)
            
            # Update interaction cache
            cache_key = f"{interaction_event.player_id}_{interaction_event.content_id}"
            if cache_key not in self.interaction_cache:
                self.interaction_cache[cache_key] = []
            self.interaction_cache[cache_key].append(interaction_event)
            
            # Update content effectiveness metrics
            await self._update_interaction_effectiveness(interaction_event)
            
            # Update player satisfaction tracking
            if interaction_event.satisfaction_score:
                await self._update_satisfaction_tracking(interaction_event)
            
            logger.info(f"Tracked interaction with {interaction_event.content_id} by player {interaction_event.player_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error tracking content interaction: {e}")
            return False

    async def analyze_discovery_patterns(
        self, content_type: str, timeframe_days: int = 30
    ) -> DiscoveryPatterns:
        """
        Analyze how content is typically discovered.
        
        Args:
            content_type: Type of content to analyze
            timeframe_days: Number of days to analyze
            
        Returns:
            DiscoveryPatterns with analysis results
        """
        try:
            patterns = DiscoveryPatterns()
            
            # Get discovery events for the timeframe
            discovery_events = await self._get_discovery_events(content_type, timeframe_days)
            
            if discovery_events:
                # Calculate average discovery time
                times = [
                    event.time_to_discovery_seconds 
                    for event in discovery_events 
                    if event.time_to_discovery_seconds is not None
                ]
                if times:
                    patterns.average_discovery_time = sum(times) / len(times)
                
                # Analyze discovery methods
                method_counts = defaultdict(int)
                for event in discovery_events:
                    method_counts[event.discovery_method] += 1
                
                total_discoveries = len(discovery_events)
                patterns.common_discovery_methods = [
                    (method, count / total_discoveries)
                    for method, count in method_counts.most_common()
                ]
                
                # Calculate discovery success rate
                successful_discoveries = len([
                    event for event in discovery_events
                    if event.player_satisfaction and event.player_satisfaction >= 3
                ])
                patterns.discovery_success_rate = successful_discoveries / total_discoveries
                
                # Assess player guidance needed
                guided_discoveries = len([
                    event for event in discovery_events
                    if event.discovery_method in ["hint", "guidance"]
                ])
                patterns.player_guidance_needed = guided_discoveries / total_discoveries
            
            logger.info(f"Analyzed discovery patterns for {content_type}")
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing discovery patterns: {e}")
            return DiscoveryPatterns()

    async def get_content_effectiveness(self, content_id: UUID) -> ContentEffectiveness:
        """
        Measure effectiveness of specific generated content.
        
        Args:
            content_id: ID of the content to analyze
            
        Returns:
            ContentEffectiveness with metrics
        """
        try:
            effectiveness = ContentEffectiveness()
            
            # Get all discovery events for this content
            discoveries = await self._get_content_discoveries(content_id)
            total_players_encountered = await self._get_players_who_encountered_content(content_id)
            
            if total_players_encountered > 0:
                # Calculate discovery rate
                effectiveness.discovery_rate = len(discoveries) / total_players_encountered
                
                # Get interaction events
                interactions = await self._get_content_interactions(content_id)
                
                # Calculate interaction rate
                if discoveries:
                    effectiveness.interaction_rate = len(interactions) / len(discoveries)
                
                # Calculate average satisfaction
                satisfactions = [
                    event.player_satisfaction 
                    for event in discoveries 
                    if event.player_satisfaction is not None
                ]
                if satisfactions:
                    effectiveness.average_satisfaction = sum(satisfactions) / len(satisfactions) / 5.0  # Normalize to 0-1
                
                # Calculate completion rate (based on interaction outcomes)
                completed_interactions = len([
                    event for event in interactions
                    if event.interaction_outcome == "success"
                ])
                if interactions:
                    effectiveness.completion_rate = completed_interactions / len(interactions)
                
                # Calculate replay value (repeated interactions)
                player_interaction_counts = defaultdict(int)
                for event in interactions:
                    player_interaction_counts[event.player_id] += 1
                
                repeated_players = len([
                    count for count in player_interaction_counts.values()
                    if count > 1
                ])
                if player_interaction_counts:
                    effectiveness.replay_value = repeated_players / len(player_interaction_counts)
            
            logger.info(f"Calculated effectiveness for content {content_id}")
            return effectiveness
            
        except Exception as e:
            logger.error(f"Error calculating content effectiveness: {e}")
            return ContentEffectiveness()

    async def get_undiscovered_content(
        self, player_id: UUID, location_id: UUID
    ) -> list[UndiscoveredContent]:
        """
        Get content in area that player hasn't discovered.
        
        Args:
            player_id: ID of the player
            location_id: ID of the location to check
            
        Returns:
            List of undiscovered content in the area
        """
        try:
            undiscovered = []
            
            # Get all content in the location
            location_content = await self._get_location_content(location_id)
            
            # Get player's discoveries
            player_discoveries = await self._get_player_discoveries(player_id)
            discovered_content_ids = {d.content_id for d in player_discoveries}
            
            # Find undiscovered content
            for content in location_content:
                if content["content_id"] not in discovered_content_ids:
                    # Calculate difficulty to discover
                    difficulty = await self.calculate_discovery_difficulty(content["content_id"])
                    
                    # Get available hints
                    hints = await self._get_content_hints(content["content_id"])
                    
                    # Calculate time since generation
                    time_since_generation = await self._get_time_since_generation(content["content_id"])
                    
                    undiscovered_content = UndiscoveredContent(
                        content_id=content["content_id"],
                        content_type=content["content_type"],
                        difficulty_to_discover=difficulty,
                        hints_available=hints,
                        time_since_generation=time_since_generation,
                    )
                    undiscovered.append(undiscovered_content)
            
            # Sort by difficulty (easier to discover first)
            undiscovered.sort(key=lambda x: x.difficulty_to_discover)
            
            logger.info(f"Found {len(undiscovered)} undiscovered content items for player {player_id}")
            return undiscovered
            
        except Exception as e:
            logger.error(f"Error getting undiscovered content: {e}")
            return []

    async def calculate_discovery_difficulty(self, content_id: UUID) -> float:
        """
        Calculate how difficult content is to discover.
        
        Args:
            content_id: ID of the content
            
        Returns:
            Difficulty score between 0.0 (easy) and 1.0 (very difficult)
        """
        try:
            # Get discovery statistics for this content
            discoveries = await self._get_content_discoveries(content_id)
            
            if not discoveries:
                # No discoveries yet - use content properties to estimate
                return await self._estimate_discovery_difficulty(content_id)
            
            # Calculate difficulty based on discovery patterns
            discovery_times = [
                d.time_to_discovery_seconds 
                for d in discoveries 
                if d.time_to_discovery_seconds is not None
            ]
            
            if discovery_times:
                avg_time = sum(discovery_times) / len(discovery_times)
                # Normalize to 0-1 scale (assume 10 minutes = max difficulty)
                difficulty = min(1.0, avg_time / 600.0)
            else:
                difficulty = 0.5  # Default medium difficulty
            
            # Adjust based on discovery methods
            method_difficulty_adjustments = {
                "exploration": 0.0,      # Natural discovery
                "accident": -0.1,        # Easier than expected
                "hint": 0.2,            # Needed hints
                "guidance": 0.3,        # Needed direct guidance
                "search": 0.1,          # Required deliberate search
            }
            
            for discovery in discoveries:
                adjustment = method_difficulty_adjustments.get(discovery.discovery_method, 0.0)
                difficulty += adjustment / len(discoveries)  # Average the adjustments
            
            # Ensure difficulty stays in bounds
            difficulty = max(0.0, min(1.0, difficulty))
            
            return difficulty
            
        except Exception as e:
            logger.error(f"Error calculating discovery difficulty: {e}")
            return 0.5  # Default medium difficulty

    async def get_discovery_analytics(self) -> DiscoveryAnalytics:
        """
        Get comprehensive analytics on content discovery.
        
        Returns:
            DiscoveryAnalytics with comprehensive data
        """
        try:
            analytics = DiscoveryAnalytics()
            
            # Get overall statistics
            analytics.total_content_generated = await self._count_total_generated_content()
            analytics.total_content_discovered = await self._count_total_discovered_content()
            
            # Calculate discovery rates by content type
            content_types = ["location", "npc", "object", "connection", "quest"]
            for content_type in content_types:
                generated_count = await self._count_generated_content_by_type(content_type)
                discovered_count = await self._count_discovered_content_by_type(content_type)
                
                if generated_count > 0:
                    rate = discovered_count / generated_count
                    analytics.discovery_rate_by_type[content_type] = rate
            
            # Calculate average time to discovery by type
            for content_type in content_types:
                avg_time = await self._get_avg_discovery_time_by_type(content_type)
                analytics.average_time_to_discovery[content_type] = avg_time
            
            # Get satisfaction trends
            analytics.player_satisfaction_trends = await self._get_satisfaction_trends()
            
            logger.info("Generated comprehensive discovery analytics")
            return analytics
            
        except Exception as e:
            logger.error(f"Error generating discovery analytics: {e}")
            return DiscoveryAnalytics()

    # Private helper methods

    async def _validate_discovery_event(self, event: DiscoveryEvent) -> bool:
        """Validate discovery event data."""
        required_fields = [event.player_id, event.content_id, event.content_type, event.discovery_method]
        return all(field is not None for field in required_fields)

    async def _validate_interaction_event(self, event: InteractionEvent) -> bool:
        """Validate interaction event data."""
        required_fields = [event.player_id, event.content_id, event.interaction_type]
        return all(field is not None for field in required_fields)

    async def _store_discovery_event(self, event: DiscoveryEvent) -> None:
        """Store discovery event in database."""
        # This would insert into content_discovery_events table
        pass

    async def _store_interaction_event(self, event: InteractionEvent) -> None:
        """Store interaction event in database."""
        # This would insert into content_interactions table
        pass

    async def _calculate_discovery_difficulty_from_time(
        self, time_seconds: int, content_type: str, discovery_method: str
    ) -> float:
        """Calculate discovery difficulty from time taken."""
        base_expected_times = {
            "location": 300,    # 5 minutes
            "npc": 180,         # 3 minutes
            "object": 120,      # 2 minutes
            "connection": 240,  # 4 minutes
            "quest": 600,       # 10 minutes
        }
        
        expected_time = base_expected_times.get(content_type, 300)
        difficulty = min(1.0, time_seconds / (expected_time * 2))  # 2x expected = max difficulty
        
        return difficulty

    async def _update_discovery_difficulty(self, discovery_id: UUID, difficulty: float) -> None:
        """Update discovery event with calculated difficulty."""
        # This would update the discovery_difficulty field in database
        pass

    async def _update_content_effectiveness(self, content_id: UUID) -> None:
        """Update content effectiveness metrics."""
        # This would update aggregated effectiveness metrics
        pass

    async def _update_interaction_effectiveness(self, event: InteractionEvent) -> None:
        """Update interaction effectiveness metrics."""
        # This would update interaction-based effectiveness metrics
        pass

    async def _update_satisfaction_tracking(self, event: InteractionEvent) -> None:
        """Update satisfaction tracking data."""
        # This would update satisfaction tracking in database
        pass

    async def _get_discovery_events(self, content_type: str, timeframe_days: int) -> list[DiscoveryEvent]:
        """Get discovery events for analysis."""
        # Mock implementation - would query database
        return []

    async def _get_content_discoveries(self, content_id: UUID) -> list[DiscoveryEvent]:
        """Get all discoveries for specific content."""
        # Mock implementation - would query database
        return []

    async def _get_players_who_encountered_content(self, content_id: UUID) -> int:
        """Get count of players who encountered but may not have discovered content."""
        # Mock implementation - would analyze player paths and content visibility
        return 1

    async def _get_content_interactions(self, content_id: UUID) -> list[InteractionEvent]:
        """Get all interactions for specific content."""
        # Mock implementation - would query database
        return []

    async def _get_location_content(self, location_id: UUID) -> list[dict[str, Any]]:
        """Get all content in a location."""
        # Mock implementation - would query location content
        return []

    async def _get_player_discoveries(self, player_id: UUID) -> list[DiscoveryEvent]:
        """Get all discoveries by a player."""
        # Mock implementation - would query database
        return []

    async def _get_content_hints(self, content_id: UUID) -> list[str]:
        """Get available hints for content."""
        # Mock implementation - would get hints from content metadata
        return []

    async def _get_time_since_generation(self, content_id: UUID) -> float:
        """Get time since content was generated."""
        # Mock implementation - would calculate from generation timestamp
        return 3600.0  # 1 hour

    async def _estimate_discovery_difficulty(self, content_id: UUID) -> float:
        """Estimate discovery difficulty from content properties."""
        # Mock implementation - would analyze content properties
        return 0.5

    async def _count_total_generated_content(self) -> int:
        """Count total generated content."""
        # Mock implementation
        return 100

    async def _count_total_discovered_content(self) -> int:
        """Count total discovered content."""
        # Mock implementation
        return 75

    async def _count_generated_content_by_type(self, content_type: str) -> int:
        """Count generated content by type."""
        # Mock implementation
        return 20

    async def _count_discovered_content_by_type(self, content_type: str) -> int:
        """Count discovered content by type."""
        # Mock implementation
        return 15

    async def _get_avg_discovery_time_by_type(self, content_type: str) -> float:
        """Get average discovery time by content type."""
        # Mock implementation
        base_times = {
            "location": 240.0,
            "npc": 150.0,
            "object": 90.0,
            "connection": 180.0,
            "quest": 480.0,
        }
        return base_times.get(content_type, 200.0)

    async def _get_satisfaction_trends(self) -> list[float]:
        """Get satisfaction trends over time."""
        # Mock implementation - would analyze satisfaction over time
        return [3.2, 3.4, 3.6, 3.5, 3.7, 3.8, 3.9]