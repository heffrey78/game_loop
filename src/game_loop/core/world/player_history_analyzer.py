"""
Player History Analyzer for Dynamic World Integration.

Analyzes player behavior patterns to influence content generation decisions 
and maintain engagement.
"""

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from game_loop.state.models import (
    ContentInteraction,
    EngagementMetrics,
    ExplorationPatterns,
    InterestPrediction,
    PlayerFeedback,
    PlayerPreferences,
    PlayerState,
)

logger = logging.getLogger(__name__)


class PlayerHistoryAnalyzer:
    """
    Analyzes player behavior patterns to influence content generation.
    
    This class learns from player actions to:
    - Identify player preferences for different content types
    - Predict what content players might find interesting
    - Track engagement patterns and optimize for player satisfaction
    - Provide insights for adaptive content generation
    """

    def __init__(self, session_factory):
        """Initialize player history analysis system."""
        self.session_factory = session_factory
        self.preference_cache = {}
        self.pattern_cache = {}
        
        # Weights for different types of evidence
        self.evidence_weights = {
            "action_duration": 0.3,
            "repeated_action": 0.5,
            "completion_rate": 0.4,
            "explicit_feedback": 0.8,
            "satisfaction_score": 0.6,
        }
        
        # Decay rates for preference learning
        self.decay_rates = {
            "content_type": 0.95,    # Slow decay for content preferences
            "theme": 0.90,           # Medium decay for theme preferences
            "difficulty": 0.85,      # Faster decay for difficulty preferences
            "interaction": 0.88,     # Medium decay for interaction style
        }

    async def analyze_player_preferences(
        self, player_id: UUID, timeframe_days: int = 30
    ) -> PlayerPreferences:
        """
        Analyze player preferences from action history.
        
        Args:
            player_id: ID of the player to analyze
            timeframe_days: Number of days of history to analyze
            
        Returns:
            PlayerPreferences object with learned preferences
        """
        try:
            # Check cache first
            cache_key = f"{player_id}_{timeframe_days}"
            if cache_key in self.preference_cache:
                cached_prefs, cache_time = self.preference_cache[cache_key]
                if datetime.now() - cache_time < timedelta(hours=1):
                    return cached_prefs
            
            # Initialize preferences
            preferences = PlayerPreferences()
            
            # Analyze content type preferences
            content_prefs = await self._analyze_content_type_preferences(
                player_id, timeframe_days
            )
            preferences.content_type_preferences = content_prefs
            
            # Analyze theme preferences
            theme_prefs = await self._analyze_theme_preferences(
                player_id, timeframe_days
            )
            preferences.theme_preferences = theme_prefs
            
            # Analyze difficulty preference
            difficulty_pref = await self._analyze_difficulty_preference(
                player_id, timeframe_days
            )
            preferences.difficulty_preference = difficulty_pref
            
            # Analyze exploration style
            exploration_style = await self._analyze_exploration_style(
                player_id, timeframe_days
            )
            preferences.exploration_style = exploration_style
            
            # Analyze interaction style
            interaction_style = await self._analyze_interaction_style(
                player_id, timeframe_days
            )
            preferences.interaction_style = interaction_style
            
            # Calculate confidence scores
            preferences.confidence_scores = await self._calculate_confidence_scores(
                player_id, timeframe_days
            )
            
            # Cache the results
            self.preference_cache[cache_key] = (preferences, datetime.now())
            
            logger.info(f"Analyzed preferences for player {player_id}")
            return preferences
            
        except Exception as e:
            logger.error(f"Error analyzing player preferences: {e}")
            return PlayerPreferences()

    async def get_exploration_patterns(self, player_id: UUID) -> ExplorationPatterns:
        """
        Analyze how player explores the world.
        
        Args:
            player_id: ID of the player to analyze
            
        Returns:
            ExplorationPatterns object with exploration behavior analysis
        """
        try:
            patterns = ExplorationPatterns()
            
            # Get recent location visits and timing
            visit_data = await self._get_location_visit_data(player_id)
            
            if visit_data:
                # Calculate average time per location
                total_time = sum(visit["duration"] for visit in visit_data)
                patterns.average_time_per_location = total_time / len(visit_data)
                
                # Analyze preferred connection types
                connection_usage = await self._analyze_connection_usage(player_id)
                patterns.preferred_connection_types = list(connection_usage.keys())
                
                # Calculate backtracking frequency
                patterns.backtracking_frequency = await self._calculate_backtracking_frequency(
                    player_id
                )
                
                # Assess discovery thoroughness
                patterns.discovery_thoroughness = await self._assess_discovery_thoroughness(
                    player_id
                )
                
                # Evaluate risk tolerance
                patterns.risk_tolerance = await self._evaluate_risk_tolerance(player_id)
            
            logger.info(f"Analyzed exploration patterns for player {player_id}")
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing exploration patterns: {e}")
            return ExplorationPatterns()

    async def calculate_engagement_metrics(
        self, player_id: UUID, session_id: UUID
    ) -> EngagementMetrics:
        """
        Calculate player engagement metrics.
        
        Args:
            player_id: ID of the player
            session_id: ID of the current session
            
        Returns:
            EngagementMetrics with calculated engagement data
        """
        try:
            metrics = EngagementMetrics()
            
            # Get session data
            session_data = await self._get_session_data(player_id, session_id)
            
            if session_data:
                # Calculate session duration
                metrics.session_duration = session_data.get("duration", 0.0)
                
                # Calculate actions per minute
                action_count = session_data.get("action_count", 0)
                if metrics.session_duration > 0:
                    metrics.actions_per_minute = action_count / (metrics.session_duration / 60)
                
                # Calculate content interaction rate
                interactions = session_data.get("content_interactions", 0)
                content_encountered = session_data.get("content_encountered", 1)
                metrics.content_interaction_rate = interactions / content_encountered
                
                # Calculate exploration depth
                unique_locations = session_data.get("unique_locations_visited", 0)
                total_moves = session_data.get("movement_actions", 1)
                metrics.exploration_depth = unique_locations / total_moves
                
                # Calculate quest completion rate
                quests_completed = session_data.get("quests_completed", 0)
                quests_started = session_data.get("quests_started", 1)
                metrics.quest_completion_rate = quests_completed / quests_started
                
                # Gather satisfaction indicators
                metrics.satisfaction_indicators = await self._gather_satisfaction_indicators(
                    player_id, session_id
                )
            
            logger.info(f"Calculated engagement metrics for player {player_id}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating engagement metrics: {e}")
            return EngagementMetrics()

    async def predict_player_interests(
        self, player_state: PlayerState, context: dict[str, Any]
    ) -> list[InterestPrediction]:
        """
        Predict what content player might find interesting.
        
        Args:
            player_state: Current player state
            context: Additional context for prediction
            
        Returns:
            List of interest predictions sorted by likelihood
        """
        try:
            predictions = []
            
            # Get player preferences
            preferences = await self.analyze_player_preferences(player_state.player_id)
            
            # Predict interest in different content types
            content_types = ["location", "npc", "object", "connection", "quest"]
            
            for content_type in content_types:
                # Base interest from preferences
                base_interest = preferences.content_type_preferences.get(content_type, 0.5)
                
                # Adjust based on current context
                interest_score = await self._adjust_interest_for_context(
                    base_interest, content_type, player_state, context
                )
                
                # Calculate confidence based on evidence
                confidence = preferences.confidence_scores.get(content_type, 0.5)
                
                # Generate reasoning
                reasoning = await self._generate_interest_reasoning(
                    content_type, interest_score, preferences, context
                )
                
                prediction = InterestPrediction(
                    content_type=content_type,
                    interest_score=interest_score,
                    confidence=confidence,
                    reasoning=reasoning,
                )
                predictions.append(prediction)
            
            # Sort by interest score
            predictions.sort(key=lambda p: p.interest_score, reverse=True)
            
            logger.info(f"Generated {len(predictions)} interest predictions for player {player_state.player_id}")
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting player interests: {e}")
            return []

    async def get_content_interaction_history(
        self, player_id: UUID, content_type: str
    ) -> list[ContentInteraction]:
        """
        Get player's history with specific content types.
        
        Args:
            player_id: ID of the player
            content_type: Type of content to analyze
            
        Returns:
            List of content interactions
        """
        try:
            # This would typically query the database
            # For now, return mock data structure
            interactions = []
            
            # In a real implementation, this would query content_interactions table
            # filtered by player_id and content_type
            
            logger.info(f"Retrieved interaction history for player {player_id}, content type {content_type}")
            return interactions
            
        except Exception as e:
            logger.error(f"Error getting content interaction history: {e}")
            return []

    async def update_preference_model(
        self, player_id: UUID, feedback: PlayerFeedback
    ) -> bool:
        """
        Update player preference model based on feedback.
        
        Args:
            player_id: ID of the player
            feedback: Player feedback to incorporate
            
        Returns:
            True if update was successful
        """
        try:
            # Clear cache for this player
            cache_keys_to_remove = [
                key for key in self.preference_cache.keys()
                if key.startswith(str(player_id))
            ]
            for key in cache_keys_to_remove:
                del self.preference_cache[key]
            
            # Apply feedback to preference learning
            await self._apply_feedback_to_preferences(player_id, feedback)
            
            logger.info(f"Updated preference model for player {player_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating preference model: {e}")
            return False

    async def get_similar_players(
        self, player_id: UUID, limit: int = 10
    ) -> list[tuple[UUID, float]]:
        """
        Find players with similar behavior patterns.
        
        Args:
            player_id: ID of the reference player
            limit: Maximum number of similar players to return
            
        Returns:
            List of (player_id, similarity_score) tuples
        """
        try:
            # Get reference player's preferences
            ref_preferences = await self.analyze_player_preferences(player_id)
            
            # This would typically query all players and calculate similarity
            # For now, return empty list as this requires database implementation
            similar_players = []
            
            logger.info(f"Found {len(similar_players)} similar players for {player_id}")
            return similar_players
            
        except Exception as e:
            logger.error(f"Error finding similar players: {e}")
            return []

    # Private helper methods

    async def _analyze_content_type_preferences(
        self, player_id: UUID, timeframe_days: int
    ) -> dict[str, float]:
        """Analyze preferences for different content types."""
        # Mock implementation - would analyze actual interaction data
        content_types = ["location", "npc", "object", "connection", "quest"]
        preferences = {}
        
        for content_type in content_types:
            # Simulate preference learning from interaction data
            base_preference = 0.5
            
            # Adjust based on simulated interaction patterns
            interaction_frequency = await self._get_interaction_frequency(
                player_id, content_type, timeframe_days
            )
            satisfaction_score = await self._get_avg_satisfaction(
                player_id, content_type, timeframe_days
            )
            
            # Combine factors
            preference = (base_preference + interaction_frequency * 0.3 + satisfaction_score * 0.2)
            preferences[content_type] = min(1.0, max(0.0, preference))
        
        return preferences

    async def _analyze_theme_preferences(
        self, player_id: UUID, timeframe_days: int
    ) -> dict[str, float]:
        """Analyze preferences for different themes."""
        themes = ["Forest", "City", "Village", "Mountain", "Dungeon"]
        preferences = {}
        
        for theme in themes:
            # Simulate theme preference learning
            time_spent = await self._get_time_spent_in_theme(player_id, theme, timeframe_days)
            exploration_depth = await self._get_theme_exploration_depth(player_id, theme)
            
            preference = (time_spent * 0.4 + exploration_depth * 0.3 + 0.3)  # Base preference
            preferences[theme] = min(1.0, max(0.0, preference))
        
        return preferences

    async def _analyze_difficulty_preference(
        self, player_id: UUID, timeframe_days: int
    ) -> float:
        """Analyze player's preferred difficulty level."""
        # Simulate difficulty preference analysis
        quest_completion_rate = await self._get_quest_completion_rate(player_id, timeframe_days)
        challenge_seeking = await self._get_challenge_seeking_behavior(player_id)
        
        # Players who complete more quests might prefer moderate difficulty
        # Players who seek challenges might prefer higher difficulty
        difficulty_pref = 0.3 + quest_completion_rate * 0.4 + challenge_seeking * 0.3
        
        return min(1.0, max(0.0, difficulty_pref))

    async def _analyze_exploration_style(
        self, player_id: UUID, timeframe_days: int
    ) -> str:
        """Determine player's exploration style."""
        thoroughness = await self._assess_discovery_thoroughness(player_id)
        speed = await self._assess_exploration_speed(player_id)
        
        if thoroughness > 0.7:
            return "thorough"
        elif speed > 0.7:
            return "direct"
        else:
            return "balanced"

    async def _analyze_interaction_style(
        self, player_id: UUID, timeframe_days: int
    ) -> str:
        """Determine player's interaction style."""
        social_interactions = await self._count_social_interactions(player_id, timeframe_days)
        combat_interactions = await self._count_combat_interactions(player_id, timeframe_days)
        puzzle_interactions = await self._count_puzzle_interactions(player_id, timeframe_days)
        
        if social_interactions > combat_interactions and social_interactions > puzzle_interactions:
            return "social"
        elif combat_interactions > puzzle_interactions:
            return "combat"
        elif puzzle_interactions > 0:
            return "puzzle"
        else:
            return "balanced"

    async def _calculate_confidence_scores(
        self, player_id: UUID, timeframe_days: int
    ) -> dict[str, float]:
        """Calculate confidence scores for preferences."""
        # Confidence based on amount of data available
        data_points = await self._count_data_points(player_id, timeframe_days)
        
        base_confidence = min(1.0, data_points / 100.0)  # 100 data points = full confidence
        
        return {
            "content_type": base_confidence,
            "theme": base_confidence * 0.9,
            "difficulty": base_confidence * 0.8,
            "exploration": base_confidence * 0.85,
            "interaction": base_confidence * 0.8,
        }

    async def _get_location_visit_data(self, player_id: UUID) -> list[dict[str, Any]]:
        """Get location visit timing data."""
        # Mock implementation
        return [
            {"location_id": "loc1", "duration": 300, "actions": 15},
            {"location_id": "loc2", "duration": 180, "actions": 8},
        ]

    async def _analyze_connection_usage(self, player_id: UUID) -> dict[str, int]:
        """Analyze which connection types player uses most."""
        # Mock implementation
        return {"road": 5, "path": 8, "bridge": 2}

    async def _calculate_backtracking_frequency(self, player_id: UUID) -> float:
        """Calculate how often player backtracks."""
        # Mock implementation
        return 0.3  # 30% of moves are backtracking

    async def _assess_discovery_thoroughness(self, player_id: UUID) -> float:
        """Assess how thoroughly player explores locations."""
        # Mock implementation
        return 0.75  # 75% thorough

    async def _evaluate_risk_tolerance(self, player_id: UUID) -> float:
        """Evaluate player's risk tolerance."""
        # Mock implementation
        return 0.6  # Moderate risk tolerance

    async def _get_session_data(self, player_id: UUID, session_id: UUID) -> dict[str, Any]:
        """Get comprehensive session data."""
        # Mock implementation
        return {
            "duration": 1800.0,  # 30 minutes
            "action_count": 45,
            "content_interactions": 12,
            "content_encountered": 15,
            "unique_locations_visited": 6,
            "movement_actions": 20,
            "quests_completed": 1,
            "quests_started": 2,
        }

    async def _gather_satisfaction_indicators(
        self, player_id: UUID, session_id: UUID
    ) -> dict[str, float]:
        """Gather various satisfaction indicators."""
        # Mock implementation
        return {
            "completion_rate": 0.8,
            "time_engagement": 0.7,
            "repeat_actions": 0.6,
            "exploration_satisfaction": 0.75,
        }

    async def _adjust_interest_for_context(
        self, base_interest: float, content_type: str, player_state: PlayerState, context: dict[str, Any]
    ) -> float:
        """Adjust interest score based on current context."""
        adjusted_interest = base_interest
        
        # Adjust based on current location
        current_location = context.get("current_location")
        if current_location:
            theme = current_location.state_flags.get("theme", "")
            if theme in ["City", "Village"] and content_type == "npc":
                adjusted_interest += 0.2
            elif theme in ["Forest", "Mountain"] and content_type == "connection":
                adjusted_interest += 0.15
        
        # Adjust based on recent actions
        recent_actions = context.get("recent_actions", [])
        if "explore" in recent_actions and content_type == "location":
            adjusted_interest += 0.1
        
        return min(1.0, max(0.0, adjusted_interest))

    async def _generate_interest_reasoning(
        self, content_type: str, interest_score: float, preferences: PlayerPreferences, context: dict[str, Any]
    ) -> list[str]:
        """Generate reasoning for interest prediction."""
        reasoning = []
        
        if interest_score > 0.7:
            reasoning.append(f"High historical engagement with {content_type}")
        elif interest_score > 0.5:
            reasoning.append(f"Moderate interest in {content_type} based on past behavior")
        else:
            reasoning.append(f"Limited historical interaction with {content_type}")
        
        # Add context-specific reasoning
        if context.get("current_location"):
            reasoning.append("Current location context supports this content type")
        
        return reasoning

    async def _apply_feedback_to_preferences(
        self, player_id: UUID, feedback: PlayerFeedback
    ) -> None:
        """Apply feedback to update preferences."""
        # This would update the database with new preference data
        # based on the feedback provided
        pass

    # Mock helper methods for data that would come from database

    async def _get_interaction_frequency(
        self, player_id: UUID, content_type: str, timeframe_days: int
    ) -> float:
        """Get interaction frequency for content type."""
        return 0.6  # Mock value

    async def _get_avg_satisfaction(
        self, player_id: UUID, content_type: str, timeframe_days: int
    ) -> float:
        """Get average satisfaction score for content type."""
        return 0.7  # Mock value

    async def _get_time_spent_in_theme(
        self, player_id: UUID, theme: str, timeframe_days: int
    ) -> float:
        """Get time spent in locations with specific theme."""
        return 0.5  # Mock value

    async def _get_theme_exploration_depth(self, player_id: UUID, theme: str) -> float:
        """Get exploration depth for specific theme."""
        return 0.6  # Mock value

    async def _get_quest_completion_rate(self, player_id: UUID, timeframe_days: int) -> float:
        """Get quest completion rate."""
        return 0.75  # Mock value

    async def _get_challenge_seeking_behavior(self, player_id: UUID) -> float:
        """Get challenge seeking behavior score."""
        return 0.5  # Mock value

    async def _assess_exploration_speed(self, player_id: UUID) -> float:
        """Assess how quickly player explores."""
        return 0.6  # Mock value

    async def _count_social_interactions(self, player_id: UUID, timeframe_days: int) -> int:
        """Count social interactions."""
        return 10  # Mock value

    async def _count_combat_interactions(self, player_id: UUID, timeframe_days: int) -> int:
        """Count combat interactions."""
        return 5  # Mock value

    async def _count_puzzle_interactions(self, player_id: UUID, timeframe_days: int) -> int:
        """Count puzzle interactions."""
        return 3  # Mock value

    async def _count_data_points(self, player_id: UUID, timeframe_days: int) -> int:
        """Count available data points for analysis."""
        return 50  # Mock value