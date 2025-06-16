"""
Generation Trigger Manager for Dynamic World Integration.

Analyzes player actions and world state to determine when and what type of
content generation should occur.
"""

import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from game_loop.core.models.connection_models import ConnectionGenerationContext
from game_loop.core.models.location_models import LocationGenerationContext
from game_loop.core.models.npc_models import NPCGenerationContext
from game_loop.core.models.object_models import ObjectGenerationContext
from game_loop.state.models import (
    ActionResult,
    ContentGap,
    GenerationTrigger,
    Location,
    PlayerState,
    WorldState,
)

logger = logging.getLogger(__name__)


class GenerationTriggerManager:
    """
    Analyzes player actions and world state to determine content generation triggers.

    This class is responsible for:
    - Detecting when player actions should trigger content generation
    - Identifying gaps in world content that need to be filled
    - Calculating priority scores for different generation opportunities
    - Managing trigger history and patterns
    """

    def __init__(self, world_state: WorldState, session_factory):
        """Initialize trigger analysis system."""
        self.world_state = world_state
        self.session_factory = session_factory
        self.trigger_cache = {}
        self.recent_triggers = []

        # Configuration for trigger sensitivity
        self.trigger_thresholds = {
            "location_boundary": 0.7,  # Trigger when close to world edge
            "exploration": 0.6,  # Trigger based on exploration patterns
            "quest_need": 0.8,  # Trigger for quest-related content
            "content_gap": 0.5,  # Trigger for missing content
            "player_preference": 0.4,  # Trigger based on player preferences
        }

        # Cooldown periods to prevent over-generation
        self.cooldown_periods = {
            "location": timedelta(minutes=15),
            "npc": timedelta(minutes=10),
            "object": timedelta(minutes=5),
            "connection": timedelta(minutes=8),
        }

    async def analyze_action_for_triggers(
        self, action_result: ActionResult, player_state: PlayerState
    ) -> list[GenerationTrigger]:
        """
        Analyze player action to identify generation triggers.

        Args:
            action_result: Result of the player's action
            player_state: Current player state

        Returns:
            List of generation triggers identified from the action
        """
        triggers = []

        try:
            # Check for location boundary triggers
            if action_result.location_change or self._is_exploration_action(
                action_result
            ):
                boundary_triggers = await self._check_boundary_triggers(
                    action_result, player_state
                )
                triggers.extend(boundary_triggers)

            # Check for content interaction triggers
            interaction_triggers = await self._check_interaction_triggers(
                action_result, player_state
            )
            triggers.extend(interaction_triggers)

            # Check for quest-related triggers
            if action_result.progress_updates:
                quest_triggers = await self._check_quest_triggers(
                    action_result, player_state
                )
                triggers.extend(quest_triggers)

            # Check for exploration depth triggers
            exploration_triggers = await self._check_exploration_triggers(
                action_result, player_state
            )
            triggers.extend(exploration_triggers)

            # Filter out triggers that are on cooldown
            filtered_triggers = await self._filter_cooldown_triggers(triggers)

            # Calculate and assign priority scores
            for trigger in filtered_triggers:
                trigger.priority_score = await self.calculate_generation_priority(
                    trigger,
                    {"player_state": player_state, "action_result": action_result},
                )

            # Sort by priority
            filtered_triggers.sort(key=lambda t: t.priority_score, reverse=True)

            # Add to recent triggers for tracking
            self.recent_triggers.extend(filtered_triggers)
            self._cleanup_recent_triggers()

            logger.info(
                f"Generated {len(filtered_triggers)} triggers from player action"
            )
            return filtered_triggers

        except Exception as e:
            logger.error(f"Error analyzing action for triggers: {e}")
            return []

    async def evaluate_world_gaps(self, current_location_id: UUID) -> list[ContentGap]:
        """
        Identify missing content that should be generated.

        Args:
            current_location_id: The player's current location

        Returns:
            List of identified content gaps
        """
        gaps = []

        try:
            current_location = self.world_state.locations.get(current_location_id)
            if not current_location:
                return gaps

            # Check for missing connections
            connection_gaps = await self._evaluate_connection_gaps(current_location)
            gaps.extend(connection_gaps)

            # Check for empty locations
            empty_location_gaps = await self._evaluate_empty_location_gaps(
                current_location
            )
            gaps.extend(empty_location_gaps)

            # Check for NPC distribution
            npc_gaps = await self._evaluate_npc_gaps(current_location)
            gaps.extend(npc_gaps)

            # Check for object distribution
            object_gaps = await self._evaluate_object_gaps(current_location)
            gaps.extend(object_gaps)

            # Assess player impact for each gap
            for gap in gaps:
                gap.player_impact = await self._calculate_gap_impact(
                    gap, current_location_id
                )

            # Sort by impact and severity
            gaps.sort(
                key=lambda g: (g.player_impact, self._severity_weight(g.severity)),
                reverse=True,
            )

            logger.info(
                f"Identified {len(gaps)} content gaps around location {current_location_id}"
            )
            return gaps

        except Exception as e:
            logger.error(f"Error evaluating world gaps: {e}")
            return []

    async def calculate_generation_priority(
        self, trigger: GenerationTrigger, context: dict[str, Any]
    ) -> float:
        """
        Calculate priority score for generation trigger.

        Args:
            trigger: The generation trigger to score
            context: Additional context for scoring

        Returns:
            Priority score between 0.0 and 1.0
        """
        try:
            base_priority = 0.5

            # Adjust based on trigger type
            type_multipliers = {
                "location_boundary": 0.9,
                "exploration": 0.7,
                "quest_need": 1.0,
                "content_gap": 0.6,
                "player_preference": 0.5,
                "narrative_requirement": 0.8,
            }

            priority = base_priority * type_multipliers.get(trigger.trigger_type, 0.5)

            # Adjust based on player state
            player_state = context.get("player_state")
            if player_state:
                # Higher priority if player has been in area longer
                location_familiarity = len(
                    [
                        loc_id
                        for loc_id in player_state.visited_locations
                        if loc_id == trigger.location_id
                    ]
                )
                priority += min(location_familiarity * 0.1, 0.3)

                # Adjust based on player level/progress
                quest_count = len(player_state.progress.active_quests)
                priority += min(quest_count * 0.05, 0.2)

            # Adjust based on world state
            if trigger.location_id:
                location = self.world_state.locations.get(trigger.location_id)
                if location:
                    # Lower priority if location is already rich in content
                    content_density = (
                        len(location.npcs)
                        + len(location.objects)
                        + len(location.connections)
                    )
                    priority -= min(content_density * 0.02, 0.3)

            # Adjust based on recent generation history
            recent_generation_count = len(
                [
                    t
                    for t in self.recent_triggers
                    if t.trigger_type == trigger.trigger_type
                    and t.triggered_at > datetime.now() - timedelta(hours=1)
                ]
            )
            priority -= min(recent_generation_count * 0.1, 0.4)

            # Ensure priority stays within bounds
            priority = max(0.0, min(1.0, priority))

            return priority

        except Exception as e:
            logger.error(f"Error calculating generation priority: {e}")
            return 0.5

    async def should_generate_location(
        self, context: LocationGenerationContext
    ) -> bool:
        """Determine if new location should be generated."""
        try:
            # Check if we're at a world boundary
            current_location = self.world_state.locations.get(
                context.source_location_id
            )
            if not current_location:
                return False

            # Count existing connections
            connection_count = len(current_location.connections)

            # Generate if very few connections and player is exploring
            if connection_count < 2:
                return True

            # Check if player has exhausted current area
            if context.generation_purpose == "exploration":
                return await self._is_area_exhausted(context.source_location_id)

            return False

        except Exception as e:
            logger.error(f"Error determining location generation: {e}")
            return False

    async def should_generate_npc(self, context: NPCGenerationContext) -> bool:
        """Determine if new NPC should be generated."""
        try:
            location = self.world_state.locations.get(context.location_id)
            if not location:
                return False

            # Check NPC density
            npc_count = len(location.npcs)
            location_theme = location.state_flags.get("theme", "")

            # Different themes have different optimal NPC counts
            optimal_counts = {
                "City": 5,
                "Village": 3,
                "Forest": 1,
                "Mountain": 1,
                "Dungeon": 2,
            }

            optimal_count = optimal_counts.get(location_theme, 2)

            # Generate if below optimal count
            if npc_count < optimal_count:
                return True

            # Generate if there's a specific narrative need
            if context.generation_purpose in ["quest_giver", "merchant", "guide"]:
                return True

            return False

        except Exception as e:
            logger.error(f"Error determining NPC generation: {e}")
            return False

    async def should_generate_object(self, context: ObjectGenerationContext) -> bool:
        """Determine if new object should be generated."""
        try:
            location = self.world_state.locations.get(context.location_id)
            if not location:
                return False

            # Check object density
            object_count = len(location.objects)

            # Generate if very few objects
            if object_count < 2:
                return True

            # Generate for specific purposes
            if context.generation_purpose in ["quest_item", "treasure", "tool"]:
                return True

            # Check if objects match location theme
            theme_appropriate = await self._check_theme_object_appropriateness(location)
            if not theme_appropriate:
                return True

            return False

        except Exception as e:
            logger.error(f"Error determining object generation: {e}")
            return False

    async def should_generate_connection(
        self, context: ConnectionGenerationContext
    ) -> bool:
        """Determine if new connection should be generated."""
        try:
            source_location = self.world_state.locations.get(context.source_location_id)
            if not source_location:
                return False

            # Check connection count
            connection_count = len(source_location.connections)

            # Generate if very isolated
            if connection_count < 1:
                return True

            # Generate for world expansion
            if context.generation_purpose == "world_expansion":
                return connection_count < 4  # Max 4 connections per location

            # Generate for quest needs
            if context.generation_purpose == "quest_path":
                return True

            return False

        except Exception as e:
            logger.error(f"Error determining connection generation: {e}")
            return False

    async def get_trigger_history(
        self, location_id: UUID, hours: int = 24
    ) -> list[GenerationTrigger]:
        """Get recent generation triggers for analysis."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)

            # Filter recent triggers for the location
            relevant_triggers = [
                trigger
                for trigger in self.recent_triggers
                if trigger.location_id == location_id
                and trigger.triggered_at > cutoff_time
            ]

            return relevant_triggers

        except Exception as e:
            logger.error(f"Error getting trigger history: {e}")
            return []

    # Private helper methods

    def _is_exploration_action(self, action_result: ActionResult) -> bool:
        """Check if action is exploration-related."""
        exploration_commands = ["look", "examine", "search", "explore", "move", "go"]
        return any(
            cmd in (action_result.command or "").lower() for cmd in exploration_commands
        )

    async def _check_boundary_triggers(
        self, action_result: ActionResult, player_state: PlayerState
    ) -> list[GenerationTrigger]:
        """Check for world boundary triggers."""
        triggers = []

        current_location_id = (
            action_result.new_location_id or player_state.current_location_id
        )
        if not current_location_id:
            return triggers

        current_location = self.world_state.locations.get(current_location_id)
        if not current_location:
            return triggers

        # Check if location has few connections (indicating boundary)
        if len(current_location.connections) <= 1:
            trigger = GenerationTrigger(
                player_id=player_state.player_id,
                session_id=action_result.processed_input
                or player_state.player_id,  # Fallback
                trigger_type="location_boundary",
                trigger_context={
                    "boundary_type": "edge",
                    "connection_count": len(current_location.connections),
                    "location_theme": current_location.state_flags.get("theme"),
                },
                location_id=current_location_id,
                action_that_triggered=action_result.command,
            )
            triggers.append(trigger)

        return triggers

    async def _check_interaction_triggers(
        self, action_result: ActionResult, player_state: PlayerState
    ) -> list[GenerationTrigger]:
        """Check for content interaction triggers."""
        triggers = []

        # Check if player tried to interact with something that doesn't exist
        if (
            not action_result.success
            and "not found" in action_result.feedback_message.lower()
        ):
            current_location_id = player_state.current_location_id
            if current_location_id:
                trigger = GenerationTrigger(
                    player_id=player_state.player_id,
                    session_id=action_result.processed_input or player_state.player_id,
                    trigger_type="content_gap",
                    trigger_context={
                        "missing_content_type": "unknown",
                        "player_expectation": action_result.command,
                        "failure_reason": action_result.feedback_message,
                    },
                    location_id=current_location_id,
                    action_that_triggered=action_result.command,
                )
                triggers.append(trigger)

        return triggers

    async def _check_quest_triggers(
        self, action_result: ActionResult, player_state: PlayerState
    ) -> list[GenerationTrigger]:
        """Check for quest-related triggers."""
        triggers = []

        if action_result.progress_updates:
            current_location_id = player_state.current_location_id
            if current_location_id:
                trigger = GenerationTrigger(
                    player_id=player_state.player_id,
                    session_id=action_result.processed_input or player_state.player_id,
                    trigger_type="quest_need",
                    trigger_context={
                        "quest_updates": action_result.progress_updates,
                        "active_quests": len(player_state.progress.active_quests),
                    },
                    location_id=current_location_id,
                    action_that_triggered=action_result.command,
                )
                triggers.append(trigger)

        return triggers

    async def _check_exploration_triggers(
        self, action_result: ActionResult, player_state: PlayerState
    ) -> list[GenerationTrigger]:
        """Check for exploration depth triggers."""
        triggers = []

        current_location_id = player_state.current_location_id
        if not current_location_id:
            return triggers

        # Check how thoroughly player has explored current area
        visit_count = player_state.visited_locations.count(current_location_id)

        if visit_count > 3:  # Player has been here multiple times
            trigger = GenerationTrigger(
                player_id=player_state.player_id,
                session_id=action_result.processed_input or player_state.player_id,
                trigger_type="exploration",
                trigger_context={
                    "visit_count": visit_count,
                    "exploration_depth": "high",
                    "area_familiarity": "high",
                },
                location_id=current_location_id,
                action_that_triggered=action_result.command,
            )
            triggers.append(trigger)

        return triggers

    async def _filter_cooldown_triggers(
        self, triggers: list[GenerationTrigger]
    ) -> list[GenerationTrigger]:
        """Filter out triggers that are on cooldown."""
        filtered = []

        for trigger in triggers:
            # Determine content type from trigger
            content_type = self._infer_content_type_from_trigger(trigger)
            cooldown = self.cooldown_periods.get(content_type, timedelta(minutes=10))

            # Check if similar trigger occurred recently
            recent_cutoff = datetime.now() - cooldown
            similar_recent = any(
                t.trigger_type == trigger.trigger_type
                and t.location_id == trigger.location_id
                and t.triggered_at > recent_cutoff
                for t in self.recent_triggers
            )

            if not similar_recent:
                filtered.append(trigger)

        return filtered

    def _infer_content_type_from_trigger(self, trigger: GenerationTrigger) -> str:
        """Infer the likely content type from trigger."""
        type_mapping = {
            "location_boundary": "location",
            "exploration": "connection",
            "quest_need": "npc",
            "content_gap": "object",
        }
        return type_mapping.get(trigger.trigger_type, "object")

    async def _evaluate_connection_gaps(self, location: Location) -> list[ContentGap]:
        """Evaluate missing connections."""
        gaps = []

        connection_count = len(location.connections)
        theme = location.state_flags.get("theme", "")

        # Expected connection counts by theme
        expected_connections = {
            "City": 4,
            "Village": 3,
            "Forest": 2,
            "Mountain": 2,
            "Dungeon": 1,
        }

        expected = expected_connections.get(theme, 2)

        if connection_count < expected:
            gap = ContentGap(
                gap_type="missing_connection",
                location_id=location.location_id,
                severity="medium" if connection_count == 0 else "low",
                suggested_content=["road", "path", "bridge"],
            )
            gaps.append(gap)

        return gaps

    async def _evaluate_empty_location_gaps(
        self, location: Location
    ) -> list[ContentGap]:
        """Evaluate if location needs more content."""
        gaps = []

        total_content = len(location.npcs) + len(location.objects)

        if total_content < 2:
            gap = ContentGap(
                gap_type="empty_location",
                location_id=location.location_id,
                severity="high" if total_content == 0 else "medium",
                suggested_content=["npc", "object"],
            )
            gaps.append(gap)

        return gaps

    async def _evaluate_npc_gaps(self, location: Location) -> list[ContentGap]:
        """Evaluate NPC distribution gaps."""
        gaps = []

        npc_count = len(location.npcs)
        theme = location.state_flags.get("theme", "")

        # Some themes should have NPCs
        npc_expected_themes = ["City", "Village"]

        if theme in npc_expected_themes and npc_count == 0:
            gap = ContentGap(
                gap_type="no_npcs",
                location_id=location.location_id,
                severity="medium",
                suggested_content=["merchant", "guard", "resident"],
            )
            gaps.append(gap)

        return gaps

    async def _evaluate_object_gaps(self, location: Location) -> list[ContentGap]:
        """Evaluate object distribution gaps."""
        gaps = []

        object_count = len(location.objects)

        if object_count == 0:
            gap = ContentGap(
                gap_type="no_objects",
                location_id=location.location_id,
                severity="low",
                suggested_content=["furniture", "decoration", "tool"],
            )
            gaps.append(gap)

        return gaps

    async def _calculate_gap_impact(
        self, gap: ContentGap, current_location_id: UUID
    ) -> float:
        """Calculate player impact of content gap."""
        base_impact = 0.5

        # Higher impact if gap is in current location
        if gap.location_id == current_location_id:
            base_impact += 0.3

        # Adjust based on severity
        severity_impacts = {"low": 0.0, "medium": 0.2, "high": 0.4}
        base_impact += severity_impacts.get(gap.severity, 0.0)

        return min(1.0, base_impact)

    def _severity_weight(self, severity: str) -> float:
        """Convert severity to numeric weight."""
        weights = {"low": 0.3, "medium": 0.6, "high": 1.0}
        return weights.get(severity, 0.5)

    async def _is_area_exhausted(self, location_id: UUID) -> bool:
        """Check if player has exhausted content in area."""
        location = self.world_state.locations.get(location_id)
        if not location:
            return False

        # Simple heuristic: area is exhausted if it has few connections
        # and the player has been there multiple times
        return len(location.connections) < 2

    async def _check_theme_object_appropriateness(self, location: Location) -> bool:
        """Check if objects match location theme."""
        theme = location.state_flags.get("theme", "")

        # Simple check - if location has a theme but no objects, it's not appropriate
        if theme and len(location.objects) == 0:
            return False

        return True

    def _cleanup_recent_triggers(self):
        """Clean up old triggers from recent history."""
        cutoff = datetime.now() - timedelta(hours=24)
        self.recent_triggers = [
            t for t in self.recent_triggers if t.triggered_at > cutoff
        ]
