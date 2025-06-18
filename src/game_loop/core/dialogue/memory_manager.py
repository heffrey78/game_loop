"""
Conversation Memory Manager for tracking NPC-player interactions and relationships.

This module manages conversation history, relationship development, and contextual
memory for more engaging NPC interactions across multiple encounters.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


class ConversationMemoryManager:
    """Manage NPC memory of conversations and relationships."""

    def __init__(self):
        # In-memory storage (would be replaced with database in production)
        self.conversation_history: dict[str, list[dict[str, Any]]] = {}
        self.relationship_scores: dict[str, dict[str, Any]] = {}
        self.topic_knowledge: dict[str, dict[str, int]] = {}
        self.name_memory: dict[str, dict[str, Any]] = {}

    async def record_conversation(
        self,
        npc_id: str,
        player_id: str,
        topic: str | None,
        response: str,
        context: dict[str, Any],
    ) -> None:
        """Record a conversation for future reference."""
        conversation_key = f"{npc_id}_{player_id}"

        if conversation_key not in self.conversation_history:
            self.conversation_history[conversation_key] = []

        conversation_record = {
            "timestamp": datetime.now(),
            "topic": topic or "general",
            "response": response,
            "context": context,
            "location_id": context.get("location_id"),
            "player_name": context.get("player_name", "Unknown"),
            "interaction_type": context.get("interaction_type", "greeting"),
        }

        self.conversation_history[conversation_key].append(conversation_record)

        # Limit history to last 50 conversations per NPC-player pair
        if len(self.conversation_history[conversation_key]) > 50:
            self.conversation_history[conversation_key] = self.conversation_history[
                conversation_key
            ][-50:]

        # Update relationship based on interaction
        await self._update_relationship(npc_id, player_id, topic, context)

        # Track topic knowledge
        await self._update_topic_knowledge(npc_id, topic or "general")

        # Record name for future reference
        if context.get("player_name"):
            await self._record_player_name(npc_id, player_id, context["player_name"])

    def get_conversation_context(self, npc_id: str, player_id: str) -> dict[str, Any]:
        """Get previous conversation context for reference."""
        conversation_key = f"{npc_id}_{player_id}"
        history = self.conversation_history.get(conversation_key, [])

        if not history:
            return {
                "is_first_meeting": True,
                "previous_topics": [],
                "total_conversations": 0,
                "relationship_level": "stranger",
                "last_interaction": None,
                "player_name_known": False,
                "preferred_name": None,
            }

        # Get relationship information
        relationship_data = self._get_relationship_data(npc_id, player_id)

        # Get name information
        name_data = self._get_name_data(npc_id, player_id)

        # Calculate time since last interaction
        last_interaction = history[-1]["timestamp"]
        time_since_last = datetime.now() - last_interaction

        return {
            "is_first_meeting": False,
            "previous_topics": [conv["topic"] for conv in history[-5:]],
            "total_conversations": len(history),
            "relationship_level": relationship_data["level"],
            "relationship_score": relationship_data["score"],
            "last_interaction": last_interaction,
            "time_since_last": time_since_last,
            "player_name_known": name_data["known"],
            "preferred_name": name_data["preferred_name"],
            "conversation_frequency": self._calculate_conversation_frequency(history),
            "common_topics": self._get_common_topics(history),
            "recent_locations": self._get_recent_locations(history),
        }

    async def _update_relationship(
        self, npc_id: str, player_id: str, topic: str | None, context: dict[str, Any]
    ) -> None:
        """Update relationship score based on interaction."""
        rel_key = f"{npc_id}_{player_id}"

        if rel_key not in self.relationship_scores:
            self.relationship_scores[rel_key] = {
                "score": 0,
                "level": "stranger",
                "interactions": 0,
                "positive_interactions": 0,
                "negative_interactions": 0,
                "first_meeting": datetime.now(),
                "last_update": datetime.now(),
            }

        rel_data = self.relationship_scores[rel_key]

        # Calculate score change based on interaction
        score_change = self._calculate_score_change(topic, context)

        # Update relationship data
        rel_data["score"] += score_change
        rel_data["interactions"] += 1
        rel_data["last_update"] = datetime.now()

        if score_change > 0:
            rel_data["positive_interactions"] += 1
        elif score_change < 0:
            rel_data["negative_interactions"] += 1

        # Update relationship level
        rel_data["level"] = self._determine_relationship_level(rel_data["score"])

    def _calculate_score_change(
        self, topic: str | None, context: dict[str, Any]
    ) -> int:
        """Calculate relationship score change based on interaction."""
        base_score = 1  # Basic positive interaction

        # Topic-based modifiers
        if topic:
            if topic in ["help_request", "information_sharing"]:
                base_score += 2
            elif topic in ["complaint", "accusation"]:
                base_score -= 3
            elif topic in ["compliment", "gratitude"]:
                base_score += 3

        # Context-based modifiers
        interaction_type = context.get("interaction_type", "neutral")
        if interaction_type == "helpful":
            base_score += 1
        elif interaction_type == "rude":
            base_score -= 2
        elif interaction_type == "polite":
            base_score += 1

        # Frequency modifier (frequent interactions are slightly less valuable)
        frequency = context.get("conversation_frequency", 1)
        if frequency > 5:
            base_score = max(1, base_score - 1)

        return base_score

    def _determine_relationship_level(self, score: int) -> str:
        """Determine relationship level based on score."""
        if score >= 75:
            return "trusted_friend"
        elif score >= 50:
            return "close_friend"
        elif score >= 25:
            return "friend"
        elif score >= 10:
            return "friendly_acquaintance"
        elif score >= 0:
            return "acquaintance"
        elif score >= -10:
            return "neutral"
        elif score >= -25:
            return "unfriendly"
        else:
            return "hostile"

    async def _update_topic_knowledge(self, npc_id: str, topic: str) -> None:
        """Track topics discussed with this NPC."""
        if npc_id not in self.topic_knowledge:
            self.topic_knowledge[npc_id] = {}

        if topic not in self.topic_knowledge[npc_id]:
            self.topic_knowledge[npc_id][topic] = 0

        self.topic_knowledge[npc_id][topic] += 1

    async def _record_player_name(
        self, npc_id: str, player_id: str, player_name: str
    ) -> None:
        """Record player name for this NPC."""
        name_key = f"{npc_id}_{player_id}"

        if name_key not in self.name_memory:
            self.name_memory[name_key] = {
                "known": False,
                "preferred_name": None,
                "first_learned": None,
                "times_used": 0,
            }

        name_data = self.name_memory[name_key]

        if not name_data["known"]:
            name_data["known"] = True
            name_data["preferred_name"] = player_name
            name_data["first_learned"] = datetime.now()

        name_data["times_used"] += 1

    def _get_relationship_data(self, npc_id: str, player_id: str) -> dict[str, Any]:
        """Get relationship data for NPC-player pair."""
        rel_key = f"{npc_id}_{player_id}"
        return self.relationship_scores.get(
            rel_key, {"score": 0, "level": "stranger", "interactions": 0}
        )

    def _get_name_data(self, npc_id: str, player_id: str) -> dict[str, Any]:
        """Get name memory data for NPC-player pair."""
        name_key = f"{npc_id}_{player_id}"
        return self.name_memory.get(name_key, {"known": False, "preferred_name": None})

    def _calculate_conversation_frequency(self, history: list[dict[str, Any]]) -> float:
        """Calculate how frequently conversations occur."""
        if len(history) < 2:
            return 1.0

        # Calculate average time between conversations
        time_diffs = []
        for i in range(1, len(history)):
            diff = history[i]["timestamp"] - history[i - 1]["timestamp"]
            time_diffs.append(diff.total_seconds() / 3600)  # Convert to hours

        avg_hours = sum(time_diffs) / len(time_diffs)

        # Convert to frequency score (higher = more frequent)
        if avg_hours < 1:
            return 10.0  # Very frequent
        elif avg_hours < 24:
            return 5.0  # Daily
        elif avg_hours < 168:
            return 2.0  # Weekly
        else:
            return 1.0  # Infrequent

    def _get_common_topics(self, history: list[dict[str, Any]]) -> list[str]:
        """Get most commonly discussed topics."""
        topic_counts = {}
        for conv in history:
            topic = conv["topic"]
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        # Sort by frequency and return top 3
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, count in sorted_topics[:3]]

    def _get_recent_locations(self, history: list[dict[str, Any]]) -> list[str]:
        """Get recent interaction locations."""
        recent_locations = []
        for conv in history[-5:]:  # Last 5 conversations
            location_id = conv.get("location_id")
            if location_id and location_id not in recent_locations:
                recent_locations.append(location_id)

        return recent_locations

    def should_npc_remember_player(self, npc_id: str, player_id: str) -> bool:
        """Determine if NPC should remember this player."""
        conversation_key = f"{npc_id}_{player_id}"
        history = self.conversation_history.get(conversation_key, [])

        if not history:
            return False

        # Remember if had more than 2 conversations or last conversation was recent
        if len(history) > 2:
            return True

        last_interaction = history[-1]["timestamp"]
        time_since = datetime.now() - last_interaction

        # Remember if last conversation was within 24 hours
        return time_since < timedelta(hours=24)

    def get_relationship_summary(self, npc_id: str, player_id: str) -> str:
        """Get a text summary of the relationship."""
        context = self.get_conversation_context(npc_id, player_id)

        if context["is_first_meeting"]:
            return "This is your first meeting with this person."

        level = context["relationship_level"]
        total = context["total_conversations"]
        name_known = context["player_name_known"]

        summary = f"You have a {level} relationship after {total} conversations."

        if name_known:
            summary += f" They know you as {context['preferred_name']}."

        if context["time_since_last"]:
            hours_since = context["time_since_last"].total_seconds() / 3600
            if hours_since < 1:
                summary += " You spoke recently."
            elif hours_since < 24:
                summary += " You spoke earlier today."
            elif hours_since < 168:
                summary += " You spoke this week."
            else:
                summary += " It's been a while since you last spoke."

        return summary

    def clear_old_conversations(self, days_old: int = 30) -> int:
        """Clear conversations older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        cleared_count = 0

        for conversation_key, history in list(self.conversation_history.items()):
            # Filter out old conversations
            filtered_history = [
                conv for conv in history if conv["timestamp"] > cutoff_date
            ]

            if len(filtered_history) != len(history):
                cleared_count += len(history) - len(filtered_history)

                if filtered_history:
                    self.conversation_history[conversation_key] = filtered_history
                else:
                    # Remove empty conversation history
                    del self.conversation_history[conversation_key]

        return cleared_count
