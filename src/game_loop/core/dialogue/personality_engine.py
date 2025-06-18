"""
NPC Personality Engine for generating personality-driven dialogue responses.

This module provides personality profiles and response generation for different
NPC archetypes to create more engaging and consistent character interactions.
"""

import logging
import random
from typing import Any

logger = logging.getLogger(__name__)


class NPCPersonalityEngine:
    """Generate personality-consistent dialogue based on NPC archetype."""

    def __init__(self):
        # Define comprehensive personality profiles
        self.personality_profiles = {
            "security_guard": {
                "traits": ["dutiful", "cautious", "professional", "alert"],
                "speech_patterns": ["formal", "direct", "procedural", "authoritative"],
                "concerns": ["security", "protocol", "identification", "safety"],
                "greeting_style": "professional_inquiry",
                "knowledge_focus": [
                    "building_layout",
                    "security_procedures",
                    "personnel",
                    "access_control",
                ],
                "mood_modifiers": {
                    "alert": "extra vigilant and questioning",
                    "relaxed": "more conversational but still professional",
                    "suspicious": "terse and demanding",
                },
            },
            "scholar": {
                "traits": ["curious", "verbose", "analytical", "intellectual"],
                "speech_patterns": [
                    "academic",
                    "detailed",
                    "questioning",
                    "philosophical",
                ],
                "concerns": ["knowledge", "research", "accuracy", "learning"],
                "greeting_style": "intellectual_engagement",
                "knowledge_focus": [
                    "books",
                    "research",
                    "history",
                    "theories",
                    "archives",
                ],
                "mood_modifiers": {
                    "excited": "enthusiastic about sharing knowledge",
                    "focused": "absorbed in current research",
                    "helpful": "eager to assist with information",
                },
            },
            "merchant": {
                "traits": ["practical", "friendly", "profit-minded", "persuasive"],
                "speech_patterns": [
                    "persuasive",
                    "friendly",
                    "transactional",
                    "enthusiastic",
                ],
                "concerns": ["trade", "value", "customers", "profit"],
                "greeting_style": "commercial_welcome",
                "knowledge_focus": [
                    "items",
                    "prices",
                    "market_conditions",
                    "trade_routes",
                ],
                "mood_modifiers": {
                    "eager": "very interested in making a sale",
                    "cautious": "careful about pricing and quality",
                    "generous": "willing to offer good deals",
                },
            },
            "administrator": {
                "traits": [
                    "organized",
                    "bureaucratic",
                    "rule-focused",
                    "detail-oriented",
                ],
                "speech_patterns": ["formal", "procedural", "systematic", "official"],
                "concerns": ["procedures", "documentation", "compliance", "efficiency"],
                "greeting_style": "official_inquiry",
                "knowledge_focus": ["procedures", "regulations", "forms", "schedules"],
                "mood_modifiers": {
                    "stressed": "overwhelmed with paperwork and procedures",
                    "helpful": "willing to guide through proper channels",
                    "strict": "rigid about following rules",
                },
            },
            "generic": {
                "traits": ["neutral", "helpful", "observant"],
                "speech_patterns": ["casual", "friendly"],
                "concerns": ["general_well_being", "local_events"],
                "greeting_style": "casual_greeting",
                "knowledge_focus": ["local_area", "general_information"],
                "mood_modifiers": {
                    "friendly": "warm and welcoming",
                    "neutral": "polite but reserved",
                },
            },
        }

        # Response templates organized by greeting style
        self.greeting_templates = {
            "professional_inquiry": [
                "Good {time_of_day}. I'll need to see some identification, please.",
                "Please state your business here and show your credentials.",
                "This is a restricted area. Do you have proper authorization to be here?",
                "Excuse me, but I need to verify your clearance level.",
                "Security checkpoint. May I see your access card?",
            ],
            "intellectual_engagement": [
                "Ah, another seeker of knowledge! What brings you to these halls of learning?",
                "Welcome, fellow scholar. Are you here for research or general inquiry?",
                "Fascinating to see someone else exploring these archives.",
                "Greetings! I hope you find what you're looking for in our collection.",
                "Welcome to our repository of knowledge. How may I assist your studies?",
            ],
            "commercial_welcome": [
                "Welcome! I have many fine wares that might interest you.",
                "Good {time_of_day}! Looking for anything in particular today?",
                "Step right up! I have exactly what you need.",
                "Greetings, traveler! You've come to the right place for quality goods.",
                "Welcome to my establishment! What can I help you find?",
            ],
            "official_inquiry": [
                "Good {time_of_day}. How may I assist you with official business?",
                "Welcome to the administrative office. What paperwork brings you here today?",
                "Please take a number and state the nature of your inquiry.",
                "Official business hours are in effect. How may I direct you?",
                "Administrative services available. What documentation do you need?",
            ],
            "casual_greeting": [
                "Hello there! Nice to see someone new around here.",
                "Good {time_of_day}! How are you doing?",
                "Welcome! Hope you're finding everything alright.",
                "Hi! Can I help you with anything?",
                "Good to see you! What brings you this way?",
            ],
        }

        # Topic-specific response patterns
        self.topic_responses = {
            "location_inquiry": {
                "security_guard": [
                    "I know this area well - it's part of my patrol route.",
                    "This sector requires {security_level} clearance.",
                    "I can tell you about the security features, but some information is classified.",
                    "Been working this area for {time_period}. What do you need to know?",
                ],
                "scholar": [
                    "This location has quite a rich history, you know.",
                    "From an academic perspective, this area is particularly interesting because...",
                    "I've researched this place extensively. What aspects interest you?",
                    "The architectural and historical significance here is remarkable.",
                ],
                "administrator": [
                    "According to our records, this area is designated as {zone_type}.",
                    "I have the official documentation for this location if you need it.",
                    "This falls under regulation {regulation_code}.",
                    "Let me check our administrative files for accurate information.",
                ],
            },
            "help_request": {
                "security_guard": [
                    "I can assist, but I need to follow proper protocols.",
                    "Security procedures require me to verify your identity first.",
                    "I'm here to maintain safety and order. What's the situation?",
                    "Protocol dictates that I assess the security implications first.",
                ],
                "scholar": [
                    "I'd be delighted to help! Knowledge shared is knowledge multiplied.",
                    "What fascinating question brings you to seek assistance?",
                    "I have extensive research materials that might be relevant.",
                    "Let me think... there are several approaches we could consider.",
                ],
                "merchant": [
                    "Help is good for business! What can I do for you?",
                    "I'm always willing to assist a potential customer!",
                    "Service with a smile! What's the problem?",
                    "Helping others often leads to mutual benefit.",
                ],
            },
        }

    def generate_personality_response(
        self, npc_archetype: str, context: dict[str, Any], topic: str | None = None
    ) -> str:
        """Generate response based on NPC personality profile."""
        profile = self.personality_profiles.get(
            npc_archetype, self.personality_profiles["generic"]
        )

        # Determine mood from context
        mood = context.get("mood", "neutral")

        if topic and topic in self.topic_responses:
            # Generate topic-specific response
            return self._generate_topic_response(npc_archetype, profile, context, topic)
        else:
            # Generate greeting response
            return self._generate_greeting_response(profile, context, mood)

    def _generate_greeting_response(
        self, profile: dict[str, Any], context: dict[str, Any], mood: str
    ) -> str:
        """Generate greeting based on NPC personality and context."""
        greeting_style = profile.get("greeting_style", "casual_greeting")
        templates = self.greeting_templates.get(
            greeting_style, self.greeting_templates["casual_greeting"]
        )

        # Select template with some variety
        template = random.choice(templates)

        # Fill in context variables
        response = template.format(
            time_of_day=context.get("time_of_day", "day"),
            player_name=context.get("player_name", ""),
            security_level=context.get("security_level", "standard"),
            zone_type=context.get("zone_type", "general access"),
        )

        # Apply mood modifiers
        response = self._apply_mood_modifier(response, profile, mood)

        # Apply personality traits
        response = self._apply_personality_traits(response, profile["traits"])

        return response

    def _generate_topic_response(
        self,
        npc_archetype: str,
        profile: dict[str, Any],
        context: dict[str, Any],
        topic: str,
    ) -> str:
        """Generate response for specific topic discussion."""
        archetype_responses = self.topic_responses.get(topic, {})
        responses = archetype_responses.get(
            npc_archetype,
            archetype_responses.get(
                "generic", ["That's an interesting topic. Let me think about that."]
            ),
        )

        # Select and format response
        base_response = random.choice(responses)
        response = base_response.format(
            security_level=context.get("security_level", "standard"),
            time_period=context.get("time_period", "several years"),
            zone_type=context.get("zone_type", "general access"),
            regulation_code=context.get("regulation_code", "GR-001"),
        )

        # Apply personality modifications
        mood = context.get("mood", "neutral")
        response = self._apply_mood_modifier(response, profile, mood)
        response = self._apply_personality_traits(response, profile["traits"])

        return response

    def _apply_mood_modifier(
        self, response: str, profile: dict[str, Any], mood: str
    ) -> str:
        """Apply mood-specific modifications to response."""
        mood_modifiers = profile.get("mood_modifiers", {})
        if mood not in mood_modifiers:
            return response

        modifier = mood_modifiers[mood]

        # Apply mood-specific adjustments
        if "vigilant" in modifier:
            response += " I need to ask a few security questions."
        elif "enthusiastic" in modifier:
            response = response.replace(".", "!")
        elif "terse" in modifier:
            # Make response more direct
            response = response.split(".")[0] + "."
        elif "eager" in modifier:
            response += " I have some excellent options for you!"

        return response

    def _apply_personality_traits(self, response: str, traits: list[str]) -> str:
        """Apply personality trait modifications to response."""
        # Modify response based on dominant traits
        if "verbose" in traits:
            # Add additional detail for verbose characters
            if not any(
                phrase in response
                for phrase in ["you know", "fascinating", "particularly"]
            ):
                response += " There's quite a bit more to discuss on this topic if you're interested."

        if "cautious" in traits:
            # Add cautionary language
            if "I should mention" not in response:
                response += " I should mention that proper procedures must be followed."

        if "friendly" in traits and "professional" not in traits:
            # Make response warmer
            response = response.replace("Good day", "Good day, friend")
            response = response.replace("Welcome", "Welcome!")

        return response

    def get_archetype_knowledge_areas(self, npc_archetype: str) -> list[str]:
        """Get knowledge areas this NPC archetype would be familiar with."""
        profile = self.personality_profiles.get(
            npc_archetype, self.personality_profiles["generic"]
        )
        return profile.get("knowledge_focus", ["general_information"])

    def get_archetype_concerns(self, npc_archetype: str) -> list[str]:
        """Get primary concerns for this NPC archetype."""
        profile = self.personality_profiles.get(
            npc_archetype, self.personality_profiles["generic"]
        )
        return profile.get("concerns", ["general_well_being"])

    def suggest_conversation_topics(
        self, npc_archetype: str, context: dict[str, Any]
    ) -> list[str]:
        """Suggest conversation topics appropriate for this NPC."""
        knowledge_areas = self.get_archetype_knowledge_areas(npc_archetype)
        concerns = self.get_archetype_concerns(npc_archetype)

        topics = []

        # Add knowledge-based topics
        for area in knowledge_areas[:3]:  # Limit to top 3
            topics.append(f"Ask about {area.replace('_', ' ')}")

        # Add concern-based topics
        for concern in concerns[:2]:  # Limit to top 2
            topics.append(f"Discuss {concern.replace('_', ' ')}")

        # Add context-specific topics
        location_type = context.get("location_type", "")
        if location_type:
            topics.append(f"Ask about this {location_type}")

        return topics[:5]  # Return max 5 suggestions
