"""Tests for NPCPersonalityEngine."""

from game_loop.core.dialogue.personality_engine import NPCPersonalityEngine


class TestNPCPersonalityEngine:
    """Test cases for NPCPersonalityEngine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = NPCPersonalityEngine()

    def test_initialization(self):
        """Test engine initialization."""
        assert self.engine is not None
        assert hasattr(self.engine, "personality_profiles")
        assert "security_guard" in self.engine.personality_profiles
        assert "scholar" in self.engine.personality_profiles

    def test_security_guard_greeting(self):
        """Test security guard personality greeting."""
        context = {"time_of_day": "morning", "player_name": "TestPlayer"}

        response = self.engine.generate_personality_response("security_guard", context)

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        # Security guards should be formal and professional
        assert any(
            word in response.lower()
            for word in [
                "identification",
                "credentials",
                "authorization",
                "clearance",
                "business",
                "good",
                "morning",
                "security",
            ]
        )

    def test_scholar_greeting(self):
        """Test scholar personality greeting."""
        context = {"time_of_day": "afternoon", "player_name": "TestPlayer"}

        response = self.engine.generate_personality_response("scholar", context)

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        # Scholars should be intellectually engaging (response should contain academic-related words)
        # The personality engine generates different responses, so we check for any scholarly indicators
        scholarly_indicators = [
            "knowledge",
            "learning",
            "research",
            "scholar",
            "seeker",
            "halls",
            "archives",
            "fellow",
            "academic",
            "inquiry",
            "study",
            "fascinating",
            "repository",
            "collection",
        ]
        assert any(word in response.lower() for word in scholarly_indicators)

    def test_topic_response(self):
        """Test topic-specific responses."""
        context = {"time_of_day": "day", "player_name": "TestPlayer"}

        response = self.engine.generate_personality_response(
            "security_guard", context, "location_inquiry"
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    def test_get_archetype_knowledge_areas(self):
        """Test getting knowledge areas for archetypes."""
        security_knowledge = self.engine.get_archetype_knowledge_areas("security_guard")
        assert isinstance(security_knowledge, list)
        assert len(security_knowledge) > 0
        assert "building_layout" in security_knowledge

        scholar_knowledge = self.engine.get_archetype_knowledge_areas("scholar")
        assert isinstance(scholar_knowledge, list)
        assert len(scholar_knowledge) > 0
        assert "research" in " ".join(scholar_knowledge)

    def test_suggest_conversation_topics(self):
        """Test conversation topic suggestions."""
        context = {"location_type": "office"}

        topics = self.engine.suggest_conversation_topics("security_guard", context)
        assert isinstance(topics, list)
        assert len(topics) > 0

        # Should suggest security-related topics
        topic_text = " ".join(topics).lower()
        assert any(word in topic_text for word in ["security", "building", "access"])

    def test_unknown_archetype_fallback(self):
        """Test fallback for unknown archetype."""
        context = {"time_of_day": "day"}

        response = self.engine.generate_personality_response(
            "unknown_archetype", context
        )

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0

    def test_mood_modifiers(self):
        """Test mood modifier application."""
        context = {
            "time_of_day": "day",
            "player_name": "TestPlayer",
            "mood": "suspicious",
        }

        response = self.engine.generate_personality_response("security_guard", context)

        assert response is not None
        assert isinstance(response, str)
        # With suspicious mood, security guard should ask questions or show vigilance
        assert any(
            word in response.lower()
            for word in ["question", "need", "ask", "security", "must", "procedure"]
        )

    def test_personality_trait_application(self):
        """Test personality trait application to responses."""
        # Test verbose trait (scholar)
        context = {"time_of_day": "day"}

        response = self.engine.generate_personality_response("scholar", context)

        # Scholars tend to be more verbose
        assert len(response) > 20  # Should generate substantial response

    def test_context_formatting(self):
        """Test proper context variable formatting."""
        context = {
            "time_of_day": "evening",
            "security_level": "high",
            "zone_type": "restricted",
        }

        response = self.engine.generate_personality_response(
            "security_guard", context, "location_inquiry"
        )

        assert response is not None
        assert "{" not in response  # No unformatted template variables
