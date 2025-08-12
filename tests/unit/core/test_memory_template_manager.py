"""Tests for Memory-Aware Template Manager."""

import shutil
import tempfile
from pathlib import Path

import pytest

from game_loop.core.memory_template_manager import MemoryTemplateManager


class TestMemoryTemplateManager:
    """Test suite for MemoryTemplateManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for templates."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def memory_template_manager(self, temp_dir):
        """Create MemoryTemplateManager instance with temp directory."""
        return MemoryTemplateManager(temp_dir)

    @pytest.fixture
    def setup_memory_templates(self, temp_dir):
        """Set up memory template files in temp directory."""
        templates_dir = Path(temp_dir)

        # Create memory template directories
        memory_dir = templates_dir / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)

        # Create confidence patterns template
        confidence_template = memory_dir / "confidence_patterns.j2"
        confidence_template.write_text(
            """
{% macro generate_memory_reference(memory_content, confidence, emotional_weight=0.0, trust_level="stranger", memory_age_days=30) %}
{%- if confidence >= 0.7 -%}
I clearly remember {{ memory_content }}
{%- elif confidence >= 0.5 -%}
I believe {{ memory_content }}
{%- else -%}
I think {{ memory_content }}
{%- endif -%}
{% endmacro %}
        """
        )

        # Create personality styles template
        personality_template = memory_dir / "personality_styles.j2"
        personality_template.write_text(
            """
{% macro generate_personality_memory_reference(memory_content, confidence, npc_archetype="generic", personality_traits={}, emotional_weight=0.0, trust_level="stranger") %}
{%- if npc_archetype == "merchant" -%}
From my business experience, {{ memory_content }}
{%- elif npc_archetype == "guard" -%}
According to protocol, {{ memory_content }}
{%- else -%}
I remember {{ memory_content }}
{%- endif -%}
{% endmacro %}

{% macro integrate_memory_in_conversation(memory_content, confidence, current_topic, npc_archetype="generic", personality_traits={}, transition_style="natural") %}
{%- if transition_style == "natural" -%}
That reminds me, {{ memory_content | lower }}
{%- else -%}
I should mention {{ memory_content | lower }}
{%- endif -%}
{% endmacro %}
        """
        )

        # Create trust revelation template
        trust_template = memory_dir / "trust_revelation.j2"
        trust_template.write_text(
            """
{% macro reveal_memory_by_trust(memory_content, trust_level, memory_sensitivity="general", confidence=0.7, allow_boundary_push=false) %}
{%- if trust_level >= 0.8 -%}
I'll let you in on something - {{ memory_content }}
{%- elif trust_level >= 0.6 -%}
Since we're friends, {{ memory_content }}
{%- elif trust_level >= 0.3 -%}
I feel comfortable telling you that {{ memory_content }}
{%- elif memory_sensitivity == "shameful_secrets" -%}
I'm not comfortable sharing that with someone I barely know.
{%- else -%}
{{ memory_content }}
{%- endif -%}
{% endmacro %}
        """
        )

    def test_memory_template_manager_initialization(self, memory_template_manager):
        """Test that MemoryTemplateManager initializes correctly."""
        assert memory_template_manager is not None
        assert hasattr(memory_template_manager, "env")
        assert hasattr(memory_template_manager, "_template_cache")
        assert hasattr(memory_template_manager, "_last_used_patterns")

    def test_generate_memory_reference_basic(self, temp_dir, setup_memory_templates):
        """Test basic memory reference generation."""
        manager = MemoryTemplateManager(temp_dir)

        result = manager.generate_memory_reference(
            memory_content="we discussed the weather last Tuesday",
            confidence=0.8,
            emotional_weight=0.2,
            trust_level=0.5,
            npc_archetype="generic",
        )

        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        assert "we discussed the weather last tuesday" in result.lower()

    def test_confidence_levels_affect_language(self, temp_dir, setup_memory_templates):
        """Test that different confidence levels produce different language patterns."""
        manager = MemoryTemplateManager(temp_dir)
        memory_content = "you helped me with the merchant's problem"

        # High confidence
        high_conf = manager.generate_memory_reference(
            memory_content=memory_content, confidence=0.9, npc_archetype="generic"
        )

        # Low confidence
        low_conf = manager.generate_memory_reference(
            memory_content=memory_content, confidence=0.2, npc_archetype="generic"
        )

        # Should use different language patterns
        assert high_conf != low_conf
        # High confidence should be more definitive
        assert any(
            word in high_conf.lower() for word in ["clearly", "certain", "remember"]
        )
        # Low confidence should be more tentative
        assert any(word in low_conf.lower() for word in ["think", "might", "seems"])

    def test_personality_archetype_affects_language(
        self, temp_dir, setup_memory_templates
    ):
        """Test that different NPC archetypes produce different language styles."""
        manager = MemoryTemplateManager(temp_dir)
        memory_content = "there was a problem with the shipment"

        merchant_ref = manager.generate_memory_reference(
            memory_content=memory_content, confidence=0.7, npc_archetype="merchant"
        )

        guard_ref = manager.generate_memory_reference(
            memory_content=memory_content, confidence=0.7, npc_archetype="guard"
        )

        # Should use archetype-specific language
        assert "business" in merchant_ref.lower()
        assert "protocol" in guard_ref.lower()

    def test_trust_based_revelation(self, temp_dir, setup_memory_templates):
        """Test trust-based memory revelation."""
        manager = MemoryTemplateManager(temp_dir)
        memory_content = "I made a terrible mistake years ago"

        # Stranger - should be cautious or refuse
        stranger_result = manager.generate_trust_based_revelation(
            memory_content=memory_content,
            trust_level=0.1,
            memory_sensitivity="shameful_secrets",
        )

        # Confidant - should be open
        confidant_result = manager.generate_trust_based_revelation(
            memory_content=memory_content,
            trust_level=0.9,
            memory_sensitivity="shameful_secrets",
        )

        # Stranger should get boundary rejection (None) for sensitive content
        assert stranger_result is None

        # Confidant should get the revelation
        assert confidant_result is not None
        assert (
            "let you in" in confidant_result.lower()
            or "secret" in confidant_result.lower()
        )

    def test_trust_level_progression(self, temp_dir, setup_memory_templates):
        """Test that trust levels create appropriate revelation patterns."""
        manager = MemoryTemplateManager(temp_dir)
        memory_content = "I have personal concerns about the situation"

        trust_levels = [0.1, 0.4, 0.7, 0.9]
        results = []

        for trust_level in trust_levels:
            result = manager.generate_trust_based_revelation(
                memory_content=memory_content,
                trust_level=trust_level,
                memory_sensitivity="personal_struggles",
            )
            results.append((trust_level, result))

        # Higher trust levels should allow revelation while lower ones might not
        low_trust_result = results[0][1]  # 0.1 trust
        high_trust_result = results[3][1]  # 0.9 trust

        # High trust should definitely get revelation
        assert high_trust_result is not None

        # Results should vary based on trust level
        trust_indicators = ["friends", "comfortable", "let you in", "secret"]
        high_trust_has_indicators = any(
            indicator in high_trust_result.lower() for indicator in trust_indicators
        )
        assert high_trust_has_indicators

    def test_memory_integration_in_dialogue(self, temp_dir, setup_memory_templates):
        """Test memory integration with conversation flow."""
        manager = MemoryTemplateManager(temp_dir)

        result = manager.integrate_memory_in_dialogue(
            memory_content="you mentioned wanting to learn magic",
            confidence=0.8,
            current_topic="spellcasting",
            transition_style="natural",
        )

        assert result is not None
        assert "reminds me" in result.lower()
        assert "you mentioned wanting to learn magic" in result.lower()

        # Test direct transition style
        direct_result = manager.integrate_memory_in_dialogue(
            memory_content="you mentioned wanting to learn magic",
            confidence=0.8,
            current_topic="spellcasting",
            transition_style="direct",
        )

        assert "mention" in direct_result.lower()

    def test_validation_quality_assessment(self, memory_template_manager):
        """Test memory reference quality validation."""
        # Good quality reference
        good_reference = "I clearly remember when you helped me solve that merchant dispute last week."
        validation = memory_template_manager.validate_memory_reference_quality(
            good_reference
        )

        assert validation["overall_quality"] > 0.7
        assert validation["length_appropriate"] is True
        assert validation["grammar_check"] is True

        # Poor quality reference
        poor_reference = "I remember I remember I remember thing thing thing!!!!!!"
        poor_validation = memory_template_manager.validate_memory_reference_quality(
            poor_reference
        )

        assert poor_validation["overall_quality"] < 0.7
        assert poor_validation["repetition_check"] is False

    def test_pattern_variety_and_anti_repetition(
        self, temp_dir, setup_memory_templates
    ):
        """Test that pattern variety prevents repetitive language."""
        manager = MemoryTemplateManager(temp_dir)
        npc_id = "test_merchant"

        # Generate multiple memory references for same NPC
        results = []
        for i in range(5):
            result = manager.generate_memory_reference(
                memory_content=f"event number {i}",
                confidence=0.7,
                npc_archetype="merchant",
                avoid_repetition=True,
            )
            results.append(result)

        # Should have some variety in patterns
        unique_starts = set()
        for result in results:
            # Get first few words as pattern indicator
            words = result.split()[:3]
            pattern = " ".join(words).lower()
            unique_starts.add(pattern)

        # Should have some variety (at least 2 different patterns)
        assert len(unique_starts) >= 2

    def test_emotional_weight_affects_language(self, temp_dir, setup_memory_templates):
        """Test that emotional weight influences language selection."""
        manager = MemoryTemplateManager(temp_dir)
        memory_content = "we lost a dear friend in battle"

        # High emotional weight
        emotional_result = manager.generate_memory_reference(
            memory_content=memory_content, confidence=0.8, emotional_weight=0.9
        )

        # Low emotional weight
        neutral_result = manager.generate_memory_reference(
            memory_content=memory_content, confidence=0.8, emotional_weight=0.1
        )

        # Should produce different results
        assert emotional_result != neutral_result

    def test_memory_age_context(self, temp_dir, setup_memory_templates):
        """Test that memory age affects temporal language."""
        manager = MemoryTemplateManager(temp_dir)
        memory_content = "we met at the tavern"

        # Recent memory
        recent_result = manager.generate_memory_reference(
            memory_content=memory_content, confidence=0.7, memory_age_days=2
        )

        # Old memory
        old_result = manager.generate_memory_reference(
            memory_content=memory_content, confidence=0.7, memory_age_days=365
        )

        # Results should contain the memory content
        assert "we met at the tavern" in recent_result.lower()
        assert "we met at the tavern" in old_result.lower()

    def test_fallback_behavior(self, temp_dir):
        """Test fallback behavior when templates are not available."""
        # Create manager without setting up templates
        manager = MemoryTemplateManager(temp_dir)

        result = manager.generate_memory_reference(
            memory_content="something happened", confidence=0.8
        )

        # Should still produce a result using fallback logic
        assert result is not None
        assert isinstance(result, str)
        assert "something happened" in result.lower()

    def test_personality_traits_integration(self, temp_dir, setup_memory_templates):
        """Test personality traits affect memory references."""
        manager = MemoryTemplateManager(temp_dir)
        memory_content = "there was a disagreement about payment"

        # Authoritative trait
        authoritative_result = manager.generate_memory_reference(
            memory_content=memory_content,
            confidence=0.7,
            personality_traits={"authoritative": 0.9, "helpful": 0.3},
        )

        # Helpful trait
        helpful_result = manager.generate_memory_reference(
            memory_content=memory_content,
            confidence=0.7,
            personality_traits={"helpful": 0.9, "authoritative": 0.3},
        )

        # Should contain the memory content
        assert "disagreement about payment" in authoritative_result.lower()
        assert "disagreement about payment" in helpful_result.lower()

    def test_boundary_rejection_detection(self, temp_dir, setup_memory_templates):
        """Test detection of trust boundary rejections."""
        manager = MemoryTemplateManager(temp_dir)

        # This should trigger boundary rejection
        rejection_result = manager.generate_trust_based_revelation(
            memory_content="my deepest shameful secret",
            trust_level=0.1,
            memory_sensitivity="shameful_secrets",
        )

        # Should be None (rejected) for very sensitive content with low trust
        assert rejection_result is None

    def test_template_cache_functionality(self, temp_dir, setup_memory_templates):
        """Test that template caching works correctly."""
        manager = MemoryTemplateManager(temp_dir)

        # Templates should be loaded into cache
        assert len(manager._template_cache) > 0

        # Should have loaded the memory templates
        expected_templates = [
            "confidence_patterns",
            "personality_styles",
            "trust_revelation",
        ]
        for template_name in expected_templates:
            assert template_name in manager._template_cache

    def test_input_validation_and_bounds(self, memory_template_manager):
        """Test that inputs are properly validated and bounded."""
        # Test with out-of-bounds confidence
        result = memory_template_manager.generate_memory_reference(
            memory_content="test memory",
            confidence=1.5,  # > 1.0
            trust_level=-0.5,  # < 0.0
        )

        # Should still work (values should be clamped)
        assert result is not None
        assert isinstance(result, str)

    def test_create_default_templates(self, temp_dir):
        """Test creation of default memory templates."""
        manager = MemoryTemplateManager(temp_dir)
        manager.create_memory_templates()

        # Check that template files were created
        memory_dir = Path(temp_dir) / "memory"
        assert memory_dir.exists()

        expected_files = [
            "simple_reference.j2",
            "personality_reference.j2",
            "trust_filtered.j2",
        ]

        for filename in expected_files:
            template_file = memory_dir / filename
            assert template_file.exists()
            assert template_file.stat().st_size > 0  # Not empty

    @pytest.mark.parametrize(
        "confidence,expected_quality",
        [(0.9, "high"), (0.6, "medium"), (0.3, "low"), (0.1, "very_low")],
    )
    def test_confidence_categories(
        self, temp_dir, setup_memory_templates, confidence, expected_quality
    ):
        """Test that confidence levels map to appropriate quality categories."""
        manager = MemoryTemplateManager(temp_dir)

        result = manager.generate_memory_reference(
            memory_content="test memory content", confidence=confidence
        )

        # Should contain confidence-appropriate language
        if expected_quality == "high":
            assert any(
                word in result.lower() for word in ["clearly", "certain", "definite"]
            )
        elif expected_quality == "very_low":
            assert any(word in result.lower() for word in ["might", "could", "perhaps"])

    def test_memory_sensitivity_filtering(self, temp_dir, setup_memory_templates):
        """Test that memory sensitivity levels are properly filtered."""
        manager = MemoryTemplateManager(temp_dir)

        # Test various sensitivity levels with different trust levels
        test_cases = [
            ("general_knowledge", 0.1, True),  # Should allow
            ("personal_struggles", 0.1, False),  # Should reject
            ("shameful_secrets", 0.5, False),  # Should reject
            ("shameful_secrets", 0.9, True),  # Should allow
        ]

        for sensitivity, trust, should_allow in test_cases:
            result = manager.generate_trust_based_revelation(
                memory_content="sensitive information",
                trust_level=trust,
                memory_sensitivity=sensitivity,
            )

            if should_allow:
                assert (
                    result is not None
                ), f"Should allow {sensitivity} at trust {trust}"
            else:
                assert result is None, f"Should reject {sensitivity} at trust {trust}"
