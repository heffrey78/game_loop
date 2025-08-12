"""Memory-Aware Template Manager for natural language memory integration."""

import random
import re
from typing import Any

import jinja2

from game_loop.core.template_manager import TemplateManager


class MemoryTemplateManager(TemplateManager):
    """Extended template manager with memory-aware response capabilities."""

    def __init__(self, template_dir: str = "templates"):
        super().__init__(template_dir)
        self._template_cache: dict[str, Any] = {}
        self._last_used_patterns: dict[str, list[str]] = {}
        self._setup_memory_templates()

    def _setup_memory_templates(self) -> None:
        """Set up memory-specific template environment and filters."""
        # Add memory-specific filters
        self.env.filters["validate_memory_reference"] = self._validate_memory_reference
        self.env.filters["randomize_pattern"] = self._randomize_pattern
        self.env.filters["assess_naturalness"] = self._assess_naturalness

        # Load memory template macros
        self._load_memory_macros()

    def _load_memory_macros(self) -> None:
        """Load memory template macros for reuse."""
        try:
            # Load confidence patterns
            if self._template_exists("memory/confidence_patterns.j2"):
                self._template_cache["confidence_patterns"] = self.env.get_template(
                    "memory/confidence_patterns.j2"
                )

            # Load personality styles
            if self._template_exists("memory/personality_styles.j2"):
                self._template_cache["personality_styles"] = self.env.get_template(
                    "memory/personality_styles.j2"
                )

            # Load trust revelation patterns
            if self._template_exists("memory/trust_revelation.j2"):
                self._template_cache["trust_revelation"] = self.env.get_template(
                    "memory/trust_revelation.j2"
                )

        except jinja2.TemplateError:
            # Templates not available - will fall back to basic patterns
            pass

    def generate_memory_reference(
        self,
        memory_content: str,
        confidence: float,
        emotional_weight: float = 0.0,
        trust_level: float = 0.0,
        npc_archetype: str = "generic",
        personality_traits: dict[str, float] | None = None,
        memory_age_days: int = 30,
        memory_sensitivity: str = "general",
        avoid_repetition: bool = True,
    ) -> str:
        """
        Generate a natural memory reference using confidence and personality patterns.

        Args:
            memory_content: The actual memory content to reference
            confidence: Memory confidence score (0.0-1.0)
            emotional_weight: Emotional significance (0.0-1.0)
            trust_level: Relationship trust level (0.0-1.0)
            npc_archetype: NPC personality archetype (merchant, guard, etc.)
            personality_traits: Dict of personality trait strengths
            memory_age_days: How many days ago the memory occurred
            memory_sensitivity: Sensitivity level of the memory content
            avoid_repetition: Whether to avoid recently used patterns

        Returns:
            Natural language memory reference string
        """
        if personality_traits is None:
            personality_traits = {}

        # Validate inputs
        confidence = max(0.0, min(1.0, confidence))
        trust_level = max(0.0, min(1.0, trust_level))

        try:
            # Use personality-specific patterns if available
            if "personality_styles" in self._template_cache:
                template = self._template_cache["personality_styles"]

                # Prepare context for template
                context = {
                    "memory_content": memory_content,
                    "confidence": confidence,
                    "npc_archetype": npc_archetype,
                    "personality_traits": personality_traits,
                    "emotional_weight": emotional_weight,
                    "trust_level": self._get_trust_category(trust_level),
                }

                # Generate using personality macro
                result = template.module.generate_personality_memory_reference(
                    **context
                )

                # Apply anti-repetition if requested
                if avoid_repetition:
                    result = self._ensure_pattern_variety(result, npc_archetype)

                return self._validate_and_clean_result(result)

        except (jinja2.TemplateError, AttributeError):
            pass

        # Fallback to confidence-based patterns
        return self._generate_confidence_based_reference(
            memory_content, confidence, emotional_weight, trust_level
        )

    def generate_trust_based_revelation(
        self,
        memory_content: str,
        trust_level: float,
        memory_sensitivity: str = "general",
        confidence: float = 0.7,
        allow_boundary_push: bool = False,
    ) -> str | None:
        """
        Generate trust-appropriate memory revelation.

        Args:
            memory_content: Memory content to potentially reveal
            trust_level: Current relationship trust level (0.0-1.0)
            memory_sensitivity: Sensitivity classification of memory
            confidence: Confidence in the memory
            allow_boundary_push: Allow slight trust boundary violations

        Returns:
            Memory revelation string or None if inappropriate for trust level
        """
        try:
            if "trust_revelation" in self._template_cache:
                template = self._template_cache["trust_revelation"]

                context = {
                    "memory_content": memory_content,
                    "trust_level": trust_level,
                    "memory_sensitivity": memory_sensitivity,
                    "confidence": confidence,
                    "allow_boundary_push": allow_boundary_push,
                }

                # Use trust-based revelation macro
                result = template.module.reveal_memory_by_trust(**context)

                # Check if result indicates boundary rejection
                if self._is_boundary_rejection(result):
                    return None

                return self._validate_and_clean_result(result)

        except (jinja2.TemplateError, AttributeError):
            pass

        # Fallback logic
        return self._basic_trust_filter(memory_content, trust_level, memory_sensitivity)

    def integrate_memory_in_dialogue(
        self,
        memory_content: str,
        confidence: float,
        current_topic: str,
        npc_archetype: str = "generic",
        personality_traits: dict[str, float] | None = None,
        transition_style: str = "natural",
    ) -> str:
        """
        Integrate a memory reference naturally into ongoing dialogue.

        Args:
            memory_content: Memory to integrate
            confidence: Memory confidence score
            current_topic: Current conversation topic
            npc_archetype: NPC personality archetype
            personality_traits: Personality trait strengths
            transition_style: How to transition to the memory (natural, direct, etc.)

        Returns:
            Integrated memory reference with appropriate transition
        """
        if personality_traits is None:
            personality_traits = {}

        try:
            if "personality_styles" in self._template_cache:
                template = self._template_cache["personality_styles"]

                context = {
                    "memory_content": memory_content,
                    "confidence": confidence,
                    "current_topic": current_topic,
                    "npc_archetype": npc_archetype,
                    "personality_traits": personality_traits,
                    "transition_style": transition_style,
                }

                result = template.module.integrate_memory_in_conversation(**context)
                return self._validate_and_clean_result(result)

        except (jinja2.TemplateError, AttributeError):
            pass

        # Simple fallback integration
        transitions = {
            "natural": [
                "That reminds me,",
                "Speaking of that,",
                "Oh, that brings back",
            ],
            "direct": ["I should mention", "You should know", "Let me tell you"],
        }

        transition_phrases = transitions.get(transition_style, transitions["natural"])
        transition = random.choice(transition_phrases)

        return f"{transition} {memory_content.lower()}"

    def validate_memory_reference_quality(
        self, reference: str, target_confidence: float = 0.9
    ) -> dict[str, Any]:
        """
        Validate the quality and naturalness of a memory reference.

        Args:
            reference: Generated memory reference to validate
            target_confidence: Target quality confidence threshold

        Returns:
            Validation results with metrics and suggestions
        """
        results = {
            "naturalness_score": 0.0,
            "confidence_appropriateness": 0.0,
            "repetition_check": True,
            "length_appropriate": True,
            "grammar_check": True,
            "overall_quality": 0.0,
            "suggestions": [],
        }

        # Check naturalness
        results["naturalness_score"] = self._assess_naturalness(reference)

        # Check length (1-3 sentences is ideal)
        sentence_count = len(re.findall(r"[.!?]+", reference))
        results["length_appropriate"] = 1 <= sentence_count <= 3
        if not results["length_appropriate"]:
            results["suggestions"].append(
                f"Reference has {sentence_count} sentences; 1-3 is ideal"
            )

        # Basic grammar check
        results["grammar_check"] = self._basic_grammar_check(reference)
        if not results["grammar_check"]:
            results["suggestions"].append("Grammar or formatting issues detected")

        # Check for repetitive patterns
        results["repetition_check"] = not self._has_repetitive_patterns(reference)
        if not results["repetition_check"]:
            results["suggestions"].append("Contains repetitive language patterns")

        # Calculate overall quality
        metrics = [
            results["naturalness_score"],
            1.0 if results["length_appropriate"] else 0.5,
            1.0 if results["grammar_check"] else 0.3,
            1.0 if results["repetition_check"] else 0.7,
        ]
        results["overall_quality"] = sum(metrics) / len(metrics)

        return results

    # Private helper methods

    def _generate_confidence_based_reference(
        self,
        memory_content: str,
        confidence: float,
        emotional_weight: float,
        trust_level: float,
    ) -> str:
        """Fallback confidence-based memory reference generation."""
        if confidence >= 0.7:
            if emotional_weight > 0.7:
                patterns = [
                    "I'll never forget",
                    "I clearly remember",
                    "That moment is burned into my memory",
                ]
            else:
                patterns = [
                    "I remember clearly",
                    "I'm certain that",
                    "I recall exactly",
                ]
        elif confidence >= 0.5:
            patterns = ["I believe", "From what I recall", "I'm pretty sure"]
        elif confidence >= 0.3:
            patterns = ["I think", "That rings a bell", "I seem to recall"]
        else:
            patterns = [
                "I might be wrong, but",
                "I could be imagining this",
                "There's something familiar",
            ]

        pattern = random.choice(patterns)
        return f"{pattern} {memory_content.lower()}"

    def _get_trust_category(self, trust_level: float) -> str:
        """Convert numeric trust level to category."""
        if trust_level < 0.3:
            return "stranger"
        elif trust_level < 0.6:
            return "acquaintance"
        elif trust_level < 0.8:
            return "friend"
        else:
            return "confidant"

    def _basic_trust_filter(
        self, memory_content: str, trust_level: float, memory_sensitivity: str
    ) -> str | None:
        """Basic trust-based filtering for memory content."""
        sensitive_types = [
            "shameful_secrets",
            "intimate_relationships",
            "trauma",
            "family_secrets",
        ]
        personal_types = ["personal_struggles", "emotional_memories", "past_mistakes"]

        if memory_sensitivity in sensitive_types and trust_level < 0.8:
            return None
        elif memory_sensitivity in personal_types and trust_level < 0.6:
            return None
        elif trust_level < 0.3 and memory_sensitivity != "general_knowledge":
            return None

        return memory_content

    def _ensure_pattern_variety(self, result: str, npc_id: str) -> str:
        """Ensure pattern variety by avoiding recently used phrases."""
        # Track last used patterns per NPC
        if npc_id not in self._last_used_patterns:
            self._last_used_patterns[npc_id] = []

        # Extract key phrases from result
        key_phrases = self._extract_key_phrases(result)

        # Check for repetition
        recent_patterns = self._last_used_patterns[npc_id][-5:]  # Last 5 patterns
        for phrase in key_phrases:
            if phrase in recent_patterns:
                # Try to regenerate with different pattern
                return self._regenerate_with_different_pattern(result, phrase)

        # Update pattern history
        if key_phrases:
            self._last_used_patterns[npc_id].append(key_phrases[0])
            # Keep only last 10 patterns
            self._last_used_patterns[npc_id] = self._last_used_patterns[npc_id][-10:]

        return result

    def _extract_key_phrases(self, text: str) -> list[str]:
        """Extract key phrases for repetition checking."""
        # Look for common memory reference starters
        patterns = [
            r"^([^,]*remember[^,]*)",
            r"^([^,]*recall[^,]*)",
            r"^([^,]*think[^,]*)",
            r"^([^,]*believe[^,]*)",
            r"^([^,]*certain[^,]*)",
        ]

        phrases = []
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                phrases.append(match.group(1).strip().lower())

        return phrases

    def _regenerate_with_different_pattern(
        self, original: str, avoid_phrase: str
    ) -> str:
        """Attempt to regenerate avoiding a specific phrase pattern."""
        # Simple approach - if we detect repetition, modify the start
        alternatives = {
            "i remember": "from what i recall",
            "i recall": "i believe",
            "i think": "it seems to me",
            "i believe": "i'm pretty sure",
            "i'm certain": "i clearly remember",
        }

        for original_phrase, alternative in alternatives.items():
            if original_phrase in avoid_phrase.lower():
                return original.replace(avoid_phrase, alternative, 1)

        return original

    def _validate_and_clean_result(self, result: str) -> str:
        """Validate and clean the generated result."""
        if not result or not isinstance(result, str):
            return "I have some thoughts about that."

        # Clean up extra whitespace
        result = re.sub(r"\s+", " ", result.strip())

        # Ensure proper capitalization
        if result and not result[0].isupper():
            result = result[0].upper() + result[1:]

        # Ensure proper ending punctuation
        if result and result[-1] not in ".!?":
            result += "."

        return result

    def _is_boundary_rejection(self, text: str) -> bool:
        """Check if text indicates a trust boundary rejection."""
        rejection_indicators = [
            "not comfortable",
            "too personal",
            "prefer to keep",
            "don't know each other",
            "rather personal",
            "not ready to",
            "never told anyone",
        ]

        text_lower = text.lower()
        return any(indicator in text_lower for indicator in rejection_indicators)

    def _validate_memory_reference(self, text: str) -> bool:
        """Template filter to validate memory references."""
        return self.validate_memory_reference_quality(text)["overall_quality"] > 0.7

    def _randomize_pattern(self, patterns: list[str]) -> str:
        """Template filter to randomize pattern selection."""
        return random.choice(patterns) if patterns else ""

    def _assess_naturalness(self, text: str) -> float:
        """Assess how natural a memory reference sounds."""
        # Simple heuristics for naturalness
        score = 1.0

        # Penalize overly formal language
        formal_indicators = ["furthermore", "moreover", "consequently", "therefore"]
        if any(word in text.lower() for word in formal_indicators):
            score -= 0.2

        # Reward conversational language
        conversational_indicators = ["you know", "i mean", "actually", "really"]
        if any(word in text.lower() for word in conversational_indicators):
            score += 0.1

        # Penalize very long sentences
        if len(text.split()) > 30:
            score -= 0.3

        # Reward appropriate emotional language
        emotional_indicators = ["feel", "felt", "emotion", "heart", "moved"]
        if any(word in text.lower() for word in emotional_indicators):
            score += 0.1

        return max(0.0, min(1.0, score))

    def _basic_grammar_check(self, text: str) -> bool:
        """Basic grammar and formatting check."""
        # Check for balanced quotes
        quote_count = text.count('"')
        if quote_count % 2 != 0:
            return False

        # Check for reasonable sentence structure
        if len(re.findall(r"[A-Z]", text)) == 0:
            return False

        # Check for excessive punctuation
        punct_ratio = len(re.findall(r"[.!?,:;]", text)) / max(len(text.split()), 1)
        if punct_ratio > 0.3:
            return False

        return True

    def _has_repetitive_patterns(self, text: str) -> bool:
        """Check for repetitive language patterns."""
        words = text.lower().split()
        if len(words) < 4:
            return False

        # Check for repeated words in close proximity
        for i in range(len(words) - 2):
            if words[i] in words[i + 1 : i + 4]:
                return True

        return False

    def create_memory_templates(self) -> None:
        """Create default memory template files if they don't exist."""
        memory_templates = {
            "memory/simple_reference.j2": """
{% from 'memory/confidence_patterns.j2' import generate_memory_reference %}
{{ generate_memory_reference(memory_content, confidence, emotional_weight, trust_level, memory_age_days) }}
            """.strip(),
            "memory/personality_reference.j2": """
{% from 'memory/personality_styles.j2' import generate_personality_memory_reference %}
{{ generate_personality_memory_reference(memory_content, confidence, npc_archetype, personality_traits, emotional_weight, trust_level) }}
            """.strip(),
            "memory/trust_filtered.j2": """
{% from 'memory/trust_revelation.j2' import reveal_memory_by_trust %}
{{ reveal_memory_by_trust(memory_content, trust_level, memory_sensitivity, confidence, allow_boundary_push) }}
            """.strip(),
        }

        for template_name, content in memory_templates.items():
            template_path = self.template_dir / template_name
            template_path.parent.mkdir(parents=True, exist_ok=True)

            if not template_path.exists():
                template_path.write_text(content)
