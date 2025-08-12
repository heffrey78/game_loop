"""Conversation flow templates and memory integration patterns."""

import enum
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class ConversationStage(enum.Enum):
    """Stages of relationship progression."""

    INITIAL_ENCOUNTER = "initial_encounter"
    ACQUAINTANCE = "acquaintance"
    RELATIONSHIP_BUILDING = "relationship_building"
    TRUST_DEVELOPMENT = "trust_development"
    DEEP_CONNECTION = "deep_connection"
    CONFIDANT = "confidant"


class TrustLevel(enum.Enum):
    """Trust levels for memory disclosure."""

    STRANGER = "stranger"  # 0.0-0.3
    ACQUAINTANCE = "acquaintance"  # 0.3-0.6
    FRIEND = "friend"  # 0.6-0.8
    CONFIDANT = "confidant"  # 0.8-1.0


class MemoryDisclosureThreshold(enum.Enum):
    """Memory disclosure thresholds with confidence ranges."""

    SUBTLE_HINTS = "subtle_hints"  # 0.3-0.5
    CLEAR_REFERENCES = "clear_references"  # 0.5-0.7
    DETAILED_MEMORIES = "detailed_memories"  # 0.7-0.9
    PERSONAL_SECRETS = "personal_secrets"  # 0.9+


@dataclass
class TransitionPhrase:
    """A phrase used for memory integration transitions."""

    text: str
    confidence_range: tuple[float, float]
    emotional_tone: str
    usage_context: str

    def matches_confidence(self, confidence: float) -> bool:
        """Check if phrase matches confidence level."""
        return self.confidence_range[0] <= confidence <= self.confidence_range[1]


@dataclass
class MemoryIntegrationPattern:
    """Pattern for integrating memories into conversation flow."""

    pattern_name: str
    disclosure_level: MemoryDisclosureThreshold
    trust_requirement: TrustLevel
    introduction_phrases: list[TransitionPhrase] = field(default_factory=list)
    continuation_phrases: list[TransitionPhrase] = field(default_factory=list)
    uncertainty_expressions: list[TransitionPhrase] = field(default_factory=list)
    emotional_modifiers: dict[str, list[str]] = field(default_factory=dict)
    flow_validation_rules: list[str] = field(default_factory=list)

    def get_appropriate_phrase(
        self,
        confidence: float,
        emotional_tone: str = "neutral",
        phrase_type: str = "introduction",
    ) -> TransitionPhrase | None:
        """Get appropriate transition phrase for context."""
        phrase_list = {
            "introduction": self.introduction_phrases,
            "continuation": self.continuation_phrases,
            "uncertainty": self.uncertainty_expressions,
        }.get(phrase_type, self.introduction_phrases)

        # Filter by confidence and emotion - prioritize exact matches
        exact_matches = [
            phrase
            for phrase in phrase_list
            if phrase.matches_confidence(confidence)
            and phrase.emotional_tone == emotional_tone
        ]

        if exact_matches:
            return exact_matches[0]

        # Fallback to neutral phrases if no exact emotional tone match
        neutral_matches = [
            phrase
            for phrase in phrase_list
            if phrase.matches_confidence(confidence)
            and phrase.emotional_tone == "neutral"
        ]

        return neutral_matches[0] if neutral_matches else None


@dataclass
class ConversationFlowTemplate:
    """Template for conversation flow management."""

    template_name: str
    stage: ConversationStage
    memory_patterns: list[MemoryIntegrationPattern] = field(default_factory=list)
    personality_modifiers: dict[str, float] = field(default_factory=dict)
    context_requirements: dict[str, Any] = field(default_factory=dict)
    flow_transitions: dict[ConversationStage, float] = field(default_factory=dict)

    def get_pattern_for_trust_level(
        self, trust_level: TrustLevel
    ) -> MemoryIntegrationPattern | None:
        """Get appropriate memory pattern for trust level."""
        for pattern in self.memory_patterns:
            if pattern.trust_requirement == trust_level:
                return pattern
        return None


class ConversationFlowLibrary:
    """Library of conversation flow templates and memory integration patterns."""

    def __init__(self) -> None:
        self.templates: dict[str, ConversationFlowTemplate] = {}
        self.transition_phrases: dict[str, list[TransitionPhrase]] = {}
        self.memory_patterns: dict[str, MemoryIntegrationPattern] = {}

        # Initialize built-in templates
        self._initialize_transition_phrases()
        self._initialize_memory_patterns()
        self._initialize_flow_templates()

    def _initialize_transition_phrases(self) -> None:
        """Initialize built-in transition phrases."""

        # Uncertainty expressions by confidence level
        uncertainty_phrases = [
            TransitionPhrase(
                "That rings a bell...", (0.2, 0.4), "neutral", "memory_recall"
            ),
            TransitionPhrase(
                "I vaguely recall...", (0.1, 0.3), "neutral", "memory_recall"
            ),
            TransitionPhrase(
                "If I remember correctly...", (0.4, 0.6), "neutral", "memory_recall"
            ),
            TransitionPhrase(
                "I think I remember...", (0.3, 0.5), "neutral", "memory_recall"
            ),
            TransitionPhrase("I believe...", (0.5, 0.7), "neutral", "memory_recall"),
            TransitionPhrase(
                "I clearly remember...", (0.7, 0.9), "confident", "memory_recall"
            ),
            TransitionPhrase(
                "I'll never forget...", (0.8, 1.0), "emotional", "memory_recall"
            ),
        ]

        # Confidence statements
        confidence_phrases = [
            TransitionPhrase(
                "I know for certain...", (0.8, 1.0), "confident", "knowledge_assertion"
            ),
            TransitionPhrase(
                "I'm quite sure...", (0.6, 0.8), "confident", "knowledge_assertion"
            ),
            TransitionPhrase(
                "From what I recall...", (0.4, 0.7), "neutral", "knowledge_assertion"
            ),
            TransitionPhrase(
                "As I understand it...", (0.5, 0.8), "neutral", "knowledge_assertion"
            ),
        ]

        # Emotional memory recalls
        emotional_phrases = [
            TransitionPhrase(
                "That brings back memories...",
                (0.5, 0.8),
                "nostalgic",
                "emotional_recall",
            ),
            TransitionPhrase(
                "I remember how that made me feel...",
                (0.6, 0.9),
                "emotional",
                "emotional_recall",
            ),
            TransitionPhrase(
                "That reminds me of...", (0.4, 0.7), "connected", "emotional_recall"
            ),
            TransitionPhrase(
                "I can still picture...", (0.7, 0.9), "vivid", "emotional_recall"
            ),
        ]

        self.transition_phrases = {
            "uncertainty": uncertainty_phrases,
            "confidence": confidence_phrases,
            "emotional": emotional_phrases,
        }

    def _initialize_memory_patterns(self) -> None:
        """Initialize memory integration patterns."""

        # Subtle hints pattern (0.3-0.5 confidence)
        subtle_pattern = MemoryIntegrationPattern(
            pattern_name="subtle_hints",
            disclosure_level=MemoryDisclosureThreshold.SUBTLE_HINTS,
            trust_requirement=TrustLevel.STRANGER,
            introduction_phrases=[
                TransitionPhrase(
                    "That sounds familiar...", (0.3, 0.5), "neutral", "hint"
                ),
                TransitionPhrase(
                    "Something about that...", (0.3, 0.5), "neutral", "hint"
                ),
                TransitionPhrase(
                    "I've heard something like that before...",
                    (0.3, 0.5),
                    "neutral",
                    "hint",
                ),
            ],
            uncertainty_expressions=self.transition_phrases["uncertainty"][:3],
            flow_validation_rules=[
                "avoid_specific_details",
                "use_vague_language",
                "maintain_mystery",
            ],
        )

        # Clear references pattern (0.5-0.7 confidence)
        clear_pattern = MemoryIntegrationPattern(
            pattern_name="clear_references",
            disclosure_level=MemoryDisclosureThreshold.CLEAR_REFERENCES,
            trust_requirement=TrustLevel.ACQUAINTANCE,
            introduction_phrases=[
                TransitionPhrase(
                    "I remember when...", (0.5, 0.7), "neutral", "reference"
                ),
                TransitionPhrase(
                    "That reminds me of...", (0.5, 0.7), "connected", "reference"
                ),
                TransitionPhrase(
                    "You mentioned something like that before...",
                    (0.5, 0.7),
                    "attentive",
                    "reference",
                ),
            ],
            continuation_phrases=[
                TransitionPhrase("And then...", (0.5, 0.7), "neutral", "continuation"),
                TransitionPhrase(
                    "Which made me think...", (0.5, 0.7), "thoughtful", "continuation"
                ),
            ],
            flow_validation_rules=[
                "reference_past_conversations",
                "maintain_consistency",
                "avoid_intimate_details",
            ],
        )

        # Detailed memories pattern (0.7-0.9 confidence)
        detailed_pattern = MemoryIntegrationPattern(
            pattern_name="detailed_memories",
            disclosure_level=MemoryDisclosureThreshold.DETAILED_MEMORIES,
            trust_requirement=TrustLevel.FRIEND,
            introduction_phrases=[
                TransitionPhrase(
                    "I clearly remember when you said...",
                    (0.7, 0.9),
                    "confident",
                    "detailed_recall",
                ),
                TransitionPhrase(
                    "That takes me back to when...",
                    (0.7, 0.9),
                    "nostalgic",
                    "detailed_recall",
                ),
                TransitionPhrase(
                    "I can still see the look on your face when...",
                    (0.7, 0.9),
                    "vivid",
                    "detailed_recall",
                ),
            ],
            emotional_modifiers={
                "happy": ["with a smile", "enthusiastically", "warmly"],
                "sad": ["with a heavy heart", "somberly", "thoughtfully"],
                "excited": ["with excitement", "eagerly", "with bright eyes"],
            },
            flow_validation_rules=[
                "include_specific_details",
                "reference_emotions",
                "maintain_authenticity",
            ],
        )

        # Personal secrets pattern (0.9+ confidence)
        secrets_pattern = MemoryIntegrationPattern(
            pattern_name="personal_secrets",
            disclosure_level=MemoryDisclosureThreshold.PERSONAL_SECRETS,
            trust_requirement=TrustLevel.CONFIDANT,
            introduction_phrases=[
                TransitionPhrase(
                    "I've never told anyone this, but...",
                    (0.9, 1.0),
                    "confessional",
                    "secret",
                ),
                TransitionPhrase(
                    "Since I trust you...", (0.9, 1.0), "trusting", "secret"
                ),
                TransitionPhrase(
                    "This is just between us...", (0.9, 1.0), "intimate", "secret"
                ),
                TransitionPhrase(
                    "I can share this with you...", (0.9, 1.0), "neutral", "secret"
                ),
            ],
            flow_validation_rules=[
                "ensure_high_trust",
                "reveal_personal_information",
                "create_intimacy",
            ],
        )

        self.memory_patterns = {
            "subtle_hints": subtle_pattern,
            "clear_references": clear_pattern,
            "detailed_memories": detailed_pattern,
            "personal_secrets": secrets_pattern,
        }

    def _initialize_flow_templates(self) -> None:
        """Initialize conversation flow templates."""

        # Initial encounter template
        initial_template = ConversationFlowTemplate(
            template_name="initial_encounter",
            stage=ConversationStage.INITIAL_ENCOUNTER,
            memory_patterns=[self.memory_patterns["subtle_hints"]],
            personality_modifiers={"cautious": 1.2, "open": 0.8, "mysterious": 1.5},
            context_requirements={"first_meeting": True, "trust_level": "stranger"},
            flow_transitions={ConversationStage.ACQUAINTANCE: 0.3},
        )

        # Relationship building template
        relationship_template = ConversationFlowTemplate(
            template_name="relationship_building",
            stage=ConversationStage.RELATIONSHIP_BUILDING,
            memory_patterns=[
                self.memory_patterns["subtle_hints"],
                self.memory_patterns["clear_references"],
            ],
            personality_modifiers={"friendly": 0.9, "reserved": 1.1, "talkative": 0.8},
            context_requirements={
                "prior_meetings": True,
                "trust_level": "acquaintance",
            },
            flow_transitions={ConversationStage.TRUST_DEVELOPMENT: 0.6},
        )

        # Trust development template
        trust_template = ConversationFlowTemplate(
            template_name="trust_development",
            stage=ConversationStage.TRUST_DEVELOPMENT,
            memory_patterns=[
                self.memory_patterns["clear_references"],
                self.memory_patterns["detailed_memories"],
            ],
            personality_modifiers={"trusting": 0.8, "cautious": 1.2, "open": 0.9},
            context_requirements={
                "developing_relationship": True,
                "trust_level": "friend",
            },
            flow_transitions={ConversationStage.DEEP_CONNECTION: 0.7},
        )

        # Deep connection template
        deep_template = ConversationFlowTemplate(
            template_name="deep_connection",
            stage=ConversationStage.DEEP_CONNECTION,
            memory_patterns=[
                self.memory_patterns["clear_references"],
                self.memory_patterns["detailed_memories"],
                self.memory_patterns["personal_secrets"],
            ],
            personality_modifiers={
                "empathetic": 0.7,
                "protective": 1.3,
                "vulnerable": 0.9,
            },
            context_requirements={
                "established_relationship": True,
                "trust_level": "friend",
            },
            flow_transitions={ConversationStage.CONFIDANT: 0.8},
        )

        # Acquaintance template (missing from original implementation)
        acquaintance_template = ConversationFlowTemplate(
            template_name="acquaintance",
            stage=ConversationStage.ACQUAINTANCE,
            memory_patterns=[self.memory_patterns["subtle_hints"]],
            personality_modifiers={"friendly": 0.9, "cautious": 1.1, "talkative": 0.8},
            context_requirements={"met_before": True, "trust_level": "acquaintance"},
            flow_transitions={ConversationStage.RELATIONSHIP_BUILDING: 0.5},
        )

        # Confidant template (missing from original implementation)
        confidant_template = ConversationFlowTemplate(
            template_name="confidant",
            stage=ConversationStage.CONFIDANT,
            memory_patterns=[
                self.memory_patterns["detailed_memories"],
                self.memory_patterns["personal_secrets"],
            ],
            personality_modifiers={
                "trustworthy": 0.7,
                "secretive": 1.5,
                "loyal": 0.6,
            },
            context_requirements={
                "deep_trust": True,
                "trust_level": "confidant",
            },
            flow_transitions={},  # Highest level, no further progression
        )

        self.templates = {
            "initial_encounter": initial_template,
            "acquaintance": acquaintance_template,
            "relationship_building": relationship_template,
            "trust_development": trust_template,
            "deep_connection": deep_template,
            "confidant": confidant_template,
        }

    def get_template(self, template_name: str) -> ConversationFlowTemplate | None:
        """Get flow template by name."""
        return self.templates.get(template_name)

    def get_template_for_stage(
        self, stage: ConversationStage
    ) -> ConversationFlowTemplate | None:
        """Get template for conversation stage."""
        for template in self.templates.values():
            if template.stage == stage:
                return template
        return None

    def get_memory_pattern(self, pattern_name: str) -> MemoryIntegrationPattern | None:
        """Get memory integration pattern by name."""
        return self.memory_patterns.get(pattern_name)

    def get_trust_level_from_relationship(
        self, relationship_score: float
    ) -> TrustLevel:
        """Convert relationship score to trust level."""
        if relationship_score >= 0.8:
            return TrustLevel.CONFIDANT
        elif relationship_score >= 0.6:
            return TrustLevel.FRIEND
        elif relationship_score >= 0.3:
            return TrustLevel.ACQUAINTANCE
        else:
            return TrustLevel.STRANGER

    def get_disclosure_threshold_from_confidence(
        self, confidence: float
    ) -> MemoryDisclosureThreshold:
        """Get appropriate disclosure threshold for confidence level."""
        if confidence >= 0.9:
            return MemoryDisclosureThreshold.PERSONAL_SECRETS
        elif confidence >= 0.7:
            return MemoryDisclosureThreshold.DETAILED_MEMORIES
        elif confidence >= 0.5:
            return MemoryDisclosureThreshold.CLEAR_REFERENCES
        elif confidence >= 0.3:
            return MemoryDisclosureThreshold.SUBTLE_HINTS
        else:
            return MemoryDisclosureThreshold.SUBTLE_HINTS  # Default to subtle

    def validate_memory_integration(
        self,
        pattern: MemoryIntegrationPattern,
        trust_level: TrustLevel,
        confidence: float,
        context: dict[str, Any],
    ) -> tuple[bool, list[str]]:
        """Validate if memory integration is appropriate."""
        validation_errors = []

        # Check trust level requirement
        trust_hierarchy = {
            TrustLevel.STRANGER: 0,
            TrustLevel.ACQUAINTANCE: 1,
            TrustLevel.FRIEND: 2,
            TrustLevel.CONFIDANT: 3,
        }

        if trust_hierarchy[trust_level] < trust_hierarchy[pattern.trust_requirement]:
            validation_errors.append(
                f"Trust level {trust_level.value} insufficient for pattern {pattern.pattern_name}"
            )

        # Check confidence appropriateness
        disclosure_threshold = self.get_disclosure_threshold_from_confidence(confidence)
        if disclosure_threshold != pattern.disclosure_level:
            validation_errors.append(
                f"Confidence {confidence} doesn't match pattern disclosure level"
            )

        # Apply flow validation rules
        for rule in pattern.flow_validation_rules:
            if rule == "ensure_high_trust" and trust_level != TrustLevel.CONFIDANT:
                validation_errors.append("High trust required but not met")
            elif rule == "avoid_specific_details" and confidence > 0.6:
                validation_errors.append(
                    "Pattern requires vague language but confidence is high"
                )
            elif rule == "maintain_consistency" and not context.get(
                "conversation_history"
            ):
                validation_errors.append(
                    "Consistency check requires conversation history"
                )

        return len(validation_errors) == 0, validation_errors

    def suggest_conversation_progression(
        self,
        current_stage: ConversationStage,
        relationship_score: float,
        conversation_count: int,
    ) -> ConversationStage | None:
        """Suggest next conversation stage based on progression."""
        template = self.get_template_for_stage(current_stage)
        if not template:
            return None

        # Check if ready for progression
        for next_stage, threshold in template.flow_transitions.items():
            if relationship_score >= threshold and conversation_count >= 3:
                return next_stage

        return current_stage  # Stay at current stage

    def get_transition_phrase(
        self, phrase_type: str, confidence: float, emotional_tone: str = "neutral"
    ) -> TransitionPhrase | None:
        """Get appropriate transition phrase."""
        if phrase_type not in self.transition_phrases:
            return None

        candidates = [
            phrase
            for phrase in self.transition_phrases[phrase_type]
            if phrase.matches_confidence(confidence)
            and (
                phrase.emotional_tone == emotional_tone
                or phrase.emotional_tone == "neutral"
            )
        ]

        return candidates[0] if candidates else None
