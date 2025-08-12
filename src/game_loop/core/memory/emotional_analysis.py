"""Emotional weighting analysis for conversation memories."""

import re
import time
from dataclasses import dataclass

from .config import MemoryAlgorithmConfig


@dataclass
class EmotionalAnalysisResult:
    """Result of emotional analysis for a conversation memory."""

    emotional_weight: float  # Overall emotional significance (0.0-1.0)
    sentiment_score: float  # Sentiment analysis result (-1.0 to 1.0)
    emotional_keywords: list[str]  # Detected emotional keywords
    relationship_impact: float  # Impact on relationship (0.0-1.0)
    emotional_intensity: float  # Intensity of emotional content (0.0-1.0)
    analysis_confidence: float  # Confidence in the analysis (0.0-1.0)


class EmotionalWeightingAnalyzer:
    """
    Analyzes emotional significance of conversation memories using multiple factors:
    - Sentiment analysis of text content
    - Emotional keyword detection
    - Relationship impact assessment
    - Context-based emotional indicators
    """

    # Emotional keyword categories with intensity weights
    EMOTIONAL_KEYWORDS = {
        # Positive emotions (high retention)
        "joy": {
            "keywords": {
                "happy",
                "joy",
                "excited",
                "delighted",
                "thrilled",
                "elated",
                "cheerful",
                "pleased",
                "glad",
                "wonderful",
                "amazing",
                "fantastic",
            },
            "intensity": 0.8,
        },
        "love": {
            "keywords": {
                "love",
                "adore",
                "cherish",
                "treasure",
                "devoted",
                "caring",
                "affection",
                "fondness",
                "dear",
                "precious",
                "beloved",
            },
            "intensity": 0.9,
        },
        "gratitude": {
            "keywords": {
                "thank",
                "grateful",
                "appreciate",
                "blessed",
                "thankful",
                "indebted",
                "obliged",
                "gracious",
            },
            "intensity": 0.7,
        },
        # Negative emotions (very high retention due to survival importance)
        "anger": {
            "keywords": {
                "angry",
                "furious",
                "rage",
                "mad",
                "irritated",
                "annoyed",
                "frustrated",
                "outraged",
                "livid",
                "irate",
                "hostile",
            },
            "intensity": 0.9,
        },
        "fear": {
            "keywords": {
                "afraid",
                "scared",
                "terrified",
                "frightened",
                "anxious",
                "worried",
                "nervous",
                "panic",
                "dread",
                "horror",
                "terror",
            },
            "intensity": 0.95,  # Fear has highest retention for survival
        },
        "sadness": {
            "keywords": {
                "sad",
                "depressed",
                "grief",
                "sorrow",
                "misery",
                "despair",
                "heartbroken",
                "devastated",
                "mourning",
                "melancholy",
            },
            "intensity": 0.85,
        },
        "betrayal": {
            "keywords": {
                "betrayed",
                "deceived",
                "lied",
                "cheated",
                "backstabbed",
                "disappointed",
                "let down",
                "trust",
                "broken promise",
            },
            "intensity": 0.9,
        },
        # Relationship emotions
        "trust": {
            "keywords": {
                "trust",
                "reliable",
                "dependable",
                "honest",
                "truthful",
                "loyal",
                "faithful",
                "integrity",
                "confide",
            },
            "intensity": 0.75,
        },
        "conflict": {
            "keywords": {
                "argue",
                "fight",
                "dispute",
                "conflict",
                "disagree",
                "quarrel",
                "clash",
                "confrontation",
            },
            "intensity": 0.8,
        },
    }

    # Emotional amplifiers - phrases that increase emotional intensity
    EMOTIONAL_AMPLIFIERS = {
        "very",
        "extremely",
        "incredibly",
        "absolutely",
        "completely",
        "totally",
        "utterly",
        "deeply",
        "intensely",
        "overwhelmingly",
        "profoundly",
    }

    # Relationship impact indicators
    RELATIONSHIP_INDICATORS = {
        "positive": {
            "keywords": {
                "helped",
                "supported",
                "saved",
                "protected",
                "assisted",
                "encouraged",
                "comforted",
                "befriended",
                "trusted",
            },
            "impact": 0.8,
        },
        "negative": {
            "keywords": {
                "hurt",
                "betrayed",
                "abandoned",
                "ignored",
                "rejected",
                "insulted",
                "threatened",
                "attacked",
                "deceived",
            },
            "impact": -0.8,
        },
        "significant": {
            "keywords": {
                "first time",
                "never forget",
                "always remember",
                "changed everything",
                "life changing",
                "turning point",
                "milestone",
                "special moment",
            },
            "impact": 0.9,
        },
    }

    def __init__(self, config: MemoryAlgorithmConfig):
        self.config = config
        self._analysis_cache: dict[str, EmotionalAnalysisResult] = {}
        self._performance_stats = {"analyses": 0, "cache_hits": 0, "avg_time_ms": 0.0}

    def analyze_emotional_weight(
        self,
        message_content: str,
        conversation_context: dict | None = None,
        participant_info: dict | None = None,
    ) -> EmotionalAnalysisResult:
        """
        Analyze emotional significance of conversation memory.

        Args:
            message_content: The conversation message text
            conversation_context: Context about the conversation
            participant_info: Information about conversation participants

        Returns:
            EmotionalAnalysisResult with weighted emotional significance
        """
        if not message_content or not message_content.strip():
            return EmotionalAnalysisResult(
                emotional_weight=0.0,
                sentiment_score=0.0,
                emotional_keywords=[],
                relationship_impact=0.0,
                emotional_intensity=0.0,
                analysis_confidence=0.0,
            )

        start_time = time.perf_counter()
        cache_key = self._create_cache_key(message_content, conversation_context)

        # Check cache first
        if cache_key in self._analysis_cache:
            self._performance_stats["cache_hits"] += 1
            return self._analysis_cache[cache_key]

        # Perform analysis
        result = self._perform_emotional_analysis(
            message_content, conversation_context, participant_info
        )

        # Cache result and update performance stats
        self._analysis_cache[cache_key] = result
        processing_time_ms = (time.perf_counter() - start_time) * 1000

        self._performance_stats["analyses"] += 1
        total_analyses = self._performance_stats["analyses"]
        self._performance_stats["avg_time_ms"] = (
            self._performance_stats["avg_time_ms"] * (total_analyses - 1)
            + processing_time_ms
        ) / total_analyses

        return result

    def _perform_emotional_analysis(
        self,
        message_content: str,
        conversation_context: dict | None,
        participant_info: dict | None,
    ) -> EmotionalAnalysisResult:
        """Perform the actual emotional analysis."""

        # Clean and normalize text
        cleaned_text = self._clean_text(message_content)

        # Analyze sentiment
        sentiment_score = self._analyze_sentiment(cleaned_text)

        # Detect emotional keywords
        emotional_keywords, keyword_intensity = self._detect_emotional_keywords(
            cleaned_text
        )

        # Analyze relationship impact
        relationship_impact = self._analyze_relationship_impact(
            cleaned_text, conversation_context, participant_info
        )

        # Calculate emotional intensity
        emotional_intensity = self._calculate_emotional_intensity(
            cleaned_text, emotional_keywords, keyword_intensity
        )

        # Calculate overall emotional weight
        emotional_weight = self._calculate_emotional_weight(
            sentiment_score, keyword_intensity, relationship_impact, emotional_intensity
        )

        # Determine analysis confidence
        analysis_confidence = self._calculate_analysis_confidence(
            len(cleaned_text), len(emotional_keywords), abs(sentiment_score)
        )

        return EmotionalAnalysisResult(
            emotional_weight=emotional_weight,
            sentiment_score=sentiment_score,
            emotional_keywords=emotional_keywords,
            relationship_impact=relationship_impact,
            emotional_intensity=emotional_intensity,
            analysis_confidence=analysis_confidence,
        )

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis."""
        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace and punctuation for analysis
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _analyze_sentiment(self, text: str) -> float:
        """
        Simple sentiment analysis using keyword matching.
        Returns sentiment score from -1.0 (negative) to 1.0 (positive).
        """
        words = set(text.split())
        positive_score = 0.0
        negative_score = 0.0

        # Count emotional keywords by category
        for category, data in self.EMOTIONAL_KEYWORDS.items():
            keyword_matches = words.intersection(data["keywords"])
            if keyword_matches:
                intensity = data["intensity"]
                if category in ["joy", "love", "gratitude", "trust"]:
                    positive_score += len(keyword_matches) * intensity
                else:  # negative emotions
                    negative_score += len(keyword_matches) * intensity

        # Apply emotional amplifiers
        amplifier_count = len(words.intersection(self.EMOTIONAL_AMPLIFIERS))
        amplifier_boost = min(amplifier_count * 0.2, 0.5)  # Max 0.5 boost

        positive_score *= 1.0 + amplifier_boost
        negative_score *= 1.0 + amplifier_boost

        # Calculate normalized sentiment
        if positive_score + negative_score == 0:
            return 0.0

        sentiment = (positive_score - negative_score) / (
            positive_score + negative_score
        )
        return max(-1.0, min(1.0, sentiment))

    def _detect_emotional_keywords(self, text: str) -> tuple[list[str], float]:
        """Detect emotional keywords and calculate intensity."""
        words = set(text.split())
        found_keywords = []
        total_intensity = 0.0

        for category, data in self.EMOTIONAL_KEYWORDS.items():
            matches = words.intersection(data["keywords"])
            found_keywords.extend(matches)
            if matches:
                total_intensity += len(matches) * data["intensity"]

        # Normalize intensity (0.0 to 1.0)
        normalized_intensity = min(
            total_intensity / 3.0, 1.0
        )  # Cap at reasonable level

        return list(found_keywords), normalized_intensity

    def _analyze_relationship_impact(
        self,
        text: str,
        conversation_context: dict | None,
        participant_info: dict | None,
    ) -> float:
        """Analyze impact on relationship between participants."""
        words = set(text.split())
        impact_score = 0.0

        # Check relationship indicators
        for category, data in self.RELATIONSHIP_INDICATORS.items():
            matches = words.intersection(data["keywords"])
            if matches:
                impact_score += len(matches) * data["impact"]

        # Context-based adjustments
        if conversation_context:
            # First conversation has higher relationship impact
            if conversation_context.get("is_first_meeting", False):
                impact_score *= 1.3

            # Conflict conversations have higher impact
            if conversation_context.get("conversation_type") == "conflict":
                impact_score *= 1.2

        # Normalize to 0.0-1.0 range (absolute value for magnitude)
        return min(abs(impact_score) / 2.0, 1.0)

    def _calculate_emotional_intensity(
        self, text: str, keywords: list[str], keyword_intensity: float
    ) -> float:
        """Calculate overall emotional intensity of the text."""

        # Base intensity from keywords
        intensity = keyword_intensity

        # Boost for multiple emotional indicators
        if len(keywords) > 2:
            intensity *= 1.2

        # Boost for emotional amplifiers
        words = set(text.split())
        amplifier_count = len(words.intersection(self.EMOTIONAL_AMPLIFIERS))
        if amplifier_count > 0:
            intensity *= 1.0 + amplifier_count * 0.15

        # Boost for exclamation marks in original text (indicates intensity)
        exclamation_count = text.count("!")
        if exclamation_count > 0:
            intensity *= 1.0 + exclamation_count * 0.1

        return min(intensity, 1.0)

    def _calculate_emotional_weight(
        self,
        sentiment_score: float,
        keyword_intensity: float,
        relationship_impact: float,
        emotional_intensity: float,
    ) -> float:
        """Calculate overall emotional weight using configured factors."""

        # Use absolute sentiment score (both positive and negative emotions are memorable)
        sentiment_magnitude = abs(sentiment_score)

        # Weighted combination of factors
        emotional_weight = (
            sentiment_magnitude * self.config.sentiment_weight
            + relationship_impact * self.config.relationship_weight
            + keyword_intensity * self.config.keyword_weight
        )

        # Boost by emotional intensity
        emotional_weight *= 1.0 + emotional_intensity * 0.5

        # Apply emotional keyword bonus
        if keyword_intensity > 0.3:  # Significant emotional content
            emotional_weight += self.config.emotional_keyword_bonus

        return max(0.0, min(1.0, emotional_weight))

    def _calculate_analysis_confidence(
        self, text_length: int, keyword_count: int, sentiment_magnitude: float
    ) -> float:
        """Calculate confidence in the emotional analysis."""

        # Base confidence from text length (more text = more reliable)
        length_confidence = min(text_length / 100.0, 1.0)  # Cap at 100 characters

        # Keyword confidence (more emotional keywords = higher confidence)
        keyword_confidence = min(keyword_count / 5.0, 1.0)  # Cap at 5 keywords

        # Sentiment confidence (stronger sentiment = higher confidence)
        sentiment_confidence = sentiment_magnitude

        # Combined confidence
        confidence = (
            length_confidence + keyword_confidence + sentiment_confidence
        ) / 3.0

        return max(0.1, min(1.0, confidence))  # Minimum 0.1 confidence

    def _create_cache_key(
        self, message_content: str, conversation_context: dict | None
    ) -> str:
        """Create cache key for analysis result."""
        context_key = ""
        if conversation_context:
            # Include relevant context in cache key
            context_parts = [
                str(conversation_context.get("is_first_meeting", False)),
                conversation_context.get("conversation_type", "normal"),
            ]
            context_key = "_".join(context_parts)

        # Use hash of content to keep key manageable
        content_hash = str(hash(message_content))
        return f"{content_hash}_{context_key}"

    def get_performance_stats(self) -> dict[str, float]:
        """Get analysis performance statistics."""
        total_requests = (
            self._performance_stats["analyses"] + self._performance_stats["cache_hits"]
        )
        cache_hit_rate = (
            (self._performance_stats["cache_hits"] / total_requests * 100)
            if total_requests > 0
            else 0.0
        )

        return {
            "total_analyses": self._performance_stats["analyses"],
            "cache_hits": self._performance_stats["cache_hits"],
            "cache_hit_rate_percent": round(cache_hit_rate, 1),
            "avg_processing_time_ms": round(self._performance_stats["avg_time_ms"], 2),
        }

    def clear_cache(self) -> None:
        """Clear the analysis cache."""
        self._analysis_cache.clear()
        self._performance_stats = {"analyses": 0, "cache_hits": 0, "avg_time_ms": 0.0}
