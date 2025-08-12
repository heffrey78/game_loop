"""Trauma-sensitive memory handling and protection mechanisms."""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from game_loop.core.conversation.conversation_models import (
    ConversationContext,
    ConversationExchange,
    NPCPersonality,
)
from game_loop.database.session_factory import DatabaseSessionFactory

from .config import MemoryAlgorithmConfig
from .constants import EmotionalThresholds, ProtectionMechanismConfig
from .emotional_context import (
    EmotionalMemoryType,
    EmotionalSignificance,
    MoodState,
    MemoryProtectionLevel,
)
from .emotional_preservation import EmotionalMemoryRecord, EmotionalRetrievalQuery
from .exceptions import (
    TraumaProtectionError,
    InvalidEmotionalDataError,
    SecurityError,
    handle_emotional_memory_error,
)
from .mood_memory_engine import MoodDependentMemoryEngine
from .validation import (
    validate_uuid,
    validate_probability,
    validate_positive_number,
    validate_string_content,
    default_validator,
)

logger = logging.getLogger(__name__)


class TraumaResponseType(Enum):
    """Types of trauma responses in memory access."""

    AVOIDANCE = "avoidance"  # Complete avoidance of trauma memories
    HYPERVIGILANCE = "hypervigilance"  # Heightened sensitivity to trauma cues
    INTRUSION = "intrusion"  # Unwanted trauma memory intrusions
    DISSOCIATION = "dissociation"  # Disconnection from trauma memories
    NUMBING = "numbing"  # Emotional numbing around trauma
    FLOODING = "flooding"  # Overwhelming trauma memory activation


class ProtectionMechanism(Enum):
    """Memory protection mechanisms."""

    ACCESS_CONTROL = "access_control"  # Control who can access memories
    GRADUAL_EXPOSURE = "gradual_exposure"  # Gradual therapeutic exposure
    SAFETY_CHECKS = "safety_checks"  # Safety checks before trauma access
    THERAPEUTIC_SUPPORT = "therapeutic"  # Provide therapeutic context
    EMERGENCY_CONTAINMENT = "emergency"  # Emergency containment protocols
    TRUST_GATING = "trust_gating"  # Trust-based access controls


class TherapeuticApproach(Enum):
    """Approaches for therapeutic trauma memory handling."""

    COGNITIVE_PROCESSING = "cognitive"  # Cognitive processing therapy approach
    EXPOSURE_THERAPY = "exposure"  # Gradual exposure therapy
    NARRATIVE_THERAPY = "narrative"  # Narrative reconstruction
    SOMATIC_APPROACH = "somatic"  # Body-based trauma therapy
    RESOURCE_BUILDING = "resource_building"  # Building internal resources first
    INTEGRATION_FOCUSED = "integration"  # Memory integration approach


@dataclass
class TraumaMemoryProfile:
    """Profile of trauma-related memory patterns for an NPC."""

    npc_id: str

    # Trauma characteristics
    trauma_memories: List[str] = field(default_factory=list)  # Traumatic memory IDs
    trauma_severity: float = 0.0  # 0.0-1.0 overall trauma severity
    trauma_recency: float = 0.0  # Time since most recent trauma (hours)

    # Response patterns
    primary_response_type: TraumaResponseType = TraumaResponseType.AVOIDANCE
    secondary_responses: List[TraumaResponseType] = field(default_factory=list)
    trigger_sensitivity: float = (
        EmotionalThresholds.MODERATE_SIGNIFICANCE
    )  # 0.0-1.0 sensitivity to triggers

    # Protection mechanisms active
    active_protections: Set[ProtectionMechanism] = field(default_factory=set)
    protection_effectiveness: Dict[ProtectionMechanism, float] = field(
        default_factory=dict
    )

    # Therapeutic context
    therapeutic_progress: float = 0.0  # 0.0-1.0 progress in processing
    preferred_approach: Optional[TherapeuticApproach] = None
    safe_memory_anchors: List[str] = field(
        default_factory=list
    )  # Safe memories for grounding

    # Thresholds and limits
    daily_trauma_exposure_limit: int = (
        EmotionalThresholds.MAX_DAILY_TRAUMA_EXPOSURE
    )  # Max trauma memories per day
    current_daily_exposure: int = 0
    last_exposure_reset: float = field(default_factory=time.time)

    # Recovery tracking
    recovery_indicators: Dict[str, float] = field(default_factory=dict)
    regression_warnings: List[str] = field(default_factory=list)

    # Emergency protocols
    emergency_contacts: List[str] = field(default_factory=list)  # Support person IDs
    crisis_prevention_active: bool = False
    last_crisis_event: Optional[float] = None


@dataclass
class TraumaAccessRequest:
    """Request to access trauma-related memories."""

    npc_id: str
    requesting_context: str  # Context of the request
    trauma_memory_ids: List[str]  # Specific trauma memories requested
    trust_level: float  # Current trust level
    therapeutic_intent: bool = False  # Whether access is for therapeutic purposes

    # Safety context
    support_available: bool = False  # Whether support is available
    safe_environment: bool = True  # Whether environment is safe
    time_availability: float = 1.0  # Available time for processing (hours)

    # Request metadata
    request_timestamp: float = field(default_factory=time.time)
    urgency_level: float = (
        EmotionalThresholds.MODERATE_SIGNIFICANCE
    )  # 0.0-1.0 urgency of access


@dataclass
class TraumaAccessDecision:
    """Decision about trauma memory access."""

    access_granted: bool
    granted_memories: List[str] = field(default_factory=list)
    denied_memories: List[str] = field(default_factory=list)

    # Access conditions
    required_preparations: List[str] = field(default_factory=list)
    recommended_support: List[str] = field(default_factory=list)
    time_limit: Optional[float] = None  # Max exposure time

    # Protection measures
    active_protections: List[ProtectionMechanism] = field(default_factory=list)
    safety_monitoring: bool = False
    emergency_protocols: bool = False

    # Rationale
    decision_rationale: str = ""
    risk_assessment: Dict[str, float] = field(default_factory=dict)
    therapeutic_value: float = 0.0

    # Follow-up requirements
    post_access_monitoring: bool = False
    integration_support: bool = False

    decision_timestamp: float = field(default_factory=time.time)


class TraumaProtectionEngine:
    """Engine for trauma-sensitive memory handling and protection."""

    def __init__(
        self,
        session_factory: DatabaseSessionFactory,
        config: MemoryAlgorithmConfig,
        mood_engine: MoodDependentMemoryEngine,
        enable_strict_protection: bool = True,
        therapeutic_mode: bool = False,
    ):
        self.session_factory = session_factory
        self.config = config
        self.mood_engine = mood_engine
        self.enable_strict_protection = enable_strict_protection
        self.therapeutic_mode = therapeutic_mode

        # Trauma profiles
        self._trauma_profiles: Dict[str, TraumaMemoryProfile] = {}

        # Access decision history
        self._access_decisions: Dict[str, List[TraumaAccessDecision]] = {}

        # Safety thresholds using constants
        self.safety_thresholds = {
            "min_trust_for_trauma": EmotionalThresholds.MIN_TRUST_FOR_TRAUMA,
            "max_daily_trauma_exposure": EmotionalThresholds.MAX_DAILY_TRAUMA_EXPOSURE,
            "min_time_between_exposures": EmotionalThresholds.MIN_TIME_BETWEEN_TRAUMA_HOURS,
            "crisis_sensitivity_threshold": EmotionalThresholds.CRISIS_SENSITIVITY_THRESHOLD,
            "emergency_response_threshold": EmotionalThresholds.EMERGENCY_RESPONSE_THRESHOLD,
        }

        # Caches
        self._risk_assessment_cache: Dict[str, Dict[str, float]] = {}
        self._therapeutic_value_cache: Dict[str, float] = {}

        # Performance tracking
        self._performance_stats = {
            "access_requests": 0,
            "access_granted": 0,
            "access_denied": 0,
            "safety_interventions": 0,
            "therapeutic_accesses": 0,
            "crisis_preventions": 0,
            "avg_decision_time_ms": 0.0,
        }

    async def evaluate_trauma_access_request(
        self,
        npc_id: uuid.UUID,
        access_request: TraumaAccessRequest,
        personality: NPCPersonality,
        current_mood: Optional[MoodState] = None,
    ) -> TraumaAccessDecision:
        """Evaluate whether to grant access to trauma memories."""
        try:
            # Validate inputs
            if not npc_id:
                raise InvalidEmotionalDataError("NPC ID is required")
            if not access_request:
                raise InvalidEmotionalDataError("Access request is required")
            if not personality:
                raise InvalidEmotionalDataError("NPC personality is required")

            # Validate NPC ID and access request fields
            npc_id_str = str(validate_uuid(npc_id, "npc_id"))
            validate_probability(access_request.trust_level, "trust_level")
            validate_probability(access_request.urgency_level, "urgency_level")
            validate_positive_number(
                access_request.time_availability, "time_availability", allow_zero=True
            )

            # Validate trauma memory IDs
            for i, memory_id in enumerate(access_request.trauma_memory_ids):
                validate_string_content(
                    memory_id, f"trauma_memory_ids[{i}]", min_length=1
                )

            # Security check - ensure strict protection is enabled for trauma access
            if not self.enable_strict_protection and access_request.trauma_memory_ids:
                raise SecurityError(
                    "Trauma memory access attempted with strict protection disabled",
                    security_violation="unprotected_trauma_access",
                    attempted_action="trauma_memory_access",
                )

        except Exception as e:
            raise handle_emotional_memory_error(
                e, "Failed to validate trauma access request"
            )

        start_time = time.perf_counter()

        try:
            # Get or create trauma profile
            trauma_profile = self._get_trauma_profile(npc_id_str)

            # Update daily exposure tracking
            self._update_daily_exposure_tracking(trauma_profile)

            # Perform comprehensive risk assessment
            risk_assessment = await self._assess_trauma_access_risk(
                trauma_profile, access_request, personality, current_mood
            )

            # Calculate therapeutic value
            therapeutic_value = await self._calculate_therapeutic_value(
                trauma_profile, access_request, personality
            )

            # Make access decision
            decision = await self._make_access_decision(
                trauma_profile,
                access_request,
                risk_assessment,
                therapeutic_value,
                personality,
            )

            # Apply protection mechanisms if access granted
            if decision.access_granted and decision.granted_memories:
                await self._apply_protection_mechanisms(
                    trauma_profile, decision, personality
                )

            # Record decision
            if npc_id_str not in self._access_decisions:
                self._access_decisions[npc_id_str] = []
            self._access_decisions[npc_id_str].append(decision)

            # Update trauma profile based on decision
            await self._update_trauma_profile_post_decision(
                trauma_profile, access_request, decision
            )

            # Update performance stats
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            self._performance_stats["access_requests"] += 1
            if decision.access_granted:
                self._performance_stats["access_granted"] += 1
            else:
                self._performance_stats["access_denied"] += 1

            if access_request.therapeutic_intent:
                self._performance_stats["therapeutic_accesses"] += 1

            self._update_avg_decision_time(processing_time_ms)

            logger.debug(
                f"Trauma access decision for NPC {npc_id}: {decision.access_granted}"
            )
            return decision

        except Exception as e:
            raise TraumaProtectionError(
                f"Failed to evaluate trauma access for NPC {npc_id}",
                npc_id=npc_id_str,
                safety_violation="evaluation_failure",
            )

    async def build_trauma_memory_profile(
        self,
        npc_id: uuid.UUID,
        memories: List[EmotionalMemoryRecord],
        personality: NPCPersonality,
    ) -> TraumaMemoryProfile:
        """Build comprehensive trauma memory profile for an NPC."""

        npc_id_str = str(npc_id)

        # Identify trauma memories
        trauma_memories = [
            m.exchange_id
            for m in memories
            if m.emotional_significance.emotional_type == EmotionalMemoryType.TRAUMATIC
        ]

        # Calculate trauma severity
        if trauma_memories:
            trauma_severities = [
                m.emotional_significance.overall_significance
                for m in memories
                if m.exchange_id in trauma_memories
            ]
            trauma_severity = max(trauma_severities)
        else:
            trauma_severity = 0.0

        # Calculate trauma recency
        if trauma_memories:
            most_recent_trauma = max(
                m.preserved_at for m in memories if m.exchange_id in trauma_memories
            )
            trauma_recency = (time.time() - most_recent_trauma) / 3600  # Hours
        else:
            trauma_recency = float("inf")

        # Determine primary response type based on personality
        primary_response = self._determine_primary_trauma_response(
            personality, trauma_severity, len(trauma_memories)
        )

        # Identify safe memory anchors
        safe_anchors = [
            m.exchange_id
            for m in memories
            if (
                m.emotional_significance.emotional_type
                in [
                    EmotionalMemoryType.CORE_ATTACHMENT,
                    EmotionalMemoryType.PEAK_POSITIVE,
                    EmotionalMemoryType.BREAKTHROUGH,
                ]
                and m.emotional_significance.overall_significance > 0.7
            )
        ]

        # Create profile
        profile = TraumaMemoryProfile(
            npc_id=npc_id_str,
            trauma_memories=trauma_memories,
            trauma_severity=trauma_severity,
            trauma_recency=trauma_recency,
            primary_response_type=primary_response,
            trigger_sensitivity=self._calculate_trigger_sensitivity(
                personality, trauma_severity
            ),
            safe_memory_anchors=safe_anchors,
            preferred_approach=self._determine_preferred_therapeutic_approach(
                personality
            ),
        )

        # Set up initial protection mechanisms
        profile.active_protections = self._initialize_protection_mechanisms(
            profile, personality
        )

        # Calculate protection effectiveness
        for protection in profile.active_protections:
            effectiveness = self._calculate_protection_effectiveness(
                protection, personality, trauma_severity
            )
            profile.protection_effectiveness[protection] = effectiveness

        # Store profile
        self._trauma_profiles[npc_id_str] = profile

        logger.debug(
            f"Built trauma profile for NPC {npc_id}: {len(trauma_memories)} trauma memories"
        )
        return profile

    async def provide_therapeutic_memory_access(
        self,
        npc_id: uuid.UUID,
        therapeutic_goals: List[str],
        session_duration: float,
        support_level: float,
        personality: NPCPersonality,
    ) -> Dict[str, Any]:
        """Provide therapeutic access to trauma memories with appropriate support."""

        npc_id_str = str(npc_id)
        trauma_profile = self._get_trauma_profile(npc_id_str)

        if not trauma_profile.trauma_memories:
            return {
                "status": "no_trauma_memories",
                "message": "No trauma memories identified for therapeutic work",
                "recommendations": [
                    "Focus on building emotional resources and resilience"
                ],
            }

        # Create therapeutic access request
        therapeutic_request = TraumaAccessRequest(
            npc_id=npc_id_str,
            requesting_context="therapeutic_session",
            trauma_memory_ids=trauma_profile.trauma_memories[
                :2
            ],  # Start with limited exposure
            trust_level=0.9,  # High trust for therapeutic context
            therapeutic_intent=True,
            support_available=True,
            safe_environment=True,
            time_availability=session_duration,
        )

        # Get access decision
        decision = await self.evaluate_trauma_access_request(
            npc_id, therapeutic_request, personality
        )

        if not decision.access_granted:
            return {
                "status": "access_denied",
                "rationale": decision.decision_rationale,
                "recommendations": decision.recommended_support,
            }

        # Provide therapeutic framework
        therapeutic_framework = await self._create_therapeutic_framework(
            trauma_profile, decision, therapeutic_goals, personality
        )

        # Monitor therapeutic process
        monitoring_protocol = self._create_monitoring_protocol(trauma_profile, decision)

        self._performance_stats["therapeutic_accesses"] += 1

        return {
            "status": "access_granted",
            "accessible_memories": decision.granted_memories,
            "therapeutic_framework": therapeutic_framework,
            "monitoring_protocol": monitoring_protocol,
            "safety_measures": decision.active_protections,
            "time_limit": decision.time_limit,
            "integration_support": decision.integration_support,
        }

    async def detect_trauma_triggers(
        self,
        npc_id: uuid.UUID,
        conversation_content: str,
        conversation_context: ConversationContext,
        personality: NPCPersonality,
    ) -> Dict[str, Any]:
        """Detect potential trauma triggers in conversation content."""

        npc_id_str = str(npc_id)
        trauma_profile = self._get_trauma_profile(npc_id_str)

        if not trauma_profile.trauma_memories:
            return {"triggers_detected": False}

        # Analyze content for trigger keywords and themes
        triggers_detected = await self._analyze_trigger_content(
            conversation_content, trauma_profile, personality
        )

        if not triggers_detected["triggers_found"]:
            return {"triggers_detected": False}

        # Assess trigger severity and response
        trigger_response = await self._assess_trigger_response(
            triggers_detected, trauma_profile, personality
        )

        # Recommend protective responses
        protective_responses = self._recommend_protective_responses(
            trigger_response, trauma_profile
        )

        return {
            "triggers_detected": True,
            "trigger_details": triggers_detected,
            "response_assessment": trigger_response,
            "protective_responses": protective_responses,
            "safety_monitoring_required": trigger_response["severity"] > 0.7,
            "crisis_risk": trigger_response["severity"] > 0.9,
        }

    async def monitor_trauma_recovery_progress(
        self,
        npc_id: uuid.UUID,
        assessment_period_days: int = 7,
    ) -> Dict[str, Any]:
        """Monitor and assess trauma recovery progress."""

        npc_id_str = str(npc_id)
        trauma_profile = self._get_trauma_profile(npc_id_str)

        if not trauma_profile.trauma_memories:
            return {"status": "no_trauma_to_monitor"}

        # Assess recovery indicators
        recovery_metrics = await self._assess_recovery_metrics(
            trauma_profile, assessment_period_days
        )

        # Check for regression warning signs
        regression_assessment = self._assess_regression_risk(
            trauma_profile, recovery_metrics
        )

        # Update therapeutic progress
        previous_progress = trauma_profile.therapeutic_progress
        trauma_profile.therapeutic_progress = recovery_metrics["overall_progress"]

        progress_change = trauma_profile.therapeutic_progress - previous_progress

        # Generate recommendations
        recommendations = self._generate_recovery_recommendations(
            trauma_profile, recovery_metrics, regression_assessment
        )

        return {
            "recovery_metrics": recovery_metrics,
            "progress_change": progress_change,
            "regression_risk": regression_assessment,
            "recommendations": recommendations,
            "next_assessment_date": time.time() + (assessment_period_days * 86400),
        }

    def _get_trauma_profile(self, npc_id: str) -> TraumaMemoryProfile:
        """Get or create trauma profile for an NPC."""
        if npc_id not in self._trauma_profiles:
            self._trauma_profiles[npc_id] = TraumaMemoryProfile(npc_id=npc_id)
        return self._trauma_profiles[npc_id]

    def _update_daily_exposure_tracking(self, profile: TraumaMemoryProfile) -> None:
        """Update daily trauma exposure tracking."""
        current_time = time.time()

        # Reset daily counter if it's a new day
        if current_time - profile.last_exposure_reset > 86400:  # 24 hours
            profile.current_daily_exposure = 0
            profile.last_exposure_reset = current_time

    async def _assess_trauma_access_risk(
        self,
        profile: TraumaMemoryProfile,
        request: TraumaAccessRequest,
        personality: NPCPersonality,
        current_mood: Optional[MoodState],
    ) -> Dict[str, float]:
        """Assess risk of granting trauma memory access."""

        risk_factors = {}

        # Trust level risk
        if request.trust_level < self.safety_thresholds["min_trust_for_trauma"]:
            risk_factors["insufficient_trust"] = 1.0 - request.trust_level
        else:
            risk_factors["insufficient_trust"] = 0.0

        # Daily exposure risk
        if profile.current_daily_exposure >= profile.daily_trauma_exposure_limit:
            risk_factors["excessive_daily_exposure"] = 0.8
        else:
            risk_factors["excessive_daily_exposure"] = (
                profile.current_daily_exposure
                / profile.daily_trauma_exposure_limit
                * 0.3
            )

        # Mood state risk
        if current_mood:
            mood_risks = {
                MoodState.FEARFUL: 0.9,
                MoodState.ANXIOUS: 0.7,
                MoodState.MELANCHOLY: 0.6,
                MoodState.ANGRY: 0.5,
                MoodState.NEUTRAL: 0.2,
                MoodState.CONTENT: 0.1,
                MoodState.JOYFUL: 0.05,
            }
            risk_factors["mood_vulnerability"] = mood_risks.get(current_mood, 0.3)
        else:
            risk_factors["mood_vulnerability"] = 0.3

        # Trauma severity risk
        risk_factors["trauma_severity"] = profile.trauma_severity * 0.6

        # Trigger sensitivity risk
        risk_factors["trigger_sensitivity"] = profile.trigger_sensitivity * 0.4

        # Recent trauma risk
        if profile.trauma_recency < 48:  # Recent trauma (less than 48 hours)
            risk_factors["recent_trauma"] = max(0.6, 1.0 - profile.trauma_recency / 48)
        else:
            risk_factors["recent_trauma"] = 0.0

        # Environment and support risk
        if not request.support_available:
            risk_factors["no_support"] = 0.5
        else:
            risk_factors["no_support"] = 0.0

        if not request.safe_environment:
            risk_factors["unsafe_environment"] = 0.7
        else:
            risk_factors["unsafe_environment"] = 0.0

        # Time availability risk
        if request.time_availability < 0.5:  # Less than 30 minutes
            risk_factors["insufficient_time"] = 0.4
        else:
            risk_factors["insufficient_time"] = 0.0

        # Personality vulnerability factors
        vulnerability_traits = ["trauma_sensitive", "anxious", "emotional_sensitivity"]
        personality_vulnerability = 0.0
        for trait in vulnerability_traits:
            personality_vulnerability += personality.get_trait_strength(trait)

        risk_factors["personality_vulnerability"] = min(
            1.0, personality_vulnerability / len(vulnerability_traits)
        )

        return risk_factors

    async def _calculate_therapeutic_value(
        self,
        profile: TraumaMemoryProfile,
        request: TraumaAccessRequest,
        personality: NPCPersonality,
    ) -> float:
        """Calculate therapeutic value of trauma memory access."""

        if not request.therapeutic_intent:
            return 0.0

        value_factors = []

        # Progress readiness
        if profile.therapeutic_progress > 0.5:
            value_factors.append(0.8)  # Good progress indicates readiness
        else:
            value_factors.append(0.3)  # Early stages, lower value

        # Safe anchors availability
        anchor_strength = min(1.0, len(profile.safe_memory_anchors) / 3)
        value_factors.append(anchor_strength * 0.6)

        # Support context
        if request.support_available and request.safe_environment:
            value_factors.append(0.7)
        else:
            value_factors.append(0.2)

        # Time availability
        if request.time_availability >= 1.0:  # At least 1 hour
            value_factors.append(0.6)
        else:
            value_factors.append(0.3)

        # Personality factors
        resilience_traits = ["resilient", "self_aware", "growth_oriented"]
        resilience_score = 0.0
        for trait in resilience_traits:
            resilience_score += personality.get_trait_strength(trait)

        resilience_factor = min(1.0, resilience_score / len(resilience_traits))
        value_factors.append(resilience_factor * 0.5)

        # Trauma recency (older trauma may be more processable)
        if profile.trauma_recency > 168:  # More than a week old
            value_factors.append(0.6)
        elif profile.trauma_recency > 48:  # More than 2 days
            value_factors.append(0.4)
        else:
            value_factors.append(0.1)  # Very recent trauma

        return sum(value_factors) / len(value_factors) if value_factors else 0.0

    async def _make_access_decision(
        self,
        profile: TraumaMemoryProfile,
        request: TraumaAccessRequest,
        risk_assessment: Dict[str, float],
        therapeutic_value: float,
        personality: NPCPersonality,
    ) -> TraumaAccessDecision:
        """Make final decision about trauma memory access."""

        # Calculate overall risk score
        overall_risk = sum(risk_assessment.values()) / len(risk_assessment)

        # Decision logic
        access_granted = False
        granted_memories = []
        denied_memories = list(request.trauma_memory_ids)
        rationale = ""

        if self.enable_strict_protection:
            # Strict protection mode - very conservative
            if overall_risk < 0.3 and therapeutic_value > 0.7:
                access_granted = True
                granted_memories = request.trauma_memory_ids[:1]  # Only one memory
                denied_memories = request.trauma_memory_ids[1:]
                rationale = "Limited access granted under strict protection with high therapeutic value"
            elif overall_risk < 0.2:
                access_granted = True
                granted_memories = request.trauma_memory_ids[:1]
                denied_memories = request.trauma_memory_ids[1:]
                rationale = "Limited access granted due to very low risk"
            else:
                rationale = f"Access denied: risk too high ({overall_risk:.2f}) for strict protection mode"

        else:
            # Standard protection mode
            if therapeutic_value > 0.6 and overall_risk < 0.5:
                access_granted = True
                granted_memories = request.trauma_memory_ids
                denied_memories = []
                rationale = (
                    "Access granted: high therapeutic value with acceptable risk"
                )
            elif overall_risk < 0.3:
                access_granted = True
                granted_memories = request.trauma_memory_ids[:2]  # Limited access
                denied_memories = request.trauma_memory_ids[2:]
                rationale = "Limited access granted due to low risk"
            else:
                rationale = f"Access denied: risk level too high ({overall_risk:.2f})"

        # Create decision
        decision = TraumaAccessDecision(
            access_granted=access_granted,
            granted_memories=granted_memories,
            denied_memories=denied_memories,
            decision_rationale=rationale,
            risk_assessment=risk_assessment,
            therapeutic_value=therapeutic_value,
        )

        # Set access conditions if granted
        if access_granted:
            decision.required_preparations = self._determine_required_preparations(
                profile, request, risk_assessment
            )
            decision.recommended_support = self._determine_recommended_support(
                profile, risk_assessment, personality
            )

            # Set time limits for high-risk access
            if overall_risk > 0.4:
                decision.time_limit = min(
                    0.5, request.time_availability
                )  # Max 30 minutes

            # Enable safety monitoring for moderate to high risk
            decision.safety_monitoring = overall_risk > 0.3
            decision.emergency_protocols = overall_risk > 0.6

            # Post-access support for therapeutic access
            if request.therapeutic_intent:
                decision.post_access_monitoring = True
                decision.integration_support = True

        return decision

    async def _apply_protection_mechanisms(
        self,
        profile: TraumaMemoryProfile,
        decision: TraumaAccessDecision,
        personality: NPCPersonality,
    ) -> None:
        """Apply active protection mechanisms for trauma memory access."""

        # Determine which protections to activate
        protections_to_activate = set()

        # Always use access control
        protections_to_activate.add(ProtectionMechanism.ACCESS_CONTROL)

        # Safety checks for any trauma access
        protections_to_activate.add(ProtectionMechanism.SAFETY_CHECKS)

        # Trust gating for high-risk access
        if decision.risk_assessment.get("insufficient_trust", 0) > 0.2:
            protections_to_activate.add(ProtectionMechanism.TRUST_GATING)

        # Therapeutic support for therapeutic access
        if decision.therapeutic_value > 0.5:
            protections_to_activate.add(ProtectionMechanism.THERAPEUTIC_SUPPORT)

        # Gradual exposure for high-severity trauma
        if profile.trauma_severity > 0.7:
            protections_to_activate.add(ProtectionMechanism.GRADUAL_EXPOSURE)

        # Emergency containment for high-risk situations
        overall_risk = sum(decision.risk_assessment.values()) / len(
            decision.risk_assessment
        )
        if overall_risk > 0.7:
            protections_to_activate.add(ProtectionMechanism.EMERGENCY_CONTAINMENT)

        decision.active_protections = list(protections_to_activate)
        profile.active_protections.update(protections_to_activate)

    async def _update_trauma_profile_post_decision(
        self,
        profile: TraumaMemoryProfile,
        request: TraumaAccessRequest,
        decision: TraumaAccessDecision,
    ) -> None:
        """Update trauma profile based on access decision."""

        if decision.access_granted:
            # Increment daily exposure
            profile.current_daily_exposure += len(decision.granted_memories)

            # Track therapeutic progress if applicable
            if request.therapeutic_intent:
                # Small progress for each therapeutic access
                progress_increment = 0.05 * len(decision.granted_memories)
                profile.therapeutic_progress = min(
                    1.0, profile.therapeutic_progress + progress_increment
                )

                # Update recovery indicators
                profile.recovery_indicators["therapeutic_sessions"] = (
                    profile.recovery_indicators.get("therapeutic_sessions", 0) + 1
                )

        # Check for crisis indicators
        overall_risk = sum(decision.risk_assessment.values()) / len(
            decision.risk_assessment
        )
        if overall_risk > self.safety_thresholds["crisis_sensitivity_threshold"]:
            self._performance_stats["crisis_preventions"] += 1
            profile.crisis_prevention_active = True
            profile.last_crisis_event = time.time()

    def _determine_primary_trauma_response(
        self,
        personality: NPCPersonality,
        trauma_severity: float,
        trauma_count: int,
    ) -> TraumaResponseType:
        """Determine primary trauma response type based on personality and trauma characteristics."""

        # Personality factors influence response type
        avoidant_traits = ["conflict_averse", "anxious", "withdrawal_tendency"]
        hypervigilant_traits = ["paranoid", "suspicious", "hyperalert"]
        dissociative_traits = ["detached", "unemotional", "withdrawn"]

        avoidance_score = sum(
            personality.get_trait_strength(trait) for trait in avoidant_traits
        )
        hypervigilance_score = sum(
            personality.get_trait_strength(trait) for trait in hypervigilant_traits
        )
        dissociation_score = sum(
            personality.get_trait_strength(trait) for trait in dissociative_traits
        )

        # Trauma severity and count influence response
        if trauma_severity > 0.8:
            if dissociation_score > 1.0:
                return TraumaResponseType.DISSOCIATION
            elif hypervigilance_score > 1.0:
                return TraumaResponseType.HYPERVIGILANCE
            else:
                return TraumaResponseType.AVOIDANCE

        elif trauma_count > 3:  # Multiple traumas
            if hypervigilance_score > avoidance_score:
                return TraumaResponseType.HYPERVIGILANCE
            else:
                return TraumaResponseType.AVOIDANCE

        else:
            # Mild to moderate single trauma
            if avoidance_score > 1.0:
                return TraumaResponseType.AVOIDANCE
            else:
                return TraumaResponseType.NUMBING

    def _calculate_trigger_sensitivity(
        self, personality: NPCPersonality, trauma_severity: float
    ) -> float:
        """Calculate trauma trigger sensitivity."""

        sensitivity_traits = ["emotional_sensitivity", "trauma_sensitive", "anxious"]
        sensitivity_score = sum(
            personality.get_trait_strength(trait) for trait in sensitivity_traits
        )

        # Base sensitivity from personality
        base_sensitivity = min(1.0, sensitivity_score / len(sensitivity_traits))

        # Trauma severity amplifies sensitivity
        trauma_amplifier = 1.0 + (trauma_severity * 0.5)

        final_sensitivity = min(1.0, base_sensitivity * trauma_amplifier)
        return final_sensitivity

    def _determine_preferred_therapeutic_approach(
        self, personality: NPCPersonality
    ) -> Optional[TherapeuticApproach]:
        """Determine preferred therapeutic approach based on personality."""

        analytical_traits = ["analytical", "rational", "logical"]
        experiential_traits = ["experiential", "somatic", "body_aware"]
        narrative_traits = ["storytelling", "creative", "expressive"]

        analytical_score = sum(
            personality.get_trait_strength(trait) for trait in analytical_traits
        )
        experiential_score = sum(
            personality.get_trait_strength(trait) for trait in experiential_traits
        )
        narrative_score = sum(
            personality.get_trait_strength(trait) for trait in narrative_traits
        )

        scores = {
            TherapeuticApproach.COGNITIVE_PROCESSING: analytical_score,
            TherapeuticApproach.SOMATIC_APPROACH: experiential_score,
            TherapeuticApproach.NARRATIVE_THERAPY: narrative_score,
        }

        if max(scores.values()) > 0.5:
            return max(scores, key=scores.get)
        else:
            return TherapeuticApproach.RESOURCE_BUILDING  # Default safe approach

    def _initialize_protection_mechanisms(
        self, profile: TraumaMemoryProfile, personality: NPCPersonality
    ) -> Set[ProtectionMechanism]:
        """Initialize protection mechanisms based on trauma profile."""

        protections = set()

        # Always enable basic protections
        protections.add(ProtectionMechanism.ACCESS_CONTROL)
        protections.add(ProtectionMechanism.SAFETY_CHECKS)

        # Trust gating for high sensitivity
        if profile.trigger_sensitivity > 0.7:
            protections.add(ProtectionMechanism.TRUST_GATING)

        # Gradual exposure for severe trauma
        if profile.trauma_severity > 0.6:
            protections.add(ProtectionMechanism.GRADUAL_EXPOSURE)

        # Therapeutic support if therapeutic approach identified
        if profile.preferred_approach:
            protections.add(ProtectionMechanism.THERAPEUTIC_SUPPORT)

        # Emergency containment for high-risk profiles
        if profile.trauma_severity > 0.8 and profile.trigger_sensitivity > 0.8:
            protections.add(ProtectionMechanism.EMERGENCY_CONTAINMENT)

        return protections

    def _calculate_protection_effectiveness(
        self,
        protection: ProtectionMechanism,
        personality: NPCPersonality,
        trauma_severity: float,
    ) -> float:
        """Calculate effectiveness of a protection mechanism for this NPC."""

        # Base effectiveness by protection type
        base_effectiveness = {
            ProtectionMechanism.ACCESS_CONTROL: 0.8,
            ProtectionMechanism.GRADUAL_EXPOSURE: 0.7,
            ProtectionMechanism.SAFETY_CHECKS: 0.9,
            ProtectionMechanism.THERAPEUTIC_SUPPORT: 0.6,
            ProtectionMechanism.EMERGENCY_CONTAINMENT: 0.9,
            ProtectionMechanism.TRUST_GATING: 0.7,
        }

        effectiveness = base_effectiveness.get(protection, 0.5)

        # Personality factors affect effectiveness
        if protection == ProtectionMechanism.THERAPEUTIC_SUPPORT:
            therapy_receptiveness = personality.get_trait_strength("therapy_receptive")
            effectiveness *= 0.5 + therapy_receptiveness * 0.5

        elif protection == ProtectionMechanism.TRUST_GATING:
            trust_issues = personality.get_trait_strength("trust_issues")
            effectiveness *= 1.0 + trust_issues * 0.3  # More effective for trust issues

        # Trauma severity affects some protections
        if protection in [
            ProtectionMechanism.EMERGENCY_CONTAINMENT,
            ProtectionMechanism.SAFETY_CHECKS,
        ]:
            effectiveness *= (
                0.7 + trauma_severity * 0.3
            )  # More effective for severe trauma

        return min(1.0, effectiveness)

    def _determine_required_preparations(
        self,
        profile: TraumaMemoryProfile,
        request: TraumaAccessRequest,
        risk_assessment: Dict[str, float],
    ) -> List[str]:
        """Determine required preparations before trauma access."""

        preparations = []

        # Always require grounding techniques
        preparations.append("establish_grounding_techniques")

        # Safe memory anchors for high-risk access
        overall_risk = sum(risk_assessment.values()) / len(risk_assessment)
        if overall_risk > 0.4 or not profile.safe_memory_anchors:
            preparations.append("activate_safe_memory_anchors")

        # Support person contact for high trauma severity
        if profile.trauma_severity > 0.7:
            preparations.append("ensure_support_person_availability")

        # Environment preparation for sensitive NPCs
        if profile.trigger_sensitivity > 0.6:
            preparations.append("create_safe_physical_environment")

        # Time boundaries for therapeutic access
        if request.therapeutic_intent:
            preparations.append("establish_clear_time_boundaries")

        # Emergency protocols for high-risk
        if overall_risk > 0.7:
            preparations.append("activate_emergency_protocols")

        return preparations

    def _determine_recommended_support(
        self,
        profile: TraumaMemoryProfile,
        risk_assessment: Dict[str, float],
        personality: NPCPersonality,
    ) -> List[str]:
        """Determine recommended support for trauma memory access."""

        support = []

        # Basic support for all trauma access
        support.append("emotional_regulation_techniques")

        # Therapeutic support for high-value therapeutic access
        if profile.preferred_approach:
            approach_name = profile.preferred_approach.value
            support.append(f"therapeutic_support_{approach_name}")

        # Grounding techniques for high sensitivity
        if profile.trigger_sensitivity > 0.6:
            support.append("advanced_grounding_techniques")

        # Social support for attachment-oriented personalities
        attachment_score = personality.get_trait_strength("attachment_oriented")
        if attachment_score > 0.5:
            support.append("social_support_activation")

        # Crisis prevention for high-risk profiles
        overall_risk = sum(risk_assessment.values()) / len(risk_assessment)
        if overall_risk > 0.6:
            support.append("crisis_prevention_protocols")

        # Integration support for therapeutic access
        if profile.therapeutic_progress > 0.3:
            support.append("memory_integration_techniques")

        return support

    async def _create_therapeutic_framework(
        self,
        profile: TraumaMemoryProfile,
        decision: TraumaAccessDecision,
        therapeutic_goals: List[str],
        personality: NPCPersonality,
    ) -> Dict[str, Any]:
        """Create therapeutic framework for trauma memory work."""

        framework = {
            "approach": (
                profile.preferred_approach.value
                if profile.preferred_approach
                else "resource_building"
            ),
            "session_goals": therapeutic_goals,
            "memory_targets": decision.granted_memories,
            "progress_markers": [],
            "safety_protocols": [],
            "integration_methods": [],
        }

        # Progress markers based on approach
        if profile.preferred_approach == TherapeuticApproach.COGNITIVE_PROCESSING:
            framework["progress_markers"] = [
                "identify_trauma_cognitions",
                "challenge_trauma_beliefs",
                "develop_balanced_perspectives",
                "integrate_new_understanding",
            ]
        elif profile.preferred_approach == TherapeuticApproach.EXPOSURE_THERAPY:
            framework["progress_markers"] = [
                "establish_safety_and_stabilization",
                "gradual_memory_exposure",
                "process_emotional_responses",
                "consolidate_therapeutic_gains",
            ]
        elif profile.preferred_approach == TherapeuticApproach.NARRATIVE_THERAPY:
            framework["progress_markers"] = [
                "externalize_trauma_narrative",
                "identify_alternative_stories",
                "strengthen_preferred_narrative",
                "integrate_coherent_life_story",
            ]

        # Safety protocols
        framework["safety_protocols"] = [
            "continuous_safety_monitoring",
            "grounding_technique_availability",
            "session_pacing_control",
            "emergency_containment_ready",
        ]

        if decision.emergency_protocols:
            framework["safety_protocols"].append("emergency_response_activated")

        # Integration methods
        framework["integration_methods"] = [
            "post_session_processing",
            "between_session_homework",
            "progress_tracking",
            "relapse_prevention_planning",
        ]

        return framework

    def _create_monitoring_protocol(
        self, profile: TraumaMemoryProfile, decision: TraumaAccessDecision
    ) -> Dict[str, Any]:
        """Create monitoring protocol for trauma memory access."""

        protocol = {
            "monitoring_level": "standard",
            "check_in_frequency": "every_15_minutes",
            "warning_signs": [],
            "intervention_thresholds": {},
            "emergency_procedures": [],
        }

        # Adjust monitoring based on risk
        overall_risk = sum(decision.risk_assessment.values()) / len(
            decision.risk_assessment
        )

        if overall_risk > 0.7:
            protocol["monitoring_level"] = "intensive"
            protocol["check_in_frequency"] = "every_5_minutes"
        elif overall_risk > 0.4:
            protocol["monitoring_level"] = "enhanced"
            protocol["check_in_frequency"] = "every_10_minutes"

        # Warning signs to monitor
        protocol["warning_signs"] = [
            "dissociation_indicators",
            "panic_response_signs",
            "emotional_flooding",
            "shutdown_responses",
            "hyperarousal_symptoms",
        ]

        # Intervention thresholds
        protocol["intervention_thresholds"] = {
            "mild_distress": 0.4,
            "moderate_distress": 0.6,
            "severe_distress": 0.8,
            "crisis_level": 0.9,
        }

        # Emergency procedures
        if decision.emergency_protocols:
            protocol["emergency_procedures"] = [
                "immediate_session_termination",
                "crisis_stabilization_techniques",
                "emergency_contact_activation",
                "professional_support_referral",
            ]

        return protocol

    async def _analyze_trigger_content(
        self,
        content: str,
        profile: TraumaMemoryProfile,
        personality: NPCPersonality,
    ) -> Dict[str, Any]:
        """Analyze content for potential trauma triggers."""

        # Common trauma trigger patterns
        trigger_keywords = {
            "violence": ["attack", "hit", "hurt", "pain", "violence", "aggressive"],
            "abandonment": ["left", "alone", "abandoned", "rejected", "isolated"],
            "betrayal": ["betrayed", "lied", "deceived", "cheated", "unfaithful"],
            "loss": ["died", "death", "lost", "gone", "never", "goodbye"],
            "powerlessness": ["helpless", "trapped", "couldn't", "forced", "powerless"],
            "shame": ["ashamed", "guilty", "fault", "blame", "worthless", "stupid"],
        }

        triggers_found = []
        content_lower = content.lower()

        for trigger_category, keywords in trigger_keywords.items():
            category_triggers = []
            for keyword in keywords:
                if keyword in content_lower:
                    category_triggers.append(keyword)

            if category_triggers:
                triggers_found.append(
                    {
                        "category": trigger_category,
                        "keywords": category_triggers,
                        "severity": len(category_triggers) / len(keywords),
                    }
                )

        return {
            "triggers_found": len(triggers_found) > 0,
            "trigger_details": triggers_found,
            "overall_trigger_load": sum(t["severity"] for t in triggers_found),
        }

    async def _assess_trigger_response(
        self,
        triggers_detected: Dict[str, Any],
        profile: TraumaMemoryProfile,
        personality: NPCPersonality,
    ) -> Dict[str, Any]:
        """Assess likely trauma response to detected triggers."""

        trigger_load = triggers_detected["overall_trigger_load"]

        # Base response severity
        base_severity = min(1.0, trigger_load * profile.trigger_sensitivity)

        # Personality factors modify response
        resilience_score = personality.get_trait_strength("resilient")
        emotional_regulation = personality.get_trait_strength("emotional_regulation")

        personality_modifier = 1.0 - (
            (resilience_score + emotional_regulation) / 2 * 0.3
        )

        final_severity = min(1.0, base_severity * personality_modifier)

        # Determine response type
        response_type = profile.primary_response_type

        # Predict specific responses
        responses = {
            TraumaResponseType.AVOIDANCE: [
                "topic_avoidance",
                "conversation_withdrawal",
                "emotional_numbing",
            ],
            TraumaResponseType.HYPERVIGILANCE: [
                "increased_alertness",
                "threat_scanning",
                "jumpiness",
            ],
            TraumaResponseType.INTRUSION: [
                "unwanted_memories",
                "flashback_risk",
                "thought_intrusions",
            ],
            TraumaResponseType.DISSOCIATION: [
                "emotional_detachment",
                "unreality_feelings",
                "memory_gaps",
            ],
            TraumaResponseType.NUMBING: [
                "emotional_flatness",
                "reduced_engagement",
                "apathy",
            ],
            TraumaResponseType.FLOODING: [
                "emotional_overwhelm",
                "panic_response",
                "crisis_risk",
            ],
        }

        predicted_responses = responses.get(response_type, ["general_distress"])

        return {
            "severity": final_severity,
            "response_type": response_type.value,
            "predicted_responses": predicted_responses,
            "intervention_needed": final_severity > 0.6,
            "crisis_risk": final_severity > 0.8,
        }

    def _recommend_protective_responses(
        self,
        trigger_response: Dict[str, Any],
        profile: TraumaMemoryProfile,
    ) -> List[str]:
        """Recommend protective responses to trauma triggers."""

        responses = []
        severity = trigger_response["severity"]
        response_type = TraumaResponseType(trigger_response["response_type"])

        # Basic protective responses
        responses.append("acknowledge_trigger_detected")

        if severity < 0.3:
            responses.extend(["gentle_topic_redirect", "increase_safety_cues"])
        elif severity < 0.6:
            responses.extend(
                [
                    "active_grounding_techniques",
                    "safe_memory_activation",
                    "reduce_trigger_exposure",
                ]
            )
        elif severity < 0.8:
            responses.extend(
                [
                    "immediate_safety_measures",
                    "crisis_prevention_protocols",
                    "support_person_contact",
                ]
            )
        else:  # High severity
            responses.extend(
                [
                    "emergency_stabilization",
                    "professional_intervention",
                    "crisis_management_activation",
                ]
            )

        # Response-type specific recommendations
        if response_type == TraumaResponseType.AVOIDANCE:
            responses.append("respect_avoidance_need")
        elif response_type == TraumaResponseType.HYPERVIGILANCE:
            responses.append("increase_predictability")
        elif response_type == TraumaResponseType.DISSOCIATION:
            responses.append("grounding_and_orientation")
        elif response_type == TraumaResponseType.FLOODING:
            responses.append("containment_techniques")

        return responses

    async def _assess_recovery_metrics(
        self, profile: TraumaMemoryProfile, assessment_period_days: int
    ) -> Dict[str, Any]:
        """Assess trauma recovery metrics over assessment period."""

        metrics = {
            "overall_progress": profile.therapeutic_progress,
            "stability_improvement": 0.0,
            "trigger_sensitivity_change": 0.0,
            "protection_effectiveness": 0.0,
            "functional_improvement": 0.0,
            "setback_frequency": 0.0,
        }

        # Calculate protection effectiveness
        if profile.protection_effectiveness:
            avg_effectiveness = sum(profile.protection_effectiveness.values()) / len(
                profile.protection_effectiveness
            )
            metrics["protection_effectiveness"] = avg_effectiveness

        # Estimate other metrics based on available data
        # In a real implementation, these would be calculated from tracked data

        # If therapeutic progress is increasing, other metrics likely improving
        if profile.therapeutic_progress > 0.5:
            metrics["stability_improvement"] = profile.therapeutic_progress * 0.6
            metrics["trigger_sensitivity_change"] = -0.1  # Slight improvement
            metrics["functional_improvement"] = profile.therapeutic_progress * 0.7
            metrics["setback_frequency"] = max(
                0.0, 0.3 - profile.therapeutic_progress * 0.2
            )

        return metrics

    def _assess_regression_risk(
        self, profile: TraumaMemoryProfile, recovery_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess risk of trauma recovery regression."""

        risk_factors = []
        overall_risk = 0.0

        # Recent crisis events increase regression risk
        if (
            profile.last_crisis_event
            and (time.time() - profile.last_crisis_event) < 86400
        ):  # Last 24 hours
            risk_factors.append("recent_crisis_event")
            overall_risk += 0.4

        # Low stability indicates regression risk
        if recovery_metrics["stability_improvement"] < 0.0:
            risk_factors.append("decreasing_stability")
            overall_risk += 0.3

        # High setback frequency
        if recovery_metrics["setback_frequency"] > 0.5:
            risk_factors.append("frequent_setbacks")
            overall_risk += 0.2

        # Increasing trigger sensitivity
        if recovery_metrics["trigger_sensitivity_change"] > 0.0:
            risk_factors.append("increasing_trigger_sensitivity")
            overall_risk += 0.2

        # Low protection effectiveness
        if recovery_metrics["protection_effectiveness"] < 0.5:
            risk_factors.append("ineffective_protections")
            overall_risk += 0.1

        overall_risk = min(1.0, overall_risk)

        return {
            "overall_risk": overall_risk,
            "risk_factors": risk_factors,
            "risk_level": (
                "high"
                if overall_risk > 0.7
                else "moderate" if overall_risk > 0.4 else "low"
            ),
            "monitoring_recommended": overall_risk > 0.5,
        }

    def _generate_recovery_recommendations(
        self,
        profile: TraumaMemoryProfile,
        recovery_metrics: Dict[str, Any],
        regression_assessment: Dict[str, Any],
    ) -> List[str]:
        """Generate recommendations for trauma recovery."""

        recommendations = []

        # Basic recommendations
        recommendations.append("continue_regular_progress_monitoring")

        # Progress-based recommendations
        if recovery_metrics["overall_progress"] < 0.3:
            recommendations.extend(
                [
                    "focus_on_safety_and_stabilization",
                    "build_coping_resources",
                    "strengthen_support_system",
                ]
            )
        elif recovery_metrics["overall_progress"] < 0.7:
            recommendations.extend(
                [
                    "continue_trauma_processing_work",
                    "practice_integration_techniques",
                    "maintain_therapeutic_engagement",
                ]
            )
        else:
            recommendations.extend(
                [
                    "focus_on_relapse_prevention",
                    "strengthen_resilience_factors",
                    "prepare_for_maintenance_phase",
                ]
            )

        # Regression risk recommendations
        if regression_assessment["overall_risk"] > 0.5:
            recommendations.extend(
                [
                    "increase_monitoring_frequency",
                    "strengthen_protective_factors",
                    "review_and_adjust_treatment_approach",
                ]
            )

        # Protection effectiveness recommendations
        if recovery_metrics["protection_effectiveness"] < 0.6:
            recommendations.append("review_and_enhance_protection_mechanisms")

        # Trigger sensitivity recommendations
        if recovery_metrics["trigger_sensitivity_change"] > 0.0:
            recommendations.append("focus_on_trigger_management_skills")

        return recommendations

    def _update_avg_decision_time(self, processing_time_ms: float) -> None:
        """Update average decision time statistic."""
        total_requests = self._performance_stats["access_requests"]
        if total_requests > 0:
            current_avg = self._performance_stats["avg_decision_time_ms"]
            self._performance_stats["avg_decision_time_ms"] = (
                current_avg * (total_requests - 1) + processing_time_ms
            ) / total_requests

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_requests = self._performance_stats["access_requests"]
        approval_rate = (
            (self._performance_stats["access_granted"] / total_requests * 100)
            if total_requests > 0
            else 0.0
        )

        return {
            "total_access_requests": total_requests,
            "access_granted": self._performance_stats["access_granted"],
            "access_denied": self._performance_stats["access_denied"],
            "approval_rate_percent": round(approval_rate, 1),
            "safety_interventions": self._performance_stats["safety_interventions"],
            "therapeutic_accesses": self._performance_stats["therapeutic_accesses"],
            "crisis_preventions": self._performance_stats["crisis_preventions"],
            "avg_decision_time_ms": round(
                self._performance_stats["avg_decision_time_ms"], 2
            ),
            "active_trauma_profiles": len(self._trauma_profiles),
        }

    def clear_caches(self) -> None:
        """Clear all caches."""
        self._risk_assessment_cache.clear()
        self._therapeutic_value_cache.clear()
        self._performance_stats = {
            "access_requests": 0,
            "access_granted": 0,
            "access_denied": 0,
            "safety_interventions": 0,
            "therapeutic_accesses": 0,
            "crisis_preventions": 0,
            "avg_decision_time_ms": 0.0,
        }
