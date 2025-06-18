"""
NPC Knowledge Engine for providing contextual and role-specific knowledge.

This module manages what NPCs know about locations, objects, and situations
based on their roles and provides contextually appropriate information.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from game_loop.search.service import SemanticSearchService

logger = logging.getLogger(__name__)


class NPCKnowledgeEngine:
    """Manage NPC knowledge about locations, objects, and situations."""

    def __init__(
        self, semantic_search_service: Optional["SemanticSearchService"] = None
    ):
        self.semantic_search = semantic_search_service
        self.knowledge_cache: dict[str, dict[str, Any]] = {}

        # Define role-specific knowledge patterns
        self.role_knowledge_patterns = {
            "security_guard": {
                "primary_knowledge": [
                    "building_layout",
                    "security_procedures",
                    "access_control",
                    "patrol_routes",
                    "emergency_protocols",
                    "personnel_clearances",
                ],
                "secondary_knowledge": [
                    "visitor_policies",
                    "restricted_areas",
                    "incident_reports",
                    "communication_channels",
                    "shift_schedules",
                ],
                "knowledge_depth": "detailed",
                "information_sharing": "cautious",  # Guards are careful about what they share
            },
            "scholar": {
                "primary_knowledge": [
                    "research_materials",
                    "historical_information",
                    "academic_resources",
                    "book_collections",
                    "archive_systems",
                    "research_methodologies",
                ],
                "secondary_knowledge": [
                    "academic_protocols",
                    "library_systems",
                    "publication_records",
                    "research_networks",
                    "scholarly_debates",
                ],
                "knowledge_depth": "comprehensive",
                "information_sharing": "generous",  # Scholars love to share knowledge
            },
            "administrator": {
                "primary_knowledge": [
                    "procedures",
                    "regulations",
                    "documentation_requirements",
                    "organizational_structure",
                    "policy_guidelines",
                    "scheduling",
                ],
                "secondary_knowledge": [
                    "form_processing",
                    "approval_workflows",
                    "compliance_requirements",
                    "record_keeping",
                    "administrative_contacts",
                ],
                "knowledge_depth": "procedural",
                "information_sharing": "official",  # Administrators share official information
            },
            "merchant": {
                "primary_knowledge": [
                    "item_properties",
                    "market_values",
                    "trade_routes",
                    "customer_preferences",
                    "inventory_management",
                    "negotiation",
                ],
                "secondary_knowledge": [
                    "supply_chains",
                    "quality_assessment",
                    "pricing_strategies",
                    "competitor_analysis",
                    "market_trends",
                ],
                "knowledge_depth": "practical",
                "information_sharing": "transactional",  # Merchants share info that helps sales
            },
        }

    async def get_npc_knowledge(
        self,
        npc_archetype: str,
        location_id: str,
        topic: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get relevant knowledge for NPC based on their role and location."""
        knowledge_key = f"{npc_archetype}_{location_id}_{topic or 'general'}"

        if knowledge_key in self.knowledge_cache:
            return self.knowledge_cache[knowledge_key]

        # Gather different types of knowledge
        location_knowledge = await self._get_location_knowledge(
            npc_archetype, location_id, context
        )
        role_knowledge = await self._get_role_specific_knowledge(
            npc_archetype, topic, context
        )
        situational_knowledge = await self._get_situational_knowledge(
            location_id, topic, context
        )

        combined_knowledge = {
            "location": location_knowledge,
            "role": role_knowledge,
            "situation": situational_knowledge,
            "sharing_style": self._get_sharing_style(npc_archetype),
            "knowledge_confidence": self._calculate_knowledge_confidence(
                npc_archetype, topic
            ),
        }

        # Cache for future use
        self.knowledge_cache[knowledge_key] = combined_knowledge
        return combined_knowledge

    async def _get_location_knowledge(
        self,
        npc_archetype: str,
        location_id: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get what this NPC type would know about this location."""
        # Get basic location details (would normally come from database)
        location_details = await self._get_location_details(location_id)

        # Filter and enhance based on NPC role
        if npc_archetype == "security_guard":
            return {
                "layout": {
                    "exits": location_details.get("exits", {}),
                    "alternate_routes": location_details.get("alternate_routes", []),
                    "restricted_access_points": location_details.get(
                        "restricted_areas", []
                    ),
                },
                "security_features": {
                    "cameras": location_details.get("surveillance", []),
                    "access_controls": location_details.get("access_controls", []),
                    "alarms": location_details.get("security_systems", []),
                },
                "access_restrictions": location_details.get(
                    "clearance_requirements", []
                ),
                "patrol_information": {
                    "patrol_frequency": location_details.get(
                        "patrol_schedule", "hourly"
                    ),
                    "key_checkpoints": location_details.get("checkpoints", []),
                    "incident_history": location_details.get("security_incidents", []),
                },
            }

        elif npc_archetype == "scholar":
            return {
                "research_materials": {
                    "books": location_details.get("book_collections", []),
                    "archives": location_details.get("archived_materials", []),
                    "databases": location_details.get("digital_resources", []),
                },
                "historical_significance": {
                    "establishment_date": location_details.get("founded", "unknown"),
                    "notable_events": location_details.get("historical_events", []),
                    "architectural_features": location_details.get("architecture", {}),
                },
                "academic_resources": {
                    "research_facilities": location_details.get(
                        "research_equipment", []
                    ),
                    "collaboration_spaces": location_details.get("meeting_areas", []),
                    "specialized_collections": location_details.get(
                        "special_collections", []
                    ),
                },
            }

        elif npc_archetype == "administrator":
            return {
                "administrative_functions": {
                    "office_purposes": location_details.get(
                        "administrative_functions", []
                    ),
                    "departments": location_details.get("departments", []),
                    "service_hours": location_details.get("operating_hours", {}),
                },
                "procedural_information": {
                    "required_forms": location_details.get(
                        "required_documentation", []
                    ),
                    "processing_times": location_details.get(
                        "processing_timeframes", {}
                    ),
                    "approval_chains": location_details.get("approval_hierarchy", []),
                },
                "regulatory_compliance": {
                    "applicable_regulations": location_details.get("regulations", []),
                    "inspection_schedules": location_details.get("inspections", []),
                    "compliance_status": location_details.get(
                        "compliance_level", "current"
                    ),
                },
            }

        elif npc_archetype == "merchant":
            return {
                "commercial_aspects": {
                    "customer_traffic": location_details.get(
                        "foot_traffic", "moderate"
                    ),
                    "trade_opportunities": location_details.get("trade_potential", []),
                    "competitor_presence": location_details.get("other_merchants", []),
                },
                "logistics": {
                    "supply_access": location_details.get("supply_routes", []),
                    "storage_facilities": location_details.get("storage_options", []),
                    "transportation": location_details.get("transport_links", []),
                },
            }

        # Generic/fallback knowledge
        return {
            "general_information": {
                "description": location_details.get("description", ""),
                "notable_features": location_details.get("features", []),
                "typical_activities": location_details.get("activities", []),
            }
        }

    async def _get_role_specific_knowledge(
        self,
        npc_archetype: str,
        topic: str | None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get knowledge specific to the NPC's professional role."""
        role_pattern = self.role_knowledge_patterns.get(npc_archetype, {})

        primary_knowledge = role_pattern.get("primary_knowledge", [])
        secondary_knowledge = role_pattern.get("secondary_knowledge", [])

        knowledge = {
            "expertise_areas": primary_knowledge,
            "supporting_knowledge": secondary_knowledge,
            "professional_experience": self._generate_experience_knowledge(
                npc_archetype
            ),
            "role_specific_insights": self._generate_role_insights(
                npc_archetype, topic
            ),
        }

        return knowledge

    async def _get_situational_knowledge(
        self,
        location_id: str,
        topic: str | None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get knowledge about current situation and context."""
        situation = {
            "current_conditions": {
                "time_context": (
                    context.get("time_of_day", "unknown") if context else "unknown"
                ),
                "activity_level": (
                    context.get("activity_level", "normal") if context else "normal"
                ),
                "special_circumstances": (
                    context.get("special_events", []) if context else []
                ),
            },
            "relevant_context": (
                self._analyze_contextual_relevance(topic, context) if topic else {}
            ),
            "environmental_factors": {
                "accessibility": "standard",
                "noise_level": "normal",
                "lighting": "adequate",
            },
        }

        return situation

    def _get_sharing_style(self, npc_archetype: str) -> dict[str, Any]:
        """Get how this NPC archetype shares information."""
        role_pattern = self.role_knowledge_patterns.get(npc_archetype, {})
        sharing_approach = role_pattern.get("information_sharing", "neutral")

        sharing_styles = {
            "cautious": {
                "verification_required": True,
                "detail_level": "minimal",
                "follow_up_questions": True,
                "official_channels_preferred": True,
            },
            "generous": {
                "verification_required": False,
                "detail_level": "comprehensive",
                "follow_up_questions": False,
                "additional_resources_offered": True,
            },
            "official": {
                "verification_required": True,
                "detail_level": "procedural",
                "documentation_referenced": True,
                "formal_language": True,
            },
            "transactional": {
                "verification_required": False,
                "detail_level": "practical",
                "value_proposition_included": True,
                "reciprocity_expected": True,
            },
        }

        return sharing_styles.get(
            sharing_approach,
            {"verification_required": False, "detail_level": "standard"},
        )

    def _calculate_knowledge_confidence(
        self, npc_archetype: str, topic: str | None
    ) -> float:
        """Calculate how confident this NPC would be about the topic."""
        if not topic:
            return 0.7  # Moderate confidence for general topics

        role_pattern = self.role_knowledge_patterns.get(npc_archetype, {})
        primary_areas = role_pattern.get("primary_knowledge", [])
        secondary_areas = role_pattern.get("secondary_knowledge", [])

        # Check if topic matches knowledge areas
        topic_lower = topic.lower().replace("_", " ")

        for area in primary_areas:
            if (
                area.lower().replace("_", " ") in topic_lower
                or topic_lower in area.lower()
            ):
                return 0.9  # High confidence

        for area in secondary_areas:
            if (
                area.lower().replace("_", " ") in topic_lower
                or topic_lower in area.lower()
            ):
                return 0.7  # Good confidence

        return 0.4  # Low confidence for unfamiliar topics

    def _generate_experience_knowledge(self, npc_archetype: str) -> dict[str, Any]:
        """Generate professional experience-based knowledge."""
        experience_templates = {
            "security_guard": {
                "years_experience": "5-15 years",
                "key_experiences": [
                    "Managing access control during high-security events",
                    "Coordinating emergency response procedures",
                    "Training new security personnel",
                ],
                "professional_insights": [
                    "Security is about prevention, not reaction",
                    "Communication is key to effective security",
                    "Every location has unique security challenges",
                ],
            },
            "scholar": {
                "years_experience": "10-20 years",
                "key_experiences": [
                    "Publishing research in peer-reviewed journals",
                    "Mentoring graduate students",
                    "Organizing academic conferences",
                ],
                "professional_insights": [
                    "Knowledge builds upon itself over time",
                    "Collaboration enhances research quality",
                    "Primary sources are invaluable",
                ],
            },
            "administrator": {
                "years_experience": "8-18 years",
                "key_experiences": [
                    "Streamlining organizational processes",
                    "Managing regulatory compliance audits",
                    "Implementing new administrative systems",
                ],
                "professional_insights": [
                    "Clear procedures prevent most problems",
                    "Documentation is essential for consistency",
                    "Stakeholder communication drives success",
                ],
            },
        }

        return experience_templates.get(npc_archetype, {})

    def _generate_role_insights(
        self, npc_archetype: str, topic: str | None
    ) -> list[str]:
        """Generate role-specific insights about the topic."""
        if not topic:
            return []

        role_insights = {
            "security_guard": {
                "location_inquiry": [
                    "From a security perspective, this area requires special attention",
                    "I've noticed some unusual patterns in this location recently",
                    "The security protocols here have been updated recently",
                ],
                "access_request": [
                    "Access control is taken very seriously in this facility",
                    "There are specific procedures that must be followed",
                    "Authorization must come through proper channels",
                ],
            },
            "scholar": {
                "research_inquiry": [
                    "This topic has been extensively studied in recent literature",
                    "There are several theoretical frameworks that apply here",
                    "I recall some fascinating research findings on this subject",
                ],
                "information_request": [
                    "The information you seek has interesting historical context",
                    "Multiple sources should be consulted for comprehensive understanding",
                    "This connects to broader academic discussions",
                ],
            },
        }

        archetype_insights = role_insights.get(npc_archetype, {})
        return archetype_insights.get(topic, [])

    def _analyze_contextual_relevance(
        self, topic: str, context: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Analyze how the topic relates to current context."""
        if not context:
            return {}

        relevance = {
            "context_match": False,
            "urgency_level": "normal",
            "additional_considerations": [],
        }

        # Analyze time sensitivity
        if "urgent" in topic.lower() or "emergency" in topic.lower():
            relevance["urgency_level"] = "high"
            relevance["additional_considerations"].append(
                "immediate_attention_required"
            )

        # Analyze location relevance
        location_type = context.get("location_type", "")
        if location_type and location_type.lower() in topic.lower():
            relevance["context_match"] = True
            relevance["additional_considerations"].append("location_specific_expertise")

        return relevance

    async def _get_location_details(self, location_id: str) -> dict[str, Any]:
        """Get detailed information about a location."""
        # This would normally query the database
        # For now, return mock data structure
        return {
            "description": f"Location {location_id} details",
            "exits": {"north": "corridor", "south": "lobby"},
            "features": ["modern furniture", "good lighting"],
            "clearance_requirements": ["standard_access"],
            "activities": ["meetings", "paperwork"],
            "operating_hours": {"start": "09:00", "end": "17:00"},
        }

    def get_knowledge_summary(self, npc_archetype: str, location_id: str) -> str:
        """Get a summary of what this NPC knows."""
        role_pattern = self.role_knowledge_patterns.get(npc_archetype, {})
        primary_areas = role_pattern.get("primary_knowledge", [])

        if not primary_areas:
            return "This person has general knowledge about the area."

        expertise_desc = ", ".join(primary_areas[:3])  # Show top 3 areas
        return (
            f"This {npc_archetype} has expertise in {expertise_desc} and related areas."
        )

    def suggest_knowledge_topics(self, npc_archetype: str) -> list[str]:
        """Suggest topics this NPC would be knowledgeable about."""
        role_pattern = self.role_knowledge_patterns.get(npc_archetype, {})
        primary_knowledge = role_pattern.get("primary_knowledge", [])
        secondary_knowledge = role_pattern.get("secondary_knowledge", [])

        # Format for user-friendly display
        suggestions = []
        for area in primary_knowledge[:3]:  # Top 3 primary areas
            suggestions.append(area.replace("_", " ").title())

        for area in secondary_knowledge[:2]:  # Top 2 secondary areas
            suggestions.append(area.replace("_", " ").title())

        return suggestions
