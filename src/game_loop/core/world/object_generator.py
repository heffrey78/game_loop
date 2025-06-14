"""
Object Generator Engine for creating dynamic objects with LLM integration.

This module coordinates object generation using context analysis, LLM prompts,
and validation to create contextually appropriate objects.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from uuid import uuid4

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

from game_loop.core.models.object_models import (
    GeneratedObject,
    GenerationMetrics,
    ObjectGenerationContext,
    ObjectInteractions,
    ObjectProperties,
    ObjectValidationResult,
)
from game_loop.core.world.object_theme_manager import ObjectThemeManager
from game_loop.database.session_factory import DatabaseSessionFactory
from game_loop.llm.ollama.client import OllamaClient
from game_loop.state.models import WorldObject, WorldState

logger = logging.getLogger(__name__)


class ObjectGenerator:
    """Main object generation engine with LLM integration."""

    def __init__(
        self,
        world_state: WorldState,
        session_factory: DatabaseSessionFactory,
        llm_client: OllamaClient,
        theme_manager: ObjectThemeManager,
        template_path: str = "templates",
    ):
        self.world_state = world_state
        self.session_factory = session_factory
        self.llm_client = llm_client
        self.theme_manager = theme_manager
        self.template_env = Environment(loader=FileSystemLoader(template_path))
        self._generation_cache: dict[str, GeneratedObject] = {}

    async def generate_object(
        self, context: ObjectGenerationContext
    ) -> GeneratedObject:
        """Generate a complete object with properties and interactions."""
        metrics = GenerationMetrics()

        try:
            logger.info(
                f"Generating object for {context.location.name} ({context.generation_purpose})"
            )

            # Step 1: Determine object type
            start_time = datetime.now()
            object_type = await self.theme_manager.determine_object_type(context)
            metrics.archetype_selection_time_ms = (
                datetime.now() - start_time
            ).total_seconds() * 1000

            logger.debug(f"Selected object type: {object_type}")

            # Step 2: Generate properties
            start_time = datetime.now()
            properties = await self.create_object_properties(object_type, context)
            property_time = (datetime.now() - start_time).total_seconds() * 1000

            # Step 3: Generate interactions
            start_time = datetime.now()
            interactions = await self.create_object_interactions(properties, context)
            interaction_time = (datetime.now() - start_time).total_seconds() * 1000

            metrics.llm_generation_time_ms = property_time + interaction_time

            # Step 4: Create base WorldObject
            base_object = WorldObject(
                object_id=uuid4(),
                name=properties.name,
                description=properties.description,
            )

            # Step 5: Create complete generated object
            generated_object = GeneratedObject(
                base_object=base_object,
                properties=properties,
                interactions=interactions,
                generation_metadata={
                    "object_type": object_type,
                    "location_theme": context.location_theme.name,
                    "generation_purpose": context.generation_purpose,
                    "timestamp": datetime.now().isoformat(),
                    "context_constraints": context.constraints,
                },
                embedding_vector=[],  # Will be populated by storage system
            )

            # Step 6: Validate generated object
            start_time = datetime.now()
            validation_result = await self.validate_generated_object(
                generated_object, context
            )
            metrics.validation_time_ms = (
                datetime.now() - start_time
            ).total_seconds() * 1000

            if not validation_result.is_valid:
                logger.warning(
                    f"Generated object failed validation: {validation_result.validation_errors}"
                )
                # For now, we'll still return it but log the issues

            metrics.mark_complete()
            generated_object.generation_metadata["metrics"] = {
                "total_time_ms": metrics.total_time_ms,
                "llm_time_ms": metrics.llm_generation_time_ms,
                "validation_score": validation_result.consistency_score,
            }

            logger.info(
                f"Generated object '{properties.name}' in {metrics.total_time_ms:.2f}ms"
            )
            return generated_object

        except Exception as e:
            logger.error(f"Error generating object: {e}")
            metrics.mark_complete()

            # Create fallback object
            fallback_properties = ObjectProperties(
                name="mysterious object",
                object_type="mystery",
                description="An object of unknown origin and purpose.",
            )

            fallback_interactions = ObjectInteractions(
                available_actions=["examine"],
                examination_text="This object defies easy categorization.",
            )

            fallback_object = WorldObject(
                object_id=uuid4(),
                name=fallback_properties.name,
                description=fallback_properties.description,
            )

            return GeneratedObject(
                base_object=fallback_object,
                properties=fallback_properties,
                interactions=fallback_interactions,
                generation_metadata={
                    "error": str(e),
                    "fallback": True,
                    "timestamp": datetime.now().isoformat(),
                },
            )

    async def create_object_properties(
        self, object_type: str, context: ObjectGenerationContext
    ) -> ObjectProperties:
        """Generate detailed object properties using LLM."""
        try:
            # Get base template from theme manager
            template_properties = await self.theme_manager.get_object_template(
                object_type, context.location_theme.name
            )

            # Apply cultural variations
            varied_properties = await self.theme_manager.generate_cultural_variations(
                template_properties, context.location
            )

            # Generate enhanced properties with LLM
            try:
                enhanced_properties = await self._generate_properties_with_llm(
                    varied_properties, context
                )
                if enhanced_properties:
                    return enhanced_properties
            except Exception as llm_error:
                logger.warning(
                    f"LLM property generation failed, using template: {llm_error}"
                )

            # Fallback to varied properties if LLM fails
            return varied_properties

        except Exception as e:
            logger.error(f"Error creating object properties: {e}")
            return ObjectProperties(
                name=f"generic {object_type}",
                object_type=object_type,
                description=f"A simple {object_type} found in {context.location.name}.",
            )

    async def create_object_interactions(
        self, properties: ObjectProperties, context: ObjectGenerationContext
    ) -> ObjectInteractions:
        """Generate interaction capabilities for the object."""
        try:
            # Create base interactions based on object type
            base_interactions = self._create_base_interactions(properties)

            # Enhance with LLM if possible
            try:
                enhanced_interactions = await self._generate_interactions_with_llm(
                    properties, base_interactions, context
                )
                if enhanced_interactions:
                    return enhanced_interactions
            except Exception as llm_error:
                logger.warning(
                    f"LLM interaction generation failed, using base: {llm_error}"
                )

            return base_interactions

        except Exception as e:
            logger.error(f"Error creating object interactions: {e}")
            return ObjectInteractions(
                available_actions=["examine"],
                examination_text=f"You examine the {properties.name}.",
            )

    async def validate_generated_object(
        self, generated_object: GeneratedObject, context: ObjectGenerationContext
    ) -> ObjectValidationResult:
        """Validate generated object meets quality and consistency requirements."""
        try:
            errors = []
            warnings = []

            # Basic validation
            if not generated_object.properties.name:
                errors.append("Object name is empty")

            if not generated_object.properties.object_type:
                errors.append("Object type is empty")

            if not generated_object.interactions.available_actions:
                warnings.append("Object has no available actions")

            # Theme consistency check
            theme_consistent = await self.theme_manager.validate_object_consistency(
                generated_object, context.location
            )

            theme_alignment = 1.0 if theme_consistent else 0.5
            if not theme_consistent:
                warnings.append("Object may not fit location theme perfectly")

            # Calculate consistency score
            consistency_score = 1.0
            if errors:
                consistency_score = 0.0
            elif warnings:
                consistency_score = 0.8

            # Placement suitability (simplified)
            placement_suitability = 1.0
            if (
                generated_object.properties.size == "huge"
                and context.location_theme.name == "Cave"
            ):
                placement_suitability = 0.3
                warnings.append("Object may be too large for location")

            return ObjectValidationResult(
                is_valid=len(errors) == 0,
                validation_errors=errors,
                warnings=warnings,
                consistency_score=consistency_score,
                theme_alignment=theme_alignment,
                placement_suitability=placement_suitability,
            )

        except Exception as e:
            logger.error(f"Error validating generated object: {e}")
            return ObjectValidationResult(
                is_valid=False,
                validation_errors=[f"Validation error: {e}"],
                consistency_score=0.0,
            )

    async def _generate_properties_with_llm(
        self, base_properties: ObjectProperties, context: ObjectGenerationContext
    ) -> ObjectProperties | None:
        """Use LLM to enhance object properties."""
        try:
            # Load and render template
            template = self.template_env.get_template(
                "object_generation/object_prompts.j2"
            )
            prompt = template.render(
                object_type=base_properties.object_type,
                location=context.location,
                location_theme=context.location_theme,
                purpose=context.generation_purpose,
                existing_objects=context.existing_objects[:3],  # Limit for context
                constraints=context.constraints,
                base_properties=base_properties,
            )

            # Generate with LLM
            response = self.llm_client.generate(
                model="llama3.1:8b",
                prompt=prompt,
                options={"temperature": 0.7, "max_tokens": 500},
            )

            # Parse response
            object_data = json.loads(response["response"])

            # Create enhanced properties
            enhanced = ObjectProperties(
                name=object_data.get("name", base_properties.name),
                object_type=base_properties.object_type,
                material=object_data.get("material", base_properties.material),
                size=object_data.get("size", base_properties.size),
                weight=object_data.get("weight", base_properties.weight),
                durability=object_data.get("durability", base_properties.durability),
                value=object_data.get("value", base_properties.value),
                special_properties=object_data.get(
                    "special_properties", base_properties.special_properties
                ),
                cultural_significance=object_data.get(
                    "cultural_significance", base_properties.cultural_significance
                ),
                description=object_data.get("description", ""),
            )

            return enhanced

        except (TemplateNotFound, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"LLM property generation failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in LLM property generation: {e}")
            return None

    async def _generate_interactions_with_llm(
        self,
        properties: ObjectProperties,
        base_interactions: ObjectInteractions,
        context: ObjectGenerationContext,
    ) -> ObjectInteractions | None:
        """Use LLM to enhance object interactions."""
        try:
            template = self.template_env.get_template(
                "object_generation/interaction_templates.j2"
            )
            prompt = template.render(
                properties=properties,
                base_interactions=base_interactions,
                location=context.location,
                location_theme=context.location_theme,
                purpose=context.generation_purpose,
            )

            response = self.llm_client.generate(
                model="llama3.1:8b",
                prompt=prompt,
                options={"temperature": 0.6, "max_tokens": 400},
            )

            interaction_data = json.loads(response["response"])

            enhanced = ObjectInteractions(
                available_actions=interaction_data.get(
                    "available_actions", base_interactions.available_actions
                ),
                use_requirements=interaction_data.get(
                    "use_requirements", base_interactions.use_requirements
                ),
                interaction_results=interaction_data.get(
                    "interaction_results", base_interactions.interaction_results
                ),
                state_changes=interaction_data.get(
                    "state_changes", base_interactions.state_changes
                ),
                consumable=interaction_data.get(
                    "consumable", base_interactions.consumable
                ),
                portable=interaction_data.get("portable", base_interactions.portable),
                examination_text=interaction_data.get(
                    "examination_text", base_interactions.examination_text
                ),
                hidden_properties=interaction_data.get(
                    "hidden_properties", base_interactions.hidden_properties
                ),
            )

            return enhanced

        except (TemplateNotFound, json.JSONDecodeError, KeyError) as e:
            logger.warning(f"LLM interaction generation failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in LLM interaction generation: {e}")
            return None

    def _create_base_interactions(
        self, properties: ObjectProperties
    ) -> ObjectInteractions:
        """Create basic interactions based on object type and properties."""
        try:
            actions = ["examine"]
            use_requirements = {}
            interaction_results = {}
            state_changes = {}
            consumable = False
            portable = True
            examination_text = (
                f"You examine the {properties.name}. {properties.description}"
            )

            # Object type specific interactions
            object_type = properties.object_type

            if object_type == "weapon":
                actions.extend(["wield", "attack"])
                use_requirements["attack"] = "target"
                interaction_results["wield"] = f"You grip the {properties.name} firmly."
                portable = True

            elif object_type == "tool":
                actions.extend(["use", "apply"])
                use_requirements["use"] = "target_or_material"
                interaction_results["use"] = (
                    f"You use the {properties.name} skillfully."
                )

            elif object_type == "container":
                actions.extend(["open", "close", "search"])
                interaction_results["open"] = f"You open the {properties.name}."
                interaction_results["close"] = f"You close the {properties.name}."
                portable = properties.size not in ["large", "huge"]

            elif object_type == "book" or object_type == "knowledge":
                actions.extend(["read", "study"])
                interaction_results["read"] = f"You read from the {properties.name}."
                interaction_results["study"] = (
                    f"You study the {properties.name} carefully."
                )

            elif object_type == "natural":
                if "medicinal" in properties.special_properties:
                    actions.extend(["gather", "consume"])
                    consumable = True
                    interaction_results["consume"] = (
                        f"You consume the {properties.name}."
                    )
                else:
                    actions.append("gather")

            elif object_type == "treasure":
                actions.extend(["appraise", "admire"])
                interaction_results["appraise"] = (
                    f"The {properties.name} appears valuable."
                )

            elif object_type == "furniture":
                if (
                    "chair" in properties.name.lower()
                    or "bench" in properties.name.lower()
                ):
                    actions.append("sit")
                    interaction_results["sit"] = f"You sit on the {properties.name}."
                portable = False

            # Size-based portability
            if properties.size in ["huge", "massive"]:
                portable = False
                if "take" in actions:
                    actions.remove("take")

            # Add standard portable actions
            if portable:
                actions.extend(["take", "drop"])
                interaction_results["take"] = f"You take the {properties.name}."
                interaction_results["drop"] = f"You drop the {properties.name}."

            return ObjectInteractions(
                available_actions=actions,
                use_requirements=use_requirements,
                interaction_results=interaction_results,
                state_changes=state_changes,
                consumable=consumable,
                portable=portable,
                examination_text=examination_text,
                hidden_properties={},
            )

        except Exception as e:
            logger.error(f"Error creating base interactions: {e}")
            return ObjectInteractions(
                available_actions=["examine"],
                examination_text=f"You examine the {properties.name}.",
            )
