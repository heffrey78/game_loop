"""Input validation utilities for the emotional memory system."""

import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from .constants import EmotionalThresholds
from .emotional_context import EmotionalMemoryType, MoodState, MemoryProtectionLevel
from .exceptions import ValidationError, SecurityError


def validate_uuid(value: Any, field_name: str) -> uuid.UUID:
    """Validate and convert UUID values."""
    if value is None:
        raise ValidationError(f"{field_name} cannot be None", field_name=field_name)

    if isinstance(value, uuid.UUID):
        return value

    if isinstance(value, str):
        try:
            return uuid.UUID(value)
        except ValueError as e:
            raise ValidationError(
                f"{field_name} must be a valid UUID",
                field_name=field_name,
                actual_value=value,
            ) from e

    raise ValidationError(
        f"{field_name} must be UUID or string, got {type(value).__name__}",
        field_name=field_name,
        expected_type="UUID",
        actual_value=value,
    )


def validate_probability(
    value: Any, field_name: str, allow_none: bool = False
) -> float:
    """Validate probability values (0.0 to 1.0)."""
    if value is None and allow_none:
        return None

    if value is None:
        raise ValidationError(f"{field_name} cannot be None", field_name=field_name)

    try:
        float_value = float(value)
    except (TypeError, ValueError) as e:
        raise ValidationError(
            f"{field_name} must be a number",
            field_name=field_name,
            expected_type="float",
            actual_value=value,
        ) from e

    if not (0.0 <= float_value <= 1.0):
        raise ValidationError(
            f"{field_name} must be between 0.0 and 1.0",
            field_name=field_name,
            actual_value=float_value,
        )

    return float_value


def validate_intensity(value: Any, field_name: str = "intensity") -> float:
    """Validate emotional intensity values."""
    return validate_probability(value, field_name)


def validate_significance(value: Any, field_name: str = "significance") -> float:
    """Validate significance scores."""
    return validate_probability(value, field_name)


def validate_trust_level(value: Any, field_name: str = "trust_level") -> float:
    """Validate trust level values."""
    return validate_probability(value, field_name)


def validate_relationship_impact(
    value: Any, field_name: str = "relationship_impact"
) -> float:
    """Validate relationship impact values (-1.0 to 1.0)."""
    if value is None:
        raise ValidationError(f"{field_name} cannot be None", field_name=field_name)

    try:
        float_value = float(value)
    except (TypeError, ValueError) as e:
        raise ValidationError(
            f"{field_name} must be a number",
            field_name=field_name,
            expected_type="float",
            actual_value=value,
        ) from e

    if not (-1.0 <= float_value <= 1.0):
        raise ValidationError(
            f"{field_name} must be between -1.0 and 1.0",
            field_name=field_name,
            actual_value=float_value,
        )

    return float_value


def validate_mood_state(value: Any, field_name: str = "mood_state") -> MoodState:
    """Validate mood state values."""
    if value is None:
        raise ValidationError(f"{field_name} cannot be None", field_name=field_name)

    if isinstance(value, MoodState):
        return value

    if isinstance(value, str):
        try:
            return MoodState(value.lower())
        except ValueError:
            valid_moods = [mood.value for mood in MoodState]
            raise ValidationError(
                f"{field_name} must be one of: {valid_moods}",
                field_name=field_name,
                actual_value=value,
            )

    raise ValidationError(
        f"{field_name} must be MoodState or string, got {type(value).__name__}",
        field_name=field_name,
        expected_type="MoodState",
        actual_value=value,
    )


def validate_emotional_memory_type(
    value: Any, field_name: str = "emotional_type"
) -> EmotionalMemoryType:
    """Validate emotional memory type values."""
    if value is None:
        raise ValidationError(f"{field_name} cannot be None", field_name=field_name)

    if isinstance(value, EmotionalMemoryType):
        return value

    if isinstance(value, str):
        try:
            return EmotionalMemoryType(value.lower())
        except ValueError:
            valid_types = [mem_type.value for mem_type in EmotionalMemoryType]
            raise ValidationError(
                f"{field_name} must be one of: {valid_types}",
                field_name=field_name,
                actual_value=value,
            )

    raise ValidationError(
        f"{field_name} must be EmotionalMemoryType or string, got {type(value).__name__}",
        field_name=field_name,
        expected_type="EmotionalMemoryType",
        actual_value=value,
    )


def validate_protection_level(
    value: Any, field_name: str = "protection_level"
) -> MemoryProtectionLevel:
    """Validate memory protection level values."""
    if value is None:
        raise ValidationError(f"{field_name} cannot be None", field_name=field_name)

    if isinstance(value, MemoryProtectionLevel):
        return value

    if isinstance(value, str):
        try:
            return MemoryProtectionLevel(value.lower())
        except ValueError:
            valid_levels = [level.value for level in MemoryProtectionLevel]
            raise ValidationError(
                f"{field_name} must be one of: {valid_levels}",
                field_name=field_name,
                actual_value=value,
            )

    raise ValidationError(
        f"{field_name} must be MemoryProtectionLevel or string, got {type(value).__name__}",
        field_name=field_name,
        expected_type="MemoryProtectionLevel",
        actual_value=value,
    )


def validate_positive_number(
    value: Any, field_name: str, allow_zero: bool = True
) -> Union[int, float]:
    """Validate positive numbers."""
    if value is None:
        raise ValidationError(f"{field_name} cannot be None", field_name=field_name)

    try:
        if isinstance(value, int):
            num_value = value
        else:
            num_value = float(value)
    except (TypeError, ValueError) as e:
        raise ValidationError(
            f"{field_name} must be a number",
            field_name=field_name,
            expected_type="number",
            actual_value=value,
        ) from e

    min_value = 0 if allow_zero else 0.001
    if num_value < min_value:
        raise ValidationError(
            f"{field_name} must be {'non-negative' if allow_zero else 'positive'}",
            field_name=field_name,
            actual_value=num_value,
        )

    return num_value


def validate_string_content(
    value: Any,
    field_name: str,
    min_length: int = 0,
    max_length: int = 10000,
    allow_empty: bool = True,
) -> str:
    """Validate and sanitize string content."""
    if value is None:
        if allow_empty:
            return ""
        raise ValidationError(f"{field_name} cannot be None", field_name=field_name)

    if not isinstance(value, str):
        raise ValidationError(
            f"{field_name} must be a string, got {type(value).__name__}",
            field_name=field_name,
            expected_type="str",
            actual_value=value,
        )

    # Check length constraints
    if len(value) < min_length:
        raise ValidationError(
            f"{field_name} must be at least {min_length} characters",
            field_name=field_name,
            actual_value=f"'{value}' (length: {len(value)})",
        )

    if len(value) > max_length:
        raise ValidationError(
            f"{field_name} cannot exceed {max_length} characters",
            field_name=field_name,
            actual_value=f"length: {len(value)}",
        )

    # Basic content validation
    if not allow_empty and not value.strip():
        raise ValidationError(
            f"{field_name} cannot be empty or only whitespace",
            field_name=field_name,
            actual_value=value,
        )

    return value.strip()


def sanitize_user_content(content: str, max_length: int = 5000) -> str:
    """Sanitize user-provided content for security."""
    if not content:
        return ""

    # Truncate if too long
    if len(content) > max_length:
        content = content[:max_length]

    # Remove potentially dangerous characters
    # Keep alphanumeric, common punctuation, and whitespace
    sanitized = re.sub(r'[^\w\s\-.,!?;:()\'"]+', "", content)

    # Normalize whitespace
    sanitized = re.sub(r"\s+", " ", sanitized).strip()

    return sanitized


def validate_keywords_list(
    value: Any,
    field_name: str = "keywords",
    max_keywords: int = 50,
    max_keyword_length: int = 100,
) -> List[str]:
    """Validate list of keywords."""
    if value is None:
        return []

    if not isinstance(value, (list, tuple)):
        raise ValidationError(
            f"{field_name} must be a list, got {type(value).__name__}",
            field_name=field_name,
            expected_type="list",
            actual_value=value,
        )

    if len(value) > max_keywords:
        raise ValidationError(
            f"{field_name} cannot contain more than {max_keywords} items",
            field_name=field_name,
            actual_value=f"{len(value)} items",
        )

    validated_keywords = []
    for i, keyword in enumerate(value):
        try:
            validated_keyword = validate_string_content(
                keyword,
                f"{field_name}[{i}]",
                min_length=1,
                max_length=max_keyword_length,
                allow_empty=False,
            )
            # Additional sanitization for keywords
            validated_keyword = sanitize_user_content(
                validated_keyword, max_keyword_length
            )
            if validated_keyword:  # Only add non-empty keywords after sanitization
                validated_keywords.append(validated_keyword.lower())
        except ValidationError:
            # Skip invalid keywords instead of failing
            continue

    return validated_keywords


def validate_memory_age(age_hours: Any, field_name: str = "memory_age_hours") -> float:
    """Validate memory age in hours."""
    age = validate_positive_number(age_hours, field_name, allow_zero=True)

    # Reasonable upper limit - 10 years
    max_age_hours = 10 * 365 * 24
    if age > max_age_hours:
        raise ValidationError(
            f"{field_name} exceeds reasonable maximum age ({max_age_hours} hours)",
            field_name=field_name,
            actual_value=age,
        )

    return float(age)


def validate_timestamp(value: Any, field_name: str = "timestamp") -> float:
    """Validate timestamp values."""
    if value is None:
        raise ValidationError(f"{field_name} cannot be None", field_name=field_name)

    if isinstance(value, datetime):
        return value.timestamp()

    try:
        timestamp = float(value)
    except (TypeError, ValueError) as e:
        raise ValidationError(
            f"{field_name} must be a timestamp (float) or datetime",
            field_name=field_name,
            expected_type="float",
            actual_value=value,
        ) from e

    # Reasonable timestamp range (1970 to 2100)
    min_timestamp = 0
    max_timestamp = 4102444800  # 2100-01-01

    if not (min_timestamp <= timestamp <= max_timestamp):
        raise ValidationError(
            f"{field_name} must be a valid timestamp between 1970 and 2100",
            field_name=field_name,
            actual_value=timestamp,
        )

    return timestamp


def validate_personality_traits(
    traits: Any, field_name: str = "traits"
) -> Dict[str, float]:
    """Validate personality traits dictionary."""
    if traits is None:
        return {}

    if not isinstance(traits, dict):
        raise ValidationError(
            f"{field_name} must be a dictionary, got {type(traits).__name__}",
            field_name=field_name,
            expected_type="dict",
            actual_value=traits,
        )

    validated_traits = {}
    for trait_name, trait_value in traits.items():
        # Validate trait name
        if not isinstance(trait_name, str):
            continue  # Skip invalid trait names

        trait_name = trait_name.strip().lower()
        if not trait_name or len(trait_name) > 50:
            continue  # Skip invalid trait names

        # Validate trait value
        try:
            trait_value = validate_probability(
                trait_value, f"{field_name}.{trait_name}"
            )
            validated_traits[trait_name] = trait_value
        except ValidationError:
            continue  # Skip invalid trait values

    return validated_traits


def validate_mood_accessibility(
    accessibility: Any, field_name: str = "mood_accessibility"
) -> Dict[MoodState, float]:
    """Validate mood accessibility dictionary."""
    if accessibility is None:
        return {}

    if not isinstance(accessibility, dict):
        raise ValidationError(
            f"{field_name} must be a dictionary, got {type(accessibility).__name__}",
            field_name=field_name,
            expected_type="dict",
            actual_value=accessibility,
        )

    validated_accessibility = {}
    for mood_key, access_value in accessibility.items():
        try:
            # Validate mood state
            if isinstance(mood_key, str):
                mood_state = validate_mood_state(mood_key, f"{field_name}.key")
            elif isinstance(mood_key, MoodState):
                mood_state = mood_key
            else:
                continue  # Skip invalid mood keys

            # Validate accessibility value
            access_value = validate_probability(
                access_value, f"{field_name}.{mood_state.value}"
            )
            validated_accessibility[mood_state] = access_value

        except ValidationError:
            continue  # Skip invalid entries

    return validated_accessibility


class InputValidator:
    """Comprehensive input validator for emotional memory system."""

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode

    def validate_emotional_significance_input(self, **kwargs) -> Dict[str, Any]:
        """Validate inputs for emotional significance analysis."""
        validated = {}

        # Required fields
        if "overall_significance" in kwargs:
            validated["overall_significance"] = validate_significance(
                kwargs["overall_significance"], "overall_significance"
            )

        if "emotional_type" in kwargs:
            validated["emotional_type"] = validate_emotional_memory_type(
                kwargs["emotional_type"], "emotional_type"
            )

        if "intensity_score" in kwargs:
            validated["intensity_score"] = validate_intensity(
                kwargs["intensity_score"], "intensity_score"
            )

        if "personal_relevance" in kwargs:
            validated["personal_relevance"] = validate_probability(
                kwargs["personal_relevance"], "personal_relevance"
            )

        if "relationship_impact" in kwargs:
            validated["relationship_impact"] = validate_relationship_impact(
                kwargs["relationship_impact"], "relationship_impact"
            )

        if "protection_level" in kwargs:
            validated["protection_level"] = validate_protection_level(
                kwargs["protection_level"], "protection_level"
            )

        if "mood_accessibility" in kwargs:
            validated["mood_accessibility"] = validate_mood_accessibility(
                kwargs["mood_accessibility"], "mood_accessibility"
            )

        if "contributing_factors" in kwargs:
            validated["contributing_factors"] = validate_keywords_list(
                kwargs["contributing_factors"], "contributing_factors"
            )

        return validated

    def validate_mood_update_input(self, **kwargs) -> Dict[str, Any]:
        """Validate inputs for mood updates."""
        validated = {}

        if "npc_id" in kwargs:
            validated["npc_id"] = validate_uuid(kwargs["npc_id"], "npc_id")

        if "new_mood" in kwargs:
            validated["new_mood"] = validate_mood_state(kwargs["new_mood"], "new_mood")

        if "intensity" in kwargs:
            validated["intensity"] = validate_intensity(
                kwargs["intensity"], "intensity"
            )

        if "trigger_source" in kwargs:
            validated["trigger_source"] = validate_string_content(
                kwargs["trigger_source"], "trigger_source", max_length=200
            )

        return validated

    def validate_memory_retrieval_input(self, **kwargs) -> Dict[str, Any]:
        """Validate inputs for memory retrieval."""
        validated = {}

        if "npc_id" in kwargs:
            validated["npc_id"] = validate_uuid(kwargs["npc_id"], "npc_id")

        if "significance_threshold" in kwargs:
            validated["significance_threshold"] = validate_probability(
                kwargs["significance_threshold"], "significance_threshold"
            )

        if "trust_level" in kwargs:
            validated["trust_level"] = validate_trust_level(
                kwargs["trust_level"], "trust_level"
            )

        if "max_results" in kwargs:
            max_results = validate_positive_number(kwargs["max_results"], "max_results")
            if max_results > 1000:  # Reasonable limit
                raise ValidationError(
                    "max_results cannot exceed 1000",
                    field_name="max_results",
                    actual_value=max_results,
                )
            validated["max_results"] = int(max_results)

        return validated

    def handle_validation_error(
        self, error: Exception, context: str = ""
    ) -> ValidationError:
        """Convert various errors to ValidationError with context."""
        if isinstance(error, ValidationError):
            return error

        context_info = f" in {context}" if context else ""
        return ValidationError(f"Validation failed{context_info}: {error}")


# Global validator instance
default_validator = InputValidator(strict_mode=True)
