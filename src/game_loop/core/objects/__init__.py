"""
Object interaction and management systems.

This module provides comprehensive object interaction mechanics, condition tracking,
and integration with other game systems.
"""

from .condition_manager import (
    ObjectCondition,
    ObjectConditionManager,
    QualityAspect,
)
from .interaction_system import (
    InteractionResult,
    ObjectInteractionSystem,
    ObjectInteractionType,
)
from .object_integration import ObjectSystemIntegration

__all__ = [
    "ObjectInteractionSystem",
    "ObjectInteractionType",
    "InteractionResult",
    "ObjectConditionManager",
    "ObjectCondition",
    "QualityAspect",
    "ObjectSystemIntegration",
]
