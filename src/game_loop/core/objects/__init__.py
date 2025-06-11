"""
Object interaction and management systems.

This module provides comprehensive object interaction mechanics, condition tracking,
and integration with other game systems.
"""

from .interaction_system import (
    ObjectInteractionSystem,
    ObjectInteractionType,
    InteractionResult,
)
from .condition_manager import (
    ObjectConditionManager,
    ObjectCondition,
    QualityAspect,
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