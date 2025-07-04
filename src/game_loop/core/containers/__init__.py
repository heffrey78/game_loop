"""
Container management system for nested storage and organization.

This module provides sophisticated container management with nested hierarchies,
specialized container types, and advanced organization features.
"""

from .container_manager import (
    ContainerManager,
    ContainerSpecification,
    ContainerType,
)

__all__ = [
    "ContainerManager",
    "ContainerType",
    "ContainerSpecification",
]
