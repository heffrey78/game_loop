"""
Inventory management system for comprehensive item handling.

This module provides sophisticated inventory management with realistic constraints,
multi-container support, and advanced organization features.
"""

from .inventory_manager import (
    InventoryConstraint,
    InventoryConstraintType,
    InventoryManager,
    InventorySlot,
)

__all__ = [
    "InventoryManager",
    "InventoryConstraint",
    "InventoryConstraintType",
    "InventorySlot",
]
