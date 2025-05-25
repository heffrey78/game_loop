"""
Factory for creating usage handlers.
"""

from .base import UsageHandler
from .container_handler import ContainerUsageHandler
from .self_handler import SelfUsageHandler
from .target_handler import TargetUsageHandler


class UsageHandlerFactory:
    """Factory for creating appropriate usage handlers."""

    def get_handler(self, command_target: str | None) -> UsageHandler:
        """
        Get the appropriate handler for the usage scenario.

        Args:
            command_target: The target part of the command (if any)

        Returns:
            UsageHandler: The appropriate handler for this usage scenario
        """
        if command_target and self._is_container_usage(command_target):
            return ContainerUsageHandler()
        elif command_target:
            return TargetUsageHandler()
        else:
            return SelfUsageHandler()

    def _is_container_usage(self, target_name: str) -> bool:
        """
        Check if this is a 'put X in Y' usage pattern.

        Args:
            target_name: The target part of the command

        Returns:
            bool: True if this is a container usage, False otherwise
        """
        return " in " in target_name or " into " in target_name
