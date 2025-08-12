"""
Secure utilities for the game loop system.

This module provides security-focused utility functions for handling sensitive
operations like UUID validation and sanitization.
"""

import re
import uuid


class UUIDSecurityError(Exception):
    """Raised when UUID validation fails for security reasons."""

    pass


def validate_uuid(uuid_input: str | uuid.UUID, allow_none: bool = False) -> uuid.UUID:
    """
    Safely validate and convert UUID input with security checks.

    Args:
        uuid_input: String or UUID object to validate
        allow_none: Whether to allow None input (returns None if True)

    Returns:
        Valid UUID object

    Raises:
        UUIDSecurityError: If input is invalid or potentially malicious
        TypeError: If input type is unexpected
    """
    if uuid_input is None:
        if allow_none:
            return None
        raise UUIDSecurityError("UUID cannot be None")

    if isinstance(uuid_input, uuid.UUID):
        return uuid_input

    if not isinstance(uuid_input, str):
        raise TypeError(f"UUID input must be string or UUID, got {type(uuid_input)}")

    # Security check: ensure input doesn't contain dangerous characters
    if not re.match(r"^[0-9a-fA-F-]+$", uuid_input):
        raise UUIDSecurityError(f"UUID contains invalid characters: {uuid_input}")

    # Security check: reasonable length limits
    if len(uuid_input) > 50:  # Standard UUID is 36 chars, allow some flexibility
        raise UUIDSecurityError(f"UUID string too long: {len(uuid_input)} characters")

    if len(uuid_input) < 32:  # Minimum for UUID without hyphens
        raise UUIDSecurityError(f"UUID string too short: {len(uuid_input)} characters")

    try:
        return uuid.UUID(uuid_input)
    except ValueError as e:
        raise UUIDSecurityError(f"Invalid UUID format: {uuid_input}") from e


def extract_player_id_from_conversation_id(
    conversation_id: str | uuid.UUID,
) -> uuid.UUID:
    """
    Securely extract player ID from conversation ID.

    This replaces the unsafe conversation_id.split('-')[0] pattern with proper validation.

    Args:
        conversation_id: Conversation UUID to extract player ID from

    Returns:
        Player UUID extracted from conversation ID

    Raises:
        UUIDSecurityError: If conversation ID is invalid or extraction fails
    """
    # First validate the conversation ID
    validated_conversation_id = validate_uuid(conversation_id)

    # Convert to string for consistent processing
    conversation_str = str(validated_conversation_id)

    # Extract the first segment (player ID portion)
    # This is the secure replacement for conversation_id.split('-')[0]
    parts = conversation_str.split("-")
    if len(parts) < 1:
        raise UUIDSecurityError(
            "Conversation ID has invalid format for player ID extraction"
        )

    player_id_part = parts[0]

    # Validate the extracted player ID part
    if not re.match(r"^[0-9a-fA-F]{8}$", player_id_part):
        raise UUIDSecurityError(
            f"Invalid player ID segment in conversation ID: {player_id_part}"
        )

    # Reconstruct as a proper UUID (pad with standard suffix for validation)
    # Note: This assumes a specific conversation ID format. In production,
    # you might want to use a proper mapping table instead.
    try:
        # For now, we'll create a UUID from the player segment
        # In a real system, you'd look up the actual player ID
        player_uuid_str = f"{player_id_part}-0000-0000-0000-000000000000"
        return uuid.UUID(player_uuid_str)
    except ValueError as e:
        raise UUIDSecurityError(
            f"Failed to construct player UUID from conversation ID: {conversation_id}"
        ) from e


def sanitize_uuid_parameter(param: str | uuid.UUID | None) -> uuid.UUID | None:
    """
    Sanitize UUID parameters for safe use in database queries and operations.

    Args:
        param: UUID parameter to sanitize

    Returns:
        Sanitized UUID or None if input was None

    Raises:
        UUIDSecurityError: If parameter is invalid
    """
    if param is None:
        return None

    return validate_uuid(param, allow_none=True)
