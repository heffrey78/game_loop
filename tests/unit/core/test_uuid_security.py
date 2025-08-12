"""Security tests for UUID manipulation utilities."""

import uuid

import pytest

from game_loop.core.utils import (
    UUIDSecurityError,
    extract_player_id_from_conversation_id,
    sanitize_uuid_parameter,
    validate_uuid,
)


class TestValidateUUID:
    """Test UUID validation security functions."""

    def test_validate_valid_uuid_string(self):
        """Test validation of valid UUID string."""
        valid_uuid = "550e8400-e29b-41d4-a716-446655440000"
        result = validate_uuid(valid_uuid)
        assert isinstance(result, uuid.UUID)
        assert str(result) == valid_uuid

    def test_validate_valid_uuid_object(self):
        """Test validation of valid UUID object."""
        valid_uuid = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        result = validate_uuid(valid_uuid)
        assert result == valid_uuid

    def test_validate_none_not_allowed(self):
        """Test that None raises error when not allowed."""
        with pytest.raises(UUIDSecurityError, match="UUID cannot be None"):
            validate_uuid(None)

    def test_validate_none_allowed(self):
        """Test that None returns None when allowed."""
        result = validate_uuid(None, allow_none=True)
        assert result is None

    def test_validate_invalid_type(self):
        """Test that invalid types raise TypeError."""
        with pytest.raises(TypeError, match="UUID input must be string or UUID"):
            validate_uuid(123)

    def test_validate_malicious_characters(self):
        """Test that malicious characters are rejected."""
        malicious_inputs = [
            (
                "550e8400-e29b-41d4-a716-446655440000'; DROP TABLE users; --",
                "UUID contains invalid characters",
            ),
            (
                "550e8400-e29b-41d4-a716-446655440000<script>",
                "UUID contains invalid characters",
            ),
            (
                "550e8400-e29b-41d4-a716-446655440000\x00",
                "UUID contains invalid characters",
            ),
            ("550e8400-e29b-41d4-a716-446655440000\n", "Invalid UUID format"),
        ]
        for malicious_input, expected_error in malicious_inputs:
            with pytest.raises(UUIDSecurityError, match=expected_error):
                validate_uuid(malicious_input)

    def test_validate_too_long(self):
        """Test that overly long strings are rejected."""
        too_long = "a" * 51
        with pytest.raises(UUIDSecurityError, match="UUID string too long"):
            validate_uuid(too_long)

    def test_validate_too_short(self):
        """Test that overly short strings are rejected."""
        too_short = "abc"
        with pytest.raises(UUIDSecurityError, match="UUID string too short"):
            validate_uuid(too_short)

    def test_validate_invalid_format(self):
        """Test that malformed UUIDs are rejected."""
        invalid_formats = [
            ("not-a-uuid-at-all-really-not", "UUID contains invalid characters"),
            (
                "550e8400-e29b-41d4-a716-44665544000x",
                "UUID contains invalid characters",
            ),
            ("550e8400-e29b-41d4-a716", "UUID string too short"),
        ]
        for invalid_format, expected_error in invalid_formats:
            with pytest.raises(UUIDSecurityError, match=expected_error):
                validate_uuid(invalid_format)


class TestExtractPlayerID:
    """Test secure player ID extraction from conversation ID."""

    def test_extract_valid_conversation_id(self):
        """Test extraction from valid conversation ID."""
        conversation_id = "550e8400-e29b-41d4-a716-446655440000"
        result = extract_player_id_from_conversation_id(conversation_id)
        assert isinstance(result, uuid.UUID)
        # Should construct player UUID from first segment
        assert str(result).startswith("550e8400")

    def test_extract_from_uuid_object(self):
        """Test extraction from UUID object."""
        conversation_uuid = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        result = extract_player_id_from_conversation_id(conversation_uuid)
        assert isinstance(result, uuid.UUID)
        assert str(result).startswith("550e8400")

    def test_extract_invalid_conversation_id(self):
        """Test that invalid conversation IDs raise security error."""
        invalid_ids = [
            "not-a-uuid",
            "550e8400-e29b-41d4-a716-446655440000'; DROP TABLE users;",
            None,
            "",
        ]
        for invalid_id in invalid_ids:
            with pytest.raises(UUIDSecurityError):
                extract_player_id_from_conversation_id(invalid_id)

    def test_extract_malformed_uuid_parts(self):
        """Test that malformed UUID parts are rejected."""
        # Create a UUID with invalid player ID segment
        with pytest.raises(UUIDSecurityError):
            extract_player_id_from_conversation_id(
                "zzzzzzzz-e29b-41d4-a716-446655440000"
            )


class TestSanitizeUUIDParameter:
    """Test UUID parameter sanitization."""

    def test_sanitize_valid_uuid_string(self):
        """Test sanitizing valid UUID string."""
        valid_uuid = "550e8400-e29b-41d4-a716-446655440000"
        result = sanitize_uuid_parameter(valid_uuid)
        assert isinstance(result, uuid.UUID)
        assert str(result) == valid_uuid

    def test_sanitize_valid_uuid_object(self):
        """Test sanitizing valid UUID object."""
        valid_uuid = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        result = sanitize_uuid_parameter(valid_uuid)
        assert result == valid_uuid

    def test_sanitize_none(self):
        """Test that None is handled properly."""
        result = sanitize_uuid_parameter(None)
        assert result is None

    def test_sanitize_invalid_input(self):
        """Test that invalid inputs raise security error."""
        invalid_inputs = [
            "malicious'; DROP TABLE users; --",
            "not-a-uuid",
            123,
        ]
        for invalid_input in invalid_inputs:
            with pytest.raises((UUIDSecurityError, TypeError)):
                sanitize_uuid_parameter(invalid_input)


class TestSecurityEdgeCases:
    """Test security edge cases and attack vectors."""

    def test_sql_injection_attempts(self):
        """Test various SQL injection attack patterns."""
        sql_injection_attempts = [
            "'; DROP TABLE conversations; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM users --",
            "'; INSERT INTO conversations VALUES ('malicious'); --",
        ]
        for attack in sql_injection_attempts:
            with pytest.raises(UUIDSecurityError):
                validate_uuid(attack)

    def test_script_injection_attempts(self):
        """Test script injection attempts."""
        script_attempts = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
        ]
        for attack in script_attempts:
            with pytest.raises(UUIDSecurityError):
                validate_uuid(attack)

    def test_null_byte_injection(self):
        """Test null byte injection attempts."""
        null_byte_attempts = [
            "550e8400-e29b-41d4-a716-446655440000\x00.txt",
            "\x00550e8400-e29b-41d4-a716-446655440000",
        ]
        for attack in null_byte_attempts:
            with pytest.raises(UUIDSecurityError):
                validate_uuid(attack)

    def test_path_traversal_attempts(self):
        """Test path traversal attempts."""
        path_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "550e8400-e29b-41d4-a716-446655440000/../",
        ]
        for attack in path_attempts:
            with pytest.raises(UUIDSecurityError):
                validate_uuid(attack)

    def test_buffer_overflow_attempts(self):
        """Test extremely long inputs for buffer overflow."""
        # Test with extremely long string
        overflow_attempt = "a" * 10000
        with pytest.raises(UUIDSecurityError):
            validate_uuid(overflow_attempt)

    def test_unicode_attacks(self):
        """Test Unicode normalization attacks."""
        unicode_attacks = [
            "550e8400\u0000e29b-41d4-a716-446655440000",
            "550e8400\u202ee29b-41d4-a716-446655440000",  # Right-to-left override
        ]
        for attack in unicode_attacks:
            with pytest.raises(UUIDSecurityError):
                validate_uuid(attack)


class TestIntegrationSecurity:
    """Test security in integration scenarios."""

    def test_conversation_id_extraction_security(self):
        """Test that conversation ID extraction is secure."""
        # Test with realistic attack scenarios
        attack_scenarios = [
            "550e8400'; DELETE FROM conversations WHERE '1'='1",
            "550e8400-e29b<script>alert('xss')</script>",
        ]

        for attack in attack_scenarios:
            with pytest.raises(UUIDSecurityError):
                extract_player_id_from_conversation_id(attack)

    def test_repository_parameter_security(self):
        """Test that repository parameters are secured."""
        # This would be an integration test with actual repository
        # For now, test the validation directly
        malicious_npc_id = "npc123'; DROP TABLE npc_personalities; --"

        with pytest.raises(UUIDSecurityError):
            sanitize_uuid_parameter(malicious_npc_id)
