"""
Exception classes for action classification system.

This module defines specific exception types for better error handling
and debugging in the action classification pipeline.
"""


class ActionClassificationError(Exception):
    """Base exception for action classification errors."""

    def __init__(self, message: str, input_text: str | None = None):
        """
        Initialize the exception.

        Args:
            message: Error message
            input_text: Original input text that caused the error
        """
        super().__init__(message)
        self.input_text = input_text


class PatternMatchError(ActionClassificationError):
    """Error in pattern matching during rule-based classification."""

    def __init__(
        self,
        message: str,
        pattern: str | None = None,
        input_text: str | None = None,
    ):
        """
        Initialize the pattern match error.

        Args:
            message: Error message
            pattern: Pattern that caused the error
            input_text: Original input text
        """
        super().__init__(message, input_text)
        self.pattern = pattern


class LLMClassificationError(ActionClassificationError):
    """Error in LLM-based classification."""

    def __init__(
        self,
        message: str,
        llm_response: str | None = None,
        input_text: str | None = None,
    ):
        """
        Initialize the LLM classification error.

        Args:
            message: Error message
            llm_response: Raw LLM response that caused the error
            input_text: Original input text
        """
        super().__init__(message, input_text)
        self.llm_response = llm_response


class ClassificationTimeoutError(ActionClassificationError):
    """Error when classification takes too long."""

    def __init__(
        self,
        message: str,
        timeout_seconds: float,
        input_text: str | None = None,
    ):
        """
        Initialize the timeout error.

        Args:
            message: Error message
            timeout_seconds: Timeout threshold that was exceeded
            input_text: Original input text
        """
        super().__init__(message, input_text)
        self.timeout_seconds = timeout_seconds


class InvalidConfigurationError(ActionClassificationError):
    """Error in action classification configuration."""

    def __init__(self, message: str, config_key: str | None = None):
        """
        Initialize the configuration error.

        Args:
            message: Error message
            config_key: Configuration key that is invalid
        """
        super().__init__(message)
        self.config_key = config_key


class CacheError(ActionClassificationError):
    """Error in classification cache operations."""

    def __init__(
        self,
        message: str,
        cache_key: str | None = None,
        input_text: str | None = None,
    ):
        """
        Initialize the cache error.

        Args:
            message: Error message
            cache_key: Cache key that caused the error
            input_text: Original input text
        """
        super().__init__(message, input_text)
        self.cache_key = cache_key
