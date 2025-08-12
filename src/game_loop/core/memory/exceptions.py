"""Custom exceptions for the emotional memory system."""

from typing import Any, Dict, Optional


class EmotionalMemoryError(Exception):
    """Base exception for the emotional memory system."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        self.message = message
    
    def __str__(self):
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class EmotionalAnalysisError(EmotionalMemoryError):
    """Error during emotional significance analysis."""
    pass


class InvalidEmotionalDataError(EmotionalMemoryError):
    """Error when emotional data is invalid or corrupted."""
    pass


class MemoryAccessError(EmotionalMemoryError):
    """Error accessing protected or restricted memories."""
    
    def __init__(self, message: str, memory_id: Optional[str] = None, 
                 protection_level: Optional[str] = None, required_trust: Optional[float] = None):
        context = {}
        if memory_id:
            context['memory_id'] = memory_id
        if protection_level:
            context['protection_level'] = protection_level
        if required_trust is not None:
            context['required_trust'] = required_trust
        
        super().__init__(message, context)
        self.memory_id = memory_id
        self.protection_level = protection_level
        self.required_trust = required_trust


class TraumaProtectionError(EmotionalMemoryError):
    """Error in trauma protection system."""
    
    def __init__(self, message: str, npc_id: Optional[str] = None, 
                 trauma_type: Optional[str] = None, safety_violation: Optional[str] = None):
        context = {}
        if npc_id:
            context['npc_id'] = npc_id
        if trauma_type:
            context['trauma_type'] = trauma_type
        if safety_violation:
            context['safety_violation'] = safety_violation
        
        super().__init__(message, context)
        self.npc_id = npc_id
        self.trauma_type = trauma_type
        self.safety_violation = safety_violation


class MoodEngineError(EmotionalMemoryError):
    """Error in mood tracking and management."""
    
    def __init__(self, message: str, npc_id: Optional[str] = None, 
                 current_mood: Optional[str] = None, attempted_mood: Optional[str] = None):
        context = {}
        if npc_id:
            context['npc_id'] = npc_id
        if current_mood:
            context['current_mood'] = current_mood
        if attempted_mood:
            context['attempted_mood'] = attempted_mood
        
        super().__init__(message, context)


class MemoryClusteringError(EmotionalMemoryError):
    """Error during memory clustering operations."""
    
    def __init__(self, message: str, clustering_method: Optional[str] = None,
                 memory_count: Optional[int] = None, error_stage: Optional[str] = None):
        context = {}
        if clustering_method:
            context['clustering_method'] = clustering_method
        if memory_count is not None:
            context['memory_count'] = memory_count
        if error_stage:
            context['error_stage'] = error_stage
        
        super().__init__(message, context)


class DialogueIntegrationError(EmotionalMemoryError):
    """Error during dialogue system integration."""
    
    def __init__(self, message: str, npc_id: Optional[str] = None,
                 conversation_id: Optional[str] = None, integration_stage: Optional[str] = None):
        context = {}
        if npc_id:
            context['npc_id'] = npc_id
        if conversation_id:
            context['conversation_id'] = conversation_id
        if integration_stage:
            context['integration_stage'] = integration_stage
        
        super().__init__(message, context)


class MemoryPreservationError(EmotionalMemoryError):
    """Error during memory preservation to database."""
    
    def __init__(self, message: str, exchange_id: Optional[str] = None,
                 operation: Optional[str] = None, database_error: Optional[str] = None):
        context = {}
        if exchange_id:
            context['exchange_id'] = exchange_id
        if operation:
            context['operation'] = operation
        if database_error:
            context['database_error'] = database_error
        
        super().__init__(message, context)


class ValidationError(EmotionalMemoryError):
    """Error when input validation fails."""
    
    def __init__(self, message: str, field_name: Optional[str] = None,
                 expected_type: Optional[str] = None, actual_value: Optional[Any] = None):
        context = {}
        if field_name:
            context['field'] = field_name
        if expected_type:
            context['expected_type'] = expected_type
        if actual_value is not None:
            context['actual_value'] = str(actual_value)[:100]  # Truncate long values
        
        super().__init__(message, context)
        self.field_name = field_name
        self.expected_type = expected_type
        self.actual_value = actual_value


class ConfigurationError(EmotionalMemoryError):
    """Error in system configuration."""
    
    def __init__(self, message: str, config_section: Optional[str] = None,
                 config_key: Optional[str] = None, config_value: Optional[Any] = None):
        context = {}
        if config_section:
            context['config_section'] = config_section
        if config_key:
            context['config_key'] = config_key
        if config_value is not None:
            context['config_value'] = str(config_value)
        
        super().__init__(message, context)


class PerformanceError(EmotionalMemoryError):
    """Error when performance limits are exceeded."""
    
    def __init__(self, message: str, operation: Optional[str] = None,
                 limit_exceeded: Optional[str] = None, actual_value: Optional[float] = None,
                 limit_value: Optional[float] = None):
        context = {}
        if operation:
            context['operation'] = operation
        if limit_exceeded:
            context['limit_exceeded'] = limit_exceeded
        if actual_value is not None:
            context['actual_value'] = actual_value
        if limit_value is not None:
            context['limit_value'] = limit_value
        
        super().__init__(message, context)


class SecurityError(EmotionalMemoryError):
    """Security-related error in the emotional memory system."""
    
    def __init__(self, message: str, security_violation: Optional[str] = None,
                 user_id: Optional[str] = None, attempted_action: Optional[str] = None):
        context = {}
        if security_violation:
            context['security_violation'] = security_violation
        if user_id:
            context['user_id'] = user_id
        if attempted_action:
            context['attempted_action'] = attempted_action
        
        super().__init__(message, context)
        
        # Don't log sensitive information
        self.user_id = user_id
        self.security_violation = security_violation


class CacheError(EmotionalMemoryError):
    """Error in cache operations."""
    
    def __init__(self, message: str, cache_name: Optional[str] = None,
                 cache_operation: Optional[str] = None, cache_size: Optional[int] = None):
        context = {}
        if cache_name:
            context['cache_name'] = cache_name
        if cache_operation:
            context['cache_operation'] = cache_operation
        if cache_size is not None:
            context['cache_size'] = cache_size
        
        super().__init__(message, context)


class TimeoutError(EmotionalMemoryError):
    """Error when operations timeout."""
    
    def __init__(self, message: str, operation: Optional[str] = None,
                 timeout_seconds: Optional[float] = None, elapsed_seconds: Optional[float] = None):
        context = {}
        if operation:
            context['operation'] = operation
        if timeout_seconds is not None:
            context['timeout_seconds'] = timeout_seconds
        if elapsed_seconds is not None:
            context['elapsed_seconds'] = elapsed_seconds
        
        super().__init__(message, context)


# Exception handling utilities
def handle_emotional_memory_error(error: Exception, default_message: str = "An error occurred") -> EmotionalMemoryError:
    """Convert generic exceptions to emotional memory system exceptions."""
    if isinstance(error, EmotionalMemoryError):
        return error
    
    # Map common exceptions to specific emotional memory exceptions
    if isinstance(error, ValueError):
        return ValidationError(str(error))
    elif isinstance(error, TypeError):
        return ValidationError(f"Type error: {error}")
    elif isinstance(error, KeyError):
        return InvalidEmotionalDataError(f"Missing required key: {error}")
    elif isinstance(error, AttributeError):
        return InvalidEmotionalDataError(f"Missing attribute: {error}")
    elif isinstance(error, ConnectionError):
        return MemoryPreservationError(f"Database connection error: {error}")
    elif isinstance(error, TimeoutError):
        return TimeoutError(f"Operation timed out: {error}")
    else:
        return EmotionalMemoryError(f"{default_message}: {error}")


def is_retriable_error(error: EmotionalMemoryError) -> bool:
    """Check if an error is retriable (temporary vs permanent)."""
    retriable_errors = {
        TimeoutError,
        MemoryPreservationError,  # Database errors might be temporary
        CacheError,              # Cache errors are usually temporary
        PerformanceError,        # Performance issues might be temporary
    }
    
    return type(error) in retriable_errors


def should_alert_admin(error: EmotionalMemoryError) -> bool:
    """Check if an error requires admin notification."""
    critical_errors = {
        SecurityError,           # Security violations always need attention
        TraumaProtectionError,   # Trauma protection failures are critical
        ConfigurationError,      # Configuration errors need fixing
    }
    
    return type(error) in critical_errors or (
        hasattr(error, 'safety_violation') and error.safety_violation
    )