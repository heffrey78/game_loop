"""Factory for conversation flow components to avoid circular dependencies."""

from .flow_templates import ConversationFlowLibrary


def create_conversation_flow_library() -> ConversationFlowLibrary:
    """
    Create a ConversationFlowLibrary instance.

    This factory function replaces the global instance pattern to avoid
    circular import dependencies and improve testability.

    Returns:
        ConversationFlowLibrary: A configured flow library instance
    """
    return ConversationFlowLibrary()


def create_default_flow_library() -> ConversationFlowLibrary:
    """
    Create the default ConversationFlowLibrary instance.

    This is used as the default when no specific flow library is provided
    to ConversationFlowManager.

    Returns:
        ConversationFlowLibrary: The default flow library instance
    """
    return create_conversation_flow_library()
