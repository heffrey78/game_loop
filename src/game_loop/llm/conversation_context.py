"""
Conversation Context Manager for Game Loop.
Tracks conversational context for more natural dialogue interactions.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class ConversationExchange:
    """Represents a single exchange in a conversation."""

    user_input: str
    system_response: str
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


class ConversationContext:
    """
    Tracks conversational context for more natural dialogue interactions.
    Maintains a history of recent exchanges to provide context for LLM.
    """

    def __init__(self, max_history: int = 5):
        """
        Initialize conversation context with history limit.

        Args:
            max_history: Maximum number of conversation exchanges to keep
        """
        self.max_history = max_history
        self.history: deque[ConversationExchange] = deque(maxlen=max_history)
        self.global_context: dict[str, Any] = {}

    def add_exchange(
        self,
        user_input: str,
        system_response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Add a conversation exchange to history.

        Args:
            user_input: User's input text
            system_response: System's response text
            metadata: Optional metadata associated with this exchange
        """
        exchange = ConversationExchange(
            user_input=user_input,
            system_response=system_response,
            metadata=metadata or {},
        )
        self.history.append(exchange)

    def get_recent_context(self, max_tokens: int | None = None) -> str:
        """
        Get recent conversation history formatted for context.

        Args:
            max_tokens: Maximum approximate token count for context

        Returns:
            Formatted conversation history string
        """
        if not self.history:
            return "No conversation history yet."

        # Start with a simple format without token counting
        # Token counting would be a more complex implementation
        context_parts = ["Recent conversation:"]

        # Add each exchange to the context
        for i, exchange in enumerate(self.history):
            context_parts.append(f"User: {exchange.user_input}")
            context_parts.append(f"System: {exchange.system_response}")

            # Add a separator between exchanges
            if i < len(self.history) - 1:
                context_parts.append("---")

        return "\n".join(context_parts)

    def update_global_context(self, key: str, value: Any) -> None:
        """
        Update the global conversation context.

        Args:
            key: Context key
            value: Context value
        """
        self.global_context[key] = value

    def get_global_context(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the global conversation context.

        Args:
            key: Context key
            default: Default value if key not found

        Returns:
            Context value or default
        """
        return self.global_context.get(key, default)

    def clear_context(self) -> None:
        """
        Reset conversation context.
        Clears both history and global context.
        """
        self.history.clear()
        self.global_context.clear()
