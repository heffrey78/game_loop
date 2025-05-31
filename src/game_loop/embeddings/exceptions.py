"""
Exception classes for the embedding service.
"""

import asyncio
import logging
from collections.abc import Callable, Coroutine
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class EmbeddingError(Exception):
    """Base exception for embedding-related errors."""

    pass


class EmbeddingGenerationError(EmbeddingError):
    """Exception raised when embedding generation fails."""

    pass


class EmbeddingCacheError(EmbeddingError):
    """Exception raised when cache operations fail."""

    pass


class EmbeddingConfigError(EmbeddingError):
    """Exception raised when configuration is invalid."""

    pass


class EmbeddingPreprocessingError(EmbeddingError):
    """Exception raised when text preprocessing fails."""

    pass


def with_retry(
    max_retries: int = 3, delay: float = 1.0, exponential_backoff: bool = True
) -> Callable[
    [Callable[..., Coroutine[Any, Any, T]]],
    Callable[..., Coroutine[Any, Any, T]],
]:
    """
    Decorator to add retry logic to async functions.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        exponential_backoff: Whether to use exponential backoff

    Returns:
        Decorated function with retry logic
    """

    def decorator(
        func: Callable[..., Coroutine[Any, Any, T]],
    ) -> Callable[..., Coroutine[Any, Any, T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"Function {func.__name__} failed after "
                            f"{max_retries} retries: {e}"
                        )
                        raise EmbeddingGenerationError(
                            f"Failed after {max_retries} retries: {e}"
                        ) from e

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed for "
                        f"{func.__name__}: {e}. Retrying in {current_delay}s..."
                    )

                    await asyncio.sleep(current_delay)

                    if exponential_backoff:
                        current_delay *= 2

            # This should never be reached, but just in case
            raise EmbeddingGenerationError(
                f"Unexpected failure in retry logic: {last_exception}"
            )

        return wrapper

    return decorator


async def with_retry_async(
    func: Callable[..., Coroutine[Any, Any, T]],
    max_retries: int = 3,
    delay: float = 1.0,
    *args: Any,
    **kwargs: Any,
) -> T:
    """
    Async function to add retry logic to function calls.

    Args:
        func: The async function to retry
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Result of the function call

    Raises:
        EmbeddingGenerationError: If all retries fail
    """
    last_exception = None
    current_delay = delay

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            if attempt == max_retries:
                logger.error(
                    f"Function {func.__name__} failed after "
                    f"{max_retries} retries: {e}"
                )
                raise EmbeddingGenerationError(
                    f"Failed after {max_retries} retries: {e}"
                ) from e

            logger.warning(
                f"Attempt {attempt + 1}/{max_retries + 1} failed for "
                f"{func.__name__}: {e}. Retrying in {current_delay}s..."
            )

            await asyncio.sleep(current_delay)
            current_delay *= 2

    # This should never be reached, but just in case
    raise EmbeddingGenerationError(
        f"Unexpected failure in retry logic: {last_exception}"
    )
