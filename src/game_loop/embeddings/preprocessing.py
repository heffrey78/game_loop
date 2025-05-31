"""
Text preprocessing functions for embedding generation.
"""

import logging
import re
from typing import Any

from .exceptions import EmbeddingPreprocessingError

logger = logging.getLogger(__name__)


def preprocess_for_embedding(
    text: str, entity_type: str = "general", max_length: int = 512
) -> str:
    """
    Preprocess text for optimal embedding generation.

    Args:
        text: The input text to preprocess
        entity_type: Type of entity for specialized preprocessing
        max_length: Maximum text length after preprocessing

    Returns:
        Preprocessed text ready for embedding generation

    Raises:
        EmbeddingPreprocessingError: If preprocessing fails
    """
    try:
        if not text or not text.strip():
            return ""

        # Start with basic text cleaning
        cleaned_text = clean_text(text)

        # Apply entity-specific preprocessing
        enriched_text = enrich_context(cleaned_text, entity_type)

        # Normalize the text
        normalized_text = normalize_text(enriched_text)

        # Truncate if necessary, but try to keep complete sentences
        if len(normalized_text) > max_length:
            normalized_text = _smart_truncate(normalized_text, max_length)

        return normalized_text.strip()

    except Exception as e:
        logger.error(f"Failed to preprocess text for {entity_type}: {e}")
        raise EmbeddingPreprocessingError(f"Preprocessing failed: {e}") from e


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent embedding generation.

    Args:
        text: Input text to normalize

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Convert to lowercase for consistency
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove excessive punctuation
    text = re.sub(r"[.]{2,}", ".", text)
    text = re.sub(r"[!]{2,}", "!", text)
    text = re.sub(r"[?]{2,}", "?", text)

    # Ensure proper sentence endings
    text = re.sub(r"([.!?])\s*([A-Za-z])", r"\1 \2", text)

    return text.strip()


def enrich_context(
    text: str, entity_type: str, additional_context: dict[str, Any] | None = None
) -> str:
    """
    Enrich text with context based on entity type for better embeddings.

    Args:
        text: Input text to enrich
        entity_type: Type of entity for context enrichment
        additional_context: Optional additional context information

    Returns:
        Context-enriched text
    """
    if not text:
        return ""

    additional_context = additional_context or {}

    # Entity-specific context enrichment
    if entity_type == "location":
        return _enrich_location_context(text, additional_context)
    elif entity_type == "npc":
        return _enrich_npc_context(text, additional_context)
    elif entity_type == "object":
        return _enrich_object_context(text, additional_context)
    elif entity_type == "quest":
        return _enrich_quest_context(text, additional_context)
    elif entity_type == "knowledge":
        return _enrich_knowledge_context(text, additional_context)
    elif entity_type == "rule":
        return _enrich_rule_context(text, additional_context)
    else:
        # General context enrichment
        return _enrich_general_context(text, additional_context)


def chunk_text(text: str, max_length: int = 512, overlap: int = 50) -> list[str]:
    """
    Split long text into chunks for embedding generation.

    Args:
        text: Input text to chunk
        max_length: Maximum length per chunk
        overlap: Number of characters to overlap between chunks

    Returns:
        List of text chunks
    """
    if not text or len(text) <= max_length:
        return [text] if text else []

    chunks = []
    sentences = _split_into_sentences(text)
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence would exceed max_length, start a new chunk
        if current_chunk and len(current_chunk) + len(sentence) + 1 > max_length:
            chunks.append(current_chunk.strip())

            # Start new chunk with overlap if possible
            if overlap > 0:
                words = current_chunk.split()
                overlap_words: list[str] = []
                overlap_length = 0

                for word in reversed(words):
                    if overlap_length + len(word) + 1 <= overlap:
                        overlap_words.insert(0, word)
                        overlap_length += len(word) + 1
                    else:
                        break

                current_chunk = " ".join(overlap_words)
                if current_chunk:
                    current_chunk += " "
            else:
                current_chunk = ""

        # Add sentence to current chunk
        if current_chunk:
            current_chunk += " " + sentence
        else:
            current_chunk = sentence

    # Add the last chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def clean_text(text: str) -> str:
    """
    Clean text by removing unwanted characters and formatting.

    Args:
        text: Input text to clean

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove control characters and non-printable characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Remove excessive HTML tags if present
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http[s]?://\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", "", text)

    # Remove excessive special characters
    text = re.sub(r"[^\w\s.,!?;:()\[\]\"'-]", " ", text)

    # Clean up whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def _smart_truncate(text: str, max_length: int) -> str:
    """
    Intelligently truncate text while preserving sentence boundaries.

    Args:
        text: Text to truncate
        max_length: Maximum allowed length

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    # Try to truncate at sentence boundary
    sentences = _split_into_sentences(text)
    truncated = ""

    for sentence in sentences:
        if len(truncated) + len(sentence) + 1 <= max_length:
            if truncated:
                truncated += " " + sentence
            else:
                truncated = sentence
        else:
            break

    # If no complete sentences fit, truncate at word boundary
    if not truncated:
        words = text.split()
        for word in words:
            if len(truncated) + len(word) + 1 <= max_length:
                if truncated:
                    truncated += " " + word
                else:
                    truncated = word
            else:
                break

    # If no complete words fit, just truncate
    if not truncated:
        truncated = text[:max_length].rstrip()

    return truncated


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Simple sentence splitting - could be enhanced with more sophisticated logic
    sentences = re.split(r"[.!?]+\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _enrich_location_context(text: str, context: dict[str, Any]) -> str:
    """Enrich location text with location-specific context."""
    enriched = f"Location: {text}"

    if context.get("location_type"):
        enriched += f" This is a {context['location_type']} type location."

    if context.get("region"):
        enriched += f" It is located in the {context['region']} region."

    return enriched


def _enrich_npc_context(text: str, context: dict[str, Any]) -> str:
    """Enrich NPC text with character-specific context."""
    enriched = f"Character: {text}"

    if context.get("race"):
        enriched += f" This character is a {context['race']}."

    if context.get("profession"):
        enriched += f" They work as a {context['profession']}."

    return enriched


def _enrich_object_context(text: str, context: dict[str, Any]) -> str:
    """Enrich object text with object-specific context."""
    enriched = f"Object: {text}"

    if context.get("object_type"):
        enriched += f" This is a {context['object_type']} type item."

    if context.get("rarity"):
        enriched += f" It has {context['rarity']} rarity."

    return enriched


def _enrich_quest_context(text: str, context: dict[str, Any]) -> str:
    """Enrich quest text with quest-specific context."""
    enriched = f"Quest: {text}"

    if context.get("quest_type"):
        enriched += f" This is a {context['quest_type']} quest."

    if context.get("difficulty"):
        enriched += f" It has {context['difficulty']} difficulty level."

    return enriched


def _enrich_knowledge_context(text: str, context: dict[str, Any]) -> str:
    """Enrich knowledge text with knowledge-specific context."""
    enriched = f"Knowledge: {text}"

    if context.get("category"):
        enriched += f" This knowledge is about {context['category']}."

    return enriched


def _enrich_rule_context(text: str, context: dict[str, Any]) -> str:
    """Enrich rule text with rule-specific context."""
    enriched = f"Game Rule: {text}"

    if context.get("rule_type"):
        enriched += f" This is a {context['rule_type']} type rule."

    return enriched


def _enrich_general_context(text: str, context: dict[str, Any]) -> str:
    """Enrich general text with available context."""
    enriched = text

    if context.get("category"):
        enriched = f"{context['category']}: {enriched}"

    if context.get("description"):
        enriched += f" {context['description']}"

    return enriched
