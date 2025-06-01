"""
Similarity search functionality for entity embeddings.

This module provides functions for calculating similarity between embeddings
and performing efficient searches across entity embeddings.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (between -1 and 1)
    """
    # Convert to numpy arrays for efficient computation
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    # Calculate norms
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Compute cosine similarity
    return float(np.dot(v1, v2) / (norm1 * norm2))


def euclidean_distance(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate Euclidean distance between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Euclidean distance (lower is more similar)
    """
    # Convert to numpy arrays for efficient computation
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    # Compute Euclidean distance
    return float(np.linalg.norm(v1 - v2))


def dot_product(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate dot product between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Dot product (higher is more similar for normalized vectors)
    """
    # Convert to numpy arrays for efficient computation
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    # Compute dot product
    return float(np.dot(v1, v2))


async def search_entities(
    query_embedding: list[float],
    entities_embeddings: dict[str, list[float]],
    top_k: int = 5,
    metric: str = "cosine",
) -> list[tuple[str, float]]:
    """
    Search for similar entities given a query embedding.

    Args:
        query_embedding: The embedding to search with
        entities_embeddings: Dictionary mapping entity_ids to embeddings
        top_k: Number of results to return
        metric: Similarity metric to use ('cosine', 'euclidean', or 'dot')

    Returns:
        List of (entity_id, similarity_score) tuples
    """
    similarity_funcs = {
        "cosine": cosine_similarity,
        "euclidean": lambda v1, v2: -euclidean_distance(
            v1, v2
        ),  # Negate to sort descending
        "dot": dot_product,
    }

    if metric not in similarity_funcs:
        logger.warning(f"Unknown similarity metric '{metric}', falling back to cosine")
        metric = "cosine"

    similarity_func = similarity_funcs[metric]

    # Calculate similarity for each entity
    similarities = [
        (entity_id, similarity_func(query_embedding, embedding))
        for entity_id, embedding in entities_embeddings.items()
    ]

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]


def create_index_from_embeddings(
    embeddings: dict[str, list[float]], index_type: str = "flat"
) -> Any:
    """
    Create a search index from entity embeddings.

    This is a simple implementation that returns a dictionary with the data
    needed for search. In a production system, this would use a specialized
    vector index like FAISS, Annoy, or similar libraries.

    Args:
        embeddings: Dictionary mapping entity_ids to embeddings
        index_type: Type of index to create

    Returns:
        Dictionary with embeddings and entity_ids for searching
    """
    if index_type != "flat":
        logger.warning(
            f"Index type '{index_type}' not supported, falling back to 'flat'"
        )

    # Simple implementation - for production, use FAISS, Annoy, etc.
    entity_ids = list(embeddings.keys())
    embedding_list = [embeddings[eid] for eid in entity_ids]

    return {"entity_ids": entity_ids, "embeddings": embedding_list, "type": "flat"}


async def query_index(
    index: dict[str, Any], query_embedding: list[float], top_k: int = 5
) -> list[tuple[str, float]]:
    """
    Query the search index with an embedding.

    Args:
        index: The search index created by create_index_from_embeddings
        query_embedding: The embedding to search with
        top_k: Number of results to return

    Returns:
        List of (entity_id, similarity_score) tuples
    """
    if index["type"] != "flat":
        logger.warning(
            f"Unknown index type '{index['type']}', results may be incorrect"
        )

    entity_ids = index["entity_ids"]
    embeddings = index["embeddings"]

    # Convert query to numpy for efficiency
    query_np = np.array(query_embedding)

    # Calculate cosine similarity for each embedding
    similarities = []
    for i, embedding in enumerate(embeddings):
        embedding_np = np.array(embedding)

        # Avoid division by zero
        norm1 = np.linalg.norm(query_np)
        norm2 = np.linalg.norm(embedding_np)
        if norm1 == 0 or norm2 == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(query_np, embedding_np) / (norm1 * norm2))

        # Ensure entity_id is a string and similarity is a float
        similarities.append((str(entity_ids[i]), float(similarity)))

    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]


def boost_by_context(
    similarities: list[tuple[str, float]], context_factors: dict[str, float]
) -> list[tuple[str, float]]:
    """
    Boost similarity scores based on contextual factors.

    Args:
        similarities: List of (entity_id, similarity_score) tuples
        context_factors: Dictionary mapping entity_id to boost factor

    Returns:
        List of boosted (entity_id, similarity_score) tuples
    """
    boosted = []
    for entity_id, score in similarities:
        # Apply boost if available, otherwise keep original score
        boost = context_factors.get(entity_id, 1.0)
        boosted.append((entity_id, score * boost))

    # Re-sort after boosting
    boosted.sort(key=lambda x: x[1], reverse=True)

    return boosted
