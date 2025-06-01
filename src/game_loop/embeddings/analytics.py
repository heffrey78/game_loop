"""
Analytics and visualization tools for entity embeddings.

This module provides functions for analyzing and visualizing entity embeddings
including dimensionality reduction, clustering, and statistics generation.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .entity_registry import EntityEmbeddingRegistry

logger = logging.getLogger(__name__)


def compute_embedding_stats(embeddings: list[list[float]]) -> dict:
    """
    Compute statistics for a collection of embeddings.

    Args:
        embeddings: List of embedding vectors

    Returns:
        Dictionary of statistics
    """
    if not embeddings:
        return {
            "count": 0,
            "dimension": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "norm_mean": None,
        }

    embeddings_np = np.array(embeddings)

    # Calculate norms of each embedding
    norms = np.linalg.norm(embeddings_np, axis=1)

    # Calculate statistics
    stats = {
        "count": len(embeddings),
        "dimension": embeddings_np.shape[1],
        "mean": embeddings_np.mean(axis=0).tolist(),
        "std": embeddings_np.std(axis=0).tolist(),
        "min": embeddings_np.min(axis=0).tolist(),
        "max": embeddings_np.max(axis=0).tolist(),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
    }

    return stats


def reduce_dimensions(
    embeddings: list[list[float]], method: str = "pca", dimensions: int = 2
) -> list[list[float]]:
    """
    Reduce embeddings to lower dimensions for visualization.

    Args:
        embeddings: List of embedding vectors
        method: Dimensionality reduction method ('pca', 'tsne', or 'umap')
        dimensions: Target number of dimensions

    Returns:
        List of reduced-dimension embeddings
    """
    # Convert embeddings to numpy array
    embeddings_np = np.array(embeddings)

    if len(embeddings) < 2:
        logger.warning("Not enough embeddings for dimensionality reduction")
        # Return original if not enough data
        return embeddings

    try:
        # Simple PCA implementation
        if method.lower() == "pca":
            # Center the data
            embeddings_centered = embeddings_np - np.mean(embeddings_np, axis=0)

            # Compute the covariance matrix
            cov_matrix = np.cov(embeddings_centered, rowvar=False)

            # Compute eigenvectors and eigenvalues
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

            # Sort eigenvectors by eigenvalues in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvectors = eigenvectors[:, idx]

            # Select top components and project data
            components = eigenvectors[:, :dimensions]
            reduced = np.dot(embeddings_centered, components)

            result: list[list[float]] = reduced.tolist()
            return result

        # For other methods, suggest installing sklearn
        else:
            logger.warning(
                f"Method {method} requires scikit-learn. Using simple PCA instead."
            )
            # Fall back to simple PCA
            pca_result: list[list[float]] = reduce_dimensions(
                embeddings, method="pca", dimensions=dimensions
            )
            return pca_result

    except Exception as e:
        logger.error(f"Dimensionality reduction failed: {e}")
        # Return original if reduction fails
        return embeddings


def cluster_embeddings(
    embeddings: list[list[float]], method: str = "kmeans", n_clusters: int = 5
) -> list[int]:
    """
    Cluster embeddings into groups.

    This is a simple implementation that uses euclidean distance
    for a basic k-means clustering approach.

    Args:
        embeddings: List of embedding vectors
        method: Clustering method ('kmeans' supported for now)
        n_clusters: Number of clusters

    Returns:
        List of cluster assignments for each embedding
    """
    if method.lower() != "kmeans":
        logger.warning(
            f"Clustering method {method} not supported. Using k-means instead."
        )

    embeddings_np = np.array(embeddings)

    if len(embeddings) < n_clusters:
        logger.warning(
            f"Not enough embeddings ({len(embeddings)}) for {n_clusters} clusters."
        )
        return [0] * len(embeddings)

    try:
        # Simple k-means implementation
        n_samples, n_features = embeddings_np.shape

        # Random initialization of centroids
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(n_samples, n_clusters, replace=False)
        centroids = embeddings_np[indices]

        # Cluster assignment array
        labels = np.zeros(n_samples, dtype=int)

        # Maximum iterations
        max_iterations = 100

        for _ in range(max_iterations):
            # Assign each point to nearest centroid
            distances = np.zeros((n_samples, n_clusters))
            for i in range(n_clusters):
                # Euclidean distance
                distances[:, i] = np.linalg.norm(embeddings_np - centroids[i], axis=1)

            new_labels = np.argmin(distances, axis=1)

            # Check for convergence
            if np.array_equal(labels, new_labels):
                break

            labels = new_labels

            # Update centroids
            for i in range(n_clusters):
                points_in_cluster = embeddings_np[labels == i]
                if len(points_in_cluster) > 0:
                    centroids[i] = points_in_cluster.mean(axis=0)

        labels_list: list[int] = labels.tolist()
        return labels_list

    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        default_labels: list[int] = [0] * len(embeddings)
        return default_labels


def calculate_distance_matrix(
    embeddings: list[list[float]], metric: str = "cosine"
) -> list[list[float]]:
    """
    Calculate a distance/similarity matrix between all embeddings.

    Args:
        embeddings: List of embedding vectors
        metric: Distance metric ('cosine', 'euclidean', or 'dot')

    Returns:
        Matrix of pairwise distances/similarities
    """
    embeddings_np = np.array(embeddings)
    n_samples = len(embeddings)
    matrix = np.zeros((n_samples, n_samples))

    # Choose metric function
    if metric.lower() == "cosine":
        # Normalize vectors
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        normalized = np.divide(
            embeddings_np, norms, out=np.zeros_like(embeddings_np), where=norms != 0
        )

        # Compute cosine similarity matrix
        matrix = np.dot(normalized, normalized.T)

    elif metric.lower() == "euclidean":
        # Compute pairwise euclidean distances
        for i in range(n_samples):
            for j in range(n_samples):
                matrix[i, j] = np.linalg.norm(embeddings_np[i] - embeddings_np[j])

    elif metric.lower() == "dot":
        # Compute dot product matrix
        matrix = np.dot(embeddings_np, embeddings_np.T)

    else:
        logger.warning(f"Unknown metric '{metric}', using cosine similarity")
        return calculate_distance_matrix(embeddings, metric="cosine")

    result: list[list[float]] = matrix.tolist()
    return result


async def generate_embedding_report(
    entity_registry: "EntityEmbeddingRegistry", output_path: Path
) -> None:
    """
    Generate a report on entity embeddings.

    Args:
        entity_registry: The entity embedding registry
        output_path: Path to save the report

    Raises:
        IOError: If writing report fails
    """
    try:
        # Collect all embeddings and metadata
        entities = entity_registry.get_all_entity_ids()
        embeddings = []
        entity_types = []
        entity_ids = []

        for entity_id in entities:
            embedding = await entity_registry.get_entity_embedding(entity_id)
            if embedding:
                embeddings.append(embedding)
                entity_types.append(
                    entity_registry.entity_types.get(entity_id, "unknown")
                )
                entity_ids.append(entity_id)

        # Generate statistics
        stats = compute_embedding_stats(embeddings)

        # Generate reduced embeddings for visualization
        reduced_embeddings = reduce_dimensions(embeddings, dimensions=2)

        # Assign clusters
        n_clusters = min(5, len(embeddings))
        clusters = (
            cluster_embeddings(embeddings, n_clusters=n_clusters) if embeddings else []
        )

        # Prepare report data
        report = {
            "statistics": stats,
            "entities": {
                "count": len(entity_ids),
                "type_distribution": {
                    t: entity_types.count(t) for t in set(entity_types)
                },
            },
            "visualization_data": [
                {
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "position": reduced_coords,
                    "cluster": cluster,
                }
                for entity_id, entity_type, reduced_coords, cluster in zip(
                    entity_ids, entity_types, reduced_embeddings, clusters, strict=False
                )
            ],
        }

        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Embedding report generated at {output_path}")

    except Exception as e:
        logger.error(f"Failed to generate embedding report: {e}")
        raise OSError(f"Failed to generate embedding report: {e}") from e


def visualize_embeddings(
    reduced_embeddings: list[list[float]],
    labels: list[str] | None = None,
    output_path: Path | None = None,
) -> None:
    """
    Visualize embeddings in a 2D plot.

    Args:
        reduced_embeddings: List of 2D embedding coordinates
        labels: Optional list of labels for each point
        output_path: Path to save the visualization

    Note:
        This function attempts to use matplotlib if available,
        otherwise it generates coordinates for external visualization.
    """
    if not reduced_embeddings:
        logger.warning("No embeddings to visualize")
        return

    # Ensure embeddings are 2D
    if len(reduced_embeddings[0]) != 2:
        logger.warning(
            f"Expected 2D embeddings, got {len(reduced_embeddings[0])}D. "
            "Using first two dimensions."
        )
        reduced_embeddings = [e[:2] for e in reduced_embeddings]

    # Try to use matplotlib if available
    try:
        import matplotlib.pyplot as plt

        # Extract x and y coordinates
        x = [e[0] for e in reduced_embeddings]
        y = [e[1] for e in reduced_embeddings]

        plt.figure(figsize=(10, 8))

        # Plot with labels if provided
        if labels:
            # Convert labels to categories if many unique values
            unique_labels = set(labels)
            if len(unique_labels) > 10:
                # Use scatter plot with hover labels
                plt.scatter(x, y, alpha=0.7)
                for i, label in enumerate(labels):
                    plt.annotate(
                        label,
                        (x[i], y[i]),
                        textcoords="offset points",
                        xytext=(0, 5),
                        ha="center",
                    )
            else:
                # Use different colors for categories
                for unique_label in unique_labels:
                    indices = [
                        i for i, label in enumerate(labels) if label == unique_label
                    ]
                    plt.scatter(
                        [x[i] for i in indices],
                        [y[i] for i in indices],
                        alpha=0.7,
                        label=unique_label,
                    )
                plt.legend()
        else:
            # Simple scatter plot
            plt.scatter(x, y, alpha=0.7)

        plt.title("Entity Embedding Visualization")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True, linestyle="--", alpha=0.7)

        # Save or show
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path)
            logger.info(f"Visualization saved to {output_path}")
        else:
            plt.show()

    except ImportError:
        logger.warning(
            "Matplotlib not available. Generating visualization data instead."
        )
        # Generate data for external visualization
        viz_data = {
            "points": reduced_embeddings,
            "labels": (
                labels
                if labels
                else ["Point " + str(i) for i in range(len(reduced_embeddings))]
            ),
        }

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(viz_data, f, indent=2)
            logger.info(f"Visualization data saved to {output_path}")
        else:
            logger.info("Visualization data generated but not saved (no output path)")

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
