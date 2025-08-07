"""K-means clustering engine for grouping related emotional memories."""

import uuid
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timezone

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .config import MemoryAlgorithmConfig


@dataclass
class MemoryClusterData:
    """Data structure for memory clustering input."""
    
    memory_id: uuid.UUID
    embedding: np.ndarray
    emotional_weight: float
    confidence_score: float
    age_days: float
    content_preview: str  # First 50 chars for debugging


@dataclass 
class ClusterResult:
    """Result of memory clustering operation."""
    
    cluster_assignments: Dict[uuid.UUID, int]  # memory_id -> cluster_id
    cluster_centers: np.ndarray  # Cluster centroid embeddings
    cluster_confidence: Dict[int, float]  # cluster_id -> average confidence
    cluster_sizes: Dict[int, int]  # cluster_id -> number of memories
    total_processing_time_ms: float
    convergence_achieved: bool
    silhouette_score: Optional[float] = None


class MemoryClusteringEngine:
    """
    K-means clustering engine for grouping semantically and emotionally related memories.
    
    Groups memories by:
    - Semantic similarity (embedding cosine similarity)
    - Emotional weight significance  
    - Temporal proximity
    - Memory confidence levels
    """
    
    def __init__(self, config: MemoryAlgorithmConfig):
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for memory clustering. "
                "Install with: pip install scikit-learn"
            )
        
        self.config = config
        self._clustering_stats = {
            'total_clusterings': 0,
            'successful_clusterings': 0,
            'avg_processing_time_ms': 0.0,
            'avg_silhouette_score': 0.0,
        }
    
    def cluster_memories(
        self,
        memories: List[MemoryClusterData],
        target_clusters: Optional[int] = None,
        include_low_confidence: bool = False,
    ) -> ClusterResult:
        """
        Cluster memories using K-means algorithm with cosine similarity.
        
        Args:
            memories: List of memory data to cluster
            target_clusters: Number of clusters (auto-determined if None)
            include_low_confidence: Include memories below confidence threshold
            
        Returns:
            ClusterResult with cluster assignments and metadata
        """
        start_time = time.perf_counter()
        self._clustering_stats['total_clusterings'] += 1
        
        try:
            # Filter and prepare memories for clustering
            filtered_memories = self._filter_memories_for_clustering(
                memories, include_low_confidence
            )
            
            if len(filtered_memories) < 2:
                return self._create_single_cluster_result(filtered_memories, start_time)
            
            # Determine optimal number of clusters
            if target_clusters is None:
                target_clusters = self._determine_optimal_clusters(filtered_memories)
            
            target_clusters = max(1, min(target_clusters, len(filtered_memories)))
            
            # Prepare feature matrix
            feature_matrix = self._prepare_feature_matrix(filtered_memories)
            
            # Perform K-means clustering
            cluster_result = self._perform_kmeans_clustering(
                filtered_memories, feature_matrix, target_clusters
            )
            
            # Calculate performance metrics
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            cluster_result.total_processing_time_ms = processing_time_ms
            
            # Update performance statistics
            self._update_clustering_stats(cluster_result)
            
            return cluster_result
            
        except Exception as e:
            print(f"Clustering failed: {e}")
            return self._create_fallback_cluster_result(memories, start_time)
    
    def _filter_memories_for_clustering(
        self, memories: List[MemoryClusterData], include_low_confidence: bool
    ) -> List[MemoryClusterData]:
        """Filter memories suitable for clustering."""
        
        filtered = []
        
        for memory in memories:
            # Skip memories that are too old
            if memory.age_days > self.config.max_memory_age_days:
                continue
            
            # Skip low-confidence memories unless explicitly included
            if (not include_low_confidence and 
                memory.confidence_score < self.config.min_confidence_for_clustering):
                continue
            
            # Skip emotionally neutral memories (unless they have high confidence)
            if (memory.emotional_weight < self.config.min_emotional_weight and
                memory.confidence_score < 0.7):
                continue
            
            # Ensure embedding is valid
            if memory.embedding is None or len(memory.embedding) == 0:
                continue
                
            filtered.append(memory)
        
        return filtered
    
    def _determine_optimal_clusters(self, memories: List[MemoryClusterData]) -> int:
        """Determine optimal number of clusters using heuristics."""
        
        num_memories = len(memories)
        
        # Use square root heuristic as baseline
        baseline_clusters = max(2, int(np.sqrt(num_memories)))
        
        # Adjust based on configuration
        optimal_clusters = min(baseline_clusters, self.config.max_clusters)
        
        # Ensure minimum cluster size
        min_clusters_for_size = max(1, num_memories // self.config.min_cluster_size)
        optimal_clusters = min(optimal_clusters, min_clusters_for_size)
        
        return max(2, optimal_clusters)  # Always at least 2 clusters for meaningful grouping
    
    def _prepare_feature_matrix(self, memories: List[MemoryClusterData]) -> np.ndarray:
        """
        Prepare feature matrix for clustering.
        
        Combines semantic embeddings with emotional and temporal features.
        """
        num_memories = len(memories)
        embedding_dim = len(memories[0].embedding)
        
        # Create feature matrix: [embeddings, emotional_weight, confidence, recency]
        feature_matrix = np.zeros((num_memories, embedding_dim + 3))
        
        for i, memory in enumerate(memories):
            # Semantic embedding (normalized)
            embedding = memory.embedding / (np.linalg.norm(memory.embedding) + 1e-8)
            feature_matrix[i, :embedding_dim] = embedding
            
            # Emotional weight feature
            feature_matrix[i, embedding_dim] = memory.emotional_weight
            
            # Confidence feature
            feature_matrix[i, embedding_dim + 1] = memory.confidence_score
            
            # Recency feature (inverse of age, normalized)
            max_age = max(m.age_days for m in memories)
            recency = (max_age - memory.age_days) / (max_age + 1e-8)
            feature_matrix[i, embedding_dim + 2] = recency
        
        # Standardize non-embedding features to give them appropriate weight
        scaler = StandardScaler()
        feature_matrix[:, embedding_dim:] = scaler.fit_transform(feature_matrix[:, embedding_dim:])
        
        return feature_matrix
    
    def _perform_kmeans_clustering(
        self,
        memories: List[MemoryClusterData],
        feature_matrix: np.ndarray,
        n_clusters: int,
    ) -> ClusterResult:
        """Perform K-means clustering on the feature matrix."""
        
        # Configure K-means
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,  # For reproducible results
            max_iter=self.config.max_iterations,
            tol=self.config.convergence_threshold,
            n_init=10,  # Multiple initializations for stability
        )
        
        # Fit the model
        cluster_labels = kmeans.fit_predict(feature_matrix)
        
        # Create cluster assignments
        cluster_assignments = {}
        for i, memory in enumerate(memories):
            cluster_assignments[memory.memory_id] = int(cluster_labels[i])
        
        # Calculate cluster metadata
        cluster_confidence = self._calculate_cluster_confidence(memories, cluster_labels)
        cluster_sizes = self._calculate_cluster_sizes(cluster_labels)
        
        # Calculate silhouette score for clustering quality
        silhouette_score = None
        if len(set(cluster_labels)) > 1:  # Need at least 2 clusters
            try:
                from sklearn.metrics import silhouette_score as calc_silhouette
                silhouette_score = calc_silhouette(feature_matrix, cluster_labels)
            except ImportError:
                pass  # Silhouette score optional
        
        return ClusterResult(
            cluster_assignments=cluster_assignments,
            cluster_centers=kmeans.cluster_centers_,
            cluster_confidence=cluster_confidence,
            cluster_sizes=cluster_sizes,
            total_processing_time_ms=0.0,  # Will be set by caller
            convergence_achieved=kmeans.n_iter_ < self.config.max_iterations,
            silhouette_score=silhouette_score,
        )
    
    def _calculate_cluster_confidence(
        self, memories: List[MemoryClusterData], cluster_labels: np.ndarray
    ) -> Dict[int, float]:
        """Calculate average confidence score for each cluster."""
        
        cluster_confidence = {}
        cluster_confidences = {}
        
        # Group memories by cluster
        for i, memory in enumerate(memories):
            cluster_id = int(cluster_labels[i])
            if cluster_id not in cluster_confidences:
                cluster_confidences[cluster_id] = []
            cluster_confidences[cluster_id].append(memory.confidence_score)
        
        # Calculate average confidence per cluster
        for cluster_id, confidences in cluster_confidences.items():
            cluster_confidence[cluster_id] = sum(confidences) / len(confidences)
        
        return cluster_confidence
    
    def _calculate_cluster_sizes(self, cluster_labels: np.ndarray) -> Dict[int, int]:
        """Calculate size of each cluster."""
        
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        return dict(zip(unique_labels.astype(int), counts.astype(int)))
    
    def find_similar_clusters(
        self,
        query_embedding: np.ndarray,
        cluster_centers: np.ndarray,
        similarity_threshold: float = None,
    ) -> List[Tuple[int, float]]:
        """
        Find clusters similar to a query embedding.
        
        Args:
            query_embedding: Embedding to find similar clusters for
            cluster_centers: Cluster centroid embeddings
            similarity_threshold: Minimum similarity (uses config default if None)
            
        Returns:
            List of (cluster_id, similarity_score) tuples, sorted by similarity
        """
        if similarity_threshold is None:
            similarity_threshold = self.config.similarity_threshold
        
        # Normalize query embedding
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        query_norm = query_norm.reshape(1, -1)
        
        # Calculate cosine similarities to cluster centers
        # Only use embedding dimensions (exclude additional features)
        embedding_dim = len(query_embedding)
        center_embeddings = cluster_centers[:, :embedding_dim]
        
        similarities = cosine_similarity(query_norm, center_embeddings)[0]
        
        # Filter and sort by similarity
        similar_clusters = []
        for i, similarity in enumerate(similarities):
            if similarity >= similarity_threshold:
                similar_clusters.append((i, float(similarity)))
        
        similar_clusters.sort(key=lambda x: x[1], reverse=True)
        return similar_clusters
    
    def _create_single_cluster_result(
        self, memories: List[MemoryClusterData], start_time: float
    ) -> ClusterResult:
        """Create result for single-cluster case (too few memories)."""
        
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        if not memories:
            return ClusterResult(
                cluster_assignments={},
                cluster_centers=np.array([]),
                cluster_confidence={},
                cluster_sizes={},
                total_processing_time_ms=processing_time_ms,
                convergence_achieved=True,
            )
        
        # Assign all memories to cluster 0
        cluster_assignments = {memory.memory_id: 0 for memory in memories}
        
        # Calculate single cluster center
        embeddings = np.array([memory.embedding for memory in memories])
        cluster_center = np.mean(embeddings, axis=0).reshape(1, -1)
        
        # Calculate cluster confidence
        avg_confidence = sum(m.confidence_score for m in memories) / len(memories)
        
        return ClusterResult(
            cluster_assignments=cluster_assignments,
            cluster_centers=cluster_center,
            cluster_confidence={0: avg_confidence},
            cluster_sizes={0: len(memories)},
            total_processing_time_ms=processing_time_ms,
            convergence_achieved=True,
        )
    
    def _create_fallback_cluster_result(
        self, memories: List[MemoryClusterData], start_time: float
    ) -> ClusterResult:
        """Create fallback result when clustering fails."""
        
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Assign each memory to its own cluster (no clustering)
        cluster_assignments = {memory.memory_id: i for i, memory in enumerate(memories)}
        
        cluster_confidence = {i: memory.confidence_score for i, memory in enumerate(memories)}
        cluster_sizes = {i: 1 for i in range(len(memories))}
        
        # Create cluster centers from individual embeddings
        if memories:
            cluster_centers = np.array([memory.embedding for memory in memories])
        else:
            cluster_centers = np.array([])
        
        return ClusterResult(
            cluster_assignments=cluster_assignments,
            cluster_centers=cluster_centers,
            cluster_confidence=cluster_confidence,
            cluster_sizes=cluster_sizes,
            total_processing_time_ms=processing_time_ms,
            convergence_achieved=False,
        )
    
    def _update_clustering_stats(self, result: ClusterResult) -> None:
        """Update clustering performance statistics."""
        
        if result.convergence_achieved:
            self._clustering_stats['successful_clusterings'] += 1
        
        # Update average processing time
        total = self._clustering_stats['total_clusterings']
        current_avg = self._clustering_stats['avg_processing_time_ms']
        new_avg = ((current_avg * (total - 1)) + result.total_processing_time_ms) / total
        self._clustering_stats['avg_processing_time_ms'] = new_avg
        
        # Update average silhouette score
        if result.silhouette_score is not None:
            current_silhouette = self._clustering_stats['avg_silhouette_score']
            successful = self._clustering_stats['successful_clusterings']
            if successful > 1:
                new_silhouette = ((current_silhouette * (successful - 1)) + result.silhouette_score) / successful
            else:
                new_silhouette = result.silhouette_score
            self._clustering_stats['avg_silhouette_score'] = new_silhouette
    
    def get_clustering_stats(self) -> Dict[str, float]:
        """Get clustering performance statistics."""
        
        success_rate = 0.0
        if self._clustering_stats['total_clusterings'] > 0:
            success_rate = (
                self._clustering_stats['successful_clusterings'] / 
                self._clustering_stats['total_clusterings'] * 100
            )
        
        return {
            'total_clusterings': self._clustering_stats['total_clusterings'],
            'successful_clusterings': self._clustering_stats['successful_clusterings'],
            'success_rate_percent': round(success_rate, 1),
            'avg_processing_time_ms': round(self._clustering_stats['avg_processing_time_ms'], 2),
            'avg_silhouette_score': round(self._clustering_stats['avg_silhouette_score'], 3),
        }
    
    def reset_stats(self) -> None:
        """Reset clustering statistics."""
        self._clustering_stats = {
            'total_clusterings': 0,
            'successful_clusterings': 0,
            'avg_processing_time_ms': 0.0,
            'avg_silhouette_score': 0.0,
        }