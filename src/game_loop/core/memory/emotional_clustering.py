"""Emotional memory clustering and association networks for NPCs."""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity

from game_loop.core.conversation.conversation_models import NPCPersonality
from game_loop.database.session_factory import DatabaseSessionFactory

from .constants import EmotionalThresholds
from .emotional_context import EmotionalMemoryType, EmotionalSignificance, MoodState
from .emotional_preservation import EmotionalMemoryRecord
from .exceptions import (
    MemoryClusteringError, InvalidEmotionalDataError, PerformanceError,
    handle_emotional_memory_error
)
from .validation import (
    validate_uuid, validate_probability, validate_positive_number,
    validate_string_content, default_validator
)

logger = logging.getLogger(__name__)


class ClusteringMethod(Enum):
    """Methods for clustering emotional memories."""
    
    KMEANS = "kmeans"                    # K-means clustering by emotional features
    DBSCAN = "dbscan"                    # Density-based clustering  
    TEMPORAL = "temporal"                # Time-based clustering
    SEMANTIC = "semantic"                # Semantic similarity clustering
    EMOTIONAL_RESONANCE = "resonance"    # Emotional resonance-based clustering
    HYBRID = "hybrid"                    # Combination of multiple methods


class AssociationType(Enum):
    """Types of associations between emotional memories."""
    
    CAUSAL = "causal"                    # One memory caused/led to another
    THEMATIC = "thematic"                # Similar themes or content
    TEMPORAL = "temporal"                # Occurred close in time
    EMOTIONAL_ECHO = "emotional_echo"    # Similar emotional signature
    TRIGGER_RESPONSE = "trigger_response" # One memory triggers another
    CONTRASTING = "contrasting"          # Opposite emotional valence
    REINFORCING = "reinforcing"          # Strengthens emotional pattern


@dataclass
class EmotionalMemoryCluster:
    """A cluster of emotionally related memories."""
    
    cluster_id: str
    npc_id: str
    
    # Core cluster properties
    dominant_emotion_type: EmotionalMemoryType
    emotional_theme: str                   # Descriptive theme
    emotional_coherence: float             # 0.0-1.0 how emotionally consistent
    temporal_span: Tuple[float, float]     # (earliest, latest) timestamps
    
    # Cluster members
    member_memories: List[str] = field(default_factory=list)  # Memory exchange IDs
    core_memories: List[str] = field(default_factory=list)    # Most central memories
    peripheral_memories: List[str] = field(default_factory=list) # Less central
    
    # Cluster characteristics
    average_significance: float = 0.0
    average_intensity: float = 0.0
    protection_level_distribution: Dict[str, int] = field(default_factory=dict)
    mood_accessibility_profile: Dict[MoodState, float] = field(default_factory=dict)
    
    # Network properties
    internal_associations: List[Tuple[str, str, AssociationType]] = field(default_factory=list)
    external_connections: List[str] = field(default_factory=list)  # Connected cluster IDs
    triggering_strength: float = 0.0       # How strongly memories trigger each other
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    access_frequency: int = 0
    stability_score: float = 0.0           # How stable the cluster is over time


@dataclass
class MemoryAssociation:
    """Association between two emotional memories."""
    
    association_id: str
    source_memory_id: str
    target_memory_id: str
    association_type: AssociationType
    
    # Strength and properties
    strength: float                        # 0.0-1.0 association strength
    bidirectional: bool = False            # Whether association works both ways
    activation_threshold: float = EmotionalThresholds.MODERATE_SIGNIFICANCE      # Threshold for activation
    
    # Context
    formed_at: float = field(default_factory=time.time)
    reinforcement_count: int = 0           # How many times association was reinforced
    last_activated: Optional[float] = None
    
    # Decay properties
    decay_rate: float = 0.05               # How fast association weakens
    base_strength: float = 0.0             # Original strength before decay
    
    # Contextual factors
    mood_dependent: bool = False           # Whether association depends on mood
    trust_dependent: bool = False          # Whether association depends on trust level
    contextual_triggers: List[str] = field(default_factory=list)


@dataclass
class EmotionalNetwork:
    """Network of emotional memory clusters and associations for an NPC."""
    
    npc_id: str
    clusters: Dict[str, EmotionalMemoryCluster] = field(default_factory=dict)
    associations: Dict[str, MemoryAssociation] = field(default_factory=dict)
    
    # Network-level properties
    network_coherence: float = 0.0         # Overall emotional coherence
    dominant_patterns: List[str] = field(default_factory=list)  # Main emotional patterns
    vulnerability_points: List[str] = field(default_factory=list) # Emotionally sensitive areas
    
    # Access patterns
    frequent_pathways: List[Tuple[str, str]] = field(default_factory=list) # Common cluster->cluster paths
    entry_points: List[str] = field(default_factory=list)  # Commonly accessed clusters
    
    # Temporal dynamics
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    major_reorganizations: List[float] = field(default_factory=list)


class EmotionalMemoryClusteringEngine:
    """Engine for clustering emotional memories and building association networks."""
    
    def __init__(
        self,
        session_factory: DatabaseSessionFactory,
        clustering_method: ClusteringMethod = ClusteringMethod.HYBRID,
        min_cluster_size: int = EmotionalThresholds.MIN_CLUSTER_SIZE,
        max_clusters: int = EmotionalThresholds.MAX_CLUSTERS,
        association_threshold: float = EmotionalThresholds.ASSOCIATION_THRESHOLD,
    ):
        self.session_factory = session_factory
        self.clustering_method = clustering_method
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.association_threshold = association_threshold
        
        # Network storage
        self._emotional_networks: Dict[str, EmotionalNetwork] = {}
        
        # Caches
        self._clustering_cache: Dict[str, List[EmotionalMemoryCluster]] = {}
        self._association_cache: Dict[str, List[MemoryAssociation]] = {}
        
        # Performance tracking
        self._performance_stats = {
            "clustering_operations": 0,
            "association_formations": 0,
            "network_updates": 0,
            "cache_hits": 0,
            "avg_clustering_time_ms": 0.0,
        }

    async def cluster_emotional_memories(
        self,
        npc_id: uuid.UUID,
        memories: List[EmotionalMemoryRecord],
        personality: NPCPersonality,
        recalculate: bool = False,
    ) -> List[EmotionalMemoryCluster]:
        """Cluster emotional memories into thematically related groups."""
        try:
            # Validate inputs
            if not npc_id:
                raise InvalidEmotionalDataError("NPC ID is required")
            if not memories:
                raise InvalidEmotionalDataError("Memories list cannot be empty")
            if not personality:
                raise InvalidEmotionalDataError("NPC personality is required")
            
            # Validate NPC ID
            npc_id_str = str(validate_uuid(npc_id, "npc_id"))
            
            # Check memory limit for performance
            if len(memories) > EmotionalThresholds.MAX_CLUSTERING_MEMORIES:
                logger.warning(f"Too many memories for clustering: {len(memories)}. Using first {EmotionalThresholds.MAX_CLUSTERING_MEMORIES}")
                memories = memories[:EmotionalThresholds.MAX_CLUSTERING_MEMORIES]
                
        except Exception as e:
            raise handle_emotional_memory_error(
                e, "Failed to validate inputs for emotional memory clustering"
            )
        
        start_time = time.perf_counter()
        
        try:
            # Check cache first
            cache_key = f"{npc_id_str}_{len(memories)}"
            if not recalculate and cache_key in self._clustering_cache:
                self._performance_stats["cache_hits"] += 1
                return self._clustering_cache[cache_key]
            
            if len(memories) < self.min_cluster_size:
                logger.debug(f"Not enough memories to cluster for NPC {npc_id}: {len(memories)}")
                return []
            
            # Extract features for clustering
            features = self._extract_clustering_features(memories, personality)
        
            # Perform clustering based on method
            cluster_assignments = await self._perform_clustering(
                features, memories, self.clustering_method
            )
            
            # Build cluster objects
            clusters = await self._build_clusters_from_assignments(
                cluster_assignments, memories, npc_id_str
            )
            
            # Calculate cluster properties
            for cluster in clusters:
                await self._calculate_cluster_properties(cluster, memories, personality)
        
            # Cache results
            self._clustering_cache[cache_key] = clusters
            
            # Update performance stats
            processing_time_ms = (time.perf_counter() - start_time) * 1000
            self._performance_stats["clustering_operations"] += 1
            self._update_avg_clustering_time(processing_time_ms)
            
            logger.debug(f"Created {len(clusters)} clusters for NPC {npc_id}")
            return clusters
            
        except Exception as e:
            raise MemoryClusteringError(
                f"Failed to cluster emotional memories for NPC {npc_id}",
                clustering_method=str(self.clustering_method.value),
                memory_count=len(memories),
                error_stage="clustering"
            )

    async def build_memory_associations(
        self,
        npc_id: uuid.UUID,
        memories: List[EmotionalMemoryRecord],
        clusters: List[EmotionalMemoryCluster],
        personality: NPCPersonality,
    ) -> List[MemoryAssociation]:
        """Build associations between memories within and across clusters."""
        
        npc_id_str = str(npc_id)
        associations = []
        
        # Create memory lookup for easy access
        memory_lookup = {m.exchange_id: m for m in memories}
        
        # Build intra-cluster associations (within clusters)
        for cluster in clusters:
            cluster_associations = await self._build_intra_cluster_associations(
                cluster, memory_lookup, personality
            )
            associations.extend(cluster_associations)
        
        # Build inter-cluster associations (between clusters)
        inter_cluster_associations = await self._build_inter_cluster_associations(
            clusters, memory_lookup, personality
        )
        associations.extend(inter_cluster_associations)
        
        # Build semantic associations based on content similarity
        semantic_associations = await self._build_semantic_associations(
            memories, personality
        )
        associations.extend(semantic_associations)
        
        # Build temporal associations for memories close in time
        temporal_associations = self._build_temporal_associations(memories)
        associations.extend(temporal_associations)
        
        # Filter associations by strength threshold
        strong_associations = [
            assoc for assoc in associations
            if assoc.strength >= self.association_threshold
        ]
        
        # Cache associations
        self._association_cache[npc_id_str] = strong_associations
        
        self._performance_stats["association_formations"] += len(strong_associations)
        logger.debug(f"Created {len(strong_associations)} associations for NPC {npc_id}")
        
        return strong_associations

    async def update_emotional_network(
        self,
        npc_id: uuid.UUID,
        memories: List[EmotionalMemoryRecord],
        personality: NPCPersonality,
        incremental: bool = True,
    ) -> EmotionalNetwork:
        """Update or create the emotional network for an NPC."""
        
        npc_id_str = str(npc_id)
        
        # Get or create network
        if npc_id_str in self._emotional_networks and incremental:
            network = self._emotional_networks[npc_id_str]
        else:
            network = EmotionalNetwork(npc_id=npc_id_str)
        
        # Cluster memories
        clusters = await self.cluster_emotional_memories(npc_id, memories, personality)
        
        # Update network clusters
        for cluster in clusters:
            network.clusters[cluster.cluster_id] = cluster
        
        # Build associations
        associations = await self.build_memory_associations(npc_id, memories, clusters, personality)
        
        # Update network associations
        for association in associations:
            network.associations[association.association_id] = association
        
        # Calculate network-level properties
        await self._calculate_network_properties(network, personality)
        
        # Store updated network
        network.last_updated = time.time()
        self._emotional_networks[npc_id_str] = network
        
        self._performance_stats["network_updates"] += 1
        logger.debug(f"Updated emotional network for NPC {npc_id}")
        
        return network

    async def get_associated_memories(
        self,
        npc_id: uuid.UUID,
        trigger_memory_id: str,
        max_associations: int = 5,
        activation_threshold: float = 0.5,
    ) -> List[Tuple[str, MemoryAssociation]]:
        """Get memories associated with a trigger memory."""
        
        npc_id_str = str(npc_id)
        network = self._emotional_networks.get(npc_id_str)
        
        if not network:
            return []
        
        # Find associations involving the trigger memory
        relevant_associations = []
        for association in network.associations.values():
            if (association.source_memory_id == trigger_memory_id and 
                association.strength >= activation_threshold):
                relevant_associations.append((association.target_memory_id, association))
            elif (association.bidirectional and 
                  association.target_memory_id == trigger_memory_id and
                  association.strength >= activation_threshold):
                relevant_associations.append((association.source_memory_id, association))
        
        # Sort by association strength
        relevant_associations.sort(key=lambda x: x[1].strength, reverse=True)
        
        # Update last_activated timestamps
        current_time = time.time()
        for _, association in relevant_associations[:max_associations]:
            association.last_activated = current_time
            association.reinforcement_count += 1
        
        return relevant_associations[:max_associations]

    def get_cluster_for_memory(self, npc_id: uuid.UUID, memory_id: str) -> Optional[EmotionalMemoryCluster]:
        """Get the cluster containing a specific memory."""
        
        npc_id_str = str(npc_id)
        network = self._emotional_networks.get(npc_id_str)
        
        if not network:
            return None
        
        for cluster in network.clusters.values():
            if memory_id in cluster.member_memories:
                return cluster
        
        return None

    def get_network_statistics(self, npc_id: uuid.UUID) -> Dict[str, Any]:
        """Get statistics about an NPC's emotional memory network."""
        
        npc_id_str = str(npc_id)
        network = self._emotional_networks.get(npc_id_str)
        
        if not network:
            return {"error": "Network not found"}
        
        # Calculate network statistics
        total_memories = sum(len(cluster.member_memories) for cluster in network.clusters.values())
        total_associations = len(network.associations)
        avg_cluster_size = total_memories / len(network.clusters) if network.clusters else 0
        
        # Association strength distribution
        association_strengths = [assoc.strength for assoc in network.associations.values()]
        avg_association_strength = sum(association_strengths) / len(association_strengths) if association_strengths else 0
        
        # Most connected clusters
        cluster_connections = {}
        for cluster_id, cluster in network.clusters.items():
            cluster_connections[cluster_id] = len(cluster.external_connections)
        
        most_connected = sorted(cluster_connections.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            "total_clusters": len(network.clusters),
            "total_memories": total_memories,
            "total_associations": total_associations,
            "average_cluster_size": round(avg_cluster_size, 1),
            "average_association_strength": round(avg_association_strength, 2),
            "network_coherence": network.network_coherence,
            "most_connected_clusters": [{"cluster_id": cid, "connections": conn} for cid, conn in most_connected],
            "dominant_patterns": network.dominant_patterns,
            "vulnerability_points": network.vulnerability_points,
            "last_updated": datetime.fromtimestamp(network.last_updated).isoformat(),
        }

    def _extract_clustering_features(
        self, 
        memories: List[EmotionalMemoryRecord],
        personality: NPCPersonality,
    ) -> np.ndarray:
        """Extract features for clustering emotional memories."""
        
        features = []
        
        for memory in memories:
            significance = memory.emotional_significance
            
            # Emotional features
            feature_vector = [
                significance.overall_significance,
                significance.intensity_score,
                significance.personal_relevance,
                significance.relationship_impact,
                significance.formative_influence,
                significance.decay_resistance,
                significance.triggering_potential,
            ]
            
            # Emotional type encoding (one-hot)
            emotion_types = list(EmotionalMemoryType)
            type_encoding = [1.0 if significance.emotional_type == et else 0.0 for et in emotion_types]
            feature_vector.extend(type_encoding)
            
            # Protection level encoding
            protection_levels = [0.2, 0.4, 0.6, 0.8, 1.0]  # PUBLIC to TRAUMATIC
            protection_encoding = [significance.protection_level.value == level for level in ['public', 'private', 'sensitive', 'protected', 'traumatic']]
            feature_vector.extend([float(x) for x in protection_encoding])
            
            # Mood accessibility features (average across moods)
            if significance.mood_accessibility:
                avg_accessibility = sum(significance.mood_accessibility.values()) / len(significance.mood_accessibility)
                accessibility_variance = np.var(list(significance.mood_accessibility.values()))
            else:
                avg_accessibility = 0.5
                accessibility_variance = 0.0
            
            feature_vector.extend([avg_accessibility, accessibility_variance])
            
            # Temporal features
            memory_age = (time.time() - memory.preserved_at) / 3600  # Age in hours
            feature_vector.append(min(1.0, memory_age / (24 * 30)))  # Normalize to ~30 days
            
            # Affective weight features
            affective = memory.affective_weight
            feature_vector.extend([
                affective.base_affective_weight,
                affective.intensity_multiplier,
                affective.personality_modifier,
                affective.mood_accessibility_modifier,
                affective.final_weight,
            ])
            
            features.append(feature_vector)
        
        return np.array(features)

    async def _perform_clustering(
        self,
        features: np.ndarray,
        memories: List[EmotionalMemoryRecord],
        method: ClusteringMethod,
    ) -> Dict[int, List[int]]:
        """Perform clustering using specified method."""
        
        if method == ClusteringMethod.KMEANS:
            n_clusters = min(self.max_clusters, max(2, len(memories) // 5))
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(features)
        
        elif method == ClusteringMethod.DBSCAN:
            clusterer = DBSCAN(eps=0.5, min_samples=self.min_cluster_size)
            cluster_labels = clusterer.fit_predict(features)
        
        elif method == ClusteringMethod.TEMPORAL:
            return self._temporal_clustering(memories)
        
        elif method == ClusteringMethod.SEMANTIC:
            return await self._semantic_clustering(memories)
        
        elif method == ClusteringMethod.EMOTIONAL_RESONANCE:
            return self._emotional_resonance_clustering(memories)
        
        elif method == ClusteringMethod.HYBRID:
            # Combine multiple clustering methods
            return await self._hybrid_clustering(features, memories)
        
        else:
            # Default to K-means
            n_clusters = min(self.max_clusters, max(2, len(memories) // 5))
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(features)
        
        # Convert cluster labels to assignment dictionary
        cluster_assignments = {}
        for i, label in enumerate(cluster_labels):
            if label == -1:  # Noise point (DBSCAN)
                continue
            if label not in cluster_assignments:
                cluster_assignments[label] = []
            cluster_assignments[label].append(i)
        
        # Filter out small clusters
        filtered_assignments = {
            k: v for k, v in cluster_assignments.items() 
            if len(v) >= self.min_cluster_size
        }
        
        return filtered_assignments

    def _temporal_clustering(self, memories: List[EmotionalMemoryRecord]) -> Dict[int, List[int]]:
        """Cluster memories based on temporal proximity."""
        
        # Sort memories by timestamp
        sorted_memories = sorted(enumerate(memories), key=lambda x: x[1].preserved_at)
        
        clusters = {}
        current_cluster = 0
        cluster_memories = []
        
        last_timestamp = None
        time_threshold = 3600 * 6  # 6 hours
        
        for original_idx, memory in sorted_memories:
            if last_timestamp is None or (memory.preserved_at - last_timestamp) <= time_threshold:
                cluster_memories.append(original_idx)
            else:
                # Start new cluster if we have enough memories
                if len(cluster_memories) >= self.min_cluster_size:
                    clusters[current_cluster] = cluster_memories
                    current_cluster += 1
                cluster_memories = [original_idx]
            
            last_timestamp = memory.preserved_at
        
        # Add final cluster
        if len(cluster_memories) >= self.min_cluster_size:
            clusters[current_cluster] = cluster_memories
        
        return clusters

    async def _semantic_clustering(self, memories: List[EmotionalMemoryRecord]) -> Dict[int, List[int]]:
        """Cluster memories based on semantic content similarity."""
        
        # This would require semantic embeddings from the actual conversation content
        # For now, use emotional features as a proxy
        
        feature_vectors = []
        for memory in memories:
            # Use contributing factors as semantic features
            factors = memory.emotional_significance.contributing_factors
            # Create simple feature vector from factors
            feature_vector = [
                len(factors),
                memory.emotional_significance.personal_relevance,
                memory.emotional_significance.relationship_impact,
            ]
            feature_vectors.append(feature_vector)
        
        if not feature_vectors:
            return {}
        
        features = np.array(feature_vectors)
        
        # Use DBSCAN for semantic clustering
        clusterer = DBSCAN(eps=0.3, min_samples=self.min_cluster_size)
        cluster_labels = clusterer.fit_predict(features)
        
        # Convert to assignment dictionary
        cluster_assignments = {}
        for i, label in enumerate(cluster_labels):
            if label == -1:  # Noise
                continue
            if label not in cluster_assignments:
                cluster_assignments[label] = []
            cluster_assignments[label].append(i)
        
        return cluster_assignments

    def _emotional_resonance_clustering(self, memories: List[EmotionalMemoryRecord]) -> Dict[int, List[int]]:
        """Cluster memories based on emotional resonance patterns."""
        
        # Group by emotional type first
        type_groups = {}
        for i, memory in enumerate(memories):
            emotion_type = memory.emotional_significance.emotional_type
            if emotion_type not in type_groups:
                type_groups[emotion_type] = []
            type_groups[emotion_type].append(i)
        
        # Then subdivide by intensity and significance
        clusters = {}
        cluster_id = 0
        
        for emotion_type, memory_indices in type_groups.items():
            if len(memory_indices) < self.min_cluster_size:
                continue
            
            # Extract intensity and significance for this type
            type_features = []
            for idx in memory_indices:
                memory = memories[idx]
                type_features.append([
                    memory.emotional_significance.intensity_score,
                    memory.emotional_significance.overall_significance,
                ])
            
            if len(type_features) >= self.min_cluster_size:
                # Further cluster within emotional type
                type_features = np.array(type_features)
                sub_clusterer = KMeans(n_clusters=min(3, len(memory_indices) // self.min_cluster_size + 1), random_state=42)
                sub_labels = sub_clusterer.fit_predict(type_features)
                
                # Create sub-clusters
                for sub_label in set(sub_labels):
                    sub_cluster_indices = [memory_indices[i] for i, l in enumerate(sub_labels) if l == sub_label]
                    if len(sub_cluster_indices) >= self.min_cluster_size:
                        clusters[cluster_id] = sub_cluster_indices
                        cluster_id += 1
        
        return clusters

    async def _hybrid_clustering(
        self, 
        features: np.ndarray, 
        memories: List[EmotionalMemoryRecord]
    ) -> Dict[int, List[int]]:
        """Combine multiple clustering methods for robust clustering."""
        
        # Get clustering results from different methods
        kmeans_clusters = {}
        n_clusters = min(self.max_clusters, max(2, len(memories) // 5))
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = clusterer.fit_predict(features)
        
        for i, label in enumerate(kmeans_labels):
            if label not in kmeans_clusters:
                kmeans_clusters[label] = []
            kmeans_clusters[label].append(i)
        
        temporal_clusters = self._temporal_clustering(memories)
        emotional_clusters = self._emotional_resonance_clustering(memories)
        
        # Combine clustering results using consensus
        memory_indices = list(range(len(memories)))
        final_clusters = {}
        used_memories = set()
        cluster_id = 0
        
        # For each memory, find which clusters it belongs to across methods
        for memory_idx in memory_indices:
            if memory_idx in used_memories:
                continue
            
            # Find all memories that co-cluster with this one across methods
            consensus_group = {memory_idx}
            
            # Check K-means co-clustering
            for k_cluster in kmeans_clusters.values():
                if memory_idx in k_cluster:
                    consensus_group.update(k_cluster)
                    break
            
            # Check temporal co-clustering
            for t_cluster in temporal_clusters.values():
                if memory_idx in t_cluster:
                    consensus_group.intersection_update(t_cluster)
            
            # Check emotional co-clustering
            for e_cluster in emotional_clusters.values():
                if memory_idx in e_cluster:
                    consensus_group.intersection_update(e_cluster)
            
            # Only create cluster if it meets size requirements
            if len(consensus_group) >= self.min_cluster_size:
                final_clusters[cluster_id] = list(consensus_group)
                used_memories.update(consensus_group)
                cluster_id += 1
        
        return final_clusters

    async def _build_clusters_from_assignments(
        self,
        cluster_assignments: Dict[int, List[int]],
        memories: List[EmotionalMemoryRecord],
        npc_id: str,
    ) -> List[EmotionalMemoryCluster]:
        """Build cluster objects from clustering assignments."""
        
        clusters = []
        
        for cluster_label, memory_indices in cluster_assignments.items():
            cluster_memories = [memories[i] for i in memory_indices]
            
            # Determine dominant emotional type
            type_counts = {}
            for memory in cluster_memories:
                emotion_type = memory.emotional_significance.emotional_type
                type_counts[emotion_type] = type_counts.get(emotion_type, 0) + 1
            
            dominant_type = max(type_counts, key=type_counts.get)
            
            # Calculate temporal span
            timestamps = [m.preserved_at for m in cluster_memories]
            temporal_span = (min(timestamps), max(timestamps))
            
            # Create cluster
            cluster = EmotionalMemoryCluster(
                cluster_id=f"{npc_id}_cluster_{cluster_label}_{int(time.time())}",
                npc_id=npc_id,
                dominant_emotion_type=dominant_type,
                emotional_theme=self._generate_emotional_theme(cluster_memories),
                emotional_coherence=0.0,  # Will be calculated later
                temporal_span=temporal_span,
                member_memories=[m.exchange_id for m in cluster_memories],
            )
            
            clusters.append(cluster)
        
        return clusters

    async def _calculate_cluster_properties(
        self,
        cluster: EmotionalMemoryCluster,
        all_memories: List[EmotionalMemoryRecord],
        personality: NPCPersonality,
    ) -> None:
        """Calculate comprehensive properties for a memory cluster."""
        
        # Get cluster memories
        cluster_memories = [
            m for m in all_memories 
            if m.exchange_id in cluster.member_memories
        ]
        
        if not cluster_memories:
            return
        
        # Calculate averages
        significances = [m.emotional_significance.overall_significance for m in cluster_memories]
        intensities = [m.emotional_significance.intensity_score for m in cluster_memories]
        
        cluster.average_significance = sum(significances) / len(significances)
        cluster.average_intensity = sum(intensities) / len(intensities)
        
        # Calculate emotional coherence
        emotion_types = [m.emotional_significance.emotional_type for m in cluster_memories]
        dominant_count = max(emotion_types.count(et) for et in set(emotion_types))
        cluster.emotional_coherence = dominant_count / len(cluster_memories)
        
        # Protection level distribution
        protection_counts = {}
        for memory in cluster_memories:
            level = memory.emotional_significance.protection_level.value
            protection_counts[level] = protection_counts.get(level, 0) + 1
        cluster.protection_level_distribution = protection_counts
        
        # Mood accessibility profile
        mood_sums = {}
        mood_counts = {}
        for memory in cluster_memories:
            for mood, accessibility in memory.mood_accessibility.items():
                mood_sums[mood] = mood_sums.get(mood, 0) + accessibility
                mood_counts[mood] = mood_counts.get(mood, 0) + 1
        
        cluster.mood_accessibility_profile = {
            mood: mood_sums[mood] / mood_counts[mood]
            for mood in mood_sums
        }
        
        # Identify core vs peripheral memories
        # Core memories are those with above-average significance and centrality
        significance_threshold = cluster.average_significance
        core_memories = []
        peripheral_memories = []
        
        for memory in cluster_memories:
            if memory.emotional_significance.overall_significance >= significance_threshold:
                core_memories.append(memory.exchange_id)
            else:
                peripheral_memories.append(memory.exchange_id)
        
        cluster.core_memories = core_memories
        cluster.peripheral_memories = peripheral_memories
        
        # Calculate stability score based on temporal distribution and significance
        time_span = cluster.temporal_span[1] - cluster.temporal_span[0]
        if time_span > 0:
            # More spread out memories = more stable cluster
            stability = min(1.0, time_span / (30 * 24 * 3600))  # Normalize to 30 days
        else:
            stability = 0.5  # Single time point
        
        # Adjust for significance
        stability *= cluster.average_significance
        cluster.stability_score = stability

    def _generate_emotional_theme(self, memories: List[EmotionalMemoryRecord]) -> str:
        """Generate descriptive theme for a cluster of memories."""
        
        if not memories:
            return "empty_cluster"
        
        # Get dominant emotional type
        type_counts = {}
        for memory in memories:
            emotion_type = memory.emotional_significance.emotional_type
            type_counts[emotion_type] = type_counts.get(emotion_type, 0) + 1
        
        dominant_type = max(type_counts, key=type_counts.get)
        
        # Get average significance level
        avg_significance = sum(m.emotional_significance.overall_significance for m in memories) / len(memories)
        
        # Get temporal characteristics
        timestamps = [m.preserved_at for m in memories]
        time_span = max(timestamps) - min(timestamps)
        
        # Generate theme based on characteristics
        theme_components = []
        
        # Add type component
        theme_components.append(dominant_type.value)
        
        # Add significance component
        if avg_significance > 0.8:
            theme_components.append("high_impact")
        elif avg_significance > 0.6:
            theme_components.append("significant")
        elif avg_significance > 0.4:
            theme_components.append("moderate")
        else:
            theme_components.append("minor")
        
        # Add temporal component
        if time_span < 3600:  # Less than 1 hour
            theme_components.append("acute_episode")
        elif time_span < 86400:  # Less than 1 day
            theme_components.append("daily_experience")
        elif time_span < 604800:  # Less than 1 week
            theme_components.append("extended_period")
        else:
            theme_components.append("longterm_pattern")
        
        return "_".join(theme_components)

    async def _build_intra_cluster_associations(
        self,
        cluster: EmotionalMemoryCluster,
        memory_lookup: Dict[str, EmotionalMemoryRecord],
        personality: NPCPersonality,
    ) -> List[MemoryAssociation]:
        """Build associations within a single cluster."""
        
        associations = []
        member_ids = cluster.member_memories
        
        # Create associations between all pairs in cluster
        for i in range(len(member_ids)):
            for j in range(i + 1, len(member_ids)):
                memory1_id = member_ids[i]
                memory2_id = member_ids[j]
                
                memory1 = memory_lookup.get(memory1_id)
                memory2 = memory_lookup.get(memory2_id)
                
                if not memory1 or not memory2:
                    continue
                
                # Calculate association strength
                strength = self._calculate_association_strength(memory1, memory2, personality)
                
                if strength >= 0.3:  # Minimum threshold for intra-cluster associations
                    association = MemoryAssociation(
                        association_id=f"intra_{cluster.cluster_id}_{memory1_id}_{memory2_id}",
                        source_memory_id=memory1_id,
                        target_memory_id=memory2_id,
                        association_type=AssociationType.THEMATIC,
                        strength=strength,
                        bidirectional=True,
                        base_strength=strength,
                    )
                    associations.append(association)
        
        return associations

    async def _build_inter_cluster_associations(
        self,
        clusters: List[EmotionalMemoryCluster],
        memory_lookup: Dict[str, EmotionalMemoryRecord],
        personality: NPCPersonality,
    ) -> List[MemoryAssociation]:
        """Build associations between different clusters."""
        
        associations = []
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster1 = clusters[i]
                cluster2 = clusters[j]
                
                # Check if clusters should be connected
                connection_strength = self._calculate_cluster_connection_strength(
                    cluster1, cluster2, memory_lookup
                )
                
                if connection_strength >= 0.5:
                    # Create associations between core memories of connected clusters
                    for mem1_id in cluster1.core_memories[:3]:  # Limit to top 3
                        for mem2_id in cluster2.core_memories[:3]:
                            memory1 = memory_lookup.get(mem1_id)
                            memory2 = memory_lookup.get(mem2_id)
                            
                            if not memory1 or not memory2:
                                continue
                            
                            strength = self._calculate_association_strength(memory1, memory2, personality)
                            strength *= connection_strength  # Scale by cluster connection
                            
                            if strength >= 0.4:  # Higher threshold for inter-cluster
                                association = MemoryAssociation(
                                    association_id=f"inter_{cluster1.cluster_id}_{cluster2.cluster_id}_{mem1_id}_{mem2_id}",
                                    source_memory_id=mem1_id,
                                    target_memory_id=mem2_id,
                                    association_type=AssociationType.TRIGGER_RESPONSE,
                                    strength=strength,
                                    bidirectional=False,
                                    base_strength=strength,
                                )
                                associations.append(association)
                                
                                # Add cluster connection
                                if cluster2.cluster_id not in cluster1.external_connections:
                                    cluster1.external_connections.append(cluster2.cluster_id)
                                if cluster1.cluster_id not in cluster2.external_connections:
                                    cluster2.external_connections.append(cluster1.cluster_id)
        
        return associations

    async def _build_semantic_associations(
        self,
        memories: List[EmotionalMemoryRecord],
        personality: NPCPersonality,
    ) -> List[MemoryAssociation]:
        """Build associations based on semantic content similarity."""
        
        associations = []
        
        # This would ideally use actual content embeddings
        # For now, use contributing factors as a proxy for semantic content
        
        for i in range(len(memories)):
            for j in range(i + 1, len(memories)):
                memory1 = memories[i]
                memory2 = memories[j]
                
                # Calculate semantic similarity from contributing factors
                factors1 = set(memory1.emotional_significance.contributing_factors)
                factors2 = set(memory2.emotional_significance.contributing_factors)
                
                if factors1 and factors2:
                    # Jaccard similarity
                    intersection = len(factors1.intersection(factors2))
                    union = len(factors1.union(factors2))
                    semantic_similarity = intersection / union if union > 0 else 0.0
                    
                    if semantic_similarity >= 0.3:
                        strength = semantic_similarity * 0.8  # Scale down semantic associations
                        
                        association = MemoryAssociation(
                            association_id=f"semantic_{memory1.exchange_id}_{memory2.exchange_id}",
                            source_memory_id=memory1.exchange_id,
                            target_memory_id=memory2.exchange_id,
                            association_type=AssociationType.THEMATIC,
                            strength=strength,
                            bidirectional=True,
                            base_strength=strength,
                        )
                        associations.append(association)
        
        return associations

    def _build_temporal_associations(self, memories: List[EmotionalMemoryRecord]) -> List[MemoryAssociation]:
        """Build associations between temporally proximate memories."""
        
        associations = []
        
        # Sort memories by timestamp
        sorted_memories = sorted(memories, key=lambda m: m.preserved_at)
        
        # Create associations between consecutive memories within time threshold
        time_threshold = 3600 * 2  # 2 hours
        
        for i in range(len(sorted_memories) - 1):
            memory1 = sorted_memories[i]
            memory2 = sorted_memories[i + 1]
            
            time_diff = memory2.preserved_at - memory1.preserved_at
            
            if time_diff <= time_threshold:
                # Strength decreases with time distance
                strength = max(0.3, 1.0 - (time_diff / time_threshold))
                
                association = MemoryAssociation(
                    association_id=f"temporal_{memory1.exchange_id}_{memory2.exchange_id}",
                    source_memory_id=memory1.exchange_id,
                    target_memory_id=memory2.exchange_id,
                    association_type=AssociationType.TEMPORAL,
                    strength=strength,
                    bidirectional=False,  # Temporal associations are directional
                    base_strength=strength,
                )
                associations.append(association)
        
        return associations

    def _calculate_association_strength(
        self,
        memory1: EmotionalMemoryRecord,
        memory2: EmotionalMemoryRecord,
        personality: NPCPersonality,
    ) -> float:
        """Calculate association strength between two memories."""
        
        sig1 = memory1.emotional_significance
        sig2 = memory2.emotional_significance
        
        # Base similarity factors
        factors = []
        
        # Emotional type similarity
        if sig1.emotional_type == sig2.emotional_type:
            factors.append(0.8)
        else:
            # Some types are more similar than others
            type_similarities = {
                (EmotionalMemoryType.PEAK_POSITIVE, EmotionalMemoryType.BREAKTHROUGH): 0.6,
                (EmotionalMemoryType.TRAUMATIC, EmotionalMemoryType.SIGNIFICANT_LOSS): 0.7,
                (EmotionalMemoryType.CONFLICT, EmotionalMemoryType.TRUST_EVENT): 0.5,
                # Add more type similarities as needed
            }
            
            type_pair = (sig1.emotional_type, sig2.emotional_type)
            reverse_pair = (sig2.emotional_type, sig1.emotional_type)
            
            similarity = type_similarities.get(type_pair, type_similarities.get(reverse_pair, 0.2))
            factors.append(similarity)
        
        # Intensity similarity
        intensity_diff = abs(sig1.intensity_score - sig2.intensity_score)
        intensity_similarity = 1.0 - intensity_diff
        factors.append(intensity_similarity * 0.6)
        
        # Significance similarity
        significance_diff = abs(sig1.overall_significance - sig2.overall_significance)
        significance_similarity = 1.0 - significance_diff
        factors.append(significance_similarity * 0.7)
        
        # Personal relevance similarity
        relevance_diff = abs(sig1.personal_relevance - sig2.personal_relevance)
        relevance_similarity = 1.0 - relevance_diff
        factors.append(relevance_similarity * 0.5)
        
        # Protection level similarity
        protection_diff = abs(
            list(MemoryProtectionLevel).index(sig1.protection_level) -
            list(MemoryProtectionLevel).index(sig2.protection_level)
        )
        protection_similarity = 1.0 - (protection_diff / len(MemoryProtectionLevel))
        factors.append(protection_similarity * 0.4)
        
        # Calculate weighted average
        strength = sum(factors) / len(factors) if factors else 0.0
        
        return max(0.0, min(1.0, strength))

    def _calculate_cluster_connection_strength(
        self,
        cluster1: EmotionalMemoryCluster,
        cluster2: EmotionalMemoryCluster,
        memory_lookup: Dict[str, EmotionalMemoryRecord],
    ) -> float:
        """Calculate how strongly two clusters should be connected."""
        
        # Emotional type compatibility
        if cluster1.dominant_emotion_type == cluster2.dominant_emotion_type:
            type_compatibility = 0.8
        else:
            # Check if emotion types typically associate
            type_associations = {
                (EmotionalMemoryType.TRAUMATIC, EmotionalMemoryType.CORE_ATTACHMENT): 0.7,  # Trauma->attachment seeking
                (EmotionalMemoryType.CONFLICT, EmotionalMemoryType.SIGNIFICANT_LOSS): 0.6,  # Conflict->loss
                (EmotionalMemoryType.BREAKTHROUGH, EmotionalMemoryType.PEAK_POSITIVE): 0.8, # Achievement->joy
                # Add more associations
            }
            
            type_pair = (cluster1.dominant_emotion_type, cluster2.dominant_emotion_type)
            reverse_pair = (cluster2.dominant_emotion_type, cluster1.dominant_emotion_type)
            
            type_compatibility = type_associations.get(type_pair, type_associations.get(reverse_pair, 0.3))
        
        # Temporal proximity of clusters
        cluster1_center = sum(cluster1.temporal_span) / 2
        cluster2_center = sum(cluster2.temporal_span) / 2
        time_diff = abs(cluster2_center - cluster1_center)
        
        # Connections stronger for temporally close clusters
        max_time_for_connection = 30 * 24 * 3600  # 30 days
        temporal_factor = max(0.0, 1.0 - (time_diff / max_time_for_connection))
        
        # Significance factor
        significance_factor = (cluster1.average_significance + cluster2.average_significance) / 2
        
        # Combine factors
        connection_strength = (
            type_compatibility * 0.5 +
            temporal_factor * 0.3 +
            significance_factor * 0.2
        )
        
        return connection_strength

    async def _calculate_network_properties(
        self,
        network: EmotionalNetwork,
        personality: NPCPersonality,
    ) -> None:
        """Calculate network-level properties."""
        
        if not network.clusters:
            return
        
        # Network coherence - how emotionally consistent the network is
        cluster_coherences = [cluster.emotional_coherence for cluster in network.clusters.values()]
        network.network_coherence = sum(cluster_coherences) / len(cluster_coherences)
        
        # Dominant patterns - most common emotional themes
        themes = [cluster.emotional_theme for cluster in network.clusters.values()]
        theme_counts = {}
        for theme in themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        # Get top 3 patterns
        sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
        network.dominant_patterns = [theme for theme, count in sorted_themes[:3]]
        
        # Vulnerability points - clusters with high trauma/negative content
        vulnerability_points = []
        for cluster in network.clusters.values():
            if (cluster.dominant_emotion_type in [EmotionalMemoryType.TRAUMATIC, EmotionalMemoryType.SIGNIFICANT_LOSS]
                and cluster.average_significance > 0.7):
                vulnerability_points.append(cluster.cluster_id)
        network.vulnerability_points = vulnerability_points
        
        # Entry points - clusters that are frequently accessed
        access_scores = [(cluster.cluster_id, cluster.access_frequency) for cluster in network.clusters.values()]
        access_scores.sort(key=lambda x: x[1], reverse=True)
        network.entry_points = [cluster_id for cluster_id, _ in access_scores[:5]]
        
        # Frequent pathways - most common association paths
        pathway_counts = {}
        for association in network.associations.values():
            # Find clusters for source and target
            source_cluster = None
            target_cluster = None
            
            for cluster in network.clusters.values():
                if association.source_memory_id in cluster.member_memories:
                    source_cluster = cluster.cluster_id
                if association.target_memory_id in cluster.member_memories:
                    target_cluster = cluster.cluster_id
            
            if source_cluster and target_cluster and source_cluster != target_cluster:
                pathway = (source_cluster, target_cluster)
                pathway_counts[pathway] = pathway_counts.get(pathway, 0) + 1
        
        # Get top pathways
        sorted_pathways = sorted(pathway_counts.items(), key=lambda x: x[1], reverse=True)
        network.frequent_pathways = [pathway for pathway, count in sorted_pathways[:10]]

    def _update_avg_clustering_time(self, processing_time_ms: float) -> None:
        """Update average clustering time statistic."""
        total_operations = self._performance_stats["clustering_operations"]
        if total_operations > 0:
            current_avg = self._performance_stats["avg_clustering_time_ms"]
            self._performance_stats["avg_clustering_time_ms"] = (
                current_avg * (total_operations - 1) + processing_time_ms
            ) / total_operations

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "clustering_operations": self._performance_stats["clustering_operations"],
            "association_formations": self._performance_stats["association_formations"],
            "network_updates": self._performance_stats["network_updates"],
            "cache_hits": self._performance_stats["cache_hits"],
            "avg_clustering_time_ms": round(self._performance_stats["avg_clustering_time_ms"], 2),
            "active_networks": len(self._emotional_networks),
            "total_clusters": sum(len(net.clusters) for net in self._emotional_networks.values()),
            "total_associations": sum(len(net.associations) for net in self._emotional_networks.values()),
        }

    def clear_caches(self) -> None:
        """Clear all caches."""
        self._clustering_cache.clear()
        self._association_cache.clear()
        self._performance_stats = {
            "clustering_operations": 0,
            "association_formations": 0,
            "network_updates": 0,
            "cache_hits": 0,
            "avg_clustering_time_ms": 0.0,
        }