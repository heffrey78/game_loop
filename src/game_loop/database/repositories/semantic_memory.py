"""Repository layer for semantic memory operations with optimized batch processing."""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

from pgvector.sqlalchemy import Vector
from sqlalchemy import and_, case, desc, func, or_, text, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import select

from ..models.conversation import (
    ConversationContext,
    ConversationExchange,
    EmotionalContext,
    MemoryCluster,
    MemoryEmbedding,
    NPCPersonality,
)
from .base import BaseRepository


class SemanticMemoryRepository(BaseRepository[ConversationExchange]):
    """Repository for semantic memory operations with batch processing and vector similarity."""

    def __init__(self, session: AsyncSession):
        super().__init__(ConversationExchange, session)

    # Batch Vector Retrieval Methods

    async def get_memories_with_embeddings_batch(
        self,
        npc_id: Optional[uuid.UUID] = None,
        min_confidence: float = 0.0,
        min_emotional_weight: float = 0.0,
        max_trust_level: float = 1.0,
        batch_size: int = 1000,
        offset: int = 0,
    ) -> List[ConversationExchange]:
        """Get batch of memories with embeddings for clustering analysis."""
        stmt = (
            select(ConversationExchange)
            .options(
                selectinload(ConversationExchange.memory_embedding_entry),
                selectinload(ConversationExchange.emotional_context_entry),
                selectinload(ConversationExchange.conversation),
            )
            .where(ConversationExchange.memory_embedding.is_not(None))
        )

        # Apply filters
        conditions = []
        if npc_id:
            # Filter by NPC through conversation join
            stmt = stmt.join(ConversationContext, ConversationExchange.conversation_id == ConversationContext.conversation_id)
            conditions.append(ConversationContext.npc_id == npc_id)

        if min_confidence > 0.0:
            conditions.append(ConversationExchange.confidence_score >= min_confidence)

        if min_emotional_weight > 0.0:
            conditions.append(ConversationExchange.emotional_weight >= min_emotional_weight)

        if max_trust_level < 1.0:
            conditions.append(ConversationExchange.trust_level_required <= max_trust_level)

        if conditions:
            stmt = stmt.where(and_(*conditions))

        # Apply batch pagination
        stmt = stmt.order_by(ConversationExchange.timestamp.desc()).offset(offset).limit(batch_size)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def stream_memories_for_clustering(
        self,
        npc_id: Optional[uuid.UUID] = None,
        min_confidence: float = 0.3,
        batch_size: int = 1000,
    ) -> AsyncGenerator[List[ConversationExchange], None]:
        """Stream memories in batches for memory-efficient clustering operations."""
        offset = 0
        while True:
            batch = await self.get_memories_with_embeddings_batch(
                npc_id=npc_id,
                min_confidence=min_confidence,
                batch_size=batch_size,
                offset=offset,
            )
            if not batch:
                break
            yield batch
            offset += batch_size
            if len(batch) < batch_size:
                break

    async def get_clustering_data_preparation(
        self,
        npc_id: uuid.UUID,
        min_cluster_size: int = 5,
        max_memories: int = 10000,
    ) -> Dict[str, Any]:
        """Prepare data for K-means clustering with optimized queries."""
        # Get memory count
        count_stmt = (
            select(func.count(ConversationExchange.exchange_id))
            .join(ConversationContext, ConversationExchange.conversation_id == ConversationContext.conversation_id)
            .where(
                and_(
                    ConversationContext.npc_id == npc_id,
                    ConversationExchange.memory_embedding.is_not(None),
                    ConversationExchange.confidence_score >= 0.3,
                )
            )
        )
        result = await self.session.execute(count_stmt)
        total_memories = result.scalar() or 0

        if total_memories < min_cluster_size:
            return {
                "total_memories": total_memories,
                "suitable_for_clustering": False,
                "suggested_clusters": 0,
                "memory_ids": [],
                "embeddings": [],
            }

        # Calculate optimal cluster count (rule of thumb: sqrt(n/2))
        import math
        suggested_clusters = max(2, min(10, int(math.sqrt(total_memories / 2))))

        # Get memory data for clustering
        memories_batch = await self.get_memories_with_embeddings_batch(
            npc_id=npc_id,
            min_confidence=0.3,
            batch_size=min(max_memories, total_memories),
        )

        memory_ids = [str(memory.exchange_id) for memory in memories_batch]
        embeddings = [memory.memory_embedding for memory in memories_batch if memory.memory_embedding]

        return {
            "total_memories": total_memories,
            "suitable_for_clustering": total_memories >= min_cluster_size,
            "suggested_clusters": suggested_clusters,
            "memory_ids": memory_ids,
            "embeddings": embeddings,
            "batch_size": len(memories_batch),
        }

    # Batch Update Operations

    async def batch_update_confidence_scores(
        self,
        updates: List[Tuple[uuid.UUID, float]],
        update_timestamp: bool = True,
    ) -> int:
        """Batch update confidence scores without N+1 queries."""
        if not updates:
            return 0

        # Input validation for security
        self._validate_batch_updates(updates)

        # Use SQLAlchemy's case construct for safe parameterized queries
        exchange_ids = [exchange_id for exchange_id, _ in updates]
        
        # Create case statement with parameterized values (secure against SQL injection)
        confidence_case = case(
            *[(ConversationExchange.exchange_id == eid, conf) for eid, conf in updates],
            else_=ConversationExchange.confidence_score
        )
        
        # Build update statement with proper type handling
        stmt = (
            update(ConversationExchange)
            .where(ConversationExchange.exchange_id.in_(exchange_ids))
            .values(confidence_score=confidence_case)
        )
        
        if update_timestamp:
            now_utc = datetime.now(timezone.utc)
            stmt = stmt.values(last_accessed=now_utc)

        result = await self.session.execute(stmt)
        # Note: Transaction commit/rollback is handled by external session management
        # Caller is responsible for proper transaction boundaries
        return result.rowcount

    async def batch_update_cluster_assignments(
        self,
        assignments: List[Tuple[uuid.UUID, int, float]],  # (exchange_id, cluster_id, confidence)
    ) -> int:
        """Batch update memory cluster assignments."""
        if not assignments:
            return 0

        # Input validation for security
        self._validate_cluster_assignments(assignments)

        exchange_ids = [assignment[0] for assignment in assignments]
        
        # Create secure case statements with parameterized values
        cluster_case = case(
            *[(ConversationExchange.exchange_id == eid, cluster_id) 
              for eid, cluster_id, _ in assignments],
            else_=ConversationExchange.memory_cluster_id
        )
        
        confidence_case = case(
            *[(ConversationExchange.exchange_id == eid, confidence) 
              for eid, _, confidence in assignments],
            else_=ConversationExchange.cluster_confidence_score
        )

        now_utc = datetime.now(timezone.utc)
        stmt = (
            update(ConversationExchange)
            .where(ConversationExchange.exchange_id.in_(exchange_ids))
            .values(
                memory_cluster_id=cluster_case,
                cluster_confidence_score=confidence_case,
                cluster_assignment_timestamp=now_utc,
            )
        )

        result = await self.session.execute(stmt)
        # Note: Transaction commit/rollback is handled by external session management
        # Caller is responsible for proper transaction boundaries
        return result.rowcount

    async def batch_increment_access_counts(
        self,
        exchange_ids: List[uuid.UUID],
    ) -> int:
        """Batch increment access counts for memory retrieval tracking."""
        if not exchange_ids:
            return 0

        stmt = (
            update(ConversationExchange)
            .where(ConversationExchange.exchange_id.in_(exchange_ids))
            .values(
                access_count=ConversationExchange.access_count + 1,
                last_accessed=datetime.now(timezone.utc),
            )
        )

        result = await self.session.execute(stmt)
        # Note: Transaction commit/rollback is handled by external session management
        # Caller is responsible for proper transaction boundaries  
        return result.rowcount

    # Vector Similarity Operations

    async def find_similar_memories(
        self,
        query_embedding: List[float],
        npc_id: Optional[uuid.UUID] = None,
        similarity_threshold: float = 0.7,
        min_confidence: float = 0.3,
        min_emotional_weight: float = 0.0,
        max_trust_level: float = 1.0,
        limit: int = 10,
        exclude_recent_hours: int = 0,
    ) -> List[Tuple[ConversationExchange, float]]:
        """Find similar memories using pgvector cosine similarity with sub-50ms performance."""
        # Input validation for security
        self._validate_embedding_dimensions(query_embedding)
        self._validate_similarity_threshold(similarity_threshold)
        
        if limit <= 0 or limit > 1000:
            raise ValueError(f"Limit must be between 1 and 1000: {limit}")
        
        # Build the similarity query using pgvector's cosine similarity
        stmt = (
            select(
                ConversationExchange,
                ConversationExchange.memory_embedding.cosine_distance(query_embedding).label("distance"),
            )
            .options(
                selectinload(ConversationExchange.conversation),
                selectinload(ConversationExchange.emotional_context_entry),
            )
            .where(ConversationExchange.memory_embedding.is_not(None))
        )

        # Apply filters
        conditions = []

        # Similarity threshold (convert distance to similarity: 1 - distance >= threshold)
        conditions.append(
            ConversationExchange.memory_embedding.cosine_distance(query_embedding) <= (1 - similarity_threshold)
        )

        if npc_id:
            stmt = stmt.join(ConversationContext, ConversationExchange.conversation_id == ConversationContext.conversation_id)
            conditions.append(ConversationContext.npc_id == npc_id)

        if min_confidence > 0.0:
            conditions.append(ConversationExchange.confidence_score >= min_confidence)

        if min_emotional_weight > 0.0:
            conditions.append(ConversationExchange.emotional_weight >= min_emotional_weight)

        if max_trust_level < 1.0:
            conditions.append(ConversationExchange.trust_level_required <= max_trust_level)

        if exclude_recent_hours > 0:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=exclude_recent_hours)
            conditions.append(ConversationExchange.timestamp < cutoff_time)

        if conditions:
            stmt = stmt.where(and_(*conditions))

        # Order by similarity (ascending distance = descending similarity)
        # Also consider temporal importance - more recent memories get slight boost
        stmt = stmt.order_by(
            ConversationExchange.memory_embedding.cosine_distance(query_embedding),
            desc(ConversationExchange.timestamp),
        ).limit(limit)

        result = await self.session.execute(stmt)
        rows = result.all()

        # Convert distance to similarity and return
        return [(row[0], 1.0 - row[1]) for row in rows]

    async def find_emotionally_similar_memories(
        self,
        target_emotional_profile: Dict[str, float],
        npc_id: Optional[uuid.UUID] = None,
        emotional_threshold: float = 0.6,
        limit: int = 20,
    ) -> List[ConversationExchange]:
        """Find memories with similar emotional profiles."""
        # This would ideally use vector operations on emotional profiles
        # For now, we'll use a simplified approach with emotional weight filtering
        
        min_emotional_weight = target_emotional_profile.get("intensity", 0.5)
        
        stmt = (
            select(ConversationExchange)
            .options(
                selectinload(ConversationExchange.emotional_context_entry),
                selectinload(ConversationExchange.conversation),
            )
            .where(
                and_(
                    ConversationExchange.emotional_weight >= min_emotional_weight,
                    ConversationExchange.memory_embedding.is_not(None),
                )
            )
        )

        if npc_id:
            stmt = stmt.join(ConversationContext, ConversationExchange.conversation_id == ConversationContext.conversation_id)
            stmt = stmt.where(ConversationContext.npc_id == npc_id)

        stmt = stmt.order_by(
            desc(ConversationExchange.emotional_weight),
            desc(ConversationExchange.confidence_score),
            desc(ConversationExchange.timestamp),
        ).limit(limit)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    # Advanced Query Operations

    async def get_memory_cluster_analysis(
        self,
        npc_id: uuid.UUID,
        include_unclustered: bool = False,
    ) -> Dict[str, Any]:
        """Get comprehensive memory cluster analysis for an NPC."""
        # Get cluster statistics
        cluster_stats_stmt = (
            select(
                MemoryCluster.cluster_id,
                MemoryCluster.cluster_name,
                MemoryCluster.cluster_theme,
                MemoryCluster.member_count,
                MemoryCluster.avg_confidence,
                MemoryCluster.emotional_profile,
            )
            .where(MemoryCluster.npc_id == npc_id)
            .order_by(desc(MemoryCluster.member_count))
        )

        cluster_result = await self.session.execute(cluster_stats_stmt)
        clusters = [
            {
                "cluster_id": row[0],
                "name": row[1],
                "theme": row[2],
                "member_count": row[3],
                "avg_confidence": float(row[4]) if row[4] else 0.0,
                "emotional_profile": row[5] or {},
            }
            for row in cluster_result.all()
        ]

        # Get unclustered memory count if requested
        unclustered_count = 0
        if include_unclustered:
            unclustered_stmt = (
                select(func.count(ConversationExchange.exchange_id))
                .join(ConversationContext, ConversationExchange.conversation_id == ConversationContext.conversation_id)
                .where(
                    and_(
                        ConversationContext.npc_id == npc_id,
                        ConversationExchange.memory_cluster_id.is_(None),
                        ConversationExchange.memory_embedding.is_not(None),
                    )
                )
            )
            result = await self.session.execute(unclustered_stmt)
            unclustered_count = result.scalar() or 0

        total_clustered = sum(cluster["member_count"] for cluster in clusters)

        return {
            "npc_id": str(npc_id),
            "clusters": clusters,
            "total_clusters": len(clusters),
            "total_clustered_memories": total_clustered,
            "unclustered_memories": unclustered_count,
            "clustering_coverage": (
                total_clustered / (total_clustered + unclustered_count)
                if (total_clustered + unclustered_count) > 0
                else 0.0
            ),
        }

    async def get_temporal_memory_distribution(
        self,
        npc_id: uuid.UUID,
        days_back: int = 30,
        bucket_hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """Get temporal distribution of memories for trend analysis."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        # Use PostgreSQL's date_trunc for efficient time bucketing
        time_bucket_expr = func.date_trunc("day", ConversationExchange.timestamp)
        stmt = (
            select(
                time_bucket_expr.label("time_bucket"),
                func.count(ConversationExchange.exchange_id).label("memory_count"),
                func.avg(ConversationExchange.confidence_score).label("avg_confidence"),
                func.avg(ConversationExchange.emotional_weight).label("avg_emotional_weight"),
            )
            .join(ConversationContext, ConversationExchange.conversation_id == ConversationContext.conversation_id)
            .where(
                and_(
                    ConversationContext.npc_id == npc_id,
                    ConversationExchange.timestamp >= cutoff_date,
                )
            )
            .group_by(time_bucket_expr)
            .order_by(time_bucket_expr)
        )

        result = await self.session.execute(stmt)
        
        return [
            {
                "date": row[0].isoformat() if row[0] else None,
                "memory_count": row[1],
                "avg_confidence": float(row[2]) if row[2] else 0.0,
                "avg_emotional_weight": float(row[3]) if row[3] else 0.0,
            }
            for row in result.all()
        ]

    # Health and Performance Methods

    async def get_repository_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics for semantic memory operations."""
        # Total memories with embeddings
        total_stmt = select(func.count(ConversationExchange.exchange_id)).where(
            ConversationExchange.memory_embedding.is_not(None)
        )
        total_result = await self.session.execute(total_stmt)
        total_memories = total_result.scalar() or 0

        # Clustered memories
        clustered_stmt = select(func.count(ConversationExchange.exchange_id)).where(
            ConversationExchange.memory_cluster_id.is_not(None)
        )
        clustered_result = await self.session.execute(clustered_stmt)
        clustered_memories = clustered_result.scalar() or 0

        # Average confidence
        avg_confidence_stmt = select(func.avg(ConversationExchange.confidence_score)).where(
            ConversationExchange.memory_embedding.is_not(None)
        )
        avg_conf_result = await self.session.execute(avg_confidence_stmt)
        avg_confidence = avg_conf_result.scalar() or 0.0

        # Recent activity (last 24 hours)
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_stmt = select(func.count(ConversationExchange.exchange_id)).where(
            and_(
                ConversationExchange.memory_embedding.is_not(None),
                ConversationExchange.last_accessed >= recent_cutoff,
            )
        )
        recent_result = await self.session.execute(recent_stmt)
        recent_activity = recent_result.scalar() or 0

        return {
            "total_memories_with_embeddings": total_memories,
            "clustered_memories": clustered_memories,
            "clustering_coverage": clustered_memories / total_memories if total_memories > 0 else 0.0,
            "average_confidence_score": float(avg_confidence),
            "recent_activity_24h": recent_activity,
            "activity_rate": recent_activity / total_memories if total_memories > 0 else 0.0,
            "repository_status": "healthy" if total_memories > 0 and avg_confidence > 0.3 else "needs_attention",
        }

    # Input Validation Methods (Security)

    def _validate_batch_updates(self, updates: List[Tuple[uuid.UUID, float]]) -> None:
        """Validate batch confidence score updates for security and data integrity."""
        if not updates:
            return
            
        if len(updates) > 10000:  # Prevent resource exhaustion
            raise ValueError("Batch size exceeds maximum limit of 10,000 updates")
            
        for exchange_id, confidence in updates:
            if not isinstance(exchange_id, uuid.UUID):
                raise ValueError(f"Invalid UUID format for exchange_id: {exchange_id}")
            if not isinstance(confidence, (int, float)):
                raise ValueError(f"Confidence score must be numeric: {confidence}")
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"Confidence score must be between 0.0 and 1.0: {confidence}")

    def _validate_cluster_assignments(self, assignments: List[Tuple[uuid.UUID, int, float]]) -> None:
        """Validate cluster assignments for security and data integrity."""
        if not assignments:
            return
            
        if len(assignments) > 10000:  # Prevent resource exhaustion
            raise ValueError("Batch size exceeds maximum limit of 10,000 assignments")
            
        for exchange_id, cluster_id, confidence in assignments:
            if not isinstance(exchange_id, uuid.UUID):
                raise ValueError(f"Invalid UUID format for exchange_id: {exchange_id}")
            if not isinstance(cluster_id, int) or cluster_id < 0:
                raise ValueError(f"Cluster ID must be a non-negative integer: {cluster_id}")
            if not isinstance(confidence, (int, float)):
                raise ValueError(f"Confidence score must be numeric: {confidence}")
            if not (0.0 <= confidence <= 1.0):
                raise ValueError(f"Confidence score must be between 0.0 and 1.0: {confidence}")

    def _validate_embedding_dimensions(self, embedding: List[float], expected_dim: int = 384) -> None:
        """Validate embedding dimensions for consistency."""
        if len(embedding) != expected_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {expected_dim}, got {len(embedding)}")
        
        if not all(isinstance(val, (int, float)) for val in embedding):
            raise ValueError("All embedding values must be numeric")

    def _validate_similarity_threshold(self, threshold: float) -> None:
        """Validate similarity threshold is within valid range."""
        if not isinstance(threshold, (int, float)):
            raise ValueError(f"Similarity threshold must be numeric: {threshold}")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Similarity threshold must be between 0.0 and 1.0: {threshold}")

    # Transaction Management Utilities

    async def _execute_batch_with_transaction(
        self, operation_func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Execute batch operation with proper transaction management and rollback on failure."""
        try:
            # Note: We rely on the external session transaction management
            # The session should be within a transaction context when this repository is used
            result = await operation_func(*args, **kwargs)
            # Flush to detect any constraint violations early
            await self.session.flush()
            return result
        except Exception as e:
            # Log the error for monitoring
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Batch operation failed: {e}", exc_info=True)
            # Re-raise to allow external transaction to handle rollback
            raise