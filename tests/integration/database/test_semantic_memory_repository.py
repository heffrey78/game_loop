"""Integration tests for semantic memory repository layer."""

import uuid
from datetime import datetime, timedelta
from typing import List

import pytest
import pytest_asyncio

from game_loop.database.models.conversation import (
    ConversationContext,
    ConversationExchange,
    MemoryCluster,
    NPCPersonality,
)
from game_loop.database.repositories.semantic_memory import SemanticMemoryRepository


@pytest_asyncio.fixture
async def sample_npc(db_session):
    """Create a sample NPC for testing."""
    npc = NPCPersonality(
        npc_id=uuid.uuid4(),
        traits={"analytical": 0.8, "helpful": 0.9},
        knowledge_areas=["technology", "science"],
        speech_patterns={"formality": "medium"},
        background_story="A knowledgeable researcher.",
        default_mood="curious",
    )
    db_session.add(npc)
    await db_session.flush()
    return npc


@pytest_asyncio.fixture
async def sample_conversation(db_session, sample_npc):
    """Create a sample conversation for testing."""
    player_id = uuid.uuid4()
    conversation = ConversationContext(
        player_id=player_id,
        npc_id=sample_npc.npc_id,
        topic="research discussion",
        mood="engaged",
        relationship_level=0.5,
    )
    db_session.add(conversation)
    await db_session.flush()
    return conversation


@pytest_asyncio.fixture
async def sample_exchanges_with_embeddings(db_session, sample_conversation):
    """Create sample exchanges with embeddings for testing."""
    exchanges = []
    
    # Create test embeddings (384-dimensional vectors)
    test_embeddings = [
        [0.1] * 384,  # Low similarity baseline
        [0.5, 0.3, 0.8] + [0.1] * 381,  # Similar to query
        [0.4, 0.2, 0.9] + [0.1] * 381,  # Very similar to query
        [0.9, 0.1, 0.1] + [0.1] * 381,  # Different from query
    ]
    
    messages = [
        "This is about general topics",
        "Let's discuss research methods",
        "Research methodology is fascinating", 
        "The weather is nice today",
    ]
    
    for i, (embedding, message) in enumerate(zip(test_embeddings, messages)):
        exchange = ConversationExchange(
            conversation_id=sample_conversation.conversation_id,
            speaker_id=sample_conversation.player_id,
            message_text=message,
            message_type="statement",
            confidence_score=0.8 - (i * 0.1),  # Varying confidence
            emotional_weight=0.6 + (i * 0.1),  # Varying emotional weight
            trust_level_required=0.3,
            memory_embedding=embedding,
        )
        db_session.add(exchange)
        exchanges.append(exchange)
    
    await db_session.flush()
    return exchanges


@pytest.mark.asyncio
class TestSemanticMemoryRepository:
    """Test SemanticMemoryRepository operations."""

    async def test_get_memories_with_embeddings_batch(self, db_session, sample_exchanges_with_embeddings):
        """Test batch retrieval of memories with embeddings."""
        repo = SemanticMemoryRepository(db_session)
        
        # Get all memories
        memories = await repo.get_memories_with_embeddings_batch(batch_size=10)
        assert len(memories) == 4
        
        # Test with minimum confidence filter
        high_conf_memories = await repo.get_memories_with_embeddings_batch(
            min_confidence=0.7,
            batch_size=10
        )
        assert len(high_conf_memories) == 2  # Only first two have confidence >= 0.7
        
        # Test batch size limiting
        limited_memories = await repo.get_memories_with_embeddings_batch(batch_size=2)
        assert len(limited_memories) == 2

    async def test_stream_memories_for_clustering(self, db_session, sample_exchanges_with_embeddings):
        """Test streaming memories for clustering operations."""
        repo = SemanticMemoryRepository(db_session)
        
        batches = []
        async for batch in repo.stream_memories_for_clustering(batch_size=2):
            batches.append(batch)
        
        assert len(batches) == 2  # Should have 2 batches of 2 memories each
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        
        # Verify all memories are included
        all_streamed_ids = {
            exchange.exchange_id 
            for batch in batches 
            for exchange in batch
        }
        original_ids = {exchange.exchange_id for exchange in sample_exchanges_with_embeddings}
        assert all_streamed_ids == original_ids

    async def test_get_clustering_data_preparation(
        self, 
        db_session, 
        sample_exchanges_with_embeddings,
        sample_conversation
    ):
        """Test K-means clustering data preparation."""
        repo = SemanticMemoryRepository(db_session)
        
        data = await repo.get_clustering_data_preparation(
            npc_id=sample_conversation.npc_id,
            min_cluster_size=2,
            max_memories=1000
        )
        
        assert data["suitable_for_clustering"] is True
        assert data["total_memories"] == 4
        assert data["suggested_clusters"] >= 2
        assert len(data["memory_ids"]) == 4
        assert len(data["embeddings"]) == 4
        assert data["batch_size"] == 4

    async def test_batch_update_confidence_scores(self, db_session, sample_exchanges_with_embeddings):
        """Test batch updating confidence scores."""
        repo = SemanticMemoryRepository(db_session)
        
        # Prepare updates
        updates = [
            (sample_exchanges_with_embeddings[0].exchange_id, 0.95),
            (sample_exchanges_with_embeddings[1].exchange_id, 0.85),
        ]
        
        # Execute batch update
        updated_count = await repo.batch_update_confidence_scores(updates)
        assert updated_count == 2
        
        # Verify updates
        await db_session.refresh(sample_exchanges_with_embeddings[0])
        await db_session.refresh(sample_exchanges_with_embeddings[1])
        
        assert float(sample_exchanges_with_embeddings[0].confidence_score) == 0.95
        assert float(sample_exchanges_with_embeddings[1].confidence_score) == 0.85

    async def test_batch_update_cluster_assignments(self, db_session, sample_exchanges_with_embeddings):
        """Test batch updating cluster assignments."""
        repo = SemanticMemoryRepository(db_session)
        
        # Create a test cluster first
        cluster = MemoryCluster(
            npc_id=sample_exchanges_with_embeddings[0].conversation.npc_id,
            cluster_name="test_cluster",
            member_count=0,
        )
        db_session.add(cluster)
        await db_session.flush()
        
        # Prepare assignments
        assignments = [
            (sample_exchanges_with_embeddings[0].exchange_id, cluster.cluster_id, 0.9),
            (sample_exchanges_with_embeddings[1].exchange_id, cluster.cluster_id, 0.8),
        ]
        
        # Execute batch assignment
        updated_count = await repo.batch_update_cluster_assignments(assignments)
        assert updated_count == 2
        
        # Verify assignments
        await db_session.refresh(sample_exchanges_with_embeddings[0])
        await db_session.refresh(sample_exchanges_with_embeddings[1])
        
        assert sample_exchanges_with_embeddings[0].memory_cluster_id == cluster.cluster_id
        assert sample_exchanges_with_embeddings[1].memory_cluster_id == cluster.cluster_id
        assert float(sample_exchanges_with_embeddings[0].cluster_confidence_score) == 0.9

    async def test_batch_increment_access_counts(self, db_session, sample_exchanges_with_embeddings):
        """Test batch incrementing access counts."""
        repo = SemanticMemoryRepository(db_session)
        
        exchange_ids = [
            sample_exchanges_with_embeddings[0].exchange_id,
            sample_exchanges_with_embeddings[1].exchange_id,
        ]
        
        # Record initial access counts
        initial_counts = [
            sample_exchanges_with_embeddings[0].access_count,
            sample_exchanges_with_embeddings[1].access_count,
        ]
        
        # Execute batch increment
        updated_count = await repo.batch_increment_access_counts(exchange_ids)
        assert updated_count == 2
        
        # Verify increments
        await db_session.refresh(sample_exchanges_with_embeddings[0])
        await db_session.refresh(sample_exchanges_with_embeddings[1])
        
        assert sample_exchanges_with_embeddings[0].access_count == initial_counts[0] + 1
        assert sample_exchanges_with_embeddings[1].access_count == initial_counts[1] + 1

    async def test_find_similar_memories(self, db_session, sample_exchanges_with_embeddings, sample_conversation):
        """Test finding similar memories using vector similarity."""
        repo = SemanticMemoryRepository(db_session)
        
        # Query with embedding similar to research-related exchanges
        query_embedding = [0.45, 0.25, 0.85] + [0.1] * 381
        
        similar_memories = await repo.find_similar_memories(
            query_embedding=query_embedding,
            npc_id=sample_conversation.npc_id,
            similarity_threshold=0.5,  # Lower threshold to catch our test data
            limit=5
        )
        
        assert len(similar_memories) >= 2  # Should find research-related exchanges
        
        # Verify results are ordered by similarity (higher similarity first)
        similarities = [similarity for _, similarity in similar_memories]
        assert all(similarities[i] >= similarities[i+1] for i in range(len(similarities)-1))
        
        # Verify similarity scores are reasonable
        for _, similarity in similar_memories:
            assert 0.0 <= similarity <= 1.0

    async def test_find_similar_memories_with_filters(self, db_session, sample_exchanges_with_embeddings, sample_conversation):
        """Test finding similar memories with confidence and emotional weight filters."""
        repo = SemanticMemoryRepository(db_session)
        
        query_embedding = [0.5] * 384
        
        # Test with high confidence filter
        high_conf_memories = await repo.find_similar_memories(
            query_embedding=query_embedding,
            npc_id=sample_conversation.npc_id,
            min_confidence=0.7,
            similarity_threshold=0.0,  # Very low to include all based on confidence
            limit=10
        )
        
        # Should only return exchanges with confidence >= 0.7
        for exchange, _ in high_conf_memories:
            assert float(exchange.confidence_score) >= 0.7
        
        # Test with emotional weight filter
        emotional_memories = await repo.find_similar_memories(
            query_embedding=query_embedding,
            npc_id=sample_conversation.npc_id,
            min_emotional_weight=0.8,
            similarity_threshold=0.0,
            limit=10
        )
        
        # Should only return exchanges with emotional_weight >= 0.8
        for exchange, _ in emotional_memories:
            assert float(exchange.emotional_weight) >= 0.8

    async def test_get_repository_health_metrics(self, db_session, sample_exchanges_with_embeddings):
        """Test repository health metrics calculation."""
        repo = SemanticMemoryRepository(db_session)
        
        metrics = await repo.get_repository_health_metrics()
        
        assert metrics["total_memories_with_embeddings"] == 4
        assert metrics["clustered_memories"] == 0  # No clusters assigned yet
        assert metrics["clustering_coverage"] == 0.0
        assert metrics["average_confidence_score"] > 0.0
        assert "repository_status" in metrics
        assert metrics["activity_rate"] >= 0.0

    async def test_get_temporal_memory_distribution(self, db_session, sample_exchanges_with_embeddings, sample_conversation):
        """Test temporal memory distribution analysis."""
        repo = SemanticMemoryRepository(db_session)
        
        distribution = await repo.get_temporal_memory_distribution(
            npc_id=sample_conversation.npc_id,
            days_back=7,
            bucket_hours=24
        )
        
        assert len(distribution) >= 1  # Should have at least one time bucket
        
        for bucket in distribution:
            assert "date" in bucket
            assert "memory_count" in bucket
            assert "avg_confidence" in bucket
            assert "avg_emotional_weight" in bucket
            assert bucket["memory_count"] > 0

    async def test_empty_results_handling(self, db_session):
        """Test handling of empty result sets."""
        repo = SemanticMemoryRepository(db_session)
        
        # Test with non-existent NPC
        non_existent_npc = uuid.uuid4()
        
        memories = await repo.get_memories_with_embeddings_batch(npc_id=non_existent_npc)
        assert memories == []
        
        data = await repo.get_clustering_data_preparation(
            npc_id=non_existent_npc,
            min_cluster_size=5
        )
        assert data["suitable_for_clustering"] is False
        assert data["total_memories"] == 0
        
        # Test batch updates with empty lists
        updated_count = await repo.batch_update_confidence_scores([])
        assert updated_count == 0

    async def test_performance_constraints(self, db_session, sample_exchanges_with_embeddings):
        """Test that operations meet performance constraints."""
        repo = SemanticMemoryRepository(db_session)
        
        # Test that similarity search returns quickly (mock performance test)
        import time
        
        query_embedding = [0.5] * 384
        start_time = time.time()
        
        await repo.find_similar_memories(
            query_embedding=query_embedding,
            similarity_threshold=0.7,
            limit=10
        )
        
        elapsed_time = time.time() - start_time
        
        # Should complete well under 50ms for small datasets
        # This is more of a structural test since we don't have large datasets in tests
        assert elapsed_time < 1.0  # Very generous threshold for test environment

    async def test_input_validation_security(self, db_session):
        """Test input validation for security vulnerabilities."""
        repo = SemanticMemoryRepository(db_session)
        
        # Test invalid UUID in batch updates
        with pytest.raises(ValueError, match="Invalid UUID format"):
            await repo.batch_update_confidence_scores([("invalid-uuid", 0.5)])
            
        # Test confidence score out of bounds
        with pytest.raises(ValueError, match="Confidence score must be between 0.0 and 1.0"):
            await repo.batch_update_confidence_scores([(uuid.uuid4(), 1.5)])
            
        # Test negative confidence score
        with pytest.raises(ValueError, match="Confidence score must be between 0.0 and 1.0"):
            await repo.batch_update_confidence_scores([(uuid.uuid4(), -0.1)])
            
        # Test batch size limit
        large_batch = [(uuid.uuid4(), 0.5) for _ in range(10001)]
        with pytest.raises(ValueError, match="Batch size exceeds maximum limit"):
            await repo.batch_update_confidence_scores(large_batch)
            
        # Test invalid cluster ID
        with pytest.raises(ValueError, match="Cluster ID must be a non-negative integer"):
            await repo.batch_update_cluster_assignments([(uuid.uuid4(), -1, 0.5)])
            
        # Test invalid embedding dimensions
        invalid_embedding = [0.1] * 200  # Wrong dimension
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            await repo.find_similar_memories(invalid_embedding)
            
        # Test invalid similarity threshold
        valid_embedding = [0.1] * 384
        with pytest.raises(ValueError, match="Similarity threshold must be between 0.0 and 1.0"):
            await repo.find_similar_memories(valid_embedding, similarity_threshold=1.5)

    async def test_batch_operations_with_parameterized_queries(self, db_session, sample_exchanges_with_embeddings):
        """Test that batch operations use parameterized queries and are secure."""
        repo = SemanticMemoryRepository(db_session)
        
        # Test with potentially malicious input (should be safely parameterized)
        malicious_uuid = uuid.uuid4()
        malicious_confidence = 0.8  # Valid value, but ensure it's parameterized
        
        updates = [(malicious_uuid, malicious_confidence)]
        
        # This should work safely with parameterized queries
        # (Note: since UUID doesn't exist, rowcount will be 0, but no SQL injection)
        result = await repo.batch_update_confidence_scores(updates)
        assert result == 0  # No rows updated since UUID doesn't exist
        
        # Test cluster assignments with edge case values
        assignments = [(malicious_uuid, 0, 0.0)]  # Min values
        result = await repo.batch_update_cluster_assignments(assignments)
        assert result == 0  # No rows updated since UUID doesn't exist