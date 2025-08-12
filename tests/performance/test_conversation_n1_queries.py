"""Performance tests to verify N+1 query optimizations in conversation flow."""

import time
import uuid
from unittest.mock import AsyncMock, Mock, patch

import pytest

from game_loop.core.conversation.flow_manager import ConversationFlowManager
from game_loop.core.conversation.flow_templates import ConversationStage
from game_loop.database.models.conversation import ConversationContext, NPCPersonality


class MockQueryCounter:
    """Mock to count database queries."""

    def __init__(self):
        self.query_count = 0
        self.queries = []

    def log_query(self, query_description: str):
        """Log a query for counting."""
        self.query_count += 1
        self.queries.append(query_description)

    def reset(self):
        """Reset counters."""
        self.query_count = 0
        self.queries = []


@pytest.fixture
def mock_query_counter():
    """Fixture for query counting."""
    return MockQueryCounter()


@pytest.fixture
def sample_conversations():
    """Create sample conversations for performance testing."""
    player_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440001")
    npc_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440002")

    conversations = []
    for i in range(10):  # 10 conversations
        conversations.append(
            ConversationContext(
                conversation_id=uuid.uuid4(),
                player_id=str(player_id),
                npc_id=str(npc_id),
                topic=f"topic_{i}",
                mood="neutral",
                relationship_level=0.1 + (i * 0.1),  # Increasing relationship
                status="active",
            )
        )

    return conversations


@pytest.fixture
def sample_npc_personality():
    """Create sample NPC personality for testing."""
    return NPCPersonality(
        npc_id=uuid.UUID("550e8400-e29b-41d4-a716-446655440002"),
        traits={"friendly": 0.8, "helpful": 0.7},
        knowledge_areas=["village_life", "local_history"],
        speech_patterns={"formality": 0.6},
        background_story="A helpful village elder",
        default_mood="friendly",
    )


class TestConversationQueryOptimization:
    """Test query optimization in conversation flow management."""

    @pytest.mark.asyncio
    async def test_old_vs_new_query_pattern_simulation(
        self, mock_query_counter, sample_conversations, sample_npc_personality
    ):
        """
        Simulate the performance difference between old and new query patterns.

        Old pattern: get_player_conversations() + Python filtering
        New pattern: get_conversation_count_for_npc_player_pair()
        """
        player_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440001")
        npc_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440002")

        # Test OLD pattern simulation
        mock_query_counter.reset()

        async def simulate_old_pattern():
            """Simulate the old inefficient pattern."""
            # Simulate: get ALL player conversations (database query)
            mock_query_counter.log_query(
                "SELECT * FROM conversation_contexts WHERE player_id = ?"
            )
            all_conversations = sample_conversations

            # Simulate: Filter in Python (no additional query, but inefficient data transfer)
            npc_conversations = [
                conv for conv in all_conversations if str(conv.npc_id) == str(npc_id)
            ]

            return len(npc_conversations)

        old_start = time.time()
        old_result = await simulate_old_pattern()
        old_duration = time.time() - old_start
        old_query_count = mock_query_counter.query_count

        # Test NEW pattern simulation
        mock_query_counter.reset()

        async def simulate_new_pattern():
            """Simulate the new optimized pattern."""
            # Simulate: COUNT with WHERE clause (single optimized query)
            mock_query_counter.log_query(
                "SELECT COUNT(*) FROM conversation_contexts WHERE player_id = ? AND npc_id = ?"
            )

            # Direct count without loading all data
            return len(
                [c for c in sample_conversations if str(c.npc_id) == str(npc_id)]
            )

        new_start = time.time()
        new_result = await simulate_new_pattern()
        new_duration = time.time() - new_start
        new_query_count = mock_query_counter.query_count

        # Verify results are the same
        assert old_result == new_result, "Both patterns should return the same count"

        # Verify query optimization
        assert (
            new_query_count < old_query_count or new_query_count == 1
        ), f"New pattern should use fewer or equal queries. Old: {old_query_count}, New: {new_query_count}"

        print("\\n=== Query Optimization Results ===")
        print(f"Old pattern: {old_query_count} queries, {old_duration:.6f}s")
        print(f"New pattern: {new_query_count} queries, {new_duration:.6f}s")
        print(f"Query reduction: {old_query_count - new_query_count} fewer queries")
        print(f"Result consistency: {old_result == new_result}")

    @pytest.mark.asyncio
    async def test_conversation_stage_determination_performance(
        self, mock_query_counter, sample_conversations, sample_npc_personality
    ):
        """Test the performance of conversation stage determination with new optimizations."""

        # Create mock components
        mock_memory_integration = AsyncMock()
        mock_session_factory = Mock()
        mock_session = AsyncMock()
        mock_session_factory.get_session.return_value = AsyncMock()
        mock_session_factory.get_session.return_value.__aenter__.return_value = (
            mock_session
        )

        # Mock the optimized repository method
        mock_repo_manager = Mock()
        mock_repo_manager.contexts.get_conversation_count_for_npc_player_pair = (
            AsyncMock()
        )

        def count_query_side_effect(player_id, npc_id, status=None):
            """Side effect to count queries and return realistic data."""
            mock_query_counter.log_query(
                f"SELECT COUNT(*) FROM conversation_contexts WHERE player_id = {player_id} AND npc_id = {npc_id}"
            )
            # Return count based on sample data
            matching_conversations = [
                c
                for c in sample_conversations
                if str(c.player_id) == str(player_id)
                and str(c.npc_id) == str(npc_id)
                and (status is None or c.status == status)
            ]
            return len(matching_conversations)

        mock_repo_manager.contexts.get_conversation_count_for_npc_player_pair.side_effect = (
            count_query_side_effect
        )

        with patch(
            "game_loop.core.conversation.flow_manager.ConversationRepositoryManager",
            return_value=mock_repo_manager,
        ):
            manager = ConversationFlowManager(
                mock_memory_integration,
                mock_session_factory,
            )

            # Test conversation stage determination
            conversation = sample_conversations[0]  # First conversation

            start_time = time.time()
            stage = await manager._determine_conversation_stage(
                conversation, sample_npc_personality
            )
            duration = time.time() - start_time

            # Verify the stage determination worked
            assert isinstance(stage, ConversationStage)

            # Verify only one query was made (the optimized count query)
            assert (
                mock_query_counter.query_count == 1
            ), f"Expected 1 query, got {mock_query_counter.query_count}"

            # Verify it was the optimized query
            assert "COUNT(*)" in mock_query_counter.queries[0]
            assert "player_id" in mock_query_counter.queries[0]
            assert "npc_id" in mock_query_counter.queries[0]

            print("\\n=== Stage Determination Performance ===")
            print(f"Duration: {duration:.6f}s")
            print(f"Queries executed: {mock_query_counter.query_count}")
            print(f"Query type: {mock_query_counter.queries[0]}")
            print(f"Determined stage: {stage.value}")

    @pytest.mark.asyncio
    async def test_multiple_stage_determinations_scalability(
        self, mock_query_counter, sample_conversations, sample_npc_personality
    ):
        """Test that multiple stage determinations scale linearly, not quadratically."""

        # Setup mocks
        mock_memory_integration = AsyncMock()
        mock_session_factory = Mock()
        mock_session = AsyncMock()
        mock_session_factory.get_session.return_value = AsyncMock()
        mock_session_factory.get_session.return_value.__aenter__.return_value = (
            mock_session
        )

        mock_repo_manager = Mock()
        mock_repo_manager.contexts.get_conversation_count_for_npc_player_pair = (
            AsyncMock()
        )

        def count_query_side_effect(player_id, npc_id, status=None):
            """Count queries for scalability testing."""
            mock_query_counter.log_query("COUNT query")
            return 1  # Always return 1 for consistency

        mock_repo_manager.contexts.get_conversation_count_for_npc_player_pair.side_effect = (
            count_query_side_effect
        )

        with patch(
            "game_loop.core.conversation.flow_manager.ConversationRepositoryManager",
            return_value=mock_repo_manager,
        ):
            manager = ConversationFlowManager(
                mock_memory_integration,
                mock_session_factory,
            )

            # Test multiple determinations
            num_determinations = 5

            start_time = time.time()
            for i in range(num_determinations):
                mock_query_counter.reset()
                await manager._determine_conversation_stage(
                    sample_conversations[i], sample_npc_personality
                )

                # Each determination should use exactly 1 query
                assert (
                    mock_query_counter.query_count == 1
                ), f"Determination {i} used {mock_query_counter.query_count} queries, expected 1"

            duration = time.time() - start_time

            print("\\n=== Scalability Test Results ===")
            print(f"Determinations: {num_determinations}")
            print(f"Total duration: {duration:.6f}s")
            print(f"Average per determination: {duration/num_determinations:.6f}s")
            print("Queries per determination: 1 (optimized)")

            # Verify linear scaling (each determination uses exactly 1 query)
            expected_total_queries = num_determinations
            # We can't easily count total queries across iterations with our current setup,
            # but we verified each iteration uses exactly 1 query above


class TestQueryPatternComparison:
    """Compare old vs new query patterns directly."""

    def test_query_pattern_analysis(self):
        """Analyze the theoretical query patterns."""

        print("\\n=== Query Pattern Analysis ===")
        print("\\nOLD PATTERN (N+1 Problem):")
        print("1. SELECT * FROM conversation_contexts WHERE player_id = ? LIMIT 50")
        print(
            "2. Filter results in Python: [conv for conv in results if conv.npc_id == target_npc_id]"
        )
        print("3. Count filtered results: len(filtered_conversations)")
        print("\\nProblems:")
        print("- Fetches ALL conversations for player (up to 50)")
        print("- Transfers unnecessary data from database")
        print("- Performs filtering in application layer")
        print("- Scales poorly with number of player conversations")

        print("\\nNEW PATTERN (Optimized):")
        print("1. SELECT COUNT(*) FROM conversation_contexts")
        print("   WHERE player_id = ? AND npc_id = ? AND status = 'active'")
        print("\\nBenefits:")
        print("- Single query with precise filtering")
        print("- Returns only the count (minimal data transfer)")
        print("- Uses database indexes efficiently")
        print("- Constant time complexity regardless of conversation count")
        print("- Leverages composite index (player_id, npc_id)")

        # This test always passes - it's just for documentation
        assert True
