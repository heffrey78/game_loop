"""Integration tests for conversation threading system."""

import uuid
import pytest
from datetime import datetime, timezone

from game_loop.core.conversation.conversation_threading import (
    ConversationThreadingService,
)
from game_loop.core.conversation.conversation_models import (
    ConversationContext,
    NPCPersonality,
)
from game_loop.database.models.conversation_threading import (
    ConversationThread,
    PlayerMemoryProfile,
    TopicEvolution,
)
from game_loop.database.repositories.conversation_threading import (
    ConversationThreadingManager,
)


@pytest.mark.asyncio
@pytest.mark.integration
class TestConversationThreadingIntegration:
    """Integration tests for the complete conversation threading system."""

    async def test_complete_threading_workflow(
        self, async_session_factory, mock_memory_integration
    ):
        """Test the complete conversation threading workflow from start to finish."""
        # Create test data
        player_id = uuid.uuid4()
        npc_id = uuid.uuid4()

        # Initialize threading service
        threading_service = ConversationThreadingService(
            session_factory=async_session_factory,
            memory_integration=mock_memory_integration,
            enable_threading=True,
        )

        # Create NPC personality
        async with async_session_factory.get_session() as session:
            npc = NPCPersonality(
                npc_id=npc_id,
                traits={"talkative": 0.8, "wise": 0.6, "helpful": 0.7},
                knowledge_areas=["exploration", "combat"],
                speech_patterns={"formal": 0.3, "casual": 0.7},
                background_story="A veteran adventurer who helps newcomers",
                default_mood="friendly",
            )
            session.add(npc)
            await session.commit()

        # Step 1: Initiate first conversation
        conversation1 = ConversationContext(
            conversation_id=str(uuid.uuid4()),
            player_id=str(player_id),
            npc_id=str(npc_id),
            topic="adventure planning",
            mood="excited",
            relationship_level=0.3,
        )

        threading_context1 = await threading_service.initiate_conversation_session(
            player_id=player_id,
            npc_id=npc_id,
            conversation=conversation1,
            initial_topic="adventure planning",
        )

        assert threading_context1.active_thread is not None
        assert threading_context1.player_profile is not None
        assert threading_context1.topic_continuity_score >= 0.0

        initial_thread_id = threading_context1.active_thread.thread_id

        # Step 2: Record topic evolution
        await threading_service.record_conversation_evolution(
            conversation=conversation1,
            previous_topic="adventure planning",
            new_topic="equipment needs",
            player_initiated=True,
            evolution_quality="natural",
        )

        # Step 3: Finalize first session
        await threading_service.finalize_conversation_session(
            conversation=conversation1,
            session_success=True,
            relationship_change=0.1,
            memorable_moments=["Discussed dungeon exploration strategy"],
        )

        # Step 4: Start second conversation (should continue thread)
        conversation2 = ConversationContext(
            conversation_id=str(uuid.uuid4()),
            player_id=str(player_id),
            npc_id=str(npc_id),
            topic="equipment selection",
            mood="focused",
            relationship_level=0.4,
        )

        threading_context2 = await threading_service.initiate_conversation_session(
            player_id=player_id,
            npc_id=npc_id,
            conversation=conversation2,
            initial_topic="equipment selection",
        )

        # Should continue the same thread
        assert threading_context2.active_thread.thread_id == initial_thread_id
        assert threading_context2.player_profile.relationship_level >= 0.4

        # Step 5: Analyze threading opportunities
        analysis = await threading_service.analyze_threading_opportunity(
            conversation=conversation2,
            personality=npc,
            player_message="I think we need better armor for the dungeon",
            current_topic="equipment selection",
        )

        # Should suggest references due to topic continuity
        assert analysis.should_reference_past is True
        assert analysis.reference_confidence > 0.0
        assert len(analysis.suggested_references) > 0

        # Step 6: Enhance response with threading
        base_response = "Good thinking about armor selection."
        enhanced_response, threading_data = (
            await threading_service.enhance_response_with_threading(
                base_response=base_response,
                threading_analysis=analysis,
                conversation=conversation2,
                personality=npc,
                current_topic="equipment selection",
            )
        )

        assert threading_data["threading_enhanced"] is True
        assert enhanced_response != base_response
        assert len(enhanced_response) > len(base_response)

        # Step 7: Verify persistence
        async with async_session_factory.get_session() as session:
            threading_manager = ConversationThreadingManager(session)

            # Check thread persistence
            threads = await threading_manager.threads.get_player_threads(
                player_id=player_id, npc_id=npc_id
            )
            assert len(threads) == 1
            assert threads[0].thread_id == initial_thread_id
            assert threads[0].session_count >= 2

            # Check profile persistence
            profile = await threading_manager.profiles.get_profile(player_id, npc_id)
            assert profile is not None
            assert float(profile.relationship_level) >= 0.4
            assert profile.total_interactions >= 2

            # Check topic evolution persistence
            evolutions = (
                await threading_manager.evolutions.get_thread_evolution_history(
                    initial_thread_id
                )
            )
            assert len(evolutions) > 0

    async def test_threading_across_multiple_sessions(
        self, async_session_factory, mock_memory_integration
    ):
        """Test threading continuity across multiple conversation sessions."""
        player_id = uuid.uuid4()
        npc_id = uuid.uuid4()

        threading_service = ConversationThreadingService(
            session_factory=async_session_factory,
            memory_integration=mock_memory_integration,
        )

        # Create NPC
        async with async_session_factory.get_session() as session:
            npc = NPCPersonality(
                npc_id=npc_id,
                traits={"talkative": 0.9, "memory_keeper": 0.8},
            )
            session.add(npc)
            await session.commit()

        session_topics = [
            ("quest introduction", "What quests are available?"),
            ("quest details", "Tell me more about the dragon quest"),
            ("preparation planning", "What should I bring for the quest?"),
            ("team formation", "Who else should join us?"),
            ("final preparations", "Are we ready to start?"),
        ]

        thread_id = None
        relationship_level = 0.1

        for i, (topic, message) in enumerate(session_topics):
            # Create conversation
            conversation = ConversationContext(
                conversation_id=str(uuid.uuid4()),
                player_id=str(player_id),
                npc_id=str(npc_id),
                topic=topic,
                relationship_level=relationship_level,
            )

            # Initiate session
            threading_context = await threading_service.initiate_conversation_session(
                player_id=player_id,
                npc_id=npc_id,
                conversation=conversation,
                initial_topic=topic,
            )

            # First session creates thread, subsequent sessions should continue it
            if thread_id is None:
                thread_id = threading_context.active_thread.thread_id
            else:
                assert threading_context.active_thread.thread_id == thread_id

            # Analyze threading - should improve over sessions
            analysis = await threading_service.analyze_threading_opportunity(
                conversation=conversation,
                personality=npc,
                player_message=message,
                current_topic=topic,
            )

            # As relationship develops, references should become more likely
            if i >= 2:  # After a few sessions
                assert analysis.should_reference_past is True
                assert len(analysis.suggested_references) > 0

            # Record evolution if not first session
            if i > 0:
                await threading_service.record_conversation_evolution(
                    conversation=conversation,
                    previous_topic=session_topics[i - 1][0],
                    new_topic=topic,
                    player_initiated=True,
                    evolution_quality="natural",
                )

            # Finalize session
            relationship_change = 0.05 + (i * 0.02)  # Increasing relationship
            await threading_service.finalize_conversation_session(
                conversation=conversation,
                session_success=True,
                relationship_change=relationship_change,
                memorable_moments=[f"Session {i+1}: {topic}"],
            )

            relationship_level += relationship_change

        # Verify final state
        async with async_session_factory.get_session() as session:
            threading_manager = ConversationThreadingManager(session)

            # Check thread has progressed
            thread = await threading_manager.threads.get_by_id(thread_id)
            assert thread.session_count == len(session_topics)
            assert len(thread.topic_evolution) >= len(session_topics) - 1

            # Check relationship progression
            profile = await threading_manager.profiles.get_profile(player_id, npc_id)
            assert float(profile.relationship_level) > 0.3
            assert len(profile.memorable_moments) == len(session_topics)

            # Check topic evolution quality
            evolutions = (
                await threading_manager.evolutions.get_thread_evolution_history(
                    thread_id
                )
            )
            assert len(evolutions) >= len(session_topics) - 1

            # Most evolutions should be natural/smooth
            quality_scores = [
                1.0 if evo.evolution_quality == "natural" else 0.5 for evo in evolutions
            ]
            avg_quality = (
                sum(quality_scores) / len(quality_scores) if quality_scores else 0
            )
            assert avg_quality >= 0.8

    async def test_conversation_preparation_integration(
        self, async_session_factory, mock_memory_integration
    ):
        """Test conversation preparation with real database data."""
        player_id = uuid.uuid4()
        npc_id = uuid.uuid4()

        threading_service = ConversationThreadingService(
            session_factory=async_session_factory,
            memory_integration=mock_memory_integration,
        )

        # Set up initial data
        async with async_session_factory.get_session() as session:
            # Create NPC
            npc = NPCPersonality(npc_id=npc_id, traits={"helpful": 0.8})
            session.add(npc)

            # Create thread with history
            thread = ConversationThread(
                player_id=player_id,
                npc_id=npc_id,
                primary_topic="quest planning",
                importance_score=0.8,
                session_count=3,
                next_conversation_hooks=[
                    "Ask about equipment status",
                    "Discuss team formation",
                    "Review quest timeline",
                ],
            )
            session.add(thread)

            # Create player profile
            profile = PlayerMemoryProfile(
                player_id=player_id,
                npc_id=npc_id,
                relationship_level=0.7,
                trust_level=0.6,
                conversation_style="casual",
                preferred_topics=["combat", "exploration", "strategy"],
            )
            session.add(profile)

            # Create topic evolutions
            for i, (source, target) in enumerate(
                [
                    ("general", "quest planning"),
                    ("quest planning", "team formation"),
                    ("team formation", "equipment needs"),
                ]
            ):
                evolution = TopicEvolution(
                    thread_id=thread.thread_id,
                    session_id=uuid.uuid4(),
                    source_topic=source,
                    target_topic=target,
                    transition_type="natural",
                    confidence_score=0.8,
                    evolution_quality="smooth",
                )
                session.add(evolution)

            await session.commit()

        # Get conversation preparation
        preparation = await threading_service.get_conversation_preparation(
            player_id=player_id,
            npc_id=npc_id,
            suggested_topics=["new quest opportunity"],
        )

        # Verify comprehensive preparation data
        assert "conversation_context" in preparation
        assert "threading_patterns" in preparation
        assert "suggested_openers" in preparation
        assert "relationship_notes" in preparation
        assert "topic_preferences" in preparation

        # Check conversation hooks
        context = preparation["conversation_context"]
        hooks = context["conversation_hooks"]
        assert len(hooks) > 0

        hook_texts = [hook["hook"] for hook in hooks]
        assert any("equipment status" in hook.lower() for hook in hook_texts)

        # Check threading patterns
        patterns = preparation["threading_patterns"]
        assert patterns["total_threads"] >= 1
        assert patterns["engagement_level"] in ["low", "medium", "high"]

        # Check suggested openers variety
        openers = preparation["suggested_openers"]
        assert len(openers) >= 3

        # Should have hook-based, topic-based, and relationship-based openers
        opener_text = " ".join(openers).lower()
        assert any(keyword in opener_text for keyword in ["equipment", "team", "quest"])

        # Check relationship notes
        relationship = preparation["relationship_notes"]
        assert relationship["level"] == 0.7
        assert relationship["trust"] == 0.6

        # Check topic preferences
        topic_prefs = preparation["topic_preferences"]
        assert len(topic_prefs) > 0

    async def test_threading_performance_and_memory_usage(
        self, async_session_factory, mock_memory_integration
    ):
        """Test threading system performance with multiple concurrent conversations."""
        import time
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        threading_service = ConversationThreadingService(
            session_factory=async_session_factory,
            memory_integration=mock_memory_integration,
        )

        # Create test NPCs
        npc_ids = []
        async with async_session_factory.get_session() as session:
            for i in range(5):
                npc = NPCPersonality(
                    npc_id=uuid.uuid4(),
                    traits={"talkative": 0.8, "helpful": 0.7},
                )
                session.add(npc)
                npc_ids.append(npc.npc_id)
            await session.commit()

        # Simulate multiple players having conversations
        player_ids = [uuid.uuid4() for _ in range(10)]

        start_time = time.time()

        # Each player has conversations with each NPC
        for player_id in player_ids:
            for npc_id in npc_ids:
                conversation = ConversationContext(
                    conversation_id=str(uuid.uuid4()),
                    player_id=str(player_id),
                    npc_id=str(npc_id),
                    topic="general conversation",
                    relationship_level=0.5,
                )

                # Initiate session
                threading_context = (
                    await threading_service.initiate_conversation_session(
                        player_id=player_id,
                        npc_id=npc_id,
                        conversation=conversation,
                        initial_topic="general conversation",
                    )
                )

                # Analyze threading
                npc = NPCPersonality(npc_id=npc_id, traits={"talkative": 0.8})
                analysis = await threading_service.analyze_threading_opportunity(
                    conversation=conversation,
                    personality=npc,
                    player_message="Hello there",
                    current_topic="general conversation",
                )

                # Finalize session
                await threading_service.finalize_conversation_session(
                    conversation=conversation,
                    session_success=True,
                    relationship_change=0.05,
                )

        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        total_conversations = len(player_ids) * len(npc_ids)
        processing_time = end_time - start_time
        memory_increase = final_memory - initial_memory

        # Performance assertions
        assert (
            processing_time < total_conversations * 0.1
        )  # Should be under 100ms per conversation
        assert memory_increase < 100  # Should not use more than 100MB additional memory

        # Verify all data was persisted
        async with async_session_factory.get_session() as session:
            threading_manager = ConversationThreadingManager(session)

            # Count total threads created
            all_threads = []
            for player_id in player_ids:
                for npc_id in npc_ids:
                    threads = await threading_manager.threads.get_player_threads(
                        player_id=player_id, npc_id=npc_id
                    )
                    all_threads.extend(threads)

            assert len(all_threads) == total_conversations

            # Verify threading context cache was properly cleaned up
            assert len(threading_service._threading_contexts) == 0
