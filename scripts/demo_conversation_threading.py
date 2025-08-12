#!/usr/bin/env python3
"""
Demonstration script for conversation threading system.

This script shows how NPCs maintain conversation threads and topic continuity
across multiple game sessions, ensuring natural relationship building.
"""

import asyncio
import uuid
from datetime import datetime, timezone

from game_loop.core.conversation.conversation_threading import (
    ConversationThreadingService,
)
from game_loop.core.conversation.conversation_models import (
    ConversationContext,
    NPCPersonality,
)
from game_loop.database.session_factory import DatabaseSessionFactory
from game_loop.llm.config import LLMConfig


class MockMemoryIntegration:
    """Mock memory integration for demonstration."""

    async def extract_memory_context(self, conversation, player_message, personality):
        """Extract memory context."""
        return type(
            "MemoryContext",
            (),
            {
                "current_topic": conversation.topic,
                "emotional_tone": conversation.mood,
                "query_context": "conversational",
            },
        )()

    async def retrieve_relevant_memories(self, memory_context, npc_id):
        """Retrieve relevant memories."""
        return type(
            "MemoryResult",
            (),
            {
                "relevant_memories": [],
                "confidence_scores": [],
            },
        )()


async def demonstrate_conversation_threading():
    """Demonstrate the conversation threading system."""
    print("🧠 Conversation Threading System Demonstration")
    print("=" * 50)

    # Initialize components
    llm_config = LLMConfig.from_yaml("tests/configs/test_ollama_config.yaml")
    session_factory = DatabaseSessionFactory.create_from_config(llm_config)
    memory_integration = MockMemoryIntegration()

    # Create threading service
    threading_service = ConversationThreadingService(
        session_factory=session_factory,
        memory_integration=memory_integration,
        enable_threading=True,
    )

    # Create test entities
    player_id = uuid.uuid4()
    npc_id = uuid.uuid4()

    # Create NPC personality
    mentor_npc = NPCPersonality(
        npc_id=str(npc_id),
        traits={
            "wise": 0.8,
            "helpful": 0.9,
            "patient": 0.7,
            "talkative": 0.6,
            "memory_keeper": 0.8,
        },
        knowledge_areas=["magic", "history", "combat", "exploration"],
        speech_patterns={"formal": 0.7, "teaching": 0.8},
        relationships={},
        background_story="An ancient wizard who mentors young adventurers",
        default_mood="wise",
    )

    # Simulate multiple conversation sessions
    conversation_sessions = [
        {
            "topic": "magic basics",
            "message": "I want to learn magic. Where should I start?",
            "response": "Magic requires patience and practice. Let's begin with understanding magical theory.",
        },
        {
            "topic": "spell casting",
            "message": "How do I cast my first spell?",
            "response": "Focus your mind and channel your energy. Start with a simple light spell.",
        },
        {
            "topic": "advanced techniques",
            "message": "I've mastered the light spell. What's next?",
            "response": "Excellent progress! Now we can explore elemental magic.",
        },
        {
            "topic": "magical theory",
            "message": "Why is understanding theory important for magic?",
            "response": "Theory provides the foundation for safe and powerful spellcasting.",
        },
        {
            "topic": "final mastery",
            "message": "I feel ready for more advanced magic now.",
            "response": "Your dedication has been remarkable. Let's unlock your true potential.",
        },
    ]

    print(f"👤 Player: {player_id}")
    print(f"🧙 NPC: {mentor_npc.background_story}")
    print()

    current_thread_id = None
    relationship_level = 0.1

    for i, session in enumerate(conversation_sessions, 1):
        print(f"📖 Session {i}: {session['topic'].title()}")
        print("-" * 30)

        # Create conversation context
        conversation = ConversationContext(
            conversation_id=str(uuid.uuid4()),
            player_id=str(player_id),
            npc_id=str(npc_id),
            topic=session["topic"],
            mood="curious" if i <= 2 else "confident" if i <= 4 else "determined",
            relationship_level=relationship_level,
        )

        # Initiate conversation session
        print(f"🔄 Initiating session (Relationship: {relationship_level:.1f})...")
        threading_context = await threading_service.initiate_conversation_session(
            player_id=player_id,
            npc_id=npc_id,
            conversation=conversation,
            initial_topic=session["topic"],
        )

        if current_thread_id is None:
            current_thread_id = threading_context.active_thread.thread_id
            print(f"📝 Created new conversation thread: {current_thread_id}")
        else:
            assert threading_context.active_thread.thread_id == current_thread_id
            print(f"🔗 Continued existing thread: {current_thread_id}")

        print(
            f"📊 Topic continuity score: {threading_context.topic_continuity_score:.2f}"
        )

        # Analyze threading opportunity
        analysis = await threading_service.analyze_threading_opportunity(
            conversation=conversation,
            personality=mentor_npc,
            player_message=session["message"],
            current_topic=session["topic"],
        )

        print(f"💭 Should reference past: {analysis.should_reference_past}")
        print(f"🎯 Reference confidence: {analysis.reference_confidence:.2f}")

        # Enhance response with threading
        enhanced_response, threading_data = (
            await threading_service.enhance_response_with_threading(
                base_response=session["response"],
                threading_analysis=analysis,
                conversation=conversation,
                personality=mentor_npc,
                current_topic=session["topic"],
            )
        )

        print(f"💬 Player: \"{session['message']}\"")
        print(f'🧙 NPC: "{enhanced_response}"')

        if threading_data["threading_enhanced"]:
            print(f"✨ Enhanced with {threading_data['references_added']} references")
            for ref in threading_data["references"]:
                print(f"   - {ref['reference']} (style: {ref['integration_style']})")

        # Record topic evolution if not first session
        if i > 1:
            previous_topic = conversation_sessions[i - 2]["topic"]
            await threading_service.record_conversation_evolution(
                conversation=conversation,
                previous_topic=previous_topic,
                new_topic=session["topic"],
                player_initiated=True,
                evolution_quality="natural",
            )
            print(f"📈 Recorded evolution: {previous_topic} → {session['topic']}")

        # Finalize session
        relationship_change = 0.05 + (i * 0.02)  # Increasing relationship
        await threading_service.finalize_conversation_session(
            conversation=conversation,
            session_success=True,
            relationship_change=relationship_change,
            memorable_moments=[f"Session {i}: Learned about {session['topic']}"],
        )

        relationship_level += relationship_change
        print(
            f"🤝 Relationship increased by {relationship_change:.2f} to {relationship_level:.2f}"
        )
        print()

    # Demonstrate conversation preparation
    print("🎯 Getting Conversation Preparation for Next Session")
    print("-" * 50)

    preparation = await threading_service.get_conversation_preparation(
        player_id=player_id,
        npc_id=npc_id,
        suggested_topics=["mastery test", "graduation ceremony"],
    )

    print("📋 Suggested conversation openers:")
    for opener in preparation["suggested_openers"]:
        print(f'   • "{opener}"')

    print(f"\n🎭 Relationship summary:")
    relationship = preparation["relationship_notes"]
    print(f"   • Level: {relationship['level']:.2f}")
    print(f"   • Trust: {relationship['trust']:.2f}")
    print(f"   • Style: {relationship['style']}")

    print(f"\n📚 Preferred topics:")
    for topic in preparation["topic_preferences"]:
        if isinstance(topic, dict):
            print(f"   • {topic['topic']} (frequency: {topic['frequency']})")
        else:
            print(f"   • {topic}")

    print("\n🎉 Conversation Threading Demonstration Complete!")
    print("Key Features Demonstrated:")
    print("✓ Persistent conversation threads across sessions")
    print("✓ Topic continuity and evolution tracking")
    print("✓ Relationship progression and memory building")
    print("✓ Natural reference integration based on relationship level")
    print("✓ Conversation preparation for future interactions")


if __name__ == "__main__":
    asyncio.run(demonstrate_conversation_threading())
