#!/usr/bin/env python3
"""Test script for semantic memory database schema."""

import asyncio
import uuid

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.game_loop.database.models.conversation import (
    ConversationContext,
    ConversationExchange,
    EmotionalContext,
    MemoryEmbedding,
    MemoryPersonalityConfig,
    NPCPersonality,
)


async def test_semantic_memory_schema():
    """Test the semantic memory database schema."""
    # Database connection - using same config as game
    DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/game_loop"

    engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        pool_size=10,
        max_overflow=20,
    )

    async_session = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with async_session() as session:
            print("🔄 Testing semantic memory schema...")

            # Test 1: Create NPC Personality with Memory Config
            print("1. Creating NPC personality with memory configuration...")
            npc_id = uuid.uuid4()

            npc_personality = NPCPersonality(
                npc_id=npc_id,
                traits={"detail-oriented": 0.8, "friendly": 0.7},
                knowledge_areas=["castle_layout", "security"],
                background_story="A veteran guard with excellent memory for details",
                default_mood="professional",
            )
            session.add(npc_personality)

            # Create memory personality config
            memory_config = MemoryPersonalityConfig(
                npc_id=npc_id,
                decay_rate_modifier=0.7,  # Slower decay (better memory)
                emotional_sensitivity=1.2,  # More sensitive to emotions
                detail_retention_strength=0.9,  # Excellent at details
                name_retention_strength=0.4,  # Not as good with names
                uncertainty_threshold=0.25,  # Lower threshold for uncertainty
            )
            session.add(memory_config)

            await session.commit()
            print("✅ NPC and memory config created")

            # Test 2: Create Conversation Context
            print("2. Creating conversation context...")
            player_id = uuid.uuid4()
            conversation = ConversationContext(
                player_id=player_id,
                npc_id=npc_id,
                topic="Security protocols",
                mood="serious",
                relationship_level=0.3,
            )
            session.add(conversation)
            await session.commit()
            print("✅ Conversation context created")

            # Test 3: Create Conversation Exchange with Semantic Memory Fields
            print("3. Creating conversation exchange with semantic memory...")
            exchange = ConversationExchange(
                conversation_id=conversation.conversation_id,
                speaker_id=player_id,
                message_text="What are the security protocols for the east wing?",
                message_type="question",
                emotion="concerned",
                confidence_score=1.0,  # New memory, full confidence
                emotional_weight=0.7,  # Moderately emotional (security concern)
                trust_level_required=0.2,  # Low security question
                access_count=0,
            )
            session.add(exchange)
            await session.flush()  # Get exchange_id
            print("✅ Exchange with semantic memory fields created")

            # Test 4: Create Emotional Context
            print("4. Adding emotional context...")
            emotional_context = EmotionalContext(
                exchange_id=exchange.exchange_id,
                sentiment_score=0.1,  # Slightly positive
                emotional_keywords=["protocols", "security", "concerned"],
                participant_emotions={"player": "concerned", "npc": "professional"},
                emotional_intensity=0.7,
                relationship_impact_score=0.2,
            )
            session.add(emotional_context)
            print("✅ Emotional context created")

            # Test 5: Create Memory Embedding (simulated)
            print("5. Adding memory embedding...")
            # Simulate a 384-dimensional embedding
            fake_embedding = [0.1] * 384  # Simple test embedding

            memory_embedding = MemoryEmbedding(
                exchange_id=exchange.exchange_id,
                embedding=fake_embedding,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                embedding_metadata={"confidence": 0.95, "source": "test"},
            )
            session.add(memory_embedding)
            print("✅ Memory embedding created")

            await session.commit()

            # Test 6: Query the enhanced data
            print("6. Querying semantic memory data...")

            # Query conversation with semantic memory fields
            result = await session.get(ConversationExchange, exchange.exchange_id)
            print(f"   Exchange confidence: {result.confidence_score}")
            print(f"   Emotional weight: {result.emotional_weight}")
            print(f"   Trust level required: {result.trust_level_required}")
            print(f"   Access count: {result.access_count}")
            print(f"   Has embedding: {result.memory_embedding is not None}")

            # Query memory config
            from sqlalchemy import select

            stmt = select(MemoryPersonalityConfig).where(
                MemoryPersonalityConfig.npc_id == npc_id
            )
            config_result = await session.execute(stmt)
            config = config_result.scalar_one()
            print(f"   NPC decay rate modifier: {config.decay_rate_modifier}")
            print(f"   Detail retention strength: {config.detail_retention_strength}")

            print("\n🎉 All semantic memory schema tests passed!")
            return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False
    finally:
        await engine.dispose()


if __name__ == "__main__":
    success = asyncio.run(test_semantic_memory_schema())
    if success:
        print("\n✨ Semantic memory database schema is working correctly!")
    else:
        print("\n💥 Schema test failed - check errors above")
        exit(1)
