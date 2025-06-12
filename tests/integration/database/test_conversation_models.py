"""Integration tests for conversation database models."""

import uuid

import pytest
from sqlalchemy.exc import IntegrityError

from game_loop.database.models.conversation import (
    ConversationContext,
    ConversationExchange,
    ConversationKnowledge,
    NPCPersonality,
)


@pytest.mark.asyncio
class TestNPCPersonalityModel:
    """Test NPCPersonality database model."""

    async def test_create_npc_personality(self, db_session):
        """Test creating an NPC personality."""
        npc_id = uuid.uuid4()
        personality = NPCPersonality(
            npc_id=npc_id,
            traits={"friendly": 0.8, "helpful": 0.9},
            knowledge_areas=["history", "local_lore"],
            speech_patterns={"formality": "medium"},
            relationships={"player_123": 0.5},
            background_story="A wise village elder.",
            default_mood="contemplative",
        )

        db_session.add(personality)
        await db_session.commit()

        # Verify it was saved
        result = await db_session.get(NPCPersonality, npc_id)
        assert result is not None
        assert result.npc_id == npc_id
        assert result.traits == {"friendly": 0.8, "helpful": 0.9}
        assert result.knowledge_areas == ["history", "local_lore"]
        assert result.background_story == "A wise village elder."

    async def test_npc_personality_trait_methods(self, db_session):
        """Test NPC personality trait helper methods."""
        npc_id = uuid.uuid4()
        personality = NPCPersonality(
            npc_id=npc_id,
            traits={"friendly": 0.7, "wise": 0.9},
            relationships={"player_1": 0.3},
        )

        db_session.add(personality)
        await db_session.commit()

        # Test trait strength
        assert personality.get_trait_strength("friendly") == 0.7
        assert personality.get_trait_strength("nonexistent") == 0.0

        # Test relationship level
        assert personality.get_relationship_level("player_1") == 0.3
        assert personality.get_relationship_level("unknown_player") == 0.0

        # Test updating relationship
        personality.update_relationship("player_1", 0.2)
        assert personality.get_relationship_level("player_1") == 0.5

        # Test relationship bounds
        personality.update_relationship("player_1", 1.0)  # Should cap at 1.0
        assert personality.get_relationship_level("player_1") == 1.0

        personality.update_relationship("player_1", -2.5)  # Should cap at -1.0
        assert personality.get_relationship_level("player_1") == -1.0

    async def test_npc_personality_to_dict(self, db_session):
        """Test NPC personality serialization."""
        npc_id = uuid.uuid4()
        personality = NPCPersonality(npc_id=npc_id, background_story="Test NPC")

        db_session.add(personality)
        await db_session.commit()

        data = personality.to_dict()
        assert data["npc_id"] == str(npc_id)
        assert data["background_story"] == "Test NPC"
        assert "created_at" in data
        assert "updated_at" in data


@pytest.mark.asyncio
class TestConversationContextModel:
    """Test ConversationContext database model."""

    async def test_create_conversation_context(self, db_session):
        """Test creating a conversation context."""
        # First create an NPC personality
        npc_id = uuid.uuid4()
        personality = NPCPersonality(npc_id=npc_id, background_story="Test guard")
        db_session.add(personality)

        # Create conversation context
        player_id = uuid.uuid4()
        conversation = ConversationContext(
            player_id=player_id,
            npc_id=npc_id,
            topic="Security matters",
            mood="professional",
            relationship_level=0.2,
            context_data={"location": "castle_gate"},
        )

        db_session.add(conversation)
        await db_session.commit()

        # Verify it was saved
        result = await db_session.get(ConversationContext, conversation.conversation_id)
        assert result is not None
        assert result.player_id == player_id
        assert result.npc_id == npc_id
        assert result.topic == "Security matters"
        assert result.mood == "professional"
        assert float(result.relationship_level) == 0.2
        assert result.context_data == {"location": "castle_gate"}

    async def test_conversation_context_methods(self, db_session):
        """Test conversation context helper methods."""
        npc_id = uuid.uuid4()
        personality = NPCPersonality(npc_id=npc_id)
        db_session.add(personality)

        conversation = ConversationContext(
            player_id=uuid.uuid4(),
            npc_id=npc_id,
            relationship_level=0.3,
        )
        db_session.add(conversation)
        await db_session.commit()

        # Test mood update
        conversation.update_mood("pleased")
        assert conversation.mood == "pleased"

        # Test relationship update
        conversation.update_relationship(0.2)
        assert float(conversation.relationship_level) == 0.5

        # Test bounds
        conversation.update_relationship(1.0)  # Should cap at 1.0
        assert float(conversation.relationship_level) == 1.0

        # Test ending conversation
        conversation.end_conversation("player_left")
        assert conversation.status == "ended"
        assert conversation.ended_at is not None
        assert conversation.context_data["end_reason"] == "player_left"

    async def test_conversation_relationship_constraint(self, db_session):
        """Test relationship level constraint."""
        npc_id = uuid.uuid4()
        personality = NPCPersonality(npc_id=npc_id)
        db_session.add(personality)

        # Test valid relationship level
        conversation = ConversationContext(
            player_id=uuid.uuid4(),
            npc_id=npc_id,
            relationship_level=0.5,
        )
        db_session.add(conversation)
        await db_session.commit()  # Should succeed

        # Test invalid relationship level
        with pytest.raises(IntegrityError):
            bad_conversation = ConversationContext(
                player_id=uuid.uuid4(),
                npc_id=npc_id,
                relationship_level=1.5,  # Invalid: > 1.0
            )
            db_session.add(bad_conversation)
            await db_session.commit()

    async def test_conversation_foreign_key_constraint(self, db_session):
        """Test foreign key constraint for NPC ID."""
        # Try to create conversation without NPC personality
        with pytest.raises(IntegrityError):
            conversation = ConversationContext(
                player_id=uuid.uuid4(),
                npc_id=uuid.uuid4(),  # Non-existent NPC
            )
            db_session.add(conversation)
            await db_session.commit()


@pytest.mark.asyncio
class TestConversationExchangeModel:
    """Test ConversationExchange database model."""

    async def test_create_conversation_exchange(self, db_session):
        """Test creating a conversation exchange."""
        # Setup conversation
        npc_id = uuid.uuid4()
        personality = NPCPersonality(npc_id=npc_id)
        db_session.add(personality)

        conversation = ConversationContext(
            player_id=uuid.uuid4(),
            npc_id=npc_id,
        )
        db_session.add(conversation)
        await db_session.commit()

        # Create exchange
        exchange = ConversationExchange(
            conversation_id=conversation.conversation_id,
            speaker_id=uuid.uuid4(),
            message_text="Hello there!",
            message_type="greeting",
            emotion="friendly",
            exchange_metadata={"intent": "greeting"},
        )

        db_session.add(exchange)
        await db_session.commit()

        # Verify it was saved
        result = await db_session.get(ConversationExchange, exchange.exchange_id)
        assert result is not None
        assert result.conversation_id == conversation.conversation_id
        assert result.message_text == "Hello there!"
        assert result.message_type == "greeting"
        assert result.emotion == "friendly"
        assert result.exchange_metadata == {"intent": "greeting"}

    async def test_exchange_cascade_delete(self, db_session):
        """Test that exchanges are deleted when conversation is deleted."""
        # Setup conversation
        npc_id = uuid.uuid4()
        personality = NPCPersonality(npc_id=npc_id)
        db_session.add(personality)

        conversation = ConversationContext(
            player_id=uuid.uuid4(),
            npc_id=npc_id,
        )
        db_session.add(conversation)
        await db_session.commit()

        # Create exchanges
        exchange1 = ConversationExchange(
            conversation_id=conversation.conversation_id,
            speaker_id=uuid.uuid4(),
            message_text="Hello",
            message_type="greeting",
        )
        exchange2 = ConversationExchange(
            conversation_id=conversation.conversation_id,
            speaker_id=uuid.uuid4(),
            message_text="How are you?",
            message_type="question",
        )

        db_session.add_all([exchange1, exchange2])
        await db_session.commit()

        exchange1_id = exchange1.exchange_id
        exchange2_id = exchange2.exchange_id

        # Delete conversation
        await db_session.delete(conversation)
        await db_session.commit()

        # Verify exchanges were deleted
        result1 = await db_session.get(ConversationExchange, exchange1_id)
        result2 = await db_session.get(ConversationExchange, exchange2_id)
        assert result1 is None
        assert result2 is None

    async def test_exchange_message_type_constraint(self, db_session):
        """Test message type constraint."""
        # Setup conversation
        npc_id = uuid.uuid4()
        personality = NPCPersonality(npc_id=npc_id)
        db_session.add(personality)

        conversation = ConversationContext(
            player_id=uuid.uuid4(),
            npc_id=npc_id,
        )
        db_session.add(conversation)
        await db_session.commit()

        # Test valid message type
        exchange = ConversationExchange(
            conversation_id=conversation.conversation_id,
            speaker_id=uuid.uuid4(),
            message_text="Hello",
            message_type="greeting",
        )
        db_session.add(exchange)
        await db_session.commit()  # Should succeed

        # Test invalid message type
        with pytest.raises(IntegrityError):
            bad_exchange = ConversationExchange(
                conversation_id=conversation.conversation_id,
                speaker_id=uuid.uuid4(),
                message_text="Hello",
                message_type="invalid_type",
            )
            db_session.add(bad_exchange)
            await db_session.commit()


@pytest.mark.asyncio
class TestConversationKnowledgeModel:
    """Test ConversationKnowledge database model."""

    async def test_create_knowledge_entry(self, db_session):
        """Test creating a knowledge entry."""
        # Setup conversation and exchange
        npc_id = uuid.uuid4()
        personality = NPCPersonality(npc_id=npc_id)
        db_session.add(personality)

        conversation = ConversationContext(
            player_id=uuid.uuid4(),
            npc_id=npc_id,
        )
        db_session.add(conversation)
        await db_session.flush()  # Ensure conversation_id is generated

        exchange = ConversationExchange(
            conversation_id=conversation.conversation_id,
            speaker_id=uuid.uuid4(),
            message_text="The dragon lives in the northern mountains.",
            message_type="statement",
        )
        db_session.add(exchange)
        await db_session.commit()

        # Create knowledge entry
        knowledge = ConversationKnowledge(
            conversation_id=conversation.conversation_id,
            information_type="location",
            extracted_info={
                "entity": "dragon",
                "location": "northern mountains",
                "confidence": 0.9,
            },
            confidence_score=0.9,
            source_exchange_id=exchange.exchange_id,
        )

        db_session.add(knowledge)
        await db_session.commit()

        # Verify it was saved
        result = await db_session.get(ConversationKnowledge, knowledge.knowledge_id)
        assert result is not None
        assert result.conversation_id == conversation.conversation_id
        assert result.information_type == "location"
        assert result.extracted_info["entity"] == "dragon"
        assert float(result.confidence_score) == 0.9
        assert result.source_exchange_id == exchange.exchange_id

    async def test_knowledge_confidence_constraint(self, db_session):
        """Test confidence score constraint."""
        # Setup conversation
        npc_id = uuid.uuid4()
        personality = NPCPersonality(npc_id=npc_id)
        db_session.add(personality)

        conversation = ConversationContext(
            player_id=uuid.uuid4(),
            npc_id=npc_id,
        )
        db_session.add(conversation)
        await db_session.commit()

        # Test valid confidence score
        knowledge = ConversationKnowledge(
            conversation_id=conversation.conversation_id,
            information_type="test",
            extracted_info={"test": "data"},
            confidence_score=0.5,
        )
        db_session.add(knowledge)
        await db_session.commit()  # Should succeed

        # Test invalid confidence score
        with pytest.raises(IntegrityError):
            bad_knowledge = ConversationKnowledge(
                conversation_id=conversation.conversation_id,
                information_type="test",
                extracted_info={"test": "data"},
                confidence_score=1.5,  # Invalid: > 1.0
            )
            db_session.add(bad_knowledge)
            await db_session.commit()

    async def test_knowledge_cascade_delete(self, db_session):
        """Test that knowledge entries are deleted when conversation is deleted."""
        # Setup conversation
        npc_id = uuid.uuid4()
        personality = NPCPersonality(npc_id=npc_id)
        db_session.add(personality)

        conversation = ConversationContext(
            player_id=uuid.uuid4(),
            npc_id=npc_id,
        )
        db_session.add(conversation)
        await db_session.commit()

        # Create knowledge entries
        knowledge1 = ConversationKnowledge(
            conversation_id=conversation.conversation_id,
            information_type="fact",
            extracted_info={"fact": "test1"},
        )
        knowledge2 = ConversationKnowledge(
            conversation_id=conversation.conversation_id,
            information_type="fact",
            extracted_info={"fact": "test2"},
        )

        db_session.add_all([knowledge1, knowledge2])
        await db_session.commit()

        knowledge1_id = knowledge1.knowledge_id
        knowledge2_id = knowledge2.knowledge_id

        # Delete conversation
        await db_session.delete(conversation)
        await db_session.commit()

        # Verify knowledge entries were deleted
        result1 = await db_session.get(ConversationKnowledge, knowledge1_id)
        result2 = await db_session.get(ConversationKnowledge, knowledge2_id)
        assert result1 is None
        assert result2 is None


@pytest.mark.asyncio
class TestConversationRelationships:
    """Test relationships between conversation models."""

    async def test_conversation_personality_relationship(self, db_session):
        """Test relationship between conversation and NPC personality."""
        npc_id = uuid.uuid4()
        personality = NPCPersonality(
            npc_id=npc_id,
            background_story="Test NPC",
        )
        db_session.add(personality)

        conversation = ConversationContext(
            player_id=uuid.uuid4(),
            npc_id=npc_id,
        )
        db_session.add(conversation)
        await db_session.commit()

        # Test accessing personality from conversation
        result = await db_session.get(ConversationContext, conversation.conversation_id)
        await db_session.refresh(result, ["npc_personality"])
        assert result.npc_personality is not None
        assert result.npc_personality.background_story == "Test NPC"

    async def test_conversation_exchanges_relationship(self, db_session):
        """Test relationship between conversation and exchanges."""
        npc_id = uuid.uuid4()
        personality = NPCPersonality(npc_id=npc_id)
        db_session.add(personality)

        conversation = ConversationContext(
            player_id=uuid.uuid4(),
            npc_id=npc_id,
        )
        db_session.add(conversation)
        await db_session.commit()

        # Add multiple exchanges
        exchanges = [
            ConversationExchange(
                conversation_id=conversation.conversation_id,
                speaker_id=uuid.uuid4(),
                message_text=f"Message {i}",
                message_type="statement",
            )
            for i in range(3)
        ]

        db_session.add_all(exchanges)
        await db_session.commit()

        # Test accessing exchanges from conversation
        result = await db_session.get(ConversationContext, conversation.conversation_id)
        await db_session.refresh(result, ["exchanges"])
        assert len(result.exchanges) == 3
        assert result.exchanges[0].message_text == "Message 0"
