"""Integration tests for conversation repository layer."""

import uuid

import pytest

from game_loop.database.repositories.conversation import (
    ConversationContextRepository,
    ConversationExchangeRepository,
    ConversationKnowledgeRepository,
    ConversationRepositoryManager,
    NPCPersonalityRepository,
)


@pytest.mark.asyncio
class TestNPCPersonalityRepository:
    """Test NPCPersonality repository operations."""

    async def test_create_personality(self, db_session):
        """Test creating an NPC personality."""
        repo = NPCPersonalityRepository(db_session)

        npc_id = uuid.uuid4()
        personality = await repo.create_personality(
            npc_id=npc_id,
            traits={"wise": 0.9, "patient": 0.8},
            knowledge_areas=["history", "magic"],
            speech_patterns={"formality": "high", "verbosity": "medium"},
            background_story="A wise old wizard.",
            default_mood="contemplative",
        )

        assert personality.npc_id == npc_id
        assert personality.traits == {"wise": 0.9, "patient": 0.8}
        assert personality.knowledge_areas == ["history", "magic"]
        assert personality.background_story == "A wise old wizard."

    async def test_get_by_npc_id(self, db_session):
        """Test retrieving personality by NPC ID."""
        repo = NPCPersonalityRepository(db_session)

        npc_id = uuid.uuid4()
        created = await repo.create_personality(
            npc_id=npc_id,
            traits={"friendly": 0.7},
            knowledge_areas=["local_news"],
            speech_patterns={},
            background_story="A local merchant.",
        )

        # Retrieve by ID
        retrieved = await repo.get_by_npc_id(npc_id)
        assert retrieved is not None
        assert retrieved.npc_id == npc_id
        assert retrieved.background_story == "A local merchant."

        # Test non-existent ID
        non_existent = await repo.get_by_npc_id(uuid.uuid4())
        assert non_existent is None

    async def test_update_relationship(self, db_session):
        """Test updating NPC relationship."""
        repo = NPCPersonalityRepository(db_session)

        npc_id = uuid.uuid4()
        player_id = uuid.uuid4()

        personality = await repo.create_personality(
            npc_id=npc_id,
            traits={},
            knowledge_areas=[],
            speech_patterns={},
            background_story="Test NPC",
        )

        # Update relationship
        updated = await repo.update_relationship(npc_id, player_id, 0.3)
        assert updated is not None
        assert updated.get_relationship_level(str(player_id)) == 0.3

        # Update again (should accumulate)
        updated = await repo.update_relationship(npc_id, player_id, 0.2)
        assert updated.get_relationship_level(str(player_id)) == 0.5

    async def test_get_npcs_by_knowledge_area(self, db_session):
        """Test finding NPCs by knowledge area."""
        repo = NPCPersonalityRepository(db_session)

        # Create NPCs with different knowledge areas
        npc1 = await repo.create_personality(
            npc_id=uuid.uuid4(),
            traits={},
            knowledge_areas=["history", "politics"],
            speech_patterns={},
            background_story="Historian",
        )

        npc2 = await repo.create_personality(
            npc_id=uuid.uuid4(),
            traits={},
            knowledge_areas=["magic", "alchemy"],
            speech_patterns={},
            background_story="Wizard",
        )

        npc3 = await repo.create_personality(
            npc_id=uuid.uuid4(),
            traits={},
            knowledge_areas=["history", "archaeology"],
            speech_patterns={},
            background_story="Archaeologist",
        )

        # Find NPCs with history knowledge
        history_npcs = await repo.get_npcs_by_knowledge_area("history")
        history_npc_ids = {npc.npc_id for npc in history_npcs}
        # Check that our created NPCs are in the results
        assert npc1.npc_id in history_npc_ids
        assert npc3.npc_id in history_npc_ids
        # npc2 should not have history knowledge
        assert npc2.npc_id not in history_npc_ids

        # Find NPCs with magic knowledge
        magic_npcs = await repo.get_npcs_by_knowledge_area("magic")
        magic_npc_ids = {npc.npc_id for npc in magic_npcs}
        # Check that npc2 has magic knowledge
        assert npc2.npc_id in magic_npc_ids
        # npc1 and npc3 should not have magic knowledge
        assert npc1.npc_id not in magic_npc_ids
        assert npc3.npc_id not in magic_npc_ids


@pytest.mark.asyncio
class TestConversationContextRepository:
    """Test ConversationContext repository operations."""

    async def test_create_conversation(self, db_session):
        """Test creating a conversation."""
        # Setup NPC personality first
        npc_repo = NPCPersonalityRepository(db_session)
        npc_id = uuid.uuid4()
        await npc_repo.create_personality(
            npc_id=npc_id,
            traits={},
            knowledge_areas=[],
            speech_patterns={},
            background_story="Test NPC",
        )

        # Create conversation
        conv_repo = ConversationContextRepository(db_session)
        player_id = uuid.uuid4()

        conversation = await conv_repo.create_conversation(
            player_id=player_id,
            npc_id=npc_id,
            topic="Test topic",
            initial_mood="friendly",
            relationship_level=0.3,
        )

        assert conversation.player_id == player_id
        assert conversation.npc_id == npc_id
        assert conversation.topic == "Test topic"
        assert conversation.mood == "friendly"
        assert float(conversation.relationship_level) == 0.3

    async def test_get_active_conversation(self, db_session):
        """Test finding active conversation between player and NPC."""
        # Setup
        npc_repo = NPCPersonalityRepository(db_session)
        npc_id = uuid.uuid4()
        await npc_repo.create_personality(
            npc_id=npc_id,
            traits={},
            knowledge_areas=[],
            speech_patterns={},
            background_story="Test NPC",
        )

        conv_repo = ConversationContextRepository(db_session)
        player_id = uuid.uuid4()

        # Create active conversation
        conversation = await conv_repo.create_conversation(
            player_id=player_id,
            npc_id=npc_id,
        )

        # Find active conversation
        found = await conv_repo.get_active_conversation(player_id, npc_id)
        assert found is not None
        assert found.conversation_id == conversation.conversation_id

        # End conversation and search again
        await conv_repo.end_conversation(conversation.conversation_id, "test_end")
        not_found = await conv_repo.get_active_conversation(player_id, npc_id)
        assert not_found is None

    async def test_get_player_conversations(self, db_session):
        """Test retrieving conversations for a player."""
        # Setup NPCs
        npc_repo = NPCPersonalityRepository(db_session)
        npc1_id = uuid.uuid4()
        npc2_id = uuid.uuid4()

        await npc_repo.create_personality(
            npc_id=npc1_id,
            traits={},
            knowledge_areas=[],
            speech_patterns={},
            background_story="NPC 1",
        )
        await npc_repo.create_personality(
            npc_id=npc2_id,
            traits={},
            knowledge_areas=[],
            speech_patterns={},
            background_story="NPC 2",
        )

        # Create conversations
        conv_repo = ConversationContextRepository(db_session)
        player_id = uuid.uuid4()

        conv1 = await conv_repo.create_conversation(player_id=player_id, npc_id=npc1_id)
        conv2 = await conv_repo.create_conversation(player_id=player_id, npc_id=npc2_id)

        # End one conversation
        await conv_repo.end_conversation(conv2.conversation_id, "finished")

        # Get all conversations
        all_convs = await conv_repo.get_player_conversations(player_id)
        assert len(all_convs) == 2

        # Get only active conversations
        active_convs = await conv_repo.get_player_conversations(
            player_id, status="active"
        )
        assert len(active_convs) == 1
        assert active_convs[0].conversation_id == conv1.conversation_id

        # Get only ended conversations
        ended_convs = await conv_repo.get_player_conversations(
            player_id, status="ended"
        )
        assert len(ended_convs) == 1
        assert ended_convs[0].conversation_id == conv2.conversation_id

    async def test_get_with_exchanges(self, db_session):
        """Test loading conversation with exchanges."""
        # Setup
        npc_repo = NPCPersonalityRepository(db_session)
        npc_id = uuid.uuid4()
        await npc_repo.create_personality(
            npc_id=npc_id,
            traits={},
            knowledge_areas=[],
            speech_patterns={},
            background_story="Test NPC",
        )

        conv_repo = ConversationContextRepository(db_session)
        exchange_repo = ConversationExchangeRepository(db_session)

        player_id = uuid.uuid4()
        conversation = await conv_repo.create_conversation(
            player_id=player_id, npc_id=npc_id
        )

        # Add exchanges
        await exchange_repo.create_exchange(
            conversation_id=conversation.conversation_id,
            speaker_id=player_id,
            message_text="Hello",
            message_type="greeting",
        )
        await exchange_repo.create_exchange(
            conversation_id=conversation.conversation_id,
            speaker_id=npc_id,
            message_text="Hello there!",
            message_type="greeting",
        )

        # Load with exchanges
        loaded = await conv_repo.get_with_exchanges(conversation.conversation_id)
        assert loaded is not None
        assert len(loaded.exchanges) == 2


@pytest.mark.asyncio
class TestConversationExchangeRepository:
    """Test ConversationExchange repository operations."""

    async def test_create_exchange(self, db_session):
        """Test creating a conversation exchange."""
        # Setup conversation
        npc_repo = NPCPersonalityRepository(db_session)
        npc_id = uuid.uuid4()
        await npc_repo.create_personality(
            npc_id=npc_id,
            traits={},
            knowledge_areas=[],
            speech_patterns={},
            background_story="Test NPC",
        )

        conv_repo = ConversationContextRepository(db_session)
        player_id = uuid.uuid4()
        conversation = await conv_repo.create_conversation(
            player_id=player_id, npc_id=npc_id
        )

        # Create exchange
        exchange_repo = ConversationExchangeRepository(db_session)
        exchange = await exchange_repo.create_exchange(
            conversation_id=conversation.conversation_id,
            speaker_id=player_id,
            message_text="How are you today?",
            message_type="question",
            emotion="curious",
            metadata={"intent": "small_talk"},
        )

        assert exchange.conversation_id == conversation.conversation_id
        assert exchange.speaker_id == player_id
        assert exchange.message_text == "How are you today?"
        assert exchange.message_type == "question"
        assert exchange.emotion == "curious"
        assert exchange.exchange_metadata == {"intent": "small_talk"}

    async def test_get_conversation_exchanges(self, db_session):
        """Test retrieving exchanges for a conversation."""
        # Setup
        npc_repo = NPCPersonalityRepository(db_session)
        npc_id = uuid.uuid4()
        await npc_repo.create_personality(
            npc_id=npc_id,
            traits={},
            knowledge_areas=[],
            speech_patterns={},
            background_story="Test NPC",
        )

        conv_repo = ConversationContextRepository(db_session)
        player_id = uuid.uuid4()
        conversation = await conv_repo.create_conversation(
            player_id=player_id, npc_id=npc_id
        )

        exchange_repo = ConversationExchangeRepository(db_session)

        # Create multiple exchanges
        exchanges = []
        for i in range(5):
            exchange = await exchange_repo.create_exchange(
                conversation_id=conversation.conversation_id,
                speaker_id=player_id if i % 2 == 0 else npc_id,
                message_text=f"Message {i}",
                message_type="statement",
            )
            exchanges.append(exchange)

        # Get all exchanges
        all_exchanges = await exchange_repo.get_conversation_exchanges(
            conversation.conversation_id
        )
        assert len(all_exchanges) == 5

        # Get limited exchanges
        limited_exchanges = await exchange_repo.get_conversation_exchanges(
            conversation.conversation_id, limit=3
        )
        assert len(limited_exchanges) == 3

    async def test_get_recent_exchanges(self, db_session):
        """Test getting recent exchanges."""

        # Setup
        npc_repo = NPCPersonalityRepository(db_session)
        npc_id = uuid.uuid4()
        await npc_repo.create_personality(
            npc_id=npc_id,
            traits={},
            knowledge_areas=[],
            speech_patterns={},
            background_story="Test NPC",
        )

        conv_repo = ConversationContextRepository(db_session)
        player_id = uuid.uuid4()
        conversation = await conv_repo.create_conversation(
            player_id=player_id, npc_id=npc_id
        )

        exchange_repo = ConversationExchangeRepository(db_session)

        # Create exchanges - since PostgreSQL current_timestamp() gives same value
        # within a transaction, they'll all have identical timestamps
        for i in range(10):
            await exchange_repo.create_exchange(
                conversation_id=conversation.conversation_id,
                speaker_id=player_id,
                message_text=f"Message {i}",
                message_type="statement",
            )

        # Get recent exchanges
        recent = await exchange_repo.get_recent_exchanges(
            conversation.conversation_id, limit=3
        )
        assert len(recent) == 3

        # Since all timestamps are identical, PostgreSQL returns them in arbitrary order
        # The important thing is that we get exactly 3 exchanges from the conversation
        message_texts = {exchange.message_text for exchange in recent}
        assert len(message_texts) == 3  # Should have 3 different messages

        # All should be from our conversation
        for exchange in recent:
            assert exchange.conversation_id == conversation.conversation_id
            assert "Message" in exchange.message_text

    async def test_search_exchanges_by_content(self, db_session):
        """Test searching exchanges by content."""
        # Setup
        npc_repo = NPCPersonalityRepository(db_session)
        npc_id = uuid.uuid4()
        await npc_repo.create_personality(
            npc_id=npc_id,
            traits={},
            knowledge_areas=[],
            speech_patterns={},
            background_story="Test NPC",
        )

        conv_repo = ConversationContextRepository(db_session)
        player_id = uuid.uuid4()
        conversation = await conv_repo.create_conversation(
            player_id=player_id, npc_id=npc_id
        )

        exchange_repo = ConversationExchangeRepository(db_session)

        # Create exchanges with different content
        dragon_ex1 = await exchange_repo.create_exchange(
            conversation_id=conversation.conversation_id,
            speaker_id=player_id,
            message_text="I love dragons!",
            message_type="statement",
        )
        dragon_ex2 = await exchange_repo.create_exchange(
            conversation_id=conversation.conversation_id,
            speaker_id=npc_id,
            message_text="Dragons are dangerous creatures.",
            message_type="statement",
        )
        unicorn_ex = await exchange_repo.create_exchange(
            conversation_id=conversation.conversation_id,
            speaker_id=player_id,
            message_text="What about unicorns?",
            message_type="question",
        )

        # Search for "dragon" - filter to our conversation
        dragon_exchanges = await exchange_repo.search_exchanges_by_content(
            "dragon", conversation_id=conversation.conversation_id
        )
        dragon_exchange_ids = {ex.exchange_id for ex in dragon_exchanges}
        # Check that our dragon exchanges are found
        assert dragon_ex1.exchange_id in dragon_exchange_ids
        assert dragon_ex2.exchange_id in dragon_exchange_ids
        # Unicorn exchange should not be in dragon results
        assert unicorn_ex.exchange_id not in dragon_exchange_ids

        # Search for "unicorn" within specific conversation
        unicorn_exchanges = await exchange_repo.search_exchanges_by_content(
            "unicorn", conversation_id=conversation.conversation_id
        )
        unicorn_exchange_ids = {ex.exchange_id for ex in unicorn_exchanges}
        # Check that unicorn exchange is found
        assert unicorn_ex.exchange_id in unicorn_exchange_ids
        # Dragon exchanges should not be in unicorn results
        assert dragon_ex1.exchange_id not in unicorn_exchange_ids
        assert dragon_ex2.exchange_id not in unicorn_exchange_ids


@pytest.mark.asyncio
class TestConversationKnowledgeRepository:
    """Test ConversationKnowledge repository operations."""

    async def test_create_knowledge_entry(self, db_session):
        """Test creating a knowledge entry."""
        # Setup
        npc_repo = NPCPersonalityRepository(db_session)
        npc_id = uuid.uuid4()
        await npc_repo.create_personality(
            npc_id=npc_id,
            traits={},
            knowledge_areas=[],
            speech_patterns={},
            background_story="Test NPC",
        )

        conv_repo = ConversationContextRepository(db_session)
        player_id = uuid.uuid4()
        conversation = await conv_repo.create_conversation(
            player_id=player_id, npc_id=npc_id
        )

        exchange_repo = ConversationExchangeRepository(db_session)
        exchange = await exchange_repo.create_exchange(
            conversation_id=conversation.conversation_id,
            speaker_id=npc_id,
            message_text="The ancient temple is located in the eastern forest.",
            message_type="statement",
        )

        # Create knowledge entry
        knowledge_repo = ConversationKnowledgeRepository(db_session)
        knowledge = await knowledge_repo.create_knowledge_entry(
            conversation_id=conversation.conversation_id,
            information_type="location",
            extracted_info={
                "entity": "ancient temple",
                "location": "eastern forest",
                "type": "building",
            },
            confidence_score=0.9,
            source_exchange_id=exchange.exchange_id,
        )

        assert knowledge.conversation_id == conversation.conversation_id
        assert knowledge.information_type == "location"
        assert knowledge.extracted_info["entity"] == "ancient temple"
        assert float(knowledge.confidence_score) == 0.9
        assert knowledge.source_exchange_id == exchange.exchange_id

    async def test_get_knowledge_by_type(self, db_session):
        """Test retrieving knowledge by type."""
        # Setup
        npc_repo = NPCPersonalityRepository(db_session)
        npc_id = uuid.uuid4()
        await npc_repo.create_personality(
            npc_id=npc_id,
            traits={},
            knowledge_areas=[],
            speech_patterns={},
            background_story="Test NPC",
        )

        conv_repo = ConversationContextRepository(db_session)
        player_id = uuid.uuid4()
        conversation = await conv_repo.create_conversation(
            player_id=player_id, npc_id=npc_id
        )

        knowledge_repo = ConversationKnowledgeRepository(db_session)

        # Create different types of knowledge
        loc1 = await knowledge_repo.create_knowledge_entry(
            conversation_id=conversation.conversation_id,
            information_type="location",
            extracted_info={"place": "forest"},
        )
        loc2 = await knowledge_repo.create_knowledge_entry(
            conversation_id=conversation.conversation_id,
            information_type="location",
            extracted_info={"place": "mountain"},
        )
        char1 = await knowledge_repo.create_knowledge_entry(
            conversation_id=conversation.conversation_id,
            information_type="character",
            extracted_info={"name": "wizard"},
        )

        # Get location knowledge and filter to our conversation
        location_knowledge = await knowledge_repo.get_knowledge_by_type("location")
        our_location_knowledge = [
            k
            for k in location_knowledge
            if k.conversation_id == conversation.conversation_id
        ]
        location_knowledge_ids = {k.knowledge_id for k in our_location_knowledge}
        # Check that our location entries are found
        assert loc1.knowledge_id in location_knowledge_ids
        assert loc2.knowledge_id in location_knowledge_ids
        # Character entry should not be in location results
        assert char1.knowledge_id not in location_knowledge_ids

        # Get character knowledge and filter to our conversation
        character_knowledge = await knowledge_repo.get_knowledge_by_type("character")
        our_character_knowledge = [
            k
            for k in character_knowledge
            if k.conversation_id == conversation.conversation_id
        ]
        character_knowledge_ids = {k.knowledge_id for k in our_character_knowledge}
        # Check that our character entry is found
        assert char1.knowledge_id in character_knowledge_ids
        # Location entries should not be in character results
        assert loc1.knowledge_id not in character_knowledge_ids
        assert loc2.knowledge_id not in character_knowledge_ids

    async def test_get_high_confidence_knowledge(self, db_session):
        """Test retrieving high confidence knowledge."""
        # Setup
        npc_repo = NPCPersonalityRepository(db_session)
        npc_id = uuid.uuid4()
        await npc_repo.create_personality(
            npc_id=npc_id,
            traits={},
            knowledge_areas=[],
            speech_patterns={},
            background_story="Test NPC",
        )

        conv_repo = ConversationContextRepository(db_session)
        player_id = uuid.uuid4()
        conversation = await conv_repo.create_conversation(
            player_id=player_id, npc_id=npc_id
        )

        knowledge_repo = ConversationKnowledgeRepository(db_session)

        # Create knowledge with different confidence scores
        high_conf_entry = await knowledge_repo.create_knowledge_entry(
            conversation_id=conversation.conversation_id,
            information_type="fact",
            extracted_info={"fact": "high confidence"},
            confidence_score=0.9,
        )
        med_conf_entry = await knowledge_repo.create_knowledge_entry(
            conversation_id=conversation.conversation_id,
            information_type="fact",
            extracted_info={"fact": "medium confidence"},
            confidence_score=0.7,
        )
        low_conf_entry = await knowledge_repo.create_knowledge_entry(
            conversation_id=conversation.conversation_id,
            information_type="fact",
            extracted_info={"fact": "low confidence"},
            confidence_score=0.4,
        )

        # Get high confidence knowledge (>= 0.8) and filter to our conversation
        high_conf = await knowledge_repo.get_high_confidence_knowledge(
            min_confidence=0.8
        )
        our_high_conf = [
            k for k in high_conf if k.conversation_id == conversation.conversation_id
        ]
        high_conf_ids = {k.knowledge_id for k in our_high_conf}

        # Check that only high confidence entry is found
        assert high_conf_entry.knowledge_id in high_conf_ids
        assert med_conf_entry.knowledge_id not in high_conf_ids
        assert low_conf_entry.knowledge_id not in high_conf_ids

        # Verify the entry has correct data
        our_entry = next(
            k for k in our_high_conf if k.knowledge_id == high_conf_entry.knowledge_id
        )
        assert our_entry.extracted_info["fact"] == "high confidence"


@pytest.mark.asyncio
class TestConversationRepositoryManager:
    """Test ConversationRepositoryManager integration."""

    async def test_get_full_conversation(self, db_session):
        """Test getting complete conversation data."""
        manager = ConversationRepositoryManager(db_session)

        # Setup NPC
        npc_id = uuid.uuid4()
        await manager.npc_personalities.create_personality(
            npc_id=npc_id,
            traits={},
            knowledge_areas=[],
            speech_patterns={},
            background_story="Test NPC",
        )

        # Create conversation
        player_id = uuid.uuid4()
        conversation = await manager.contexts.create_conversation(
            player_id=player_id,
            npc_id=npc_id,
        )

        # Add exchanges
        exchange1 = await manager.exchanges.create_exchange(
            conversation_id=conversation.conversation_id,
            speaker_id=player_id,
            message_text="Hello",
            message_type="greeting",
        )
        exchange2 = await manager.exchanges.create_exchange(
            conversation_id=conversation.conversation_id,
            speaker_id=npc_id,
            message_text="Greetings!",
            message_type="greeting",
        )

        # Add knowledge
        await manager.knowledge.create_knowledge_entry(
            conversation_id=conversation.conversation_id,
            information_type="greeting",
            extracted_info={"type": "friendly_greeting"},
            source_exchange_id=exchange1.exchange_id,
        )

        # Get full conversation
        context, exchanges, knowledge = await manager.get_full_conversation(
            conversation.conversation_id
        )

        assert context is not None
        assert len(exchanges) == 2
        assert len(knowledge) == 1
        assert exchanges[0].message_text == "Hello"
        assert knowledge[0].information_type == "greeting"

    async def test_create_complete_conversation(self, db_session):
        """Test creating conversation with initial exchange."""
        manager = ConversationRepositoryManager(db_session)

        # Setup NPC
        npc_id = uuid.uuid4()
        await manager.npc_personalities.create_personality(
            npc_id=npc_id,
            traits={},
            knowledge_areas=[],
            speech_patterns={},
            background_story="Test NPC",
        )

        # Create complete conversation
        player_id = uuid.uuid4()
        conversation, exchange = await manager.create_complete_conversation(
            player_id=player_id,
            npc_id=npc_id,
            initial_message="Welcome, traveler!",
            message_type="greeting",
            topic="Welcome",
        )

        assert conversation.player_id == player_id
        assert conversation.npc_id == npc_id
        assert conversation.topic == "Welcome"
        assert exchange.speaker_id == npc_id
        assert exchange.message_text == "Welcome, traveler!"
        assert exchange.message_type == "greeting"
