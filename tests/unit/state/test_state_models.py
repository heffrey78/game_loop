from datetime import datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from game_loop.state.models import (
    ActionResult,
    GameSession,
    InventoryItem,
    Location,
    NonPlayerCharacter,
    PlayerKnowledge,
    PlayerProgress,
    PlayerState,
    PlayerStats,
    WorldObject,
    WorldState,
)


# --- Test InventoryItem ---
def test_inventory_item_creation():
    item_id = uuid4()
    item = InventoryItem(
        item_id=item_id,
        name="Sword",
        description="A sharp sword.",
        quantity=1,
        attributes={"damage": "1d6"},
    )
    assert item.item_id == item_id
    assert item.name == "Sword"
    assert item.description == "A sharp sword."
    assert item.quantity == 1
    assert item.attributes == {"damage": "1d6"}


def test_inventory_item_defaults():
    item = InventoryItem(name="Shield", description="A sturdy shield.")
    assert isinstance(item.item_id, UUID)
    assert item.quantity == 1
    assert item.attributes == {}


# --- Test PlayerKnowledge ---
def test_player_knowledge_creation():
    knowledge_id = uuid4()
    now = datetime.now()
    knowledge = PlayerKnowledge(
        knowledge_id=knowledge_id,
        topic="Ancient Ruins",
        content="Discovered a map to the old ruins.",
        discovered_at=now,
        source="Old Scroll",
    )
    assert knowledge.knowledge_id == knowledge_id
    assert knowledge.topic == "Ancient Ruins"
    assert knowledge.content == "Discovered a map to the old ruins."
    assert knowledge.discovered_at == now
    assert knowledge.source == "Old Scroll"


def test_player_knowledge_defaults():
    knowledge = PlayerKnowledge(topic="Local Flora", content="Rosehips are edible.")
    assert isinstance(knowledge.knowledge_id, UUID)
    assert isinstance(knowledge.discovered_at, datetime)
    assert knowledge.source is None


# --- Test PlayerStats ---
def test_player_stats_creation():
    stats = PlayerStats(
        health=90,
        max_health=110,
        mana=40,
        max_mana=60,
        strength=12,
        dexterity=11,
        intelligence=13,
    )
    assert stats.health == 90
    assert stats.max_health == 110
    assert stats.mana == 40
    assert stats.max_mana == 60
    assert stats.strength == 12
    assert stats.dexterity == 11
    assert stats.intelligence == 13


def test_player_stats_defaults():
    stats = PlayerStats()
    assert stats.health == 100
    assert stats.max_health == 100
    assert stats.mana == 50
    assert stats.max_mana == 50
    assert stats.strength == 10
    assert stats.dexterity == 10
    assert stats.intelligence == 10


# --- Test PlayerProgress ---
def test_player_progress_creation():
    quest_id = uuid4()
    progress = PlayerProgress(
        active_quests={quest_id: {"stage": 1, "objective": "Find key"}},
        completed_quests=[uuid4()],
        achievements=["Tutorial Complete"],
        flags={"met_king": True},
    )
    assert quest_id in progress.active_quests
    assert len(progress.completed_quests) == 1
    assert "Tutorial Complete" in progress.achievements
    assert progress.flags["met_king"] is True


def test_player_progress_defaults():
    progress = PlayerProgress()
    assert progress.active_quests == {}
    assert progress.completed_quests == []
    assert progress.achievements == []
    assert progress.flags == {}


# --- Test PlayerState ---
def test_player_state_creation():
    player_id = uuid4()
    loc_id = uuid4()
    item = InventoryItem(name="Potion", description="Heals 10 HP")
    knowledge = PlayerKnowledge(topic="Weakness", content="Goblins hate fire.")

    player_state = PlayerState(
        player_id=player_id,
        name="Alice",
        current_location_id=loc_id,
        inventory=[item],
        knowledge=[knowledge],
        stats=PlayerStats(health=80),
        progress=PlayerProgress(flags={"started_game": True}),
        visited_locations=[uuid4()],
    )
    assert player_state.player_id == player_id
    assert player_state.name == "Alice"
    assert player_state.current_location_id == loc_id
    assert player_state.inventory[0].name == "Potion"
    assert player_state.knowledge[0].topic == "Weakness"
    assert player_state.stats.health == 80
    assert player_state.progress.flags["started_game"] is True
    assert len(player_state.visited_locations) == 1
    assert player_state.state_data_json is None


def test_player_state_defaults():
    player_state = PlayerState()
    assert isinstance(player_state.player_id, UUID)
    assert player_state.name == "Player"
    assert player_state.current_location_id is None
    assert player_state.inventory == []
    assert player_state.knowledge == []
    assert isinstance(player_state.stats, PlayerStats)
    assert isinstance(player_state.progress, PlayerProgress)
    assert player_state.visited_locations == []


# --- Test WorldObject ---
def test_world_object_creation():
    obj_id = uuid4()
    obj = WorldObject(
        object_id=obj_id,
        name="Chest",
        description="A wooden chest.",
        is_takeable=False,
        is_container=True,
        state={"locked": True},
        contained_items=[uuid4()],
    )
    assert obj.object_id == obj_id
    assert obj.name == "Chest"
    assert obj.is_container is True
    assert obj.state["locked"] is True
    assert len(obj.contained_items) == 1


# --- Test NonPlayerCharacter ---
def test_npc_creation():
    npc_id = uuid4()
    npc = NonPlayerCharacter(
        npc_id=npc_id,
        name="Guard",
        description="A stern-looking guard.",
        dialogue_state="hostile",
        current_behavior="patrolling",
        state={"alert_level": 1},
    )
    assert npc.npc_id == npc_id
    assert npc.name == "Guard"
    assert npc.dialogue_state == "hostile"
    assert npc.state["alert_level"] == 1


# --- Test Location ---
def test_location_creation():
    loc_id = uuid4()
    obj_id = uuid4()
    npc_id = uuid4()
    exit_loc_id = uuid4()

    world_obj = WorldObject(
        object_id=obj_id, name="Fountain", description="A water fountain."
    )
    npc = NonPlayerCharacter(npc_id=npc_id, name="Merchant", description="Sells wares.")

    location = Location(
        location_id=loc_id,
        name="Town Square",
        description="A bustling square.",
        objects={obj_id: world_obj},
        npcs={npc_id: npc},
        connections={"north": exit_loc_id},
        state_flags={"market_open": True},
    )
    assert location.location_id == loc_id
    assert location.name == "Town Square"
    assert location.objects[obj_id].name == "Fountain"
    assert location.npcs[npc_id].name == "Merchant"
    assert location.connections["north"] == exit_loc_id
    assert location.state_flags["market_open"] is True
    assert location.first_visited is None
    assert location.last_visited is None


def test_location_defaults():
    location = Location(name="Cave Entrance", description="Dark and foreboding.")
    assert isinstance(location.location_id, UUID)
    assert location.objects == {}
    assert location.npcs == {}
    assert location.connections == {}
    assert location.state_flags == {}


# --- Test WorldState ---
def test_world_state_creation():
    world_id = uuid4()
    loc_id = uuid4()
    location = Location(
        location_id=loc_id, name="Forest Path", description="A winding path."
    )
    now = datetime.now()

    from game_loop.state.models import EvolutionEvent

    evolution_event = EvolutionEvent(trigger="meteor_shower", timestamp=now)

    world_state = WorldState(
        world_id=world_id,
        locations={loc_id: location},
        global_flags={"dragon_defeated": False},
        current_time=now,
        evolution_queue=[evolution_event],
    )
    assert world_state.world_id == world_id
    assert world_state.locations[loc_id].name == "Forest Path"
    assert world_state.global_flags["dragon_defeated"] is False
    assert world_state.current_time == now
    assert len(world_state.evolution_queue) == 1
    assert world_state.state_data_json is None


def test_world_state_defaults():
    world_state = WorldState()
    assert isinstance(world_state.world_id, UUID)
    assert world_state.locations == {}
    assert world_state.global_flags == {}
    assert isinstance(world_state.current_time, datetime)
    assert world_state.evolution_queue == []


# --- Test GameSession ---
def test_game_session_creation():
    session_id = uuid4()
    player_state_id = uuid4()
    world_state_id = uuid4()
    now = datetime.now()

    session = GameSession(
        session_id=session_id,
        player_state_id=player_state_id,
        world_state_id=world_state_id,
        save_name="My Epic Adventure",
        created_at=now,
        updated_at=now,
        game_version="0.2.0",
    )
    assert session.session_id == session_id
    assert session.player_state_id == player_state_id
    assert session.world_state_id == world_state_id
    assert session.save_name == "My Epic Adventure"
    assert session.created_at == now
    assert session.updated_at == now
    assert session.game_version == "0.2.0"


def test_game_session_defaults():
    player_state_id = uuid4()
    world_state_id = uuid4()
    session = GameSession(
        player_state_id=player_state_id, world_state_id=world_state_id
    )
    assert isinstance(session.session_id, UUID)
    assert session.save_name == "New Save"
    assert isinstance(session.created_at, datetime)
    assert isinstance(session.updated_at, datetime)
    assert session.game_version == "0.1.0"


# --- Test ActionResult ---
def test_action_result_creation():
    new_loc_id = uuid4()
    knowledge_item = PlayerKnowledge(topic="test", content="test content")
    item_id_to_remove = uuid4()
    obj_id_to_change = uuid4()
    npc_id_to_change = uuid4()
    door_id_for_evolution = uuid4()

    action_result = ActionResult(
        success=False,
        feedback_message="You failed to open the door.",
        location_change=True,
        new_location_id=new_loc_id,
        inventory_changes=[{"action": "remove", "item_id": item_id_to_remove}],
        knowledge_updates=[knowledge_item],
        stat_changes={"health": -5},
        progress_updates={"quest_log": "updated"},
        object_changes=[{"object_id": obj_id_to_change, "new_state": {"broken": True}}],
        npc_changes=[{"npc_id": npc_id_to_change, "new_dialogue_state": "angry"}],
        location_state_changes={"door_jammed": True},
        triggers_evolution=True,
        evolution_trigger="door_break_event",
        evolution_data={"door_id": door_id_for_evolution},
        priority=10,
        command="open door",
        processed_input={"intent": "open", "target": "door"},
    )
    assert action_result.success is False
    assert action_result.feedback_message == "You failed to open the door."
    assert action_result.location_change is True
    assert action_result.new_location_id == new_loc_id

    assert action_result.inventory_changes is not None
    assert len(action_result.inventory_changes) == 1
    assert action_result.inventory_changes[0]["action"] == "remove"
    assert action_result.inventory_changes[0]["item_id"] == item_id_to_remove

    assert action_result.knowledge_updates is not None
    assert len(action_result.knowledge_updates) == 1
    assert action_result.knowledge_updates[0].topic == "test"

    assert action_result.stat_changes is not None
    assert action_result.stat_changes["health"] == -5

    assert action_result.progress_updates is not None
    assert action_result.progress_updates["quest_log"] == "updated"

    assert action_result.object_changes is not None
    assert len(action_result.object_changes) == 1
    assert action_result.object_changes[0]["object_id"] == obj_id_to_change
    assert action_result.object_changes[0]["new_state"]["broken"] is True

    assert action_result.npc_changes is not None
    assert len(action_result.npc_changes) == 1
    assert action_result.npc_changes[0]["npc_id"] == npc_id_to_change
    assert action_result.npc_changes[0]["new_dialogue_state"] == "angry"

    assert action_result.location_state_changes is not None
    assert action_result.location_state_changes["door_jammed"] is True

    assert action_result.triggers_evolution is True
    assert action_result.evolution_trigger == "door_break_event"

    assert action_result.evolution_data is not None
    assert action_result.evolution_data["door_id"] == door_id_for_evolution

    assert action_result.priority == 10
    assert isinstance(action_result.timestamp, datetime)
    assert action_result.command == "open door"

    assert action_result.processed_input is not None
    assert action_result.processed_input["intent"] == "open"


def test_action_result_defaults():
    action_result = ActionResult()
    assert action_result.success is True
    assert action_result.feedback_message == ""
    assert action_result.location_change is False
    assert action_result.new_location_id is None
    assert action_result.inventory_changes is None
    assert action_result.knowledge_updates is None
    assert action_result.stat_changes is None
    assert action_result.progress_updates is None
    assert action_result.object_changes is None
    assert action_result.npc_changes is None
    assert action_result.location_state_changes is None
    assert action_result.triggers_evolution is False
    assert action_result.evolution_trigger is None
    assert action_result.evolution_data is None
    assert action_result.priority == 5
    assert isinstance(action_result.timestamp, datetime)
    assert action_result.command is None
    assert action_result.processed_input is None


# --- Serialization/Deserialization Tests ---
def test_inventory_item_serialization():
    item = InventoryItem(
        name="Potion",
        description="Restores health",
        quantity=5,
        attributes={"type": "healing"},
    )
    item_dict = item.model_dump()
    assert item_dict["name"] == "Potion"
    assert item_dict["quantity"] == 5
    assert item_dict["attributes"]["type"] == "healing"
    assert "item_id" in item_dict

    item_json = item.model_dump_json()
    assert isinstance(item_json, str)
    assert '"name":"Potion"' in item_json
    assert '"description":"Restores health"' in item_json
    assert '"quantity":5' in item_json
    assert '"attributes":{"type":"healing"}' in item_json

    loaded_item = InventoryItem.model_validate(item_dict)
    assert loaded_item == item
    loaded_item_json = InventoryItem.model_validate_json(item_json)
    assert loaded_item_json == item


def test_player_state_serialization_deserialization():
    player_id = uuid4()
    loc_id = uuid4()
    original_player_state = PlayerState(
        player_id=player_id,
        name="Tester",
        current_location_id=loc_id,
        inventory=[InventoryItem(name="Key", description="Unlocks a door")],
        stats=PlayerStats(health=90),
        knowledge=[PlayerKnowledge(topic="secrets", content="Hidden Door location")],
        progress=PlayerProgress(flags={"found_artifact": True}),
    )

    player_state_json = original_player_state.model_dump_json()
    loaded_player_state = PlayerState.model_validate_json(player_state_json)

    assert loaded_player_state.player_id == original_player_state.player_id
    assert loaded_player_state.name == "Tester"
    assert loaded_player_state.current_location_id == loc_id
    assert len(loaded_player_state.inventory) == 1
    assert loaded_player_state.inventory[0].name == "Key"
    assert loaded_player_state.stats.health == 90
    assert len(loaded_player_state.knowledge) == 1
    assert loaded_player_state.knowledge[0].topic == "secrets"
    assert loaded_player_state.progress.flags["found_artifact"] is True


def test_world_state_with_locations_serialization():
    world_id = uuid4()
    loc1_id = uuid4()
    loc2_id = uuid4()

    location1 = Location(
        location_id=loc1_id, name="Town Square", description="Bustling."
    )
    location2 = Location(location_id=loc2_id, name="Dark Cave", description="Cold.")

    original_world_state = WorldState(
        world_id=world_id,
        locations={loc1_id: location1, loc2_id: location2},
        global_flags={"event_started": True},
    )

    world_state_json = original_world_state.model_dump_json()
    loaded_world_state = WorldState.model_validate_json(world_state_json)

    assert loaded_world_state.world_id == world_id
    assert len(loaded_world_state.locations) == 2
    assert loc1_id in loaded_world_state.locations
    assert loaded_world_state.locations[loc1_id].name == "Town Square"
    assert loaded_world_state.global_flags["event_started"] is True


# --- Validation Tests (Example) ---
# Add these if you include validators in your Pydantic models
# For example, if PlayerStats had a validator for health:
#
# class PlayerStats(BaseModel):
#     health: int = 100
#     ...
#     @validator(\'health\')
#     def health_must_be_non_negative(cls, v):
#         if v < 0:
#             raise ValueError(\'health must be non-negative\')
#         return v


def test_player_stats_health_validation_placeholder():
    # This test would fail if a validator like the one above is implemented
    # and health is set to a negative value.
    # For now, Pydantic allows any int.
    try:
        PlayerStats(health=-10)
    except ValidationError:
        # This branch would be taken if validators are active and catch this
        pytest.fail(
            "ValidationError was raised unexpectedly for health, "
            "or test needs update for active validator."
        )
    # If no validator, this just creates the object.
    # If a validator exists that prevents negative health, this test needs
    # to be:
    # with pytest.raises(ValidationError):
    # PlayerStats(health=-10)
    pass


# Add more tests for default values, specific model behaviors,
# and validation rules as your models evolve.
