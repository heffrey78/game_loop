"""Facade for managing the overall game state,
coordinating player, world, and session state."""

import logging
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import asyncpg

from ..config.manager import ConfigManager
from .models import (
    ActionResult,
    GameSession,
    Location,
    PlayerState,
    WorldState,
)
from .player_state import PlayerStateTracker
from .session_manager import SessionManager
from .world_state import WorldStateTracker

if TYPE_CHECKING:
    from ..embeddings.service import EmbeddingService

logger = logging.getLogger(__name__)


class GameStateManager:
    """Coordinates PlayerStateTracker, WorldStateTracker, and SessionManager."""

    def __init__(
        self,
        config_manager: ConfigManager,
        db_pool: asyncpg.Pool,
        # Optional dependency injection
        embedding_service: "EmbeddingService | None" = None,
    ):
        self.config_manager = config_manager
        self.db_pool = db_pool
        self.player_tracker = PlayerStateTracker(db_pool)
        self.world_tracker = WorldStateTracker(db_pool)
        self.session_manager = SessionManager(db_pool)
        self._current_session: GameSession | None = None
        # Lazy initialization for embedding service
        self._embedding_service = embedding_service

    @property
    def embedding_service(self) -> "EmbeddingService | None":
        """
        Get embedding service if features.use_embedding_search is enabled.

        Returns:
            EmbeddingService instance if enabled and available, None otherwise
        """
        # Check if embedding search is enabled via feature flags
        if not self.config_manager.is_embedding_enabled():
            return None

        # Lazy initialization
        if self._embedding_service is None:
            try:
                self._embedding_service = self.config_manager.create_embedding_service()
            except (ImportError, ValueError) as e:
                logger.warning(f"Failed to create embedding service: {e}")
                return None

        return self._embedding_service

    async def initialize(
        self,
        session_id: UUID | None = None,
        player_state_id: UUID | None = None,
        world_state_id: UUID | None = None,
    ) -> None:
        """Initializes all state trackers, loading data based on provided IDs."""
        logger.info(f"Initializing GameStateManager for session {session_id}...")

        if session_id:
            self._current_session = await self.session_manager.load_session(session_id)
            if not self._current_session:
                logger.error(
                    f"Failed to load session {session_id}. "
                    f"Cannot initialize state trackers."
                )
                return

            # Use IDs from the loaded session if specific IDs weren't provided
            player_id_to_load = player_state_id or self._current_session.player_state_id
            world_id_to_load = world_state_id or self._current_session.world_state_id

            await self.player_tracker.initialize(session_id, player_id_to_load)
            await self.world_tracker.initialize(session_id, world_id_to_load)
            logger.info("GameStateManager initialized with loaded session.")
        else:
            # Case: Starting a new game without a pre-existing session ID
            # The trackers will be initialized but won't load specific state yet.
            # A new session will be created later via create_new_game.
            # We need a temporary session ID for the trackers during creation.
            temp_session_id = uuid4()
            await self.player_tracker.initialize(temp_session_id)
            await self.world_tracker.initialize(temp_session_id)
            logger.info(
                "GameStateManager initialized without a session ID "
                "(ready for new game)."
            )

    async def create_new_game(
        self, player_name: str = "Player", save_name: str = "New Game"
    ) -> tuple[PlayerState | None, WorldState | None]:
        """Creates a new game session, player state, and world state."""
        logger.info(f"Creating new game: Player='{player_name}', Save='{save_name}'")
        try:
            # 1. Create a themed world - abandoned cubicle farm
            location_ids = {}
            initial_locations: dict[UUID, Location] = {}

            async with self.db_pool.acquire() as conn:

                # CENTRAL LOCATION - Reception Area
                reception_id = uuid4()
                location_ids["reception"] = reception_id
                await conn.execute(
                    """
                    INSERT INTO locations (
                        id,
                        name,
                        short_desc,
                        full_desc,
                        location_type,
                        is_dynamic,
                        created_by,
                        state_json
                        )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    reception_id,
                    "Abandoned Reception Area",
                    "A dusty reception desk sits in this once-bustling office entrance.",  # noqa: E501
                    "Fluorescent lights flicker weakly overhead, casting eerie shadows across the reception area. A thick layer of dust covers the once-polished desk where visitors would check in. Torn posters about workplace productivity hang crooked on the walls. The faint sound of a phone ringing somewhere deep in the building sends chills down your spine, though the lines should be long dead. Paper name tags are scattered on the floor, remnants of employees who disappeared without warning when the company mysteriously shut down.",  # noqa: E501
                    "indoor",
                    False,
                    "system",
                    "{}",
                )
                logger.info(f"Created reception area {reception_id}")

                # NORTH - Executive Suite
                north_id = uuid4()
                location_ids["north"] = north_id
                await conn.execute(
                    """
                    INSERT INTO locations (id, name, short_desc, full_desc, location_type, is_dynamic, created_by, state_json)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,  # noqa: E501
                    north_id,
                    "Executive Suite",
                    "Luxurious but abandoned offices of former executives.",
                    "Unlike the rest of the office, these rooms once housed the company elites. Mahogany desks are now warped with moisture, and plush leather chairs are cracked and moldy. A large portrait of the CEO hangs on one wall, the eyes seemingly following you around the room. Someone has scratched out the face with what looks like fingernails. The calendar on the central desk is still open to the date of the 'incident.' Expensive pens have rolled into corners, and a half-empty bottle of whiskey sits on a credenza, its contents long evaporated. A newspaper clipping about 'experimental corporate wellness programs' lies partially burned in a wastepaper basket.",  # noqa: E501
                    "indoor",
                    False,
                    "system",
                    "{}",
                )
                logger.info(f"Created north location {north_id}")

                # EAST - Server Room
                east_id = uuid4()
                location_ids["east"] = east_id
                await conn.execute(
                    """
                    INSERT INTO locations (id, name, short_desc, full_desc, location_type, is_dynamic, created_by, state_json)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,  # noqa: E501
                    east_id,
                    "Server Room",
                    "Rows of old servers hum ominously in this cold room.",
                    "The temperature drops noticeably as you enter the server room. Against all logic, some of the ancient servers are still running, blinking with cryptic patterns of lights. Thick cables snake across the floor like dormant serpents. The constant hum has an almost hypnotic quality, occasionally interrupted by clicks and whirs from machines that should have died years ago. A terminal screen flickers in the corner, displaying fragmented code and what might be employee records. One section of the wall has been torn open, revealing unusual modifications to the building's electrical system. Something about the arrangement of wires reminds you of a neural network, as if the building itself had been turned into some kind of crude brain.",  # noqa: E501
                    "indoor",
                    False,
                    "system",
                    "{}",
                )
                logger.info(f"Created east location {east_id}")

                # SOUTH - Cafeteria
                south_id = uuid4()
                location_ids["south"] = south_id
                await conn.execute(
                    """
                    INSERT INTO locations (id, name, short_desc, full_desc, location_type, is_dynamic, created_by, state_json)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,  # noqa: E501
                    south_id,
                    "Abandoned Cafeteria",
                    "Once-bustling eating area now frozen in time.",
                    "The cafeteria looks as though everyone vanished mid-meal. Trays with half-eaten food remain on tables, now covered in a fuzzy layer of mold that has grown into unnatural colors. The serving counter still holds industrial-sized containers of unidentifiable substances. A motivational banner reading 'PRODUCTIVITY FEEDS THE SOUL' hangs askew above the food line. Several chairs are overturned, suggesting people left in a hurry. The vending machines stand dark, except for one that occasionally sputters to life, its internal mechanics grinding painfully. The wall clock is frozen at 2:37, though whether it's AM or PM is anyone's guess. A strange dark stain spreads from beneath the door to the kitchen.",  # noqa: E501
                    "indoor",
                    False,
                    "system",
                    "{}",
                )
                logger.info(f"Created south location {south_id}")

                # WEST - Cubicle Farm
                west_id = uuid4()
                location_ids["west"] = west_id
                await conn.execute(
                    """
                    INSERT INTO locations (id, name, short_desc, full_desc, location_type, is_dynamic, created_by, state_json)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,  # noqa: E501
                    west_id,
                    "Cubicle Farm",
                    "Endless rows of gray partitions form a maze of abandoned workspaces.",  # noqa: E501
                    "Gray fabric partitions create a labyrinth of identical workspaces, stretching far beyond what the building's exterior would suggest possible. Personal items are still at each desk—family photos, coffee mugs with cheerful slogans, now a mockery in this desolate place. Some cubicles have been personalized with plants, now brown and desiccated, reaching out like withered claws. Computer monitors display error messages or peculiar screensavers of fractal patterns that never quite repeat. Motivational posters about teamwork and persistence take on a sinister tone in the eerie silence. Occasionally, the distant sound of typing can be heard, though the source is impossible to locate. Post-it notes with cryptic messages are stuck to monitors: 'IT'S IN THE WALLS' and 'DON'T STAY AFTER HOURS.' The air feels charged, as if the collective despair of a thousand meaningless workdays has become a tangible presence.",  # noqa: E501
                    "indoor",
                    False,
                    "system",
                    "{}",
                )
                logger.info(f"Created west location {west_id}")

                # Now add connections between locations
                # Connect reception to all four directions
                await self._create_bidirectional_connection(
                    conn, reception_id, north_id, "north", "south"
                )
                await self._create_bidirectional_connection(
                    conn, reception_id, east_id, "east", "west"
                )
                await self._create_bidirectional_connection(
                    conn, reception_id, south_id, "south", "north"
                )
                await self._create_bidirectional_connection(
                    conn, reception_id, west_id, "west", "east"
                )

                # Add items to the world
                # Reception area items
                keycard_id = uuid4()
                await conn.execute(
                    """
                    INSERT INTO objects (id, name, short_desc, full_desc, object_type, is_takeable, location_id, properties_json, state_json)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,  # noqa: E501
                    keycard_id,
                    "Keycard",
                    "An employee access card with faded ID photo.",
                    "The plastic keycard displays a smiling employee photo that has faded to an unsettling blur. The name 'J. Cooper' is still visible, along with 'Level 3 Access.' The magnetic strip appears to be intact, though whether any security systems still function is unknown.",  # noqa: E501
                    "key",
                    True,
                    reception_id,
                    "{}",
                    "{}",
                )

                visitor_log_id = uuid4()
                await conn.execute(
                    """
                    INSERT INTO objects (id, name, short_desc, full_desc, object_type, is_takeable, location_id, properties_json, state_json)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,  # noqa: E501
                    visitor_log_id,
                    "Visitor Log",
                    "A leather-bound book with a list of visitors.",
                    "The visitor log is open to the final day of operation. The neat handwriting deteriorates throughout the day, becoming increasingly frantic and illegible. The final entry simply reads 'THEY'RE COMING FROM INSIDE' with a pen stroke trailing off the page.",  # noqa: E501
                    "document",
                    True,
                    reception_id,
                    "{}",
                    "{}",
                )

                # Executive Suite items
                letter_id = uuid4()
                await conn.execute(
                    """
                    INSERT INTO objects (id, name, short_desc, full_desc, object_type, is_takeable, location_id, properties_json, state_json)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,  # noqa: E501
                    letter_id,
                    "Resignation Letter",
                    "An unsigned letter of resignation.",
                    "The typed letter begins formally but ends in a hasty scrawl: 'I can no longer be part of what we're doing here. The tests have gone too far. Something is happening to the employees on the 4th floor. If you're reading this, GET OUT NOW.'",  # noqa: E501
                    "document",
                    True,
                    north_id,
                    "{}",
                    "{}",
                )

                executive_pen_id = uuid4()
                await conn.execute(
                    """
                    INSERT INTO objects (id, name, short_desc, full_desc, object_type, is_takeable, location_id, properties_json, state_json)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,  # noqa: E501
                    executive_pen_id,
                    "Executive Pen",
                    "An expensive fountain pen with 'Monolith Corp' engraved on it.",
                    "The heavy gold pen feels surprisingly warm to the touch. The ink window shows a dark red liquid that doesn't quite look like ordinary ink. The company name 'Monolith Corp' is engraved along the side, along with their slogan: 'Reshaping Human Potential.'",  # noqa: E501
                    "tool",
                    True,
                    north_id,
                    "{}",
                    "{}",
                )

                # Server Room items
                usb_id = uuid4()
                await conn.execute(
                    """
                    INSERT INTO objects (id, name, short_desc, full_desc, object_type, is_takeable, location_id, properties_json, state_json)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,  # noqa: E501
                    usb_id,
                    "USB Drive",
                    "A black USB drive with a biohazard symbol on it.",
                    "The USB drive is unmarked except for a yellow biohazard symbol. It feels unusually heavy and sometimes seems to vibrate slightly, as if containing something barely contained. The metal connector has unusual corrosion patterns.",  # noqa: E501
                    "electronic",
                    True,
                    east_id,
                    "{}",
                    "{}",
                )

                server_manual_id = uuid4()
                await conn.execute(
                    """
                    INSERT INTO objects (id, name, short_desc, full_desc, object_type, is_takeable, location_id, properties_json, state_json)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,  # noqa: E501
                    server_manual_id,
                    "Server Manual",
                    "A thick technical manual with handwritten notes.",
                    "Standard server maintenance procedures are described in the first half of this manual, but the second half has been heavily annotated by hand. Diagrams show modifications to connect the servers to 'organic components.' One page is dog-eared, featuring instructions for 'consciousness transfer protocols' with a handwritten note: 'It worked, but at what cost?'",  # noqa: E501
                    "document",
                    True,
                    east_id,
                    "{}",
                    "{}",
                )

                # Cafeteria items
                energy_drink_id = uuid4()
                await conn.execute(
                    """
                    INSERT INTO objects (id, name, short_desc, full_desc, object_type, is_takeable, location_id, properties_json, state_json)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,  # noqa: E501
                    energy_drink_id,
                    "Energy Drink",
                    "A can of 'Productive!' brand energy drink.",
                    "The garish can promises '24-Hour Productivity Enhancement.' Despite being years old, the liquid inside still fizzes when you shake it. The ingredients list includes several chemical compounds with very long names and concludes with 'proprietary cognitive enhancers.' A warning label states it's for 'Monolith Employees Only' and 'Not For Human Consumption Outside Controlled Environments.'",  # noqa: E501
                    "consumable",
                    True,
                    south_id,
                    "{}",
                    "{}",
                )

                cafeteria_note_id = uuid4()
                await conn.execute(
                    """
                    INSERT INTO objects (id, name, short_desc, full_desc, object_type, is_takeable, location_id, properties_json, state_json)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,  # noqa: E501
                    cafeteria_note_id,
                    "Cafeteria Note",
                    "A handwritten note about strange food.",
                    "The hastily written note reads: 'Don't drink the coffee. Don't eat the Tuesday special. People who do start acting strange. Jenkins drank three cups yesterday and today his skin looks different. I'm keeping track of who eats what. There's a pattern here.'",  # noqa: E501
                    "document",
                    True,
                    south_id,
                    "{}",
                    "{}",
                )

                # Cubicle Farm items
                stapler_id = uuid4()
                await conn.execute(
                    """
                    INSERT INTO objects (id, name, short_desc, full_desc, object_type, is_takeable, location_id, properties_json, state_json)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,  # noqa: E501
                    stapler_id,
                    "Red Stapler",
                    "A bright red stapler, someone's prized possession.",
                    "This bright red Swingline stapler has been meticulously maintained, unlike most items in the abandoned office. Someone has etched their initials on the bottom and attached a note reading 'RETURN IF FOUND - THIS MEANS YOU MICHAEL!' The stapler feels unusually heavy and makes a strange clicking sound occasionally, even when not in use.",  # noqa: E501
                    "tool",
                    True,
                    west_id,
                    "{}",
                    "{}",
                )

                performance_review_id = uuid4()
                await conn.execute(
                    """
                    INSERT INTO objects (id, name, short_desc, full_desc, object_type, is_takeable, location_id, properties_json, state_json)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,  # noqa: E501
                    performance_review_id,
                    "Performance Review",
                    "An employee's concerning performance review.",
                    "This performance review for employee #2773 (name redacted) begins with standard metrics but becomes increasingly strange: 'Subject shows remarkable integration with new protocols. Cognitive enhancement at 86% above baseline. Minor side effects include sleeplessness, aggression, and occasional non-human vocalizations. Recommend promotion to Level 4 and increased monitoring. Potential candidate for full integration test.'",  # noqa: E501
                    "document",
                    True,
                    west_id,
                    "{}",
                    "{}",
                )

                # Add NPCs to the world
                security_guard_id = uuid4()
                await conn.execute(
                    """
                    INSERT INTO npcs (id, name, short_desc, full_desc, npc_type, personality_json, knowledge_json, location_id, state_json)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,  # noqa: E501
                    security_guard_id,
                    "Security Guard",
                    "A motionless security guard seated at the reception desk.",
                    "The security guard sits unnaturally still, facing away from you. He wears a faded uniform with the Monolith Corp logo. When you approach, he slowly turns, revealing a gaunt face with deeply sunken eyes. His movements are jerky, almost mechanical. 'Employees... must... check in,' he mumbles in a monotone voice, though there's no sign of any check-in system still operating. His name tag reads 'Davis,' and his skin has a grayish tint.",  # noqa: E501
                    "humanoid",
                    '{"traits": {"suspicious": true, "helpful": false}, "dialogue_state": "neutral"}',  # noqa: E501
                    "{}",
                    reception_id,
                    '{"dialogue_state": "neutral", "hostile": false}',
                )

                janitor_id = uuid4()
                await conn.execute(
                    """
                    INSERT INTO npcs (id, name, short_desc, full_desc, npc_type, personality_json, knowledge_json, location_id, state_json)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,  # noqa: E501
                    janitor_id,
                    "Janitor",
                    "An old janitor endlessly mopping the same spot.",
                    "The elderly janitor moves back and forth, mopping the same spot on the floor over and over. His jumpsuit has the name 'Frank' stitched on it. Despite the building's abandonment, his uniform is surprisingly clean. He hums a tune that sounds familiar but just off-key enough to be unsettling. When he notices you watching, he stops and stares with unnervingly alert eyes. 'They don't let anyone leave, you know,' he whispers, before returning to his endless task as if you'd never interacted.",  # noqa: E501
                    "humanoid",
                    '{"traits": {"observant": true, "cryptic": true}, "dialogue_state": "friendly"}',  # noqa: E501
                    '{"items": [{"topic": "Building Layout", "content": "There\'s a hidden room behind the vending machine in the cafeteria. That\'s where they took people who asked too many questions."}, {"topic": "Incident", "content": "It wasn\'t an accident. They were testing something new that day. Something they put in the water."}]}',  # noqa: E501
                    south_id,
                    '{"dialogue_state": "friendly", "hostile": false}',
                )

                programmer_id = uuid4()
                await conn.execute(
                    """
                    INSERT INTO npcs (id, name, short_desc, full_desc, npc_type, personality_json, knowledge_json, location_id, state_json)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    """,  # noqa: E501
                    programmer_id,
                    "Programmer",
                    "A disheveled programmer typing frantically at a dead computer.",
                    "The programmer's fingers fly over the keyboard of a computer that isn't even plugged in. Her eyes are fixed on the blank screen with intense concentration. Dark circles under her eyes suggest she hasn't slept in days, and her company ID badge is splattered with something that might be coffee—or blood. Her movements are erratic, and she occasionally mutters about 'fixing the code before it spreads.' When addressed, she looks up with bloodshot eyes: 'Are you real? They sent the avatars again, didn't they? I won't go back to the server room. YOU CAN'T MAKE ME GO BACK THERE.'",  # noqa: E501
                    "humanoid",
                    '{"traits": {"paranoid": true, "intelligent": true}, "dialogue_state": "hostile"}',  # noqa: E501
                    '{"items": [{"topic": "Server Room", "content": "The servers aren\'t processing data anymore. They\'re growing it. I saw something move inside the cooling tanks."}, {"topic": "Executive Suite", "content": "The executives knew what was happening. They were the first to change."}]}',  # noqa: E501
                    west_id,
                    '{"dialogue_state": "hostile", "hostile": true}',
                )

                # Create Location objects for the initial world state
                for _loc_name, loc_id in location_ids.items():
                    loc_data = await conn.fetchrow(
                        "SELECT * FROM locations WHERE id = $1", loc_id
                    )
                    if loc_data:
                        # Create basic Location object
                        location = Location(
                            location_id=loc_id,
                            name=loc_data["name"],
                            description=loc_data["full_desc"],
                        )

                        # Get connections for this location
                        connections = await conn.fetch(
                            """
                            SELECT direction, to_location_id FROM location_connections
                            WHERE from_location_id = $1
                            """,
                            loc_id,
                        )
                        for conn_data in connections:
                            location.connections[conn_data["direction"]] = conn_data[
                                "to_location_id"
                            ]

                        # Add objects for this location
                        objects = await conn.fetch(
                            "SELECT * FROM objects WHERE location_id = $1",
                            loc_id,
                        )
                        for obj_data in objects:
                            from game_loop.state.models import WorldObject

                            obj = WorldObject(
                                object_id=obj_data["id"],
                                name=obj_data["name"],
                                description=obj_data["full_desc"],
                                is_takeable=obj_data["is_takeable"],
                                is_container=obj_data.get("is_container", False),
                                is_hidden=obj_data.get("is_hidden", False),
                            )
                            location.objects[obj_data["id"]] = obj

                        # Add NPCs for this location
                        npcs = await conn.fetch(
                            "SELECT * FROM npcs WHERE location_id = $1", loc_id
                        )
                        for npc_data in npcs:
                            import json

                            from game_loop.state.models import (
                                NonPlayerCharacter,
                            )

                            # Parse JSON fields
                            (
                                json.loads(npc_data["personality_json"])
                                if npc_data.get("personality_json")
                                else {}
                            )
                            knowledge = (
                                json.loads(npc_data["knowledge_json"])
                                if npc_data.get("knowledge_json")
                                else {}
                            )
                            state = (
                                json.loads(npc_data["state_json"])
                                if npc_data.get("state_json")
                                else {}
                            )

                            # Create knowledge objects if available
                            knowledge_items = []
                            if "items" in knowledge:
                                from game_loop.state.models import (
                                    PlayerKnowledge,
                                )

                                for k_item in knowledge["items"]:
                                    knowledge_items.append(
                                        PlayerKnowledge(
                                            topic=k_item.get("topic", "general"),
                                            content=k_item["content"],
                                            source=f"From {npc_data['name']}",
                                        )
                                    )

                            npc = NonPlayerCharacter(
                                npc_id=npc_data["id"],
                                name=npc_data["name"],
                                description=npc_data["full_desc"],
                                dialogue_state=state.get("dialogue_state", "neutral"),
                                current_behavior=npc_data.get(
                                    "current_behavior", "idle"
                                ),
                                knowledge=knowledge_items,
                                state=state,
                            )
                            location.npcs[npc_data["id"]] = npc

                        # Add to initial locations dictionary
                        initial_locations[loc_id] = location

            # Set starting location to reception area
            starting_location_id = location_ids.get("reception")
            if not starting_location_id:
                logger.warning(
                    "Failed to find reception area, using first available location"
                )
                starting_location_id = (
                    next(iter(location_ids.values())) if location_ids else None
                )

            if not starting_location_id:
                raise RuntimeError("No locations created for the world")

            # 2. Create World State with the locations
            world_state = await self.world_tracker.create_new_world(initial_locations)
            if not world_state:
                raise RuntimeError("Failed to create new world state.")

            # 3. Create Player State with the starting location
            player_state = await self.player_tracker.create_new_player(
                player_name, starting_location_id
            )
            if not player_state:
                raise RuntimeError("Failed to create new player state.")

            # 4. Create Game Session linking the states
            self._current_session = await self.session_manager.create_session(
                player_state_id=player_state.player_id,
                world_state_id=world_state.world_id,
                save_name=save_name,
            )

            if not self._current_session:
                raise RuntimeError("Failed to create new game session.")

            # 5. Update trackers with the final session ID
            self.player_tracker._current_session_id = self._current_session.session_id
            self.world_tracker._current_session_id = self._current_session.session_id
            # Resave states with the correct session ID
            await self.player_tracker.save_state()
            await self.world_tracker.save_state()

            logger.info(
                f"New game created successfully. Session ID: "
                f"{self._current_session.session_id}"
            )
            return player_state, world_state

        except Exception as e:
            logger.exception(f"Error during new game creation: {e}")
            # TODO: Add cleanup logic if partial creation occurred
            # (e.g., delete created states/session)
            self._current_session = None
            return None, None

    async def _create_bidirectional_connection(
        self,
        conn: asyncpg.Connection,
        source_id: UUID,
        target_id: UUID,
        direction: str,
        return_direction: str,
    ) -> None:
        """Helper to create bidirectional connections between locations."""
        # Create forward connection
        await conn.execute(
            """
            INSERT INTO location_connections
            (from_location_id, to_location_id, connection_type, direction,
             is_visible, requirements_json)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            source_id,
            target_id,
            "path",  # Default connection type
            direction,
            True,
            "{}",
        )

        # Create return connection
        await conn.execute(
            """
            INSERT INTO location_connections
            (from_location_id, to_location_id, connection_type, direction,
             is_visible, requirements_json)
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            target_id,
            source_id,
            "path",  # Default connection type
            return_direction,
            True,
            "{}",
        )

    async def load_game(
        self, session_id: UUID
    ) -> tuple[PlayerState | None, WorldState | None]:
        """Loads a game session and associated player/world states."""
        logger.info(f"Loading game session {session_id}...")
        session = await self.session_manager.load_session(session_id)
        if not session:
            logger.error(f"Failed to load session metadata for {session_id}.")
            return None, None

        self._current_session = session
        player_state = await self.player_tracker.load_state(
            session.player_state_id, session.session_id
        )
        world_state = await self.world_tracker.load_state(
            session.world_state_id, session.session_id
        )

        if player_state and world_state:
            logger.info(f"Game session {session_id} loaded successfully.")
            return player_state, world_state
        else:
            logger.error(
                f"Failed to load player or world state for session {session_id}."
            )
            self._current_session = None  # Clear session if loading failed
            return None, None

    async def save_game(self, save_name: str | None = None) -> None:
        """Saves the current game state (player, world, session metadata)."""
        if not self._current_session:
            logger.error("Cannot save game: No active session.")
            return

        player_state = self.player_tracker.get_state()
        world_state = self.world_tracker.get_state()

        if not player_state or not world_state:
            logger.error("Cannot save game: Player or World state is not loaded.")
            return

        logger.info(f"Saving game for session {self._current_session.session_id}...")
        try:
            # Update save name if provided
            if save_name:
                self._current_session.save_name = save_name

            # Trigger saves in trackers (important if they don't auto-save)
            await self.player_tracker.save_state()
            await self.world_tracker.save_state()

            # Save the session metadata (updates timestamps, play time etc.)
            await self.session_manager.save_session(
                self._current_session, player_state, world_state
            )
            logger.info(
                f"Game saved successfully as '{self._current_session.save_name}'."
            )

        except Exception as e:
            logger.exception(
                f"Error saving game session {self._current_session.session_id}: {e}"
            )
            # Decide if error should be propagated

    async def update_after_action(self, action_result: ActionResult) -> None:
        """Updates player and world states based on an action result."""
        if not self._current_session:
            logger.warning("Cannot update state after action: No active session.")
            return

        logger.debug(f"Updating state after action: {action_result.feedback_message}")
        player_state = self.player_tracker.get_state()
        if not player_state:
            logger.warning(
                "Cannot update world state after action: Player state not loaded."
            )
            return

        # Update Player State first (e.g., inventory changes, stat changes)
        await self.player_tracker.update_from_action(action_result)

        # Update World State (e.g., object state changes, NPC changes)
        # Pass the player's current location ID for context
        current_location_id = await self.player_tracker.get_current_location_id()
        await self.world_tracker.update_from_action(action_result, current_location_id)

        # Decide on save strategy: Auto-save after every action?
        # Or only on explicit save command?
        # If auto-saving:
        # await self.save_game() # Might be too frequent

    def get_current_state(
        self,
    ) -> tuple[PlayerState | None, WorldState | None]:
        """Returns the current in-memory player and world states."""
        return self.player_tracker.get_state(), self.world_tracker.get_state()

    async def get_current_location_description(self) -> str | None:
        """Gets the description of the player's current location."""
        location_id = await self.player_tracker.get_current_location_id()
        if location_id:
            return await self.world_tracker.get_location_description(location_id)
        return "You are nowhere."  # Or None

    async def get_current_location_details(self) -> Location | None:
        """Gets the full details of the player's current location."""
        location_id = await self.player_tracker.get_current_location_id()
        if location_id:
            return await self.world_tracker.get_location_details(location_id)
        return None

    async def list_saved_games(self, limit: int = 10) -> list[GameSession]:
        """Lists available saved games."""
        return await self.session_manager.list_saved_games(limit)

    async def delete_saved_game(self, session_id: UUID) -> bool:
        """Deletes a specific saved game."""
        # Consider implications: If deleting the *current* session, what should happen?
        if self._current_session and self._current_session.session_id == session_id:
            logger.warning(
                f"Attempting to delete the currently active session {session_id}. "
                f"Clearing current session state."
            )
            # Reset internal state as the session is gone
            self._current_session = None
            await self.player_tracker.initialize(uuid4())  # Re-init with temp ID
            await self.world_tracker.initialize(uuid4())

        return await self.session_manager.delete_saved_game(session_id)

    async def shutdown(self) -> None:
        """Shuts down all managed state components."""
        logger.info("Shutting down GameStateManager...")
        # Decide if auto-save on shutdown is desired
        # if self._current_session:
        #     await self.save_game()

        await self.player_tracker.shutdown()
        await self.world_tracker.shutdown()
        await self.session_manager.shutdown()
        logger.info("GameStateManager shut down.")

    def get_current_session_id(self) -> UUID | None:
        """Returns the ID of the currently active session, if any."""
        return self._current_session.session_id if self._current_session else None
