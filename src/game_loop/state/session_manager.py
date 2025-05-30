"""Manages game sessions, including saving, loading, and listing."""

import logging
from datetime import datetime
from uuid import UUID

import asyncpg
from pydantic import ValidationError

from .models import GameSession, PlayerState, WorldState

logger = logging.getLogger(__name__)


class SessionManager:
    """Handles the creation, loading, saving, and deletion of game sessions."""

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    async def create_session(
        self, player_state_id: UUID, world_state_id: UUID, save_name: str = "New Game"
    ) -> GameSession:
        """Creates a new game session record in the database."""
        new_session = GameSession(
            player_state_id=player_state_id,
            world_state_id=world_state_id,
            save_name=save_name,
            # game_version can be fetched from config or set here
        )

        async with self.db_pool.acquire() as conn:
            try:
                # First check or create player in players table
                player_row = await conn.fetchrow(
                    """
                    SELECT id FROM players WHERE username = $1
                    """,
                    str(player_state_id)[:50],
                )

                # If no player entry exists, create one
                if not player_row:
                    player_id = await conn.fetchval(
                        """
                        INSERT INTO players (username, created_at, settings_json)
                        VALUES ($1, CURRENT_TIMESTAMP, $2) RETURNING id
                        """,
                        str(player_state_id)[
                            :50
                        ],  # Username has a 50-char limit in schema
                        "{}",  # Default empty settings
                    )
                else:
                    player_id = player_row["id"]

                # Check if the table has necessary columns
                columns = await conn.fetch(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'game_sessions'
                """
                )
                column_names = {col["column_name"] for col in columns}

                # Adapt the INSERT statement based on available columns
                if "created_at" in column_names and "updated_at" in column_names:
                    # New schema with created_at and updated_at
                    await conn.execute(
                        """
                        INSERT INTO game_sessions (session_id,
                        player_id,
                        player_state_id,
                        world_state_id,
                        save_name,
                        created_at,
                        updated_at,
                        game_version,
                        game_time)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        """,
                        new_session.session_id,
                        player_id,  # Use the player ID from players table
                        new_session.player_state_id,
                        new_session.world_state_id,
                        new_session.save_name,
                        new_session.created_at,
                        new_session.updated_at,
                        new_session.game_version,
                        0,  # Initial game time
                    )
                else:
                    # Old schema with started_at field
                    await conn.execute(
                        """
                        INSERT INTO game_sessions (session_id,
                        player_id,
                        player_state_id,
                        world_state_id,
                        save_name,
                        started_at,
                        game_version,
                        game_time)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """,
                        new_session.session_id,
                        player_id,  # Use the player ID from players table
                        new_session.player_state_id,
                        new_session.world_state_id,
                        new_session.save_name,
                        new_session.created_at,  # Use created_at for started_at
                        new_session.game_version,
                        0,  # Initial game time
                    )

                logger.info(
                    f"Created new game session {new_session.session_id} ('{save_name}')"
                )
                return new_session
            except Exception as e:
                logger.error(f"Error creating game session: {e}")
                raise

    async def load_session(self, session_id: UUID) -> GameSession | None:
        """Loads session metadata from the database."""
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM game_sessions WHERE session_id = $1", session_id
            )
            if row:
                try:
                    # Convert row to dict and validate with Pydantic
                    session_data = dict(row)
                    session = GameSession(**session_data)
                    logger.info(f"Loaded session metadata for {session_id}")
                    return session
                except ValidationError as e:
                    logger.error(f"Error validating session data for {session_id}: {e}")
                    return None
                except Exception as e:
                    logger.error(f"Error loading session {session_id}: {e}")
                    return None
            else:
                logger.warning(f"Game session {session_id} not found.")
                return None

    async def save_session(
        self, session: GameSession, player_state: PlayerState, world_state: WorldState
    ) -> None:
        """
        Saves the session metadata, player state, and world state.
        This assumes PlayerStateTracker and WorldStateTracker handle their own saving,
        and this method primarily updates the session metadata (like updated_at).
        If trackers don't auto-save, this method should trigger their save methods.
        """
        if not session:
            logger.error("Cannot save session: Session object is None.")
            return

        # Update session metadata before saving
        session.updated_at = datetime.now()

        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                try:
                    # 1. Save Player State (assuming tracker handles DB interaction)
                    if player_state.player_id != session.player_state_id:
                        logger.warning(
                            f"Player state ID mismatch during save: Session expects "
                            f"{session.player_state_id}, got {player_state.player_id}"
                        )

                        session.player_state_id = player_state.player_id

                    # Find or create appropriate player_id from players table
                    player_row = await conn.fetchrow(
                        """
                        SELECT id FROM players WHERE username = $1
                        """,
                        str(player_state.player_id)[:50],
                    )

                    if not player_row:
                        player_id = await conn.fetchval(
                            """
                            INSERT INTO players (username, created_at, settings_json)
                            VALUES ($1, CURRENT_TIMESTAMP, $2) RETURNING id
                            """,
                            str(player_state.player_id)[:50],
                            "{}",  # Default empty settings
                        )
                    else:
                        player_id = player_row["id"]

                    # 2. Save World State (assuming tracker handles DB interaction)
                    if world_state.world_id != session.world_state_id:
                        logger.warning(
                            f"World state ID mismatch during save: Session expects "
                            f"{session.world_state_id}, got {world_state.world_id}"
                        )
                        session.world_state_id = world_state.world_id

                    # 3. Update Session Metadata in DB
                    await conn.execute(
                        """
                        UPDATE game_sessions
                        SET player_id = $2,
                            player_state_id = $3,
                            world_state_id = $4,
                            save_name = $5,
                            updated_at = $6,
                            game_version = $7
                        WHERE session_id = $1
                        """,
                        session.session_id,
                        player_id,
                        session.player_state_id,
                        session.world_state_id,
                        session.save_name,
                        session.updated_at,
                        session.game_version,
                    )
                    logger.info(
                        f"Updated game session metadata for "
                        f"{session.session_id} ('{session.save_name}')"
                    )

                except Exception as e:
                    logger.error(f"Error saving game session {session.session_id}: {e}")
                    raise  # Re-raise after logging

    async def list_saved_games(self, limit: int = 10) -> list[GameSession]:
        """Lists available saved game sessions, ordered by most recently updated."""
        sessions = []
        async with self.db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM game_sessions ORDER BY updated_at DESC LIMIT $1", limit
            )
            for row in rows:
                try:
                    session_data = dict(row)
                    sessions.append(GameSession(**session_data))
                except ValidationError as e:
                    logger.warning(f"Skipping invalid session data during listing: {e}")
        logger.info(f"Retrieved {len(sessions)} saved games.")
        return sessions

    async def delete_saved_game(self, session_id: UUID) -> bool:
        """
        Deletes a game session record and associated state data.
        Note: This requires handling of foreign key constraints or cascading deletes.
        The current implementation only deletes the session record.
        Deleting associated player/world states might need separate calls or DB setup.
        """
        async with self.db_pool.acquire() as conn:
            async with conn.transaction():
                try:
                    # Delete the session record
                    result = await conn.execute(
                        "DELETE FROM game_sessions WHERE session_id = $1", session_id
                    )
                    deleted_count = int(
                        result.split()[-1]
                    )  # Parse the count from 'DELETE N' string
                    if deleted_count > 0:
                        logger.info(f"Deleted game session {session_id}")
                        return True
                    else:
                        logger.warning(
                            f"Attempted to delete non-existent "
                            f"game session {session_id}"
                        )
                        return False
                except Exception as e:
                    logger.error(f"Error deleting game session {session_id}: {e}")
                    raise

    async def shutdown(self) -> None:
        """Perform any cleanup if needed."""
        logger.info("Shutting down SessionManager...")
        # No specific actions needed currently
        logger.info("SessionManager shut down.")
