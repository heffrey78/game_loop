"""Enhanced save/load system with multiple save slots and metadata."""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy.sql import select

from ...database.models.conversation import ConversationContext
from ...database.session_factory import DatabaseSessionFactory
from ...state.manager import GameStateManager
from ..models.system_models import LoadResult, SaveMetadata, SaveResult


class SaveManager:
    """Enhanced save/load system with multiple save slots and metadata."""

    def __init__(
        self,
        session_factory: DatabaseSessionFactory,
        game_state_manager: GameStateManager,
    ):
        self.session_factory = session_factory
        self.game_state = game_state_manager
        self.save_directory = Path("saves")
        self.auto_save_interval = 300  # 5 minutes
        self.max_auto_saves = 10
        self.max_save_slots = 20

        # Ensure save directory exists
        self.save_directory.mkdir(exist_ok=True)

    async def create_save(
        self,
        save_name: str | None = None,
        description: str | None = None,
    ) -> SaveResult:
        """Create a complete game save with metadata."""
        try:
            # Get current player ID from game state
            player_id = await self._get_current_player_id()
            if not player_id:
                return SaveResult(
                    success=False,
                    save_name=save_name or "unknown",
                    message="Failed to save game: No active player",
                    error="No player context available",
                )

            # Generate save name if not provided
            if not save_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_name = f"autosave_{timestamp}"

            # Get current game state
            game_state_data = await self._collect_game_state(player_id)
            if not game_state_data:
                return SaveResult(
                    success=False,
                    save_name=save_name,
                    message="Failed to collect game state",
                    error="No game state available",
                )

            # Create save file
            save_file_path = self.save_directory / f"{save_name}.json"
            save_data = {
                "player_id": str(player_id),
                "save_name": save_name,
                "description": description or "",
                "created_at": datetime.now().isoformat(),
                "game_state": game_state_data,
            }

            # Write to file
            with open(save_file_path, "w", encoding="utf-8") as f:
                json.dump(save_data, f, indent=2)

            file_size = save_file_path.stat().st_size

            # Create metadata
            metadata = SaveMetadata(
                save_name=save_name,
                description=description or "",
                created_at=datetime.now(),
                file_size=file_size,
                player_level=game_state_data.get("player_level", 1),
                location=game_state_data.get("current_location", "Unknown"),
                play_time=timedelta(seconds=game_state_data.get("play_time", 0)),
                player_id=player_id,
                file_path=str(save_file_path),
            )

            # Store in database
            await self._store_save_metadata(metadata, save_data)

            return SaveResult(
                success=True,
                save_name=save_name,
                message=f"Game saved as '{save_name}'",
                save_metadata=metadata,
            )

        except Exception as e:
            return SaveResult(
                success=False,
                save_name=save_name or "unknown",
                message="Failed to save game",
                error=str(e),
            )

    async def load_save(self, save_name: str) -> LoadResult:
        """Load a complete game save and restore state."""
        try:
            # Get current player ID from game state
            player_id = await self._get_current_player_id()
            if not player_id:
                return LoadResult(
                    success=False,
                    save_name=save_name,
                    message="Failed to load game: No active player",
                    error="No player context available",
                )

            # Check if save exists in database
            save_metadata = await self._get_save_metadata(player_id, save_name)
            if not save_metadata:
                return LoadResult(
                    success=False,
                    save_name=save_name,
                    message=f"Save '{save_name}' not found",
                    error="Save file does not exist",
                )

            # Load save file
            save_file_path = (
                Path(save_metadata.file_path) if save_metadata.file_path else None
            )
            if not save_file_path or not save_file_path.exists():
                # Try alternative path
                save_file_path = self.save_directory / f"{save_name}.json"
                if not save_file_path.exists():
                    return LoadResult(
                        success=False,
                        save_name=save_name,
                        message=f"Save file for '{save_name}' not found",
                        error="Save file missing from disk",
                    )

            # Read save data
            with open(save_file_path, encoding="utf-8") as f:
                save_data = json.load(f)

            # Validate save data
            if not self._validate_save_data(save_data):
                return LoadResult(
                    success=False,
                    save_name=save_name,
                    message=f"Save '{save_name}' is corrupted",
                    error="Invalid save data format",
                )

            # Restore game state
            game_state = save_data["game_state"]
            await self._restore_game_state(player_id, game_state)

            return LoadResult(
                success=True,
                save_name=save_name,
                message=f"Game loaded from '{save_name}'",
                game_state=game_state,
            )

        except Exception as e:
            return LoadResult(
                success=False,
                save_name=save_name,
                message=f"Failed to load '{save_name}'",
                error=str(e),
            )

    async def list_saves(self) -> list[SaveMetadata]:
        """List all available saves with metadata."""
        try:
            # Get current player ID from game state
            player_id = await self._get_current_player_id()
            if not player_id:
                return []

            async with self.session_factory.get_session() as session:
                result = await session.execute(
                    """
                    SELECT save_id, save_name, player_id, description, file_path, 
                           file_size, player_level, location, play_time, created_at
                    FROM game_saves 
                    WHERE player_id = $1
                    ORDER BY created_at DESC
                    """,
                    (player_id,),
                )
                saves = []

                for row in result:
                    save_metadata = SaveMetadata(
                        save_id=row[0],
                        save_name=row[1],
                        player_id=row[2],
                        description=row[3] or "",
                        file_path=row[4],
                        file_size=row[5] or 0,
                        player_level=row[6] or 1,
                        location=row[7] or "Unknown",
                        play_time=row[8] or timedelta(0),
                        created_at=row[9],
                    )
                    saves.append(save_metadata)

                return saves

        except Exception:
            return []

    async def delete_save(self, save_name: str) -> bool:
        """Delete a save file."""
        try:
            # Get current player ID from game state
            player_id = await self._get_current_player_id()
            if not player_id:
                return False

            # Get save metadata
            save_metadata = await self._get_save_metadata(player_id, save_name)
            if not save_metadata:
                return False

            # Delete file
            if save_metadata.file_path:
                save_file_path = Path(save_metadata.file_path)
                if save_file_path.exists():
                    save_file_path.unlink()

            # Delete from database
            async with self.session_factory.get_session() as session:
                result = await session.execute(
                    """
                    DELETE FROM game_saves 
                    WHERE player_id = $1 AND save_name = $2
                    """,
                    (player_id, save_name),
                )
                await session.commit()

                if result.rowcount > 0:
                    return True
                else:
                    return False

        except Exception:
            return False

    async def auto_save(self) -> SaveResult:
        """Perform automatic save with rotation."""
        try:
            # Get current player ID from game state
            player_id = await self._get_current_player_id()
            if not player_id:
                return SaveResult(
                    success=False,
                    save_name="autosave",
                    message="Auto-save failed: No active player",
                    error="No player context available",
                )

            # Create auto-save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            auto_save_name = f"autosave_{timestamp}"

            result = await self.create_save(
                save_name=auto_save_name,
                description="Automatic save",
            )

            if result.success:
                # Clean up old auto-saves
                await self._cleanup_auto_saves(player_id)

            return result

        except Exception as e:
            return SaveResult(
                success=False,
                save_name="autosave",
                message="Auto-save failed",
                error=str(e),
            )

    def generate_save_summary(self, game_state: dict[str, Any]) -> str:
        """Generate human-readable save summary."""
        location = game_state.get("current_location", "Unknown Location")
        level = game_state.get("player_level", 1)
        play_time = game_state.get("play_time", 0)

        # Convert play time to readable format
        hours = int(play_time // 3600)
        minutes = int((play_time % 3600) // 60)
        time_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"

        return f"Level {level} at {location} (Playtime: {time_str})"

    async def _collect_game_state(self, player_id: uuid.UUID) -> dict[str, Any] | None:
        """Collect complete game state for saving."""
        try:
            # Get player state
            player_state = await self.game_state.get_player_state(player_id)
            world_state = await self.game_state.get_world_state()

            if not player_state:
                return None

            # Collect conversation states
            conversation_states = await self._collect_conversation_states(player_id)

            game_state = {
                "player_state": (
                    player_state.to_dict()
                    if hasattr(player_state, "to_dict")
                    else dict(player_state)
                ),
                "world_state": (
                    world_state.to_dict()
                    if hasattr(world_state, "to_dict")
                    else dict(world_state)
                ),
                "conversation_states": conversation_states,
                "save_version": "1.0",
                "timestamp": datetime.now().isoformat(),
            }

            return game_state

        except Exception:
            return None

    async def _collect_conversation_states(
        self, player_id: uuid.UUID
    ) -> list[dict[str, Any]]:
        """Collect active conversation states."""
        try:
            async with self.session_factory.get_session() as session:
                stmt = select(ConversationContext).where(
                    ConversationContext.player_id == player_id,
                    ConversationContext.status == "active",
                )
                result = await session.execute(stmt)
                conversations = result.scalars().all()

                return [conv.to_dict() for conv in conversations]

        except Exception:
            return []

    async def _restore_game_state(
        self, player_id: uuid.UUID, game_state: dict[str, Any]
    ) -> bool:
        """Restore game state from save data."""
        try:
            # Restore player state
            if "player_state" in game_state:
                await self.game_state.restore_player_state(
                    player_id, game_state["player_state"]
                )

            # Restore world state
            if "world_state" in game_state:
                await self.game_state.restore_world_state(game_state["world_state"])

            # TODO: Restore conversation states when conversation manager supports it

            return True

        except Exception:
            return False

    def validate_save_data(self, save_data: dict[str, Any] | None) -> bool:
        """Validate save data structure."""
        if save_data is None:
            return False
        if not isinstance(save_data, dict):
            return False
        required_fields = ["player_id", "save_name", "created_at", "game_state"]
        return all(field in save_data for field in required_fields)

    def _validate_save_data(self, save_data: dict[str, Any]) -> bool:
        """Private wrapper for backward compatibility."""
        return self.validate_save_data(save_data)

    async def _store_save_metadata(
        self, metadata: SaveMetadata, save_data: dict[str, Any]
    ) -> None:
        """Store save metadata in database."""
        async with self.session_factory.get_session() as session:
            # Note: This would use the actual table model in a real implementation
            # For now, we'll use raw SQL
            await session.execute(
                """
                INSERT INTO game_saves 
                (save_id, save_name, player_id, description, save_data, metadata, 
                 file_path, file_size, player_level, location, play_time)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (player_id, save_name) 
                DO UPDATE SET
                    save_data = EXCLUDED.save_data,
                    metadata = EXCLUDED.metadata,
                    file_path = EXCLUDED.file_path,
                    file_size = EXCLUDED.file_size,
                    player_level = EXCLUDED.player_level,
                    location = EXCLUDED.location,
                    play_time = EXCLUDED.play_time,
                    created_at = CURRENT_TIMESTAMP
                """,
                (
                    metadata.save_id,
                    metadata.save_name,
                    metadata.player_id,
                    metadata.description,
                    json.dumps(save_data),
                    json.dumps(metadata.to_dict()),
                    metadata.file_path,
                    metadata.file_size,
                    metadata.player_level,
                    metadata.location,
                    metadata.play_time,
                ),
            )
            await session.commit()

    async def _get_save_metadata(
        self, player_id: uuid.UUID, save_name: str
    ) -> SaveMetadata | None:
        """Get save metadata from database."""
        try:
            async with self.session_factory.get_session() as session:
                result = await session.execute(
                    """
                    SELECT save_id, save_name, player_id, description, file_path, 
                           file_size, player_level, location, play_time, created_at
                    FROM game_saves 
                    WHERE player_id = $1 AND save_name = $2
                    """,
                    (player_id, save_name),
                )
                row = result.fetchone()

                if row:
                    return SaveMetadata(
                        save_id=row[0],
                        save_name=row[1],
                        player_id=row[2],
                        description=row[3] or "",
                        file_path=row[4],
                        file_size=row[5] or 0,
                        player_level=row[6] or 1,
                        location=row[7] or "Unknown",
                        play_time=row[8] or timedelta(0),
                        created_at=row[9],
                    )

                return None

        except Exception:
            return None

    async def cleanup_old_saves(
        self, player_id: uuid.UUID, max_saves: int | None = None
    ) -> None:
        """Clean up old save files beyond the maximum limit."""
        try:
            max_limit = max_saves or self.max_save_slots
            async with self.session_factory.get_session() as session:
                # Get saves ordered by creation date (oldest first after limit)
                result = await session.execute(
                    """
                    SELECT save_name, file_path 
                    FROM game_saves 
                    WHERE player_id = $1
                    ORDER BY created_at DESC
                    OFFSET $2
                    """,
                    (player_id, max_limit),
                )

                # Delete excess saves
                for row in result:
                    save_name, file_path = row
                    await self.delete_save(player_id, save_name)

        except Exception:
            pass  # Non-critical operation

    async def _cleanup_auto_saves(self, player_id: uuid.UUID) -> None:
        """Clean up old auto-save files."""
        try:
            async with self.session_factory.get_session() as session:
                # Get auto-saves ordered by creation date
                result = await session.execute(
                    """
                    SELECT save_name, file_path 
                    FROM game_saves 
                    WHERE player_id = $1 AND save_name LIKE 'autosave_%'
                    ORDER BY created_at DESC
                    OFFSET $2
                    """,
                    (player_id, self.max_auto_saves),
                )

                # Delete excess auto-saves
                for row in result:
                    save_name, file_path = row
                    await self.delete_save(player_id, save_name)

        except Exception:
            pass  # Non-critical operation

    async def _get_current_player_id(self) -> uuid.UUID | None:
        """Get current player ID from game state manager."""
        try:
            # This depends on how GameStateManager tracks current player
            # For now, assume there's a method to get current player
            player_state = await self.game_state.get_current_player_state()
            if player_state and hasattr(player_state, "player_id"):
                return player_state.player_id
            elif player_state and hasattr(player_state, "id"):
                return player_state.id
            else:
                # Fallback - try to get a default player ID
                return None
        except Exception:
            return None

    def _get_save_table(self):
        """Get save table for queries (placeholder for actual table model)."""
        # This would return the actual SQLAlchemy table model
        # For now, using raw queries
        return None
