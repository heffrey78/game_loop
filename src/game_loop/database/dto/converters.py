"""Data Transfer Object converters between Pydantic and SQLAlchemy models."""

import uuid  # Added uuid
from datetime import datetime
from typing import Any  # Added Optional, Dict

from pydantic import BaseModel, ConfigDict  # Added ConfigDict import

from game_loop.database.models import GameSession as SQLGameSession
from game_loop.database.models import Player as SQLPlayer
from game_loop.state.models import GameSession as PydanticGameSession
from game_loop.state.models import PlayerState as PydanticPlayerState


# --- DTO Definitions as per Step 4 Plan ---
class PlayerDTO(BaseModel):
    """Data Transfer Object for Player."""

    id: uuid.UUID
    name: str
    username: str
    level: int = 1
    created_at: datetime | None = None
    updated_at: datetime | None = None
    settings_json: dict[str, Any] | None = None
    current_location_id: uuid.UUID | None = None

    model_config = ConfigDict(from_attributes=True)


class GameSessionDTO(BaseModel):
    """Data Transfer Object for GameSession."""

    session_id: uuid.UUID
    player_id: uuid.UUID  # Added player_id
    player_state_id: uuid.UUID
    world_state_id: uuid.UUID
    save_name: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    game_version: str | None = None

    model_config = ConfigDict(from_attributes=True)


# --- End DTO Definitions ---


# Simple Player model for testing - matches test expectations
class Player:
    """Simple Player class for DTO testing."""

    def __init__(
        self, id: uuid.UUID | None = None, name: str | None = None, level: int = 1
    ) -> None:
        self.id = id
        self.name = name
        self.level = level


class GameSessionConverter:
    """Converts between Pydantic and SQLAlchemy GameSession models."""

    @staticmethod
    def pydantic_to_sqlalchemy(pydantic_session: PydanticGameSession) -> dict[str, Any]:
        """Convert Pydantic GameSession to SQLAlchemy GameSession kwargs."""
        return {
            "session_id": pydantic_session.session_id,
            "player_state_id": pydantic_session.player_state_id,
            "world_state_id": pydantic_session.world_state_id,
            "save_name": pydantic_session.save_name,
            "created_at": pydantic_session.created_at,
            "updated_at": pydantic_session.updated_at,
            "game_version": pydantic_session.game_version,
        }

    @staticmethod
    def sqlalchemy_to_pydantic(sql_session: SQLGameSession) -> PydanticGameSession:
        """Convert SQLAlchemy GameSession to Pydantic GameSession."""
        return PydanticGameSession(
            session_id=sql_session.session_id,
            player_state_id=sql_session.player_state_id,
            world_state_id=sql_session.world_state_id,
            save_name=sql_session.save_name,
            created_at=sql_session.created_at,
            updated_at=sql_session.updated_at,
            game_version=sql_session.game_version,
        )


class PlayerConverter:
    """Converts between Pydantic and SQLAlchemy Player models."""

    @staticmethod
    def pydantic_to_sqlalchemy(pydantic_player: PydanticPlayerState) -> dict[str, Any]:
        """Convert Pydantic PlayerState to SQLAlchemy Player kwargs."""
        return {
            "id": pydantic_player.player_id,
            "username": pydantic_player.name[:50],  # Truncate to schema limit
            "created_at": datetime.now(),
            "current_location_id": pydantic_player.current_location_id,
            "settings_json": {
                "state_data_json": pydantic_player.state_data_json,
                "stats": pydantic_player.stats.model_dump(),
                "progress": pydantic_player.progress.model_dump(),
            },  # Example structure, adjust as needed
        }

    @staticmethod
    def sqlalchemy_to_pydantic(sql_player: SQLPlayer) -> PydanticPlayerState:
        """Convert SQLAlchemy Player to Pydantic PlayerState."""
        state_data = (
            sql_player.settings_json.get("state_data_json")
            if sql_player.settings_json
            else None
        )
        return PydanticPlayerState(
            player_id=sql_player.id,
            name=sql_player.username,
            current_location_id=sql_player.current_location_id,
            state_data_json=state_data,
        )


class PlayerDTOConverter:
    """Converts between Player entities and DTOs."""

    @staticmethod
    def to_dto(player: SQLPlayer) -> PlayerDTO | None:
        """Converts an SQLPlayer ORM model to a PlayerDTO."""
        if not player:
            # Consider raising an error or returning a default PlayerDTO
            # For now, returning None if input is None, matching some patterns.
            return None

        player_data = {
            "id": player.id,
            "name": player.name,
            "username": player.username,
            "created_at": player.created_at,
            "updated_at": player.updated_at,
            "settings_json": player.settings_json or {},
            "current_location_id": player.current_location_id,
        }

        # Populate DTO fields from SQLPlayer's settings_json
        # Ensure types are compatible with PlayerDTO fields
        if player.settings_json:
            settings = player.settings_json
            player_data.update(
                {
                    "level": int(settings.get("level", 1)),
                    "experience": int(settings.get("experience", 0)),
                }
            )
        else:  # Ensure default values if settings_json is None or empty
            player_data.update({"level": 1})

        player_dto: PlayerDTO = PlayerDTO.model_validate(player_data)
        return player_dto

    @staticmethod
    def from_dto(dto: PlayerDTO) -> SQLPlayer | None:
        """Converts a PlayerDTO to an SQLPlayer ORM model instance."""
        if not dto:
            return None

        # Prepare settings_json for SQLPlayer from DTO fields
        settings_data = {
            "level": dto.level,
        }
        # If PlayerDTO itself has a settings_json field, merge it
        if dto.settings_json:
            settings_data.update(dto.settings_json)

        # SQLPlayer fields: id, name, username, created_at, updated_at,
        # settings_json, current_location_id.
        # TimestampMixin usually handles created_at/updated_at on the DB model.
        sql_player_instance = SQLPlayer(
            id=dto.id,
            name=dto.name,
            username=dto.username,
            current_location_id=dto.current_location_id,
            settings_json=settings_data,
        )
        # Note: created_at and updated_at from DTO are not set here,
        # assuming TimestampMixin on SQLPlayer manages these.
        return sql_player_instance


class GameStateDTOConverter:
    """Converts between GameSession entities and DTOs."""

    @staticmethod
    def to_dto(session: SQLGameSession) -> GameSessionDTO | None:
        """Converts an SQLGameSession ORM model to a GameSessionDTO."""
        if not session:
            return None
        return GameSessionDTO(
            session_id=session.session_id,
            player_id=session.player_id,
            player_state_id=session.player_state_id,
            world_state_id=session.world_state_id,
            save_name=session.save_name,
            created_at=session.created_at,
            updated_at=session.updated_at,
            game_version=session.game_version,
        )

    @staticmethod
    def from_dto(dto: GameSessionDTO) -> SQLGameSession:
        """Converts a GameSessionDTO to an SQLGameSession ORM model instance."""
        if not dto:
            return None  # type: ignore
        # created_at/updated_at are handled by TimestampMixin on SQLGameSession.
        return SQLGameSession(
            session_id=dto.session_id,
            player_id=dto.player_id,
            player_state_id=dto.player_state_id,
            world_state_id=dto.world_state_id,
            save_name=dto.save_name,
            game_version=dto.game_version,
        )
