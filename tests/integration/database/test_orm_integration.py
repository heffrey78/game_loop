"""
Integration tests for SQLAlchemy ORM implementation.

These tests validate the complete ORM integration including:
- Database schema synchronization
- Model CRUD operations
- Repository pattern implementation
- DTO conversion between Pydantic and SQLAlchemy models
- Vector operations with pgvector
- Transaction handling
"""

import asyncio  # For sleep
import uuid
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from game_loop.config.models import DatabaseConfig
from game_loop.database.dto.converters import (
    GameSessionDTO,
    GameStateDTOConverter,
    PlayerDTO,
    PlayerDTOConverter,
)
from game_loop.database.models.game_state import GameSession as SQLGameSession
from game_loop.database.models.player import Player as SQLPlayer
from game_loop.database.models.world import (
    Location as SQLLocation,
)
from game_loop.database.models.world import (
    Region as SQLRegion,
)
from game_loop.database.repositories.game_state import GameSessionRepository
from game_loop.database.repositories.player import PlayerRepository
from game_loop.database.repositories.world import LocationRepository, RegionRepository
from game_loop.database.session_factory import DatabaseSessionFactory

# Unused state models were removed in a previous step.


@pytest.mark.integration
class TestDatabaseInfrastructure:
    """Test basic database infrastructure and connectivity."""

    @pytest.mark.asyncio
    async def test_database_connection(self, db_session: AsyncSession) -> None:
        """Test basic database connectivity."""
        result = await db_session.execute(text("SELECT 1 as test"))
        assert result.scalar() == 1

    @pytest.mark.asyncio
    async def test_pgvector_extension(self, db_session: AsyncSession) -> None:
        """Test that pgvector extension is available."""
        result = await db_session.execute(
            text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
        )
        has_vector = result.scalar()

        if not has_vector:
            pytest.skip("pgvector extension not installed")

        result = await db_session.execute(
            text("SELECT '[1,2,3]'::vector <-> '[4,5,6]'::vector as distance")
        )
        distance = result.scalar()
        assert distance is not None
        assert isinstance(distance, int | float)

    @pytest.mark.asyncio
    async def test_uuid_extension(self, db_session: AsyncSession) -> None:
        """Test that uuid-ossp extension is available."""
        result = await db_session.execute(
            text(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'uuid-ossp')"
            )
        )
        has_uuid = result.scalar()

        if not has_uuid:
            pytest.skip("uuid-ossp extension not installed")

        result = await db_session.execute(text("SELECT uuid_generate_v4()"))
        generated_uuid = result.scalar()
        assert generated_uuid is not None
        assert isinstance(generated_uuid, uuid.UUID)


@pytest.mark.integration
class TestSQLAlchemyModels:
    """Test SQLAlchemy model functionality."""

    @pytest.mark.asyncio
    async def test_timestamp_mixin(self, db_session: AsyncSession) -> None:
        """Test that timestamp mixin works correctly."""
        region = SQLRegion(
            name="Test Region",
            description="A test region for timestamp testing",
            theme="test",
        )
        assert region.created_at is None
        assert region.updated_at is None

        db_session.add(region)
        await db_session.flush()

        assert region.created_at is not None
        assert region.updated_at is not None
        assert isinstance(region.created_at, datetime)
        assert isinstance(region.updated_at, datetime)

        original_created = region.created_at
        original_updated = region.updated_at

        await asyncio.sleep(0.01)

        region.description = "Updated description"
        await db_session.flush()
        await db_session.refresh(region)

        assert region.created_at == original_created
        assert region.updated_at > original_updated

    @pytest.mark.asyncio
    async def test_uuid_primary_keys(self, db_session: AsyncSession) -> None:
        """Test that UUID primary keys are generated correctly."""
        player = SQLPlayer(name="Test Player", username="uuid_user_models")
        assert player.id is None
        db_session.add(player)
        await db_session.flush()
        assert player.id is not None
        assert isinstance(player.id, uuid.UUID)

    @pytest.mark.asyncio
    async def test_vector_columns(self, db_session: AsyncSession) -> None:
        """Test vector column functionality with pgvector."""
        result = await db_session.execute(
            text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
        )
        if not result.scalar():
            pytest.skip("pgvector extension not installed")

        region = SQLRegion(
            name="Vector Test Region Models",
            description="A region for vector testing in models",
            theme="test_models",
        )
        db_session.add(region)
        await db_session.flush()

        test_embedding = [0.1] * 384
        location = SQLLocation(
            name="Vector Test Location Models",
            short_desc="A location for vector testing in models",
            full_desc="Detailed description for vector test location in models",
            region_id=region.id,
            location_type="room_models",
            is_dynamic=False,
            created_by="system_models",
            location_embedding=test_embedding,
        )
        db_session.add(location)
        await db_session.flush()

        assert location.location_embedding is not None
        assert len(location.location_embedding) == 384
        assert all(abs(val - 0.1) < 1e-6 for val in location.location_embedding)


@pytest.mark.integration
class TestRepositoryPattern:
    """Test repository pattern implementation."""

    @pytest.mark.asyncio
    async def test_player_repository_crud(self, db_session: AsyncSession) -> None:
        """Test complete CRUD operations for Player repository."""
        repo = PlayerRepository(db_session)

        player_data = SQLPlayer(
            name="Test Player Repo", username="test_repo_user", level=1
        )

        created_player = await repo.create(player_data)
        assert created_player.id is not None
        assert created_player.name == "Test Player Repo"
        assert created_player.username == "test_repo_user"
        assert created_player.level == 1

        found_player = await repo.get_by_id(created_player.id)
        assert found_player is not None
        assert found_player.name == "Test Player Repo"
        assert found_player.level == 1

        updated_player = await repo.update(found_player, level=2)
        assert updated_player.level == 2

        all_players = await repo.get_all()
        assert len(all_players) >= 1
        assert any(p.id == created_player.id for p in all_players)

        delete_successful = await repo.delete(created_player.id)
        assert delete_successful is True
        deleted_player = await repo.get_by_id(created_player.id)
        assert deleted_player is None

    @pytest.mark.asyncio
    async def test_region_location_relationship(self, db_session: AsyncSession) -> None:
        """Test relationships between regions and locations."""
        region_repo = RegionRepository(db_session)
        location_repo = LocationRepository(db_session)

        region_data = SQLRegion(
            name="Test Region Repo",
            description="A test region for relationship testing in repo",
            theme="fantasy_repo",
        )
        region = await region_repo.create(region_data)
        await db_session.flush()  # Ensure region.id is populated

        location1_data = SQLLocation(
            name="Location 1 Repo",
            short_desc="First test location in repo",
            full_desc="Detailed description of first location in repo",
            region_id=region.id,
            location_type="room_repo",
            is_dynamic=False,
            created_by="system_repo",
        )
        location2_data = SQLLocation(
            name="Location 2 Repo",
            short_desc="Second test location in repo",
            full_desc="Detailed description of second location in repo",
            region_id=region.id,
            location_type="outdoor_repo",
            is_dynamic=False,
            created_by="system_repo",
        )
        location1 = await location_repo.create(location1_data)
        location2 = await location_repo.create(location2_data)

        assert location1.region_id == region.id
        assert location2.region_id == region.id

        region_locations = await location_repo.find_by_region(region.id)
        assert len(region_locations) == 2
        location_names = {loc.name for loc in region_locations}
        assert "Location 1 Repo" in location_names
        assert "Location 2 Repo" in location_names

    @pytest.mark.asyncio
    async def test_game_session_repository(self, db_session: AsyncSession) -> None:
        """Test GameSession repository functionality."""
        repo = GameSessionRepository(db_session)
        player_repo = PlayerRepository(db_session)

        # Create a player for FK constraint
        test_player = SQLPlayer(name="Session Test Player", username="session_user")
        await player_repo.create(test_player)
        await db_session.flush()  # Ensure test_player.id is populated

        session_data = SQLGameSession(
            session_id=uuid4(),
            player_id=test_player.id,
            player_state_id=uuid4(),
            world_state_id=uuid4(),
            save_name="Test Save Repo",
            game_version="0.1.0_repo",
        )

        session = await repo.create(session_data)
        assert session.session_id is not None
        assert session.save_name == "Test Save Repo"

        found_session = await repo.get_by_id(session.session_id)
        assert found_session is not None
        assert found_session.session_id == session.session_id

        found_session.save_name = "Updated Save Repo"
        updated_session = await repo.update(found_session)
        assert updated_session.save_name == "Updated Save Repo"


@pytest.mark.integration
class TestDTOConverters:
    """Test DTO conversion between Pydantic and SQLAlchemy models."""

    @pytest.mark.asyncio
    async def test_player_dto_conversion(self, db_session: AsyncSession) -> None:
        """Test Player DTO conversion."""
        converter = PlayerDTOConverter()

        player_dto = PlayerDTO(
            id=uuid4(),
            name="Test Player DTO",
            username="test_dto_user",
            level=5,
            experience=1000,
            health=85,
            max_health=100,
            energy=75,
            max_energy=100,
            gold=250,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        sql_player_instance = converter.from_dto(player_dto)
        assert isinstance(sql_player_instance, SQLPlayer)
        assert sql_player_instance.id == player_dto.id
        assert sql_player_instance.name == player_dto.name
        assert sql_player_instance.username == player_dto.username
        assert sql_player_instance.settings_json is not None
        assert sql_player_instance.settings_json.get("level") == player_dto.level

        converted_back_dto = converter.to_dto(sql_player_instance)
        assert isinstance(converted_back_dto, PlayerDTO)
        assert converted_back_dto.id == player_dto.id
        assert converted_back_dto.name == player_dto.name
        assert converted_back_dto.username == player_dto.username
        assert converted_back_dto.level == player_dto.level

    @pytest.mark.asyncio
    async def test_game_session_dto_conversion(self, db_session: AsyncSession) -> None:
        """Test GameSession DTO conversion."""
        converter = GameStateDTOConverter()

        session_id_val = uuid4()
        player_id_val = uuid4()  # For GameSessionDTO
        player_state_id_val = uuid4()
        world_state_id_val = uuid4()

        game_session_dto = GameSessionDTO(
            session_id=session_id_val,
            player_id=player_id_val,
            player_state_id=player_state_id_val,
            world_state_id=world_state_id_val,
            save_name="Test Session DTO",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            game_version="0.1.1",
        )

        sql_session_instance = converter.from_dto(game_session_dto)
        assert isinstance(sql_session_instance, SQLGameSession)
        assert sql_session_instance.session_id == game_session_dto.session_id
        assert sql_session_instance.player_id == game_session_dto.player_id
        assert sql_session_instance.player_state_id == (
            game_session_dto.player_state_id
        )
        assert sql_session_instance.world_state_id == (game_session_dto.world_state_id)
        assert sql_session_instance.save_name == game_session_dto.save_name
        assert sql_session_instance.game_version == game_session_dto.game_version

        converted_back_dto = converter.to_dto(sql_session_instance)
        assert isinstance(converted_back_dto, GameSessionDTO)
        assert converted_back_dto.session_id == game_session_dto.session_id
        assert converted_back_dto.player_id == game_session_dto.player_id
        assert converted_back_dto.player_state_id == (game_session_dto.player_state_id)
        assert converted_back_dto.world_state_id == (game_session_dto.world_state_id)
        assert converted_back_dto.save_name == game_session_dto.save_name
        assert converted_back_dto.game_version == game_session_dto.game_version


@pytest.mark.integration
class TestSessionFactory:
    """Test SessionFactory functionality."""

    @pytest.mark.asyncio
    async def test_session_factory_lifecycle(
        self, database_config: DatabaseConfig  # Use DatabaseConfig type hint
    ) -> None:
        """Test complete SessionFactory lifecycle."""
        factory = DatabaseSessionFactory(database_config)
        await factory.initialize()
        async with factory.get_session() as session:
            assert isinstance(session, AsyncSession)
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1
        await factory.close()

    @pytest.mark.asyncio
    async def test_transaction_rollback(self, db_session: AsyncSession) -> None:
        """Test transaction rollback functionality."""
        repo = PlayerRepository(db_session)
        player_id_to_check = None
        try:
            player_data = SQLPlayer(
                name="Rollback Test Player",
                username="rollback_user_orm",
                level=1,
            )
            created_player = await repo.create(player_data)
            player_id_to_check = created_player.id

            found_player = await repo.get_by_id(player_id_to_check)
            assert found_player is not None

            await db_session.rollback()

            found_player_after_rollback = await repo.get_by_id(player_id_to_check)
            assert found_player_after_rollback is None

        except Exception:
            await db_session.rollback()
            raise


@pytest.mark.integration
class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""

    @pytest.mark.asyncio
    async def test_bulk_operations(self, db_session: AsyncSession) -> None:
        """Test bulk insert and query operations."""
        repo = PlayerRepository(db_session)

        players_data = []
        for i in range(10):
            player_data = SQLPlayer(
                name=f"Bulk Test Player {i}",
                username=f"bulk_user_orm_{i}",
                level=i + 1,
            )
            players_data.append(player_data)

        created_players = []
        for p_data in players_data:
            player = await repo.create(p_data)
            created_players.append(player)
        await db_session.commit()

        all_players = await repo.get_all()
        created_ids = {p.id for p in created_players}
        found_in_all = [p for p in all_players if p.id in created_ids]
        assert len(found_in_all) == 10

        for player in created_players:
            await repo.delete(player.id)
        await db_session.commit()

    @pytest.mark.asyncio
    async def test_vector_similarity_search(self, db_session: AsyncSession) -> None:
        """Test vector similarity search performance."""
        result = await db_session.execute(
            text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
        )
        if not result.scalar():
            pytest.skip("pgvector extension not installed")

        loc_repo = LocationRepository(db_session)
        region_repo = RegionRepository(db_session)

        region_data = SQLRegion(
            name="Vector Search Region Orm",
            description="Region for vector search testing in ORM",
            theme="test_orm",
        )
        region = await region_repo.create(region_data)
        await db_session.flush()

        embeddings = [
            [0.1] * 384,
            [0.2] * 384,
            [0.9] * 384,
        ]
        created_locations = []
        for i, embedding in enumerate(embeddings):
            location_data = SQLLocation(
                name=f"Vector Location Orm {i}",
                short_desc=f"Location Orm {i} for vector testing",
                full_desc=f"Detailed description of location Orm {i}",
                region_id=region.id,
                location_type="room_orm",
                is_dynamic=False,
                created_by="system_orm",
                location_embedding=embedding,
            )
            location = await loc_repo.create(location_data)
            created_locations.append(location)
        await db_session.commit()

        query_embedding = [0.15] * 384
        similar_locations = await loc_repo.find_similar_locations(
            query_embedding, limit=3
        )
        assert len(similar_locations) <= 3
        if similar_locations:
            # Ensure the expected location is among the results
            names = [loc.name for loc in similar_locations]
            assert "Vector Location Orm 0" in names

        for loc_item in created_locations:
            await loc_repo.delete(loc_item.id)
        await region_repo.delete(region.id)
        await db_session.commit()


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_constraint_violations(self, db_session: AsyncSession) -> None:
        """Test handling of database constraint violations."""
        repo = PlayerRepository(db_session)
        try:
            invalid_player = SQLPlayer(
                name="Invalid Player Orm", username="invalid_user_orm", level=-1
            )
            await repo.create(invalid_player)
        except Exception as e:
            assert "constraint" in str(e).lower() or "check" in str(e).lower()

    @pytest.mark.asyncio
    async def test_nonexistent_record_handling(self, db_session: AsyncSession) -> None:
        """Test handling of operations on non-existent records."""
        repo = PlayerRepository(db_session)
        fake_id = uuid4()
        player = await repo.get_by_id(fake_id)
        assert player is None
        delete_result = await repo.delete(fake_id)
        assert delete_result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
