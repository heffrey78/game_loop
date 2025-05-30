# Commit 10: Database Models and ORM Implementation Plan

## Overview

This commit implements the SQLAlchemy ORM layer for the Game Loop system, providing database models for core entities and establishing robust database access patterns. This builds upon the state management system from Commit 9 and creates the foundation for semantic search and vector embeddings in future commits.

## Prerequisites

- **Commit 9 Complete**: State management system with Pydantic models implemented ✅
- Database infrastructure (PostgreSQL with pgvector) is running ✅
- Schema migrations (001_initial_schema.sql, 002_add_state_tables.sql) have been applied ✅
- Async database connection pooling is working (`db_connection.py`) ✅
- Configuration system supports database settings ✅


## Components to Implement

### 1. Database Models (SQLAlchemy)

**Location**: `src/game_loop/database/models/`

Create the directory structure and implement SQLAlchemy models that correspond to the existing database schema:

```
src/game_loop/database/models/
├── __init__.py
├── base.py          # Base model with common functionality
├── player.py        # Player-related models
├── world.py         # World entity models
├── game_state.py    # Game state and session models
├── relationships.py # Graph relationship models
└── mixins.py        # Common mixins for timestamps, etc.
```

### 2. Repository Pattern Implementation

**Location**: `src/game_loop/database/repositories/`

Implement repository patterns for clean data access:

```
src/game_loop/database/repositories/
├── __init__.py
├── base.py          # Base repository with common CRUD operations
├── player.py        # Player data repository
├── world.py         # World entity repository
├── session.py       # Game session repository
└── search.py        # Semantic search repository (preparation for future)
```

### 3. Database Session Management

**Location**: `src/game_loop/database/`

Enhance existing database infrastructure:

- `session_factory.py`: SQLAlchemy async session factory
- `orm_connection.py`: ORM-specific connection management
- Update `db_connection.py` to work with SQLAlchemy

### 4. Data Transfer Objects (DTOs)

**Location**: `src/game_loop/database/dto/`

Bridge between Pydantic models and SQLAlchemy models:

```
src/game_loop/database/dto/
├── __init__.py
├── converters.py    # Conversion utilities
├── player.py        # Player DTO converters
├── world.py         # World DTO converters
└── session.py       # Session DTO converters
```

## Implementation Steps

### Step 1: SQLAlchemy Base Infrastructure

#### 1.1 Create Base Model (`src/game_loop/database/models/base.py`)

```python
"""Base SQLAlchemy model with common functionality."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import TIMESTAMP, UUID, Column, text
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.dialects.postgresql import JSONB


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    type_annotation_map = {
        dict[str, Any]: JSONB,
        uuid.UUID: UUID(as_uuid=True),
        datetime: TIMESTAMP(timezone=True)
    }


class TimestampMixin:
    """Mixin for models that need created_at/updated_at timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False
    )

    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=text("CURRENT_TIMESTAMP"),
        nullable=False
    )


class UUIDMixin:
    """Mixin for models that use UUID primary keys."""

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("uuid_generate_v4()"),
        nullable=False
    )


class DatabaseError(Exception):
    """Base exception for database operations."""
    pass


class EntityNotFoundError(DatabaseError):
    """Raised when an entity is not found."""
    pass


class ValidationError(DatabaseError):
    """Raised when data validation fails."""
    pass
```

#### 1.2 Create Session Factory (`src/game_loop/database/session_factory.py`)

```python
"""SQLAlchemy async session factory."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from game_loop.config.models import DatabaseConfig
from game_loop.database.models.base import Base

logger = logging.getLogger(__name__)


class DatabaseSessionFactory:
    """Factory for creating database sessions."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._engine = None
        self._session_factory = None

    async def initialize(self) -> None:
        """Initialize the database engine and session factory."""
        database_url = (
            f"postgresql+asyncpg://{self.config.username}:{self.config.password}"
            f"@{self.config.host}:{self.config.port}/{self.config.database}"
        )

        self._engine = create_async_engine(
            database_url,
            echo=self.config.echo,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            poolclass=StaticPool if self.config.echo else None,  # For testing
        )

        self._session_factory = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False
        )

        logger.info("Database session factory initialized")

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session."""
        if not self._session_factory:
            raise RuntimeError("Session factory not initialized")

        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def close(self) -> None:
        """Close the database engine."""
        if self._engine:
            await self._engine.dispose()
            logger.info("Database engine closed")
```

### Step 2: Core Entity Models

#### 2.1 Player Models (`src/game_loop/database/models/player.py`)

```python
"""SQLAlchemy models for player-related entities."""

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import UUID, VARCHAR, INTEGER, TEXT, BOOLEAN, TIMESTAMP, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

from .base import Base, TimestampMixin, UUIDMixin


class Player(Base, UUIDMixin, TimestampMixin):
    """Player account information."""

    __tablename__ = "players"

    username: Mapped[str] = mapped_column(VARCHAR(50), unique=True, nullable=False)
    last_login: Mapped[Optional[datetime]] = mapped_column(TIMESTAMP(timezone=True))
    settings_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=lambda: {})
    current_location_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("locations.id", ondelete="SET NULL")
    )

    # Relationships
    inventory_items = relationship("PlayerInventory", back_populates="player")
    knowledge_entries = relationship("PlayerKnowledge", back_populates="player")
    skills = relationship("PlayerSkill", back_populates="player")
    history_events = relationship("PlayerHistory", back_populates="player")
    game_sessions = relationship("GameSession", back_populates="player")


class PlayerInventory(Base, UUIDMixin):
    """Items in player's inventory."""

    __tablename__ = "player_inventory"

    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("players.id", ondelete="CASCADE"),
        nullable=False
    )
    object_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("objects.id", ondelete="CASCADE"),
        nullable=False
    )
    quantity: Mapped[int] = mapped_column(INTEGER, default=1, nullable=False)
    acquired_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False
    )
    state_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=lambda: {})

    # Relationships
    player = relationship("Player", back_populates="inventory_items")
    object = relationship("Object", back_populates="inventory_entries")


class PlayerKnowledge(Base, UUIDMixin):
    """Knowledge acquired by the player."""

    __tablename__ = "player_knowledge"

    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("players.id", ondelete="CASCADE"),
        nullable=False
    )
    knowledge_key: Mapped[str] = mapped_column(VARCHAR(100), nullable=False)
    knowledge_value: Mapped[str] = mapped_column(TEXT, nullable=False)
    discovered_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False
    )
    knowledge_embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(384))

    # Relationships
    player = relationship("Player", back_populates="knowledge_entries")


class PlayerSkill(Base, UUIDMixin):
    """Player skills and abilities."""

    __tablename__ = "player_skills"

    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("players.id", ondelete="CASCADE"),
        nullable=False
    )
    skill_name: Mapped[str] = mapped_column(VARCHAR(50), nullable=False)
    skill_level: Mapped[int] = mapped_column(INTEGER, nullable=False)
    skill_description: Mapped[str] = mapped_column(TEXT, nullable=False)
    skill_category: Mapped[str] = mapped_column(VARCHAR(50), nullable=False)

    # Relationships
    player = relationship("Player", back_populates="skills")


class PlayerHistory(Base, UUIDMixin):
    """Player action history."""

    __tablename__ = "player_history"

    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("players.id", ondelete="CASCADE"),
        nullable=False
    )
    event_timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=text("CURRENT_TIMESTAMP"),
        nullable=False
    )
    event_type: Mapped[str] = mapped_column(VARCHAR(50), nullable=False)
    event_data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    location_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("locations.id", ondelete="SET NULL")
    )
    event_embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(384))

    # Relationships
    player = relationship("Player", back_populates="history_events")
    location = relationship("Location")
```

#### 2.2 World Models (`src/game_loop/database/models/world.py`)

```python
"""SQLAlchemy models for world entities."""

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import UUID, VARCHAR, TEXT, BOOLEAN, TIMESTAMP, ForeignKey, text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

from .base import Base, UUIDMixin


class Region(Base, UUIDMixin):
    """World regions that contain locations."""

    __tablename__ = "regions"

    name: Mapped[str] = mapped_column(VARCHAR(100), nullable=False)
    description: Mapped[str] = mapped_column(TEXT, nullable=False)
    theme: Mapped[Optional[str]] = mapped_column(VARCHAR(50))
    parent_region_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("regions.id", ondelete="SET NULL")
    )
    region_embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(384))

    # Relationships
    locations = relationship("Location", back_populates="region")
    parent_region = relationship("Region", remote_side="Region.id")
    child_regions = relationship("Region", back_populates="parent_region")


class Location(Base, UUIDMixin):
    """Locations within the game world."""

    __tablename__ = "locations"

    name: Mapped[str] = mapped_column(VARCHAR(100), nullable=False)
    short_desc: Mapped[str] = mapped_column(TEXT, nullable=False)
    full_desc: Mapped[str] = mapped_column(TEXT, nullable=False)
    region_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("regions.id", ondelete="SET NULL")
    )
    location_type: Mapped[str] = mapped_column(VARCHAR(50), nullable=False)
    is_dynamic: Mapped[bool] = mapped_column(BOOLEAN, default=False, nullable=False)
    discovery_timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=text("CURRENT_TIMESTAMP")
    )
    created_by: Mapped[str] = mapped_column(VARCHAR(50), nullable=False)
    state_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=lambda: {})
    location_embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(384))

    # Relationships
    region = relationship("Region", back_populates="locations")
    objects = relationship("Object", back_populates="location")
    npcs = relationship("NPC", back_populates="location")
    outgoing_connections = relationship(
        "LocationConnection",
        foreign_keys="LocationConnection.source_id",
        back_populates="source_location"
    )
    incoming_connections = relationship(
        "LocationConnection",
        foreign_keys="LocationConnection.target_id",
        back_populates="target_location"
    )


class Object(Base, UUIDMixin):
    """Objects that can be found in locations or inventory."""

    __tablename__ = "objects"

    name: Mapped[str] = mapped_column(VARCHAR(100), nullable=False)
    short_desc: Mapped[str] = mapped_column(TEXT, nullable=False)
    full_desc: Mapped[str] = mapped_column(TEXT, nullable=False)
    object_type: Mapped[str] = mapped_column(VARCHAR(50), nullable=False)
    properties_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=lambda: {})
    is_takeable: Mapped[bool] = mapped_column(BOOLEAN, default=False, nullable=False)
    location_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("locations.id", ondelete="SET NULL")
    )
    state_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=lambda: {})
    object_embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(384))

    # Relationships
    location = relationship("Location", back_populates="objects")
    inventory_entries = relationship("PlayerInventory", back_populates="object")
    interactions = relationship("ObjectInteraction", back_populates="object")


class NPC(Base, UUIDMixin):
    """Non-player characters."""

    __tablename__ = "npcs"

    name: Mapped[str] = mapped_column(VARCHAR(100), nullable=False)
    short_desc: Mapped[str] = mapped_column(TEXT, nullable=False)
    full_desc: Mapped[str] = mapped_column(TEXT, nullable=False)
    npc_type: Mapped[str] = mapped_column(VARCHAR(50), nullable=False)
    personality_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=lambda: {})
    knowledge_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=lambda: {})
    location_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("locations.id", ondelete="SET NULL")
    )
    state_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=lambda: {})
    npc_embedding: Mapped[Optional[list[float]]] = mapped_column(Vector(384))

    # Relationships
    location = relationship("Location", back_populates="npcs")
    relationships = relationship("NPCRelationship", back_populates="npc")
```

### Step 3: Repository Pattern

#### 3.1 Base Repository (`src/game_loop/database/repositories/base.py`)

```python
"""Base repository with common CRUD operations."""

import uuid
from typing import Any, Generic, TypeVar, Optional, Sequence

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase

from game_loop.database.models.base import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Base repository with common CRUD operations."""

    def __init__(self, session: AsyncSession, model_class: type[ModelType]):
        self.session = session
        self.model_class = model_class

    async def get_by_id(self, id: uuid.UUID) -> Optional[ModelType]:
        """Get a single entity by ID."""
        result = await self.session.execute(
            select(self.model_class).where(self.model_class.id == id)
        )
        return result.scalar_one_or_none()

    async def get_all(self, limit: Optional[int] = None) -> Sequence[ModelType]:
        """Get all entities with optional limit."""
        query = select(self.model_class)
        if limit:
            query = query.limit(limit)

        result = await self.session.execute(query)
        return result.scalars().all()

    async def create(self, **kwargs) -> ModelType:
        """Create a new entity."""
        entity = self.model_class(**kwargs)
        self.session.add(entity)
        await self.session.flush()  # Get the ID without committing
        await self.session.refresh(entity)
        return entity

    async def update_by_id(self, id: uuid.UUID, **kwargs) -> Optional[ModelType]:
        """Update an entity by ID."""
        await self.session.execute(
            update(self.model_class)
            .where(self.model_class.id == id)
            .values(**kwargs)
        )
        return await self.get_by_id(id)

    async def delete_by_id(self, id: uuid.UUID) -> bool:
        """Delete an entity by ID."""
        result = await self.session.execute(
            delete(self.model_class).where(self.model_class.id == id)
        )
        return result.rowcount > 0

    async def exists(self, id: uuid.UUID) -> bool:
        """Check if an entity exists."""
        result = await self.session.execute(
            select(self.model_class.id).where(self.model_class.id == id)
        )
        return result.scalar_one_or_none() is not None
```

#### 3.2 Player Repository (`src/game_loop/database/repositories/player.py`)

```python
"""Repository for player-related data access."""

import uuid
from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from game_loop.database.models.player import (
    Player, PlayerInventory, PlayerKnowledge, PlayerSkill, PlayerHistory
)
from .base import BaseRepository


class PlayerRepository(BaseRepository[Player]):
    """Repository for player data operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, Player)

    async def get_by_username(self, username: str) -> Optional[Player]:
        """Get player by username."""
        result = await self.session.execute(
            select(Player).where(Player.username == username)
        )
        return result.scalar_one_or_none()

    async def get_with_full_data(self, player_id: uuid.UUID) -> Optional[Player]:
        """Get player with all related data loaded."""
        result = await self.session.execute(
            select(Player)
            .options(
                selectinload(Player.inventory_items),
                selectinload(Player.knowledge_entries),
                selectinload(Player.skills),
                selectinload(Player.history_events)
            )
            .where(Player.id == player_id)
        )
        return result.scalar_one_or_none()

    async def update_last_login(self, player_id: uuid.UUID) -> None:
        """Update player's last login timestamp."""
        from datetime import datetime
        await self.update_by_id(player_id, last_login=datetime.now())


class PlayerInventoryRepository(BaseRepository[PlayerInventory]):
    """Repository for player inventory operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, PlayerInventory)

    async def get_player_inventory(self, player_id: uuid.UUID) -> Sequence[PlayerInventory]:
        """Get all inventory items for a player."""
        result = await self.session.execute(
            select(PlayerInventory)
            .where(PlayerInventory.player_id == player_id)
            .options(selectinload(PlayerInventory.object))
        )
        return result.scalars().all()

    async def add_item(
        self,
        player_id: uuid.UUID,
        object_id: uuid.UUID,
        quantity: int = 1
    ) -> PlayerInventory:
        """Add an item to player inventory."""
        return await self.create(
            player_id=player_id,
            object_id=object_id,
            quantity=quantity
        )


class PlayerKnowledgeRepository(BaseRepository[PlayerKnowledge]):
    """Repository for player knowledge operations."""

    def __init__(self, session: AsyncSession):
        super().__init__(session, PlayerKnowledge)

    async def get_player_knowledge(self, player_id: uuid.UUID) -> Sequence[PlayerKnowledge]:
        """Get all knowledge for a player."""
        result = await self.session.execute(
            select(PlayerKnowledge).where(PlayerKnowledge.player_id == player_id)
        )
        return result.scalars().all()

    async def add_knowledge(
        self,
        player_id: uuid.UUID,
        knowledge_key: str,
        knowledge_value: str,
        embedding: Optional[list[float]] = None
    ) -> PlayerKnowledge:
        """Add knowledge to player."""
        return await self.create(
            player_id=player_id,
            knowledge_key=knowledge_key,
            knowledge_value=knowledge_value,
            knowledge_embedding=embedding
        )
```

### Step 4: Integration with State Management

#### 4.1 Update State Manager (`src/game_loop/state/manager.py`)

Modify the existing GameStateManager to use the ORM repositories:

```python
# Add to existing imports
from game_loop.database.session_factory import DatabaseSessionFactory
from game_loop.database.repositories.player import PlayerRepository
from game_loop.database.repositories.world import WorldRepository

class GameStateManager:
    """Enhanced GameStateManager with ORM integration."""

    def __init__(self, config_manager, db_pool=None):
        # ...existing code...
        self.session_factory = DatabaseSessionFactory(config_manager.config.database)
        self._repositories = {}

    async def initialize(self):
        """Initialize with ORM support."""
        await self.session_factory.initialize()
        # ...existing initialization code...

    async def get_player_repository(self) -> PlayerRepository:
        """Get player repository with current session."""
        async with self.session_factory.get_session() as session:
            return PlayerRepository(session)

    # Add similar methods for other repositories
```

## Migration Strategy

### Backward Compatibility

During implementation, maintain compatibility with existing `asyncpg` patterns:

1. **Gradual Migration**:
   - Keep existing `db_connection.py` functional
   - Add ORM capabilities alongside current SQL patterns
   - Migrate component by component to avoid breaking changes

2. **Dual Mode Support**:
   ```python
   class GameStateManager:
       def __init__(self, config_manager, db_pool=None):
           self.db_pool = db_pool  # Keep existing asyncpg pool
           self.session_factory = None  # Add ORM session factory

       async def initialize(self):
           # Initialize both connection methods
           await self._init_legacy_pool()
           await self._init_orm_sessions()
   ```

3. **Data Validation**:
   - Implement conversion validation between Pydantic ↔ SQLAlchemy
   - Add data integrity checks during migration
   - Ensure vector field compatibility

## Testing Plan

### Unit Tests

1. **Model Tests** (`tests/unit/database/test_models.py`)
   - Test model creation and validation
   - Test relationships between models
   - Test vector field handling

2. **Repository Tests** (`tests/unit/database/test_repositories.py`)
   - Test CRUD operations
   - Test complex queries
   - Test transaction handling

3. **Session Factory Tests** (`tests/unit/database/test_session_factory.py`)
   - Test session creation and cleanup
   - Test error handling and rollback
   - Test connection pooling

### Integration Tests

1. **Database Integration** (`tests/integration/database/test_orm_integration.py`)
   - Test full CRUD workflows
   - Test model relationships with real database
   - Test vector operations (preparation for embeddings)

2. **State Management Integration** (`tests/integration/state/test_orm_state_integration.py`)
   - Test Pydantic ↔ SQLAlchemy conversions
   - Test state persistence through ORM
   - Test concurrent access patterns

## Success Criteria

### Functional Requirements

1. **Model Operations**
   - All SQLAlchemy models can be created, read, updated, and deleted
   - Relationships between models work correctly
   - Vector fields can store and retrieve embedding data

2. **Repository Pattern**
   - Repositories provide clean abstraction over raw SQL
   - Complex queries work through repository methods
   - Transaction handling is robust

3. **Integration**
   - Existing state management system works with ORM
   - Pydantic models convert cleanly to/from SQLAlchemy models
   - Performance is acceptable for game use cases

### Technical Requirements

1. **Database Schema Compatibility**
   - Models match existing database schema exactly
   - Foreign key constraints work properly
   - Vector columns are properly configured

2. **Async Support**
   - All database operations are properly async
   - Session management doesn't leak connections
   - Concurrent access patterns work correctly

3. **Error Handling**
   - Database errors are caught and handled gracefully
   - Rollback works correctly on failures
   - Connection pooling handles edge cases

## Metrics and Validation

1. **Performance Metrics**
   - Single entity operations < 10ms
   - Bulk operations scale linearly
   - Connection pool utilization stays reasonable

2. **Data Integrity**
   - All foreign key constraints enforced
   - JSONB data round-trips correctly
   - Vector data preserves precision

3. **Code Quality**
   - >95% test coverage on repository code
   - All models have proper type hints
   - Documentation covers all public APIs

## Documentation Updates

1. **Code Documentation**
   - Add comprehensive docstrings to all models
   - Document repository patterns and usage
   - Include examples for complex operations

2. **Architecture Documentation**
   - Update architecture diagram with ORM layer
   - Document data flow through repositories
   - Add database access patterns guide

## Future Enhancements (Post-Commit)

1. **Query Optimization**
   - Add database query profiling
   - Implement query result caching
   - Add database index optimization

2. **Advanced Features**
   - Implement database migrations through Alembic
   - Add soft delete functionality
   - Implement audit logging for data changes

3. **Vector Operations**
   - Prepare vector similarity search methods
   - Add embedding generation hooks
   - Optimize vector index configuration

This implementation plan builds a solid ORM foundation that will support the semantic search and vector embedding features planned for future commits while maintaining compatibility with the existing state management system.
