# Game Loop Database Schema

This document defines the database schema for the Game Loop project, including relational tables, vector storage, and graph relationships within PostgreSQL.

## Overview

The Game Loop database uses PostgreSQL with extensions to support:
- Vector storage (pgvector) for semantic search and similarity matching
- Graph database patterns for entity relationships
- Traditional relational tables for structured data

## Table of Contents

- [Relational Schema](#relational-schema)
  - [Player Data](#player-data)
  - [World Data](#world-data)
  - [Game State](#game-state)
- [Vector Storage](#vector-storage)
  - [Embeddings](#embeddings)
  - [Semantic Search](#semantic-search)
- [Graph Relationships](#graph-relationships)
  - [Spatial Connections](#spatial-connections)
  - [Interactive Relationships](#interactive-relationships)
- [Schema Diagrams](#schema-diagrams)
- [SQL Initialization](#sql-initialization)

## Relational Schema

### Player Data

#### `players`
| Column            | Type        | Description                          |
|-------------------|-------------|--------------------------------------|
| id                | UUID        | Primary key                          |
| username          | VARCHAR(50) | Player's username                    |
| created_at        | TIMESTAMP   | Account creation timestamp           |
| last_login        | TIMESTAMP   | Last login timestamp                 |
| settings_json     | JSONB       | Player preferences and settings      |
| current_location_id | UUID      | FK to locations                      |

#### `player_inventory`
| Column            | Type        | Description                          |
|-------------------|-------------|--------------------------------------|
| id                | UUID        | Primary key                          |
| player_id         | UUID        | FK to players                        |
| object_id         | UUID        | FK to objects                        |
| quantity          | INTEGER     | Number of this item                  |
| acquired_at       | TIMESTAMP   | When the item was acquired           |
| state_json        | JSONB       | Item state data                      |

#### `player_knowledge`
| Column            | Type        | Description                          |
|-------------------|-------------|--------------------------------------|
| id                | UUID        | Primary key                          |
| player_id         | UUID        | FK to players                        |
| knowledge_key     | VARCHAR(100)| Knowledge identifier                 |
| knowledge_value   | TEXT        | Knowledge content                    |
| discovered_at     | TIMESTAMP   | When this was learned                |
| knowledge_embedding | VECTOR(384)| Vector embedding of knowledge       |

#### `player_skills`
| Column            | Type        | Description                          |
|-------------------|-------------|--------------------------------------|
| id                | UUID        | Primary key                          |
| player_id         | UUID        | FK to players                        |
| skill_name        | VARCHAR(50) | Name of the skill                    |
| skill_level       | INTEGER     | Current level                        |
| skill_description | TEXT        | Description of the skill             |
| skill_category    | VARCHAR(50) | Skill category                       |

#### `player_history`
| Column            | Type        | Description                          |
|-------------------|-------------|--------------------------------------|
| id                | UUID        | Primary key                          |
| player_id         | UUID        | FK to players                        |
| event_timestamp   | TIMESTAMP   | When the event occurred              |
| event_type        | VARCHAR(50) | Type of event                        |
| event_data        | JSONB       | Event details                        |
| location_id       | UUID        | FK to locations                      |
| event_embedding   | VECTOR(384) | Vector embedding of event            |

### World Data

#### `locations`
| Column            | Type        | Description                          |
|-------------------|-------------|--------------------------------------|
| id                | UUID        | Primary key                          |
| name              | VARCHAR(100)| Location name                        |
| short_desc        | TEXT        | Short description                    |
| full_desc         | TEXT        | Full description                     |
| region_id         | UUID        | FK to regions                        |
| location_type     | VARCHAR(50) | Type of location                     |
| is_dynamic        | BOOLEAN     | Whether dynamically generated        |
| discovery_timestamp| TIMESTAMP  | When first discovered                |
| created_by        | VARCHAR(50) | Origin (predefined/dynamic/player)   |
| state_json        | JSONB       | Current state                        |
| location_embedding| VECTOR(384) | Vector embedding of location desc    |

#### `regions`
| Column            | Type        | Description                          |
|-------------------|-------------|--------------------------------------|
| id                | UUID        | Primary key                          |
| name              | VARCHAR(100)| Region name                          |
| description       | TEXT        | Region description                   |
| theme             | VARCHAR(50) | Dominant theme                       |
| parent_region_id  | UUID        | FK to parent region (NULL if top)    |
| region_embedding  | VECTOR(384) | Vector embedding of region           |

#### `objects`
| Column            | Type        | Description                          |
|-------------------|-------------|--------------------------------------|
| id                | UUID        | Primary key                          |
| name              | VARCHAR(100)| Object name                          |
| short_desc        | TEXT        | Short description                    |
| full_desc         | TEXT        | Full description                     |
| object_type       | VARCHAR(50) | Type of object                       |
| properties_json   | JSONB       | Object properties                    |
| is_takeable       | BOOLEAN     | Whether can be taken                 |
| location_id       | UUID        | FK to locations (NULL if in inventory)|
| state_json        | JSONB       | Current state                        |
| object_embedding  | VECTOR(384) | Vector embedding of object desc      |

#### `npcs`
| Column            | Type        | Description                          |
|-------------------|-------------|--------------------------------------|
| id                | UUID        | Primary key                          |
| name              | VARCHAR(100)| NPC name                             |
| short_desc        | TEXT        | Short description                    |
| full_desc         | TEXT        | Full description                     |
| npc_type          | VARCHAR(50) | Type of NPC                          |
| personality_json  | JSONB       | Personality traits                   |
| knowledge_json    | JSONB       | What NPC knows                       |
| location_id       | UUID        | FK to locations                      |
| state_json        | JSONB       | Current state                        |
| npc_embedding     | VECTOR(384) | Vector embedding of NPC              |

#### `quests`
| Column            | Type        | Description                          |
|-------------------|-------------|--------------------------------------|
| id                | UUID        | Primary key                          |
| title             | VARCHAR(100)| Quest title                          |
| description       | TEXT        | Quest description                    |
| quest_type        | VARCHAR(50) | Type of quest                        |
| status            | VARCHAR(30) | Current status                       |
| requirements_json | JSONB       | Prerequisites                        |
| steps_json        | JSONB       | Quest steps                          |
| rewards_json      | JSONB       | Rewards on completion                |
| created_at        | TIMESTAMP   | Creation timestamp                   |
| is_dynamic        | BOOLEAN     | Whether dynamically generated        |
| quest_embedding   | VECTOR(384) | Vector embedding of quest            |

### Game State

#### `game_sessions`
| Column            | Type        | Description                          |
|-------------------|-------------|--------------------------------------|
| id                | UUID        | Primary key                          |
| player_id         | UUID        | FK to players                        |
| started_at        | TIMESTAMP   | Session start                        |
| ended_at          | TIMESTAMP   | Session end (NULL if active)         |
| game_time         | INTEGER     | In-game time (minutes)               |
| save_data         | BYTEA       | Serialized save data                 |
| session_summary   | TEXT        | Session summary                      |

#### `world_rules`
| Column            | Type        | Description                          |
|-------------------|-------------|--------------------------------------|
| id                | UUID        | Primary key                          |
| rule_name         | VARCHAR(100)| Rule name                            |
| rule_description  | TEXT        | Rule description                     |
| rule_type         | VARCHAR(30) | Static or dynamic                    |
| created_by        | VARCHAR(50) | System or player ID                  |
| rule_logic        | TEXT        | Rule implementation logic            |
| rule_embedding    | VECTOR(384) | Vector embedding of rule             |

#### `evolution_events`
| Column            | Type        | Description                          |
|-------------------|-------------|--------------------------------------|
| id                | UUID        | Primary key                          |
| event_type        | VARCHAR(50) | Type of evolution event              |
| target_type       | VARCHAR(30) | Location, NPC, object, etc.          |
| target_id         | UUID        | ID of the target                     |
| scheduled_time    | INTEGER     | Game time when it triggers           |
| priority          | INTEGER     | Event priority                       |
| conditions_json   | JSONB       | Triggering conditions                |
| effects_json      | JSONB       | Event effects                        |
| is_processed      | BOOLEAN     | Whether processed                    |

## Vector Storage

### Embeddings
The PostgreSQL database uses the pgvector extension to store embeddings as VECTOR type columns. These are used for semantic similarity searches.

All major entities have embedding columns:
- `player_knowledge.knowledge_embedding`
- `player_history.event_embedding`
- `locations.location_embedding`
- `regions.region_embedding`
- `objects.object_embedding`
- `npcs.npc_embedding`
- `quests.quest_embedding`
- `world_rules.rule_embedding`

### Semantic Search

Semantic searches can be performed using vector similarity operations:

```sql
-- Example: Find similar locations based on description
SELECT
  id,
  name,
  short_desc,
  location_embedding <-> query_embedding AS distance
FROM
  locations
ORDER BY
  distance
LIMIT 5;

-- Example: Find knowledge related to a concept
SELECT
  knowledge_key,
  knowledge_value,
  knowledge_embedding <-> query_embedding AS relevance
FROM
  player_knowledge
WHERE
  player_id = :player_id
ORDER BY
  relevance
LIMIT 10;
```

## Graph Relationships

The graph relationships are modeled using foreign keys and specialized relationship tables.

### Spatial Connections

#### `location_connections`
| Column            | Type        | Description                          |
|-------------------|-------------|--------------------------------------|
| id                | UUID        | Primary key                          |
| source_id         | UUID        | FK to source location                |
| target_id         | UUID        | FK to target location                |
| connection_type   | VARCHAR(50) | Door, path, portal, etc.             |
| direction         | VARCHAR(20) | Direction (N, S, E, W, UP, DOWN)     |
| requirements_json | JSONB       | Requirements to traverse             |
| description       | TEXT        | Connection description               |
| is_hidden         | BOOLEAN     | Whether hidden                       |
| state_json        | JSONB       | Current state                        |

### Interactive Relationships

#### `npc_relationships`
| Column            | Type        | Description                          |
|-------------------|-------------|--------------------------------------|
| id                | UUID        | Primary key                          |
| npc_id            | UUID        | FK to NPCs                           |
| target_type       | VARCHAR(30) | Player, NPC, faction                 |
| target_id         | UUID        | ID of the related entity             |
| relationship_type | VARCHAR(50) | Friend, enemy, neutral, etc.         |
| value             | INTEGER     | Relationship value (-100 to 100)     |
| history_json      | JSONB       | Interaction history                  |

#### `object_interactions`
| Column            | Type        | Description                          |
|-------------------|-------------|--------------------------------------|
| id                | UUID        | Primary key                          |
| object_id         | UUID        | FK to objects                        |
| interaction_type  | VARCHAR(50) | Type of interaction                  |
| effect_json       | JSONB       | Interaction effects                  |
| requirements_json | JSONB       | Requirements to interact             |
| description       | TEXT        | Interaction description              |

#### `player_quest_progress`
| Column            | Type        | Description                          |
|-------------------|-------------|--------------------------------------|
| id                | UUID        | Primary key                          |
| player_id         | UUID        | FK to players                        |
| quest_id          | UUID        | FK to quests                         |
| status            | VARCHAR(30) | In progress, completed, failed       |
| current_step      | INTEGER     | Current step number                  |
| progress_data     | JSONB       | Detailed progress data               |
| started_at        | TIMESTAMP   | When quest was started               |
| completed_at      | TIMESTAMP   | When quest was completed             |

## Schema Diagrams

```
[Player] -owns-> [Inventory Items] -instanceof-> [Objects]
[Player] -knows-> [Knowledge]
[Player] -has-> [Skills]
[Player] -at-> [Location] -contains-> [Objects]
[Location] -connected to-> [Location]
[Location] -in-> [Region] -contains-> [Region]
[Location] -has-> [NPCs]
[NPCs] -relationship-> [Player/NPCs]
[Player] -undertakes-> [Quests]
[World Rules] -affect-> [Game State]
```

## SQL Initialization

```sql
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgvector";

-- Create tables with explicit foreign key constraints
CREATE TABLE regions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    theme VARCHAR(50),
    parent_region_id UUID,
    region_embedding VECTOR(384),
    FOREIGN KEY (parent_region_id) REFERENCES regions(id) ON DELETE SET NULL
);

CREATE TABLE locations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    short_desc TEXT NOT NULL,
    full_desc TEXT NOT NULL,
    region_id UUID,
    location_type VARCHAR(50) NOT NULL,
    is_dynamic BOOLEAN NOT NULL DEFAULT false,
    discovery_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(50) NOT NULL,
    state_json JSONB DEFAULT '{}'::jsonb,
    location_embedding VECTOR(384),
    FOREIGN KEY (region_id) REFERENCES regions(id) ON DELETE SET NULL
);

CREATE TABLE players (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) NOT NULL UNIQUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    settings_json JSONB DEFAULT '{}'::jsonb,
    current_location_id UUID,
    FOREIGN KEY (current_location_id) REFERENCES locations(id) ON DELETE SET NULL
);

CREATE TABLE npcs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    short_desc TEXT NOT NULL,
    full_desc TEXT NOT NULL,
    npc_type VARCHAR(50) NOT NULL,
    personality_json JSONB DEFAULT '{}'::jsonb,
    knowledge_json JSONB DEFAULT '{}'::jsonb,
    location_id UUID,
    state_json JSONB DEFAULT '{}'::jsonb,
    npc_embedding VECTOR(384),
    FOREIGN KEY (location_id) REFERENCES locations(id) ON DELETE SET NULL
);

CREATE TABLE objects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    short_desc TEXT NOT NULL,
    full_desc TEXT NOT NULL,
    object_type VARCHAR(50) NOT NULL,
    properties_json JSONB DEFAULT '{}'::jsonb,
    is_takeable BOOLEAN NOT NULL DEFAULT false,
    location_id UUID,
    state_json JSONB DEFAULT '{}'::jsonb,
    object_embedding VECTOR(384),
    FOREIGN KEY (location_id) REFERENCES locations(id) ON DELETE SET NULL
);

CREATE TABLE player_inventory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    player_id UUID NOT NULL,
    object_id UUID NOT NULL,
    quantity INTEGER NOT NULL DEFAULT 1,
    acquired_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    state_json JSONB DEFAULT '{}'::jsonb,
    FOREIGN KEY (player_id) REFERENCES players(id) ON DELETE CASCADE,
    FOREIGN KEY (object_id) REFERENCES objects(id) ON DELETE CASCADE
);

-- More tables would follow with appropriate constraints...

-- Indexes for vector search
CREATE INDEX ON locations USING ivfflat (location_embedding vector_cosine_ops);
CREATE INDEX ON objects USING ivfflat (object_embedding vector_cosine_ops);
CREATE INDEX ON npcs USING ivfflat (npc_embedding vector_cosine_ops);

-- Indexes for frequent lookups
CREATE INDEX ON player_inventory (player_id);
CREATE INDEX ON npcs (location_id);
CREATE INDEX ON objects (location_id);
CREATE INDEX ON locations (region_id);
```

## Usage Notes

1. **Embeddings Generation**: Vector embeddings should be generated whenever a text description is created or modified.
2. **Graph Traversal**: Use recursive CTEs or specialized graph functions to traverse connections efficiently.
3. **JSON Storage**: Use JSONB for flexible property storage while keeping core properties as columns.
4. **Vector Dimensions**: Adjust vector dimensions based on the embedding model used (default 384 for most efficient models).
5. **Indexing Strategy**: Use appropriate vector indexing methods (ivfflat, hnsw) based on dataset size and query patterns.
