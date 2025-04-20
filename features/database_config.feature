Feature: Game Loop Database Configuration
  As a game developer
  I want to configure and manage the database for the Game Loop application
  So that data persistence, vector search, and relational features work correctly

  Background:
    Given PostgreSQL is installed and running
    And the pgvector extension is installed
    And the uuid-ossp extension is installed

  Scenario: Initialize database schema
    When I run the database initialization script
    Then all required tables should be created
    And all required indexes should be created
    And all required constraints should be configured

  Scenario: Configure vector embeddings
    Given the database schema is initialized
    When I configure the embedding model with 384 dimensions
    Then the system should be able to store vector embeddings for all entity types
    And the vector columns should be properly indexed for similarity search

  Scenario: Test vector search functionality
    Given the database contains sample locations with vector embeddings
    When I perform a semantic search for locations similar to "ancient ruins"
    Then the system should return locations ranked by semantic similarity
    And the search results should include relevant metadata

  Scenario: Test spatial relationship graph
    Given the database contains interconnected locations
    When I query for all paths between "Forgotten Library" and "Hidden Archives"
    Then the system should return all possible routes
    And each route should include the connection types and directions

  Scenario: Configure database backup and recovery
    Given the database contains game state data
    When I configure automated backups
    Then backups should occur on the defined schedule
    And I should be able to restore from a backup point

  Scenario: Scale vector operations
    Given the database contains over 10,000 entities with embeddings
    When I perform batch similarity searches
    Then the operations should complete within performance thresholds
    And the system should maintain index efficiency

  Scenario Outline: Store different entity embeddings
    Given I have a "<entity_type>" with a text description
    When I generate and store an embedding for the entity
    Then the embedding should be saved in the appropriate table
    And I should be able to retrieve semantically similar "<entity_type>" entities

    Examples:
      | entity_type      |
      | location         |
      | object           |
      | npc              |
      | player_knowledge |
      | quest            |
      | region           |
      | world_rule       |

  Scenario: Migrate database schema
    Given the database has an existing schema
    When I need to update the schema with new columns
    Then the migration script should preserve existing data
    And the system should validate data integrity after migration

  Scenario: Handle dynamic entity generation
    Given a player explores an undefined area
    When the system generates a new location on-the-fly
    Then the location data should be persisted in the database
    And the location should include appropriate vector embeddings
    And the location should be connected to the existing world graph

  Scenario: Database connection pooling
    Given the game has multiple concurrent players
    When database connections are configured with pooling
    Then the system should efficiently manage connection resources
    And performance should remain stable under varying load

  Scenario: Query optimization for common operations
    Given the database has standard game operation patterns
    When I analyze query performance
    Then I should identify opportunities for optimization
    And I should implement proper indexes for frequent access patterns

  Scenario: Test JSON property storage
    Given an entity has dynamic properties
    When I store those properties in a JSONB column
    Then I should be able to query based on JSON properties
    And I should be able to update nested JSON attributes efficiently

  Scenario: Database transaction integrity
    Given a player performs a complex action that affects multiple tables
    When the action is processed within a transaction
    Then either all changes should be applied successfully
    Or the entire transaction should be rolled back

  Scenario: Handle concurrent updates
    Given multiple game processes update the same entity
    When row-level locking is implemented
    Then updates should not conflict or overwrite each other
    And the database should maintain consistency

  Scenario: Database monitoring and alerts
    When I configure database monitoring
    Then the system should track key performance metrics
    And alerts should trigger when thresholds are exceeded
    And logs should capture significant database events