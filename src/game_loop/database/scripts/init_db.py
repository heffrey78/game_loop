#!/usr/bin/env python
"""
Database initialization script for Game Loop project.

This script:
1. Connects to the PostgreSQL database
2. Creates necessary extensions
3. Runs all migrations in order with proper transaction handling
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

import asyncpg

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def connect_db(retry: int = 5, delay: int = 2) -> asyncpg.Connection:
    """Connect to the database with retry logic."""
    # Get connection details from environment variables
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = int(os.environ.get("POSTGRES_PORT", "5432"))
    database = os.environ.get("POSTGRES_DB", "game_loop")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")

    logger.info(f"Connecting to PostgreSQL at {host}:{port}/{database} as {user}...")

    for attempt in range(retry):
        try:
            conn = await asyncpg.connect(
                host=host, port=port, user=user, password=password, database=database
            )
            logger.info("Database connection successful")
            return conn
        except (asyncpg.PostgresError, OSError) as e:
            if attempt < retry - 1:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                logger.info(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                logger.error(f"Failed to connect after {retry} attempts: {e}")
                raise

    # This should never be reached due to the raise above
    raise RuntimeError("Failed to connect to database")


async def create_extensions(conn: asyncpg.Connection) -> bool:
    """Create necessary database extensions."""
    logger.info("Creating required extensions...")

    try:
        # Create uuid-ossp extension
        await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
        logger.info("uuid-ossp extension created successfully")

        # Create pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        logger.info("pgvector extension created successfully")

        # Verify pgvector is working by creating a simple vector
        test_result = await conn.fetchval(
            "SELECT '[1,2,3]'::vector <-> '[4,5,6]'::vector"
        )
        logger.info(f"Vector distance test result: {test_result}")

        return True
    except asyncpg.PostgresError as e:
        logger.error(f"Failed to create extensions: {e}")
        return False


async def setup_migrations_table(conn: asyncpg.Connection) -> bool:
    """Set up the migrations tracking table."""
    logger.info("Setting up migrations tracking table...")

    try:
        await conn.execute(
            """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
        )
        """
        )
        logger.info("Migrations tracking table created successfully")
        return True
    except asyncpg.PostgresError as e:
        logger.error(f"Failed to create migrations table: {e}")
        return False


async def get_applied_migrations(conn: asyncpg.Connection) -> set:
    """Get the list of already applied migrations."""
    try:
        rows = await conn.fetch("SELECT version FROM schema_migrations")
        applied_migrations = {row["version"] for row in rows}
        logger.info(f"Found {len(applied_migrations)} already applied migrations")
        return applied_migrations
    except asyncpg.PostgresError as e:
        logger.error(f"Failed to fetch applied migrations: {e}")
        return set()


async def apply_migration(
    conn: asyncpg.Connection, migration_path: Path, record: bool = True
) -> bool:
    """Apply a single migration file."""
    migration_name = migration_path.name
    logger.info(f"Applying migration: {migration_name}")

    try:
        # Read the migration file
        with open(migration_path) as f:
            migration_sql = f.read()

        # Execute the migration in a transaction
        async with conn.transaction():
            await conn.execute(migration_sql)

            # Record the migration as applied if requested
            if record:
                await conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES ($1)",
                    migration_name,
                )

        logger.info(f"Migration {migration_name} applied successfully")
        return True
    except asyncpg.PostgresError as e:
        logger.error(f"Failed to apply migration {migration_name}: {e}")
        logger.error("Migration transaction was rolled back")

        # Check if this failure is due to tables already existing
        if "already exists" in str(e).lower():
            logger.warning(f"Objects already exist from migration {migration_name}")
            if (
                "001_initial_schema.sql" in migration_name
                or "initial" in migration_name.lower()
            ):
                logger.warning(
                    "This appears to be an initial schema migration "
                    "with existing tables."
                )
                logger.warning(
                    "Recording it as applied to allow subsequent migrations to run."
                )

                try:
                    # Record the migration as applied
                    await conn.execute(
                        "INSERT INTO schema_migrations (version) VALUES ($1)",
                        migration_name,
                    )
                    return True
                except asyncpg.PostgresError as insert_err:
                    logger.error(f"Failed to record migration as applied: {insert_err}")

        return False


async def apply_migrations(conn: asyncpg.Connection) -> bool:
    """Apply all pending migrations in order."""
    # Find migration files
    migrations_dir = Path(__file__).parent.parent / "migrations"

    if not migrations_dir.exists() or not migrations_dir.is_dir():
        logger.error(f"Migrations directory not found: {migrations_dir}")
        return False

    # Get all SQL migration files and sort them
    migration_files = sorted(migrations_dir.glob("*.sql"))

    if not migration_files:
        logger.error("No migration files found")
        return False

    logger.info(f"Found {len(migration_files)} migration files")

    # Get already applied migrations
    applied_migrations = await get_applied_migrations(conn)

    # Apply each migration in order
    all_success = True
    for migration_path in migration_files:
        migration_name = migration_path.name

        # Skip if already applied
        if migration_name in applied_migrations:
            logger.info(f"Migration {migration_name} already applied, skipping")
            continue

        # Apply the migration
        success = await apply_migration(conn, migration_path)
        if not success:
            logger.warning(f"Migration {migration_name} application had issues")
            all_success = False
            # Continue with other migrations instead of breaking

    return all_success


async def create_test_data(conn: asyncpg.Connection) -> bool:
    """Create some test data in the database."""
    logger.info("Creating test data...")

    try:
        # Check if test data already exists
        region_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM regions WHERE name = 'Test Region')"
        )

        if region_exists:
            logger.info("Test data already exists, skipping")
            return True

        # Create a test region
        region_id = await conn.fetchval(
            """
        INSERT INTO regions (name, description, theme)
        VALUES ('Test Region', 'A region for testing', 'Test')
        RETURNING id
        """
        )

        # Create a test location
        location_id = await conn.fetchval(
            """
        INSERT INTO locations (
            name, short_desc, full_desc, region_id,
            location_type, is_dynamic, created_by
        )
        VALUES (
            'Test Location',
            'A test location',
            'This is a detailed description of the test location.',
            $1,
            'room',
            false,
            'system'
        )
        RETURNING id
        """,
            region_id,
        )

        # Create a test vector - properly formatted as string
        # Format: "[0.1,0.1,0.1,...]" for pgvector
        test_vector = "[" + ",".join(["0.1"] * 384) + "]"

        # Update the location with a test embedding
        await conn.execute(
            """
        UPDATE locations
        SET location_embedding = $1::vector
        WHERE id = $2
        """,
            test_vector,
            location_id,
        )

        logger.info("Test data created successfully")
        return True
    except asyncpg.PostgresError as e:
        logger.error(f"Failed to create test data: {e}")
        return False


async def drop_database_if_requested() -> bool:
    """Drop and recreate the database if requested via environment variable."""
    if os.environ.get("DB_FRESH_START", "").lower() == "true":
        logger.warning(
            "DB_FRESH_START is set to true - dropping and recreating database"
        )

        # Get connection details for the postgres database to be able to drop/create
        host = os.environ.get("POSTGRES_HOST", "localhost")
        port = int(os.environ.get("POSTGRES_PORT", "5432"))
        target_db = os.environ.get("POSTGRES_DB", "game_loop")
        user = os.environ.get("POSTGRES_USER", "postgres")
        password = os.environ.get("POSTGRES_PASSWORD", "postgres")

        # Connect to the postgres database
        try:
            sys_conn = await asyncpg.connect(
                host=host, port=port, user=user, password=password, database="postgres"
            )

            # Check if database exists
            db_exists = await sys_conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname = $1)", target_db
            )

            if db_exists:
                # Terminate all connections to the target database
                await sys_conn.execute(
                    f"""
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = '{target_db}'
                AND pid <> pg_backend_pid()
                """
                )

                # Drop the database
                await sys_conn.execute(f"DROP DATABASE IF EXISTS {target_db}")
                logger.info(f"Dropped database {target_db}")

            # Create the database
            await sys_conn.execute(f"CREATE DATABASE {target_db}")
            logger.info(f"Created fresh database {target_db}")

            # Close the system connection
            await sys_conn.close()

            return True

        except asyncpg.PostgresError as e:
            logger.error(f"Failed to drop/recreate database: {e}")
            return False

    return True


async def main() -> bool:
    """Main initialization function."""
    try:
        # Drop and recreate database if requested
        if not await drop_database_if_requested():
            logger.error("Failed to prepare database")
            return False

        # Connect to the database
        conn = await connect_db()

        try:
            # Create necessary extensions
            if not await create_extensions(conn):
                logger.error("Failed to create required extensions")
                return False

            # Set up migrations table
            if not await setup_migrations_table(conn):
                logger.error("Failed to set up migrations table")
                return False

            # Apply all migrations
            migration_result = await apply_migrations(conn)
            if not migration_result:
                logger.warning(
                    "Some migrations had issues, but proceeding with initialization"
                )

            # Create test data
            if not await create_test_data(conn):
                logger.warning("Failed to create test data, but schema may be OK")

            logger.info("Database initialization completed successfully")
            return True

        finally:
            # Always close the connection
            await conn.close()

    except Exception as e:
        logger.error(f"Unexpected error during database initialization: {e}")
        return False


if __name__ == "__main__":
    # Run the async main function
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
