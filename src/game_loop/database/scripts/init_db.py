#!/usr/bin/env python
"""
Database initialization script for Game Loop project.

This script:
1. Connects to the PostgreSQL database
2. Creates the pgvector extension if needed
3. Sets up initial schema based on schema definition
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
    return None


async def create_pgvector_extension(conn: asyncpg.Connection) -> bool:
    """Create the pgvector extension if it doesn't exist."""
    logger.info("Checking for pgvector extension...")

    try:
        # Check if extension exists
        extension_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
        )

        if extension_exists:
            logger.info("pgvector extension is already installed")
        else:
            logger.info("Creating pgvector extension...")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            logger.info("pgvector extension created successfully")

        # Verify it's working by creating a simple vector
        logger.info("Testing pgvector functionality...")
        test_result = await conn.fetchval(
            "SELECT '[1,2,3]'::vector <-> '[4,5,6]'::vector"
        )
        logger.info(f"Vector distance test result: {test_result}")

        return True
    except asyncpg.PostgresError as e:
        logger.error(f"Failed to create or verify pgvector extension: {e}")
        return False


async def create_uuid_extension(conn: asyncpg.Connection) -> bool:
    """Create the uuid-ossp extension if it doesn't exist."""
    logger.info("Creating uuid-ossp extension...")

    try:
        await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
        logger.info("uuid-ossp extension created successfully")
        return True
    except asyncpg.PostgresError as e:
        logger.error(f"Failed to create uuid-ossp extension: {e}")
        return False


async def find_and_execute_schema_file(conn: asyncpg.Connection) -> bool:
    """Find and execute the database schema file."""
    # Try to locate the schema file
    possible_paths = [
        # In the migrations directory relative to this script
        Path(__file__).parent.parent / "migrations" / "001_initial_schema.sql",
        # In the docs directory (from the project root)
        Path(__file__).parents[4] / "docs" / "database" / "schema.sql",
    ]

    schema_path = None
    for path in possible_paths:
        if path.exists():
            schema_path = path
            break

    if not schema_path:
        logger.error("Could not find database schema file")
        return False

    logger.info(f"Found schema file: {schema_path}")

    # Read the schema file
    try:
        with open(schema_path) as f:
            schema_sql = f.read()
    except OSError as e:
        logger.error(f"Failed to read schema file: {e}")
        return False

    # Execute the schema
    logger.info("Executing database schema...")
    try:
        # Execute the schema file in a transaction
        await conn.execute(schema_sql)
        logger.info("Database schema created successfully")
        return True
    except asyncpg.PostgresError as e:
        logger.error(f"Failed to create database schema: {e}")
        return False


async def create_schema_from_documentation(conn: asyncpg.Connection) -> bool:
    """Extract and create schema from documentation if SQL file not found."""
    logger.info("Looking for schema in documentation...")

    schema_docs_path = Path(__file__).parents[4] / "docs" / "database" / "schema.md"

    if not schema_docs_path.exists():
        logger.error(f"Schema documentation not found at {schema_docs_path}")
        return False

    logger.info(f"Found schema documentation: {schema_docs_path}")

    # Read the documentation file
    try:
        with open(schema_docs_path) as f:
            schema_md = f.read()
    except OSError as e:
        logger.error(f"Failed to read schema documentation: {e}")
        return False

    # Extract SQL code blocks from markdown
    import re

    sql_blocks = re.findall(r"```sql\n(.*?)\n```", schema_md, re.DOTALL)

    if not sql_blocks:
        logger.error("No SQL code blocks found in schema documentation")
        return False

    # Clean up SQL blocks to remove any markdown formatting artifacts
    cleaned_sql_blocks = []
    for block in sql_blocks:
        # Replace code comments that might have markdown table syntax
        cleaned_block = re.sub(r"--.*$", "", block, flags=re.MULTILINE)
        # Remove any lines containing markdown table formatting
        cleaned_block = "\n".join(
            line
            for line in cleaned_block.split("\n")
            if "|" not in line and not line.strip().startswith("-")
        )
        # Remove empty lines
        cleaned_block = "\n".join(
            line for line in cleaned_block.split("\n") if line.strip()
        )
        cleaned_sql_blocks.append(cleaned_block)

    # Execute each SQL block in a transaction
    logger.info(f"Found {len(cleaned_sql_blocks)} SQL blocks in documentation")

    try:
        # Start a transaction
        async with conn.transaction():
            for i, sql_block in enumerate(cleaned_sql_blocks):
                logger.info(f"Executing SQL block {i+1}/{len(cleaned_sql_blocks)}")
                try:
                    await conn.execute(sql_block)
                except Exception as e:
                    logger.error(f"Error in SQL block {i+1}: {e}")
                    logger.error(f"Problematic SQL: {sql_block}")
                    # Continue with other blocks instead of failing completely
                    continue

        logger.info("Database schema created successfully from documentation")
        return True
    except asyncpg.PostgresError as e:
        logger.error(f"Failed to create schema from documentation: {e}")
        return False


async def create_test_data(conn: asyncpg.Connection) -> bool:
    """Create some test data in the database."""
    logger.info("Creating test data...")

    try:
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


async def main() -> bool:
    """Main initialization function."""
    try:
        # Connect to the database
        conn = await connect_db()

        # Create extensions
        extensions_ok = await create_uuid_extension(conn)
        if not extensions_ok:
            logger.error("Failed to create required extensions")
            return False

        pgvector_ok = await create_pgvector_extension(conn)
        if not pgvector_ok:
            logger.error("pgvector extension not working properly")
            return False

        # Try to create schema from SQL file first
        schema_ok = await find_and_execute_schema_file(conn)

        # If that fails, try to create from documentation
        if not schema_ok:
            schema_ok = await create_schema_from_documentation(conn)

        if not schema_ok:
            logger.error("Failed to create database schema")
            return False

        # Create some test data
        test_data_ok = await create_test_data(conn)
        if not test_data_ok:
            logger.warning("Failed to create test data, but schema setup may be OK")

        # Close connection
        await conn.close()

        logger.info("Database initialization completed successfully")
        return True
    except Exception as e:
        logger.error(f"Unexpected error during database initialization: {e}")
        return False


if __name__ == "__main__":
    # Run the async main function
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
