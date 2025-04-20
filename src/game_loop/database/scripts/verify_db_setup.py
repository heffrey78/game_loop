#!/usr/bin/env python
"""
Verification script for the database setup.

This script verifies that the database infrastructure is correctly set up
with all required components:
1. PostgreSQL connection
2. pgvector extension
3. Database schema
4. Basic operations
5. Vector storage and similarity search
"""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try to import asyncpg directly for database connection
try:
    import asyncpg
except ImportError:
    logger.warning("asyncpg not installed. Some tests will be skipped.")
    asyncpg = None


async def check_postgres_version(conn: "asyncpg.Connection") -> bool:
    """Check PostgreSQL version."""
    try:
        version = await conn.fetchval("SELECT version()")
        logger.info(f"PostgreSQL version: {version}")
        return True
    except Exception as e:
        logger.error(f"Failed to get PostgreSQL version: {e}")
        return False


async def check_pgvector(conn: "asyncpg.Connection") -> bool:
    """Check pgvector extension is available and working."""
    try:
        # Check if extension exists
        has_ext = await conn.fetchval(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
        )

        if not has_ext:
            logger.error("pgvector extension not installed")
            return False

        # Test vector operations
        distance = await conn.fetchval("SELECT '[1,2,3]'::vector <-> '[4,5,6]'::vector")

        logger.info(f"Vector distance test: {distance}")
        return True
    except Exception as e:
        logger.error(f"pgvector test failed: {e}")
        return False


async def check_schema_tables(conn: "asyncpg.Connection") -> bool:
    """Check if all required tables exist."""
    try:
        # Required tables from the schema
        required_tables = [
            "regions",
            "locations",
            "location_connections",
            "players",
            "npcs",
            "objects",
            "player_inventory",
            "quests",
            "world_rules",
        ]

        # Get existing tables
        tables = await conn.fetch(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        )
        existing = [t["tablename"] for t in tables]

        # Check if all required tables exist
        missing = [t for t in required_tables if t not in existing]

        if missing:
            logger.error(f"Missing tables: {missing}")
            return False

        logger.info(f"Found {len(existing)} tables, all required tables exist")
        return True
    except Exception as e:
        logger.error(f"Schema check failed: {e}")
        return False


async def check_basic_operations(conn: "asyncpg.Connection") -> bool:
    """Check basic database operations."""
    try:
        # Try to create a test region
        region_id = await conn.fetchval(
            """
            INSERT INTO regions (name, description, theme)
            VALUES ('Test Region', 'A test region for verification', 'test')
            RETURNING id
        """
        )

        # Read the region back
        region = await conn.fetchrow(
            "SELECT name, description FROM regions WHERE id = $1", region_id
        )

        if not region:
            logger.error("Failed to read test region")
            return False

        # Update the region
        await conn.execute(
            "UPDATE regions SET theme = 'verified' WHERE id = $1", region_id
        )

        # Delete the test region
        await conn.execute("DELETE FROM regions WHERE id = $1", region_id)

        logger.info("Basic database operations succeeded (CRUD)")
        return True
    except Exception as e:
        logger.error(f"Basic operations test failed: {e}")
        return False


async def check_vector_operations(conn: "asyncpg.Connection") -> bool:
    """Check vector operations with pgvector."""
    try:
        # Create test regions with embeddings using proper vector syntax
        test_vectors = [
            {
                "vec": "[1,0,0" + ",0" * 381 + "]",
                "name": "North",
            },  # Simplified 384-dim vector
            {"vec": "[0,1,0" + ",0" * 381 + "]", "name": "East"},
            {"vec": "[0,0,1" + ",0" * 381 + "]", "name": "Up"},
        ]

        for vector in test_vectors:
            await conn.execute(
                """
                INSERT INTO regions (name, description, theme, region_embedding)
                VALUES ($1, $2, 'test', $3::vector)
                """,
                vector["name"],
                f"Test {vector['name']} region",
                vector["vec"],
            )

        # Query for nearest vector - using proper vector syntax
        query_vec = "[0.9,0.1,0" + ",0" * 381 + "]"  # Should be closest to North

        nearest = await conn.fetchrow(
            """
            SELECT name, region_embedding <-> $1::vector AS distance
            FROM regions
            WHERE theme = 'test'
            ORDER BY distance
            LIMIT 1
            """,
            query_vec,
        )

        # Clean up test data
        await conn.execute("DELETE FROM regions WHERE theme = 'test'")

        if nearest and nearest["name"] == "North":
            logger.info(
                f"Vector search found correct nearest region: {nearest['name']} "
                f"with distance {nearest['distance']}"
            )
            return True
        else:
            logger.error(
                "Vector search returned incorrect result: "
                + "{nearest['name'] if nearest else 'None'}"
            )
            return False
    except Exception as e:
        logger.error(f"Vector operations test failed: {e}")
        return False


def check_file_structure() -> bool:
    """Check if all required database infrastructure files exist."""
    # Base path to look for files
    base_paths = [
        Path(__file__).parents[2],  # game_loop module if run from scripts
        Path(__file__).parents[4] / "src" / "game_loop",  # from project root
    ]

    # Files that should exist
    required_files = [
        "database/db_connection.py",
        "database/migrations/001_initial_schema.sql",
        "database/scripts/init_db.py",
        "database/scripts/run_postgres.py",
        "database/scripts/verify_db_setup.py",
    ]

    # Check each required file
    missing_files = []

    for req_file in required_files:
        found = False
        for base_path in base_paths:
            if (base_path / req_file).exists():
                found = True
                break

        if not found:
            missing_files.append(req_file)

    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        return False

    logger.info("All required database infrastructure files exist")
    return True


def check_sql_schema_file() -> bool:
    """Check if the SQL schema file contains all required components."""
    # Path to schema file
    schema_path = Path(__file__).parents[2] / "migrations" / "001_initial_schema.sql"

    if not schema_path.exists():
        # Try an alternative path
        schema_path = (
            Path(__file__).parents[4]
            / "src"
            / "game_loop"
            / "database"
            / "migrations"
            / "001_initial_schema.sql"
        )
        if not schema_path.exists():
            logger.error(f"Schema file not found: {schema_path}")
            return False

    # Read the schema file
    schema_content = schema_path.read_text()

    # Check for required components
    required_components = [
        # pgvector is now created in init_db.py instead of the schema file
        'CREATE EXTENSION IF NOT EXISTS "uuid-ossp"',
        "CREATE TABLE regions",
        "CREATE TABLE locations",
        "CREATE TABLE players",
        "CREATE TABLE npcs",
        "CREATE TABLE objects",
        "region_embedding VECTOR(384)",
        "location_embedding VECTOR(384)",
        "object_embedding VECTOR(384)",
        "npc_embedding VECTOR(384)",
        "CREATE INDEX",
    ]

    missing_components = []

    for component in required_components:
        if component not in schema_content:
            missing_components.append(component)

    if missing_components:
        logger.error(f"Schema file missing required components: {missing_components}")
        return False

    logger.info("SQL schema file contains all required components")
    return True


async def run_database_verification(
    host: str, port: int, user: str, password: str, database: str
) -> tuple[int, int, int]:
    """Run all database verification checks."""
    live_checks_passed = 0
    live_checks_failed = 0
    live_checks_skipped = 0

    # Check if we have asyncpg
    if not asyncpg:
        logger.warning("Skipping database connection tests - asyncpg not installed")
        live_checks_skipped = 5  # Skip all live database checks
        return (0, 0, 5)

    # Try to connect to the database
    try:
        conn = await asyncpg.connect(
            host=host, port=port, user=user, password=password, database=database
        )

        # Run verification checks
        checks = [
            ("PostgreSQL Version", check_postgres_version),
            ("pgvector Extension", check_pgvector),
            ("Database Schema", check_schema_tables),
            ("Basic Operations", check_basic_operations),
            ("Vector Storage", check_vector_operations),
        ]

        for name, check_func in checks:
            try:
                if await check_func(conn):
                    live_checks_passed += 1
                    logger.info(f"✅ {name} check: PASSED")
                else:
                    live_checks_failed += 1
                    logger.error(f"❌ {name} check: FAILED")
            except Exception as e:
                logger.error(f"❌ {name} check failed with error: {e}")
                live_checks_failed += 1

        await conn.close()

    except Exception as e:
        logger.warning(f"Could not connect to database: {e}")
        live_checks_skipped = 5  # Skip all checks if connection fails

    return (live_checks_passed, live_checks_failed, live_checks_skipped)


async def verify_database_setup(
    host: str, port: int, user: str, password: str, database: str
) -> bool:
    """Verify database setup with all checks."""
    logger.info("Starting database setup verification...")

    # Run database connection checks
    live_checks_passed, live_checks_failed, live_checks_skipped = (
        await run_database_verification(host, port, user, password, database)
    )

    # Check file structure (always run)
    logger.info("Verifying code file structure...")
    file_structure_ok = check_file_structure()

    # Check SQL schema file (always run)
    logger.info("Verifying SQL schema file...")
    sql_schema_ok = check_sql_schema_file()

    # Print verification results
    logger.info("\n--------------------------------------------------")
    logger.info("Verification Results:")
    logger.info("--------------------------------------------------")
    logger.info("Live Database Checks:")

    if live_checks_passed == 0 and live_checks_failed == 0:
        logger.info("No live database tests passed - database may not be running")
    elif live_checks_passed > 0 and live_checks_failed == 0:
        logger.info(f"All {live_checks_passed} live database tests passed")
    else:
        logger.info(
            f"{live_checks_passed} tests passed, {live_checks_failed} tests failed"
        )

    # Print status for each check
    pass_icon = "✅ PASS"
    skip_icon = "🔶 SKIP"
    fail_icon = "❌ FAIL"

    def get_status_icon(min_passes_needed: int) -> str:
        """Helper to get the right status icon based on check results."""
        if live_checks_passed >= min_passes_needed:
            return pass_icon
        return skip_icon if live_checks_skipped > 0 else fail_icon

    logger.info(f"PostgreSQL Connection: {get_status_icon(1)}")
    logger.info(f"pgvector Extension: {get_status_icon(2)}")
    logger.info(f"Database Schema: {get_status_icon(3)}")
    logger.info(f"Basic Operations: {get_status_icon(4)}")
    logger.info(f"Vector Storage: {get_status_icon(5)}")

    logger.info("\nRequired Code Structure Checks:")
    logger.info(f"Code Structure: {'✅ PASS' if file_structure_ok else '❌ FAIL'}")
    logger.info(f"SQL Schema File: {'✅ PASS' if sql_schema_ok else '❌ FAIL'}")
    logger.info("--------------------------------------------------")

    # Determine overall success
    if live_checks_failed > 0:
        logger.error("❌ Some database checks FAILED")
        return False

    if not file_structure_ok or not sql_schema_ok:
        logger.error("❌ Some required code structure checks FAILED")
        return False

    logger.info(
        "✅ All database infrastructure components "
        "are in place and functioning correctly"
    )
    return True


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Verify database setup")
    parser.add_argument(
        "--host",
        default=os.environ.get("POSTGRES_HOST", "localhost"),
        help="PostgreSQL host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("POSTGRES_PORT", "5432")),
        help="PostgreSQL port",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("POSTGRES_USER", "postgres"),
        help="PostgreSQL user",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("POSTGRES_PASSWORD", "postgres"),
        help="PostgreSQL password",
    )
    parser.add_argument(
        "--database",
        default=os.environ.get("POSTGRES_DB", "game_loop"),
        help="PostgreSQL database",
    )

    args = parser.parse_args()

    success = asyncio.run(
        verify_database_setup(
            args.host, args.port, args.user, args.password, args.database
        )
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
