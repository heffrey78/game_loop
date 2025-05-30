#!/usr/bin/env python3
"""Migration script to add timestamp columns to tables with TimestampMixin."""

import asyncio
import os
import sys

import asyncpg


async def add_timestamp_columns() -> None:
    """Add created_at and updated_at columns to all tables that need them."""

    # Tables that use TimestampMixin and need timestamp columns
    tables = [
        "players",
        "player_inventories",
        "player_knowledge",
        "player_skills",
        "player_history",
        "regions",
        "locations",
        "objects",
        "npcs",
        "quests",
        "location_connections",
        "game_sessions",
        "world_rules",
        "evolution_events",
    ]

    conn = await asyncpg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5432)),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "postgres"),
        database=os.getenv("POSTGRES_DB", "game_loop"),
    )

    try:
        print("Adding timestamp columns to tables...")

        for table in tables:
            try:
                # Add created_at column
                await conn.execute(
                    f"""
                    ALTER TABLE {table}
                    ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE
                    DEFAULT CURRENT_TIMESTAMP NOT NULL
                """
                )
                print(f"✓ Added created_at to {table}")

                # Add updated_at column
                await conn.execute(
                    f"""
                    ALTER TABLE {table}
                    ADD COLUMN IF NOT EXISTS updated_at
                    TIMESTAMP WITH TIME ZONE
                    DEFAULT CURRENT_TIMESTAMP NOT NULL
                """
                )
                print(f"✓ Added updated_at to {table}")

                # Create trigger to update updated_at on row updates
                await conn.execute(
                    f"""
                    CREATE OR REPLACE FUNCTION update_{table}_updated_at()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.updated_at = CURRENT_TIMESTAMP;
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql;
                """
                )

                await conn.execute(
                    f"""
                    DROP TRIGGER IF EXISTS
                    trigger_update_{table}_updated_at ON {table};
                    CREATE TRIGGER trigger_update_{table}_updated_at
                        BEFORE UPDATE ON {table}
                        FOR EACH ROW
                        EXECUTE FUNCTION update_{table}_updated_at();
                """
                )
                print(f"✓ Created update trigger for {table}")

            except Exception as e:
                print(f"✗ Error processing table {table}: {e}")
                continue

        print("\nMigration completed successfully!")

    except Exception as e:
        print(f"Migration failed: {e}")
        sys.exit(1)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(add_timestamp_columns())
