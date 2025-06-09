#!/usr/bin/env python
"""
Docker services management script for Game Loop project.

This script provides commands to:
1. Start/stop Docker services defined in docker-compose.yml
2. Initialize the database
3. Verify the setup
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parents[1]
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"


def run_command(
    cmd: list[str],
    check: bool = True,
    capture_output: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess | None:
    """Run a shell command."""
    logger.debug(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, check=check, capture_output=capture_output, text=True, env=env
        )
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        if capture_output:
            logger.error(f"Command output: {e.stderr}")
        if check:
            sys.exit(1)
        return None


def start_services() -> None:
    """Start Docker services using docker-compose."""
    logger.info("Starting Docker services...")

    if not COMPOSE_FILE.exists():
        logger.error(f"Docker Compose file not found at {COMPOSE_FILE}")
        sys.exit(1)

    # Check if Podman is running
    docker_check = run_command(["podman", "info"], check=False, capture_output=True)
    if docker_check and docker_check.returncode != 0:
        logger.error("Podman is not running or not accessible")
        sys.exit(1)

    # Start services
    run_command(["podman-compose", "-f", str(COMPOSE_FILE), "up", "-d"])

    logger.info("Docker services started successfully")


def stop_services() -> None:
    """Stop Docker services using docker-compose."""
    logger.info("Stopping Docker services...")

    if not COMPOSE_FILE.exists():
        logger.error(f"Docker Compose file not found at {COMPOSE_FILE}")
        sys.exit(1)

    # Stop services
    run_command(["podman-compose", "-f", str(COMPOSE_FILE), "down"])

    logger.info("Docker services stopped successfully")


def initialize_database() -> None:
    """Initialize the database using the init_db.py script."""
    logger.info("Initializing database...")

    # Wait a bit to ensure PostgreSQL is ready
    import time

    time.sleep(5)

    # Run the initialization script
    init_db_script = (
        PROJECT_ROOT / "src" / "game_loop" / "database" / "scripts" / "init_db.py"
    )

    if not init_db_script.exists():
        logger.error(f"Database initialization script not found at {init_db_script}")
        sys.exit(1)

    env = os.environ.copy()
    env.update(
        {
            "POSTGRES_HOST": "localhost",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DB": "game_loop",
            "POSTGRES_USER": "postgres",
            "POSTGRES_PASSWORD": "postgres",
        }
    )

    # Use poetry run to ensure we're using the correct Python environment
    run_command(["poetry", "run", "python", str(init_db_script)], env=env)

    logger.info("Database initialized successfully")


def verify_setup() -> None:
    """Verify the setup using the verify_db_setup.py script."""
    logger.info("Verifying setup...")

    # Run the verification script
    verify_script = (
        PROJECT_ROOT
        / "src"
        / "game_loop"
        / "database"
        / "scripts"
        / "verify_db_setup.py"
    )

    if not verify_script.exists():
        logger.error(f"Database verification script not found at {verify_script}")
        sys.exit(1)

    env = os.environ.copy()
    env.update(
        {
            "POSTGRES_HOST": "localhost",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DB": "game_loop",
            "POSTGRES_USER": "postgres",
            "POSTGRES_PASSWORD": "postgres",
        }
    )

    # Use poetry run to ensure we're using the correct Python environment
    run_command(["poetry", "run", "python", str(verify_script)], env=env)

    logger.info("Setup verification completed")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Manage Docker services for Game Loop")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Start command
    subparsers.add_parser("start", help="Start Docker services")

    # Stop command
    subparsers.add_parser("stop", help="Stop Docker services")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize database")
    init_parser.add_argument(
        "--with-start", action="store_true", help="Start services before initializing"
    )

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify setup")
    verify_parser.add_argument(
        "--with-start", action="store_true", help="Start services before verifying"
    )

    # Setup command (start + init + verify)
    subparsers.add_parser(
        "setup", help="Complete setup (start services, init database, verify)"
    )

    # Debug flag
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Execute command
    if args.command == "start":
        start_services()
    elif args.command == "stop":
        stop_services()
    elif args.command == "init":
        if args.with_start:
            start_services()
        initialize_database()
    elif args.command == "verify":
        if args.with_start:
            start_services()
        verify_setup()
    elif args.command == "setup":
        start_services()
        initialize_database()
        verify_setup()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
