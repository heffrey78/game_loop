#!/usr/bin/env python
"""
Script to run PostgreSQL with pgvector in a Docker container.

This script:
1. Checks if the container exists
2. Creates or starts the container as needed
3. Ensures PostgreSQL is running with pgvector extension
4. Provides information on how to connect to the database
"""
import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_docker_installed() -> bool:
    """Check if Docker is installed."""
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            logger.info(f"Docker is installed: {result.stdout.strip()}")
            return True
        else:
            logger.error("Docker is not installed or not in PATH")
            return False
    except FileNotFoundError:
        logger.error("Docker is not installed or not in PATH")
        return False


def check_container_exists(container_name: str) -> bool:
    """Check if a container with the given name exists."""
    try:
        result = subprocess.run(
            [
                "docker",
                "container",
                "ls",
                "-a",
                "--filter",
                f"name=^{container_name}$",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return container_name in result.stdout.strip()
    except Exception as e:
        logger.error(f"Error checking if container exists: {e}")
        return False


def check_container_running(container_name: str) -> bool:
    """Check if a container is running."""
    try:
        result = subprocess.run(
            [
                "docker",
                "container",
                "ls",
                "--filter",
                f"name=^{container_name}$",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return container_name in result.stdout.strip()
    except Exception as e:
        logger.error(f"Error checking if container is running: {e}")
        return False


def get_container_ip(container_name: str) -> str | None:
    """Get the container's IP address."""
    try:
        result = subprocess.run(
            [
                "docker",
                "container",
                "inspect",
                "--format",
                "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
                container_name,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None

        ip = result.stdout.strip()
        return ip if ip else None
    except Exception as e:
        logger.error(f"Error getting container IP: {e}")
        return None


def start_container(
    container_name: str,
    image_name: str,
    host_port: int,
    db_name: str,
    db_user: str,
    db_password: str,
    container_file: str | None = None,
) -> bool:
    """Start a PostgreSQL container with pgvector."""
    logger.info(f"Starting {container_name} container...")

    # Check if we need to build from Containerfile
    if container_file and Path(container_file).exists():
        logger.info(f"Building custom container image from {container_file}")

        # Build the image
        build_result = subprocess.run(
            [
                "docker",
                "build",
                "-t",
                f"local/{container_name}:latest",
                "-f",
                container_file,
                Path(container_file).parent,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if build_result.returncode != 0:
            logger.error(f"Failed to build image: {build_result.stderr}")
            return False

        image_name = f"local/{container_name}:latest"
        logger.info(f"Successfully built image: {image_name}")

    # Check if container exists but is stopped - then remove it
    if check_container_exists(container_name) and not check_container_running(
        container_name
    ):
        logger.info(f"Found stopped container {container_name}, removing it...")
        subprocess.run(
            ["docker", "container", "rm", container_name],
            capture_output=True,
            check=False,
        )

    # Create container if it doesn't exist
    if not check_container_exists(container_name):
        logger.info(f"Creating new container: {container_name}")

        # Ensure the data directory exists
        data_dir = Path.home() / ".local" / "share" / "game_loop" / "postgres-data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Directory exists in absolute path
        data_dir_abs = data_dir.resolve()

        # Create the container
        create_result = subprocess.run(
            [
                "docker",
                "run",
                "--name",
                container_name,
                "-e",
                f"POSTGRES_DB={db_name}",
                "-e",
                f"POSTGRES_USER={db_user}",
                "-e",
                f"POSTGRES_PASSWORD={db_password}",
                "-p",
                f"{host_port}:5432",
                "-v",
                f"{data_dir_abs}:/var/lib/postgresql/data",
                "-d",  # Detached mode
                image_name,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        if create_result.returncode != 0:
            logger.error(f"Failed to create container: {create_result.stderr}")
            return False

        logger.info("Container created and started successfully")

        # Wait for PostgreSQL to be ready
        logger.info("Waiting for PostgreSQL to be ready...")
        time.sleep(5)  # Initial wait
    elif not check_container_running(container_name):
        # Start an existing but stopped container
        logger.info(f"Starting existing container: {container_name}")

        start_result = subprocess.run(
            ["docker", "container", "start", container_name],
            capture_output=True,
            text=True,
            check=False,
        )

        if start_result.returncode != 0:
            logger.error(f"Failed to start container: {start_result.stderr}")
            return False

        logger.info("Container started successfully")

        # Wait for PostgreSQL to be ready
        logger.info("Waiting for PostgreSQL to be ready...")
        time.sleep(5)  # Initial wait
    else:
        logger.info(f"Container {container_name} is already running")

    return True


def initialize_database(
    host: str,
    port: int,
    db_name: str,
    db_user: str,
    db_password: str,
    project_root: str | None = None,
) -> bool:
    """Initialize the database with the required schema."""
    logger.info("Initializing database...")

    # Try to find the init_db.py script
    if project_root:
        script_path = (
            Path(project_root)
            / "src"
            / "game_loop"
            / "database"
            / "scripts"
            / "init_db.py"
        )
    else:
        script_path = Path(__file__).with_name("init_db.py")

    if not script_path.exists():
        logger.error(f"Could not find init_db.py at {script_path}")
        return False

    # Run the initialization script
    logger.info(f"Running database initialization script: {script_path}")

    env = os.environ.copy()
    env.update(
        {
            "POSTGRES_HOST": host,
            "POSTGRES_PORT": str(port),
            "POSTGRES_DB": db_name,
            "POSTGRES_USER": db_user,
            "POSTGRES_PASSWORD": db_password,
        }
    )

    init_result = subprocess.run(
        [sys.executable, str(script_path)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    if init_result.returncode != 0:
        logger.error(f"Database initialization failed: {init_result.stderr}")
        return False

    logger.info("Database initialization completed successfully")
    return True


def check_database_status(
    host: str, port: int, db_name: str, db_user: str, db_password: str
) -> bool:
    """Check if the database is ready and verify pgvector extension."""
    # Try to find the verify_db_setup.py script
    script_path = Path(__file__).with_name("verify_db_setup.py")

    if not script_path.exists():
        logger.error(f"Could not find verify_db_setup.py at {script_path}")
        return False

    # Run the verification script
    logger.info(f"Running database verification script: {script_path}")

    env = os.environ.copy()
    env.update(
        {
            "POSTGRES_HOST": host,
            "POSTGRES_PORT": str(port),
            "POSTGRES_DB": db_name,
            "POSTGRES_USER": db_user,
            "POSTGRES_PASSWORD": db_password,
        }
    )

    verify_result = subprocess.run(
        [sys.executable, str(script_path)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    # Output verification results regardless of status
    if verify_result.stdout:
        for line in verify_result.stdout.splitlines():
            logger.info(line)

    return verify_result.returncode == 0


def find_containerfile() -> str | None:
    """Find the Containerfile for PostgreSQL with pgvector."""
    # Look in common locations
    possible_paths = [
        # From script directory
        Path(__file__).parents[3] / "container" / "Containerfile.postgres",
        # From project root
        Path(__file__).parents[4] / "container" / "Containerfile.postgres",
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    return None


def verify_database_environment() -> bool:
    """Verify that all prerequisites are in place for database setup."""
    # Check if PostgreSQL client is installed (for CLI access)
    has_psql = shutil.which("psql") is not None
    if not has_psql:
        logger.warning(
            "PostgreSQL client (psql) not found, which may be useful for debugging"
        )

    # Check for asyncpg (for Python access)
    try:
        import asyncpg

        logger.info(f"asyncpg is installed: {asyncpg.__version__}")
    except ImportError:
        logger.warning(
            "asyncpg not found, may cause issues connecting to database from Python"
        )
        logger.info("Consider installing with: pip install asyncpg")

    # Check for the required Containerfile
    containerfile = find_containerfile()
    if not containerfile:
        logger.warning(
            "Could not find Containerfile.postgres, will use standard postgres image"
        )
        logger.warning("pgvector may not be available without a custom image")
    else:
        logger.info(f"Found containerfile at: {containerfile}")

    # Check for docker compose
    has_docker_compose = shutil.which("docker-compose") is not None
    if has_docker_compose:
        logger.info("docker-compose is available for multi-container setup")

    return True


def main() -> None:
    """Main entry point for script."""
    parser = argparse.ArgumentParser(
        description="Run PostgreSQL with pgvector in a Docker container"
    )
    parser.add_argument(
        "--container-name",
        default="game-loop-postgres",
        help="Name for the PostgreSQL container",
    )
    parser.add_argument(
        "--image",
        default="pgvector/pgvector:pg16",
        help="PostgreSQL image to use (default: pgvector/pgvector:pg16)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5432,
        help="Host port to map to PostgreSQL port 5432",
    )
    parser.add_argument("--db", default="game_loop", help="Database name to create")
    parser.add_argument("--user", default="postgres", help="PostgreSQL user")
    parser.add_argument("--password", default="postgres", help="PostgreSQL password")
    parser.add_argument(
        "--containerfile",
        default=None,
        help="Path to Containerfile for building custom image with pgvector",
    )
    parser.add_argument(
        "--project-root", default=None, help="Path to project root directory"
    )
    parser.add_argument(
        "--no-init", action="store_true", help="Skip database initialization"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--verify-vector",
        action="store_true",
        default=True,
        help="Explicitly verify that pgvector is working (default: True)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        default=True,
        help="Skip building custom image and use pre-built image instead "
        "(default: True)",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Find Containerfile if not specified and not skipping build
    containerfile = None
    if not args.skip_build:
        containerfile = args.containerfile
        if not containerfile:
            containerfile = find_containerfile()
            if containerfile:
                logger.info(f"Using Containerfile: {containerfile}")
    else:
        logger.info(f"Skipping custom build, using pre-built image: {args.image}")

    # Verify environment
    verify_database_environment()

    # Check if docker is installed
    if not check_docker_installed():
        logger.error("Docker is required to run the PostgreSQL container")
        sys.exit(1)

    # Start the container
    if not start_container(
        args.container_name,
        args.image,
        args.port,
        args.db,
        args.user,
        args.password,
        containerfile if not args.skip_build else None,
    ):
        logger.error("Failed to start PostgreSQL container")
        sys.exit(1)

    # Get container IP
    container_ip = get_container_ip(args.container_name)
    if container_ip:
        logger.info(
            f"Container {args.container_name} is running with IP: {container_ip}"
        )

    # Initialize database if requested
    if not args.no_init:
        # When connecting from host, use localhost instead of container IP
        host = "localhost"

        if not initialize_database(
            host, args.port, args.db, args.user, args.password, args.project_root
        ):
            logger.warning("Database initialization failed, may need manual setup")

    # Check database status
    db_status_ok = check_database_status(
        "localhost", args.port, args.db, args.user, args.password
    )

    # Verify pgvector specifically if requested
    if args.verify_vector and db_status_ok:
        try:
            import asyncio

            import asyncpg

            logger.info("Running explicit pgvector verification check...")

            async def verify_pgvector() -> bool:
                # Connect to database
                conn = await asyncpg.connect(
                    host="localhost",
                    port=args.port,
                    user=args.user,
                    password=args.password,
                    database=args.db,
                )

                # Check pgvector extension
                has_ext = await conn.fetchval(
                    "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
                )

                if not has_ext:
                    logger.error("pgvector extension is not installed in the database!")
                    logger.error(
                        "Please check that pgvector extension is created "
                        "during initialization"
                    )
                    return False

                # Test vector operations to confirm functionality
                try:
                    test_result = await conn.fetchval(
                        "SELECT '[1,2,3]'::vector <-> '[4,5,6]'::vector"
                    )
                    logger.info(
                        f"Vector distance calculation successful: {test_result}"
                    )
                    await conn.close()
                    return True
                except Exception as e:
                    logger.error(f"Vector distance calculation failed: {e}")
                    await conn.close()
                    return False

            pgvector_ok = asyncio.run(verify_pgvector())
            if pgvector_ok:
                logger.info("✅ pgvector is correctly installed and functioning!")
            else:
                logger.warning(
                    "❌ pgvector appears to be installed but not working correctly"
                )
        except ImportError:
            logger.warning(
                "Cannot run detailed pgvector checks - asyncpg not installed"
            )

    # Print connection information
    logger.info("\n--------------------------------------------------")
    logger.info("PostgreSQL with pgvector is now running")
    logger.info("--------------------------------------------------")
    logger.info("Connection Information:")
    logger.info("  Host:     localhost")
    logger.info(f"  Port:     {args.port}")
    logger.info(f"  Database: {args.db}")
    logger.info(f"  Username: {args.user}")
    logger.info(f"  Password: {args.password}")
    logger.info("\nConnection String:")
    logger.info(
        f"  postgresql://{args.user}:{args.password}@localhost:{args.port}/{args.db}"
    )
    logger.info("\nPsql Command:")
    logger.info(f"  psql -h localhost -p {args.port} -U {args.user} {args.db}")
    logger.info("--------------------------------------------------")
    logger.info("Container will continue running in the background")
    logger.info("To stop it: docker stop " + args.container_name)
    logger.info("--------------------------------------------------")


if __name__ == "__main__":
    main()
