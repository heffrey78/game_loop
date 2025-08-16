.PHONY: install clean lint format test coverage pre-commit docker-start docker-stop docker-init docker-setup docker-check run db-reset

# Default Python interpreter
PYTHON := python

# Install the package in development mode
install:
	poetry install
	poetry install --with dev

# Clean Python cache files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

# Run linters
lint:
	poetry run black --check .
	poetry run ruff check .
	poetry run mypy src tests

# Format code
format:
	poetry run black .
	poetry run ruff check --fix .

# Run tests
test:
	poetry run pytest tests/

# Run tests with coverage report
coverage:
	poetry run pytest --cov=src tests/ --cov-report=html

# Check if docker container is running and start if needed
docker-check:
	@if ! docker ps --format '{{.Names}}' | grep -q "game-loop-postgres"; then \
	    echo "PostgreSQL container not running. Starting it now..."; \
	    docker-compose up -d postgres; \
	    echo "Waiting for PostgreSQL to initialize..."; \
	    sleep 5; \
	else \
	    echo "PostgreSQL container is already running."; \
	fi

# Run the game (ensuring database is running and initialized)
run: docker-check docker-init
	poetry run python -m game_loop.main

# Completely reset and recreate the database from migrations
db-reset: docker-check
	@echo "Dropping and recreating the database from scratch..."
	DB_FRESH_START=true ./scripts/manage_docker.py init
	@echo "Database reset complete."

# Setup pre-commit hooks
pre-commit:
	poetry run pre-commit install
	poetry run pre-commit run --all-files

# Docker management commands
docker-start:
	./scripts/manage_docker.py start

docker-stop:
	./scripts/manage_docker.py stop

docker-init:
	./scripts/manage_docker.py init

docker-verify:
	./scripts/manage_docker.py verify

# Complete Docker setup (start, init, verify)
docker-setup:
	./scripts/manage_docker.py setup
