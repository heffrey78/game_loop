# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Core Commands
- **Start game**: `make run` or `poetry run python -m game_loop.main` or `poetry run game_loop`
- **Run tests**: `make test` or `poetry run pytest tests/`
- **Lint code**: `make lint` (runs black, ruff, mypy)
- **Format code**: `make format` (auto-fixes with black/ruff)
- **Type check**: `poetry run mypy src tests`
- **Install dependencies**: `make install` or `poetry install`
- **Clean cache**: `make clean` (removes Python cache files, coverage, mypy cache)

### Database Management
- **Start database**: `make docker-check` (auto-starts if not running)
- **Initialize database**: `make docker-init` or `./scripts/manage_docker.py init`
- **Reset database**: `make db-reset` (drops and recreates from migrations)
- **Verify database**: `./scripts/manage_docker.py verify`
- **Complete setup**: `./scripts/manage_docker.py setup` (start, init, verify all in one)
- **Docker management**: Use `./scripts/manage_docker.py start|stop|init|verify|setup`

### Testing Commands
- **Run all tests**: `poetry run pytest tests/`
- **Run with coverage**: `make coverage`
- **Run integration tests**: `poetry run pytest tests/integration/`
- **Run unit tests**: `poetry run pytest tests/unit/`
- **Run specific test file**: `poetry run pytest tests/path/to/test_file.py`
- **Run tests with markers**: `poetry run pytest -m integration` or `poetry run pytest -m asyncio`

## Architecture Overview

Game Loop is a text adventure game with natural language processing that uses a modular, async architecture:

### Core Components
- **GameLoop** (`src/game_loop/core/game_loop.py`): Main orchestrator that manages game state and coordinates all systems
- **EnhancedInputProcessor**: Processes natural language input using LLM integration
- **OutputGenerator**: Creates rich narrative responses using templates and LLM
- **GameStateManager**: Manages world and player state persistence
- **CommandHandlerFactory**: Routes commands to appropriate handlers (use, look, move, etc.)

### Key Systems
- **Database Layer**: PostgreSQL with pgvector for embeddings, SQLAlchemy ORM with async support
- **LLM Integration**: Ollama for local language processing (no external APIs)
- **Embedding System**: Entity embeddings for semantic search of game elements
- **Search System**: Semantic search capabilities for finding related game content
- **Template System**: Jinja2 templates for consistent output formatting

### Data Flow
1. User input → EnhancedInputProcessor → LLM parsing → CommandHandlerFactory
2. Command execution → GameStateManager → Database persistence
3. Response generation → OutputGenerator → Template rendering → Console output

## Configuration

- **Main config**: `src/game_loop/config/models.py` (GameConfig class)
- **LLM config**: `src/game_loop/llm/config.py`
- **Database connection**: Environment variables (POSTGRES_HOST, POSTGRES_DB, etc.)
- **Ollama**: Configured in LLM config, uses local models
- **Python version**: Requires Python 3.11+ (configured in pyproject.toml)
- **Container runtime**: Uses podman-compose (fallback: docker-compose)

## Important File Patterns

- **Models**: `src/game_loop/database/models/` - SQLAlchemy models with async support
- **Repositories**: `src/game_loop/database/repositories/` - Data access layer
- **Command Handlers**: `src/game_loop/core/command_handlers/` - Game action implementations
- **Templates**: `templates/` - Jinja2 templates for output formatting
- **Migrations**: `src/game_loop/database/migrations/` - SQL migration files

## Development Notes

- **Async/Await**: The entire system is async - use await for database and LLM calls
- **Rich Console**: Use `self.console` for formatted output in game components
- **Error Handling**: Wrap LLM and database calls in try/catch blocks
- **Type Hints**: Required - mypy enforces strict typing
- **Database**: Always containerized PostgreSQL with pgvector extension
- **LLM**: Local-only processing via Ollama, no external API calls

## Testing Strategy

- **Unit tests**: `tests/unit/` - Test individual components in isolation
- **Integration tests**: `tests/integration/` - Test component interactions
- **Database tests**: Use pytest fixtures in `tests/integration/database/conftest.py`
- **LLM tests**: Mock Ollama responses for consistent testing
- **Async tests**: Use `@pytest.mark.asyncio` for async test functions

## Common Development Tasks

When adding new game commands:
1. Create handler in `src/game_loop/core/command_handlers/`
2. Register in CommandHandlerFactory
3. Add templates in `templates/actions/`
4. Write tests in `tests/unit/core/command_handlers/`

When modifying database schema:
1. Create migration in `src/game_loop/database/migrations/`
2. Update models in `src/game_loop/database/models/`
3. Update repositories if needed
4. Run `make db-reset` to apply changes

When working with embeddings:
1. Use `EmbeddingManager` for vector operations
2. Entity embeddings are in `src/game_loop/embeddings/`
3. Search functionality in `src/game_loop/search/`
