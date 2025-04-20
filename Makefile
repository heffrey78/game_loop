.PHONY: install clean lint format test coverage pre-commit

# Default Python interpreter
PYTHON := python

# Install the package in development mode
install:
	poetry install

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

# Setup pre-commit hooks
pre-commit:
	poetry run pre-commit install
	poetry run pre-commit run --all-files
