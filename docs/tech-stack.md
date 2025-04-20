# Game Loop Project Tech Stack

## Overview
This document outlines the technology stack used in the Game Loop project, a text-based game engine with natural language processing capabilities. The project maintains a local-first architecture with all NLP processing happening on-device.

## Core Technologies

### Backend Framework
- **Language**: Python (primary implementation language)
- **Runtime Environment**: Python 3.10+
- **Dependency Management**:
  - **Poetry**: Modern Python dependency management with:
    - Deterministic builds via lock files
    - Virtual environment management
    - Dependency resolution
    - Publishing capabilities
  - **PDM** (alternative): Performance-oriented Python dependency manager with PEP 582 support

### Database
- **Primary Database**: PostgreSQL
  - **Always Containerized**: PostgreSQL runs in a dedicated Docker container
  - **Vector Storage**: pgvector extension for embedding storage and similarity search
  - **Graph Database**: Integration for relationship modeling between game entities
  - **Relational Storage**: Traditional tables for structured data
  - **Connection**: SQLAlchemy ORM with asyncpg driver
  - **Container Configuration**:
    - Persistent volume mapping for data durability
    - Custom initialization scripts for extension setup
    - Health checks and automatic recovery

### AI & NLP Components (Local-First)
- **LLM Service**:
  - **Ollama**: Primary language understanding engine
    - Handles text parsing, intent recognition, and response generation
    - Configurable model selection (Mistral, Llama, etc.)
    - Local inference capability
    - Customizable parameters (temperature, context length)
    - Hardware acceleration support (CUDA, Metal)
    - Direct handling of user input parsing and intent extraction
    - Contextual response generation based on game state
  - **Model Formats**: GGUF, GGML optimized for local inference
  - **Quantization**: Support for various quantization levels (4-bit to 16-bit)

- **Vector Storage**:
  - **PostgreSQL with pgvector**: For similarity search of game objects and state
  - **Simple Embedding Generation**: Using Ollama's embedding API
  - Storage and retrieval of game context vectors

- **Language Processing Strategy**:
  - Prompt engineering for game command interpretation
  - Few-shot examples for consistent command parsing
  - System prompts for maintaining game context
  - Simple pre-processing of user input (tokenization, normalization)
  - Context window management for game state awareness

### Game Engine Components
- **Input Processor**: Parses and validates user text input
- **Game State Manager**:
  - World State Tracker
  - Player State Tracker
- **Rules Engine**:
  - Static rule definitions
  - Dynamic rule generation system
- **Output Generator**: Creates narrative responses using LLM

### Configuration System
- **Configuration Manager**: Centralized configuration handling
- **Config Sources**:
  - YAML/TOML configuration files (using PyYAML/tomli)
  - Command-line parameters
  - Environment variables support
- **Configuration Libraries**: Pydantic for schema validation

## User Interface
- **Primary Interface**: Command-line interface (CLI)
  - **Framework**: Click/Typer for rich CLI experiences
  - **Rich**: Terminal formatting, progress bars, tables
  - **Prompt Toolkit**: Interactive prompts and input handling
  - **Textual**: TUI (Text User Interface) for more advanced displays
- **Display**: Text-based output with optional formatting
  - ANSI color support
  - Unicode characters for improved visuals
  - Optional ASCII art for location descriptions

## Development Tools
- **Version Control**: Git
- **Documentation**:
  - Markdown for documentation
  - Mermaid for diagrams
  - MkDocs with Material theme for documentation site
- **Testing Framework**:
  - Pytest for unit and integration testing
  - Hypothesis for property-based testing
  - pytest-cov for coverage reporting
- **Linting & Formatting**:
  - Black for code formatting
  - Ruff for fast linting (replacing Flake8)
  - mypy for static type checking
  - pre-commit hooks for automated checks
- **CI/CD**:
  - GitHub Actions for automated testing
  - Automated dependency updates (Dependabot)

## Architecture Pattern
- Event-driven architecture
- Component-based design
- Modular system with clear separation of concerns
- Async I/O patterns for non-blocking operations
- Local-first processing with no external API dependencies
- Privacy-focused design with no data leaving the device

## Deployment
- **Local Deployment**: Stand-alone application
- **Container Platform**: Docker
  - Multi-container deployment with Docker Compose
  - OCI-compliant image support
  - Volume mapping for model storage
- **Containerized Services**:
  - **PostgreSQL**: Always runs in a dedicated container
    - Auto-initialized with vector extensions
    - Persistent volumes for data storage
    - Configured with optimized settings for vector operations
  - **Ollama LLM service**: Optionally containerized
  - Additional service containers as needed
- **Dependencies**:
  - Python environment
  - Optional GPU support for acceleration

## Performance Considerations
- Vector database optimization for quick similarity searches
- Efficient context management for LLM requests
- Caching strategy for frequently accessed game state elements
- Async processing for I/O-bound operations
- Batch processing for embedding generation
- Model quantization techniques to reduce memory footprint
- Streaming responses for better user experience
- Memory-efficient token processing

## Security Considerations
- Local-first approach ensures no data leaves the device
- Configuration for sensitive settings (local model paths, parameters)
- Input validation and sanitization
- Dependency scanning (Safety)
- Container security best practices
- No telemetry or data collection

## Future Tech Considerations
- Web interface option (FastAPI + HTMX)
- Enhanced containerization with Kubernetes
- Additional local LLM integrations
- GPU acceleration for embedding generation and inference
- Integration with vector database alternatives (Milvus, Weaviate)
- Local fine-tuning of models for game-specific knowledge
- Neural-symbolic integration for improved reasoning

## Project Scaffolding
- **Project Structure**: src-based layout
- **Package Management**: pyproject.toml-based configuration
- **Development Containers**: Dev container configuration for consistent environments
- **Makefile**: Common development tasks automation
