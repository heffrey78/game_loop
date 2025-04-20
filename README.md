# Game Loop

A text adventure game with natural language processing capabilities.

## Overview

Game Loop is an interactive text adventure system that uses natural language processing to create dynamic and responsive game experiences. Players can interact with the game world using natural language commands, and the system will respond with contextually appropriate descriptions and interactions.

## Features

- Natural language command processing
- Dynamic world generation
- Contextual conversations with NPCs
- Vector-based semantic search for game elements
- Persistent game state management
- Rules and evolution systems
- Integration with LLM services via Ollama

## Getting Started

### Prerequisites

- Python 3.10+
- Poetry
- Docker (for running PostgreSQL with pgvector)
- Ollama with appropriate models

### Installation

1. Clone the repository
2. Install dependencies with Poetry:
   ```
   poetry install
   ```
3. Configure your environment (see Configuration)
4. Run the game:
   ```
   poetry run game_loop
   ```

## Architecture

The Game Loop system is built with a modular architecture, allowing for easy extension and modification:

- Core Game Loop: Manages the main game flow and interactions
- Input Processing: Handles natural language input parsing
- NLP Integration: Connects to LLM services for language understanding
- Database Integration: Stores game state, world data, and vector embeddings
- Output Generation: Creates rich, contextual responses

## Development

This project follows a structured development approach with clear commit checkpoints. See the implementation plan in `docs/implementation/game_loop_implementation_plan.md` for details.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
