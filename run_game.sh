#!/bin/bash

# Run Game Loop text adventure game
# This script runs the game from the project root directory

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

# Use Poetry to run the game (if poetry is set up)
if command -v poetry &> /dev/null; then
    cd "$SCRIPT_DIR" && poetry run python -m game_loop.main
else
    # Fall back to running with Python directly
    cd "$SCRIPT_DIR" && python -m src.game_loop.main
fi
