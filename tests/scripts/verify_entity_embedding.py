"""
Simple verification script for entity embedding generator.

This script demonstrates the functionality of the EntityEmbeddingGenerator
and related components.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from game_loop.embeddings.entity_embeddings import EntityEmbeddingGenerator
from game_loop.embeddings.entity_registry import EntityEmbeddingRegistry


class MockEmbeddingService:
    """A simple embedding service for testing."""

    async def generate_embedding(
        self, text: str, entity_type: str = None
    ) -> list[float]:
        """Generate a mock embedding based on text length."""
        # Create deterministic embedding based on text length
        seed = len(text) % 10
        return [(i % 10 + seed) / 100 for i in range(384)]

    async def generate_embeddings_batch(
        self, texts: list[str], entity_types: list[str] | None = None
    ) -> list[list[float]]:
        """Generate mock embeddings for multiple texts."""
        return [await self.generate_embedding(text) for text in texts]


async def test_entity_embedding_generator() -> tuple[list[float], list[float]]:
    """Test the entity embedding generator functionality."""
    print("Testing EntityEmbeddingGenerator...")

    # Create service with mock
    service = MockEmbeddingService()
    entity_generator = EntityEmbeddingGenerator(service)

    # Create test entities
    character = {
        "id": "char1",
        "name": "Wizard",
        "description": "A powerful magic user",
        "personality": "Wise and mysterious",
    }

    location = {
        "id": "loc1",
        "name": "Tower",
        "description": "A tall tower overlooking the valley",
        "environment": "Stone structure",
    }

    # Generate embeddings
    character_embedding = await entity_generator.generate_entity_embedding(
        character, "character"
    )
    location_embedding = await entity_generator.generate_entity_embedding(
        location, "location"
    )

    # Print dimensions
    print(f"Character embedding dimension: {len(character_embedding)}")
    print(f"Location embedding dimension: {len(location_embedding)}")

    # Verify different preprocessing produces different embeddings
    if character_embedding != location_embedding:
        print("✓ Different entity types produce different embeddings")
    else:
        print("✗ Different entity types produce the same embedding")

    return character_embedding, location_embedding


async def test_entity_registry():
    """Test the entity embedding registry functionality."""
    print("\nTesting EntityEmbeddingRegistry...")

    # Get embeddings from previous test
    char_embedding, loc_embedding = await test_entity_embedding_generator()

    # Create registry
    registry = EntityEmbeddingRegistry(dimension=384)

    # Register entities
    await registry.register_entity(
        "char1", "character", char_embedding, {"name": "Wizard"}
    )
    await registry.register_entity("loc1", "location", loc_embedding, {"name": "Tower"})

    # Check registry content
    all_ids = registry.get_all_entity_ids()
    print(f"Registry contains {len(all_ids)} entities: {all_ids}")

    # Test similarity search
    similar = await registry.find_similar_entities(char_embedding, top_k=2)
    print(
        f"Similar to character: {[id for id, score in similar]} with scores "
        f"{[score for id, score in similar]}"
    )

    # Test filtering by type
    characters = registry.get_all_entity_ids(entity_type="character")
    print(f"Character entities: {characters}")

    # Test export/import
    tmp_file = Path("./tmp_registry.pkl")
    await registry.export_registry(tmp_file)
    print(f"Registry exported to {tmp_file}")

    # Create new registry and import
    new_registry = EntityEmbeddingRegistry(dimension=384)
    await new_registry.import_registry(tmp_file)
    print(f"Registry imported with {len(new_registry.get_all_entity_ids())} entities")

    # Clean up
    tmp_file.unlink(missing_ok=True)


async def main():
    """Run verification tests."""
    print("=== Entity Embedding System Verification ===\n")

    # Also write output to a file
    with open("entity_embedding_verification.log", "w") as f:
        f.write("=== Entity Embedding System Verification ===\n\n")

        # Redirect standard output temporarily
        import sys

        original_stdout = sys.stdout
        sys.stdout = f

        try:
            await test_entity_registry()
            f.write("\n=== Verification Complete ===\n")
        finally:
            # Restore standard output
            sys.stdout = original_stdout

    await test_entity_registry()
    print("\n=== Verification Complete ===")
    print("Results also written to entity_embedding_verification.log")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
