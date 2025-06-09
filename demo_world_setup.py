#!/usr/bin/env python3
"""
Demo World Setup Script
Creates a demonstration world that showcases the sophisticated technology in Game Loop.

Run with: poetry run python demo_world_setup.py
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sqlalchemy.ext.asyncio import AsyncSession

from game_loop.config.models import DatabaseConfig
from game_loop.database.models.world import NPC, Location, LocationConnection, Object
from game_loop.database.session_factory import DatabaseSessionFactory
from game_loop.embeddings.manager import EmbeddingManager
from game_loop.embeddings.service import EmbeddingService
from game_loop.llm.config import LLMConfig
from game_loop.llm.ollama.client import OllamaClient


async def clear_existing_world(session: AsyncSession):
    """Clear existing world data."""
    from sqlalchemy import delete

    # Delete in correct order due to foreign key constraints
    await session.execute(delete(LocationConnection))
    await session.execute(delete(Object))
    await session.execute(delete(NPC))
    await session.execute(delete(Location))
    await session.commit()


async def create_demo_world():
    """Create a demonstration world with interesting scenarios."""
    # Initialize database session factory
    db_config = DatabaseConfig()
    session_factory = DatabaseSessionFactory(db_config)
    await session_factory.initialize()

    async with session_factory.get_session() as session:
        # Clear existing world
        print("Clearing existing world data...")
        await clear_existing_world(session)

        # Initialize embedding services
        llm_config = LLMConfig()
        ollama_client = OllamaClient(llm_config)
        embedding_service = EmbeddingService(ollama_client)
        from game_loop.embeddings.entity_generator import EntityEmbeddingGenerator

        entity_generator = EntityEmbeddingGenerator(embedding_service)
        embedding_manager = EmbeddingManager(entity_generator)

        print("Creating demonstration world...")

        # Create locations that showcase NLP understanding
        wizard_tower = Location(
            name="Archmage's Tower",
            short_desc="A grand magical library filled with ancient knowledge.",
            full_desc="""You stand in the grand library of the Archmage's Tower. Ancient tomes line the walls,
their spines glowing with ethereal light. A massive crystalline orb hovers in the center of the room,
pulsing with arcane energy. The air crackles with magic, and you can feel the weight of centuries
of knowledge pressing down upon you. To the north, a spiral staircase ascends into darkness.""",
            location_type="library",
        )

        alchemy_lab = Location(
            name="Alchemical Laboratory",
            short_desc="A chaotic laboratory filled with bubbling experiments.",
            full_desc="""The laboratory is a chaotic symphony of bubbling beakers, smoking cauldrons, and
mysterious apparatus. Shelves overflow with ingredients: glowing mushrooms, crystallized dragon tears,
and vials of shifting colors. A workbench dominates the eastern wall, covered in half-finished
experiments. The pungent smell of sulfur mingles with sweet floral notes. A heavy wooden door
leads west back to the tower.""",
            location_type="laboratory",
        )

        enchanted_garden = Location(
            name="Enchanted Garden",
            short_desc="An impossible indoor garden with magical plants.",
            full_desc="""You've entered a impossible garden that exists within the tower. Despite being indoors,
a warm sun shines overhead and a gentle breeze carries the scent of exotic flowers. Paths of luminous
stones wind between plants that seem to watch you with curious intelligence. A fountain of liquid
starlight bubbles peacefully in the center. Time feels different here - minutes might be hours,
or hours merely heartbeats.""",
            location_type="garden",
        )

        # Add locations to session
        session.add_all([wizard_tower, alchemy_lab, enchanted_garden])
        await session.flush()  # Get IDs

        # Create items that demonstrate semantic search
        items = [
            Object(
                name="Crystal of Eternal Light",
                short_desc="A glowing crystal that never dims.",
                full_desc="A perfect crystal that radiates a warm, comforting glow. It never dims and feels slightly warm to the touch.",
                is_takeable=True,
                object_type="artifact",
                properties_json={
                    "keywords": [
                        "crystal",
                        "light",
                        "glowing",
                        "eternal",
                        "warm",
                        "radiant",
                    ]
                },
                location_id=wizard_tower.id,
            ),
            Object(
                name="Tome of Forbidden Knowledge",
                short_desc="An ancient book bound in unknown scales.",
                full_desc="An ancient book bound in scales of unknown origin. The pages seem to turn themselves, revealing secrets best left unlearned.",
                is_takeable=True,
                object_type="book",
                properties_json={
                    "keywords": [
                        "book",
                        "tome",
                        "forbidden",
                        "knowledge",
                        "ancient",
                        "secrets",
                    ]
                },
                location_id=wizard_tower.id,
            ),
            Object(
                name="Philosopher's Stone",
                short_desc="A small stone that hums with latent power.",
                full_desc="A small, unassuming stone that supposedly can transmute base metals to gold and grant immortality. It hums with latent power.",
                is_takeable=True,
                object_type="alchemical",
                properties_json={
                    "keywords": [
                        "stone",
                        "philosopher",
                        "alchemy",
                        "transmute",
                        "immortality",
                        "power",
                    ]
                },
                location_id=alchemy_lab.id,
            ),
            Object(
                name="Vial of Liquid Moonlight",
                short_desc="A vial containing captured moonlight.",
                full_desc="A delicate glass vial containing what appears to be captured moonlight. It glows with a silver radiance and feels cool to the touch.",
                is_takeable=True,
                object_type="potion",
                properties_json={
                    "keywords": [
                        "vial",
                        "moonlight",
                        "liquid",
                        "silver",
                        "glowing",
                        "potion",
                    ]
                },
                location_id=alchemy_lab.id,
            ),
            Object(
                name="Seed of the World Tree",
                short_desc="A seed pulsing with life potential.",
                full_desc="A single seed that pulses with life. You can feel the potential of an entire forest within this tiny package.",
                is_takeable=True,
                object_type="nature",
                properties_json={
                    "keywords": ["seed", "tree", "world", "life", "forest", "nature"]
                },
                location_id=enchanted_garden.id,
            ),
        ]

        # Create NPCs for conversation demos
        npcs = [
            NPC(
                name="Archmagus Aldric",
                short_desc="An elderly wizard with ancient wisdom.",
                full_desc="An elderly wizard with a long silver beard and eyes that sparkle with ancient wisdom. His robes seem to be woven from starlight itself.",
                npc_type="wizard",
                knowledge_json={
                    "greeting": "Ah, a visitor! How refreshing. I am Aldric, keeper of this tower. What brings you to my domain?",
                    "knowledge": "I have studied the arcane arts for centuries. The secrets of the universe are written in the very fabric of reality, if one knows how to read them.",
                    "crystal": "The Crystal of Eternal Light? A fascinating artifact. It was gifted to me by the Sun Sprites of the Eastern Realms. Its light can reveal hidden truths.",
                    "help": "If you seek knowledge, you must prove yourself worthy. Perhaps you could help me with a small task? I've misplaced my spectacles somewhere in the tower.",
                },
                personality_json={
                    "keywords": [
                        "wizard",
                        "aldric",
                        "archmagus",
                        "old",
                        "wise",
                        "keeper",
                    ]
                },
                location_id=wizard_tower.id,
            ),
            NPC(
                name="Pip the Apprentice",
                short_desc="A young, enthusiastic apprentice.",
                full_desc="A young, enthusiastic apprentice with wild red hair and ink-stained fingers. They practically vibrate with nervous energy.",
                npc_type="apprentice",
                knowledge_json={
                    "greeting": "Oh! A visitor! Master Aldric didn't mention anyone was coming. I'm Pip, the apprentice. Don't touch anything explosive!",
                    "alchemy": "I love alchemy! It's like cooking but with more explosions and fewer edible results. Did you know that mixing moonlight with powdered star sapphire creates the most beautiful blue flame?",
                    "help": "You want to help? That's wonderful! I accidentally mixed up all the ingredient labels in the cabinet. If you could help me sort them out, I'd be forever grateful!",
                    "master": "Master Aldric is brilliant but terribly forgetful. He once spent three days looking for his hat while wearing it.",
                },
                personality_json={
                    "keywords": [
                        "pip",
                        "apprentice",
                        "young",
                        "student",
                        "helper",
                        "alchemist",
                    ]
                },
                location_id=alchemy_lab.id,
            ),
        ]

        # Add all entities
        session.add_all(items)
        session.add_all(npcs)
        await session.commit()

        # Generate embeddings for all entities
        print("Generating embeddings for semantic search...")

        try:
            # Generate location embeddings
            for location in [wizard_tower, alchemy_lab, enchanted_garden]:
                success = await embedding_manager.create_or_update_location_embedding(
                    location.id
                )
                if success:
                    print(f"Generated embedding for location: {location.name}")

            # Generate item embeddings
            for item in items:
                success = await embedding_manager.create_or_update_object_embedding(
                    item.id
                )
                if success:
                    print(f"Generated embedding for object: {item.name}")

            # Generate NPC embeddings
            for npc in npcs:
                success = await embedding_manager.create_or_update_npc_embedding(npc.id)
                if success:
                    print(f"Generated embedding for NPC: {npc.name}")

            print("Embeddings generated successfully!")
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            print("Continuing without embeddings...")

        print("Demo world created successfully!")
        print("\nKey features to demonstrate:")
        print(
            "1. Natural Language: Try 'I want to pick up the glowing crystal and examine it'"
        )
        print("2. Semantic Search: Try 'look for something that provides illumination'")
        print(
            "3. Context Awareness: Try 'talk to the wizard about the strange crystal'"
        )
        print(
            "4. Complex Commands: Try 'take the moonlight vial and put it in the cabinet'"
        )
        print("5. Rich Descriptions: Explore the locations and examine objects")


if __name__ == "__main__":
    asyncio.run(create_demo_world())
