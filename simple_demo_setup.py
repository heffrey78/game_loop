#!/usr/bin/env python3
"""
Simple Demo Setup for Game Loop Technology

This creates a basic demo world without complex database operations
to showcase the technology.

Run with: poetry run python simple_demo_setup.py
"""

import asyncio
import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sqlalchemy import delete

from game_loop.config.models import DatabaseConfig
from game_loop.database.models.world import NPC, Location, Object
from game_loop.database.session_factory import DatabaseSessionFactory


async def simple_demo():
    """Create a simple demonstration setup."""
    print("🎮 Setting up Game Loop Demo World...")

    # Initialize database session factory
    db_config = DatabaseConfig()
    session_factory = DatabaseSessionFactory(db_config)
    await session_factory.initialize()

    async with session_factory.get_session() as session:
        print("Clearing existing demo data...")

        # Clear existing data
        await session.execute(delete(Object))
        await session.execute(delete(NPC))
        await session.execute(delete(Location))
        await session.commit()

        print("Creating demo locations...")

        # Create simple locations with all required fields
        wizard_tower = Location(
            name="Wizard's Tower",
            short_desc="A mystical tower filled with magical artifacts.",
            full_desc="""You stand in the grand chamber of a wizard's tower. Ancient books float through the air, 
their pages turning themselves. A crystal orb glows softly on a pedestal in the center of the room. 
The air hums with arcane energy, and magical symbols drift lazily through the air like glowing dust.""",
            location_type="magical",
            created_by="demo_setup",
        )

        alchemy_lab = Location(
            name="Alchemy Laboratory",
            short_desc="A chaotic lab with bubbling experiments.",
            full_desc="""The laboratory is filled with the sounds of bubbling cauldrons and crackling energy. 
Shelves line the walls, overflowing with strange ingredients: glowing fungi, crystallized essences, 
and vials of swirling liquid. A large workbench dominates the room, covered in half-finished experiments.""",
            location_type="laboratory",
            created_by="demo_setup",
        )

        session.add_all([wizard_tower, alchemy_lab])
        await session.flush()  # Get IDs

        print("Creating demo objects...")

        # Create magical objects with semantic variety
        objects = [
            Object(
                name="Crystal of Eternal Light",
                short_desc="A radiant crystal that never dims.",
                full_desc="A perfect crystal that radiates warm, comforting light. Its glow never fades and it feels pleasantly warm to the touch. Ancient runes are etched into its surface.",
                object_type="artifact",
                is_takeable=True,
                properties_json={
                    "keywords": [
                        "crystal",
                        "light",
                        "glowing",
                        "radiant",
                        "magical",
                        "warm",
                    ],
                    "magical": True,
                    "light_source": True,
                },
                location_id=wizard_tower.id,
            ),
            Object(
                name="Ancient Tome",
                short_desc="A weathered book of forbidden knowledge.",
                full_desc="This ancient book is bound in midnight-blue leather. Its pages seem to turn themselves, revealing glimpses of forgotten spells and dangerous knowledge.",
                object_type="book",
                is_takeable=True,
                properties_json={
                    "keywords": [
                        "book",
                        "tome",
                        "ancient",
                        "knowledge",
                        "magical",
                        "forbidden",
                    ],
                    "readable": True,
                    "dangerous": True,
                },
                location_id=wizard_tower.id,
            ),
            Object(
                name="Philosopher's Stone",
                short_desc="A legendary alchemical artifact.",
                full_desc="A small, unremarkable stone that hums with incredible power. Legend says it can transmute base metals into gold and grant eternal life.",
                object_type="alchemical",
                is_takeable=True,
                properties_json={
                    "keywords": [
                        "stone",
                        "philosopher",
                        "alchemy",
                        "power",
                        "legendary",
                        "transmutation",
                    ],
                    "powerful": True,
                    "alchemical": True,
                },
                location_id=alchemy_lab.id,
            ),
            Object(
                name="Vial of Moonlight",
                short_desc="Captured moonlight in crystalline form.",
                full_desc="A delicate vial containing what appears to be liquid moonlight. It glows with a soft silver radiance and feels cool to the touch.",
                object_type="potion",
                is_takeable=True,
                properties_json={
                    "keywords": [
                        "vial",
                        "moonlight",
                        "silver",
                        "glowing",
                        "potion",
                        "magical",
                    ],
                    "liquid": True,
                    "light_source": True,
                },
                location_id=alchemy_lab.id,
            ),
        ]

        print("Creating demo NPCs...")

        # Create NPCs with conversational content
        npcs = [
            NPC(
                name="Master Aldric",
                short_desc="An ancient wizard with piercing eyes.",
                full_desc="Master Aldric is an elderly wizard whose long beard sparkles with stardust. His eyes hold the wisdom of centuries, and his voice carries the weight of ancient knowledge.",
                npc_type="wizard",
                knowledge_json={
                    "greeting": "Welcome, traveler. I sense great potential in you.",
                    "magic": "Magic is not just power, but understanding. The universe speaks to those who know how to listen.",
                    "tower": "This tower has stood for a thousand years. It has seen empires rise and fall.",
                    "crystal": "That crystal was a gift from the Celestial Court. Its light reveals truth.",
                },
                personality_json={
                    "disposition": "wise",
                    "keywords": [
                        "wizard",
                        "ancient",
                        "knowledgeable",
                        "master",
                        "teacher",
                    ],
                },
                location_id=wizard_tower.id,
            ),
            NPC(
                name="Apprentice Zara",
                short_desc="An eager young alchemist.",
                full_desc="Zara is a young woman with bright eyes and ink-stained fingers. She moves with barely contained enthusiasm, always eager to share her latest discoveries.",
                npc_type="apprentice",
                knowledge_json={
                    "greeting": "Oh! A visitor! Want to see my latest experiment?",
                    "alchemy": "Alchemy is like cooking, but with more explosions and stranger ingredients!",
                    "experiments": "I'm working on distilling courage into liquid form. So far I've only managed to make glowing soup.",
                    "master": "Master Aldric knows everything about magic. Sometimes I think he's older than the tower itself!",
                },
                personality_json={
                    "disposition": "enthusiastic",
                    "keywords": [
                        "apprentice",
                        "young",
                        "eager",
                        "alchemist",
                        "student",
                    ],
                },
                location_id=alchemy_lab.id,
            ),
        ]

        session.add_all(objects)
        session.add_all(npcs)
        await session.commit()

        print("✅ Demo world created successfully!")
        print("\n🎯 Demo Features Available:")
        print("1. 🧠 Natural Language Processing")
        print(
            "   Try: 'I want to pick up the glowing crystal and examine it carefully'"
        )
        print("   Try: 'Can you show me all the magical items in this room?'")
        print("")
        print("2. 🔍 Semantic Search")
        print("   Try: 'find something that provides illumination'")
        print("   Try: 'look for items related to alchemy'")
        print("   Try: 'show me sources of light'")
        print("")
        print("3. 🎯 Action Classification")
        print("   Try: 'carefully place the philosopher stone on the workbench'")
        print("   Try: 'ask the wizard about ancient magic'")
        print("")
        print("4. 🎭 Context-Aware Processing")
        print("   Try: 'talk to them' (when NPCs are present)")
        print("   Try: 'use it' (when holding specific items)")
        print("")
        print("5. 🎨 Rich Output Generation")
        print("   - Beautiful console formatting")
        print("   - Context-aware descriptions")
        print("   - Helpful error messages")
        print("")
        print("🚀 To start the demo:")
        print("   poetry run python interactive_demo.py")
        print("   poetry run python demo_scenarios.py")
        print("   make run  # Full game with demo world")


if __name__ == "__main__":
    asyncio.run(simple_demo())
