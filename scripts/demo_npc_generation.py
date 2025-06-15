#!/usr/bin/env python3
"""
Demo script for NPC Generation System.

This demonstrates the NPC generation pipeline without requiring external dependencies.
Shows how different components work together to create contextually appropriate NPCs.
"""

import asyncio
import json
from datetime import datetime
from uuid import uuid4

from game_loop.core.models.npc_models import (
    DialogueContext,
    GeneratedNPC,
    NPCDialogueState,
    NPCKnowledge,
    NPCPersonality,
)
from game_loop.core.world.npc_context_collector import NPCContextCollector
from game_loop.core.world.npc_theme_manager import NPCThemeManager
from game_loop.state.models import Location, NonPlayerCharacter, WorldState


# Mock classes for demo purposes
class MockSessionFactory:
    """Mock session factory for demo."""

    async def get_session(self):
        return MockAsyncContextManager()


class MockAsyncContextManager:
    """Mock async context manager."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockOllamaClient:
    """Mock Ollama client that generates believable NPC data."""

    def generate(self, model: str, prompt: str, options: dict = None):
        """Generate mock NPC data based on prompt analysis."""

        # Analyze prompt to determine what kind of NPC to generate
        if "merchant" in prompt.lower():
            return self._generate_merchant_response()
        elif "hermit" in prompt.lower():
            return self._generate_hermit_response()
        elif "guard" in prompt.lower():
            return self._generate_guard_response()
        elif "forest" in prompt.lower():
            return self._generate_forest_npc_response()
        elif "village" in prompt.lower():
            return self._generate_village_npc_response()
        else:
            return self._generate_generic_response()

    def _generate_merchant_response(self):
        return {
            "response": json.dumps(
                {
                    "name": "Elara Goldweaver",
                    "description": "A well-dressed woman with keen eyes and calloused hands from years of handling goods. Her jewelry gleams subtly in the light.",
                    "personality_traits": [
                        "shrewd",
                        "personable",
                        "ambitious",
                        "detail-oriented",
                    ],
                    "motivations": [
                        "expand_trade_network",
                        "accumulate_wealth",
                        "build_reputation",
                    ],
                    "fears": ["market_collapse", "theft", "losing_customers"],
                    "background": "Elara started as a simple cloth trader but built her business through careful investments and excellent customer relationships. She now deals in exotic goods from distant lands.",
                    "knowledge_areas": [
                        "trade_routes",
                        "market_prices",
                        "rare_goods",
                        "customer_psychology",
                    ],
                    "speech_style": "speaks with confident politeness, often using trade terminology",
                    "initial_dialogue": "Welcome, traveler! I have the finest goods from across the realm. What might catch your interest today?",
                    "special_abilities": ["appraisal", "negotiation"],
                }
            )
        }

    def _generate_hermit_response(self):
        return {
            "response": json.dumps(
                {
                    "name": "Old Henrik",
                    "description": "A weathered man with piercing gray eyes and a long beard streaked with silver. His simple robes are patched but clean.",
                    "personality_traits": [
                        "wise",
                        "reclusive",
                        "patient",
                        "mysterious",
                    ],
                    "motivations": [
                        "seek_inner_peace",
                        "preserve_ancient_knowledge",
                        "help_worthy_souls",
                    ],
                    "fears": [
                        "corruption_of_nature",
                        "loss_of_solitude",
                        "forgotten_wisdom",
                    ],
                    "background": "Once a scholar in the great libraries, Henrik retreated to the wilderness after witnessing the burning of ancient texts. He now guards old secrets.",
                    "knowledge_areas": [
                        "ancient_lore",
                        "herbalism",
                        "meditation",
                        "forest_spirits",
                    ],
                    "speech_style": "speaks slowly and thoughtfully, often in riddles or metaphors",
                    "initial_dialogue": "Few find their way to my dwelling. The forest itself must have guided you here, child.",
                    "special_abilities": ["herb_identification", "ancient_knowledge"],
                }
            )
        }

    def _generate_guard_response(self):
        return {
            "response": json.dumps(
                {
                    "name": "Captain Marcus Ironwood",
                    "description": "A sturdy man in polished chainmail with alert brown eyes and numerous small scars from years of service. His sword is well-maintained.",
                    "personality_traits": ["dutiful", "vigilant", "honorable", "stern"],
                    "motivations": [
                        "protect_the_people",
                        "uphold_justice",
                        "maintain_order",
                    ],
                    "fears": ["failing_in_duty", "corruption", "civilian_casualties"],
                    "background": "Marcus rose through the ranks through dedication and courage. He's known for his fairness and has never lost a person under his protection.",
                    "knowledge_areas": [
                        "combat_tactics",
                        "local_threats",
                        "patrol_routes",
                        "criminal_behavior",
                    ],
                    "speech_style": "speaks with military precision and authority, but shows warmth to civilians",
                    "initial_dialogue": "Halt, traveler! State your business in our territory. These roads can be dangerous for the unwary.",
                    "special_abilities": ["combat_expertise", "threat_assessment"],
                }
            )
        }

    def _generate_forest_npc_response(self):
        return {
            "response": json.dumps(
                {
                    "name": "Willow Greenthumb",
                    "description": "A young woman with earth-stained clothes and flowers woven into her dark hair. Her eyes hold the depth of ancient forests.",
                    "personality_traits": [
                        "nature-loving",
                        "intuitive",
                        "gentle",
                        "protective",
                    ],
                    "motivations": [
                        "protect_the_forest",
                        "heal_wounded_creatures",
                        "maintain_balance",
                    ],
                    "fears": ["deforestation", "pollution", "loss_of_wildlife"],
                    "background": "Raised by druids after her village was destroyed, Willow has become one with the forest and serves as its guardian.",
                    "knowledge_areas": [
                        "forest_ecology",
                        "animal_behavior",
                        "natural_healing",
                        "weather_patterns",
                    ],
                    "speech_style": "speaks softly with nature metaphors, often pausing to listen to forest sounds",
                    "initial_dialogue": "The trees whisper of your arrival, stranger. They sense you mean no harm to this sacred place.",
                    "special_abilities": ["animal_communication", "natural_healing"],
                }
            )
        }

    def _generate_village_npc_response(self):
        return {
            "response": json.dumps(
                {
                    "name": "Thomas the Baker",
                    "description": "A cheerful man with flour-dusted apron and strong arms from kneading dough. The scent of fresh bread always surrounds him.",
                    "personality_traits": [
                        "friendly",
                        "hardworking",
                        "generous",
                        "community-minded",
                    ],
                    "motivations": [
                        "feed_the_community",
                        "perfect_recipes",
                        "support_neighbors",
                    ],
                    "fears": ["poor_harvests", "hungry_children", "failing_health"],
                    "background": "Thomas learned baking from his father and has fed the village for twenty years. He's known for his kind heart and excellent bread.",
                    "knowledge_areas": [
                        "baking",
                        "local_families",
                        "grain_quality",
                        "community_events",
                    ],
                    "speech_style": "speaks warmly with enthusiasm about food and community",
                    "initial_dialogue": "Good day! The bread's fresh from the oven. Would you like to try a sample? Nothing brings people together like warm bread!",
                    "special_abilities": ["master_baking", "community_knowledge"],
                }
            )
        }

    def _generate_generic_response(self):
        return {
            "response": json.dumps(
                {
                    "name": "Mysterious Wanderer",
                    "description": "A figure in a worn traveling cloak with eyes that seem to hold many secrets.",
                    "personality_traits": ["enigmatic", "experienced", "cautious"],
                    "motivations": ["seek_purpose", "gather_knowledge"],
                    "fears": ["being_discovered", "staying_too_long"],
                    "background": "Little is known about this wanderer's past.",
                    "knowledge_areas": ["travel", "survival"],
                    "speech_style": "speaks carefully, revealing little",
                    "initial_dialogue": "Greetings, fellow traveler. The roads are long and full of stories.",
                    "special_abilities": ["survival_skills"],
                }
            )
        }


class MockEmbeddingManager:
    """Mock embedding manager for demo."""

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate mock embedding vector."""
        # Simple hash-based pseudo-embedding for demo
        hash_val = hash(text) % 1000
        return [float(i + hash_val) / 1000.0 for i in range(10)]


async def create_sample_world():
    """Create a sample world with different location types for demo."""
    world_state = WorldState()

    # Create Village location
    village_id = uuid4()
    village = Location(
        location_id=village_id,
        name="Millbrook Village",
        description="A peaceful farming village nestled in a valley with stone cottages and cobblestone paths.",
        connections={"north": uuid4(), "east": uuid4()},
        objects={},
        npcs={},
        state_flags={"theme": "Village", "type": "village", "population": "small"},
    )

    # Create Forest location
    forest_id = uuid4()
    forest = Location(
        location_id=forest_id,
        name="Whispering Woods",
        description="An ancient forest where sunlight filters through thick canopy and mysterious sounds echo.",
        connections={"south": village_id},
        objects={},
        npcs={},
        state_flags={"theme": "Forest", "type": "wilderness", "danger_level": "low"},
    )

    # Create Crossroads location
    crossroads_id = uuid4()
    crossroads = Location(
        location_id=crossroads_id,
        name="Trader's Crossroads",
        description="A busy intersection where four roads meet, marked by a weathered signpost and a small inn.",
        connections={
            "north": forest_id,
            "south": village_id,
            "east": uuid4(),
            "west": uuid4(),
        },
        objects={},
        npcs={},
        state_flags={"theme": "Crossroads", "type": "junction", "traffic": "high"},
    )

    world_state.locations[village_id] = village
    world_state.locations[forest_id] = forest
    world_state.locations[crossroads_id] = crossroads

    return world_state, village_id, forest_id, crossroads_id


async def demonstrate_npc_generation():
    """Demonstrate the complete NPC generation process."""

    print("🎭 NPC Generation System Demo")
    print("=" * 50)

    # Setup
    print("\n📋 Setting up demo environment...")
    world_state, village_id, forest_id, crossroads_id = await create_sample_world()

    session_factory = MockSessionFactory()
    theme_manager = NPCThemeManager(world_state, session_factory)
    context_collector = NPCContextCollector(world_state, session_factory)

    # Demo different locations and archetypes
    locations_to_demo = [
        (village_id, "Village", "populate_location"),
        (forest_id, "Forest", "quest_related"),
        (crossroads_id, "Crossroads", "random_encounter"),
    ]

    generated_npcs = []

    for location_id, theme_name, purpose in locations_to_demo:
        print(f"\n🏘️  Generating NPC for {theme_name} ({purpose})")
        print("-" * 40)

        location = world_state.locations[location_id]

        # Step 1: Collect context
        print("📊 Collecting generation context...")
        context = await context_collector.collect_generation_context(
            location_id, purpose
        )

        print(f"   Location: {context.location.name}")
        print(f"   Theme: {context.location_theme.name}")
        print(f"   Purpose: {context.generation_purpose}")
        print(f"   Player Level: {context.player_level}")

        # Step 2: Determine archetype
        print("\n🎯 Determining appropriate archetype...")
        archetype = await theme_manager.determine_npc_archetype(context)
        print(f"   Selected Archetype: {archetype}")

        # Step 3: Get personality template
        print("\n🧠 Creating personality template...")
        personality_template = await theme_manager.get_personality_template(
            archetype, theme_name
        )
        print(f"   Base Traits: {', '.join(personality_template.traits[:3])}")
        print(f"   Motivations: {', '.join(personality_template.motivations[:2])}")

        # Step 4: Apply cultural variations
        print("\n🌍 Applying cultural variations...")
        varied_personality = await theme_manager.generate_cultural_variations(
            personality_template, location
        )
        print(
            f"   Cultural Traits Added: {[t for t in varied_personality.traits if t not in personality_template.traits]}"
        )

        # Step 5: Mock LLM generation
        print("\n🤖 Generating NPC with AI...")
        ollama_client = MockOllamaClient()

        # Create a prompt-like description for the mock LLM
        prompt_context = (
            f"Generate {archetype} for {theme_name} location for {purpose} purpose"
        )
        llm_response = ollama_client.generate("mock-model", prompt_context)
        npc_data = json.loads(llm_response["response"])

        # Step 6: Create complete NPC
        print("\n🎭 Assembling complete NPC...")
        base_npc = NonPlayerCharacter(
            npc_id=uuid4(), name=npc_data["name"], description=npc_data["description"]
        )

        personality = NPCPersonality(
            name=npc_data["name"],
            archetype=archetype,
            traits=npc_data["personality_traits"],
            motivations=npc_data["motivations"],
            fears=npc_data["fears"],
            speech_patterns={"style": npc_data["speech_style"]},
            relationship_tendencies={},
        )

        knowledge = NPCKnowledge(
            expertise_areas=npc_data["knowledge_areas"],
            world_knowledge={"background": npc_data["background"]},
            local_knowledge={"location": location.name, "theme": theme_name},
        )

        dialogue_state = NPCDialogueState(
            current_mood="neutral", active_topics=["introduction", "local_area"]
        )

        # Step 7: Generate embedding
        print("\n🔗 Generating embeddings...")
        embedding_manager = MockEmbeddingManager()
        embedding_text = (
            f"{npc_data['name']} {archetype} {' '.join(npc_data['personality_traits'])}"
        )
        embedding = await embedding_manager.generate_embedding(embedding_text)

        generated_npc = GeneratedNPC(
            base_npc=base_npc,
            personality=personality,
            knowledge=knowledge,
            dialogue_state=dialogue_state,
            generation_metadata={
                "archetype": archetype,
                "location_theme": theme_name,
                "generation_purpose": purpose,
                "timestamp": datetime.now().isoformat(),
            },
            embedding_vector=embedding[:5],  # Show first 5 dimensions
        )

        generated_npcs.append(generated_npc)

        # Display results
        print(f"\n✨ Generated NPC: {generated_npc.base_npc.name}")
        print(f"   Description: {generated_npc.base_npc.description}")
        print(f"   Archetype: {generated_npc.personality.archetype}")
        print(f"   Key Traits: {', '.join(generated_npc.personality.traits[:3])}")
        print(f"   Expertise: {', '.join(generated_npc.knowledge.expertise_areas[:3])}")
        print(f"   Greeting: \"{npc_data['initial_dialogue']}\"")
        print(
            f"   Embedding: [{', '.join(f'{x:.2f}' for x in generated_npc.embedding_vector)}...]"
        )

    # Demonstrate NPC validation
    print("\n🔍 Validating NPCs...")
    print("-" * 40)

    for i, npc in enumerate(generated_npcs):
        location = world_state.locations[list(world_state.locations.keys())[i]]
        is_consistent = await theme_manager.validate_npc_consistency(npc, location)
        print(
            f"   {npc.base_npc.name}: {'✅ Consistent' if is_consistent else '❌ Inconsistent'}"
        )

    # Demonstrate dialogue generation
    print("\n💬 Dialogue Demo")
    print("-" * 40)

    sample_npc = generated_npcs[0]  # Use first NPC
    dialogue_context = DialogueContext(
        npc=sample_npc,
        player_input="Hello there!",
        conversation_history=[],
        interaction_type="casual",
    )

    print('Player: "Hello there!"')
    print(
        f"{sample_npc.base_npc.name}: \"{json.loads(MockOllamaClient().generate('', sample_npc.personality.archetype)['response'])['initial_dialogue']}\""
    )

    # Demonstrate archetype analysis
    print("\n📊 Archetype Analysis")
    print("-" * 40)

    archetype_counts = {}
    for npc in generated_npcs:
        archetype = npc.personality.archetype
        archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1

    print("Generated Archetypes:")
    for archetype, count in archetype_counts.items():
        archetype_def = theme_manager.get_archetype_definition(archetype)
        if archetype_def:
            print(f"   {archetype}: {count} - {archetype_def.description}")

    # Performance summary
    print("\n⚡ Performance Summary")
    print("-" * 40)
    print(f"   NPCs Generated: {len(generated_npcs)}")
    print(
        f"   Locations Covered: {len(set(npc.generation_metadata['location_theme'] for npc in generated_npcs))}"
    )
    print(f"   Unique Archetypes: {len(archetype_counts)}")
    print(
        f"   Average Traits per NPC: {sum(len(npc.personality.traits) for npc in generated_npcs) / len(generated_npcs):.1f}"
    )
    print("   Total Demo Time: < 1 second (mocked)")

    print("\n🎉 Demo Complete!")
    print("The NPC Generation System successfully created diverse, contextually")
    print("appropriate characters for different locations and purposes!")

    return generated_npcs


async def demonstrate_search_and_dialogue():
    """Demonstrate NPC search and dialogue capabilities."""

    print("\n🔍 Advanced Features Demo")
    print("=" * 50)

    # Generate some NPCs first
    npcs = await demonstrate_npc_generation()

    # Demonstrate search capabilities
    print("\n🔎 NPC Search Demo")
    print("-" * 30)

    print("Available NPCs:")
    for i, npc in enumerate(npcs):
        print(f"   {i+1}. {npc.base_npc.name} ({npc.personality.archetype})")
        print(f"      Location: {npc.generation_metadata['location_theme']}")
        print(f"      Traits: {', '.join(npc.personality.traits[:3])}")

    # Demonstrate relationship progression
    print("\n💖 Relationship Demo")
    print("-" * 30)

    merchant = npcs[0]  # Assume first NPC is merchant
    print(f"Simulating interactions with {merchant.base_npc.name}:")

    interactions = [
        ("positive", "Player helps NPC"),
        ("helpful", "Player provides useful information"),
        ("generous", "Player gives gift"),
    ]

    relationship_level = 0.0
    for interaction_type, description in interactions:
        # Simulate relationship changes
        change_map = {"positive": 0.1, "helpful": 0.15, "generous": 0.25}
        change = change_map.get(interaction_type, 0.0)
        relationship_level = min(1.0, relationship_level + change)

        print(f"   {description}: {relationship_level:.2f} (+{change:.2f})")

    print(f"   Final relationship: {relationship_level:.2f} (Friendly)")


if __name__ == "__main__":
    print("Starting NPC Generation Demo...")
    asyncio.run(demonstrate_npc_generation())
