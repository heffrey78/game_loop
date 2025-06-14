#!/usr/bin/env python3
"""
Demo script for Object Generation System.

This demonstrates the complete object generation pipeline including context
collection, generation, placement, and storage without requiring external dependencies.
"""

import asyncio
import json
import time
from uuid import uuid4

from game_loop.core.world.object_context_collector import ObjectContextCollector
from game_loop.core.world.object_generator import ObjectGenerator
from game_loop.core.world.object_placement_manager import ObjectPlacementManager
from game_loop.core.world.object_theme_manager import ObjectThemeManager
from game_loop.state.models import Location, WorldState


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
    """Mock Ollama client that generates believable object data."""

    def generate(self, model: str, prompt: str, options: dict = None):
        """Generate mock object data based on prompt analysis."""

        # Analyze prompt to determine what kind of object to generate
        if "weapon" in prompt.lower():
            return self._generate_weapon_response()
        elif "tool" in prompt.lower():
            return self._generate_tool_response()
        elif "container" in prompt.lower():
            return self._generate_container_response()
        elif "natural" in prompt.lower():
            return self._generate_natural_response()
        elif "treasure" in prompt.lower():
            return self._generate_treasure_response()
        elif "book" in prompt.lower() or "knowledge" in prompt.lower():
            return self._generate_book_response()
        else:
            return self._generate_generic_response()

    def _generate_weapon_response(self):
        return {
            "response": json.dumps(
                {
                    "name": "Village Blacksmith's Hammer",
                    "description": "A well-balanced war hammer with a leather-wrapped handle, clearly forged by skilled hands. Small nicks along the head tell of countless battles.",
                    "material": "iron_and_leather",
                    "size": "medium",
                    "weight": "heavy",
                    "durability": "very_sturdy",
                    "value": 85,
                    "special_properties": ["balanced", "battle-tested", "intimidating"],
                    "cultural_significance": "local",
                }
            )
        }

    def _generate_tool_response(self):
        return {
            "response": json.dumps(
                {
                    "name": "Carpenter's Precise Chisel",
                    "description": "A finely crafted wood chisel with a sharp edge and comfortable wooden handle, worn smooth by years of use.",
                    "material": "steel_and_wood",
                    "size": "small",
                    "weight": "light",
                    "durability": "sturdy",
                    "value": 35,
                    "special_properties": ["precise", "well-maintained", "sharp"],
                    "cultural_significance": "artisan",
                }
            )
        }

    def _generate_container_response(self):
        return {
            "response": json.dumps(
                {
                    "name": "Reinforced Storage Chest",
                    "description": "A sturdy oak chest bound with iron bands, featuring a complex lock mechanism and weathered brass corners.",
                    "material": "oak_and_iron",
                    "size": "large",
                    "weight": "heavy",
                    "durability": "very_sturdy",
                    "value": 120,
                    "special_properties": ["lockable", "spacious", "weatherproof"],
                    "cultural_significance": "common",
                }
            )
        }

    def _generate_natural_response(self):
        return {
            "response": json.dumps(
                {
                    "name": "Silverleaf Healing Herb",
                    "description": "A delicate herb with silver-tipped leaves that shimmer in the light, known for its potent healing properties.",
                    "material": "plant",
                    "size": "tiny",
                    "weight": "light",
                    "durability": "fragile",
                    "value": 45,
                    "special_properties": ["medicinal", "rare", "potent"],
                    "cultural_significance": "sacred",
                }
            )
        }

    def _generate_treasure_response(self):
        return {
            "response": json.dumps(
                {
                    "name": "Polished Amber Gem",
                    "description": "A perfect sphere of golden amber containing a preserved ancient flower, warm to the touch and mesmerizing to behold.",
                    "material": "amber",
                    "size": "small",
                    "weight": "light",
                    "durability": "delicate",
                    "value": 250,
                    "special_properties": ["beautiful", "ancient", "magical_resonance"],
                    "cultural_significance": "legendary",
                }
            )
        }

    def _generate_book_response(self):
        return {
            "response": json.dumps(
                {
                    "name": "Tome of Local Histories",
                    "description": "A leather-bound collection of village records and folk tales, with yellowed pages filled with careful handwriting.",
                    "material": "parchment_and_leather",
                    "size": "medium",
                    "weight": "normal",
                    "durability": "delicate",
                    "value": 75,
                    "special_properties": ["informative", "historical", "fragile"],
                    "cultural_significance": "regional",
                }
            )
        }

    def _generate_generic_response(self):
        return {
            "response": json.dumps(
                {
                    "name": "Mysterious Object",
                    "description": "An object of unclear purpose, with strange markings that suggest it may have once held great importance.",
                    "material": "unknown",
                    "size": "medium",
                    "weight": "normal",
                    "durability": "sturdy",
                    "value": 50,
                    "special_properties": ["mysterious", "ancient"],
                    "cultural_significance": "unknown",
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
        description="A peaceful farming village with cobblestone streets and thatched-roof cottages.",
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
        description="An ancient forest where sunlight filters through thick canopy and wildlife thrives.",
        connections={"south": village_id},
        objects={},
        npcs={},
        state_flags={"theme": "Forest", "type": "wilderness", "danger_level": "low"},
    )

    # Create City location
    city_id = uuid4()
    city = Location(
        location_id=city_id,
        name="Goldport City",
        description="A bustling trade city with grand buildings and busy marketplaces.",
        connections={"west": village_id},
        objects={},
        npcs={},
        state_flags={"theme": "City", "type": "urban", "wealth": "prosperous"},
    )

    # Create Dungeon location
    dungeon_id = uuid4()
    dungeon = Location(
        location_id=dungeon_id,
        name="Forgotten Crypt",
        description="A dark underground chamber filled with ancient mysteries and hidden dangers.",
        connections={"up": forest_id},
        objects={},
        npcs={},
        state_flags={"theme": "Dungeon", "type": "underground", "danger_level": "high"},
    )

    world_state.locations[village_id] = village
    world_state.locations[forest_id] = forest
    world_state.locations[city_id] = city
    world_state.locations[dungeon_id] = dungeon

    return world_state, village_id, forest_id, city_id, dungeon_id


async def demonstrate_object_generation():
    """Demonstrate the complete object generation process."""

    print("🏗️  Object Generation System Demo")
    print("=" * 60)

    # Setup
    print("\n📋 Setting up demo environment...")
    world_state, village_id, forest_id, city_id, dungeon_id = (
        await create_sample_world()
    )

    session_factory = MockSessionFactory()
    llm_client = MockOllamaClient()
    embedding_manager = MockEmbeddingManager()

    # Initialize components
    theme_manager = ObjectThemeManager(world_state, session_factory)
    context_collector = ObjectContextCollector(world_state, session_factory)
    object_generator = ObjectGenerator(
        world_state, session_factory, llm_client, theme_manager, "templates"
    )
    placement_manager = ObjectPlacementManager(world_state, session_factory)

    # Demo different locations and purposes
    locations_to_demo = [
        (village_id, "Village", "populate_location"),
        (forest_id, "Forest", "quest_related"),
        (city_id, "City", "random_encounter"),
        (dungeon_id, "Dungeon", "narrative_enhancement"),
    ]

    generated_objects = []

    for location_id, theme_name, purpose in locations_to_demo:
        print(f"\n🏘️  Generating Object for {theme_name} ({purpose})")
        print("-" * 50)

        location = world_state.locations[location_id]

        # Step 1: Collect context
        print("📊 Collecting generation context...")
        start_time = time.time()
        context = await context_collector.collect_generation_context(
            location_id, purpose
        )
        context_time = (time.time() - start_time) * 1000

        print(f"   Location: {context.location.name}")
        print(f"   Theme: {context.location_theme.name}")
        print(f"   Purpose: {context.generation_purpose}")
        print(f"   Player Level: {context.player_level}")
        print(f"   Context Collection Time: {context_time:.2f}ms")

        # Step 2: Determine object type
        print("\n🎯 Determining appropriate object type...")
        start_time = time.time()
        object_type = await theme_manager.determine_object_type(context)
        type_time = (time.time() - start_time) * 1000
        print(f"   Selected Object Type: {object_type}")
        print(f"   Type Selection Time: {type_time:.2f}ms")

        # Step 3: Generate object
        print("\n🤖 Generating object with AI...")
        start_time = time.time()
        generated_object = await object_generator.generate_object(context)
        generation_time = (time.time() - start_time) * 1000

        print(f"   Object Name: {generated_object.properties.name}")
        print(f"   Material: {generated_object.properties.material}")
        print(f"   Value: {generated_object.properties.value} copper")
        print(
            f"   Special Properties: {', '.join(generated_object.properties.special_properties)}"
        )
        print(f"   Generation Time: {generation_time:.2f}ms")

        # Step 4: Determine placement
        print("\n📍 Determining object placement...")
        start_time = time.time()
        placement = await placement_manager.determine_placement(
            generated_object, location
        )
        placement_time = (time.time() - start_time) * 1000

        print(f"   Placement Type: {placement.placement_type}")
        print(f"   Visibility: {placement.visibility}")
        print(f"   Discovery Difficulty: {placement.discovery_difficulty}/10")
        print(f"   Spatial Description: {placement.spatial_description}")
        print(f"   Placement Time: {placement_time:.2f}ms")

        # Step 5: Validate object and placement
        print("\n🔍 Validating object and placement...")
        start_time = time.time()

        # Validate object consistency
        object_consistent = await theme_manager.validate_object_consistency(
            generated_object, location
        )

        # Validate placement
        placement_valid = await placement_manager.validate_placement(
            placement, location
        )

        # Validate generation result
        generation_validation = await object_generator.validate_generated_object(
            generated_object, context
        )

        validation_time = (time.time() - start_time) * 1000

        print(
            f"   Object Consistency: {'✅ Valid' if object_consistent else '❌ Invalid'}"
        )
        print(f"   Placement Valid: {'✅ Valid' if placement_valid else '❌ Invalid'}")
        print(
            f"   Generation Quality: {generation_validation.consistency_score:.2f}/1.0"
        )
        print(f"   Validation Time: {validation_time:.2f}ms")

        # Step 6: Generate embedding
        print("\n🔗 Generating object embeddings...")
        start_time = time.time()
        embedding_text = f"{generated_object.properties.name} {generated_object.properties.object_type} {' '.join(generated_object.properties.special_properties)}"
        embedding = await embedding_manager.generate_embedding(embedding_text)
        generated_object.embedding_vector = embedding[:5]  # Show first 5 dimensions
        embedding_time = (time.time() - start_time) * 1000

        print(f"   Embedding Dimensions: {len(embedding)}")
        print(f"   Sample Values: [{', '.join(f'{x:.3f}' for x in embedding[:5])}...]")
        print(f"   Embedding Time: {embedding_time:.2f}ms")

        # Add placement to object
        generated_object.placement_info = placement
        generated_objects.append(generated_object)

        # Add to location for density tracking
        location.objects[generated_object.base_object.object_id] = (
            generated_object.base_object
        )

        # Display complete object
        print("\n✨ Generated Complete Object:")
        print(f"   🏷️  {generated_object.properties.name}")
        print(f"   📝 {generated_object.properties.description}")
        print(f"   🎭 Type: {generated_object.properties.object_type}")
        print(
            f"   🛠️  Actions: {', '.join(generated_object.interactions.available_actions[:4])}"
        )
        print(f"   💰 Value: {generated_object.properties.value} copper")
        print(f"   📍 Placed: {placement.spatial_description}")

    # Demonstrate object analysis
    print("\n📊 Object Generation Analysis")
    print("-" * 40)

    # Type distribution
    type_counts = {}
    total_value = 0
    for obj in generated_objects:
        obj_type = obj.properties.object_type
        type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
        total_value += obj.properties.value

    print("Object Type Distribution:")
    for obj_type, count in type_counts.items():
        print(f"   {obj_type}: {count}")

    print("\nValue Statistics:")
    print(f"   Total Value: {total_value} copper")
    print(f"   Average Value: {total_value / len(generated_objects):.1f} copper")

    # Rarity distribution
    rarity_counts = {
        "common": 0,
        "local": 0,
        "regional": 0,
        "sacred": 0,
        "legendary": 0,
    }
    for obj in generated_objects:
        significance = obj.properties.cultural_significance
        if significance in rarity_counts:
            rarity_counts[significance] += 1

    print("\nCultural Significance:")
    for rarity, count in rarity_counts.items():
        if count > 0:
            print(f"   {rarity}: {count}")

    # Placement analysis
    print("\nPlacement Analysis:")
    placement_types = {}
    visibility_counts = {}

    for obj in generated_objects:
        if obj.placement_info:
            placement_type = obj.placement_info.placement_type
            visibility = obj.placement_info.visibility

            placement_types[placement_type] = placement_types.get(placement_type, 0) + 1
            visibility_counts[visibility] = visibility_counts.get(visibility, 0) + 1

    print(f"   Placement Types: {dict(placement_types)}")
    print(f"   Visibility: {dict(visibility_counts)}")

    # Location density check
    print("\n🏘️  Location Density Analysis")
    print("-" * 40)

    for location_id in [village_id, forest_id, city_id, dungeon_id]:
        location = world_state.locations[location_id]
        density_info = await placement_manager.check_placement_density(location)

        print(f"{location.name}:")
        print(
            f"   Objects: {density_info['current_count']}/{density_info['max_recommended']}"
        )
        print(
            f"   Density: {density_info['density_ratio']:.2f} ({'Sparse' if density_info['sparse'] else 'Normal' if not density_info['overcrowded'] else 'Crowded'})"
        )
        print(
            f"   Can Accommodate More: {'Yes' if density_info['can_accommodate'] else 'No'}"
        )

    # Demonstrate theme consistency
    print("\n🎨 Theme Consistency Demo")
    print("-" * 40)

    # Get available object types for each theme
    themes = ["Village", "Forest", "City", "Dungeon"]
    for theme in themes:
        available_types = await theme_manager.get_available_object_types(theme)
        theme_def = theme_manager.get_theme_definition(theme)

        print(f"{theme} Theme:")
        print(f"   Available Types: {', '.join(available_types[:4])}")
        if theme_def:
            print(f"   Materials: {', '.join(theme_def.typical_materials[:3])}")
            print(f"   Style: {', '.join(theme_def.style_descriptors[:3])}")

    # Demonstrate archetype system
    print("\n🏺 Archetype System Demo")
    print("-" * 40)

    archetypes = ["sword", "hammer", "chest", "book", "herb", "gem"]
    for archetype_name in archetypes:
        archetype = theme_manager.get_archetype_definition(archetype_name)
        if archetype:
            print(f"{archetype_name.upper()}:")
            print(f"   Description: {archetype.description}")
            print(f"   Object Type: {archetype.typical_properties.object_type}")
            print(f"   Rarity: {archetype.rarity}")
            # Show best location affinity
            if archetype.location_affinities:
                best_location = max(
                    archetype.location_affinities.items(), key=lambda x: x[1]
                )
                print(
                    f"   Best Location: {best_location[0]} ({best_location[1]:.1f} affinity)"
                )

    # Performance summary
    print("\n⚡ Performance Summary")
    print("-" * 40)
    print(f"   Objects Generated: {len(generated_objects)}")
    print(
        f"   Locations Covered: {len(set(obj.generation_metadata.get('location_theme', 'Unknown') for obj in generated_objects))}"
    )
    print(f"   Unique Object Types: {len(type_counts)}")
    print(
        f"   Average Special Properties: {sum(len(obj.properties.special_properties) for obj in generated_objects) / len(generated_objects):.1f}"
    )
    print("   Total Demo Time: < 1 second (mocked LLM)")

    print("\n🎉 Demo Complete!")
    print("The Object Generation System successfully created diverse, contextually")
    print("appropriate objects for different locations and purposes!")

    return generated_objects


async def demonstrate_search_and_interactions():
    """Demonstrate object search and interaction capabilities."""

    print("\n🔍 Advanced Object Features Demo")
    print("=" * 60)

    # Generate some objects first
    objects = await demonstrate_object_generation()

    # Demonstrate interaction complexity
    print("\n💫 Object Interaction Complexity")
    print("-" * 40)

    for obj in objects[:3]:  # Show first 3
        print(f"\n{obj.properties.name}:")
        print(f"   Basic Actions: {', '.join(obj.interactions.available_actions[:3])}")

        if obj.interactions.use_requirements:
            print(f"   Requirements: {list(obj.interactions.use_requirements.keys())}")

        if obj.interactions.interaction_results:
            result_keys = list(obj.interactions.interaction_results.keys())[:2]
            for key in result_keys:
                print(f"   {key}: {obj.interactions.interaction_results[key][:50]}...")

        print(f"   Portable: {'Yes' if obj.interactions.portable else 'No'}")
        print(f"   Consumable: {'Yes' if obj.interactions.consumable else 'No'}")

    # Demonstrate cultural significance
    print("\n🌍 Cultural Significance Analysis")
    print("-" * 40)

    significance_descriptions = {
        "common": "Everyday items found in most settlements",
        "local": "Items with local importance or craftsmanship",
        "regional": "Items known throughout the region",
        "sacred": "Items with religious or spiritual significance",
        "legendary": "Items of great renown and power",
    }

    for obj in objects:
        significance = obj.properties.cultural_significance
        description = significance_descriptions.get(
            significance, "Unknown significance"
        )
        print(f"   {obj.properties.name}: {significance} - {description}")

    # Demonstrate placement intelligence
    print("\n📍 Intelligent Placement Analysis")
    print("-" * 40)

    for obj in objects:
        if obj.placement_info:
            print(f"\n{obj.properties.name}:")
            print(f"   Placement: {obj.placement_info.placement_type} placement")
            print(f"   Visibility: {obj.placement_info.visibility}")
            print(f"   Accessibility: {obj.placement_info.accessibility}")
            print(
                f"   Discovery: {obj.placement_info.discovery_difficulty}/10 difficulty"
            )
            print(f"   Location: {obj.placement_info.spatial_description}")

    print("\n✨ Advanced Features Complete!")


if __name__ == "__main__":
    print("Starting Object Generation Demo...")
    asyncio.run(demonstrate_object_generation())
    print("\nStarting Advanced Features Demo...")
    asyncio.run(demonstrate_search_and_interactions())
