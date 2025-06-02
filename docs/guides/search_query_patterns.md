# Common Search Queries and Usage Patterns

## Introduction

This document provides examples of common search query patterns, their optimal handling approaches, and implementation examples. These patterns represent typical ways that game systems and players interact with the semantic search functionality.

## Query Patterns by Usage Category

### 1. Entity Lookup Queries

These queries aim to find specific entities by name, attribute, or function.

#### Pattern: Direct Name Lookup

**Example Queries:**
- "Magic sword"
- "Healing potion"
- "Ancient ruins map"

**Optimal Strategy:** Hybrid (with keyword preference)

**Implementation:**

```python
async def lookup_by_name(name):
    results = await search_service.search(
        query=name,
        strategy="hybrid",
        semantic_weight=0.3,  # Favor keyword matching for names
        top_k=5,
        threshold=0.8
    )
    return results
```

#### Pattern: Functional Description Lookup

**Example Queries:**
- "Weapon that deals fire damage"
- "Item that helps with climbing"
- "Potion that cures poison"

**Optimal Strategy:** Semantic

**Implementation:**

```python
async def lookup_by_function(description):
    results = await search_service.search(
        query=description,
        strategy="semantic",
        entity_types=["item", "weapon", "consumable"],
        top_k=10,
        threshold=0.65  # Lower threshold for functional matches
    )
    return results
```

### 2. Contextual Discovery Queries

These queries aim to find entities that fit the current game context.

#### Pattern: Location-Based Discovery

**Example Queries:**
- "What can I find in the forest?"
- "What's available in this shop?"
- "What's hidden in this dungeon?"

**Optimal Strategy:** Hybrid with context filtering

**Implementation:**

```python
async def discover_in_location(location_id, query_text):
    # Get location data
    location = await location_service.get_location(location_id)

    # Create context-enhanced query
    context = {
        "location_type": location.type,
        "location_attributes": location.attributes,
        "location_id": location_id
    }

    # Search with context
    results = await search_service.search(
        query=query_text,
        strategy="hybrid",
        context=context,
        filter_criteria={"location_ids": [location_id]}
    )

    return results
```

#### Pattern: Situational Relevance

**Example Queries:**
- "What can help me in combat?"
- "What do I need for crafting?"
- "What would be useful for this puzzle?"

**Optimal Strategy:** Semantic with contextual weighting

**Implementation:**

```python
async def find_for_situation(situation_type, player_id):
    # Get player context
    player = await player_service.get_player(player_id)

    # Create situation-specific context
    context = {
        "situation": situation_type,
        "player_skills": player.skills,
        "player_inventory": player.inventory,
        "current_objective": player.current_objective
    }

    # Generate appropriate query based on situation
    if situation_type == "combat":
        query = "items weapons or abilities useful in combat"
    elif situation_type == "crafting":
        query = "materials ingredients or tools useful for crafting"
    elif situation_type == "puzzle":
        query = "items or knowledge useful for solving puzzles"
    else:
        query = f"items useful for {situation_type}"

    # Search with context
    results = await search_service.semantic_search(
        query=query,
        context=context,
        top_k=5
    )

    return results
```

### 3. Relational Queries

These queries aim to find relationships between entities.

#### Pattern: Similar Entity Search

**Example Queries:**
- "Show me items similar to Flame Sword"
- "What else is like this potion?"
- "Find similar monsters to Frost Giant"

**Optimal Strategy:** Similarity search

**Implementation:**

```python
async def find_similar_entities(entity_id):
    # Get entity data
    entity = await entity_service.get_entity(entity_id)

    # Find similar entities
    similar = await similarity_analyzer.find_similar_entities(
        entity_id=entity_id,
        top_k=10,
        min_similarity=0.7
    )

    return similar
```

#### Pattern: Complementary Entity Search

**Example Queries:**
- "What works well with this shield?"
- "What ingredients go with dragon scales?"
- "What should I pair with this spell?"

**Optimal Strategy:** Hybrid with relation filtering

**Implementation:**

```python
async def find_complementary_entities(entity_id):
    # Get entity data
    entity = await entity_service.get_entity(entity_id)

    # Create relation-specific query
    query = f"items that complement or work well with {entity.name}"

    # Search for complementary items
    results = await search_service.search(
        query=query,
        strategy="hybrid",
        context={"reference_entity": entity.to_dict()},
        filter_criteria={"relation_types": ["complements", "enhances"]}
    )

    return results
```

### 4. Knowledge Queries

These queries aim to find information rather than specific entities.

#### Pattern: Lore and Information Search

**Example Queries:**
- "Tell me about the history of the kingdom"
- "What is known about ancient dragons?"
- "Information about potion brewing"

**Optimal Strategy:** Semantic search across knowledge entities

**Implementation:**

```python
async def search_knowledge_base(query):
    results = await search_service.search(
        query=query,
        strategy="semantic",
        entity_types=["lore", "book", "scroll", "knowledge"],
        top_k=5
    )

    # Extract and format knowledge content
    knowledge = []
    for item in results:
        if "content" in item:
            knowledge.append({
                "title": item.get("title", "Unknown"),
                "content": item["content"],
                "relevance": item["score"]
            })

    return knowledge
```

#### Pattern: Game Mechanics Information

**Example Queries:**
- "How does crafting work?"
- "Rules for combat"
- "How to level up skills"

**Optimal Strategy:** Exact match with fallback to semantic

**Implementation:**

```python
async def get_game_mechanics_info(topic):
    # Try exact match first
    exact_results = await search_service.exact_match_search(
        query=topic,
        field="mechanic_name",
        entity_types=["game_rule", "tutorial", "help_topic"]
    )

    if exact_results:
        return exact_results

    # Fall back to semantic search
    semantic_results = await search_service.semantic_search(
        query=f"information about how {topic} works in the game",
        entity_types=["game_rule", "tutorial", "help_topic"],
        threshold=0.6
    )

    return semantic_results
```

### 5. Compound Queries

These queries combine multiple search intents.

#### Pattern: Multi-criteria Search

**Example Queries:**
- "Powerful weapons that deal ice damage"
- "Rare potions that boost strength"
- "Light armor with magic resistance"

**Optimal Strategy:** Combined semantic search with filtering

**Implementation:**

```python
async def multi_criteria_search(query):
    # Extract criteria from query
    query_analysis = await query_processor.analyze_query(query)

    # Build filter criteria
    filters = {}
    if "attributes" in query_analysis:
        filters["attributes"] = query_analysis["attributes"]

    if "rarity" in query_analysis:
        filters["rarity"] = query_analysis["rarity"]

    if "element" in query_analysis:
        filters["element"] = query_analysis["element"]

    # Perform search with extracted filters
    results = await search_service.search(
        query=query,
        strategy="semantic",
        filter_criteria=filters,
        top_k=10
    )

    return results
```

#### Pattern: Comparative Search

**Example Queries:**
- "Which is better, the fire sword or ice sword?"
- "Compare healing potions and bandages"
- "Strongest armor in the kingdom"

**Optimal Strategy:** Batch entity retrieval with comparison logic

**Implementation:**

```python
async def compare_entities(query):
    # Extract entities to compare
    comparison = await query_processor.extract_comparison(query)

    if comparison.get("type") == "direct":
        # Direct comparison between specific entities
        entity_ids = comparison.get("entity_ids", [])

        # Retrieve entities
        entities = []
        for entity_id in entity_ids:
            entity = await entity_service.get_entity(entity_id)
            entities.append(entity)

        # Perform comparison
        comparison_result = await comparison_service.compare(
            entities,
            aspect=comparison.get("aspect")
        )

        return comparison_result

    elif comparison.get("type") == "superlative":
        # Finding highest/lowest for a category
        category = comparison.get("category")
        aspect = comparison.get("aspect")
        direction = comparison.get("direction", "highest")

        # Search for entities in category
        category_entities = await search_service.search(
            query=category,
            strategy="keyword",
            top_k=50
        )

        # Sort by aspect
        sorted_entities = await sorting_service.sort_by_attribute(
            category_entities,
            attribute=aspect,
            direction=direction
        )

        return sorted_entities[:10]  # Return top 10
```

## Usage Patterns by Game System

### 1. Quest System Integration

#### Example: Dynamic Clue Generation

```python
async def generate_quest_clues(quest_objective):
    # Get objective data
    objective_entity = await quest_service.get_objective_entity(quest_objective.id)

    # Generate clues of different difficulty levels
    clues = []

    # Easy clue - more direct
    easy_clue_query = f"direct explicit clue about {objective_entity.name}"
    easy_clue = await search_service.semantic_search(
        query=easy_clue_query,
        entity_types=["knowledge", "clue"],
        context={"difficulty": "easy"}
    )
    if easy_clue:
        clues.append({"difficulty": "easy", "clue": easy_clue[0]})

    # Medium clue - less direct
    medium_clue_query = f"subtle hint about {objective_entity.name}"
    medium_clue = await search_service.semantic_search(
        query=medium_clue_query,
        entity_types=["knowledge", "clue"],
        context={"difficulty": "medium"}
    )
    if medium_clue:
        clues.append({"difficulty": "medium", "clue": medium_clue[0]})

    # Hard clue - cryptic
    hard_clue_query = f"cryptic reference to {objective_entity.name}"
    hard_clue = await search_service.semantic_search(
        query=hard_clue_query,
        entity_types=["knowledge", "clue"],
        context={"difficulty": "hard"}
    )
    if hard_clue:
        clues.append({"difficulty": "hard", "clue": hard_clue[0]})

    return clues
```

#### Example: Related Quest Discovery

```python
async def find_related_quests(current_quest_id):
    # Get current quest
    current_quest = await quest_service.get_quest(current_quest_id)

    # Find quests with similar themes or objectives
    related_quests = await search_service.search(
        query=current_quest.description,
        entity_types=["quest"],
        filter_criteria={"exclude_ids": [current_quest_id]},
        top_k=3
    )

    return related_quests
```

### 2. Dialogue System Integration

#### Example: Dynamic NPC Responses

```python
async def generate_npc_response(npc_id, player_query):
    # Get NPC data
    npc = await npc_service.get_npc(npc_id)

    # Search NPC's knowledge base
    knowledge_results = await search_service.search(
        query=player_query,
        context={
            "npc_knowledge": npc.knowledge,
            "npc_personality": npc.personality,
            "conversation_history": npc.conversation_history
        },
        filter_criteria={"knowledge_access": npc.knowledge_access}
    )

    # Generate response based on search results
    response = await dialogue_service.generate_response(
        npc=npc,
        player_query=player_query,
        knowledge=knowledge_results
    )

    return response
```

#### Example: Topic Detection

```python
async def detect_conversation_topic(dialogue_text):
    # Search for entities mentioned in dialogue
    mentioned_entities = await search_service.search(
        query=dialogue_text,
        strategy="keyword",
        threshold=0.8
    )

    # Classify the conversation topic
    topics = await topic_classifier.classify(
        text=dialogue_text,
        mentioned_entities=mentioned_entities
    )

    return topics
```

### 3. Crafting System Integration

#### Example: Recipe Discovery

```python
async def discover_recipes(inventory_items):
    # Convert inventory items to query
    item_names = [item.name for item in inventory_items]
    query = f"recipes using {', '.join(item_names)}"

    # Search for possible recipes
    recipes = await search_service.search(
        query=query,
        entity_types=["recipe"],
        context={"inventory_items": item_names}
    )

    # Filter for valid recipes (having all required items)
    valid_recipes = []
    inventory_item_ids = [item.id for item in inventory_items]

    for recipe in recipes:
        required_items = recipe.get("required_items", [])
        if all(item in inventory_item_ids for item in required_items):
            valid_recipes.append(recipe)

    return valid_recipes
```

#### Example: Finding Missing Ingredients

```python
async def find_missing_ingredients(recipe_id, inventory_items):
    # Get recipe
    recipe = await crafting_service.get_recipe(recipe_id)

    # Check what's missing
    inventory_item_ids = [item.id for item in inventory_items]
    missing_item_ids = [
        item_id for item_id in recipe.required_items
        if item_id not in inventory_item_ids
    ]

    # Get info on missing items
    missing_items = []
    for item_id in missing_item_ids:
        item = await entity_service.get_entity(item_id)

        # Find locations where this item is commonly found
        locations = await search_service.search(
            query=f"locations where {item.name} can be found",
            entity_types=["location"],
            top_k=3
        )

        missing_items.append({
            "item": item,
            "likely_locations": locations
        })

    return missing_items
```

### 4. Combat System Integration

#### Example: Weakness Analysis

```python
async def analyze_enemy_weaknesses(enemy_id):
    # Get enemy data
    enemy = await entity_service.get_entity(enemy_id)

    # Search for weakness information
    weakness_info = await search_service.search(
        query=f"weaknesses vulnerabilities or counters for {enemy.name}",
        entity_types=["knowledge", "combat_info"],
        top_k=3
    )

    # Search for recommended items against this enemy
    recommended_items = await search_service.search(
        query=f"effective weapons items or abilities against {enemy.name}",
        entity_types=["item", "weapon", "ability"],
        top_k=5
    )

    return {
        "weaknesses": weakness_info,
        "recommended_items": recommended_items
    }
```

#### Example: Tactical Suggestion

```python
async def generate_combat_advice(player_id, enemy_ids):
    # Get player and enemies
    player = await player_service.get_player(player_id)
    enemies = [await entity_service.get_entity(enemy_id) for enemy_id in enemy_ids]

    # Create context
    combat_context = {
        "player": {
            "level": player.level,
            "class": player.class_type,
            "abilities": player.abilities,
            "equipment": player.equipment,
            "statistics": player.statistics,
            "health_percentage": player.current_health / player.max_health
        },
        "enemies": [
            {
                "type": enemy.type,
                "level": enemy.level,
                "abilities": enemy.abilities,
                "statistics": enemy.statistics
            } for enemy in enemies
        ],
        "environment": player.current_location.environment_type
    }

    # Generate tactical query based on context
    query = "combat tactics and strategy advice for current situation"

    # Search for tactical advice
    advice = await search_service.semantic_search(
        query=query,
        entity_types=["combat_tactic", "knowledge"],
        context=combat_context,
        top_k=3
    )

    return advice
```

## Advanced Usage Patterns

### 1. Progressive Information Revelation

This pattern gradually reveals information based on player actions and discoveries.

```python
async def get_progressive_information(topic_id, player_knowledge_level):
    # Get basic information always available
    basic_info = await search_service.exact_match_search(
        query=topic_id,
        field="topic_id",
        entity_types=["knowledge"]
    )

    if player_knowledge_level < 2:
        # New player - only basic info
        return basic_info

    # Add intermediate information
    if player_knowledge_level >= 2:
        intermediate_info = await search_service.search(
            query=f"detailed information about {basic_info[0].name}",
            entity_types=["knowledge"],
            filter_criteria={"knowledge_level": "intermediate"},
            top_k=2
        )
        result = basic_info + intermediate_info

    # Add advanced information for experts
    if player_knowledge_level >= 4:
        advanced_info = await search_service.search(
            query=f"expert advanced information about {basic_info[0].name}",
            entity_types=["knowledge"],
            filter_criteria={"knowledge_level": "advanced"},
            top_k=3
        )
        result = result + advanced_info

    return result
```

### 2. Adaptive Difficulty

This pattern adjusts search quality based on game difficulty settings.

```python
async def puzzle_hint(puzzle_id, difficulty_setting):
    # Get puzzle
    puzzle = await puzzle_service.get_puzzle(puzzle_id)

    # Determine hint clarity based on difficulty
    if difficulty_setting == "easy":
        clarity = "clear direct hint"
        threshold = 0.5
    elif difficulty_setting == "normal":
        clarity = "subtle hint"
        threshold = 0.7
    else:  # hard
        clarity = "cryptic vague reference"
        threshold = 0.9

    # Search for appropriate hint
    hint = await search_service.search(
        query=f"{clarity} for solving {puzzle.name}",
        entity_types=["hint", "knowledge"],
        threshold=threshold,
        top_k=1
    )

    return hint
```

### 3. Contextual Tutorials

This pattern delivers tutorials and help based on player context.

```python
async def get_contextual_help(player_id, current_action):
    # Get player state
    player = await player_service.get_player(player_id)

    # Build context
    context = {
        "player_level": player.level,
        "action_history": player.recent_actions,
        "current_location": player.current_location.type,
        "current_action": current_action,
        "completed_tutorials": player.tutorial_progress
    }

    # Find relevant tutorials
    tutorials = await search_service.search(
        query=f"tutorial help or guide for {current_action}",
        entity_types=["tutorial", "help_topic"],
        context=context,
        filter_criteria={"exclude_ids": player.tutorial_progress}
    )

    # Sort by relevance to current context
    tutorials = await tutorial_service.sort_by_relevance(tutorials, context)

    return tutorials
```

## Conclusion

These examples demonstrate common search query patterns and their implementations across different game systems. They should be adapted to your specific game mechanics, entity structures, and player experience requirements.

When implementing these patterns:

1. Consider the balance between search accuracy and performance
2. Use appropriate search strategies for each query type
3. Leverage game context to enhance search relevance
4. Implement comprehensive error handling
5. Cache frequent queries for performance
6. Monitor search performance and optimize as needed

By understanding and implementing these patterns, you can create rich, context-aware gameplay experiences that leverage the full capabilities of the semantic search system.
