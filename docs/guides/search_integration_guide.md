# Search Integration Guide for Game Developers

## Introduction

This guide helps game developers integrate the semantic search system into their game mechanics, events, and systems. The search system provides powerful capabilities that can enhance gameplay through intelligent entity discovery, contextual awareness, and dynamic relationships between game elements.

## Integration Overview

The `SearchGameIntegrator` class serves as the primary interface for integrating search functionality with game systems. It provides methods for handling player search queries, generating contextual searches, finding related entities, and processing search-triggered events.

## Basic Integration Steps

1. **Initialize the Search Game Integrator:**

```python
from game_loop.search.game_integration import SearchGameIntegrator
from game_loop.search.semantic_search import SemanticSearchService

# Initialize with your existing services
search_integrator = SearchGameIntegrator(
    search_service=semantic_search_service,
    game_state_manager=game_state_manager
)
```

2. **Handle Player Search Queries:**

```python
async def process_player_search(player_id, query_text, game_context):
    # Get current context from player's game state
    player_context = game_state_manager.get_player_context(player_id)

    # Process the search with context
    search_results = await search_integrator.handle_player_search_query(
        query=query_text,
        context={
            "player_id": player_id,
            "location": player_context.current_location,
            "inventory": player_context.inventory,
            "game_state": game_context
        }
    )

    return search_results
```

3. **Generate Contextual Searches:**

```python
async def suggest_relevant_entities(game_context):
    # Generate search results based on current game context
    contextual_results = await search_integrator.generate_contextual_search(
        current_context=game_context
    )

    return contextual_results
```

4. **Find Related Entities:**

```python
async def find_related_items(item_id, relation_type="similar"):
    # Find entities related to a specific item
    related_results = await search_integrator.search_related_entities(
        entity_id=item_id,
        relation_type=relation_type  # "similar", "complementary", "opposing", etc.
    )

    return related_results
```

5. **Search Environment:**

```python
async def search_location(location_id, search_query=None):
    # Search for entities within a specific location
    environment_results = await search_integrator.search_environment(
        location_id=location_id,
        query=search_query
    )

    return environment_results
```

## Integrating with Game Events

The search system can be integrated with your game's event system to trigger searches or respond to search results:

```python
# Register search-triggered event handlers
game_event_system.register_handler(
    "item_discovery",
    search_integrator.handle_search_triggered_event
)
```

## Common Integration Patterns

### 1. Contextual Item Discovery

Reveal relevant items to players based on their current context:

```python
async def discover_contextual_items(player, location):
    context = {
        "player": player.to_dict(),
        "location": location.to_dict(),
        "quest_state": player.active_quests,
        "time_of_day": game_world.time_of_day
    }

    discovered_items = await search_integrator.generate_contextual_search(context)

    # Filter by relevance threshold
    relevant_items = [item for item in discovered_items if item["score"] > 0.75]

    return relevant_items
```

### 2. Knowledge-Based Puzzles

Create puzzles that require players to find related information:

```python
async def check_puzzle_solution(player_query, puzzle_context):
    # Process the player's search query
    search_result = await search_integrator.handle_player_search_query(
        query=player_query,
        context=puzzle_context
    )

    # Check if any results meet the solution criteria
    solution_found = any(
        result["entity_id"] == puzzle_context["solution_entity_id"]
        for result in search_result["results"]
    )

    return solution_found
```

### 3. Dynamic NPC Knowledge

Let NPCs provide information based on semantic search:

```python
async def npc_knowledge_response(npc, player_question):
    npc_context = {
        "npc_knowledge": npc.knowledge_base,
        "npc_personality": npc.personality,
        "player_relationship": npc.get_relationship(player.id)
    }

    # Find information the NPC would know about
    knowledge_results = await search_integrator.handle_player_search_query(
        query=player_question,
        context=npc_context
    )

    # Filter based on NPC knowledge level
    known_results = [r for r in knowledge_results["results"]
                    if r["entity_id"] in npc.known_entities]

    return known_results
```

## Best Practices

1. **Provide Rich Context:** Always include as much relevant context as possible when making search queries to improve relevance.

2. **Cache Common Searches:** Use the search caching system for frequently performed searches.

3. **Progressive Revelation:** Use search to progressively reveal information to players based on their actions and discoveries.

4. **Respect Performance:** Avoid making too many complex searches in performance-sensitive code paths.

5. **Use Appropriate Strategy:** Select the right search strategy based on your use case (see the Search Strategy Selection Guide).

6. **Threshold Tuning:** Experiment with similarity thresholds to find the right balance for your game mechanics.

7. **Handle Empty Results:** Always have fallback behaviors for when searches return no results.

## Advanced Integration

### Custom Search Event Handlers

You can create custom event handlers for search-triggered events:

```python
class CustomSearchHandler:
    def __init__(self, search_integrator):
        self.search_integrator = search_integrator

    async def handle_rare_item_discovery(self, search_result):
        if any(result["score"] > 0.9 and result["entity_type"] == "rare_item"
               for result in search_result["results"]):
            # Trigger special game event
            await game_event_system.trigger("rare_discovery", search_result)
```

### Search-Based Game Mechanics

Implement mechanics that leverage the search system's capabilities:

- **Memory Systems:** Track what players have searched for and use it to influence game behavior
- **Learning Mechanics:** Improve character abilities based on information discovered
- **Reputation Systems:** Track how players interact with information to influence NPC reactions
- **Adaptive Difficulty:** Use search patterns to adjust puzzle difficulty

## Troubleshooting

Common issues when integrating search with game systems:

1. **Irrelevant Results:** Add more specific context or increase the similarity threshold
2. **Poor Performance:** Check for too many concurrent searches or overly complex queries
3. **Missing Results:** Try different search strategies or lower the similarity threshold
4. **Inconsistent Behavior:** Ensure proper cache invalidation when game state changes

## Example: Complete Search Integration

Here's a complete example integrating search with a quest system:

```python
class QuestSearchIntegration:
    def __init__(self, search_integrator, quest_manager):
        self.search = search_integrator
        self.quests = quest_manager

    async def initialize(self):
        # Register event handlers
        self.quests.on_quest_accepted(self.handle_quest_accepted)
        self.quests.on_objective_complete(self.handle_objective_complete)

    async def handle_quest_accepted(self, player_id, quest_id):
        quest = self.quests.get_quest(quest_id)

        # Generate contextual hints based on quest
        context = {
            "quest_description": quest.description,
            "quest_objectives": quest.objectives,
            "player_id": player_id,
            "difficulty": quest.difficulty
        }

        # Generate relevant search results as hints
        hints = await self.search.generate_contextual_search(context)

        # Add hints to player's quest log
        self.quests.add_quest_hints(player_id, quest_id, hints)

    async def handle_objective_complete(self, player_id, quest_id, objective_id):
        # Get remaining objectives
        remaining = self.quests.get_remaining_objectives(quest_id)

        if remaining:
            next_objective = remaining[0]

            # Search for entities related to next objective
            related_entities = await self.search.search_related_entities(
                entity_id=next_objective.related_entity_id
            )

            # Update quest hints
            self.quests.update_quest_hints(player_id, quest_id, related_entities)
```

## Conclusion

Integrating the semantic search system with your game systems enables rich, contextual gameplay experiences. By following this guide, you can create dynamic interactions between your game world and the search capabilities to enhance player engagement and enable new types of gameplay mechanics.

Remember to test search integration thoroughly, as the relevance of search results can significantly impact the player experience.
