"""
Game integration module for semantic search.

This module provides integration between semantic search capabilities
and game systems to enable search-based gameplay mechanics.
"""

import logging
import time
from collections.abc import Callable
from typing import Any

from game_loop.search.semantic_search import SemanticSearchService
from game_loop.search.similarity import EntitySimilarityAnalyzer
from game_loop.state.manager import GameStateManager

logger = logging.getLogger(__name__)


class SearchGameIntegrator:
    """Integrate search functionality with game systems."""

    def __init__(
        self,
        search_service: SemanticSearchService,
        game_state_manager: GameStateManager,
    ) -> None:
        """
        Initialize the search game integrator.

        Args:
            search_service: Semantic search service
            game_state_manager: Game state manager
        """
        self.search_service = search_service
        self.game_state_manager = game_state_manager
        self._event_handlers: dict[str, Callable[..., Any]] = {}
        self._query_history: dict[str, list[dict[str, Any]]] = (
            {}
        )  # Map player_id -> [recent queries]
        self._search_callbacks: dict[str, Callable[..., Any]] = (
            {}
        )  # Map event_type -> callback
        self._register_event_handlers()

    async def handle_player_search_query(
        self, query: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Process a player's search query within game context.

        Args:
            query: Player's search query
            context: Additional context like player location, current quest, etc.

        Returns:
            Search results with game context integration
        """
        try:
            # Extract relevant context
            player_id = context.get("player_id") if context else None
            context.get("location_id") if context else None
            context.get("current_quest") if context else None

            # Record query in history
            if player_id:
                self._record_player_query(player_id, query)

            # Determine entity types to search based on context
            entity_types = self._determine_entity_types_from_context(context)

            # Perform search with appropriate strategy
            results = await self.search_service.search(
                query=query,
                entity_types=entity_types or [],
                strategy="hybrid",
                top_k=10,
            )

            # Apply game context filters
            if context:
                results = self._filter_results_by_game_context(results, context)

            # Enhance results with game-specific information
            enhanced_results = await self._enhance_results_with_game_data(
                results, context
            )

            # Trigger any relevant game events based on search
            await self._trigger_search_events(query, enhanced_results, context)

            return {
                "results": enhanced_results,
                "context_applied": bool(context),
                "game_interaction": self._generate_game_interactions(
                    query, enhanced_results, context
                ),
            }

        except Exception as e:
            logger.error(f"Error in handle_player_search_query: {e}")
            return {"error": f"Search processing error: {str(e)}", "results": []}

    async def generate_contextual_search(
        self, current_context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Generate search results based on current game context.

        Args:
            current_context: Current game context

        Returns:
            Contextually relevant search results
        """
        try:
            # Extract key context elements
            location = current_context.get("location")
            quest = current_context.get("current_quest")
            recent_interactions = current_context.get("recent_interactions", [])
            current_context.get("player_state", {})

            # Build implicit search query from context
            query_elements = []

            if location and "name" in location:
                query_elements.append(location["name"])

            if location and "description" in location:
                # Extract key phrases from description
                query_elements.append(
                    self._extract_key_phrases(location["description"], 2)
                )

            if quest and "objective" in quest:
                query_elements.append(quest["objective"])

            if recent_interactions:
                # Get most recent interaction
                last_interaction = recent_interactions[-1]
                if "entity_id" in last_interaction:
                    # Find related entities instead of text search
                    similar_entities = await self.search_related_entities(
                        last_interaction["entity_id"]
                    )

                    # These are already in result format
                    return similar_entities

            # If we have query elements, perform search
            if query_elements:
                implicit_query = " ".join(query_elements)

                # Perform search with context-appropriate strategy
                results = await self.search_service.search(
                    query=implicit_query,
                    strategy="semantic",  # Semantic better for implicit queries
                    top_k=5,
                    threshold=0.6,  # Lower threshold for contextual search
                )

                # Filter and rank by relevance to current context
                return self._rank_by_context_relevance(results, current_context)

            # Fallback to empty results
            return []

        except Exception as e:
            logger.error(f"Error in generate_contextual_search: {e}")
            return []

    async def search_related_entities(
        self, entity_id: str, relation_type: str = "any"
    ) -> list[dict[str, Any]]:
        """
        Find entities related to a specific entity by relationship type.

        Args:
            entity_id: Reference entity ID
            relation_type: Type of relationship to find

        Returns:
            List of related entities
        """
        try:
            related_entities = []

            # Use similarity for 'similar' relation type
            if relation_type in ["similar", "any"]:
                similarity_analyzer = EntitySimilarityAnalyzer(
                    self.search_service.registry, self.search_service.db_manager
                )

                similar_entities = await similarity_analyzer.find_similar_entities(
                    entity_id=entity_id, top_k=5, min_similarity=0.7
                )

                # Convert to results format
                for similar_id, score in similar_entities:
                    # Get entity data
                    entity_data = self._get_entity_data(similar_id)

                    related_entities.append(
                        {
                            "entity_id": similar_id,
                            "relation_type": "similar",
                            "score": score,
                            "data": entity_data or {"entity_id": similar_id},
                        }
                    )

            # Use explicit relationships for other relation types
            if relation_type in ["connected", "interacts_with", "contains", "any"]:
                # This would query your game's relationship system
                # For now, we'll use a placeholder implementation
                connected_entities = await self._get_connected_entities(entity_id)

                related_entities.extend(connected_entities)

            return related_entities

        except Exception as e:
            logger.error(f"Error in search_related_entities: {e}")
            return []

    async def search_environment(
        self, location_id: str, query: str | None = None
    ) -> list[dict[str, Any]]:
        """
        Search for entities within a specific game location.

        Args:
            location_id: Location ID to search within
            query: Optional search query to filter results

        Returns:
            Entities in the location matching query
        """
        try:
            # Get location data
            location_data = self._get_location_data(location_id)
            if not location_data:
                logger.warning(f"Location not found: {location_id}")
                return []

            # Get entities in this location
            entities_in_location = await self._get_entities_in_location(location_id)

            # If no query, return all entities in location
            if not query:
                return entities_in_location

            # With query, filter entities by similarity to query
            entity_ids = [e["entity_id"] for e in entities_in_location]

            # Perform search restricted to these entities
            search_results = await self.search_service.search(
                query=query,
                entity_types=[],  # Don't restrict types
                strategy="hybrid",
                top_k=len(entity_ids),  # Get all matches
            )

            # Filter to only include entities in this location
            filtered_results = [
                result for result in search_results if result["entity_id"] in entity_ids
            ]

            return filtered_results

        except Exception as e:
            logger.error(f"Error in search_environment: {e}")
            return []

    async def handle_search_triggered_event(
        self, search_result: dict[str, Any], event_type: str
    ) -> dict[str, Any]:
        """
        Handle game events triggered by search results.

        Args:
            search_result: Search result that triggered the event
            event_type: Type of event triggered

        Returns:
            Event handling result
        """
        try:
            # Check if we have a handler for this event type
            if event_type in self._search_callbacks:
                handler = self._search_callbacks[event_type]
                return await handler(search_result)  # type: ignore[no-any-return]

            # Default handling for common event types
            if event_type == "entity_discovered":
                return await self._handle_entity_discovered(search_result)

            elif event_type == "quest_progressed":
                return await self._handle_quest_progressed(search_result)

            else:
                logger.warning(f"No handler for search event type: {event_type}")
                return {"status": "unhandled", "event_type": event_type}

        except Exception as e:
            logger.error(f"Error in handle_search_triggered_event: {e}")
            return {"error": str(e)}

    def register_search_callback(self, event_type: str, callback: Callable) -> None:
        """
        Register a callback for a search-related event.

        Args:
            event_type: Event type to register for
            callback: Async callback function
        """
        self._search_callbacks[event_type] = callback
        logger.debug(f"Registered callback for search event: {event_type}")

    def _register_event_handlers(self) -> None:
        """Register handlers for search-related game events."""
        # This would integrate with your game's event system
        # For now, we'll just set up the handler dictionary
        self._event_handlers = {
            "player_search": self.handle_player_search_query,
            "contextual_search": self.generate_contextual_search,
            "entity_relation_search": self.search_related_entities,
        }

        logger.info("Registered search event handlers")

    def _extract_search_context(self, game_state: dict[str, Any]) -> dict[str, Any]:
        """
        Extract relevant search context from current game state.

        Args:
            game_state: Current game state

        Returns:
            Extracted search context
        """
        context = {}

        # Extract player info
        if "player" in game_state:
            context["player_id"] = game_state["player"].get("id")
            context["player_state"] = {
                "level": game_state["player"].get("level"),
                "skills": game_state["player"].get("skills", {}),
            }

        # Extract location info
        if "location" in game_state:
            context["location_id"] = game_state["location"].get("id")
            context["location_name"] = game_state["location"].get("name")

        # Extract quest info
        if "quests" in game_state and "active_quests" in game_state["quests"]:
            active_quests = game_state["quests"]["active_quests"]
            if active_quests:
                context["current_quest"] = active_quests[0]

        # Extract recent interactions
        if "interactions" in game_state:
            recent = game_state["interactions"].get("recent", [])
            context["recent_interactions"] = recent[-5:]  # Last 5 interactions

        return context

    def _apply_search_results_to_game_state(
        self, results: list[dict[str, Any]], game_state: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Update game state based on search results.

        Args:
            results: Search results
            game_state: Current game state

        Returns:
            Updated game state
        """
        # This is a placeholder implementation
        # In a real game, this would update the game state based on search results

        # Create a copy to avoid modifying the original
        updated_state = dict(game_state)

        # Update discovered entities
        if "discovered_entities" not in updated_state:
            updated_state["discovered_entities"] = {}

        for result in results:
            entity_id = result.get("entity_id")
            if entity_id:
                updated_state["discovered_entities"][entity_id] = {
                    "discovered_at": time.time(),
                    "relevance": result.get("score", 0.0),
                }

        # Add to recent searches
        if (
            "player" in updated_state
            and "recent_searches" not in updated_state["player"]
        ):
            updated_state["player"]["recent_searches"] = []

        if "player" in updated_state and len(results) > 0:
            updated_state["player"]["recent_searches"].append(
                {
                    "timestamp": time.time(),
                    "results": [r["entity_id"] for r in results[:3]],
                }
            )

        return updated_state

    def _record_player_query(self, player_id: str, query: str) -> None:
        """
        Record a player's search query in history.

        Args:
            player_id: Player ID
            query: Search query
        """
        if player_id not in self._query_history:
            self._query_history[player_id] = []

        # Add to history with timestamp
        self._query_history[player_id].append(
            {"query": query, "timestamp": time.time()}
        )

        # Keep history bounded
        if len(self._query_history[player_id]) > 20:
            self._query_history[player_id] = self._query_history[player_id][-20:]

    def _determine_entity_types_from_context(
        self, context: dict[str, Any] | None
    ) -> list[str] | None:
        """
        Determine which entity types to search based on context.

        Args:
            context: Game context

        Returns:
            List of entity types to search or None for all types
        """
        if not context:
            return None

        entity_types = []

        # Check location for relevant entity types
        location_id = context.get("location_id")
        if location_id:
            location_data = self._get_location_data(location_id)
            if location_data and "primary_entity_types" in location_data:
                entity_types.extend(location_data["primary_entity_types"])

        # Check quest for relevant entity types
        quest = context.get("current_quest")
        if quest and "relevant_entity_types" in quest:
            entity_types.extend(quest["relevant_entity_types"])

        # Ensure uniqueness
        return list(set(entity_types)) if entity_types else None

    def _filter_results_by_game_context(
        self, results: list[dict[str, Any]], context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Filter search results based on game context.

        Args:
            results: Search results
            context: Game context

        Returns:
            Filtered search results
        """
        filtered_results = []

        # Get location ID from context
        location_id = context.get("location_id")
        player_level = context.get("player_state", {}).get("level", 1)

        for result in results:
            # Filter by location if applicable
            if location_id and "available_locations" in result.get("data", {}):
                if location_id not in result["data"]["available_locations"]:
                    continue

            # Filter by player level if applicable
            if (
                "min_level" in result.get("data", {})
                and player_level < result["data"]["min_level"]
            ):
                continue

            filtered_results.append(result)

        return filtered_results

    async def _enhance_results_with_game_data(
        self, results: list[dict[str, Any]], context: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """
        Enhance search results with game-specific information.

        Args:
            results: Search results
            context: Game context

        Returns:
            Enhanced search results
        """
        enhanced_results = []

        for result in results:
            # Create a copy to avoid modifying the original
            enhanced = dict(result)

            # Ensure data field exists
            if "data" not in enhanced:
                enhanced["data"] = {}

            # Add game-specific information
            entity_id = enhanced.get("entity_id")

            if entity_id:
                # Add discovery status
                enhanced["data"]["discovered"] = self._is_entity_discovered(
                    entity_id, context
                )

                # Add relevance to current quest if applicable
                if context and "current_quest" in context:
                    quest_relevance = await self._calculate_quest_relevance(
                        entity_id, context["current_quest"]
                    )
                    enhanced["data"]["quest_relevance"] = quest_relevance

                # Add interaction options if applicable
                interaction_options = self._get_interaction_options(entity_id, context)
                if interaction_options:
                    enhanced["data"]["interactions"] = interaction_options

            enhanced_results.append(enhanced)

        return enhanced_results

    def _generate_game_interactions(
        self,
        query: str,
        results: list[dict[str, Any]],
        context: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Generate possible game interactions based on search results.

        Args:
            query: Search query
            results: Search results
            context: Game context

        Returns:
            Dictionary of possible game interactions
        """
        interactions = {"can_interact": False, "actions": []}

        # Check if we have any results
        if not results:
            return interactions

        # Check if top result is interactable
        if results and "data" in results[0]:
            top_result = results[0]
            entity_id = top_result.get("entity_id")

            if entity_id:
                # Check if discovered
                discovered = self._is_entity_discovered(entity_id, context)

                # Get possible interactions
                possible_actions = []

                # Entity type specific actions
                entity_type = top_result.get("entity_type", "")

                if entity_type == "item":
                    possible_actions.extend(["examine", "pickup", "use"])
                elif entity_type == "character":
                    possible_actions.extend(["talk", "trade", "gift"])
                elif entity_type == "location":
                    possible_actions.extend(["travel", "explore", "search"])

                if discovered:
                    interactions["can_interact"] = True
                    interactions["actions"] = possible_actions
                else:
                    # If not discovered, only allow discovery
                    interactions["can_interact"] = True
                    interactions["actions"] = ["discover"]

        return interactions

    async def _trigger_search_events(
        self,
        query: str,
        results: list[dict[str, Any]],
        context: dict[str, Any] | None,
    ) -> None:
        """
        Trigger any relevant game events based on search results.

        Args:
            query: Search query
            results: Search results
            context: Game context
        """
        # Check for discovery events
        if results:
            for result in results:
                entity_id = result.get("entity_id")

                if entity_id and not self._is_entity_discovered(entity_id, context):
                    # Trigger discovery event
                    await self.handle_search_triggered_event(
                        result, "entity_discovered"
                    )

        # Check for quest progression
        if context and "current_quest" in context and results:
            quest = context["current_quest"]

            # Check if any result is relevant to quest
            for result in results:
                entity_id = result.get("entity_id")

                if entity_id:
                    relevance = await self._calculate_quest_relevance(entity_id, quest)

                    if relevance > 0.8:  # High relevance threshold
                        # Trigger quest progression event
                        await self.handle_search_triggered_event(
                            {**result, "quest": quest}, "quest_progressed"
                        )

    async def _handle_entity_discovered(
        self, search_result: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle entity discovery event.

        Args:
            search_result: Search result that triggered discovery

        Returns:
            Event handling result
        """
        # This would integrate with your game's entity discovery system
        # For now, we'll provide a placeholder implementation
        entity_id = search_result.get("entity_id")

        if not entity_id:
            return {"status": "failed", "reason": "No entity ID"}

        return {
            "status": "discovered",
            "entity_id": entity_id,
            "entity_type": search_result.get("entity_type", "unknown"),
            "discovery_message": "Discovered new "
            f"{search_result.get('entity_type', 'entity')}!",
        }

    async def _handle_quest_progressed(
        self, search_result: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle quest progression event.

        Args:
            search_result: Search result that triggered quest progression

        Returns:
            Event handling result
        """
        # This would integrate with your game's quest system
        # For now, we'll provide a placeholder implementation
        entity_id = search_result.get("entity_id")
        quest = search_result.get("quest", {})

        if not entity_id or not quest:
            return {"status": "failed", "reason": "Missing entity ID or quest info"}

        return {
            "status": "progressed",
            "quest_id": quest.get("id", "unknown"),
            "entity_id": entity_id,
            "progress_message": "Made progress on your quest!",
        }

    def _is_entity_discovered(
        self, entity_id: str, context: dict[str, Any] | None
    ) -> bool:
        """
        Check if an entity has been discovered by the player.

        Args:
            entity_id: Entity ID
            context: Game context

        Returns:
            True if entity is discovered
        """
        # This would integrate with your game's discovery tracking system
        # For now, we'll provide a placeholder implementation
        if not context:
            return True  # Default to discovered if no context

        # Check discovered entities
        if "game_state" in context and "discovered_entities" in context["game_state"]:
            return entity_id in context["game_state"]["discovered_entities"]

        return False

    async def _calculate_quest_relevance(
        self, entity_id: str, quest: dict[str, Any]
    ) -> float:
        """
        Calculate how relevant an entity is to a quest.

        Args:
            entity_id: Entity ID
            quest: Quest data

        Returns:
            Relevance score (0-1)
        """
        # This would integrate with your game's quest system
        # For now, we'll provide a placeholder implementation

        # Get entity data
        entity_data = self._get_entity_data(entity_id)
        if not entity_data or not quest:
            return 0.0

        # Check if entity is explicitly mentioned in quest
        if "required_entities" in quest and entity_id in quest["required_entities"]:
            return 1.0

        # Check if entity type is relevant to quest
        if (
            "relevant_entity_types" in quest
            and entity_data.get("entity_type") in quest["relevant_entity_types"]
        ):
            return 0.8

        # Calculate semantic similarity between entity and quest objective
        if "objective" in quest and (
            "name" in entity_data or "description" in entity_data
        ):
            query = quest["objective"]

            # Use search service to calculate similarity
            entity_type = entity_data.get("entity_type")
            entity_types = [entity_type] if entity_type is not None else []
            results = await self.search_service.search(
                query=query,
                entity_types=entity_types,
                strategy="semantic",
                top_k=1,
            )

            # Check if our entity is in results
            for result in results:
                if result.get("entity_id") == entity_id:
                    return float(result.get("score", 0.0))

        return 0.0

    def _get_interaction_options(
        self, entity_id: str, context: dict[str, Any] | None
    ) -> list[dict[str, Any]]:
        """
        Get available interaction options for an entity.

        Args:
            entity_id: Entity ID
            context: Game context

        Returns:
            List of interaction options
        """
        # This would integrate with your game's interaction system
        # For now, we'll provide a placeholder implementation
        entity_data = self._get_entity_data(entity_id)

        if not entity_data:
            return []

        entity_type = entity_data.get("entity_type", "unknown")
        interactions = []

        # Generate interactions based on entity type
        if entity_type == "item":
            interactions.append(
                {
                    "type": "examine",
                    "label": "Examine Item",
                    "action": "examine",
                    "params": {"entity_id": entity_id},
                }
            )

            if self._is_entity_discovered(entity_id, context):
                interactions.append(
                    {
                        "type": "use",
                        "label": "Use Item",
                        "action": "use",
                        "params": {"entity_id": entity_id},
                    }
                )

        elif entity_type == "character":
            interactions.append(
                {
                    "type": "talk",
                    "label": "Talk to Character",
                    "action": "talk",
                    "params": {"entity_id": entity_id},
                }
            )

        elif entity_type == "location":
            interactions.append(
                {
                    "type": "travel",
                    "label": "Travel to Location",
                    "action": "travel",
                    "params": {"entity_id": entity_id},
                }
            )

        return interactions

    def _extract_key_phrases(self, text: str, max_phrases: int = 2) -> str:
        """
        Extract key phrases from text.

        Args:
            text: Source text
            max_phrases: Maximum number of phrases to extract

        Returns:
            Extracted key phrases
        """
        # This is a very simple implementation
        # In a real system, you might use NLP techniques

        # Split into sentences
        sentences = text.split(".")

        # Take first few sentences
        key_sentences = sentences[:max_phrases]

        # Join and return
        return " ".join(s.strip() for s in key_sentences if s.strip())

    def _rank_by_context_relevance(
        self, results: list[dict[str, Any]], context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        Rank search results by relevance to current context.

        Args:
            results: Search results
            context: Game context

        Returns:
            Results ranked by context relevance
        """
        # Assign context relevance scores
        for result in results:
            context_score = 0.0
            entity_data = result.get("data", {})

            # Relevance to location
            if "location_id" in context and "available_locations" in entity_data:
                if context["location_id"] in entity_data["available_locations"]:
                    context_score += 0.3

            # Relevance to current quest
            if "current_quest" in context and "quest_relevance" in entity_data:
                context_score += entity_data["quest_relevance"] * 0.4

            # Factor in search relevance
            base_score = result.get("score", 0.0)

            # Combine scores (60% search relevance, 40% context relevance)
            result["context_score"] = context_score
            result["combined_score"] = (base_score * 0.6) + (context_score * 0.4)

        # Sort by combined score
        return sorted(results, key=lambda x: x.get("combined_score", 0.0), reverse=True)

    def _get_entity_data(self, entity_id: str) -> dict[str, Any] | None:
        """
        Get entity data.

        Args:
            entity_id: Entity ID

        Returns:
            Entity data or None if not found
        """
        # Try to get from registry
        if hasattr(self.search_service.registry, "get_entity_by_id"):
            entity = self.search_service.registry.get_entity_by_id(entity_id)
            if entity and isinstance(entity, dict):
                return entity  # type: ignore[no-any-return]

        # Try to get from game state manager
        if hasattr(self.game_state_manager, "get_entity"):
            entity = self.game_state_manager.get_entity(entity_id)
            if entity and isinstance(entity, dict):
                return entity  # type: ignore[no-any-return]

        return None

    def _get_location_data(self, location_id: str) -> dict[str, Any] | None:
        """
        Get location data.

        Args:
            location_id: Location ID

        Returns:
            Location data or None if not found
        """
        # This would integrate with your game's location system
        # For now, use the generic entity getter
        return self._get_entity_data(location_id)

    async def _get_entities_in_location(self, location_id: str) -> list[dict[str, Any]]:
        """
        Get entities in a specific location.

        Args:
            location_id: Location ID

        Returns:
            List of entities in the location
        """
        # This would integrate with your game's spatial system
        # For now, we'll provide a placeholder implementation

        return []

    async def _get_connected_entities(self, entity_id: str) -> list[dict[str, Any]]:
        """
        Get entities explicitly connected to an entity.

        Args:
            entity_id: Entity ID

        Returns:
            List of connected entities
        """
        # This would integrate with your game's relationship system
        # For now, we'll provide a placeholder implementation

        return []
