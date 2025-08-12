"""Query processor for handling player information requests."""

import time

from game_loop.llm.ollama.client import OllamaClient
from game_loop.search.semantic_search import SemanticSearchService
from game_loop.state.manager import GameStateManager

from .information_aggregator import InformationAggregator
from .query_models import (
    InformationSource,
    QueryContext,
    QueryRequest,
    QueryResponse,
    QueryType,
)


class QueryProcessor:
    """Processes player information requests using LLM and semantic search."""

    def __init__(
        self,
        semantic_search: SemanticSearchService,
        game_state_manager: GameStateManager,
        llm_client: OllamaClient,
    ):
        self.semantic_search = semantic_search
        self.game_state_manager = game_state_manager
        self.llm_client = llm_client
        self.information_aggregator = InformationAggregator(
            semantic_search, game_state_manager
        )

        # Query type handlers
        self._query_handlers = {
            QueryType.WORLD_INFO: self._handle_world_info_query,
            QueryType.OBJECT_INFO: self._handle_object_info_query,
            QueryType.NPC_INFO: self._handle_npc_info_query,
            QueryType.LOCATION_INFO: self._handle_location_info_query,
            QueryType.HELP: self._handle_help_query,
            QueryType.STATUS: self._handle_status_query,
            QueryType.INVENTORY: self._handle_inventory_query,
            QueryType.QUEST_INFO: self._handle_quest_info_query,
        }

    async def process_query(self, query_request: QueryRequest) -> QueryResponse:
        """Process a player query and generate response."""
        start_time = time.time()

        try:
            # Classify query type if not already specified
            if query_request.query_type is None:
                query_type = await self._classify_query_type(query_request.query_text)
                query_request.query_type = query_type

            # Get query handler
            handler = self._query_handlers.get(query_request.query_type)
            if not handler:
                return QueryResponse.error_response(
                    f"Unknown query type: {query_request.query_type.value}"
                )

            # Process the query
            response = await handler(query_request)

            # Add processing time to metadata
            processing_time = int((time.time() - start_time) * 1000)
            if response.metadata is None:
                response.metadata = {}
            response.metadata["processing_time_ms"] = processing_time

            return response

        except Exception as e:
            return QueryResponse.error_response(
                "An error occurred while processing your query",
                errors=[str(e)],
            )

    async def _classify_query_type(self, query_text: str) -> QueryType:
        """Determine the type of query using LLM."""
        classification_prompt = f"""
        Classify this player query into one of these categories:
        - WORLD_INFO: Questions about the game world, lore, history
        - OBJECT_INFO: Questions about specific objects or items
        - NPC_INFO: Questions about characters or NPCs
        - LOCATION_INFO: Questions about current or other locations
        - HELP: Requests for help or instructions
        - STATUS: Requests for player status or progress
        - INVENTORY: Questions about player's inventory
        - QUEST_INFO: Questions about quests or objectives

        Query: "{query_text}"

        Respond with just the category name.
        """

        try:
            response = await self.llm_client.generate_response(
                classification_prompt, model="qwen3:1.7b"
            )

            # Extract category from response
            category = response.strip().upper()

            # Map to QueryType enum
            for query_type in QueryType:
                if query_type.value.upper() == category:
                    return query_type

            # Default fallback
            return QueryType.WORLD_INFO

        except Exception:
            # Default to world info if classification fails
            return QueryType.WORLD_INFO

    async def _search_relevant_information(
        self, query: QueryRequest
    ) -> list[InformationSource]:
        """Find relevant information using semantic search."""
        try:
            # Use semantic search to find relevant entities/content
            search_results = await self.semantic_search.search_entities(
                query.query_text, limit=10, min_similarity=0.3
            )

            information_sources = []
            for result in search_results:
                source = InformationSource(
                    source_id=result.get("entity_id", "unknown"),
                    source_type=result.get("entity_type", "entity"),
                    source_name=result.get("name", "Unknown"),
                    content=result.get("description", ""),
                    relevance_score=result.get("similarity", 0.0),
                    metadata=result.get("metadata", {}),
                )
                information_sources.append(source)

            return information_sources

        except Exception:
            return []

    async def _generate_response(
        self,
        query: QueryRequest,
        information_sources: list[InformationSource],
        context: QueryContext,
    ) -> str:
        """Generate natural language response using LLM."""
        # Format information for the prompt
        sources_text = ""
        for i, source in enumerate(information_sources[:5], 1):  # Limit to top 5
            sources_text += f"{i}. {source.source_name}: {source.content}\n"

        prompt = f"""
        Based on the following information sources, provide a helpful answer to the player's query.

        Query: "{query.query_text}"
        
        Information Sources:
        {sources_text}

        Player Context:
        - Location: {context.current_location_id or "Unknown"}
        - Recent Actions: {", ".join(context.recent_actions or [])}
        - Active Quests: {", ".join(context.active_quests or [])}

        Provide a natural, conversational response that directly answers the query using the available information. 
        If information is incomplete, mention what aspects you're uncertain about.
        Keep the response concise but informative.
        """

        try:
            response = await self.llm_client.generate_response(
                prompt, model="qwen3:1.7b"
            )
            return response.strip()

        except Exception as e:
            return f"I apologize, but I encountered an issue generating a response: {str(e)}"

    async def _handle_world_info_query(self, query: QueryRequest) -> QueryResponse:
        """Handle queries about the game world and lore."""
        context = QueryContext(player_id=query.player_id, **query.context)

        # Search for relevant world information
        information_sources = await self._search_relevant_information(query)

        # Get additional world information from aggregator
        world_info = await self.information_aggregator.gather_world_information(
            query.query_text, query.context
        )

        # Add aggregated info as sources
        if world_info:
            for key, value in world_info.items():
                if isinstance(value, str) and value:
                    source = InformationSource(
                        source_id=f"world_{key}",
                        source_type="world",
                        source_name=key.replace("_", " ").title(),
                        content=value,
                        relevance_score=0.8,
                    )
                    information_sources.append(source)

        if not information_sources:
            return QueryResponse.success_response(
                "I don't have specific information about that aspect of the world. "
                "You might want to explore or talk to NPCs to learn more.",
                information_type="world_info",
                confidence=0.3,
            )

        # Generate response
        response_text = await self._generate_response(
            query, information_sources, context
        )

        return QueryResponse.success_response(
            response_text,
            information_type="world_info",
            sources=[source.source_name for source in information_sources[:3]],
            confidence=0.8,
        )

    async def _handle_object_info_query(self, query: QueryRequest) -> QueryResponse:
        """Handle queries about specific objects or items."""
        context = QueryContext(player_id=query.player_id, **query.context)

        # Extract object name from query
        object_info = await self.information_aggregator.gather_object_information(
            query.query_text, query.context
        )

        information_sources = await self._search_relevant_information(query)

        if object_info:
            for key, value in object_info.items():
                if isinstance(value, str) and value:
                    source = InformationSource(
                        source_id=f"object_{key}",
                        source_type="object",
                        source_name=key.replace("_", " ").title(),
                        content=value,
                        relevance_score=0.9,
                    )
                    information_sources.append(source)

        if not information_sources:
            return QueryResponse.success_response(
                "I don't have information about that object. "
                "Try examining it directly or looking around for more details.",
                information_type="object_info",
                confidence=0.3,
            )

        response_text = await self._generate_response(
            query, information_sources, context
        )

        return QueryResponse.success_response(
            response_text,
            information_type="object_info",
            sources=[source.source_name for source in information_sources[:3]],
            confidence=0.8,
        )

    async def _handle_npc_info_query(self, query: QueryRequest) -> QueryResponse:
        """Handle queries about NPCs and characters."""
        context = QueryContext(player_id=query.player_id, **query.context)

        # Get NPC information
        npc_info = await self.information_aggregator.gather_npc_information(
            query.query_text, query.context
        )

        information_sources = await self._search_relevant_information(query)

        if npc_info:
            for key, value in npc_info.items():
                if isinstance(value, str) and value:
                    source = InformationSource(
                        source_id=f"npc_{key}",
                        source_type="npc",
                        source_name=key.replace("_", " ").title(),
                        content=value,
                        relevance_score=0.9,
                    )
                    information_sources.append(source)

        if not information_sources:
            return QueryResponse.success_response(
                "I don't have information about that character. "
                "Try talking to them or other NPCs who might know them.",
                information_type="npc_info",
                confidence=0.3,
            )

        response_text = await self._generate_response(
            query, information_sources, context
        )

        return QueryResponse.success_response(
            response_text,
            information_type="npc_info",
            sources=[source.source_name for source in information_sources[:3]],
            confidence=0.8,
        )

    async def _handle_location_info_query(self, query: QueryRequest) -> QueryResponse:
        """Handle queries about locations."""
        context = QueryContext(player_id=query.player_id, **query.context)

        # Get current location details from game state
        location_info = {}
        if context.current_location_id:
            try:
                location_state = await self.game_state_manager.get_location_state(
                    context.current_location_id
                )
                if location_state:
                    location_info = location_state.to_dict()
            except Exception:
                pass

        information_sources = await self._search_relevant_information(query)

        # Add location info as source
        if location_info:
            source = InformationSource(
                source_id=f"location_{context.current_location_id}",
                source_type="location",
                source_name="Current Location",
                content=location_info.get("description", ""),
                relevance_score=1.0,
                metadata=location_info,
            )
            information_sources.insert(0, source)

        response_text = await self._generate_response(
            query, information_sources, context
        )

        return QueryResponse.success_response(
            response_text,
            information_type="location_info",
            sources=[source.source_name for source in information_sources[:3]],
            confidence=0.9,
        )

    async def _handle_help_query(self, query: QueryRequest) -> QueryResponse:
        """Handle help requests."""
        help_text = """
        Here are some things you can do:
        
        • **Movement**: Use commands like 'go north', 'enter door', 'climb stairs'
        • **Interaction**: Try 'look at', 'examine', 'use', 'take', 'give'
        • **Conversation**: Talk to NPCs with 'talk to [name]' or 'ask [name] about [topic]'
        • **Inventory**: Check your items with 'inventory' or 'check inventory'
        • **Quests**: View your quests with 'quests' or 'quest status'
        • **Information**: Ask questions like 'what is [object]?' or 'tell me about [topic]'
        
        You can also just describe what you want to do in natural language!
        """

        return QueryResponse.success_response(
            help_text,
            information_type="help",
            sources=["Game System"],
            confidence=1.0,
        )

    async def _handle_status_query(self, query: QueryRequest) -> QueryResponse:
        """Handle player status requests."""
        try:
            # Get player state
            player_state = await self.game_state_manager.get_player_state(
                query.player_id
            )

            if not player_state:
                return QueryResponse.error_response("Could not retrieve player status")

            status_info = f"""
            **Player Status:**
            • Location: {player_state.current_location_id or "Unknown"}
            • Health: {getattr(player_state, 'health', 'Unknown')}
            • Level: {getattr(player_state, 'level', 'Unknown')}
            • Experience: {getattr(player_state, 'experience', 'Unknown')}
            """

            return QueryResponse.success_response(
                status_info,
                information_type="status",
                sources=["Player State"],
                confidence=1.0,
            )

        except Exception as e:
            return QueryResponse.error_response(
                f"Could not retrieve player status: {str(e)}"
            )

    async def _handle_inventory_query(self, query: QueryRequest) -> QueryResponse:
        """Handle inventory requests."""
        try:
            # Get player inventory from game state
            player_state = await self.game_state_manager.get_player_state(
                query.player_id
            )

            if not player_state:
                return QueryResponse.error_response("Could not access inventory")

            # This would be implemented based on the actual inventory system
            inventory_info = "Your inventory system is not yet fully implemented."

            return QueryResponse.success_response(
                inventory_info,
                information_type="inventory",
                sources=["Inventory System"],
                confidence=0.5,
            )

        except Exception as e:
            return QueryResponse.error_response(f"Could not access inventory: {str(e)}")

    async def _handle_quest_info_query(self, query: QueryRequest) -> QueryResponse:
        """Handle quest information requests."""
        try:
            # This would integrate with the quest system
            quest_info = "Quest information system is available but not yet integrated."

            return QueryResponse.success_response(
                quest_info,
                information_type="quest_info",
                sources=["Quest System"],
                confidence=0.5,
            )

        except Exception as e:
            return QueryResponse.error_response(
                f"Could not retrieve quest information: {str(e)}"
            )
