"""Context-aware help system that provides relevant assistance."""

from typing import Any

from ...database.session_factory import DatabaseSessionFactory
from ...llm.ollama.client import OllamaClient
from ...search.semantic_search import SemanticSearchService
from ..models.system_models import (
    HelpContext,
    HelpResponse,
    HelpTopic,
    PlayerSkillLevel,
)


class HelpSystem:
    """Context-aware help system that provides relevant assistance."""

    def __init__(
        self,
        session_factory: DatabaseSessionFactory,
        llm_client: OllamaClient,
        semantic_search: SemanticSearchService,
    ):
        self.session_factory = session_factory
        self.llm_client = llm_client
        self.semantic_search = semantic_search
        self.help_topics: dict[str, HelpTopic] = {}
        self._load_help_topics_task = None

    async def initialize(self) -> None:
        """Initialize the help system by loading topics."""
        await self._load_help_topics()

    async def get_help(
        self, topic: str | None = None, context: dict[str, Any] | None = None
    ) -> HelpResponse:
        """Get contextual help based on current game state."""
        try:
            if not self.help_topics:
                await self._load_help_topics()

            # If no specific topic, provide contextual help
            if not topic:
                return await self._get_contextual_help(context or {})

            # Clean and normalize topic
            topic = topic.lower().strip()

            # Try exact match first
            if topic in self.help_topics:
                help_topic = self.help_topics[topic]
                return await self._create_help_response(help_topic, context or {})

            # Try partial matches
            matching_topics = self._find_matching_topics(topic)
            if matching_topics:
                # Return the best match
                best_match = matching_topics[0]
                help_topic = self.help_topics[best_match]
                return await self._create_help_response(help_topic, context or {})

            # Use semantic search to find relevant topics
            semantic_matches = await self._search_help_semantic(topic)
            if semantic_matches:
                return await self._create_help_response(
                    semantic_matches[0], context or {}
                )

            # Generate help using LLM if no matches found
            return await self._generate_llm_help(topic, context or {})

        except Exception as e:
            return HelpResponse(
                topic=topic or "help",
                content=f"Sorry, I encountered an error while trying to help: {str(e)}",
                category="error",
            )

    async def get_contextual_suggestions(self, context: dict[str, Any]) -> list[str]:
        """Suggest relevant commands based on current situation."""
        try:
            help_context = self._analyze_context_for_help(context)
            suggestions = []

            # Basic suggestions based on context
            if help_context.current_location:
                suggestions.extend(["look around", "examine room"])

            if help_context.nearby_objects:
                suggestions.extend(
                    [f"examine {obj}" for obj in help_context.nearby_objects[:3]]
                )
                suggestions.extend(
                    [f"take {obj}" for obj in help_context.nearby_objects[:2]]
                )

            if help_context.nearby_npcs:
                suggestions.extend(
                    [f"talk to {npc}" for npc in help_context.nearby_npcs[:2]]
                )

            # Skill-based suggestions
            if help_context.player_skill_level == PlayerSkillLevel.BEGINNER:
                suggestions.extend(["help basics", "help movement", "inventory"])

            # Quest-based suggestions
            if help_context.current_quest:
                suggestions.extend(["quest status", "help quests"])

            # Add general helpful commands
            suggestions.extend(["save game", "help", "settings"])

            return list(
                dict.fromkeys(suggestions)
            )  # Remove duplicates while preserving order

        except Exception:
            return ["help", "look around", "inventory", "save game"]

    async def search_help(self, query: str) -> list[HelpTopic]:
        """Search help content using semantic similarity."""
        try:
            if not self.help_topics:
                await self._load_help_topics()

            # First try keyword matching
            keyword_matches = []
            query_lower = query.lower()

            for topic in self.help_topics.values():
                if any(keyword in query_lower for keyword in topic.keywords):
                    keyword_matches.append(topic)

            if keyword_matches:
                return keyword_matches[:5]

            # Fall back to semantic search
            return await self._search_help_semantic(query)

        except Exception:
            return []

    def _analyze_context_for_help(self, context: dict[str, Any]) -> HelpContext:
        """Analyze game context to provide relevant help."""
        return HelpContext(
            current_location=context.get("current_location"),
            available_commands=context.get("available_commands", []),
            nearby_objects=context.get("nearby_objects", []),
            nearby_npcs=context.get("nearby_npcs", []),
            player_level=context.get("player_level", 1),
            player_skill_level=self._assess_player_skill_level(context),
            recent_actions=context.get("recent_actions", []),
            current_quest=context.get("current_quest"),
        )

    def _assess_player_skill_level(self, context: dict[str, Any]) -> PlayerSkillLevel:
        """Assess player skill level based on context."""
        player_level = context.get("player_level", 1)
        recent_actions = context.get("recent_actions", [])
        total_actions = context.get("total_actions", 0)

        # Simple heuristic for skill assessment
        if player_level < 3 or total_actions < 10:
            return PlayerSkillLevel.BEGINNER
        elif player_level < 10 or total_actions < 100:
            return PlayerSkillLevel.INTERMEDIATE
        else:
            return PlayerSkillLevel.ADVANCED

    async def _load_help_topics(self) -> None:
        """Load help content from database."""
        try:
            async with self.session_factory.get_session() as session:
                result = await session.execute(
                    """
                    SELECT topic_key, title, content, category, keywords, examples, related_topics
                    FROM help_topics
                    ORDER BY category, title
                    """
                )

                self.help_topics = {}
                for row in result:
                    topic = HelpTopic(
                        topic_id=row[0],
                        title=row[1],
                        content=row[2],
                        category=row[3],
                        keywords=row[4] or [],
                        examples=row[5] or [],
                        related_topics=row[6] or [],
                    )
                    self.help_topics[row[0]] = topic

        except Exception:
            # Load default help topics if database fails
            self._load_default_help_topics()

    def _load_default_help_topics(self) -> None:
        """Load default help topics as fallback."""
        default_topics = {
            "basic_commands": HelpTopic(
                topic_id="basic_commands",
                title="Basic Commands",
                content="Basic commands for interacting with the game:\n\n"
                "• look/examine - Look at things\n"
                "• go/move - Move to different locations\n"
                "• take/get - Pick up objects\n"
                "• use - Use items or interact with objects\n"
                "• talk - Speak with characters\n"
                "• inventory - Check your belongings\n"
                "• help - Get help on topics",
                category="getting_started",
                keywords=["commands", "basic", "start", "begin"],
                examples=["look around", "go north", "take key", "talk to guard"],
            ),
            "movement": HelpTopic(
                topic_id="movement",
                title="Movement and Navigation",
                content="Moving around the game world:\n\n"
                "• Use directional commands: north, south, east, west, up, down\n"
                "• Try 'go <direction>' or just the direction name\n"
                "• Some locations have special exits like 'enter building'\n"
                "• Use 'look' to see available exits",
                category="gameplay",
                keywords=["movement", "navigation", "go", "direction"],
                examples=["north", "go south", "enter tavern", "climb stairs"],
            ),
            "save_load": HelpTopic(
                topic_id="save_load",
                title="Saving and Loading",
                content="Managing your game progress:\n\n"
                "• 'save game' - Quick save\n"
                "• 'save as <name>' - Save with custom name\n"
                "• 'load game' - Load most recent save\n"
                "• 'load <name>' - Load specific save\n"
                "• 'list saves' - See all your saves",
                category="system",
                keywords=["save", "load", "progress", "continue"],
                examples=[
                    "save game",
                    "save as my_adventure",
                    "load game",
                    "list saves",
                ],
            ),
        }
        self.help_topics = default_topics

    def _find_matching_topics(self, query: str) -> list[str]:
        """Find topics that match the query."""
        matches = []
        query_lower = query.lower()

        for topic_key, topic in self.help_topics.items():
            # Check topic key
            if query_lower in topic_key.lower():
                matches.append(topic_key)
                continue

            # Check title
            if query_lower in topic.title.lower():
                matches.append(topic_key)
                continue

            # Check keywords
            if any(query_lower in keyword.lower() for keyword in topic.keywords):
                matches.append(topic_key)

        return matches

    async def _search_help_semantic(self, query: str) -> list[HelpTopic]:
        """Search help topics using semantic similarity."""
        try:
            # Create search corpus from help topics
            topic_texts = []
            topic_keys = []

            for topic_key, topic in self.help_topics.items():
                text = f"{topic.title} {topic.content} {' '.join(topic.keywords)}"
                topic_texts.append(text)
                topic_keys.append(topic_key)

            # Use semantic search to find relevant topics
            if hasattr(self.semantic_search, "search_similar_texts"):
                results = await self.semantic_search.search_similar_texts(
                    query, topic_texts, limit=3
                )
                return [
                    self.help_topics[topic_keys[result["index"]]] for result in results
                ]

            return []

        except Exception:
            return []

    async def _get_contextual_help(self, context: dict[str, Any]) -> HelpResponse:
        """Get contextual help based on current situation."""
        help_context = self._analyze_context_for_help(context)

        # Generate contextual content
        content_parts = []

        if help_context.player_skill_level == PlayerSkillLevel.BEGINNER:
            content_parts.append("**Getting Started:**")
            content_parts.append("Try these basic commands to explore:")
            content_parts.append("• 'look around' - See your surroundings")
            content_parts.append("• 'inventory' - Check what you're carrying")
            content_parts.append("• 'help basic_commands' - Learn essential commands")
        else:
            content_parts.append("**Available Help Topics:**")
            content_parts.append("• help movement - Navigation and travel")
            content_parts.append("• help inventory - Managing items")
            content_parts.append("• help conversation - Talking to NPCs")
            content_parts.append("• help save_load - Saving your progress")

        if help_context.current_location:
            content_parts.append(
                f"\n**Current Location:** {help_context.current_location}"
            )

        # Add contextual suggestions
        suggestions = await self.get_contextual_suggestions(context)
        if suggestions:
            content_parts.append("\n**Suggested Actions:**")
            for suggestion in suggestions[:5]:
                content_parts.append(f"• {suggestion}")

        content = "\n".join(content_parts)

        return HelpResponse(
            topic="contextual_help",
            content=content,
            contextual_suggestions=suggestions,
            category="contextual",
        )

    async def _create_help_response(
        self, topic: HelpTopic, context: dict[str, Any]
    ) -> HelpResponse:
        """Create a help response from a topic."""
        # Get contextual suggestions
        suggestions = await self.get_contextual_suggestions(context)

        return HelpResponse(
            topic=topic.title,
            content=topic.content,
            related_topics=topic.related_topics,
            contextual_suggestions=suggestions,
            examples=topic.examples,
            category=topic.category,
        )

    async def _generate_llm_help(
        self, topic: str, context: dict[str, Any]
    ) -> HelpResponse:
        """Generate help using LLM when no existing help is found."""
        try:
            help_context = self._analyze_context_for_help(context)

            prompt = f"""
            You are a helpful assistant for a text adventure game. The player is asking for help about: "{topic}"
            
            Current context:
            - Location: {help_context.current_location or 'Unknown'}
            - Player level: {help_context.player_level}
            - Skill level: {help_context.player_skill_level.value}
            - Recent actions: {', '.join(help_context.recent_actions[-3:]) if help_context.recent_actions else 'None'}
            
            Provide helpful information about "{topic}" in the context of a text adventure game.
            Keep the response concise but informative. Include specific examples if relevant.
            """

            response = await self.llm_client.generate_response(
                prompt=prompt,
                system_prompt="You are a helpful game assistant. Provide clear, concise help.",
                max_tokens=300,
            )

            suggestions = await self.get_contextual_suggestions(context)

            return HelpResponse(
                topic=topic,
                content=response.get(
                    "response", "Sorry, I couldn't generate help for that topic."
                ),
                contextual_suggestions=suggestions,
                category="generated",
            )

        except Exception:
            return HelpResponse(
                topic=topic,
                content=f"Sorry, I don't have specific help for '{topic}'. Try 'help' for general assistance.",
                category="error",
            )
