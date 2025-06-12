"""Knowledge extractor for learning information from conversations."""

import json
from typing import Any

from game_loop.llm.ollama.client import OllamaClient
from game_loop.search.semantic_search import SemanticSearchService

from .conversation_models import ConversationContext


class KnowledgeExtractor:
    """Extracts and stores knowledge from conversations."""

    def __init__(
        self,
        llm_client: OllamaClient,
        semantic_search: SemanticSearchService,
    ):
        self.llm_client = llm_client
        self.semantic_search = semantic_search

    async def extract_information(
        self, conversation: ConversationContext
    ) -> list[dict[str, Any]]:
        """Extract new information from conversation."""
        if conversation.get_exchange_count() < 2:
            return []

        try:
            # Format conversation for analysis
            conversation_text = self._format_conversation_for_analysis(conversation)

            # Extract information using LLM
            extracted_info = await self._analyze_conversation_for_knowledge(
                conversation_text
            )

            return extracted_info

        except Exception as e:
            return [{"error": f"Failed to extract knowledge: {str(e)}"}]

    async def store_knowledge(
        self, information: list[dict[str, Any]], source_context: dict[str, Any]
    ) -> bool:
        """Store extracted knowledge in the game state."""
        try:
            # For now, we'll just validate and return success
            # In a full implementation, this would integrate with the database
            # and entity embedding system

            stored_count = 0
            for info in information:
                if self._validate_extracted_info(info):
                    # Here we would:
                    # 1. Store in database
                    # 2. Create embeddings for semantic search
                    # 3. Update relevant game entities
                    stored_count += 1

            return stored_count > 0

        except Exception:
            return False

    async def update_npc_knowledge(
        self, npc_id: str, new_knowledge: dict[str, Any]
    ) -> bool:
        """Update NPC's knowledge base."""
        try:
            # This would update the NPC's personality and knowledge areas
            # For now, just validate the input
            if not npc_id or not new_knowledge:
                return False

            # In full implementation:
            # 1. Validate new knowledge
            # 2. Update NPC personality in database
            # 3. Update knowledge areas and relationships
            # 4. Create embeddings for new knowledge

            return True

        except Exception:
            return False

    def _format_conversation_for_analysis(
        self, conversation: ConversationContext
    ) -> str:
        """Format conversation exchanges for LLM analysis."""
        formatted_exchanges = []

        for exchange in conversation.conversation_history:
            speaker = (
                "Player"
                if exchange.speaker_id == conversation.player_id
                else conversation.npc_id
            )
            formatted_exchanges.append(f"{speaker}: {exchange.message_text}")

        return "\n".join(formatted_exchanges)

    async def _analyze_conversation_for_knowledge(
        self, conversation_text: str
    ) -> list[dict[str, Any]]:
        """Analyze conversation and extract factual information."""
        analysis_prompt = f"""
        Analyze the following conversation and extract any new factual information that was revealed.
        Focus on concrete facts, not opinions or speculation.

        Conversation:
        {conversation_text}

        Extract information in these categories:
        1. World/Lore information (historical facts, locations, events)
        2. Character relationships and backgrounds
        3. Location descriptions or connections
        4. Object information or properties
        5. Quest or objective information
        6. Skills, abilities, or game mechanics

        For each piece of information, specify:
        - Category (world_lore, character_info, location_info, object_info, quest_info, game_mechanic)
        - Specific information learned (be precise and factual)
        - Confidence level (high/medium/low) based on how clearly it was stated
        - Source (which speaker revealed it)

        Format as JSON with this structure:
        {{
          "extracted_information": [
            {{
              "category": "world_lore",
              "information": "specific fact learned",
              "confidence": "high",
              "source": "speaker_name",
              "keywords": ["key", "terms", "for", "search"]
            }}
          ]
        }}

        If no significant factual information was revealed, return an empty list.
        """

        try:
            response = await self.llm_client.generate_response(
                analysis_prompt, model="qwen2.5:3b"
            )

            # Try to parse JSON response
            try:
                parsed_response = json.loads(response)
                return parsed_response.get("extracted_information", [])
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract information manually
                return self._parse_fallback_response(response)

        except Exception:
            return []

    def _parse_fallback_response(self, response: str) -> list[dict[str, Any]]:
        """Parse response when JSON parsing fails."""
        # Simple fallback parsing
        information = []

        lines = response.split("\n")
        for line in lines:
            line = line.strip()
            if (
                line
                and ":" in line
                and any(
                    category in line.lower()
                    for category in [
                        "world",
                        "character",
                        "location",
                        "object",
                        "quest",
                    ]
                )
            ):
                # Try to extract basic information
                try:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        category = self._classify_information_category(parts[0])
                        info_text = parts[1].strip()

                        if info_text:
                            information.append(
                                {
                                    "category": category,
                                    "information": info_text,
                                    "confidence": "medium",
                                    "source": "unknown",
                                    "keywords": self._extract_keywords(info_text),
                                }
                            )
                except Exception:
                    continue

        return information

    def _classify_information_category(self, text: str) -> str:
        """Classify information into categories."""
        text_lower = text.lower()

        if any(word in text_lower for word in ["world", "lore", "history", "ancient"]):
            return "world_lore"
        elif any(
            word in text_lower
            for word in ["character", "npc", "person", "relationship"]
        ):
            return "character_info"
        elif any(
            word in text_lower for word in ["location", "place", "area", "region"]
        ):
            return "location_info"
        elif any(
            word in text_lower for word in ["object", "item", "artifact", "weapon"]
        ):
            return "object_info"
        elif any(
            word in text_lower for word in ["quest", "mission", "task", "objective"]
        ):
            return "quest_info"
        else:
            return "general_info"

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keywords from information text."""
        # Simple keyword extraction
        import re

        # Remove common words and extract meaningful terms
        common_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "been",
            "be",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "this",
            "that",
            "these",
            "those",
            "they",
            "them",
            "their",
            "there",
            "where",
            "when",
            "what",
            "who",
            "why",
            "how",
            "it",
            "its",
            "he",
            "she",
            "him",
            "her",
            "we",
            "us",
            "our",
            "you",
            "your",
        }

        # Extract words
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        keywords = [word for word in words if word not in common_words]

        # Return unique keywords, limited to 5
        return list(set(keywords))[:5]

    def _validate_extracted_info(self, info: dict[str, Any]) -> bool:
        """Validate extracted information structure."""
        required_fields = ["category", "information", "confidence", "source"]

        # Check required fields exist
        for field in required_fields:
            if field not in info:
                return False

        # Check field types and values
        if not isinstance(info["information"], str) or not info["information"].strip():
            return False

        if info["confidence"] not in ["high", "medium", "low"]:
            return False

        if info["category"] not in [
            "world_lore",
            "character_info",
            "location_info",
            "object_info",
            "quest_info",
            "game_mechanic",
            "general_info",
        ]:
            return False

        return True

    async def create_knowledge_embeddings(
        self, information: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Create embeddings for extracted knowledge for semantic search."""
        embeddings = []

        try:
            for info in information:
                if not self._validate_extracted_info(info):
                    continue

                # Create embedding for the information
                # This would integrate with the embedding system
                embedding_data = {
                    "entity_id": f"knowledge_{hash(info['information'])}",
                    "entity_type": info["category"],
                    "name": (
                        info.get("keywords", ["unknown"])[0]
                        if info.get("keywords")
                        else "unknown"
                    ),
                    "description": info["information"],
                    "metadata": {
                        "confidence": info["confidence"],
                        "source": info["source"],
                        "keywords": info.get("keywords", []),
                        "extracted_at": info.get("extracted_at"),
                    },
                }

                embeddings.append(embedding_data)

        except Exception:
            pass

        return embeddings
