"""
Embedding Manager stub for NPC system type checking.
"""

from typing import Any


class EmbeddingManager:
    """Stub for embedding manager to support type checking."""
    
    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        return []
        
    async def store_embedding(self, entity_id: str, embedding: list[float]) -> bool:
        """Store embedding for entity."""
        return True
        
    async def search_similar(self, query_embedding: list[float], limit: int = 10) -> list[dict[str, Any]]:
        """Search for similar embeddings."""
        return []