"""
Entity embedding model for storing vector embeddings in the database.

This module defines the ORM model for entity embeddings.
"""

from sqlalchemy import JSON, Column, String
from sqlalchemy.dialects.postgresql import ARRAY, FLOAT

from .base import Base


class EntityEmbedding(Base):
    """Database model for entity embeddings."""

    __tablename__ = "entity_embeddings"

    entity_id = Column(String, primary_key=True, index=True)
    entity_type = Column(String, index=True, nullable=False)
    embedding: Column = Column(ARRAY(FLOAT), nullable=False)
    metadata_json = Column(JSON, nullable=False, default={})

    def __repr__(self) -> str:
        """Return string representation of the entity embedding."""
        return (
            f"EntityEmbedding(entity_id={self.entity_id}, "
            f"entity_type={self.entity_type})"
        )
