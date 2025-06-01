"""
Entity-specific preprocessing for different game entity types.

This module provides specialized text preprocessing optimized for
different entity types such as characters, locations, items, and events.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Template registry for different entity types
ENTITY_TEMPLATES = {
    "character": (
        "Character Information:\n"
        "Name: {name}\n"
        "Description: {description}\n"
        "Role: {role}\n"
        "Background: {background}\n"
        "Personality: {personality}\n"
        "Motivations: {motivations}\n"
        "Key Relationships: {relationships}"
    ),
    "location": (
        "Location Information:\n"
        "Name: {name}\n"
        "Description: {description}\n"
        "Environment: {environment}\n"
        "Atmosphere: {atmosphere}\n"
        "Notable Features: {features}\n"
        "Connections: {connections}"
    ),
    "item": (
        "Item Information:\n"
        "Name: {name}\n"
        "Description: {description}\n"
        "Properties: {properties}\n"
        "Usage: {usage}\n"
        "Origin: {origin}\n"
        "Value: {value}"
    ),
    "event": (
        "Event Information:\n"
        "Name: {name}\n"
        "Description: {description}\n"
        "Timeline: {timeline}\n"
        "Participants: {participants}\n"
        "Consequences: {consequences}\n"
        "Significance: {significance}"
    ),
    "general": (
        "Entity Information:\n"
        "Name: {name}\n"
        "Type: {type}\n"
        "Description: {description}\n"
        "Additional Details: {details}"
    ),
}


def preprocess_character(character_entity: dict) -> str:
    """
    Preprocess a character entity for optimal embedding.

    Args:
        character_entity: Dictionary containing character data

    Returns:
        Preprocessed text optimized for character embeddings
    """
    # Extract relevant fields with fallbacks
    name = character_entity.get("name", "Unknown")
    description = character_entity.get("description", "")
    role = character_entity.get("role", "")
    background = character_entity.get("background", "")
    personality = character_entity.get("personality", "")
    motivations = character_entity.get("motivations", "")

    # Process relationships into a text format
    relationships = character_entity.get("relationships", {})
    if isinstance(relationships, dict):
        relationships_text = "; ".join([f"{k}: {v}" for k, v in relationships.items()])
    elif isinstance(relationships, list):
        relationships_text = "; ".join(relationships)
    else:
        relationships_text = str(relationships)

    # Combine all fields using the character template
    template = ENTITY_TEMPLATES["character"]
    processed_text = template.format(
        name=name,
        description=description,
        role=role,
        background=background,
        personality=personality,
        motivations=motivations,
        relationships=relationships_text,
    )

    # Clean up the text
    processed_text = clean_text(processed_text)

    return processed_text


def preprocess_location(location_entity: dict) -> str:
    """
    Preprocess a location entity for optimal embedding.

    Args:
        location_entity: Dictionary containing location data

    Returns:
        Preprocessed text optimized for location embeddings
    """
    # Extract relevant fields with fallbacks
    name = location_entity.get("name", "Unknown")
    description = location_entity.get("description", "")
    environment = location_entity.get("environment", "")
    atmosphere = location_entity.get("atmosphere", "")

    # Process features into a text format
    features_list = location_entity.get("features", [])
    if isinstance(features_list, list):
        features_text = "; ".join(features_list)
    else:
        features_text = str(features_list)

    # Process connections into a text format
    connections = location_entity.get("connections", [])
    if isinstance(connections, list):
        connections_text = "; ".join(connections)
    else:
        connections_text = str(connections)

    # Combine all fields using the location template
    template = ENTITY_TEMPLATES["location"]
    processed_text = template.format(
        name=name,
        description=description,
        environment=environment,
        atmosphere=atmosphere,
        features=features_text,
        connections=connections_text,
    )

    # Clean up the text
    processed_text = clean_text(processed_text)

    return processed_text


def preprocess_item(item_entity: dict) -> str:
    """
    Preprocess an item entity for optimal embedding.

    Args:
        item_entity: Dictionary containing item data

    Returns:
        Preprocessed text optimized for item embeddings
    """
    # Extract relevant fields with fallbacks
    name = item_entity.get("name", "Unknown")
    description = item_entity.get("description", "")
    usage = item_entity.get("usage", "")
    origin = item_entity.get("origin", "")
    value = item_entity.get("value", "")

    # Process properties into a text format
    properties = item_entity.get("properties", {})
    if isinstance(properties, dict):
        properties_text = "; ".join([f"{k}: {v}" for k, v in properties.items()])
    elif isinstance(properties, list):
        properties_text = "; ".join(properties)
    else:
        properties_text = str(properties)

    # Combine all fields using the item template
    template = ENTITY_TEMPLATES["item"]
    processed_text = template.format(
        name=name,
        description=description,
        properties=properties_text,
        usage=usage,
        origin=origin,
        value=value,
    )

    # Clean up the text
    processed_text = clean_text(processed_text)

    return processed_text


def preprocess_event(event_entity: dict) -> str:
    """
    Preprocess an event entity for optimal embedding.

    Args:
        event_entity: Dictionary containing event data

    Returns:
        Preprocessed text optimized for event embeddings
    """
    # Extract relevant fields with fallbacks
    name = event_entity.get("name", "Unknown")
    description = event_entity.get("description", "")
    timeline = event_entity.get("timeline", "")
    consequences = event_entity.get("consequences", "")
    significance = event_entity.get("significance", "")

    # Process participants into a text format
    participants = event_entity.get("participants", [])
    if isinstance(participants, list):
        participants_text = "; ".join(participants)
    else:
        participants_text = str(participants)

    # Combine all fields using the event template
    template = ENTITY_TEMPLATES["event"]
    processed_text = template.format(
        name=name,
        description=description,
        timeline=timeline,
        participants=participants_text,
        consequences=consequences,
        significance=significance,
    )

    # Clean up the text
    processed_text = clean_text(processed_text)

    return processed_text


def extract_salient_features(entity: dict, entity_type: str) -> list[str]:
    """
    Extract the most important features from an entity based on its type.

    Args:
        entity: Entity data dictionary
        entity_type: Type of entity (character, location, item, event)

    Returns:
        List of salient feature strings
    """
    features = []

    # Common features for all entity types
    if "name" in entity:
        features.append(f"Name: {entity['name']}")
    if "description" in entity:
        features.append(f"Description: {entity['description']}")

    # Entity type-specific features
    if entity_type.lower() == "character":
        if "personality" in entity:
            features.append(f"Personality: {entity['personality']}")
        if "background" in entity and len(entity["background"]) > 0:
            features.append(f"Background: {entity['background']}")
        if "motivations" in entity:
            features.append(f"Motivations: {entity['motivations']}")

    elif entity_type.lower() == "location":
        if "environment" in entity:
            features.append(f"Environment: {entity['environment']}")
        if "atmosphere" in entity:
            features.append(f"Atmosphere: {entity['atmosphere']}")

    elif entity_type.lower() == "item":
        if "usage" in entity:
            features.append(f"Usage: {entity['usage']}")
        if "value" in entity:
            features.append(f"Value: {entity['value']}")

    elif entity_type.lower() == "event":
        if "timeline" in entity:
            features.append(f"Timeline: {entity['timeline']}")
        if "consequences" in entity:
            features.append(f"Consequences: {entity['consequences']}")

    return features


def create_entity_context(entity: dict, entity_type: str) -> str:
    """
    Create contextual information for an entity to enhance embedding.

    Args:
        entity: Entity data dictionary
        entity_type: Type of entity

    Returns:
        Contextual string that enriches the entity representation
    """
    # Start with the entity type for context
    context_parts = [f"Entity Type: {entity_type.capitalize()}"]

    # Add salient features
    features = extract_salient_features(entity, entity_type)
    context_parts.extend(features)

    # Add relationships or connections if available
    if entity_type.lower() == "character" and "relationships" in entity:
        relationships = entity["relationships"]
        if isinstance(relationships, dict):
            rel_text = ", ".join([f"{k}: {v}" for k, v in relationships.items()])
            context_parts.append(f"Relationships: {rel_text}")

    if entity_type.lower() == "location" and "connections" in entity:
        connections = entity["connections"]
        if isinstance(connections, list):
            conn_text = ", ".join(connections)
            context_parts.append(f"Connections: {conn_text}")

    # Join all context parts
    return "\n".join(context_parts)


def build_entity_embedding_template(entity_type: str) -> str:
    """
    Get the embedding template for a given entity type.

    Args:
        entity_type: Type of entity

    Returns:
        Template string for the entity type
    """
    return ENTITY_TEMPLATES.get(entity_type.lower(), ENTITY_TEMPLATES["general"])


def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    # Remove multiple spaces
    cleaned = re.sub(r"\s+", " ", text)

    # Remove empty lines
    cleaned = re.sub(r"\n\s*\n", "\n", cleaned)

    # Replace None values with empty strings
    cleaned = cleaned.replace("None", "")

    # Normalize line endings
    cleaned = cleaned.replace("\r\n", "\n")

    return cleaned.strip()
