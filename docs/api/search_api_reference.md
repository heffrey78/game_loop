# Search API Reference

## Overview

The Search API provides RESTful endpoints for executing semantic searches across game entities, finding similar entities, retrieving contextual information, and more. This document details the available endpoints, their parameters, and response formats.

## Base URL

All API endpoints are prefixed with: `/api/search`

## Authentication

API requests require authentication through the standard game authentication mechanism. Include the authentication token in the request header:

```
Authorization: Bearer {your_auth_token}
```

## Endpoints

### 1. Search Entities

**Endpoint:** `GET /api/search`

**Description:** Main search endpoint for finding game entities using text queries.

**Parameters:**
- `query` (required): Search query text
- `entity_types` (optional): Comma-separated list of entity types to search
- `strategy` (optional): Search strategy (semantic, keyword, hybrid, exact). Defaults to "hybrid"
- `top_k` (optional): Maximum number of results to return. Defaults to 10
- `threshold` (optional): Minimum similarity threshold. Defaults to 0.7
- `page` (optional): Page number for pagination. Defaults to 1
- `page_size` (optional): Number of results per page. Defaults to 10
- `format` (optional): Result format (detailed, summary, compact). Defaults to "detailed"

**Response:**
```json
{
  "results": [
    {
      "entity_id": "item_123",
      "entity_type": "item",
      "score": 0.92,
      "snippet": "A <em>magical</em> sword with fire damage...",
      "data": {
        "name": "Flame Blade",
        "description": "A magical sword that deals fire damage.",
        "attributes": { ... }
      }
    },
    // More results
  ],
  "pagination": {
    "page": 1,
    "page_size": 10,
    "total_results": 42,
    "total_pages": 5,
    "has_next": true,
    "has_prev": false
  },
  "metadata": {
    "query": "magical sword",
    "strategy": "hybrid",
    "entity_types": ["item", "weapon"],
    "threshold": 0.7
  }
}
```

### 2. Find Similar Entities

**Endpoint:** `GET /api/search/similar/{entity_id}`

**Description:** Find entities similar to a reference entity.

**Parameters:**
- `entity_id` (required): Reference entity ID
- `top_k` (optional): Maximum number of results to return. Defaults to 10
- `threshold` (optional): Minimum similarity threshold. Defaults to 0.7
- `entity_types` (optional): Comma-separated list of entity types to search
- `page` (optional): Page number for pagination. Defaults to 1
- `page_size` (optional): Number of results per page. Defaults to 10

**Response:**
```json
{
  "results": [
    {
      "entity_id": "item_456",
      "entity_type": "item",
      "score": 0.89,
      "data": { ... }
    },
    // More results
  ],
  "pagination": { ... },
  "metadata": {
    "reference_entity": "item_123",
    "threshold": 0.7,
    "entity_types": ["item"]
  }
}
```

### 3. Get Entity Context

**Endpoint:** `GET /api/search/context/{entity_id}`

**Description:** Get contextually relevant entities for a specific entity.

**Parameters:**
- `entity_id` (required): Entity ID to get context for
- `context_types` (optional): Comma-separated list of context types (related, similar, hierarchy). Defaults to "related,similar"
- `top_k` (optional): Maximum number of results per context type. Defaults to 5

**Response:**
```json
{
  "entity_id": "npc_789",
  "entity_type": "character",
  "contexts": {
    "similar": [
      {
        "entity_id": "npc_790",
        "similarity": 0.85,
        "data": { ... }
      },
      // More similar entities
    ],
    "related": [
      {
        "entity_id": "quest_123",
        "relation_type": "quest_giver",
        "data": { ... }
      },
      // More related entities
    ],
    "hierarchy": {
      "parents": [ ... ],
      "children": [ ... ]
    }
  }
}
```

### 4. Search By Example

**Endpoint:** `POST /api/search/by-example`

**Description:** Search using an example entity as the query.

**Request Body:**
```json
{
  "name": "Example Item",
  "description": "This is an example item that has certain properties.",
  "entity_type": "item",
  "attributes": {
    "rarity": "rare",
    "damage": 15
  }
}
```

**Response:**
```json
{
  "results": [ ... ],
  "metadata": {
    "derived_query": "Example Item This is an example item that has certain properties. rare",
    "entity_type": "item"
  }
}
```

### 5. Get Search Suggestions

**Endpoint:** `GET /api/search/suggestions`

**Description:** Get search query suggestions based on partial input.

**Parameters:**
- `partial_query` (required): Partial search query
- `max_suggestions` (optional): Maximum number of suggestions to return. Defaults to 5

**Response:**
```json
{
  "partial_query": "mag",
  "suggestions": [
    "magic sword",
    "mage staff",
    "magical potion",
    "magnetic ore",
    "magnitude spell"
  ]
}
```

### 6. Get Recent Searches

**Endpoint:** `GET /api/search/recent`

**Description:** Get user's recent searches.

**Parameters:**
- `max_results` (optional): Maximum number of recent searches to return. Defaults to 10

**Response:**
```json
{
  "recent_searches": [
    {
      "query": "magical sword",
      "timestamp": 1622505600,
      "result_count": 12
    },
    // More recent searches
  ]
}
```

## Error Responses

All endpoints return standard error responses in this format:

```json
{
  "error": "Error message",
  "status_code": 400,
  "details": "Additional error details"
}
```

Common error status codes:
- `400`: Bad request (invalid parameters)
- `401`: Unauthorized
- `404`: Entity not found
- `429`: Rate limit exceeded
- `500`: Server error

## Rate Limiting

Search API endpoints are rate limited to prevent abuse. Current limits:
- 60 requests per minute per user
- 10 concurrent requests per user

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1622505660
```
