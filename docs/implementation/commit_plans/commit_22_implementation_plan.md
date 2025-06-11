# Commit 22 Implementation Plan: Query and Conversation System

**Commit ID**: commit_22  
**Phase**: 4 - Action Processing System  
**Focus**: Implement query processing and NPC conversation system with context tracking and knowledge updates

## Overview

This commit implements the Query and Conversation System as outlined in the main game loop implementation plan. The system handles information requests from players and manages NPC dialogues with context tracking and knowledge acquisition.

## Implementation Goals

### Primary Objectives
1. **QueryProcessor**: Handle information requests about the game world
2. **ConversationManager**: Manage NPC dialogues with context tracking
3. **Context Tracking**: Maintain conversation history and state
4. **Dialogue Generation**: Generate context-aware NPC responses using LLM
5. **Knowledge Updates**: Learn and store new information from conversations

### Secondary Objectives
- Integration with existing semantic search system
- Rich conversation history tracking
- Dynamic NPC personality expression
- Context-aware query responses
- Conversation state persistence

## Detailed Implementation Plan

### 1. Query Processing System (`src/game_loop/core/query/`)

#### 1.1 Query Models (`query_models.py`)
```python
@dataclass
class QueryRequest:
    """Represents a player information request."""
    query_id: str
    player_id: str
    query_text: str
    query_type: QueryType  # WORLD_INFO, OBJECT_INFO, NPC_INFO, HELP, STATUS
    context: dict[str, Any]
    timestamp: float

@dataclass
class QueryResponse:
    """Response to a player query."""
    success: bool
    response_text: str
    information_type: str
    sources: list[str]  # What sources provided the information
    related_queries: list[str]  # Suggested follow-up questions
    confidence: float  # How confident we are in the response
    errors: list[str]

class QueryType(Enum):
    WORLD_INFO = "world_info"
    OBJECT_INFO = "object_info"
    NPC_INFO = "npc_info"
    LOCATION_INFO = "location_info"
    HELP = "help"
    STATUS = "status"
    INVENTORY = "inventory"
    QUEST_INFO = "quest_info"
```

#### 1.2 Query Processor (`query_processor.py`)
```python
class QueryProcessor:
    """Processes player information requests."""
    
    def __init__(
        self,
        semantic_search: SemanticSearchService,
        game_state_manager: GameStateManager,
        llm_client: OllamaClient
    ):
        self.semantic_search = semantic_search
        self.game_state_manager = game_state_manager
        self.llm_client = llm_client
        self._query_templates = {}
    
    async def process_query(
        self, 
        query_request: QueryRequest
    ) -> QueryResponse:
        """Process a player query and generate response."""
        
    async def _classify_query_type(self, query_text: str) -> QueryType:
        """Determine the type of query using LLM."""
        
    async def _search_relevant_information(
        self, 
        query: QueryRequest
    ) -> list[dict[str, Any]]:
        """Find relevant information using semantic search."""
        
    async def _generate_response(
        self, 
        query: QueryRequest, 
        information: list[dict[str, Any]]
    ) -> str:
        """Generate natural language response using LLM."""
```

#### 1.3 Information Aggregator (`information_aggregator.py`)
```python
class InformationAggregator:
    """Aggregates information from multiple game systems."""
    
    async def gather_world_information(
        self, 
        query: str, 
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """Gather information about the game world."""
        
    async def gather_object_information(
        self, 
        object_name: str, 
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """Gather information about specific objects."""
        
    async def gather_npc_information(
        self, 
        npc_name: str, 
        context: dict[str, Any]
    ) -> dict[str, Any]:
        """Gather information about NPCs."""
```

### 2. Conversation System (`src/game_loop/core/conversation/`)

#### 2.1 Conversation Models (`conversation_models.py`)
```python
@dataclass
class ConversationContext:
    """Tracks conversation state and history."""
    conversation_id: str
    player_id: str
    npc_id: str
    topic: str | None
    mood: str  # friendly, hostile, neutral, excited, etc.
    relationship_level: float  # -1.0 to 1.0
    conversation_history: list[ConversationExchange]
    context_data: dict[str, Any]
    started_at: float
    last_updated: float

@dataclass
class ConversationExchange:
    """Single exchange in a conversation."""
    exchange_id: str
    speaker_id: str  # player_id or npc_id
    message_text: str
    message_type: str  # greeting, question, statement, farewell
    emotion: str | None
    timestamp: float
    metadata: dict[str, Any]

@dataclass
class NPCPersonality:
    """Defines NPC personality traits."""
    npc_id: str
    traits: dict[str, float]  # friendly, talkative, helpful, etc.
    knowledge_areas: list[str]
    speech_patterns: dict[str, Any]
    relationships: dict[str, float]  # relationship with other entities
    background_story: str
```

#### 2.2 Conversation Manager (`conversation_manager.py`)
```python
class ConversationManager:
    """Manages NPC conversations with context tracking."""
    
    def __init__(
        self,
        llm_client: OllamaClient,
        game_state_manager: GameStateManager,
        semantic_search: SemanticSearchService
    ):
        self.llm_client = llm_client
        self.game_state_manager = game_state_manager
        self.semantic_search = semantic_search
        self._active_conversations: dict[str, ConversationContext] = {}
        self._npc_personalities: dict[str, NPCPersonality] = {}
    
    async def start_conversation(
        self, 
        player_id: str, 
        npc_id: str, 
        context: dict[str, Any]
    ) -> ConversationContext:
        """Start a new conversation with an NPC."""
        
    async def process_player_message(
        self, 
        conversation_id: str, 
        message: str,
        context: dict[str, Any]
    ) -> ConversationExchange:
        """Process player message and generate NPC response."""
        
    async def end_conversation(
        self, 
        conversation_id: str,
        reason: str = "natural_end"
    ) -> dict[str, Any]:
        """End conversation and extract learned information."""
        
    async def _generate_npc_response(
        self, 
        context: ConversationContext, 
        player_message: str
    ) -> str:
        """Generate contextual NPC response using LLM."""
        
    async def _update_relationship(
        self, 
        context: ConversationContext, 
        interaction_type: str
    ) -> None:
        """Update relationship based on interaction."""
```

#### 2.3 Knowledge Extractor (`knowledge_extractor.py`)
```python
class KnowledgeExtractor:
    """Extracts and stores knowledge from conversations."""
    
    def __init__(
        self,
        llm_client: OllamaClient,
        embedding_manager: EmbeddingManager
    ):
        self.llm_client = llm_client
        self.embedding_manager = embedding_manager
    
    async def extract_information(
        self, 
        conversation: ConversationContext
    ) -> list[dict[str, Any]]:
        """Extract new information from conversation."""
        
    async def store_knowledge(
        self, 
        information: list[dict[str, Any]], 
        source_context: dict[str, Any]
    ) -> bool:
        """Store extracted knowledge in the game state."""
        
    async def update_npc_knowledge(
        self, 
        npc_id: str, 
        new_knowledge: dict[str, Any]
    ) -> bool:
        """Update NPC's knowledge base."""
```

### 3. Integration Components

#### 3.1 Query Command Handler (`src/game_loop/core/command_handlers/query_handler.py`)
```python
class QueryCommandHandler(BaseCommandHandler):
    """Handles query-type commands."""
    
    def __init__(
        self,
        query_processor: QueryProcessor,
        conversation_manager: ConversationManager
    ):
        self.query_processor = query_processor
        self.conversation_manager = conversation_manager
    
    async def handle(
        self, 
        command: str, 
        context: dict[str, Any]
    ) -> ActionResult:
        """Handle query commands like 'ask about', 'help', 'status'."""
        
    async def _handle_information_query(
        self, 
        query: str, 
        context: dict[str, Any]
    ) -> ActionResult:
        """Handle general information queries."""
        
    async def _handle_npc_conversation(
        self, 
        npc_target: str, 
        message: str, 
        context: dict[str, Any]
    ) -> ActionResult:
        """Handle talking to NPCs."""
```

#### 3.2 Conversation Integration (`src/game_loop/core/conversation/conversation_integration.py`)
```python
class ConversationIntegration:
    """Integrates conversation system with game systems."""
    
    def __init__(
        self,
        conversation_manager: ConversationManager,
        quest_manager: QuestManager,
        object_manager: ObjectManager
    ):
        self.conversation_manager = conversation_manager
        self.quest_manager = quest_manager
        self.object_manager = object_manager
    
    async def process_conversation_effects(
        self, 
        conversation_result: dict[str, Any]
    ) -> list[ActionResult]:
        """Process side effects from conversations (quest updates, etc.)."""
        
    async def check_conversation_triggers(
        self, 
        player_id: str, 
        context: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Check if any conversations should be triggered by events."""
```

### 4. Database Schema

#### 4.1 Database Migration (`src/game_loop/database/migrations/023_conversation_system.sql`)
```sql
-- Conversation contexts table
CREATE TABLE conversation_contexts (
    conversation_id VARCHAR(255) PRIMARY KEY,
    player_id VARCHAR(255) NOT NULL,
    npc_id VARCHAR(255) NOT NULL,
    topic VARCHAR(500),
    mood VARCHAR(100) NOT NULL DEFAULT 'neutral',
    relationship_level FLOAT NOT NULL DEFAULT 0.0,
    context_data JSONB DEFAULT '{}',
    started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    status VARCHAR(50) NOT NULL DEFAULT 'active'
);

-- Conversation exchanges table
CREATE TABLE conversation_exchanges (
    exchange_id VARCHAR(255) PRIMARY KEY,
    conversation_id VARCHAR(255) NOT NULL REFERENCES conversation_contexts(conversation_id),
    speaker_id VARCHAR(255) NOT NULL,
    message_text TEXT NOT NULL,
    message_type VARCHAR(100) NOT NULL,
    emotion VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- NPC personalities table
CREATE TABLE npc_personalities (
    npc_id VARCHAR(255) PRIMARY KEY,
    traits JSONB NOT NULL DEFAULT '{}',
    knowledge_areas JSONB NOT NULL DEFAULT '[]',
    speech_patterns JSONB NOT NULL DEFAULT '{}',
    relationships JSONB NOT NULL DEFAULT '{}',
    background_story TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Query logs table
CREATE TABLE query_logs (
    query_id VARCHAR(255) PRIMARY KEY,
    player_id VARCHAR(255) NOT NULL,
    query_text TEXT NOT NULL,
    query_type VARCHAR(100) NOT NULL,
    response_text TEXT,
    confidence FLOAT,
    sources JSONB DEFAULT '[]',
    context_data JSONB DEFAULT '{}',
    processing_time_ms INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_conversation_contexts_player_id ON conversation_contexts(player_id);
CREATE INDEX idx_conversation_contexts_npc_id ON conversation_contexts(npc_id);
CREATE INDEX idx_conversation_contexts_status ON conversation_contexts(status);
CREATE INDEX idx_conversation_exchanges_conversation_id ON conversation_exchanges(conversation_id);
CREATE INDEX idx_conversation_exchanges_created_at ON conversation_exchanges(created_at);
CREATE INDEX idx_query_logs_player_id ON query_logs(player_id);
CREATE INDEX idx_query_logs_query_type ON query_logs(query_type);
CREATE INDEX idx_query_logs_created_at ON query_logs(created_at);
```

### 5. Template System

#### 5.1 Conversation Templates (`templates/conversation/`)
```jinja2
<!-- npc_response.j2 -->
{% if npc_emotion %}[{{ npc_emotion }}] {% endif %}{{ npc_name }}: "{{ response_text }}"

{% if relationship_change %}
Your relationship with {{ npc_name }} has {{ "improved" if relationship_change > 0 else "deteriorated" }}.
{% endif %}

{% if new_information %}
You learned: {{ new_information }}
{% endif %}

<!-- query_response.j2 -->
{{ response_text }}

{% if sources %}
Sources: {{ sources | join(", ") }}
{% endif %}

{% if related_queries %}
You might also want to ask about:
{% for query in related_queries %}
- {{ query }}
{% endfor %}
{% endif %}
```

### 6. LLM Prompt Templates (`src/game_loop/llm/prompts/`)

#### 6.1 Query Processing Prompts
```text
# query_classification.txt
Classify the following player query into one of these categories:
- WORLD_INFO: Questions about the game world, lore, locations
- OBJECT_INFO: Questions about specific objects or items
- NPC_INFO: Questions about characters or NPCs
- LOCATION_INFO: Questions about current or other locations
- HELP: Requests for help or instructions
- STATUS: Requests for player status or progress
- INVENTORY: Questions about player's inventory
- QUEST_INFO: Questions about quests or objectives

Query: {query_text}
Context: {context}

Respond with just the category name.

# information_synthesis.txt
Based on the following information sources, provide a helpful answer to the player's query.

Query: {query_text}
Information Sources:
{information_sources}

Player Context:
- Location: {current_location}
- Recent Actions: {recent_actions}
- Active Quests: {active_quests}

Provide a natural, conversational response that directly answers the query using the available information. If information is incomplete, mention what aspects you're uncertain about.
```

#### 6.2 Conversation Prompts
```text
# npc_response_generation.txt
You are {npc_name}, an NPC in a text adventure game with the following personality:

Personality Traits: {personality_traits}
Background: {background_story}
Current Mood: {current_mood}
Relationship with Player: {relationship_level} (-1.0 to 1.0, where -1 is hostile, 0 is neutral, 1 is friendly)

Conversation Context:
{conversation_history}

Knowledge Areas: {knowledge_areas}
Current Location: {current_location}
Time/Setting: {time_context}

The player just said: "{player_message}"

Respond as this character would, considering:
1. Their personality and current mood
2. Their relationship with the player
3. What they would realistically know
4. Their speech patterns and mannerisms
5. The conversation context and history

Provide only the character's dialogue response, without quotes or character name.

# knowledge_extraction.txt
Analyze the following conversation and extract any new factual information that was revealed:

Conversation:
{conversation_history}

Extract information in these categories:
1. World/Lore information
2. Character relationships and backgrounds
3. Location descriptions or connections
4. Object information or properties
5. Quest or objective information
6. Historical events or timelines

For each piece of information, specify:
- Category
- Specific information learned
- Confidence level (high/medium/low)
- Source (which speaker revealed it)

Format as JSON with this structure:
{
  "extracted_information": [
    {
      "category": "world_lore",
      "information": "specific fact learned",
      "confidence": "high",
      "source": "npc_name"
    }
  ]
}
```

### 7. Testing Strategy

#### 7.1 Unit Tests (`tests/unit/core/query/`, `tests/unit/core/conversation/`)
- Query classification accuracy
- Information aggregation from multiple sources
- Conversation context tracking
- NPC personality expression
- Knowledge extraction from conversations
- Response generation quality

#### 7.2 Integration Tests (`tests/integration/conversation/`)
- End-to-end query processing
- Multi-turn conversation flows
- Knowledge persistence and retrieval
- Cross-system integration (quests, objects, etc.)
- Conversation state management

### 8. Configuration and Settings

#### 8.1 Conversation Configuration (`src/game_loop/config/conversation_config.py`)
```python
@dataclass
class ConversationConfig:
    """Configuration for conversation system."""
    max_conversation_history: int = 50
    relationship_decay_rate: float = 0.01
    default_npc_mood: str = "neutral"
    conversation_timeout_minutes: int = 30
    knowledge_extraction_threshold: float = 0.7
    response_generation_temperature: float = 0.8
    max_query_response_length: int = 500
```

## Testing and Verification

### Verification Criteria
1. **Query Processing**: Test queries about the game world with accurate, helpful responses
2. **NPC Conversations**: Verify NPCs maintain context over multiple exchanges and express personality
3. **Knowledge Updates**: Confirm new information is learned and stored through conversations
4. **Context Tracking**: Verify conversation history and relationship changes persist
5. **Integration**: Test query and conversation systems work with existing game systems

### Test Scenarios
1. **Information Queries**:
   - "What do you know about the ancient temple?"
   - "Tell me about the magical artifacts in this area"
   - "What can you tell me about the local history?"

2. **NPC Conversations**:
   - Multi-turn dialogue with context tracking
   - Relationship changes based on conversation choices
   - Knowledge sharing and information gathering

3. **System Integration**:
   - Quest information through conversations
   - Object information through queries
   - Location details through both systems

## Implementation Order

1. **Foundation** (Days 1-2):
   - Query and conversation models
   - Database schema and migration
   - Basic query processor structure

2. **Query System** (Days 3-4):
   - Information aggregator
   - Query classification and processing
   - Response generation with LLM

3. **Conversation System** (Days 5-7):
   - Conversation manager
   - NPC personality system
   - Context tracking and history

4. **Knowledge System** (Days 8-9):
   - Knowledge extraction from conversations
   - Information storage and retrieval
   - NPC knowledge updates

5. **Integration** (Days 10-11):
   - Command handlers
   - Game system integration
   - Template system

6. **Testing and Polish** (Days 12-14):
   - Unit and integration tests
   - Performance optimization
   - Documentation and examples

## Dependencies

### Required Components
- Semantic Search System (from Commit 16)
- Game State Management (from Commit 9)
- LLM Integration (Ollama client)
- Database Models and ORM (from Commit 10)
- Template System (from Commit 11)

### Integration Points
- Quest System (for quest-related queries)
- Object System (for object information)
- Location System (for location queries)
- Player State (for status queries)

## Success Metrics

1. **Query Accuracy**: >85% of queries receive relevant, helpful responses
2. **Conversation Quality**: NPCs maintain consistent personality and context
3. **Knowledge Retention**: Information learned in conversations is stored and retrievable
4. **Performance**: Query responses generated within 2 seconds
5. **Integration**: Smooth interaction with all existing game systems

This implementation provides a comprehensive query and conversation system that enhances player engagement through meaningful information gathering and NPC interactions while maintaining consistency with the existing game architecture.