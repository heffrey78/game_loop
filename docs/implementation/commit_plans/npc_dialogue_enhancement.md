# NPC and Dialogue Enhancement Plan

## Overview

Enhance the existing `ConversationCommandHandler` through composition and additional components to create more engaging, personality-driven NPC interactions with memory and context awareness.

## Current Issues from User Testing

- NPC responses somewhat generic: "Good evening, sir/madam. How may I assist you?"
- Lack of personality consistency in dialogue
- No memory of previous interactions
- Limited contextual awareness of location and situation
- Missing name recognition and relationship tracking

## Enhancement Strategy: Composition over Replacement

Rather than replacing the existing `ConversationCommandHandler`, enhance it with specialized components that can be plugged in to provide richer functionality.

## Core Enhancement Components

### 1. Personality Engine

**Purpose**: Provide consistent personality-driven responses for each NPC archetype.

```python
class NPCPersonalityEngine:
    """Generate personality-consistent dialogue based on NPC archetype."""
    
    PERSONALITY_PROFILES = {
        'security_guard': {
            'traits': ['dutiful', 'cautious', 'professional'],
            'speech_patterns': ['formal', 'direct', 'procedural'],
            'concerns': ['security', 'protocol', 'identification'],
            'greeting_style': 'professional_inquiry',
            'knowledge_focus': ['building_layout', 'security_procedures', 'personnel']
        },
        'scholar': {
            'traits': ['curious', 'verbose', 'analytical'],
            'speech_patterns': ['academic', 'detailed', 'questioning'],
            'concerns': ['knowledge', 'research', 'accuracy'],
            'greeting_style': 'intellectual_engagement',
            'knowledge_focus': ['books', 'research', 'history', 'theories']
        },
        'merchant': {
            'traits': ['practical', 'friendly', 'profit-minded'],
            'speech_patterns': ['persuasive', 'friendly', 'transactional'],
            'concerns': ['trade', 'value', 'customers'],
            'greeting_style': 'commercial_welcome',
            'knowledge_focus': ['items', 'prices', 'market_conditions']
        }
    }
    
    def generate_personality_response(self, npc_archetype, context, topic=None):
        """Generate response based on NPC personality profile."""
        profile = self.PERSONALITY_PROFILES.get(npc_archetype, self._get_default_profile())
        
        # Build response based on personality traits
        response_style = self._determine_response_style(profile, context, topic)
        base_response = self._generate_base_response(profile, context, topic)
        
        # Apply personality modifiers
        personality_response = self._apply_personality_traits(base_response, profile['traits'])
        speech_response = self._apply_speech_patterns(personality_response, profile['speech_patterns'])
        
        return speech_response
    
    def _generate_base_response(self, profile, context, topic):
        """Generate base response content."""
        if topic:
            return self._generate_topic_response(profile, context, topic)
        else:
            return self._generate_greeting_response(profile, context)
    
    def _generate_greeting_response(self, profile, context):
        """Generate greeting based on NPC type and context."""
        greeting_templates = {
            'professional_inquiry': [
                "Good {time_of_day}. I'll need to see some identification.",
                "Please state your business here and show your credentials.",
                "This is a restricted area. Do you have authorization?"
            ],
            'intellectual_engagement': [
                "Ah, another seeker of knowledge! What brings you to these halls of learning?",
                "Welcome, fellow scholar. Are you here for research or general inquiry?",
                "Fascinating to see someone else exploring these archives."
            ],
            'commercial_welcome': [
                "Welcome! I have many fine wares that might interest you.",
                "Good {time_of_day}! Looking for anything in particular today?",
                "Step right up! I have exactly what you need."
            ]
        }
        
        greeting_style = profile.get('greeting_style', 'neutral')
        templates = greeting_templates.get(greeting_style, ["Hello there."])
        
        # Select template and fill in context
        template = random.choice(templates)
        return template.format(
            time_of_day=context.get('time_of_day', 'day'),
            player_name=context.get('player_name', '')
        )
```

### 2. Conversation Memory System

**Purpose**: Track conversation history and relationship development.

```python
class ConversationMemoryManager:
    """Manage NPC memory of conversations and relationships."""
    
    def __init__(self):
        self.conversation_history = {}  # npc_id -> conversation records
        self.relationship_scores = {}   # (npc_id, player_id) -> relationship data
        self.topic_knowledge = {}       # npc_id -> topics discussed
    
    async def record_conversation(self, npc_id, player_id, topic, response, context):
        """Record a conversation for future reference."""
        conversation_key = f"{npc_id}_{player_id}"
        
        if conversation_key not in self.conversation_history:
            self.conversation_history[conversation_key] = []
        
        self.conversation_history[conversation_key].append({
            'timestamp': datetime.now(),
            'topic': topic,
            'response': response,
            'context': context,
            'location': context.get('location_id')
        })
        
        # Update relationship based on interaction
        await self._update_relationship(npc_id, player_id, topic, context)
        
        # Track topic knowledge
        self._update_topic_knowledge(npc_id, topic)
    
    def get_conversation_context(self, npc_id, player_id):
        """Get previous conversation context for reference."""
        conversation_key = f"{npc_id}_{player_id}"
        history = self.conversation_history.get(conversation_key, [])
        
        if not history:
            return {'is_first_meeting': True, 'previous_topics': []}
        
        return {
            'is_first_meeting': False,
            'previous_topics': [conv['topic'] for conv in history[-5:]],
            'last_interaction': history[-1]['timestamp'],
            'total_conversations': len(history),
            'relationship_level': self._get_relationship_level(npc_id, player_id)
        }
    
    def _get_relationship_level(self, npc_id, player_id):
        """Determine relationship level based on interaction history."""
        rel_key = f"{npc_id}_{player_id}"
        rel_data = self.relationship_scores.get(rel_key, {'score': 0})
        
        score = rel_data['score']
        if score >= 50:
            return 'trusted_friend'
        elif score >= 20:
            return 'friendly_acquaintance'
        elif score >= 0:
            return 'neutral'
        else:
            return 'suspicious'
```

### 3. Contextual Knowledge Engine

**Purpose**: Provide NPCs with location-specific and situational knowledge.

```python
class NPCKnowledgeEngine:
    """Manage NPC knowledge about locations, objects, and situations."""
    
    def __init__(self, semantic_search_service):
        self.semantic_search = semantic_search_service
        self.knowledge_cache = {}
    
    async def get_npc_knowledge(self, npc_archetype, location_id, topic=None):
        """Get relevant knowledge for NPC based on their role and location."""
        knowledge_key = f"{npc_archetype}_{location_id}_{topic or 'general'}"
        
        if knowledge_key in self.knowledge_cache:
            return self.knowledge_cache[knowledge_key]
        
        # Gather contextual knowledge
        location_knowledge = await self._get_location_knowledge(npc_archetype, location_id)
        role_knowledge = await self._get_role_specific_knowledge(npc_archetype, topic)
        situational_knowledge = await self._get_situational_knowledge(location_id, topic)
        
        combined_knowledge = {
            'location': location_knowledge,
            'role': role_knowledge,
            'situation': situational_knowledge
        }
        
        self.knowledge_cache[knowledge_key] = combined_knowledge
        return combined_knowledge
    
    async def _get_location_knowledge(self, npc_archetype, location_id):
        """Get what this NPC type would know about this location."""
        location_details = await self._get_location_details(location_id)
        
        # Filter knowledge based on NPC role
        if npc_archetype == 'security_guard':
            return {
                'layout': location_details.get('exits', {}),
                'security_features': location_details.get('security', {}),
                'access_restrictions': location_details.get('restrictions', []),
                'patrol_routes': location_details.get('routes', [])
            }
        elif npc_archetype == 'scholar':
            return {
                'research_materials': location_details.get('books', []),
                'historical_significance': location_details.get('history', ''),
                'academic_resources': location_details.get('resources', [])
            }
        
        return location_details
```

### 4. Enhanced Conversation Handler Integration

**Purpose**: Integrate new components with existing ConversationCommandHandler.

```python
class EnhancedConversationCommandHandler(ConversationCommandHandler):
    """Enhanced version of existing handler with personality and memory."""
    
    def __init__(self, console, state_manager):
        super().__init__(console, state_manager)
        
        # Add enhancement components
        self.personality_engine = NPCPersonalityEngine()
        self.memory_manager = ConversationMemoryManager()
        self.knowledge_engine = NPCKnowledgeEngine(semantic_search_service)
        
        # Keep existing conversation history for compatibility
        # New components supplement rather than replace
    
    async def _generate_npc_response(self, npc, context, topic=None):
        """Enhanced response generation using new components."""
        
        # Get NPC information
        npc_id = getattr(npc, 'id', getattr(npc, 'name', 'unknown'))
        npc_archetype = getattr(npc, 'archetype', 'generic')
        player_id = context.get('player_id', 'player')
        
        # Gather enhanced context
        conversation_context = self.memory_manager.get_conversation_context(npc_id, player_id)
        npc_knowledge = await self.knowledge_engine.get_npc_knowledge(
            npc_archetype, context.get('location_id'), topic
        )
        
        # Build enhanced context
        enhanced_context = {
            **context,
            **conversation_context,
            'knowledge': npc_knowledge,
            'npc_archetype': npc_archetype
        }
        
        # Generate personality-driven response
        if not conversation_context['is_first_meeting']:
            response = self._generate_return_visitor_response(npc, enhanced_context, topic)
        else:
            response = self.personality_engine.generate_personality_response(
                npc_archetype, enhanced_context, topic
            )
        
        # Record conversation for future reference
        await self.memory_manager.record_conversation(
            npc_id, player_id, topic or 'general', response, enhanced_context
        )
        
        return response
    
    def _generate_return_visitor_response(self, npc, context, topic):
        """Generate response for returning visitors with memory references."""
        npc_name = getattr(npc, 'name', 'Someone')
        player_name = context.get('player_name', 'you')
        previous_topics = context.get('previous_topics', [])
        relationship_level = context.get('relationship_level', 'neutral')
        
        # Reference previous interactions
        if 'general' in previous_topics:
            greeting = f"Good to see you again, {player_name}!"
        else:
            greeting = f"Hello again, {player_name}."
        
        # Add relationship-specific context
        if relationship_level == 'trusted_friend':
            greeting += " How can I help my friend today?"
        elif relationship_level == 'friendly_acquaintance':
            greeting += " What brings you back?"
        elif relationship_level == 'suspicious':
            greeting += " You again... what do you want this time?"
        
        return greeting
```

## Implementation Strategy

### Phase 1: Personality Engine (Week 1)
1. Implement `NPCPersonalityEngine` with basic personality profiles
2. Create personality-specific response templates
3. Integrate with existing `ConversationCommandHandler`
4. Test with existing NPCs (security guard, etc.)

### Phase 2: Memory System (Week 2)  
1. Implement `ConversationMemoryManager` with history tracking
2. Add relationship scoring system
3. Create database schema for conversation persistence
4. Add memory references to NPC responses

### Phase 3: Knowledge Integration (Week 3)
1. Implement `NPCKnowledgeEngine` with role-specific knowledge
2. Integrate with semantic search for contextual information
3. Add location-specific knowledge for NPCs
4. Create topic-specific response generation

### Phase 4: Advanced Features (Week 4)
1. Add name recognition and player identification
2. Implement emotional state tracking for NPCs
3. Create conversation trees for complex topics
4. Add NPC-initiated conversations based on context

## Success Criteria

### Immediate Goals
- NPCs use player names in conversation
- Security guard responses feel professional and security-focused
- Conversations reference previous interactions
- NPC responses vary based on relationship level

### Advanced Goals
- NPCs provide location-specific information based on their role
- Emotional relationships develop over multiple interactions
- NPCs remember specific topics discussed previously
- Conversation quality feels natural and engaging

## Integration with Existing Systems

### Maintaining Compatibility
- Existing `ConversationCommandHandler` functionality preserved
- New features added through composition, not replacement
- Backward compatibility with current NPC data structures
- Gradual migration path for enhanced features

### Database Integration
```sql
-- Add tables for enhanced conversation features
CREATE TABLE npc_conversations (
    id UUID PRIMARY KEY,
    npc_id UUID NOT NULL,
    player_id UUID NOT NULL,
    topic VARCHAR(255),
    response_text TEXT,
    context_data JSONB,
    location_id UUID,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE npc_relationships (
    id UUID PRIMARY KEY,
    npc_id UUID NOT NULL,
    player_id UUID NOT NULL,
    relationship_score INTEGER DEFAULT 0,
    relationship_level VARCHAR(50),
    last_interaction TIMESTAMP,
    total_interactions INTEGER DEFAULT 0
);
```

This enhancement approach builds upon the existing conversation system while adding the depth and personality that make NPC interactions memorable and engaging.