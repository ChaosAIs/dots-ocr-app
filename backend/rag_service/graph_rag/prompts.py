"""
LLM Prompts for GraphRAG.

This module contains all prompts used for:
- Query mode detection
- Entity extraction
- Relationship extraction
- Multi-turn reasoning
"""

# =============================================================================
# Query Mode Detection Prompts
# =============================================================================

QUERY_MODE_DETECTION_PROMPT = """You are a query analyzer for a knowledge graph RAG system.

Analyze the following user query and determine:
1. The best retrieval mode
2. An enhanced version of the query for better retrieval

MODES:
- LOCAL: Use when the query asks about a specific entity (person, organization, concept)
  Examples: "Who is John Smith?", "What is machine learning?", "Tell me about Company X"

- GLOBAL: Use when the query asks about relationships between entities
  Examples: "How does X relate to Y?", "What is the connection between A and B?"

- HYBRID: Use when the query requires both entity information and relationships
  Examples: "Explain the authentication system and how it connects to the database"

- NAIVE: Use for simple factual lookups or when graph context isn't needed
  Examples: "What is the date of the meeting?", "List all files"

USER QUERY: {query}

Respond in JSON format:
{{"mode": "LOCAL|GLOBAL|HYBRID|NAIVE", "enhanced_query": "improved query for retrieval"}}
"""

# =============================================================================
# Entity Extraction Prompts
# =============================================================================

ENTITY_EXTRACTION_PROMPT = """You are an expert at extracting entities and relationships from text.

Given the following text, extract all entities and relationships.

TEXT:
{text}

Extract entities and relationships in the following format:

For entities, use:
(entity|<entity_type>|<entity_name>|<entity_description>|<importance_score>)

Where:
- entity_type: person, organization, location, concept, technology, event, etc.
- entity_name: The name of the entity
- entity_description: A brief description of the entity
- importance_score: 1-100 indicating how important this entity is in the text

For relationships, use:
(relationship|<source_entity>|<target_entity>|<relationship_description>|<keywords>|<weight>)

Where:
- source_entity: Name of the source entity
- target_entity: Name of the target entity
- relationship_description: Description of how they are related
- keywords: Comma-separated keywords describing the relationship
- weight: 1-10 indicating relationship strength

Example output:
(entity|person|John Smith|CEO of Acme Corporation|90)
(entity|organization|Acme Corporation|Technology company founded in 2010|85)
(relationship|John Smith|Acme Corporation|John Smith is the CEO and founder|CEO, founder, leadership|9)

Now extract all entities and relationships from the text above:
"""

ENTITY_CONTINUE_EXTRACTION_PROMPT = """MANY knowledge fragments with entities and relationships were missed in the last extraction.

Please review the original text again and add any additional entities and relationships that were not captured.

Use the same format:
(entity|<entity_type>|<entity_name>|<entity_description>|<importance_score>)
(relationship|<source_entity>|<target_entity>|<relationship_description>|<keywords>|<weight>)

Add the missing entities and relationships below:
"""

ENTITY_CHECK_CONTINUE_PROMPT = """Please check whether the extracted knowledge fragments cover all the important entities and relationships in the given text.

Answer with only YES or NO:
- YES: There are still important entities or relationships that were missed
- NO: The extraction is complete and covers all important information

Answer:"""

# =============================================================================
# Multi-Turn Reasoning Prompts
# =============================================================================

MULTI_TURN_REASONING_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using a knowledge base.

When answering, follow this process:
1. Think about what information you need in <think>...</think>
2. If you need to search the knowledge base, use: <query>{{"query": "your search"}}</query>
3. When you have enough information, provide your answer in <answer>...</answer>

Rules:
- Always show your reasoning in <think> tags
- You can search multiple times if needed
- Only provide <answer> when you are confident
- Keep searches focused and specific

Example:
<think>
The user asks about X. I need to find information about Y first.
</think>
<query>{{"query": "information about Y"}}</query>

[After receiving knowledge]

<think>
Now I know about Y. I can answer the question.
</think>
<answer>
Based on the knowledge base, X is...
</answer>
"""

KNOWLEDGE_INJECTION_TEMPLATE = """<knowledge>
{knowledge}
</knowledge>

Continue your reasoning based on this knowledge."""

# =============================================================================
# Entity Summary Prompts
# =============================================================================

ENTITY_SUMMARY_PROMPT = """Summarize the following entity information into a concise description.

Entity Name: {entity_name}
Entity Type: {entity_type}
Descriptions from different sources:
{descriptions}

Provide a single, comprehensive summary (max 200 words):
"""

RELATIONSHIP_SUMMARY_PROMPT = """Summarize the relationship between these entities.

Source Entity: {source_entity}
Target Entity: {target_entity}
Relationship descriptions from different sources:
{descriptions}

Provide a single, comprehensive summary of their relationship (max 100 words):
"""

