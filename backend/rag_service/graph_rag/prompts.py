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
Following the Graph-R1 paper design with 3 retrieval modes.

Analyze the following user query and determine:
1. The best retrieval mode
2. An enhanced version of the query for better retrieval

MODES (from Graph-R1 paper):

- LOCAL: Entity-focused retrieval
  Use for: Simple entity definition/identification queries
  Examples: "Who is John Smith?", "What is machine learning?", "Tell me about Company X"
  Use when: Query asks "who/what IS something" (identity/definition only)

- GLOBAL: Relationship-focused retrieval
  Use for: Queries about relationships BETWEEN two or more entities
  Examples: "How does X relate to Y?", "What is the connection between A and B?",
            "Compare X and Y", "What's the difference between A and B?"
  Use when: Query explicitly mentions two entities and asks about their relationship

- HYBRID: Combined entity and relationship retrieval (DEFAULT)
  Use for: Complex queries about mechanisms, processes, or multi-faceted topics
  Examples: "How does X work?", "Explain the authentication system",
            "What are the components of X?", "How is X implemented?",
            "List all products", "When was X created?", "How many employees?"
  Use when: Query asks about mechanisms, processes, lists, counts, or any complex topic
  IMPORTANT: This is the DEFAULT mode - use when in doubt

DECISION GUIDE:
- "What IS X?" (simple definition) → LOCAL
- "How does X RELATE to Y?" (cross-entity relationship) → GLOBAL
- Everything else (mechanisms, lists, counts, explanations) → HYBRID

USER QUERY: {query}

Respond in JSON format:
{{"mode": "LOCAL|GLOBAL|HYBRID", "enhanced_query": "improved query for retrieval"}}
"""

# =============================================================================
# Entity Extraction Prompts
# =============================================================================

ENTITY_EXTRACTION_PROMPT = """You are an expert at extracting KEY entities and relationships from text for knowledge graph construction.

IMPORTANT: Extract ONLY the most important and salient entities. Focus on quality over quantity.

Extract entities that are:
✓ Central to understanding the document's main topics
✓ Proper nouns (people, organizations, locations, products, technologies, projects)
✓ Key concepts that define the domain or subject matter
✓ Referenced multiple times or in important contexts
✓ Have significant relationships with other entities

DO NOT extract:
✗ Common nouns or generic terms (e.g., "system", "process", "method")
✗ Entities mentioned only once in passing without context
✗ Low-importance supporting details or examples
✗ Adjectives or descriptive phrases
✗ Pronouns or references without clear antecedents

TEXT:
{text}

Extract entities and relationships in the following format:

For entities, use:
(entity|<entity_type>|<entity_name>|<entity_description>|<importance_score>)

Where:
- entity_type: person, organization, location, concept, technology, event, product, project, etc.
- entity_name: The name of the entity (use proper capitalization)
- entity_description: A concise description of the entity and its significance
- importance_score: 1-100 indicating importance (be selective - use high scores for truly important entities)
  * 80-100: Critical entities (main subjects, key people/organizations, core concepts)
  * 60-79: Important entities (significant technologies, major concepts, notable relationships)
  * 40-59: Moderate entities (supporting details with some significance)
  * Below 40: Skip these - not important enough for the knowledge graph

For relationships, use:
(relationship|<source_entity>|<target_entity>|<relationship_description>|<keywords>|<weight>)

Where:
- source_entity: Name of the source entity (must match an extracted entity name)
- target_entity: Name of the target entity (must match an extracted entity name)
- relationship_description: Clear description of how they are related
- keywords: Comma-separated keywords describing the relationship type
- weight: 1-10 indicating relationship strength and importance

QUALITY GUIDELINES:
- Aim for 5-15 high-quality entities per chunk (not 50+)
- Only extract relationships between entities you've already extracted
- Use high importance scores (60+) for entities worth storing
- Be selective - a smaller, high-quality graph is better than a large, noisy one

Example output:
(entity|person|John Smith|CEO and founder of Acme Corporation, leading AI innovation initiatives|90)
(entity|organization|Acme Corporation|Technology company specializing in artificial intelligence and machine learning solutions|85)
(entity|technology|Machine Learning|Core technology used by Acme Corporation for product development|75)
(relationship|John Smith|Acme Corporation|John Smith founded and currently leads Acme Corporation as CEO|CEO, founder, leadership|9)
(relationship|Acme Corporation|Machine Learning|Acme Corporation specializes in developing machine learning solutions|specialization, technology, development|8)

Now extract ONLY the key entities and relationships from the text above:
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

