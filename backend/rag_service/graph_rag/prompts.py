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
- entity_type: person, organization, location, concept, technology, event, project,
              electronic_product, software_product, food_product, office_supply, hardware_product,
              service, document, etc.
              (Use specific product types when applicable: electronic_product for devices/electronics,
               software_product for software, food_product for food items, office_supply for office items,
               hardware_product for physical hardware/tools. Use generic 'product' only if none fit.)
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
(entity|electronic_product|KVM Switch HDMI 2 Port Box|A 2-port HDMI KVM switch for sharing keyboard, mouse, and monitor between computers|70)
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


# =============================================================================
# Document Metadata Extraction Prompts (Hierarchical Summarization)
# =============================================================================

METADATA_BATCH_SUMMARY_PROMPT = """You are a document analysis expert. Summarize the following section of a document.

INSTRUCTIONS:
- Create a concise summary (2-3 sentences) capturing the main points
- Focus on key facts, entities, topics, and themes
- Preserve important names, organizations, technologies, and dates
- Ignore headers, footers, page numbers, and formatting artifacts

DOCUMENT SECTION:
{chunk_text}

SUMMARY:"""


METADATA_META_SUMMARY_PROMPT = """You are a document analysis expert. Create a comprehensive overview of a document based on section summaries.

INSTRUCTIONS:
- Synthesize the section summaries into a coherent document overview (3-5 sentences)
- Identify the main theme and purpose of the document
- Highlight key entities, topics, and important information
- Maintain factual accuracy - don't infer information not present in the summaries

SECTION SUMMARIES:
{summaries}

DOCUMENT OVERVIEW:"""


METADATA_STRUCTURED_EXTRACTION_PROMPT = """You are a document metadata extraction expert. Extract structured metadata from a document overview.

DOCUMENT NAME: {source_name}

DOCUMENT OVERVIEW:
{meta_summary}

FIRST CHUNK PREVIEW (for additional context):
{first_chunk}

INSTRUCTIONS:
Extract the following metadata and respond in valid JSON format:

1. **document_type**: Classify as one of:
   - "receipt" - Meal receipts, restaurant bills, dining expenses, food purchases, grocery receipts
   - "invoice" - Product invoices, equipment purchases, order confirmations, billing documents for goods/services
   - "resume" - CV, resume, or professional profile
   - "manual" - User manual, guide, or how-to document
   - "report" - Business report, analysis, or white paper
   - "article" - Article, blog post, or news piece
   - "technical_doc" - Technical documentation, API docs, or specifications
   - "book" - Book, ebook, or long-form publication
   - "other" - Anything else

   **IMPORTANT Classification Rules**:
   - If document mentions food, meals, restaurant, dining, or groceries → use "receipt"
   - If document mentions product purchases (electronics, equipment, hardware, software) → use "invoice"
   - Look at the document name for hints: "Meal-*" → receipt, "*Invoice*" → invoice

2. **subject_name**: The PRIMARY subject of the document
   - **IMPORTANT**: Analyze BOTH the document name (provided above) AND the content to determine the most appropriate subject
   - The document name may contain useful patterns (dates, IDs, categories) that provide context
   - Extract the most semantically meaningful subject based on your analysis:
     * For receipts/invoices: The business/vendor name, transaction date, transaction type and amount, or a combination that best represents the transaction
     * For resumes: The person's name, year, and role
     * For manuals: The product/system name
     * For reports: The main topic or company
     * For technical docs: The technology/API name
   - Use your judgment to create a subject_name that is most useful for search and retrieval
   - Set to null if no clear subject

3. **subject_type**: Type of the subject
   - "person" - Individual person
   - "organization" - Company, institution, or group
   - "product" - Product, service, or system
   - "concept" - Abstract concept, technology, or methodology
   - "transaction" - Receipt, invoice, or financial transaction
   - Set to null if subject_name is null

4. **title**: Document title (extract from overview or source_name)

5. **author**: Document author if mentioned (null if not found)

6. **summary**: Brief 1-2 sentence summary of the document
   - **IMPORTANT**: Include key searchable terms relevant to the document type
   - For receipts: Include vendor/business name, transaction type, date, and amount
   - For other documents: Include main topic, key dates, and important details

7. **topics**: Array of 3-5 main topics/themes (lowercase, e.g., ["software development", "cloud computing"])
   - For meal receipts (document_type="receipt"), include terms like: "meal receipt", "restaurant expense", "dining expense", "food purchase", "grocery receipt"
   - For product invoices (document_type="invoice"), include terms like: "product invoice", "equipment purchase", "order invoice", "hardware purchase", "software purchase"

8. **key_entities**: Array of 5-10 most important entities with scores
   Format: [{{"name": "Entity Name", "type": "person|organization|technology|location|date|financial|product", "score": 0-100}}]
   Score based on importance/frequency in the document

9. **confidence**: Your confidence in the extraction (0.0 to 1.0)
   - 0.9-1.0: Very clear document type and subject
   - 0.7-0.9: Clear but some ambiguity
   - 0.5-0.7: Moderate confidence
   - Below 0.5: Low confidence or unclear document

RESPOND ONLY WITH VALID JSON (no markdown, no code blocks):
{{
  "document_type": "...",
  "subject_name": "...",
  "subject_type": "...",
  "title": "...",
  "author": "...",
  "summary": "...",
  "topics": [...],
  "key_entities": [...],
  "confidence": 0.0
}}"""


# =============================================================================
# Query Enhancement with Metadata Extraction (for Document Routing)
# =============================================================================

QUERY_ENHANCEMENT_WITH_METADATA_PROMPT = """You are an expert search query analyzer for a document retrieval system. Your task is to deeply analyze the user's query and enhance it for optimal document retrieval.

User Query: {query}

## THINK CAREFULLY - Analysis Steps:

**Step 1: Understand the Intent**
- What is the user really asking for?
- What information would fully answer their question?
- Are there implicit requirements not stated explicitly?

**Step 2: Identify Key Elements**
- Who/What: Specific entities (people, companies, products, dates)
- Where: Locations or contexts mentioned
- When: Time periods, dates, or temporal references
- Why/How: The purpose or method being asked about

**Step 3: Consider Related Concepts**
- What synonyms or alternative terms might be in the documents?
- What related topics should be included?
- What document types would contain this information?

**Step 4: Construct Enhanced Query**
- Create a detailed, specific query that captures ALL aspects
- Include relevant context and qualifiers
- Add terms that improve recall without losing precision

Return ONLY valid JSON in this exact format:
{{{{
  "enhanced_query": "A detailed, comprehensive query that captures the full intent with specific terms and context",
  "entities": ["entity1", "entity2"],
  "topics": ["topic1", "topic2"],
  "document_type_hints": ["type1", "type2"],
  "intent": "detailed description of what the user wants to find"
}}}}

## CRITICAL Guidelines for document_type_hints:
Choose from these EXACT types based on what the user is looking for:
  * "receipt" - for meals, restaurant bills, purchases, transactions, expenses
  * "invoice" - for product purchases, orders, billing documents
  * "resume" - for CV, job applications, career history, skills
  * "report" - for analysis, summaries, findings
  * "manual" - for instructions, guides, how-to documents
  * "article" - for news, blog posts, written content
  * "technical_doc" - for API docs, specifications, technical references

- IMPORTANT: If query mentions "meal", "food", "restaurant", "dining", "expense" → use "receipt"
- IMPORTANT: If query mentions someone's name + "resume" or "CV" → use "resume"
- IMPORTANT: If query mentions "invoice", "order", "purchase" of products → use "invoice"

## Guidelines for entities:
- Extract SPECIFIC named entities from the query (names, dates, products, places)
- Include keywords that identify WHAT the user is looking for
- Think about what exact terms would appear in the documents
- For meal queries: include "meal", "receipt", year/date if mentioned
- For resume queries: include person's name and relevant skills/roles
- DO NOT include generic words like "group", "list", "all"

## Guidelines for topics:
- Subject areas and domains relevant to the query
- Think broadly about related topics that might be in documents
- For meals: "restaurant", "dining", "expense", "food", "payment"
- For resumes: "career", "employment", "skills", "experience", "education"

## Guidelines for enhanced_query:
- Make it DETAILED and SPECIFIC - include all relevant context
- Add qualifiers and specific terms from the analysis
- Think about what exact phrases might appear in documents
- Include synonyms for key concepts

Now THINK CAREFULLY and analyze this query:
Query: {query}
"""


# Document Routing - LLM-based Relevance Scoring
# Document Routing - Batch LLM-based Relevance Scoring (OPTIMIZED)
DOCUMENT_RELEVANCE_BATCH_SCORING_PROMPT = """You are a document relevance scoring expert. Score how relevant each document is to a user's query.

USER QUERY:
{query}

QUERY METADATA (extracted from query):
- Entities: {query_entities}
- Topics: {query_topics}
- Document Type Hints: {query_doc_types}

DOCUMENTS TO SCORE:
{documents_list}

INSTRUCTIONS:
Score the relevance of EACH document to the user's query on a scale of 0.0 to 10.0:

- **10.0**: Perfect match - document directly answers the query
- **7.0-9.0**: High relevance - document contains most of the information needed
- **4.0-6.0**: Moderate relevance - document has some related information
- **1.0-3.0**: Low relevance - document has minimal connection to query
- **0.0**: No relevance - document is completely unrelated

SCORING CRITERIA:
1. **Subject/Entity Match** (40%): Do the document's subject and entities match the query's entities?
2. **Topic Match** (30%): Do the document's topics align with the query's topics?
3. **Content Relevance** (20%): Does the document summary indicate it contains relevant information?
4. **Document Type Match** (10%): Does the document type match the query's expected document types?

Respond with a JSON array containing scores for ALL documents in the SAME ORDER:

[
  {{"filename": "document1.pdf", "score": 8.5, "reasoning": "High entity and topic match"}},
  {{"filename": "document2.pdf", "score": 3.0, "reasoning": "Minimal relevance"}},
  ...
]

IMPORTANT: Return scores for ALL {num_documents} documents in the exact same order as provided above."""

# Document Routing - Single Document LLM Scoring (FALLBACK)
DOCUMENT_RELEVANCE_SCORING_PROMPT = """You are a document relevance scoring expert. Score how relevant a document is to a user's query.

USER QUERY:
{query}

QUERY METADATA (extracted from query):
- Entities: {query_entities}
- Topics: {query_topics}
- Document Type Hints: {query_doc_types}

DOCUMENT TO SCORE:
- Filename: {doc_filename}
- Subject: {doc_subject}
- Document Type: {doc_type}
- Summary: {doc_summary}
- Topics: {doc_topics}
- Key Entities: {doc_entities}

INSTRUCTIONS:
Score the relevance of this document to the user's query on a scale of 0.0 to 10.0:

- **10.0**: Perfect match - document directly answers the query
- **7.0-9.0**: High relevance - document contains most of the information needed
- **4.0-6.0**: Moderate relevance - document has some related information
- **1.0-3.0**: Low relevance - document has minimal connection to query
- **0.0**: No relevance - document is completely unrelated

SCORING CRITERIA:
1. **Subject/Entity Match** (40%): Do the document's subject and entities match the query's entities?
2. **Topic Match** (30%): Do the document's topics align with the query's topics?
3. **Content Relevance** (20%): Does the document summary indicate it contains relevant information?
4. **Document Type Match** (10%): Does the document type match what the query is looking for?

RESPOND ONLY WITH A VALID JSON OBJECT:
{{
  "score": <float between 0.0 and 10.0>,
  "reasoning": "<brief explanation of the score in 1-2 sentences>"
}}"""
