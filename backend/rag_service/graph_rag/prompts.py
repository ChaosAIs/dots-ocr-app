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
   - "receipt" - Receipt, invoice, or transaction record
   - "resume" - CV, resume, or professional profile
   - "manual" - User manual, guide, or how-to document
   - "report" - Business report, analysis, or white paper
   - "article" - Article, blog post, or news piece
   - "technical_doc" - Technical documentation, API docs, or specifications
   - "book" - Book, ebook, or long-form publication
   - "other" - Anything else

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
   - For receipts, include terms like: "meal receipt", "restaurant expense", "credit card transaction", "purchase invoice", "ticket invoice", "grocery receipt", "utility bill", "transportation ticket", "event ticket", "parking receipt", "medical receipt", "service receipt", "rental receipt", "membership receipt", "subscription receipt", "donation receipt", "tax receipt", "warranty receipt", "guarantee receipt", "refund receipt", "return receipt", "cancellation receipt", "adjustment receipt", "correction receipt", "dispute receipt", "reversal receipt"

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

QUERY_ENHANCEMENT_WITH_METADATA_PROMPT = """You are a search query analyzer. Your task is to:
1. Enhance the user's query with more context and details for better vector search
2. Extract structured metadata for intelligent document routing

User Query: {query}

Analyze the query and return JSON with two parts:

**Part 1: Enhanced Query**
- Expand abbreviations and implicit concepts
- Add related terms and synonyms
- Make implicit entities explicit (e.g., "Felix" → "Felix Yang")
- Add context that would help find relevant information
- Keep it natural and readable
- **CRITICAL FOR DATES - ALWAYS NORMALIZE**: When the query mentions dates or months:
  - ALWAYS expand abbreviated month names to full names (Oct → October)
  - ALWAYS add the ISO year-month format in parentheses
  - ALWAYS include multiple date format variations
  - Examples:
    * "2025 Oct" → "October 2025 (2025-10, 10/2025)"
    * "Oct 2025" → "October 2025 (2025-10, 10/2025)"
    * "meals in Oct" → "meals in October (month 10)"
    * "do we have meal in 2025 oct?" → "Do we have any meals or receipts from October 2025 (2025-10, 10/2025, month 10)?"
  - This is CRITICAL because documents contain dates in various formats (10/14/2025, 2025-10-14, October 2025)
  - The enhanced query MUST include all these format variations to ensure matching

**Part 2: Metadata for Document Routing**
- **entities**: Specific named entities (people, organizations, products, technologies)
  - Use full names when possible
  - Include both explicit and implicit entities
  - Examples: "Felix Yang", "AWS", "Microsoft Azure", "Docker", "BDO"

- **topics**: Subject areas and domains
  - Be specific and granular
  - Include both broad and narrow topics
  - Examples: "cloud computing", "microservices architecture", "resume experience", "work history"

- **document_type_hints**: Types of documents likely to contain this information
  - Options: "resume", "manual", "report", "article", "technical_doc", "book", "other"
  - Can include multiple types

- **intent**: Brief description of what the user wants to find

Return ONLY valid JSON in this exact format:
{{{{
  "enhanced_query": "your enhanced query text here",
  "entities": ["entity1", "entity2", ...],
  "topics": ["topic1", "topic2", ...],
  "document_type_hints": ["type1", "type2"],
  "intent": "brief intent description"
}}}}

Examples:

Example 1:
Query: "What does Felix know about cloud?"
Response:
{{{{
  "enhanced_query": "What cloud computing technologies, platforms, and architectures does Felix Yang have experience with? Include AWS, Azure, GCP, cloud-native development, containerization, Kubernetes, serverless technologies, and cloud infrastructure management.",
  "entities": ["Felix Yang", "AWS", "Azure", "GCP", "Kubernetes"],
  "topics": ["cloud computing", "cloud platforms", "cloud architecture", "containerization", "serverless computing", "cloud-native development"],
  "document_type_hints": ["resume", "technical_doc"],
  "intent": "Find Felix Yang's cloud technology experience and expertise"
}}}}

Example 2:
Query: "How do microservices communicate?"
Response:
{{{{
  "enhanced_query": "How do microservices communicate with each other? What communication patterns, protocols, and technologies are used for inter-service communication in microservices architecture? Include REST APIs, message queues, event-driven architecture, gRPC, and service mesh.",
  "entities": ["REST", "gRPC", "message queues", "service mesh"],
  "topics": ["microservices architecture", "inter-service communication", "distributed systems", "API design", "event-driven architecture"],
  "document_type_hints": ["technical_doc", "manual", "article"],
  "intent": "Understand microservices communication patterns and technologies"
}}}}

Example 3:
Query: "Felix's work at BDO"
Response:
{{{{
  "enhanced_query": "What work experience, projects, and responsibilities did Felix Yang have at BDO? What technologies, methodologies, and achievements were involved in his role at BDO?",
  "entities": ["Felix Yang", "BDO"],
  "topics": ["work experience", "professional background", "projects", "career history"],
  "document_type_hints": ["resume"],
  "intent": "Find Felix Yang's work experience and projects at BDO"
}}}}

Example 4:
Query: "Tell me about the authentication system"
Response:
{{{{
  "enhanced_query": "Explain the authentication system architecture, implementation, and components. How does user authentication work? What authentication methods, protocols, and security measures are used?",
  "entities": ["authentication", "security"],
  "topics": ["authentication systems", "security architecture", "user authentication", "access control"],
  "document_type_hints": ["technical_doc", "manual", "article"],
  "intent": "Understand the authentication system design and implementation"
}}}}

Example 5 (DATE QUERY - CRITICAL):
Query: "do we have meal in 2025 oct?"
Response:
{{{{
  "enhanced_query": "Do we have any meals, receipts, or invoices from October 2025 (2025-10, 10/2025, month 10)? Show me meal records from 2025-10.",
  "entities": [],
  "topics": ["meal records", "receipts", "invoices", "October 2025"],
  "document_type_hints": ["receipt", "invoice"],
  "intent": "Find meal receipts or records from October 2025"
}}}}

Now analyze this query:
Query: {query}
"""


# Document Routing - LLM-based Relevance Scoring
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
