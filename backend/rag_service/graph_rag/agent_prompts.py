"""
Agent prompts for Graph-R1 style iterative reasoning.

These prompts guide the LLM through the think-query-retrieve-answer loop.
Enhanced with Universal Deep Reasoning Framework for domain-agnostic analysis.
"""

# =============================================================================
# ENHANCED AGENT THINK PROMPT - Universal Deep Reasoning Framework
# =============================================================================
AGENT_THINK_PROMPT = """## Current State
- **Original Question**: {question}
- **Reasoning Step**: {step}/{max_steps}
- **Previous Queries**: {previous_queries}
- **Retrieved Knowledge So Far**:
{retrieved_knowledge}

---

## STAGE 1: QUESTION DECOMPOSITION
Analyze the question to identify its components:

**1.1 Information Type Sought** (check all that apply):
- [ ] Specific facts/values (names, dates, numbers, identifiers)
- [ ] Descriptions/definitions (what is X, explain Y)
- [ ] Lists/collections (all items matching criteria)
- [ ] Comparisons (differences, similarities)
- [ ] Processes/procedures (how does X work, steps)
- [ ] Causes/effects (why did X happen, what results)
- [ ] Aggregations (count, sum, average, total)

**1.2 Target Entities**:
- Primary entities: [List the main subjects of the question]
- Secondary entities: [List related/supporting entities]

**1.3 Constraints**:
- Temporal: [Date range or time period, if any]
- Scope: [Specific documents or categories, if any]
- Conditions: [Filters or criteria, if any]

---

## STAGE 2: EVIDENCE MAPPING
For each question component, assess the retrieved knowledge:

| Component | Evidence Found | Status | Confidence |
|-----------|----------------|--------|------------|
| [Component 1] | [Specific quote/data from chunks] | FOUND/PARTIAL/MISSING | HIGH/MED/LOW |
| [Component 2] | [Specific quote/data from chunks] | FOUND/PARTIAL/MISSING | HIGH/MED/LOW |
| [Add more rows as needed] | | | |

**Evidence Quality Notes**:
- EXPLICIT: Directly stated in documents
- IMPLICIT: Can be inferred from context
- CONFLICTING: Multiple sources disagree

---

## STAGE 3: GAP ANALYSIS
Identify what information is missing:

**BLOCKING GAPS** (Cannot answer without these):
- [ ] Missing core entity: [description]
- [ ] Missing critical attribute: [description]
- [ ] Missing key relationship: [description]

**QUALITY GAPS** (Would improve answer):
- [ ] Incomplete coverage: [may have more instances]
- [ ] Low confidence: [single source only]
- [ ] Unresolved ambiguity: [multiple interpretations]

**Priority Gap to Address**: [Identify the single most important gap]

---

## STAGE 4: CROSS-REFERENCE CHECK
Connect information across sources:

**Corroborated Facts** (multiple sources agree):
- [Fact] - confirmed by [Source 1, Source 2]

**Single-Source Facts** (note limitation):
- [Fact] - from [Source] only

**Potential Conflicts**:
- [Conflict description and how to resolve]

---

## STAGE 5: DECISION

**Readiness Assessment**:
- Core entities found: [YES/NO]
- Critical attributes found: [YES/NO]
- Can perform required operation: [YES/NO]
- Blocking gaps remaining: [COUNT]
- Overall readiness: [0-100]%

---

## CRITICAL: Response Format Requirements

Based on your readiness assessment, respond with EXACTLY ONE of these formats:

**IF readiness >= 75% (sufficient to answer):**
<answer>
## Direct Answer
[Clear, concise response to the exact question]

## Evidence
Based on the retrieved documents:
- **[Source 1]**: "[relevant excerpt]"
- **[Source 2]**: "[relevant excerpt]"

## Details
[Structured breakdown appropriate to the question type:
- For LISTS: Numbered/bulleted items
- For COMPARISONS: Side-by-side or table format
- For AGGREGATIONS: Itemized with totals
- For PROCESSES: Step-by-step sequence]

## Confidence & Notes
- Confidence: [HIGH/MEDIUM/LOW]
- Based on: [N] source(s)
- Limitations: [Any caveats or missing information]
</answer>

**IF readiness < 75% (need more information):**
<query>
[Targeted search query that:
- Addresses the highest priority gap identified above
- Uses DIFFERENT keywords than previous queries: {previous_queries}
- Follows one of these strategies:
  * ENTITY SEARCH: "[entity name] [entity type] [context]"
  * ATTRIBUTE SEARCH: "[known entity] [missing attribute] details"
  * RELATIONSHIP SEARCH: "[Entity A] [Entity B] connection"
  * EXPANSION SEARCH: "[category] [synonyms] [alternatives]"
  * TEMPORAL SEARCH: "[topic] [time period] [event type]"
  * SOURCE SEARCH: "[topic] [unexplored document type]"]
</query>

## IMPORTANT RULES:
1. Output ONLY <answer>...</answer> OR <query>...</query> - not both
2. Do NOT include any text outside the tags
3. If this is the final step ({step}/{max_steps}), you MUST provide an <answer> with available information
4. Never repeat a previous query - always use different keywords

Your response:"""


# =============================================================================
# ENHANCED INITIAL QUERY PROMPT
# =============================================================================
INITIAL_QUERY_PROMPT = """## Question
{question}

## Task
Generate an effective initial search query to find relevant knowledge.

## Query Construction Guidelines

**Step 1: Identify Key Elements**
- Main subject/entity: [What is the question primarily about?]
- Action/relationship: [What action, property, or connection is being asked about?]
- Context clues: [Time period, location, category, or other constraints?]

**Step 2: Select Query Strategy**
Choose the most appropriate strategy:

1. **ENTITY-FOCUSED**: For questions about specific things/people/organizations
   → "[entity name] [entity type] [descriptive terms]"

2. **RELATIONSHIP-FOCUSED**: For questions about connections between things
   → "[entity A] [entity B] [relationship terms] connection"

3. **ATTRIBUTE-FOCUSED**: For questions about specific properties
   → "[entity] [attribute type] [value indicators]"

4. **COLLECTION-FOCUSED**: For questions asking for lists or groups
   → "[category] [type] [criteria] list all"

5. **TEMPORAL-FOCUSED**: For time-specific questions
   → "[subject] [time period] [event/state terms]"

**Step 3: Add Synonyms**
Include alternative terms that might appear in documents.

## CRITICAL: Response Format
You MUST wrap your query in <query> tags. Do NOT include any other text.

<query>
[Your focused search query - 5-15 words targeting key concepts]
</query>

Your response (ONLY use <query> tags):"""


# =============================================================================
# KNOWLEDGE FORMAT TEMPLATES
# =============================================================================
KNOWLEDGE_FORMAT_TEMPLATE = """
### Retrieved Knowledge (Step {step}):

**Query Used**: "{query}"

**Entities Found** ({entity_count}):
{entities}

**Relationships Found** ({relationship_count}):
{relationships}

**Document Content** ({chunk_count} chunks):
{chunks}

**Sources Covered**: {sources}
"""


NO_KNOWLEDGE_TEMPLATE = """
### Retrieved Knowledge (Step {step}):

**Query Used**: "{query}"

No relevant entities, relationships, or document content found for this query.

**Suggestion**: Try a different query with:
- Alternative keywords or synonyms
- Broader or narrower scope
- Different entity names or types
"""


# =============================================================================
# SYSTEM PROMPT FOR REASONING AGENT
# =============================================================================
AGENT_SYSTEM_PROMPT = """You are an intelligent knowledge graph reasoning agent. You answer questions by iteratively querying a knowledge graph and document store.

## Core Capabilities
- Search for entities (people, organizations, objects, concepts)
- Find relationships between entities
- Retrieve relevant document chunks
- Synthesize information from multiple sources

## Reasoning Process (Graph-R1 Style)
1. **DECOMPOSE**: Break the question into components
2. **MAP**: Match retrieved knowledge to question components
3. **ANALYZE**: Identify gaps and cross-reference sources
4. **DECIDE**: Answer if ready, or query for missing information

## Output Format Rules
- Respond with ONLY <answer>...</answer> OR <query>...</query>
- Never use both tags in one response
- Never include text outside the tags
- Provide structured, evidence-based answers

## Quality Standards
- Always cite sources when answering
- Acknowledge uncertainty or limitations
- Never fabricate information not in the retrieved knowledge
- Use exact quotes from documents when possible

Maximum {max_steps} reasoning steps allowed. Use them wisely."""


# =============================================================================
# QUERY REFINEMENT PROMPT
# =============================================================================
REFINE_QUERY_PROMPT = """## Context
You need to generate a refined query because the previous query did not provide sufficient information.

## Original Question
{question}

## Previous Query
"{previous_query}"

## What Was Found
{previous_results}

## What Is Still Missing
{missing_info}

## Query Refinement Strategies

Choose ONE strategy based on what's missing:

1. **SYNONYM STRATEGY**: If the topic exists but wasn't found
   → Use alternative terms: "{topic}" → synonyms, related terms

2. **SPECIFICITY STRATEGY**: If results were too broad
   → Add qualifiers: "[topic] [specific type] [specific attribute]"

3. **BROADENING STRATEGY**: If results were too narrow
   → Remove constraints: "[broader category] [general terms]"

4. **RELATIONSHIP STRATEGY**: If entity found but connections missing
   → Query connections: "[found entity] related [missing aspect]"

5. **SOURCE STRATEGY**: If information might be in different documents
   → Target source type: "[topic] [document type not yet searched]"

## CRITICAL: Response Format
Wrap your refined query in <query> tags. The query must be DIFFERENT from: "{previous_query}"

<query>
[Your refined search query using a different approach]
</query>

Your refined query:"""


# =============================================================================
# SUFFICIENCY CHECK PROMPT
# =============================================================================
SUFFICIENCY_CHECK_PROMPT = """## Task
Evaluate whether the retrieved knowledge is sufficient to answer the question.

## Question
{question}

## Retrieved Knowledge
{knowledge}

## Evaluation Criteria

**1. Core Coverage** (Required):
- [ ] Main subject/entity is identified
- [ ] Primary question is addressable
- [ ] Key facts/values are present

**2. Completeness** (Important):
- [ ] All parts of a multi-part question covered
- [ ] Sufficient detail for the answer type requested
- [ ] No critical gaps in the reasoning chain

**3. Confidence** (Quality):
- [ ] Information from reliable sources
- [ ] Multiple sources corroborate (if applicable)
- [ ] No major contradictions

## Scoring
- 3/3 Core + 3/3 Completeness + High Confidence = SUFFICIENT
- 3/3 Core + 2/3 Completeness = SUFFICIENT (with caveats)
- 2/3 Core or less = INSUFFICIENT

## Response
Respond with EXACTLY one word:
- **SUFFICIENT**: Knowledge is adequate to answer the question
- **INSUFFICIENT**: Critical information is missing

Your evaluation:"""


# =============================================================================
# ANSWER FORMATTING GUIDELINES (for reference in prompts)
# =============================================================================
ANSWER_FORMAT_GUIDELINES = """
## Answer Structure Templates

### For FACTUAL Questions (who, what, when, where):
<answer>
## Direct Answer
[Concise answer to the question]

## Evidence
- **[Source]**: "[exact quote or data]"

## Confidence: [HIGH/MEDIUM/LOW]
</answer>

### For LIST Questions (all X, which Y):
<answer>
## Items Found
1. **[Item 1]** - [brief description] (Source: [doc])
2. **[Item 2]** - [brief description] (Source: [doc])
3. **[Item 3]** - [brief description] (Source: [doc])

## Summary
Total: [N] items found

## Confidence: [HIGH/MEDIUM/LOW]
- Note: [Any limitations on completeness]
</answer>

### For AGGREGATION Questions (how much, how many, total):
<answer>
## Result
**[Aggregation Type]**: [Value]

## Breakdown
| Item | Value | Source |
|------|-------|--------|
| [Item 1] | [Value] | [Doc] |
| [Item 2] | [Value] | [Doc] |
| **Total** | **[Sum]** | |

## Confidence: [HIGH/MEDIUM/LOW]
</answer>

### For COMPARISON Questions (difference between, compare):
<answer>
## Comparison: [X] vs [Y]

| Aspect | [X] | [Y] |
|--------|-----|-----|
| [Aspect 1] | [Value] | [Value] |
| [Aspect 2] | [Value] | [Value] |

## Key Differences
- [Difference 1]
- [Difference 2]

## Confidence: [HIGH/MEDIUM/LOW]
</answer>

### For PROCESS Questions (how does, steps to):
<answer>
## Process: [Name]

**Step 1**: [Action]
**Step 2**: [Action]
**Step 3**: [Action]

## Source
Based on: [Document name/section]

## Confidence: [HIGH/MEDIUM/LOW]
</answer>

### For EXPLANATION Questions (why, explain):
<answer>
## Explanation

[Main explanation paragraph]

## Key Points
- [Point 1]
- [Point 2]

## Evidence
- **[Source]**: "[supporting quote]"

## Confidence: [HIGH/MEDIUM/LOW]
</answer>
"""


# =============================================================================
# DOMAIN-SPECIFIC FILTERING RULES (Applied contextually)
# =============================================================================
FINANCIAL_FILTERING_RULES = """
## Financial Document Filtering Rules

When answering questions about products, purchases, or items:

**EXCLUDE from product/item lists:**
- Shipping Charges, Delivery Fees
- Environmental/Recycling Fees
- Federal/Provincial/State Tax, GST/HST, VAT, Sales Tax
- Discounts, Rebates, Credits
- Service Fees, Processing Fees, Handling Fees
- Subtotals, Totals, Grand Totals

**INCLUDE as products/items:**
- Physical goods with descriptions
- Software licenses or subscriptions
- Services that are the primary purchase

**FILTERING TEST**: Ask "Is this an actual product/service purchased, or is it a fee/tax/charge?"
"""

TECHNICAL_FILTERING_RULES = """
## Technical Document Filtering Rules

When answering questions about systems, components, or specifications:

**Distinguish between:**
- Requirements vs. Implementations
- Current state vs. Planned state
- Mandatory vs. Optional features
- Deprecated vs. Active components

**Version awareness:**
- Note version numbers when available
- Distinguish between different releases
- Identify superseded information
"""

LEGAL_FILTERING_RULES = """
## Legal Document Filtering Rules

When answering questions about contracts, agreements, or policies:

**Pay attention to:**
- Effective dates and expiration dates
- Amendment and addendum references
- Defined terms (capitalized terms with specific meanings)
- Conditions and exceptions

**Distinguish between:**
- Obligations vs. Rights
- Mandatory vs. Permissive language ("shall" vs "may")
- Primary terms vs. Boilerplate
"""
