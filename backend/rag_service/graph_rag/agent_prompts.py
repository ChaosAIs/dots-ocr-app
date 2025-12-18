"""
Agent prompts for Graph-R1 style iterative reasoning.

These prompts guide the LLM through the think-query-retrieve-answer loop.
"""

# Prompt for the agent to think and decide whether to continue querying
AGENT_THINK_PROMPT = """## Current State
- **Original Question**: {question}
- **Reasoning Step**: {step}/{max_steps}
- **Retrieved Knowledge So Far**:
{retrieved_knowledge}

## Task
Analyze the retrieved knowledge and decide your next action.

## CRITICAL: Response Format Requirements
You MUST respond with EXACTLY ONE of these two formats:

**Option A - If you have enough information to answer:**
<answer>
[Your complete, detailed answer to the question based on the retrieved knowledge]
</answer>

**Option B - If you need more information:**
<query>
[A specific search query to retrieve additional knowledge]
</query>

## Decision Guidelines
- Choose <answer> if the retrieved knowledge contains sufficient information to fully address the question
- Choose <query> if critical information is missing or unclear
- Do NOT include any text before or after the tags
- Do NOT use both tags - pick only ONE

Your response (use ONLY <answer> or <query> tags):"""


# Prompt for initial query generation
INITIAL_QUERY_PROMPT = """## Question
{question}

## Task
Generate an initial search query to find relevant knowledge from the knowledge graph.

## CRITICAL: Response Format
You MUST wrap your query in <query> tags. Do NOT include any other text.

<query>
[Your focused search query targeting key entities and concepts]
</query>

Your response (ONLY use <query> tags):"""


# Prompt for formatting retrieved knowledge
KNOWLEDGE_FORMAT_TEMPLATE = """
### Retrieved Knowledge (Step {step}):
**Entities Found**:
{entities}

**Relationships Found**:
{relationships}

**Document Content**:
{chunks}
"""


# Prompt for when no knowledge is retrieved
NO_KNOWLEDGE_TEMPLATE = """
### Retrieved Knowledge (Step {step}):
No relevant entities or relationships found for the query.
"""


# System prompt for the reasoning agent
AGENT_SYSTEM_PROMPT = """You are a knowledge graph reasoning agent. You answer questions by iteratively querying a knowledge graph.

## STRICT OUTPUT FORMAT RULES
1. You MUST respond using ONLY <answer>...</answer> OR <query>...</query> tags
2. Do NOT include explanations or text outside these tags
3. Do NOT use both tags in one response - pick ONE

## Reasoning Process
1. Analyze the question and current knowledge
2. If knowledge is SUFFICIENT: Output <answer>complete answer</answer>
3. If knowledge is INSUFFICIENT: Output <query>search query</query>

Maximum {max_steps} reasoning steps allowed. Choose wisely."""


# Prompt for query refinement based on previous results
REFINE_QUERY_PROMPT = """Based on the previous retrieval results, generate a refined query to find additional relevant information.

## Original Question
{question}

## Previous Query
{previous_query}

## Previous Results Summary
{previous_results}

## Instructions
Generate a new, refined query that:
1. Explores different aspects of the question
2. Targets information that was missing from previous results
3. Uses different keywords or entity names

Wrap your refined query in <query>...</query> tags.

Your refined query:"""


# Prompt for knowledge sufficiency check
SUFFICIENCY_CHECK_PROMPT = """Evaluate whether the retrieved knowledge is sufficient to answer the question.

## Question
{question}

## Retrieved Knowledge
{knowledge}

## Instructions
Analyze the retrieved knowledge and determine:
1. Does it contain relevant information about the key entities mentioned in the question?
2. Does it explain the relationships or concepts asked about?
3. Is there enough context to provide a complete answer?

Respond with:
- SUFFICIENT: if the knowledge is enough to answer the question
- INSUFFICIENT: if more information is needed

Your evaluation:"""

