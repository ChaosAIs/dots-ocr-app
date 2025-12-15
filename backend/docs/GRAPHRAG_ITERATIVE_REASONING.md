# GraphRAG Iterative Reasoning (Graph-R1 Implementation)

## Overview

This document describes the iterative "think-query-retrieve-rethink" reasoning cycle implementation in the GraphRAG system, following the **Graph-R1 paper** design.

## What is Iterative Reasoning?

Iterative reasoning is a multi-step process where the system:
1. **THINK**: Analyzes the current knowledge and decides the next action
2. **QUERY**: Generates a search query if more information is needed
3. **RETRIEVE**: Uses LOCAL/GLOBAL/HYBRID modes to get graph + vector results
4. **RETHINK**: Evaluates if sufficient knowledge has been gathered
5. **REPEAT**: Continues up to `max_steps` iterations or until an answer is found

This approach allows the system to:
- Break down complex questions into multiple retrieval steps
- Explore different aspects of the knowledge graph iteratively
- Accumulate knowledge across multiple queries
- Decide when enough information has been gathered

## Configuration

### Environment Variables

Add to `backend/.env`:

```bash
# Iterative reasoning max steps (Graph-R1 think-query-retrieve-rethink cycle)
# Set to 1 for single-step retrieval (default)
# Set to 3-5 for iterative reasoning (recommended for complex queries)
GRAPH_RAG_MAX_STEPS=1
```

### Programmatic Configuration

```python
from rag_service.graph_rag.base import QueryParam

# Single-step retrieval (default)
params = QueryParam(max_steps=1)

# Iterative reasoning (3-5 steps recommended)
params = QueryParam(max_steps=5)
```

## How It Works

### Single-Step Mode (max_steps=1)

When `max_steps=1`, the system performs traditional single-step retrieval:

```
User Query → Mode Detection → Retrieval (LOCAL/GLOBAL/HYBRID) → Return Results
```

### Iterative Mode (max_steps>1)

When `max_steps>1`, the system enters the iterative reasoning loop:

```
User Query
    ↓
Generate Initial Query (LLM)
    ↓
┌─────────────────────────────────────┐
│  ITERATION LOOP (up to max_steps)  │
├─────────────────────────────────────┤
│  1. RETRIEVE                        │
│     - Auto-detect mode or use fixed │
│     - LOCAL/GLOBAL/HYBRID retrieval │
│     - Accumulate results            │
│  2. THINK                           │
│     - Format knowledge summary      │
│     - Ask LLM: continue or answer?  │
│  3. DECIDE                          │
│     - <answer>: Terminate           │
│     - <query>: Continue with new Q  │
│     - No tags: Terminate            │
└─────────────────────────────────────┘
    ↓
Return Accumulated Results
```

## Example Usage

### Python API

```python
from rag_service.graph_rag import GraphRAG
from rag_service.graph_rag.base import QueryParam

# Initialize GraphRAG
graph_rag = GraphRAG()

# Single-step query
context = await graph_rag.query(
    "What is machine learning?",
    params=QueryParam(max_steps=1)
)

# Iterative reasoning query (5 steps)
context = await graph_rag.query(
    "How does machine learning relate to artificial intelligence and what are the key differences?",
    params=QueryParam(max_steps=5)
)
```

### REST API

The `max_steps` parameter is automatically read from the environment variable `GRAPH_RAG_MAX_STEPS`.

To enable iterative reasoning globally, set in `.env`:
```bash
GRAPH_RAG_MAX_STEPS=5
```

## Iterative Reasoning Flow Details

### Phase 1: Initial Query Generation

The LLM generates an initial search query from the user's question:

**Prompt Template**: `INITIAL_QUERY_PROMPT`
- Input: User's original question
- Output: `<query>initial search query</query>`

### Phase 2: Iteration Loop

For each iteration (up to `max_steps`):

#### Step 1: RETRIEVE
- Auto-detect best mode (LOCAL/GLOBAL/HYBRID) for current query
- Perform graph retrieval + Qdrant vector search
- Accumulate entities, relationships, and chunks
- Deduplicate entities by ID

#### Step 2: THINK
- Format all retrieved knowledge into a summary
- Present to LLM with current state (step number, queries made)
- Ask LLM to decide: continue or answer?

**Prompt Template**: `AGENT_THINK_PROMPT`
- Input: Question, current step, retrieved knowledge
- Output: Either `<answer>...</answer>` OR `<query>...</query>`

#### Step 3: DECIDE
- **If `<answer>` tag found**: Terminate loop, return accumulated results
- **If `<query>` tag found**: 
  - Check if query is duplicate → terminate if yes
  - Otherwise, use new query for next iteration
- **If no valid tags**: Terminate loop

### Phase 3: Return Results

Return `GraphRAGContext` with:
- All accumulated entities (deduplicated)
- All accumulated relationships
- All accumulated chunks
- Metadata: steps taken, queries made, modes used

## Termination Conditions

The iterative loop terminates when ANY of these conditions are met:

1. **LLM provides answer**: `<answer>` tag found in response
2. **Max steps reached**: Iteration count >= `max_steps`
3. **Duplicate query**: LLM generates a query already used
4. **Invalid response**: LLM response missing both `<answer>` and `<query>` tags

## Logging

Iterative reasoning produces detailed logs for debugging:

```
[GraphRAG] 🚀 STARTING ITERATIVE REASONING (Graph-R1)
[GraphRAG] Question: How does X relate to Y?
[GraphRAG] Max steps: 5
[GraphRAG] 📝 PHASE 1: Generating initial search query...
[GraphRAG] Initial query: 'X and Y relationship'
[GraphRAG] 🔄 ITERATION 1/5
[GraphRAG] Using HYBRID mode for retrieval
[GraphRAG] Retrieved: Entities: 15, Relationships: 8, Chunks: 20
[GraphRAG] 🤔 THINKING: Should I continue or answer?
[GraphRAG] ✅ DECISION: TERMINATE - Answer found
[GraphRAG] 🏁 ITERATIVE REASONING COMPLETE
[GraphRAG] Summary: Total iterations: 1, Queries: ['X and Y relationship']
```

## Performance Considerations

- **Single-step (max_steps=1)**: Fast, suitable for simple queries
- **Iterative (max_steps=3-5)**: Slower but more comprehensive
  - Each iteration requires: 1 LLM call + 1 retrieval operation
  - Total LLM calls: 1 (initial query) + max_steps (think) = max_steps + 1
  - Recommended for complex, multi-faceted questions

## Best Practices

1. **Use single-step for simple queries**: "What is X?", "Define Y"
2. **Use iterative for complex queries**: "How does X relate to Y?", "Compare A and B"
3. **Set max_steps=3-5**: Good balance between thoroughness and performance
4. **Monitor logs**: Check termination reasons to optimize max_steps
5. **Combine with mode detection**: Let the system auto-detect best mode per iteration

## Comparison with Original Graph-R1 Paper

| Aspect | Graph-R1 Paper | Our Implementation |
|--------|----------------|-------------------|
| Iterative reasoning | Built-in | Controlled by `max_steps` parameter |
| Query modes | LOCAL, GLOBAL, HYBRID, NAIVE | LOCAL, GLOBAL, HYBRID only |
| Storage | Custom | Neo4j + Qdrant + PostgreSQL |
| Vector search | Optional | Always combined with graph results |
| Mode detection | Manual | Auto-detection via LLM + heuristics |
| Termination | LLM decision | LLM decision + max_steps limit |

## Troubleshooting

### Issue: Iterative reasoning not activating
**Solution**: Check that `max_steps > 1` in QueryParam or GRAPH_RAG_MAX_STEPS env var

### Issue: Loop terminates too early
**Solution**: Check LLM response format - ensure it uses `<query>` tags correctly

### Issue: Loop runs to max_steps every time
**Solution**: LLM may not be generating `<answer>` tags - check prompt templates

### Issue: Duplicate queries
**Solution**: LLM is repeating queries - may need better prompt engineering or higher temperature

## Related Files

- `backend/rag_service/graph_rag/graph_rag.py` - Main implementation
- `backend/rag_service/graph_rag/base.py` - QueryParam definition
- `backend/rag_service/graph_rag/agent_prompts.py` - LLM prompts for reasoning
- `backend/.env` - Configuration (GRAPH_RAG_MAX_STEPS)
- `backend/test/test_graphrag.py` - Unit tests

