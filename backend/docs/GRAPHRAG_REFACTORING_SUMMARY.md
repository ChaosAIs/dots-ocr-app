# GraphRAG Refactoring Summary - Graph-R1 Paper Alignment

## Overview

This document summarizes the major refactoring of the GraphRAG implementation to align with the original **Graph-R1 paper** design.

**Date**: 2025-12-15  
**Objective**: Remove custom modes (AGENT, NAIVE), implement iterative reasoning as core feature, ensure vector search is always included

---

## Changes Made

### 1. Query Modes - Aligned with Graph-R1 Paper

**Before**:
- 5 modes: LOCAL, GLOBAL, HYBRID, NAIVE, AGENT
- AGENT mode was a separate wrapper
- NAIVE mode was simple vector-only search

**After**:
- 3 modes only: LOCAL, GLOBAL, HYBRID (as per Graph-R1 paper)
- AGENT mode removed - iterative reasoning is now a core feature
- NAIVE mode removed - vector search is always combined with graph results

**Files Modified**:
- `backend/rag_service/graph_rag/base.py` - Removed AGENT and NAIVE from QueryMode enum
- `backend/rag_service/graph_rag/query_mode_detector.py` - Removed AGENT and NAIVE detection
- `backend/rag_service/graph_rag/prompts.py` - Updated mode detection prompt
- `backend/.env` - Removed GRAPHRAG_QUERY_MODE=agent
- `backend/test/test_graphrag.py` - Removed NAIVE mode tests

### 2. Iterative Reasoning - Graph-R1 Implementation

**Before**:
- Iterative reasoning was a separate "AGENT" mode
- Required explicit mode selection
- Not integrated with core retrieval

**After**:
- Iterative reasoning is controlled by `max_steps` parameter
- When `max_steps > 1`: Activates "think-query-retrieve-rethink" cycle
- When `max_steps = 1`: Single-step retrieval (default)
- Works with all modes (LOCAL, GLOBAL, HYBRID)

**Implementation**:
- Added `_iterative_reasoning_query()` method to `GraphRAG` class
- Implements full Graph-R1 reasoning loop:
  1. Generate initial query (LLM)
  2. RETRIEVE: Use LOCAL/GLOBAL/HYBRID mode
  3. THINK: Ask LLM to decide (continue or answer)
  4. DECIDE: Parse `<answer>` or `<query>` tags
  5. REPEAT: Up to max_steps or until termination

**Files Modified**:
- `backend/rag_service/graph_rag/graph_rag.py` - Added iterative reasoning logic
- `backend/rag_service/graph_rag/base.py` - Added max_steps parameter with env var support
- `backend/.env` - Added GRAPH_RAG_MAX_STEPS=1 (default)

### 3. Vector Search Integration

**Before**:
- Vector search was optional
- Some modes didn't include Qdrant results
- Graph-only retrieval was possible

**After**:
- Vector search is ALWAYS performed
- All modes (LOCAL, GLOBAL, HYBRID) combine graph + vector results
- Added `_get_vector_search_chunks()` helper method
- Added `_merge_chunks()` for deduplication

**Files Modified**:
- `backend/rag_service/graph_rag/graph_rag.py` - Updated all retrieval methods
- `backend/.env` - Added GRAPH_RAG_VECTOR_SEARCH_ENABLED=true

### 4. File Summary Feature Cleanup

**Before**:
- References to `FILE_SUMMARY_ENABLED` and `search_file_summaries()`
- Caused runtime errors (undefined variables)

**After**:
- Removed all file summary dependencies
- Functions return empty lists (feature deprecated)
- No runtime errors

**Files Modified**:
- `backend/rag_service/rag_agent.py` - Simplified `_find_relevant_files()` and `_find_relevant_files_with_scopes()`

---

## Configuration Changes

### Environment Variables

**Added**:
```bash
# Iterative reasoning max steps (Graph-R1 think-query-retrieve-rethink cycle)
GRAPH_RAG_MAX_STEPS=1  # Set to 3-5 for iterative reasoning

# Enable/disable Qdrant vector search (always combines with graph results)
GRAPH_RAG_VECTOR_SEARCH_ENABLED=true
```

**Removed**:
```bash
GRAPHRAG_QUERY_MODE=agent  # No longer needed
```

### QueryParam Defaults

```python
@dataclass
class QueryParam:
    mode: QueryMode = QueryMode.HYBRID
    top_k: int = 60  # Reads from GRAPH_RAG_TOP_K env var
    max_steps: int = 1  # Reads from GRAPH_RAG_MAX_STEPS env var
    # ... other fields
```

---

## API Changes

### GraphRAG.query() Method

**Signature** (unchanged):
```python
async def query(
    self,
    query: str,
    mode: str = None,
    params: QueryParam = None,
) -> GraphRAGContext
```

**Behavior Changes**:
- When `params.max_steps > 1`: Activates iterative reasoning
- When `params.max_steps = 1`: Single-step retrieval (default)
- All modes now include vector search results

**Example Usage**:
```python
# Single-step (default)
context = await graph_rag.query("What is X?")

# Iterative reasoning (5 steps)
context = await graph_rag.query(
    "How does X relate to Y?",
    params=QueryParam(max_steps=5)
)
```

---

## Testing Updates

### Test Changes

**File**: `backend/test/test_graphrag.py`

**Removed**:
- `test_naive_mode_detection()` - NAIVE mode no longer exists

**Updated**:
- `test_query_modes_exist()` - Now checks only LOCAL, GLOBAL, HYBRID
- `test_default_values()` - Updated to check max_steps parameter

**Added**:
- `test_iterative_reasoning_enabled()` - Verifies max_steps > 1 enables iterative reasoning

---

## Migration Guide

### For Users

**If you were using AGENT mode**:
```python
# Before
params = QueryParam(mode=QueryMode.AGENT)
context = await graph_rag.query(query, params=params)

# After
params = QueryParam(max_steps=5)  # Enable iterative reasoning
context = await graph_rag.query(query, params=params)
```

**If you were using NAIVE mode**:
```python
# Before
params = QueryParam(mode=QueryMode.NAIVE)
context = await graph_rag.query(query, params=params)

# After
# Use HYBRID mode (default) - it now includes vector search
params = QueryParam(max_steps=1)  # Single-step retrieval
context = await graph_rag.query(query, params=params)
```

### For Developers

**Removed Methods**:
- `GraphRAG._naive_retrieval()` - No longer exists
- `GraphRAG._agent_query()` - No longer exists
- `GraphRAG._init_agent()` - No longer exists

**New Methods**:
- `GraphRAG._iterative_reasoning_query()` - Implements Graph-R1 reasoning loop
- `GraphRAG._get_vector_search_chunks()` - Direct Qdrant vector search
- `GraphRAG._merge_chunks()` - Deduplicate chunks
- `GraphRAG._format_knowledge_summary()` - Format knowledge for LLM prompts
- `GraphRAG._format_entities_for_prompt()` - Format entities for LLM
- `GraphRAG._format_relationships_for_prompt()` - Format relationships for LLM

---

## Performance Impact

### Single-Step Mode (max_steps=1)
- **No change** from previous behavior
- Fast, suitable for simple queries
- 1 LLM call for mode detection (if auto)
- 1 retrieval operation

### Iterative Mode (max_steps=3-5)
- **New capability** - more comprehensive but slower
- Each iteration: 1 LLM call + 1 retrieval
- Total LLM calls: 1 (initial query) + max_steps (think)
- Recommended for complex, multi-faceted questions

---

## Documentation

**New Documents**:
- `backend/docs/GRAPHRAG_ITERATIVE_REASONING.md` - Detailed guide on iterative reasoning
- `backend/docs/GRAPHRAG_REFACTORING_SUMMARY.md` - This document

**Updated Documents**:
- README files should be updated to reflect new modes and iterative reasoning

---

## Backward Compatibility

### Breaking Changes
- ❌ `QueryMode.AGENT` no longer exists
- ❌ `QueryMode.NAIVE` no longer exists
- ❌ `GRAPHRAG_QUERY_MODE=agent` env var no longer supported

### Compatible Changes
- ✅ `QueryParam` API unchanged (new optional parameter: max_steps)
- ✅ `GraphRAG.query()` signature unchanged
- ✅ Default behavior unchanged (single-step retrieval)
- ✅ All existing code using LOCAL/GLOBAL/HYBRID modes works as before

---

## Next Steps

1. **Restart the backend server** to load new code
2. **Test queries** with both single-step and iterative modes
3. **Monitor logs** to verify correct behavior
4. **Adjust max_steps** based on query complexity and performance needs
5. **Update client code** if using AGENT or NAIVE modes

---

## Related Files

### Core Implementation
- `backend/rag_service/graph_rag/graph_rag.py` - Main GraphRAG class
- `backend/rag_service/graph_rag/base.py` - Data structures and enums
- `backend/rag_service/graph_rag/query_mode_detector.py` - Mode detection
- `backend/rag_service/graph_rag/agent_prompts.py` - LLM prompts for reasoning

### Configuration
- `backend/.env` - Environment variables
- `backend/rag_service/graph_rag/prompts.py` - LLM prompts

### Tests
- `backend/test/test_graphrag.py` - Unit tests

### Documentation
- `backend/docs/GRAPHRAG_ITERATIVE_REASONING.md` - Iterative reasoning guide
- `backend/docs/GRAPHRAG_REFACTORING_SUMMARY.md` - This document

