# Agent Flow Optimization Plan - Quick Win

## Overview

This plan focuses on **Quick Win** optimizations for the agent flow by reusing existing non-agent services. The goal is to achieve immediate parity with the non-agent flow without building new logic.

**Principle:** Don't reinvent - reuse existing sophisticated code.

---

## Current Problems

### 1. SQL Generation (Broken)

| Issue | Impact |
|-------|--------|
| Hardcoded `filename` column that doesn't exist | SQL fails 100% of the time |
| Template-based SQL generation (127 lines) | Missing schema awareness |
| Naive retry (same broken query 4x) | No error learning/correction |
| No SchemaService integration | Wrong field names, wrong casting |

### 2. Document Filtering (Defective)

| Issue | Impact |
|-------|--------|
| No entity extraction | Can't filter by vendor/customer name |
| No fuzzy matching | Misses "Amazon" vs "Amazon.com Inc." |
| Fixed threshold (0.3) | Includes irrelevant documents |
| No boost/penalty scoring | Poor document ranking |
| No multi-stage fallback | Fails when initial match doesn't work |

---

## Quick Win Solution

### Architecture Change

```
BEFORE (Broken):
┌─────────────────────────────────────────────────────────────┐
│ Agent Flow                                                  │
│ ┌─────────────────┐    ┌─────────────────┐                 │
│ │ routing_tools.py│    │ sql_tools.py    │                 │
│ │ (naive filter)  │    │ (broken SQL)    │                 │
│ └─────────────────┘    └─────────────────┘                 │
└─────────────────────────────────────────────────────────────┘

AFTER (Fixed):
┌─────────────────────────────────────────────────────────────┐
│ Agent Flow                                                  │
│ ┌─────────────────┐    ┌─────────────────┐                 │
│ │ routing_tools.py│    │ sql_tools.py    │                 │
│ │ (thin wrapper)  │    │ (thin wrapper)  │                 │
│ └────────┬────────┘    └────────┬────────┘                 │
│          │                      │                           │
│          ▼                      ▼                           │
│ ┌─────────────────────────────────────────────────────────┐│
│ │              Shared Services (Non-Agent)                ││
│ │  • EntityExtractor (rag_service/entity_extractor.py)    ││
│ │  • Document Scoring (rag_service/chat_api.py)           ││
│ │  • LLMSQLGenerator (analytics_service/)                 ││
│ │  • SchemaService (analytics_service/)                   ││
│ │  • SQLErrorCorrector (analytics_service/)               ││
│ └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Tasks

### Part 1: SQL Generation Quick Win

#### Task 1.1: Import Existing SQL Services

**File to modify:** `backend/agents/tools/sql_tools.py`

**Services to import from `backend/analytics_service/`:**
- `LLMSQLGenerator` or `LLMSQLGeneratorV2` - Main SQL generation
- `SchemaService` - Field mappings and schema lookup
- `correct_sql_error()` function - LLM-based error correction

#### Task 1.2: Replace Template-Based SQL Generation

**Current code to replace:** `generate_schema_aware_sql()` function (lines 35-127)

**Replace with:** Call to `LLMSQLGenerator.generate_sql()` with:
- Pass document IDs
- Pass user query
- Pass agent's LLM client
- Pass schema context from SchemaService

#### Task 1.3: Replace Naive Retry with Error Correction

**Current code to replace:** `execute_sql_with_retry()` function (lines 137-231)

**Replace with:** Integration with existing error correction:
```
1. Execute SQL
2. If error:
   a. Call correct_sql_error(failed_sql, error_message, schema_context)
   b. LLM analyzes error and generates corrected SQL
   c. Retry with corrected SQL
3. Return results
```

#### Task 1.4: Fix LLM Client Dependency

**Problem:** Non-agent services need `llm_client` which agent context doesn't provide directly

**Solution:**
- Create adapter to pass agent's LLM client to shared services
- Or instantiate LLM client within sql_tools using same config as non-agent

---

### Part 2: Document Filtering Quick Win

#### Task 2.1: Import Existing Filtering Services

**File to modify:** `backend/agents/tools/routing_tools.py`

**Services to import:**
- `EntityExtractor` from `backend/rag_service/entity_extractor.py`
- Document scoring logic from `backend/rag_service/chat_api.py`
- Fuzzy matching utilities (rapidfuzz)

#### Task 2.2: Add Entity Extraction to Routing

**Current flow:**
```
User query → Check schema_type → Check field names → Return documents
```

**New flow:**
```
User query → Extract entities (vendor, customer, dates)
          → Match against vendor_normalized, customer_normalized
          → Apply scoring with boosts/penalties
          → Return ranked documents
```

#### Task 2.3: Use Normalized Entity Fields

**Fields to use from documents_data:**
- `vendor_normalized` - For vendor entity matching
- `customer_normalized` - For customer entity matching
- `all_entities_normalized` - For general entity matching
- `document_metadata.topics` - For topic filtering

#### Task 2.4: Apply Existing Scoring Logic

**Scoring to implement (from non-agent flow):**

| Match Type | Score Adjustment |
|------------|------------------|
| Entity exact match | +0.3 boost |
| Entity fuzzy match (80%+) | +0.2 boost |
| Entity no match | ×0.4 penalty |
| Document type match | +0.25 boost |
| Tabular data (for analytics) | +0.35 boost |
| Field relevance | +0.1 to +0.2 |

#### Task 2.5: Replace Fixed Threshold with Adaptive

**Current:** Fixed threshold = 0.3

**New:** Adaptive threshold = top_score × 0.5

This ensures only documents within 50% of the best match are included.

#### Task 2.6: Implement Multi-Stage Fallback

**Fallback stages (from non-agent flow):**

```
Stage 1: Entity filter (exact match)
    ↓ (if no results)
Stage 2: Entity filter (fuzzy match)
    ↓ (if no results)
Stage 3: Topic/keyword filter
    ↓ (if no results)
Stage 4: Schema type filter
    ↓ (if no results)
Stage 5: Return all available documents with low confidence warning
```

---

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `backend/agents/tools/sql_tools.py` | Major refactor | Replace template SQL with shared service calls |
| `backend/agents/tools/routing_tools.py` | Major refactor | Replace naive filtering with shared service calls |
| `backend/agents/tools/__init__.py` | Minor update | Update imports if needed |

---

## Dependencies

### Existing Services to Reuse

| Service | Location | Purpose |
|---------|----------|---------|
| `LLMSQLGenerator` | `analytics_service/llm_sql_generator.py` | SQL generation with 3-round LLM |
| `LLMSQLGeneratorV2` | `analytics_service/llm_sql_generator_v2.py` | Optimized SQL generation |
| `SchemaService` | `analytics_service/schema_service.py` | Field mappings and schema lookup |
| `correct_sql_error()` | `analytics_service/llm_sql_generator.py` | LLM-based SQL error correction |
| `EntityExtractor` | `rag_service/entity_extractor.py` | Extract entities from queries |
| `document_scoring` | `rag_service/chat_api.py` | Document relevance scoring |

### External Libraries Already in Use

- `rapidfuzz` - Fuzzy string matching
- `psycopg2` - PostgreSQL execution
- `sqlalchemy` - ORM and query execution

---

## Expected Outcomes

### SQL Generation

| Metric | Before | After |
|--------|--------|-------|
| SQL success rate | ~0% (filename error) | ~95%+ |
| Error recovery | None (retry same query) | LLM-based correction |
| Schema awareness | None | Full field mapping |
| Type casting | Hardcoded | Dynamic based on schema |

### Document Filtering

| Metric | Before | After |
|--------|--------|-------|
| Entity-based queries | Fails | Works |
| Vendor name variations | Misses | Fuzzy matched |
| False positives | High (fixed threshold) | Low (adaptive) |
| Fallback on no match | None | 5-stage fallback |

---

## Testing Checklist

### SQL Generation Tests

- [ ] Query with existing schema fields succeeds
- [ ] Query with wrong field name gets corrected
- [ ] Query across multiple documents works
- [ ] Aggregation queries (SUM, COUNT, AVG) work
- [ ] Error correction generates valid SQL on retry

### Document Filtering Tests

- [ ] "Show Amazon invoices" returns only Amazon documents
- [ ] "Amazon.com" matches "Amazon Inc." (fuzzy)
- [ ] Adaptive threshold excludes low-relevance documents
- [ ] Fallback returns results when exact match fails
- [ ] Entity extraction handles vendor/customer/date

---

## Future Optimizations (On Hold)

The following optimizations are documented but **not included** in this Quick Win phase:

### SQL Intelligence Layer (Phase 2)
- Query decomposition for complex queries
- Result validation and self-correction
- Cross-document reasoning
- Confidence scoring
- Conversation context usage

### Document Filtering Intelligence (Phase 2)
- Intent-aware document selection
- Progressive filtering with feedback loop
- Schema compatibility validation
- Temporal reasoning
- Ambiguity resolution
- Cross-query document memory

These will be addressed in future optimization phases after Quick Win is validated.

---

## Timeline Estimate

| Task | Effort |
|------|--------|
| Part 1: SQL Generation Quick Win | Medium |
| Part 2: Document Filtering Quick Win | Medium |
| Integration Testing | Small |
| **Total** | **Medium** |

---

## Implementation Status

### Completed: 2026-01-08

#### Part 1: SQL Generation Quick Win
- [x] Added helper functions: `_get_db_session()`, `_get_llm_client()`, `_get_sql_generator()`, `_get_schema_service()`
- [x] Added `_get_field_mappings_for_documents()` to get schema context
- [x] Updated `generate_schema_aware_sql()` to use `LLMSQLGenerator` with fallback to improved template
- [x] Fixed template SQL to use proper JOIN (removed non-existent `filename` column)
- [x] Updated `execute_sql_with_retry()` with LLM-based error correction via `_correct_sql_with_llm()`
- [x] Added correction history tracking in execution results

#### Part 2: Document Filtering Quick Win
- [x] Imported `EntityExtractor` functions from `rag_service`
- [x] Added scoring configuration constants (boosts/penalties)
- [x] Added `_extract_query_entities()` for entity extraction from queries
- [x] Added `_match_entity_to_document()` for entity matching with fuzzy support
- [x] Added `_is_analytics_query()` for query intent detection
- [x] Added `_apply_adaptive_threshold()` for adaptive document filtering
- [x] Enhanced `_calculate_relevance_from_header_data()` with entity matching and boosts
- [x] Updated `get_relevant_documents()` to fetch `extraction_metadata` and apply adaptive threshold
- [x] Added detailed logging for document filtering decisions

#### Part 3: Auto-Flowing Status Messages (UI Enhancement)
- [x] Added `_build_status_details()` helper in workflow.py for detailed progress info
- [x] Updated `execute_analytics_query_async()` to yield rich status details with phases, messages, sub-messages, and metrics
- [x] Updated `chat_api.py` to send "status" WebSocket messages with detailed agent progress
- [x] Added `statusMessages` state in AgenticChatBot.jsx for tracking status flow
- [x] Added WebSocket handler for "status" message type
- [x] Implemented auto-flowing status display with phase icons, animated entries, and sub-messages
- [x] Added comprehensive SCSS styling for status flow with phase-specific colors and animations
- [x] Added auto-scroll behavior for new status messages

#### Part 4: Field Mappings Format Compatibility Fix
- [x] Added extensive logging in `_validate_and_normalize_field_mappings()` to debug field mapping structures
- [x] Enhanced `_validate_and_normalize_field_mappings()` to handle nested list formats (e.g., `[field_name, field_info]`)
- [x] Updated `LLMSQLGenerator._format_grouped_field_schema_json()` with defensive `_normalize_mapping()` helper
- [x] Fixed `PromptBuilder._format_grouped_schema()` to handle various input formats defensively
- [x] All grouped schema formatters now handle: dicts, nested lists, and string items

---

## Success Criteria

1. **SQL queries no longer fail with "filename does not exist" error**
2. **SQL errors trigger LLM-based correction instead of naive retry**
3. **Entity-based document queries return correct documents**
4. **Fuzzy matching catches vendor name variations**
5. **Agent flow accuracy matches non-agent flow accuracy**
