# Semantic Query Cache Solution with Qdrant

## Overview

This document describes a complete caching solution for repeated or similar user questions using Qdrant as the sole caching layer. The solution uses **semantic similarity matching combined with permission-based validation** to identify and return cached answers securely, significantly improving response times for frequently asked questions.

**Key Principle**: A cache hit requires BOTH:
1. Semantic similarity of the question (above threshold)
2. **Current user has READ permission** to ALL source documents used to generate the cached answer

This ensures:
- Similar questions can benefit from cached answers (high cache hit rate)
- Users can only access cached answers derived from documents they are authorized to read (security)
- No information leakage between users with different document access levels

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Question                             │
│                    + User ID (for permission check)              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Query Cache Manager                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  1. Normalize question                                   │    │
│  │  2. Generate question embedding                          │    │
│  │  3. Semantic search for similar cached questions        │    │
│  │  4. For each candidate:                                  │    │
│  │     → PERMISSION CHECK: Can user access ALL source docs? │    │
│  │     → If YES: Return cached answer                       │    │
│  │     → If NO: Try next candidate                          │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
        ┌───────────────────┐   ┌───────────────────────────────┐
        │    Cache Hit      │   │       Cache Miss              │
        │                   │   │                               │
        │  Requirements:    │   │  Execute full routing:        │
        │  ✓ Similar question│  │  • Intent classification      │
        │  ✓ User has READ  │   │  • Query generation           │
        │    permission to  │   │  • Data retrieval             │
        │    ALL source docs│   │  • Response generation        │
        │                   │   │                               │
        │  Return cached    │   │  ┌─────────────────────────┐  │
        │  answer directly  │   │  │ Return response to user │  │
        │                   │   │  │ IMMEDIATELY             │  │
        │  Update hit_count │   │  └───────────┬─────────────┘  │
        └───────────────────┘   │              │                 │
                                │              ▼                 │
                                │  ┌─────────────────────────┐  │
                                │  │ BACKGROUND (async):     │  │
                                │  │ Store in cache with     │  │
                                │  │ source_document_ids     │  │
                                │  │ (non-blocking)          │  │
                                │  └─────────────────────────┘  │
                                └───────────────────────────────┘
```

## Non-Blocking Cache Storage

**CRITICAL DESIGN PRINCIPLE**: Cache storage NEVER blocks user response.

```
Timeline for Cache Miss:

User Request ──────────────────────────────────────────────────────►

    │
    ▼
┌─────────┐   ┌──────────────┐   ┌─────────────────┐
│ Cache   │──►│ Process      │──►│ Return Response │ ◄── User gets answer HERE
│ Lookup  │   │ Question     │   │ to User         │
│ (fast)  │   │ (2-5 sec)    │   │ IMMEDIATELY     │
└─────────┘   └──────────────┘   └────────┬────────┘
                                          │
                                          │ Fire-and-forget
                                          ▼
                              ┌─────────────────────────┐
                              │   BACKGROUND TASK       │
                              │   ┌─────────────────┐   │
                              │   │ Generate embed  │   │
                              │   │ Store in Qdrant │   │
                              │   │ Update indexes  │   │
                              │   └─────────────────┘   │
                              │   (runs independently)  │
                              └─────────────────────────┘

User Response Time = Cache Lookup + Question Processing
                   ≠ Cache Lookup + Question Processing + Cache Storage
```

### Benefits of Background Caching

| Aspect | Blocking (Bad) | Non-Blocking (Good) |
|--------|---------------|---------------------|
| User wait time | +100-500ms for cache write | 0ms added |
| Cache failures | User sees error | Silent, logged only |
| High load | Queued cache writes slow users | Users unaffected |
| Qdrant downtime | User requests fail | Users get responses, cache skipped |

## Intelligent Cache Decision Making (Optimized Single LLM Call)

### Design Optimization: Unified Pre-Processing Analysis

**Problem with Multiple LLM Calls:**

The original design required up to 3 separate LLM calls before cache lookup:
1. Dissatisfaction Detection (LLM Call 1)
2. Cache Worthiness Analysis (LLM Call 2)
3. Question Enhancement (LLM Call 3)

This adds **significant latency** (300-900ms per call) and **cost** before the user gets any response.

**Solution: Single Unified LLM Analysis**

Consolidate ALL pre-cache analysis into **ONE LLM call** that performs:
- Dissatisfaction detection
- Question analysis (self-contained vs context-dependent)
- Question enhancement (if needed)
- Cache worthiness determination

### Performance Comparison

| Approach | LLM Calls | Estimated Latency | Cost |
|----------|-----------|-------------------|------|
| **Original (Multiple)** | 3 calls | 600-1200ms | 3x tokens |
| **Optimized (Single)** | 1 call | 200-400ms | 1x tokens |
| **Savings** | -2 calls | -400-800ms | -66% |

### Unified Analysis Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Message Received                        │
│                     + Chat History Context                       │
│                     + Previous System Response (if any)          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│           SINGLE LLM CALL: Unified Pre-Cache Analysis            │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                                                          │    │
│  │  INPUT:                                                  │    │
│  │  - Current user message                                  │    │
│  │  - Last N messages from chat history                    │    │
│  │  - Previous system response (for dissatisfaction check) │    │
│  │                                                          │    │
│  │  ANALYZES ALL AT ONCE:                                   │    │
│  │  ✓ Is user dissatisfied with previous response?         │    │
│  │  ✓ Is question self-contained or context-dependent?     │    │
│  │  ✓ Can it be enhanced to self-contained?                │    │
│  │  ✓ What is the enhanced question (if applicable)?       │    │
│  │  ✓ Is the question worth caching?                       │    │
│  │                                                          │    │
│  │  OUTPUT: Single JSON with all decisions                  │    │
│  │                                                          │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Route Based on Analysis                       │
│                                                                  │
│  if (is_dissatisfied):                                          │
│      → BYPASS CACHE → Fresh Query → Invalidate old cache        │
│                                                                  │
│  elif (is_cacheable):                                           │
│      question_for_cache = enhanced_question or original_question│
│      → CACHE LOOKUP with question_for_cache                     │
│      → HIT: Return cached answer                                │
│      → MISS: Fresh Query → Background cache store               │
│                                                                  │
│  else: // not cacheable                                          │
│      → SKIP CACHE → Fresh Query → Do NOT store in cache         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Unified LLM Prompt

```
You are a query pre-processor for a Q&A caching system. Analyze the user's message and provide ALL of the following assessments in a single response.

## Input Context

Current User Message: "{current_message}"

Chat History (last 5 messages):
{chat_history}

Previous System Response: "{previous_response}"

## Analysis Tasks

Perform ALL of the following analyses:

### 1. DISSATISFACTION CHECK
Is the user expressing dissatisfaction with the previous response?
- Look for: complaints about correctness, requests to refresh/retry, confusion, negative sentiment
- Signals: "that's wrong", "check again", "not what I asked", "are you sure?", "refresh"

### 2. QUESTION ANALYSIS
Analyze the current message as a question:
- Is it self-contained (understandable without chat history)?
- Does it contain unresolved references (pronouns like "it", "that", "this", or relative terms like "the previous one")?
- Is it a follow-up that requires context?

### 3. QUESTION ENHANCEMENT (if applicable)
If the question is NOT self-contained but CAN be made self-contained using chat history:
- Create an enhanced version that replaces all references with actual subjects
- The enhanced question should be understandable in a brand new conversation

### 4. CACHE WORTHINESS
Should this question-answer pair be cached for future reuse?
- Worth caching: specific queries, entity lookups, aggregations, self-contained questions
- NOT worth caching: meta questions about conversation, highly personal/temporal questions

## Response Format (JSON)

{
  "dissatisfaction": {
    "is_dissatisfied": true/false,
    "type": "incorrect|unclear|refresh_request|verification|none",
    "should_bypass_cache": true/false,
    "should_invalidate_previous_cache": true/false
  },
  "question_analysis": {
    "is_self_contained": true/false,
    "has_unresolved_references": true/false,
    "reference_types": ["pronoun", "relative_term", "implicit_subject"],
    "can_be_enhanced": true/false
  },
  "enhancement": {
    "enhanced_question": "the enhanced self-contained question or null",
    "context_used": ["key context item 1", "key context item 2"]
  },
  "cache_decision": {
    "is_cacheable": true/false,
    "reason": "brief explanation",
    "cache_key_question": "the question to use for cache operations (enhanced or original)"
  }
}
```

### Unified Analysis Output Examples

**Example 1: Self-contained cacheable question**
```json
{
  "dissatisfaction": {
    "is_dissatisfied": false,
    "type": "none",
    "should_bypass_cache": false,
    "should_invalidate_previous_cache": false
  },
  "question_analysis": {
    "is_self_contained": true,
    "has_unresolved_references": false,
    "reference_types": [],
    "can_be_enhanced": false
  },
  "enhancement": {
    "enhanced_question": null,
    "context_used": []
  },
  "cache_decision": {
    "is_cacheable": true,
    "reason": "Self-contained specific query about Q3 revenue",
    "cache_key_question": "What is the Q3 2024 revenue?"
  }
}
```

**Example 2: Context-dependent enhanceable question**
```json
{
  "dissatisfaction": {
    "is_dissatisfied": false,
    "type": "none",
    "should_bypass_cache": false,
    "should_invalidate_previous_cache": false
  },
  "question_analysis": {
    "is_self_contained": false,
    "has_unresolved_references": true,
    "reference_types": ["relative_term", "implicit_subject"],
    "can_be_enhanced": true
  },
  "enhancement": {
    "enhanced_question": "What is the Q4 2024 revenue?",
    "context_used": ["Previous discussion about Q3 2024 revenue"]
  },
  "cache_decision": {
    "is_cacheable": true,
    "reason": "Context-dependent but successfully enhanced to self-contained",
    "cache_key_question": "What is the Q4 2024 revenue?"
  }
}
```

**Example 3: User dissatisfied with previous response**
```json
{
  "dissatisfaction": {
    "is_dissatisfied": true,
    "type": "incorrect",
    "should_bypass_cache": true,
    "should_invalidate_previous_cache": true
  },
  "question_analysis": {
    "is_self_contained": true,
    "has_unresolved_references": false,
    "reference_types": [],
    "can_be_enhanced": false
  },
  "enhancement": {
    "enhanced_question": null,
    "context_used": []
  },
  "cache_decision": {
    "is_cacheable": false,
    "reason": "User dissatisfied - bypass cache and fetch fresh results",
    "cache_key_question": null
  }
}
```

**Example 4: Non-cacheable context-dependent question**
```json
{
  "dissatisfaction": {
    "is_dissatisfied": false,
    "type": "none",
    "should_bypass_cache": false,
    "should_invalidate_previous_cache": false
  },
  "question_analysis": {
    "is_self_contained": false,
    "has_unresolved_references": true,
    "reference_types": ["pronoun"],
    "can_be_enhanced": false
  },
  "enhancement": {
    "enhanced_question": null,
    "context_used": []
  },
  "cache_decision": {
    "is_cacheable": false,
    "reason": "Cannot determine what 'it' refers to without broader context",
    "cache_key_question": null
  }
}
```

### Question Types Reference

#### Questions Worth Caching

| Category | Example | Reason |
|----------|---------|--------|
| **Self-contained questions** | "What is the Q3 2024 revenue?" | Complete, no context needed |
| **Specific entity queries** | "Show me John Smith's sales data" | Clear subject and intent |
| **Aggregation questions** | "What is the total inventory count?" | Standalone, reusable |
| **Definition questions** | "What are the KPIs in this report?" | General, context-free |
| **Enhanced questions** | "What about Q4?" → "What is Q4 2024 revenue?" | Context resolved |

#### Questions NOT Worth Caching

| Category | Example | Reason |
|----------|---------|--------|
| **Unresolvable context** | "What about the second one?" | Can't determine referent |
| **Meta-conversation** | "What did I ask before?" | About conversation, not data |
| **Clarification requests** | "What do you mean by that?" | Meaningless without context |
| **Highly temporal** | "What's happening right now?" | Changes constantly |

### Dissatisfaction Signals Reference

| Signal Type | Example Phrases | Action |
|-------------|-----------------|--------|
| **Incorrect result** | "That's wrong", "Not correct", "That's not right" | Bypass cache, fresh query |
| **Unclear response** | "I don't understand", "Can you clarify?", "What does that mean?" | Re-process with more detail |
| **Request refresh** | "Refresh", "Try again", "Get latest data" | Bypass cache, fresh query |
| **Double-check** | "Are you sure?", "Double check that", "Verify this" | Bypass cache, fresh query |
| **Quality complaint** | "That doesn't look right", "Check again", "That seems off" | Bypass cache, fresh query |
| **Explicit rejection** | "No, that's not what I asked", "Wrong answer" | Re-analyze question, fresh query |

### Cache Invalidation on Dissatisfaction

When the unified analysis detects `should_invalidate_previous_cache: true`:

```
┌─────────────────────────────────────────────────────────────────┐
│  Previous interaction:                                          │
│  User: "What is Q3 revenue?"                                    │
│  System: "$1.2M" (from cache)                                   │
│  User: "That's wrong, check again"  ← DISSATISFACTION          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Unified Analysis Output:                                       │
│  {                                                              │
│    "dissatisfaction": {                                         │
│      "is_dissatisfied": true,                                   │
│      "should_invalidate_previous_cache": true                   │
│    }                                                            │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  System Response:                                               │
│                                                                 │
│  1. Mark/invalidate the cached entry that was returned         │
│     (soft-delete or reduce confidence score)                    │
│                                                                 │
│  2. Execute fresh query (bypass cache):                        │
│     - Query documents/database directly                        │
│     - Generate new answer with LLM                             │
│                                                                 │
│  3. Return fresh result: "$1.5M" (corrected)                   │
│                                                                 │
│  4. Store NEW cache entry with corrected answer                │
│     (replaces the incorrect one)                               │
└─────────────────────────────────────────────────────────────────┘
```

### Complete Optimized Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Message Received                        │
│                     + Chat History Context                       │
│                     + Previous System Response                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: UNIFIED PRE-CACHE ANALYSIS (Single LLM Call)           │
│                                                                  │
│  Analyzes in ONE call:                                          │
│  ✓ Dissatisfaction detection                                    │
│  ✓ Question self-containment check                              │
│  ✓ Question enhancement (if needed)                             │
│  ✓ Cache worthiness decision                                    │
│                                                                  │
│  Output: unified_analysis JSON                                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: ROUTE BASED ON UNIFIED ANALYSIS                        │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  IF unified_analysis.dissatisfaction.is_dissatisfied:   │    │
│  │      → Invalidate previous cache entry (if applicable)  │    │
│  │      → Go to STEP 4 (Fresh Query)                       │    │
│  │                                                          │    │
│  │  ELIF unified_analysis.cache_decision.is_cacheable:     │    │
│  │      cache_key = unified_analysis.cache_decision        │    │
│  │                   .cache_key_question                   │    │
│  │      → Go to STEP 3 (Cache Lookup)                      │    │
│  │                                                          │    │
│  │  ELSE:                                                   │    │
│  │      → Go to STEP 4 (Fresh Query, no caching)           │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                                │
           ┌────────────────────┼────────────────────┐
           │                    │                    │
           ▼                    ▼                    ▼
    ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
    │ Dissatisfied│     │  Cacheable  │     │ Not Cacheable   │
    │ → Fresh     │     │  → Lookup   │     │ → Fresh Query   │
    │   Query     │     │    Cache    │     │   (no cache)    │
    └─────────────┘     └──────┬──────┘     └─────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: CACHE LOOKUP (using cache_key_question)                │
│                                                                  │
│  - Generate embedding for cache_key_question                    │
│  - Semantic search in Qdrant cache collection                   │
│  - For each candidate:                                          │
│      → Check permission (user can READ all source docs?)        │
│      → If YES: Return cached answer (CACHE HIT)                 │
│      → If NO: Try next candidate                                │
│  - If no valid candidate: CACHE MISS → Go to STEP 4             │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: FRESH QUERY PROCESSING                                 │
│                                                                  │
│  - Execute full query pipeline (intent → query → retrieve)     │
│  - Generate answer with LLM                                     │
│  - Return to user IMMEDIATELY                                   │
│                                                                  │
│  ⚡ User response time ends here                                │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 5: BACKGROUND CACHE STORAGE (if cacheable, async)         │
│                                                                  │
│  Only if unified_analysis.cache_decision.is_cacheable == true   │
│                                                                  │
│  - Store cache_key_question (enhanced or original)              │
│  - Store source document IDs for permission checks              │
│  - Generate and store question embedding                        │
│  - Fire-and-forget (non-blocking)                               │
└─────────────────────────────────────────────────────────────────┘
```

### LLM Call Count Summary

| Scenario | Old Design | Optimized Design |
|----------|------------|------------------|
| User dissatisfied | 1 LLM call (detection) | 1 LLM call (unified) |
| Self-contained, cache hit | 2 LLM calls (detection + worthiness) | 1 LLM call (unified) |
| Context-dependent, enhanceable | 3 LLM calls (detection + worthiness + enhance) | 1 LLM call (unified) |
| Not cacheable | 2 LLM calls (detection + worthiness) | 1 LLM call (unified) |

**Result: Always 1 LLM call for pre-cache analysis, regardless of scenario.**

### Cache Entry Confidence Scoring

Cache entries can have confidence scores that affect their usage:

```
Cache Entry:
{
  "question": "What is Q3 revenue?",
  "answer": "$1.5M",
  "confidence_score": 0.95,  // High confidence
  "negative_feedback_count": 0,
  "created_at": "...",
  "last_validated": "..."
}

On User Dissatisfaction:
{
  "confidence_score": 0.95 → 0.70,  // Reduced
  "negative_feedback_count": 0 → 1,
  "needs_revalidation": true
}

Threshold Rules:
- confidence_score < 0.5: Don't use this cache entry
- negative_feedback_count >= 3: Auto-invalidate entry
```

### Decision Matrix

| Question Type | Cacheable? | Cache Key | Action |
|--------------|------------|-----------|--------|
| Self-contained | ✅ Yes | Original question | Normal cache flow |
| Context-dependent, enhanceable | ✅ Yes | Enhanced question | Enhance, then cache |
| Context-dependent, not enhanceable | ❌ No | N/A | Process, don't cache |
| Dissatisfaction expressed | ❌ No | N/A | Bypass cache, fresh query |
| Refresh/retry request | ❌ No | N/A | Bypass cache, fresh query |
| Follow-up with pronouns | ⚠️ Maybe | Enhanced if possible | Enhance or skip |

### Cache Entry Confidence Scoring

Cache entries can have confidence scores that affect their usage:

```
Cache Entry:
{
  "question": "What is Q3 revenue?",
  "answer": "$1.5M",
  "confidence_score": 0.95,  // High confidence
  "negative_feedback_count": 0,
  "created_at": "...",
  "last_validated": "..."
}

On User Dissatisfaction:
{
  "confidence_score": 0.95 → 0.70,  // Reduced
  "negative_feedback_count": 0 → 1,
  "needs_revalidation": true
}

Threshold Rules:
- confidence_score < 0.5: Don't use this cache entry
- negative_feedback_count >= 3: Auto-invalidate entry
```

## Security-First Cache Design

### Core Security Principle

**CRITICAL**: A cached answer can be returned for a semantically similar question, but ONLY if the current user has **READ ACCESS PERMISSION** to ALL source documents that were used to generate that cached answer.

This prevents **information leakage** where one user could receive cached answers generated from documents they are not authorized to access.

### The Security Problem

```
Scenario WITHOUT Permission Check (DANGEROUS):

User A (has access to confidential_salary.pdf):
  Question: "What is the CEO salary?"
  → System generates answer from confidential_salary.pdf
  → Answer cached: "CEO salary is $5M"

User B (NO access to confidential_salary.pdf):
  Question: "What's the CEO's salary?"  ← Semantically similar
  → Cache HIT (similar question found)
  → Returns: "CEO salary is $5M"  ← SECURITY BREACH! User B sees confidential data
```

### The Solution: Permission-Based Cache Validation

```
Scenario WITH Permission Check (SECURE):

User A (has access to confidential_salary.pdf):
  Question: "What is the CEO salary?"
  → System generates answer from confidential_salary.pdf
  → Answer cached with source_document_ids: ["confidential_salary.pdf"]

User B (NO access to confidential_salary.pdf):
  Question: "What's the CEO's salary?"  ← Semantically similar
  → Cache search finds similar question
  → Permission check: Can User B read "confidential_salary.pdf"? NO
  → Cache MISS (permission denied)
  → System processes question using only documents User B can access
  → Different answer (or no answer) returned
```

### Cache Hit Requirements

A cache entry can be returned ONLY when ALL conditions are met:

| # | Requirement | Description |
|---|-------------|-------------|
| 1 | **Semantic Similarity** | Question similarity score > threshold (e.g., 0.90) |
| 2 | **Not Expired** | Cache entry TTL has not expired |
| 3 | **Same Workspace** | Cache entry belongs to same workspace |
| 4 | **PERMISSION CHECK** | Current user has READ access to ALL source documents used in cached answer |

### Permission Check Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cache Search Process                          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Semantic Search: Find similar questions in cache            │
│     (returns multiple candidates sorted by similarity)          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. For each candidate (highest similarity first):              │
│     ┌─────────────────────────────────────────────────────┐     │
│     │  Get source_document_ids from cached entry          │     │
│     │  ["doc_A", "doc_B", "doc_C"]                         │     │
│     └─────────────────────────────────────────────────────┘     │
│                          │                                       │
│                          ▼                                       │
│     ┌─────────────────────────────────────────────────────┐     │
│     │  PERMISSION CHECK:                                   │     │
│     │  Can current_user READ all of these documents?      │     │
│     │                                                      │     │
│     │  user_accessible_docs = get_user_documents(user_id) │     │
│     │  required_docs = cached_entry.source_document_ids   │     │
│     │                                                      │     │
│     │  if required_docs ⊆ user_accessible_docs:           │     │
│     │      → PERMISSION GRANTED → Return cached answer    │     │
│     │  else:                                               │     │
│     │      → PERMISSION DENIED → Try next candidate       │     │
│     └─────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. If no candidates pass permission check:                     │
│     → Cache MISS                                                │
│     → Process question normally with user's accessible docs     │
└─────────────────────────────────────────────────────────────────┘
```

### Why NOT Use Document IDs as Composite Key?

Previous approach: `Cache Key = Hash(question + document_ids)`
- Required EXACT document match
- Too restrictive - low cache hit rate
- User A and User B with same documents couldn't share cache

**New approach**: Question similarity + Permission validation at query time
- Higher cache hit rate for users with overlapping document access
- Flexible - similar questions can match
- Secure - permission checked on every cache read
- Users with same document access can benefit from shared cache

### Cache Sharing Between Users

```
Document Access Matrix:
┌──────────────────────────────────────────────────────┐
│         │ doc_A │ doc_B │ doc_C │ doc_D (confidential)│
├──────────────────────────────────────────────────────┤
│ User A  │   ✓   │   ✓   │   ✓   │         ✓          │
│ User B  │   ✓   │   ✓   │   ✓   │         ✗          │
│ User C  │   ✓   │   ✗   │   ✗   │         ✗          │
└──────────────────────────────────────────────────────┘

Cache Entry 1:
  Question: "What is the Q3 revenue?"
  Source Documents: [doc_A, doc_B]
  Answer: "$1.5M"

Cache Entry 2:
  Question: "What is the CEO compensation?"
  Source Documents: [doc_D]
  Answer: "$5M salary"

Query Results:
┌─────────────────────────────────────────────────────────────────┐
│ User A asks: "What's Q3 revenue?"                               │
│ → Cache search finds Entry 1 (similar question)                 │
│ → Permission check: User A can read [doc_A, doc_B]? YES         │
│ → Cache HIT → Returns "$1.5M"                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ User B asks: "What's Q3 revenue?"                               │
│ → Cache search finds Entry 1 (similar question)                 │
│ → Permission check: User B can read [doc_A, doc_B]? YES         │
│ → Cache HIT → Returns "$1.5M" (shared cache benefit!)           │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ User C asks: "What's Q3 revenue?"                               │
│ → Cache search finds Entry 1 (similar question)                 │
│ → Permission check: User C can read [doc_A, doc_B]? NO (missing doc_B)
│ → Cache MISS → Process with User C's accessible documents only  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ User B asks: "What's the CEO salary?"                           │
│ → Cache search finds Entry 2 (similar question)                 │
│ → Permission check: User B can read [doc_D]? NO                 │
│ → Cache MISS → No answer (User B has no relevant documents)     │
│ → SECURITY PROTECTED: Confidential data not leaked              │
└─────────────────────────────────────────────────────────────────┘
```

### Performance Consideration

Permission checking adds a small overhead, but it's essential for security:

| Operation | Time | Notes |
|-----------|------|-------|
| Semantic search | 10-50ms | Vector similarity search in Qdrant |
| Permission check | 1-10ms | Check user's document access list |
| Total cache lookup | 11-60ms | Still much faster than full processing (2-5s) |

**Optimization**: Pre-load user's accessible document IDs at session start and cache in memory.

## Qdrant Collection Schema

### Collection Name Pattern

```
query_cache_{workspace_id}
```

Each workspace has its own cache collection to ensure data isolation and enable workspace-specific cache management.

### Vector Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Vector Size | 1536 (OpenAI) or 384 (all-MiniLM) | Match your embedding model dimension |
| Distance Metric | Cosine | Best for semantic similarity |
| On Disk | true | For large cache sizes |
| HNSW m | 16 | Connections per element |
| HNSW ef_construct | 100 | Construction time accuracy |

### Payload Schema

```json
{
  "id": "uuid",

  "question_data": {
    "original_question": "What is the total revenue for Q3 2024?",
    "normalized_question": "what is the total revenue for q3 2024",
    "question_hash": "sha256_hash_of_normalized_question_only"
  },

  "source_documents": {
    "document_ids": ["doc_1", "doc_2"],
    "document_ids_hash": "sha256_hash_of_sorted_document_ids",
    "document_count": 2,
    "document_hashes": {
      "doc_1": "content_hash_abc",
      "doc_2": "content_hash_def"
    }
  },

  "composite_key": {
    "key_hash": "sha256_hash_of_question_plus_doc_ids",
    "key_components": "question_hash|document_ids_hash"
  },

  "answer": "The total revenue for Q3 2024 is $1,234,567...",
  "answer_metadata": {
    "sources": [
      {"document_id": "doc_1", "chunk_ids": ["chunk_1", "chunk_2"], "relevance_score": 0.95},
      {"document_id": "doc_2", "chunk_ids": ["chunk_5"], "relevance_score": 0.88}
    ],
    "sql_query": "SELECT SUM(revenue) FROM sales WHERE quarter = 'Q3' AND year = 2024",
    "charts": [],
    "confidence": 0.95
  },

  "intent": "data_query",
  "query_type": "aggregation",

  "context": {
    "workspace_id": "workspace_123",
    "dataset_id": "dataset_456",
    "table_names": ["sales", "products"]
  },

  "versioning": {
    "data_version": "v1.2.3",
    "cache_version": "1.0"
  },

  "timestamps": {
    "created_at": 1735550400.0,
    "last_accessed_at": 1735550400.0,
    "expires_at": 1735636800.0
  },

  "stats": {
    "hit_count": 0,
    "avg_response_time_ms": 2500
  },

  "ttl_seconds": 86400,
  "is_valid": true
}
```

### Payload Indexes

Create indexes on frequently filtered fields for optimal query performance:

```python
indexes = [
    # Composite key for exact matching (PRIMARY)
    ("composite_key.key_hash", PayloadSchemaType.KEYWORD),

    # Question hash for fast exact question lookup
    ("question_data.question_hash", PayloadSchemaType.KEYWORD),

    # Document IDs hash for filtering by exact document set
    ("source_documents.document_ids_hash", PayloadSchemaType.KEYWORD),

    # Individual document IDs for invalidation queries
    ("source_documents.document_ids", PayloadSchemaType.KEYWORD),

    # Other filters
    ("intent", PayloadSchemaType.KEYWORD),
    ("context.workspace_id", PayloadSchemaType.KEYWORD),
    ("context.dataset_id", PayloadSchemaType.KEYWORD),
    ("context.table_names", PayloadSchemaType.KEYWORD),
    ("timestamps.expires_at", PayloadSchemaType.FLOAT),
    ("is_valid", PayloadSchemaType.BOOL)
]
```

## Configuration

### Cache Settings

```python
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class QueryCacheConfig:
    """Configuration for query cache behavior."""

    # Similarity thresholds by query type
    similarity_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "document_qa": 0.92,      # High threshold for document questions
        "data_query": 0.90,       # Slightly lower for data queries
        "chart_generation": 0.88, # Lower for chart requests
        "general": 0.85,          # Lowest for general questions
        "default": 0.90
    })

    # TTL settings by query type (in seconds)
    ttl_settings: Dict[str, int] = field(default_factory=lambda: {
        "document_qa": 604800,      # 7 days - documents rarely change
        "data_query": 3600,         # 1 hour - data may change frequently
        "chart_generation": 86400,  # 24 hours
        "general": 2592000,         # 30 days - static knowledge
        "default": 86400            # 24 hours default
    })

    # Cache size limits
    max_cache_entries_per_workspace: int = 10000
    max_answer_length: int = 50000  # characters

    # Cleanup settings
    cleanup_batch_size: int = 1000
    cleanup_interval_hours: int = 6

    # Feature flags
    enable_partial_caching: bool = True  # Cache SQL but re-execute
    enable_cache_warming: bool = False   # Pre-populate common queries

    # Composite key settings
    require_exact_document_match: bool = True  # MUST be True for composite key logic

    # Search settings
    search_limit: int = 5  # Number of candidates to evaluate
    ef_search: int = 128   # HNSW search accuracy parameter
```

## Core Implementation

### 1. Query Cache Manager

```python
"""
Query Cache Manager - Handles all cache operations with Qdrant.

Key Feature: Composite key matching (question + source documents)
A cache hit requires BOTH semantic question similarity AND exact source document match.
"""

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue, Range, FilterSelector,
    UpdateStatus, PayloadSchemaType, CreateAliasOperation,
    OptimizersConfigDiff, HnswConfigDiff
)


class QueryCacheManager:
    """
    Manages semantic query caching using Qdrant with composite key matching.

    Features:
    - Composite key: question + source documents must both match
    - Semantic similarity search for questions
    - Exact match requirement for source documents
    - TTL-based expiration
    - Event-driven invalidation
    - Hit count tracking
    - Workspace isolation
    """

    def __init__(
        self,
        qdrant_client: QdrantClient,
        embedding_service: Any,  # Your embedding service
        config: QueryCacheConfig = None
    ):
        self.client = qdrant_client
        self.embedding_service = embedding_service
        self.config = config or QueryCacheConfig()

    def _get_collection_name(self, workspace_id: str) -> str:
        """Generate collection name for workspace."""
        return f"query_cache_{workspace_id}"

    def _normalize_question(self, question: str) -> str:
        """Normalize question for consistent matching."""
        normalized = question.lower().strip()
        normalized = " ".join(normalized.split())
        return normalized

    def _hash_question(self, normalized_question: str) -> str:
        """Generate hash for question only."""
        return hashlib.sha256(normalized_question.encode()).hexdigest()

    def _hash_document_ids(self, document_ids: List[str]) -> str:
        """
        Generate hash for document IDs.
        Documents are sorted to ensure consistent hashing regardless of order.
        """
        if not document_ids:
            return hashlib.sha256(b"__no_documents__").hexdigest()

        # Sort document IDs for consistent ordering
        sorted_ids = sorted(document_ids)
        combined = "|".join(sorted_ids)
        return hashlib.sha256(combined.encode()).hexdigest()

    def _generate_composite_key(
        self,
        normalized_question: str,
        document_ids: List[str]
    ) -> str:
        """
        Generate composite key from question and source documents.

        This is the PRIMARY cache key - both question and documents must match.

        Args:
            normalized_question: Normalized question text
            document_ids: List of source document IDs

        Returns:
            SHA256 hash of the composite key
        """
        question_hash = self._hash_question(normalized_question)
        doc_ids_hash = self._hash_document_ids(document_ids)

        # Combine both hashes
        composite = f"{question_hash}|{doc_ids_hash}"
        return hashlib.sha256(composite.encode()).hexdigest()

    async def initialize_collection(self, workspace_id: str) -> bool:
        """
        Create and configure cache collection for a workspace.

        Args:
            workspace_id: The workspace identifier

        Returns:
            True if collection was created or already exists
        """
        collection_name = self._get_collection_name(workspace_id)

        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if exists:
            return True

        # Create collection with optimized settings for cache use case
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=self.embedding_service.dimension,
                distance=Distance.COSINE,
                on_disk=True
            ),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10000
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000,
                memmap_threshold=50000
            )
        )

        # Create payload indexes for efficient filtering
        index_configs = [
            # PRIMARY: Composite key for exact matching
            ("composite_key.key_hash", PayloadSchemaType.KEYWORD),

            # Question and document hashes
            ("question_data.question_hash", PayloadSchemaType.KEYWORD),
            ("source_documents.document_ids_hash", PayloadSchemaType.KEYWORD),
            ("source_documents.document_ids", PayloadSchemaType.KEYWORD),

            # Other filters
            ("intent", PayloadSchemaType.KEYWORD),
            ("context.workspace_id", PayloadSchemaType.KEYWORD),
            ("context.dataset_id", PayloadSchemaType.KEYWORD),
            ("context.table_names", PayloadSchemaType.KEYWORD),
            ("timestamps.expires_at", PayloadSchemaType.FLOAT),
            ("is_valid", PayloadSchemaType.BOOL)
        ]

        for field_name, field_type in index_configs:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_type
            )

        return True

    async def search_cache(
        self,
        question: str,
        workspace_id: str,
        source_document_ids: List[str],
        dataset_id: Optional[str] = None,
        intent: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Search for a cached answer using composite key (question + source documents).

        IMPORTANT: A cache hit requires BOTH:
        1. Semantic similarity of the question (above threshold)
        2. EXACT match of source document IDs

        Args:
            question: The user's question
            workspace_id: Workspace identifier
            source_document_ids: List of document IDs used as sources (REQUIRED for matching)
            dataset_id: Optional dataset context
            intent: Optional pre-classified intent

        Returns:
            Cached answer dict if found, None otherwise
        """
        collection_name = self._get_collection_name(workspace_id)
        current_time = time.time()

        # Validate source documents are provided
        if source_document_ids is None:
            source_document_ids = []

        # Check if collection exists
        try:
            collection_info = self.client.get_collection(collection_name)
        except Exception:
            return None

        # Normalize question and generate keys
        normalized = self._normalize_question(question)
        question_hash = self._hash_question(normalized)
        doc_ids_hash = self._hash_document_ids(source_document_ids)
        composite_key = self._generate_composite_key(normalized, source_document_ids)

        # ============================================================
        # Step 1: Try EXACT composite key match first (fastest path)
        # ============================================================
        exact_match = await self._search_exact_composite_match(
            collection_name=collection_name,
            composite_key=composite_key,
            current_time=current_time
        )

        if exact_match:
            await self._update_hit_stats(collection_name, exact_match["id"])
            return exact_match

        # ============================================================
        # Step 2: Semantic search with EXACT document filter
        # ============================================================
        question_embedding = await self.embedding_service.embed_text(normalized)

        # Build filter conditions - MUST include exact document match
        must_conditions = [
            # Filter by exact document set (using hash)
            FieldCondition(
                key="source_documents.document_ids_hash",
                match=MatchValue(value=doc_ids_hash)
            ),
            # Valid and not expired
            FieldCondition(
                key="is_valid",
                match=MatchValue(value=True)
            ),
            FieldCondition(
                key="timestamps.expires_at",
                range=Range(gt=current_time)
            )
        ]

        # Add optional context filters
        if dataset_id:
            must_conditions.append(
                FieldCondition(
                    key="context.dataset_id",
                    match=MatchValue(value=dataset_id)
                )
            )

        if intent:
            must_conditions.append(
                FieldCondition(
                    key="intent",
                    match=MatchValue(value=intent)
                )
            )

        # Perform semantic search with document filter
        results = self.client.search(
            collection_name=collection_name,
            query_vector=question_embedding,
            query_filter=Filter(must=must_conditions),
            limit=self.config.search_limit,
            with_payload=True,
            score_threshold=self._get_similarity_threshold(intent)
        )

        if not results:
            return None

        # Get best match and verify document IDs exactly match
        best_match = results[0]

        # Double-check: Verify the cached document IDs exactly match
        cached_doc_ids = best_match.payload.get("source_documents", {}).get("document_ids", [])
        if not self._document_ids_match(source_document_ids, cached_doc_ids):
            # Documents don't match exactly - this is a cache miss
            return None

        # Update hit statistics
        await self._update_hit_stats(collection_name, best_match.id)

        return {
            "id": best_match.id,
            "answer": best_match.payload["answer"],
            "answer_metadata": best_match.payload.get("answer_metadata", {}),
            "original_question": best_match.payload["question_data"]["original_question"],
            "similarity_score": best_match.score,
            "intent": best_match.payload["intent"],
            "source_document_ids": cached_doc_ids,
            "cache_hit": True,
            "exact_match": False,
            "cached_at": best_match.payload["timestamps"]["created_at"]
        }

    def _document_ids_match(
        self,
        requested_ids: List[str],
        cached_ids: List[str]
    ) -> bool:
        """
        Check if document IDs match exactly (order-independent).

        Args:
            requested_ids: Document IDs from current request
            cached_ids: Document IDs stored in cache

        Returns:
            True if sets are identical, False otherwise
        """
        return set(requested_ids or []) == set(cached_ids or [])

    async def _search_exact_composite_match(
        self,
        collection_name: str,
        composite_key: str,
        current_time: float
    ) -> Optional[Dict[str, Any]]:
        """
        Search for exact composite key match (question + documents).

        This is the fastest path - direct lookup by composite key hash.
        """
        results = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="composite_key.key_hash",
                        match=MatchValue(value=composite_key)
                    ),
                    FieldCondition(
                        key="is_valid",
                        match=MatchValue(value=True)
                    ),
                    FieldCondition(
                        key="timestamps.expires_at",
                        range=Range(gt=current_time)
                    )
                ]
            ),
            limit=1,
            with_payload=True
        )

        if results[0]:
            point = results[0][0]
            return {
                "id": point.id,
                "answer": point.payload["answer"],
                "answer_metadata": point.payload.get("answer_metadata", {}),
                "original_question": point.payload["question_data"]["original_question"],
                "similarity_score": 1.0,
                "intent": point.payload["intent"],
                "source_document_ids": point.payload["source_documents"]["document_ids"],
                "cache_hit": True,
                "exact_match": True,
                "cached_at": point.payload["timestamps"]["created_at"]
            }

        return None

    def _get_similarity_threshold(self, intent: Optional[str]) -> float:
        """Get similarity threshold based on intent type."""
        if intent and intent in self.config.similarity_thresholds:
            return self.config.similarity_thresholds[intent]
        return self.config.similarity_thresholds["default"]

    def _get_ttl(self, intent: Optional[str]) -> int:
        """Get TTL based on intent type."""
        if intent and intent in self.config.ttl_settings:
            return self.config.ttl_settings[intent]
        return self.config.ttl_settings["default"]

    async def _update_hit_stats(self, collection_name: str, point_id: str) -> None:
        """Update cache hit statistics."""
        try:
            points = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True
            )

            if points:
                current_hits = points[0].payload.get("stats", {}).get("hit_count", 0)

                self.client.set_payload(
                    collection_name=collection_name,
                    payload={
                        "stats.hit_count": current_hits + 1,
                        "timestamps.last_accessed_at": time.time()
                    },
                    points=[point_id]
                )
        except Exception as e:
            # Non-critical operation, log and continue
            print(f"Failed to update hit stats: {e}")

    async def store_cache(
        self,
        question: str,
        answer: str,
        workspace_id: str,
        source_document_ids: List[str],
        intent: str,
        dataset_id: Optional[str] = None,
        table_names: Optional[List[str]] = None,
        answer_metadata: Optional[Dict[str, Any]] = None,
        document_content_hashes: Optional[Dict[str, str]] = None,
        data_version: Optional[str] = None,
        ttl_override: Optional[int] = None
    ) -> str:
        """
        Store a question-answer pair in the cache with composite key.

        Args:
            question: The original user question
            answer: The generated answer
            workspace_id: Workspace identifier
            source_document_ids: Document IDs used to generate the answer (REQUIRED)
            intent: Classified intent type
            dataset_id: Optional dataset context
            table_names: Optional table names referenced
            answer_metadata: Additional answer metadata (SQL, sources, etc.)
            document_content_hashes: Hashes of document contents for version tracking
            data_version: Version string for data freshness
            ttl_override: Custom TTL in seconds

        Returns:
            The cache entry ID
        """
        collection_name = self._get_collection_name(workspace_id)

        # Ensure collection exists
        await self.initialize_collection(workspace_id)

        # Ensure source_document_ids is a list
        if source_document_ids is None:
            source_document_ids = []

        # Prepare question data
        normalized = self._normalize_question(question)
        question_hash = self._hash_question(normalized)
        doc_ids_hash = self._hash_document_ids(source_document_ids)
        composite_key = self._generate_composite_key(normalized, source_document_ids)

        # Generate question embedding
        question_embedding = await self.embedding_service.embed_text(normalized)

        # Calculate timestamps
        current_time = time.time()
        ttl = ttl_override or self._get_ttl(intent)
        expires_at = current_time + ttl

        # Generate unique ID
        point_id = str(uuid.uuid4())

        # Build payload with composite key structure
        payload = {
            "question_data": {
                "original_question": question,
                "normalized_question": normalized,
                "question_hash": question_hash
            },

            "source_documents": {
                "document_ids": sorted(source_document_ids),  # Store sorted for consistency
                "document_ids_hash": doc_ids_hash,
                "document_count": len(source_document_ids),
                "document_hashes": document_content_hashes or {}
            },

            "composite_key": {
                "key_hash": composite_key,
                "key_components": f"{question_hash}|{doc_ids_hash}"
            },

            "answer": answer,
            "answer_metadata": answer_metadata or {},

            "intent": intent,
            "query_type": answer_metadata.get("query_type", "unknown") if answer_metadata else "unknown",

            "context": {
                "workspace_id": workspace_id,
                "dataset_id": dataset_id,
                "table_names": table_names or []
            },

            "versioning": {
                "data_version": data_version,
                "cache_version": "1.0"
            },

            "timestamps": {
                "created_at": current_time,
                "last_accessed_at": current_time,
                "expires_at": expires_at
            },

            "stats": {
                "hit_count": 0,
                "avg_response_time_ms": answer_metadata.get("response_time_ms", 0) if answer_metadata else 0
            },

            "ttl_seconds": ttl,
            "is_valid": True
        }

        # Check cache size limit and cleanup if needed
        await self._enforce_cache_limit(collection_name, workspace_id)

        # Store in Qdrant
        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=question_embedding,
                    payload=payload
                )
            ]
        )

        return point_id

    async def _enforce_cache_limit(self, collection_name: str, workspace_id: str) -> None:
        """Enforce maximum cache size by removing old entries."""
        try:
            collection_info = self.client.get_collection(collection_name)
            current_count = collection_info.points_count

            if current_count >= self.config.max_cache_entries_per_workspace:
                entries_to_remove = current_count - self.config.max_cache_entries_per_workspace + 100

                results = self.client.scroll(
                    collection_name=collection_name,
                    limit=entries_to_remove,
                    with_payload=True,
                    order_by="timestamps.created_at"
                )

                if results[0]:
                    point_ids = [p.id for p in results[0]]
                    self.client.delete(
                        collection_name=collection_name,
                        points_selector=point_ids
                    )
        except Exception as e:
            print(f"Failed to enforce cache limit: {e}")

    async def invalidate_by_document(
        self,
        workspace_id: str,
        document_id: str
    ) -> int:
        """
        Invalidate ALL cache entries that reference a specific document.

        When a document is updated or deleted, ALL cache entries that used
        this document as a source must be invalidated.

        Args:
            workspace_id: Workspace identifier
            document_id: Document that was modified/deleted

        Returns:
            Number of entries invalidated
        """
        collection_name = self._get_collection_name(workspace_id)

        try:
            # Find all entries that include this document in their source documents
            results = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_documents.document_ids",
                            match=MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=10000,
                with_payload=False
            )

            if not results[0]:
                return 0

            point_ids = [p.id for p in results[0]]

            # Delete all matching entries
            self.client.delete(
                collection_name=collection_name,
                points_selector=point_ids
            )

            return len(point_ids)

        except Exception as e:
            print(f"Failed to invalidate by document: {e}")
            return 0

    async def invalidate_by_documents(
        self,
        workspace_id: str,
        document_ids: List[str]
    ) -> int:
        """
        Invalidate cache entries for multiple documents.

        Args:
            workspace_id: Workspace identifier
            document_ids: List of documents that were modified/deleted

        Returns:
            Total number of entries invalidated
        """
        total_invalidated = 0

        for doc_id in document_ids:
            count = await self.invalidate_by_document(workspace_id, doc_id)
            total_invalidated += count

        return total_invalidated

    async def invalidate_by_exact_document_set(
        self,
        workspace_id: str,
        document_ids: List[str]
    ) -> int:
        """
        Invalidate cache entries that use EXACTLY this set of documents.

        Use this when you want to invalidate only entries that used
        exactly these documents (not entries that used a subset/superset).

        Args:
            workspace_id: Workspace identifier
            document_ids: Exact set of document IDs

        Returns:
            Number of entries invalidated
        """
        collection_name = self._get_collection_name(workspace_id)
        doc_ids_hash = self._hash_document_ids(document_ids)

        try:
            result = self.client.delete(
                collection_name=collection_name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="source_documents.document_ids_hash",
                                match=MatchValue(value=doc_ids_hash)
                            )
                        ]
                    )
                )
            )

            return result.status == UpdateStatus.COMPLETED

        except Exception as e:
            print(f"Failed to invalidate by exact document set: {e}")
            return 0

    async def invalidate_by_dataset(
        self,
        workspace_id: str,
        dataset_id: str
    ) -> int:
        """
        Invalidate all cache entries related to a dataset.

        Args:
            workspace_id: Workspace identifier
            dataset_id: Dataset that was modified

        Returns:
            Number of entries invalidated
        """
        collection_name = self._get_collection_name(workspace_id)

        try:
            result = self.client.delete(
                collection_name=collection_name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="context.dataset_id",
                                match=MatchValue(value=dataset_id)
                            )
                        ]
                    )
                )
            )

            return result.status == UpdateStatus.COMPLETED

        except Exception as e:
            print(f"Failed to invalidate by dataset: {e}")
            return 0

    async def invalidate_by_table(
        self,
        workspace_id: str,
        table_name: str
    ) -> int:
        """
        Invalidate all cache entries related to a table.

        Args:
            workspace_id: Workspace identifier
            table_name: Table that was modified

        Returns:
            Number of entries invalidated
        """
        collection_name = self._get_collection_name(workspace_id)

        try:
            result = self.client.delete(
                collection_name=collection_name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="context.table_names",
                                match=MatchValue(value=table_name)
                            )
                        ]
                    )
                )
            )

            return result.status == UpdateStatus.COMPLETED

        except Exception as e:
            print(f"Failed to invalidate by table: {e}")
            return 0

    async def cleanup_expired(self, workspace_id: str) -> int:
        """
        Remove all expired cache entries.

        Args:
            workspace_id: Workspace identifier

        Returns:
            Number of entries removed
        """
        collection_name = self._get_collection_name(workspace_id)
        current_time = time.time()

        try:
            result = self.client.delete(
                collection_name=collection_name,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="timestamps.expires_at",
                                range=Range(lt=current_time)
                            )
                        ]
                    )
                )
            )

            return result.status == UpdateStatus.COMPLETED

        except Exception as e:
            print(f"Failed to cleanup expired entries: {e}")
            return 0

    async def cleanup_all_workspaces(self) -> Dict[str, int]:
        """
        Cleanup expired entries across all workspace caches.

        Returns:
            Dict mapping workspace_id to number of entries cleaned
        """
        results = {}

        collections = self.client.get_collections().collections
        cache_collections = [c.name for c in collections if c.name.startswith("query_cache_")]

        for collection_name in cache_collections:
            workspace_id = collection_name.replace("query_cache_", "")
            count = await self.cleanup_expired(workspace_id)
            results[workspace_id] = count

        return results

    async def get_cache_stats(self, workspace_id: str) -> Dict[str, Any]:
        """
        Get statistics about the cache for a workspace.

        Args:
            workspace_id: Workspace identifier

        Returns:
            Cache statistics including composite key info
        """
        collection_name = self._get_collection_name(workspace_id)

        try:
            collection_info = self.client.get_collection(collection_name)
            current_time = time.time()

            # Count valid entries
            valid_results = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="is_valid",
                            match=MatchValue(value=True)
                        ),
                        FieldCondition(
                            key="timestamps.expires_at",
                            range=Range(gt=current_time)
                        )
                    ]
                ),
                limit=1,
                with_payload=False
            )

            return {
                "workspace_id": workspace_id,
                "total_entries": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "status": collection_info.status,
                "max_entries": self.config.max_cache_entries_per_workspace,
                "utilization_percent": (collection_info.points_count / self.config.max_cache_entries_per_workspace) * 100,
                "composite_key_enabled": True
            }

        except Exception as e:
            return {
                "workspace_id": workspace_id,
                "error": str(e)
            }

    async def clear_cache(self, workspace_id: str) -> bool:
        """
        Clear all cache entries for a workspace.

        Args:
            workspace_id: Workspace identifier

        Returns:
            True if successful
        """
        collection_name = self._get_collection_name(workspace_id)

        try:
            self.client.delete_collection(collection_name)
            await self.initialize_collection(workspace_id)
            return True
        except Exception as e:
            print(f"Failed to clear cache: {e}")
            return False
```

### 2. Integration with Analytics Service

```python
"""
Integration example for analytics_service.py with composite key caching.

IMPORTANT: Cache storage is performed in the BACKGROUND and does NOT block
the user response. The user receives their answer immediately while caching
happens asynchronously.
"""

import asyncio
import time
import logging
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class AnalyticsService:
    def __init__(self, ...):
        # ... existing initialization ...

        # Initialize cache manager
        self.cache_manager = QueryCacheManager(
            qdrant_client=self.qdrant_client,
            embedding_service=self.embedding_service,
            config=QueryCacheConfig()
        )

        # Background task executor for non-blocking cache operations
        self._background_executor = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix="cache_worker"
        )

        # Track background tasks for graceful shutdown
        self._background_tasks: set = set()

    async def route_and_process(
        self,
        question: str,
        workspace_id: str,
        source_document_ids: List[str],  # REQUIRED for composite key
        dataset_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main entry point with composite key cache integration.

        Cache lookup uses: question similarity + EXACT source document match

        IMPORTANT: Cache storage is NON-BLOCKING - user gets response immediately
        while cache is updated in the background.
        """
        start_time = time.time()

        # ============================================================
        # Step 1: Check cache with composite key (question + documents)
        # This is a fast read operation, acceptable to await
        # ============================================================
        cached_result = await self.cache_manager.search_cache(
            question=question,
            workspace_id=workspace_id,
            source_document_ids=source_document_ids,
            dataset_id=dataset_id
        )

        if cached_result:
            # Cache hit - both question and documents matched
            cached_result["response_time_ms"] = (time.time() - start_time) * 1000
            cached_result["from_cache"] = True
            return cached_result

        # ============================================================
        # Step 2: Cache miss - process normally
        # ============================================================
        result = await self._process_question(
            question=question,
            workspace_id=workspace_id,
            source_document_ids=source_document_ids,
            dataset_id=dataset_id,
            **kwargs
        )

        response_time_ms = (time.time() - start_time) * 1000

        # ============================================================
        # Step 3: Store result in cache - BACKGROUND (NON-BLOCKING)
        # User receives response immediately, cache update happens async
        # ============================================================
        if result.get("success", False) and result.get("confidence", 0) > 0.7:
            actual_source_docs = result.get("used_document_ids", source_document_ids)

            # Fire-and-forget background task - does NOT block user response
            self._schedule_background_cache_store(
                question=question,
                answer=result.get("answer", ""),
                workspace_id=workspace_id,
                source_document_ids=actual_source_docs,
                intent=result.get("intent", "general"),
                dataset_id=dataset_id,
                table_names=result.get("table_names", []),
                answer_metadata={
                    "sql_query": result.get("sql_query"),
                    "sources": result.get("sources", []),
                    "confidence": result.get("confidence"),
                    "response_time_ms": response_time_ms,
                    "query_type": result.get("query_type")
                }
            )

        # Return immediately to user - cache storage continues in background
        result["response_time_ms"] = response_time_ms
        result["from_cache"] = False
        return result

    def _schedule_background_cache_store(self, **cache_params):
        """
        Schedule cache storage as a background task.
        Does NOT block the calling coroutine.
        """
        task = asyncio.create_task(
            self._background_cache_store_safe(**cache_params)
        )

        # Track task for cleanup
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _background_cache_store_safe(self, **cache_params):
        """
        Safely store cache entry in background with error handling.
        Failures are logged but do not affect user experience.
        """
        try:
            await self.cache_manager.store_cache(**cache_params)
            logger.debug(
                f"Background cache store completed for workspace "
                f"{cache_params.get('workspace_id')}"
            )
        except Exception as e:
            # Log error but don't propagate - user already has their response
            logger.warning(
                f"Background cache store failed (non-blocking): {e}",
                exc_info=True
            )

    async def shutdown(self):
        """Graceful shutdown - wait for pending background tasks."""
        if self._background_tasks:
            logger.info(f"Waiting for {len(self._background_tasks)} background cache tasks...")
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        self._background_executor.shutdown(wait=True)
```

### 2.1 Background Cache Worker (Alternative: Queue-Based)

For high-traffic systems, use a dedicated background worker with a queue:

```python
"""
Queue-based background cache worker for high-throughput scenarios.

Benefits:
- Completely decoupled from request handling
- Rate limiting and batching support
- Graceful degradation under load
- No impact on user response times
"""

import asyncio
from asyncio import Queue
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheStoreRequest:
    """Request to store an entry in the cache."""
    question: str
    answer: str
    workspace_id: str
    source_document_ids: List[str]
    intent: str
    dataset_id: Optional[str] = None
    table_names: Optional[List[str]] = None
    answer_metadata: Optional[Dict[str, Any]] = None
    priority: int = 1  # Lower = higher priority


class BackgroundCacheWorker:
    """
    Background worker that processes cache store requests asynchronously.

    Features:
    - Non-blocking queue for cache requests
    - Configurable batch processing
    - Automatic retry with backoff
    - Health monitoring
    """

    def __init__(
        self,
        cache_manager: QueryCacheManager,
        max_queue_size: int = 10000,
        batch_size: int = 10,
        flush_interval_seconds: float = 1.0,
        max_retries: int = 3
    ):
        self.cache_manager = cache_manager
        self.max_queue_size = max_queue_size
        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds
        self.max_retries = max_retries

        self._queue: Queue[CacheStoreRequest] = Queue(maxsize=max_queue_size)
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None

        # Metrics
        self._processed_count = 0
        self._failed_count = 0
        self._dropped_count = 0

    async def start(self):
        """Start the background worker."""
        if self._running:
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Background cache worker started")

    async def stop(self):
        """Stop the background worker gracefully."""
        self._running = False

        if self._worker_task:
            # Process remaining items in queue
            await self._flush_queue()
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

        logger.info(
            f"Background cache worker stopped. "
            f"Processed: {self._processed_count}, "
            f"Failed: {self._failed_count}, "
            f"Dropped: {self._dropped_count}"
        )

    def enqueue(self, request: CacheStoreRequest) -> bool:
        """
        Add a cache store request to the queue.

        Returns True if enqueued, False if queue is full (request dropped).
        This method is NON-BLOCKING.
        """
        try:
            self._queue.put_nowait(request)
            return True
        except asyncio.QueueFull:
            # Queue is full - drop the request (acceptable for caching)
            self._dropped_count += 1
            logger.warning(
                f"Cache queue full, dropping request for workspace "
                f"{request.workspace_id}"
            )
            return False

    async def _worker_loop(self):
        """Main worker loop - processes cache requests in batches."""
        while self._running:
            try:
                batch = await self._collect_batch()

                if batch:
                    await self._process_batch(batch)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cache worker error: {e}", exc_info=True)
                await asyncio.sleep(1)  # Brief pause on error

    async def _collect_batch(self) -> List[CacheStoreRequest]:
        """Collect a batch of requests from the queue."""
        batch = []

        try:
            # Wait for at least one item
            first_item = await asyncio.wait_for(
                self._queue.get(),
                timeout=self.flush_interval
            )
            batch.append(first_item)

            # Collect more items if available (non-blocking)
            while len(batch) < self.batch_size:
                try:
                    item = self._queue.get_nowait()
                    batch.append(item)
                except asyncio.QueueEmpty:
                    break

        except asyncio.TimeoutError:
            pass  # No items available, return empty batch

        return batch

    async def _process_batch(self, batch: List[CacheStoreRequest]):
        """Process a batch of cache store requests."""
        for request in batch:
            success = await self._process_single_with_retry(request)
            if success:
                self._processed_count += 1
            else:
                self._failed_count += 1

    async def _process_single_with_retry(
        self,
        request: CacheStoreRequest
    ) -> bool:
        """Process a single request with retry logic."""
        for attempt in range(self.max_retries):
            try:
                await self.cache_manager.store_cache(
                    question=request.question,
                    answer=request.answer,
                    workspace_id=request.workspace_id,
                    source_document_ids=request.source_document_ids,
                    intent=request.intent,
                    dataset_id=request.dataset_id,
                    table_names=request.table_names,
                    answer_metadata=request.answer_metadata
                )
                return True

            except Exception as e:
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    wait_time = (2 ** attempt) * 0.1
                    logger.debug(
                        f"Cache store retry {attempt + 1}/{self.max_retries} "
                        f"after {wait_time}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.warning(
                        f"Cache store failed after {self.max_retries} retries: {e}"
                    )

        return False

    async def _flush_queue(self):
        """Process all remaining items in the queue."""
        remaining = []
        while not self._queue.empty():
            try:
                remaining.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        if remaining:
            logger.info(f"Flushing {len(remaining)} remaining cache requests")
            await self._process_batch(remaining)

    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "queue_size": self._queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "processed_count": self._processed_count,
            "failed_count": self._failed_count,
            "dropped_count": self._dropped_count,
            "running": self._running
        }


# Integration with AnalyticsService using queue-based worker
class AnalyticsServiceWithQueueWorker:
    """Analytics service using queue-based background caching."""

    def __init__(self, ...):
        # ... existing initialization ...

        self.cache_manager = QueryCacheManager(...)

        # Initialize background worker
        self.cache_worker = BackgroundCacheWorker(
            cache_manager=self.cache_manager,
            max_queue_size=10000,
            batch_size=10,
            flush_interval_seconds=1.0
        )

    async def startup(self):
        """Start background services."""
        await self.cache_worker.start()

    async def shutdown(self):
        """Graceful shutdown."""
        await self.cache_worker.stop()

    async def route_and_process(
        self,
        question: str,
        workspace_id: str,
        source_document_ids: List[str],
        dataset_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Process question with non-blocking cache storage."""
        start_time = time.time()

        # Check cache (fast read)
        cached_result = await self.cache_manager.search_cache(
            question=question,
            workspace_id=workspace_id,
            source_document_ids=source_document_ids,
            dataset_id=dataset_id
        )

        if cached_result:
            cached_result["response_time_ms"] = (time.time() - start_time) * 1000
            cached_result["from_cache"] = True
            return cached_result

        # Process question
        result = await self._process_question(...)
        response_time_ms = (time.time() - start_time) * 1000

        # Enqueue cache storage - NON-BLOCKING, returns immediately
        if result.get("success", False) and result.get("confidence", 0) > 0.7:
            cache_request = CacheStoreRequest(
                question=question,
                answer=result.get("answer", ""),
                workspace_id=workspace_id,
                source_document_ids=result.get("used_document_ids", source_document_ids),
                intent=result.get("intent", "general"),
                dataset_id=dataset_id,
                table_names=result.get("table_names", []),
                answer_metadata={
                    "sql_query": result.get("sql_query"),
                    "sources": result.get("sources", []),
                    "confidence": result.get("confidence"),
                    "response_time_ms": response_time_ms,
                    "query_type": result.get("query_type")
                }
            )

            # This is NON-BLOCKING - user response is not delayed
            self.cache_worker.enqueue(cache_request)

        result["response_time_ms"] = response_time_ms
        result["from_cache"] = False
        return result
```

### 3. Event Handlers for Cache Invalidation

```python
"""
Event handlers for automatic cache invalidation with composite key awareness.
"""


class CacheInvalidationHandler:
    """
    Handles cache invalidation events.

    Key behavior: When a document is updated/deleted, ALL cache entries
    that include that document in their source documents are invalidated.
    """

    def __init__(self, cache_manager: QueryCacheManager):
        self.cache_manager = cache_manager

    async def on_document_updated(self, workspace_id: str, document_id: str):
        """
        Called when a document is updated or re-processed.

        Invalidates ALL cache entries that used this document as a source,
        regardless of what other documents were also used.
        """
        invalidated = await self.cache_manager.invalidate_by_document(
            workspace_id=workspace_id,
            document_id=document_id
        )
        print(f"Invalidated {invalidated} cache entries containing document {document_id}")

    async def on_document_deleted(self, workspace_id: str, document_id: str):
        """Called when a document is deleted."""
        invalidated = await self.cache_manager.invalidate_by_document(
            workspace_id=workspace_id,
            document_id=document_id
        )
        print(f"Invalidated {invalidated} cache entries for deleted document {document_id}")

    async def on_documents_bulk_updated(
        self,
        workspace_id: str,
        document_ids: List[str]
    ):
        """
        Called when multiple documents are updated at once.
        """
        invalidated = await self.cache_manager.invalidate_by_documents(
            workspace_id=workspace_id,
            document_ids=document_ids
        )
        print(f"Invalidated {invalidated} cache entries for {len(document_ids)} documents")

    async def on_dataset_updated(self, workspace_id: str, dataset_id: str):
        """Called when dataset data is modified."""
        invalidated = await self.cache_manager.invalidate_by_dataset(
            workspace_id=workspace_id,
            dataset_id=dataset_id
        )
        print(f"Invalidated {invalidated} cache entries for dataset {dataset_id}")

    async def on_table_data_changed(self, workspace_id: str, table_name: str):
        """Called when table data is inserted/updated/deleted."""
        invalidated = await self.cache_manager.invalidate_by_table(
            workspace_id=workspace_id,
            table_name=table_name
        )
        print(f"Invalidated {invalidated} cache entries for table {table_name}")


# Integration with document processing
class DocumentProcessor:
    def __init__(self, cache_invalidation_handler: CacheInvalidationHandler, ...):
        self.cache_invalidation_handler = cache_invalidation_handler
        # ... other initialization ...

    async def process_document(self, document_id: str, workspace_id: str, ...):
        # ... existing processing logic ...

        # After successful processing, invalidate related cache entries
        # This will invalidate ALL entries that used this document
        await self.cache_invalidation_handler.on_document_updated(
            workspace_id=workspace_id,
            document_id=document_id
        )

    async def delete_document(self, document_id: str, workspace_id: str, ...):
        # ... existing deletion logic ...

        # Invalidate cache entries that used this document
        await self.cache_invalidation_handler.on_document_deleted(
            workspace_id=workspace_id,
            document_id=document_id
        )
```

### 4. Scheduled Cleanup Task

```python
"""
Scheduled task for cache maintenance.
"""

import asyncio
from apscheduler.schedulers.asyncio import AsyncIOScheduler


class CacheMaintenanceScheduler:
    """Schedules periodic cache maintenance tasks."""

    def __init__(self, cache_manager: QueryCacheManager):
        self.cache_manager = cache_manager
        self.scheduler = AsyncIOScheduler()

    def start(self):
        """Start the maintenance scheduler."""
        # Cleanup expired entries every 6 hours
        self.scheduler.add_job(
            self._cleanup_expired,
            'interval',
            hours=self.cache_manager.config.cleanup_interval_hours,
            id='cache_cleanup'
        )

        # Log cache stats every hour
        self.scheduler.add_job(
            self._log_cache_stats,
            'interval',
            hours=1,
            id='cache_stats'
        )

        self.scheduler.start()

    async def _cleanup_expired(self):
        """Cleanup expired entries across all workspaces."""
        results = await self.cache_manager.cleanup_all_workspaces()
        total = sum(results.values())
        print(f"Cache cleanup completed. Removed entries: {results}")

    async def _log_cache_stats(self):
        """Log cache statistics for monitoring."""
        collections = self.cache_manager.client.get_collections().collections
        cache_collections = [c.name for c in collections if c.name.startswith("query_cache_")]

        for collection_name in cache_collections:
            workspace_id = collection_name.replace("query_cache_", "")
            stats = await self.cache_manager.get_cache_stats(workspace_id)
            print(f"Cache stats for {workspace_id}: {stats}")
```

## API Endpoints

### Cache Management Endpoints

```python
"""
FastAPI endpoints for cache management.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter(prefix="/api/cache", tags=["cache"])


class CacheStatsResponse(BaseModel):
    workspace_id: str
    total_entries: int
    utilization_percent: float
    status: str
    composite_key_enabled: bool


class InvalidationRequest(BaseModel):
    document_id: Optional[str] = None
    document_ids: Optional[List[str]] = None  # For bulk invalidation
    dataset_id: Optional[str] = None
    table_name: Optional[str] = None


class CacheLookupRequest(BaseModel):
    question: str
    source_document_ids: List[str]
    dataset_id: Optional[str] = None
    intent: Optional[str] = None


@router.get("/stats/{workspace_id}", response_model=CacheStatsResponse)
async def get_cache_stats(workspace_id: str):
    """Get cache statistics for a workspace."""
    stats = await cache_manager.get_cache_stats(workspace_id)
    return stats


@router.post("/lookup/{workspace_id}")
async def lookup_cache(workspace_id: str, request: CacheLookupRequest):
    """
    Check if a question+documents combination exists in cache.
    Useful for debugging cache behavior.
    """
    result = await cache_manager.search_cache(
        question=request.question,
        workspace_id=workspace_id,
        source_document_ids=request.source_document_ids,
        dataset_id=request.dataset_id,
        intent=request.intent
    )

    if result:
        return {
            "cache_hit": True,
            "similarity_score": result.get("similarity_score"),
            "exact_match": result.get("exact_match"),
            "cached_at": result.get("cached_at"),
            "source_document_ids": result.get("source_document_ids")
        }

    return {"cache_hit": False}


@router.post("/invalidate/{workspace_id}")
async def invalidate_cache(workspace_id: str, request: InvalidationRequest):
    """
    Manually invalidate cache entries.

    When invalidating by document_id, ALL cache entries that used
    that document as a source will be invalidated.
    """
    invalidated = 0

    if request.document_id:
        invalidated += await cache_manager.invalidate_by_document(
            workspace_id, request.document_id
        )

    if request.document_ids:
        invalidated += await cache_manager.invalidate_by_documents(
            workspace_id, request.document_ids
        )

    if request.dataset_id:
        invalidated += await cache_manager.invalidate_by_dataset(
            workspace_id, request.dataset_id
        )

    if request.table_name:
        invalidated += await cache_manager.invalidate_by_table(
            workspace_id, request.table_name
        )

    return {"invalidated_entries": invalidated}


@router.delete("/clear/{workspace_id}")
async def clear_cache(workspace_id: str):
    """Clear all cache entries for a workspace."""
    success = await cache_manager.clear_cache(workspace_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to clear cache")
    return {"success": True}


@router.post("/cleanup")
async def trigger_cleanup():
    """Manually trigger cache cleanup."""
    results = await cache_manager.cleanup_all_workspaces()
    return {"cleaned_entries": results}
```

## Cache Hit Examples (Permission-Based)

### Example 1: Same User, Similar Question (HIT)

```
User A has access to: [doc_A, doc_B, doc_C]

Cache Entry (created by User A):
  Question: "What is the total revenue?"
  Source Documents: ["doc_A", "doc_B"]
  Answer: "$2.5M"

User A asks: "What's the total revenue amount?"  ← Semantically similar
  → Semantic search finds cached entry
  → Permission check: Can User A read [doc_A, doc_B]? YES
  → Result: Cache HIT, returns "$2.5M"
```

### Example 2: Different User, Has Permission (HIT - Shared Cache)

```
User A has access to: [doc_A, doc_B, doc_C]
User B has access to: [doc_A, doc_B, doc_D]

Cache Entry (created by User A):
  Question: "What is the total revenue?"
  Source Documents: ["doc_A", "doc_B"]
  Answer: "$2.5M"

User B asks: "What's the revenue total?"  ← Semantically similar
  → Semantic search finds cached entry
  → Permission check: Can User B read [doc_A, doc_B]? YES (both accessible)
  → Result: Cache HIT, returns "$2.5M"
  → BENEFIT: User B benefits from User A's cached answer!
```

### Example 3: Different User, Missing Permission (MISS - Security)

```
User A has access to: [doc_A, doc_B, doc_confidential]
User B has access to: [doc_A, doc_B]

Cache Entry (created by User A):
  Question: "What is the CEO salary?"
  Source Documents: ["doc_confidential"]
  Answer: "$5M"

User B asks: "What's the CEO's salary?"  ← Semantically similar
  → Semantic search finds cached entry
  → Permission check: Can User B read [doc_confidential]? NO
  → Result: Cache MISS (permission denied)
  → User B's question processed separately with their accessible documents
  → SECURITY: Confidential information NOT leaked to User B
```

### Example 4: User Has Partial Permission (MISS)

```
User C has access to: [doc_A]  (missing doc_B)

Cache Entry:
  Question: "What is the total revenue?"
  Source Documents: ["doc_A", "doc_B"]
  Answer: "$2.5M"

User C asks: "What is the total revenue?"
  → Semantic search finds cached entry
  → Permission check: Can User C read [doc_A, doc_B]?
    → doc_A: YES
    → doc_B: NO
    → Result: PERMISSION DENIED (must have ALL documents)
  → Result: Cache MISS
  → User C's question processed with only doc_A
  → May get different/partial answer based on available documents
```

### Example 5: Multiple Cache Candidates with Permission Filtering

```
User D has access to: [doc_public_1, doc_public_2]

Cache Entries:
  Entry 1 (similarity: 0.95):
    Question: "What are the sales numbers?"
    Source Documents: ["doc_confidential"]  ← User D cannot access

  Entry 2 (similarity: 0.91):
    Question: "What are the sales figures?"
    Source Documents: ["doc_public_1", doc_public_2"]  ← User D CAN access

User D asks: "What are the sales numbers?"
  → Semantic search returns [Entry 1, Entry 2] sorted by similarity
  → Check Entry 1: Permission denied (doc_confidential)
  → Check Entry 2: Permission granted
  → Result: Cache HIT with Entry 2 (lower similarity but accessible)
```

## Performance Optimization

### 1. Embedding Optimization

```python
class OptimizedEmbeddingService:
    """Embedding service with caching for repeated texts."""

    def __init__(self, base_service, max_cache_size: int = 1000):
        self.base_service = base_service
        self.embedding_cache = {}
        self.max_cache_size = max_cache_size

    async def embed_text(self, text: str) -> List[float]:
        """Embed text with local caching."""
        cache_key = hashlib.md5(text.encode()).hexdigest()

        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]

        embedding = await self.base_service.embed_text(text)

        if len(self.embedding_cache) >= self.max_cache_size:
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]

        self.embedding_cache[cache_key] = embedding
        return embedding
```

### 2. Batch Operations

```python
async def batch_store_cache(
    self,
    entries: List[Dict[str, Any]],
    workspace_id: str
) -> List[str]:
    """
    Store multiple cache entries in a single operation.

    Args:
        entries: List of cache entry dicts (must include source_document_ids)
        workspace_id: Workspace identifier

    Returns:
        List of cache entry IDs
    """
    collection_name = self._get_collection_name(workspace_id)
    await self.initialize_collection(workspace_id)

    points = []
    ids = []

    for entry in entries:
        normalized = self._normalize_question(entry["question"])
        source_docs = entry.get("source_document_ids", [])
        composite_key = self._generate_composite_key(normalized, source_docs)
        embedding = await self.embedding_service.embed_text(normalized)

        point_id = str(uuid.uuid4())
        ids.append(point_id)

        current_time = time.time()
        ttl = entry.get("ttl", self._get_ttl(entry.get("intent")))

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "question_data": {
                        "original_question": entry["question"],
                        "normalized_question": normalized,
                        "question_hash": self._hash_question(normalized)
                    },
                    "source_documents": {
                        "document_ids": sorted(source_docs),
                        "document_ids_hash": self._hash_document_ids(source_docs),
                        "document_count": len(source_docs)
                    },
                    "composite_key": {
                        "key_hash": composite_key
                    },
                    "answer": entry["answer"],
                    "intent": entry.get("intent", "general"),
                    "timestamps": {
                        "created_at": current_time,
                        "last_accessed_at": current_time,
                        "expires_at": current_time + ttl
                    },
                    "is_valid": True
                }
            )
        )

    # Batch upsert
    self.client.upsert(
        collection_name=collection_name,
        points=points
    )

    return ids
```

## Monitoring and Observability

### Metrics to Track

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `cache_hit_rate` | Percentage of requests served from cache | < 20% (low utilization) |
| `cache_miss_rate` | Percentage of cache misses | > 80% (cache not effective) |
| `cache_miss_reason` | Why cache missed (question/documents/expired) | Monitor distribution |
| `exact_match_rate` | % of hits that were exact composite key matches | Informational |
| `semantic_match_rate` | % of hits that were semantic matches | Informational |
| `avg_cache_lookup_time_ms` | Average time to search cache | > 100ms |
| `cache_size_bytes` | Total cache storage size | > 80% of limit |
| `expired_entries_cleaned` | Entries removed during cleanup | Informational |
| `invalidation_count` | Manual/event invalidations | Spike detection |
| `document_invalidation_cascade` | Avg entries invalidated per document update | Monitor for issues |

### Logging

```python
import structlog

logger = structlog.get_logger()

# In search_cache
logger.info(
    "cache_lookup",
    workspace_id=workspace_id,
    cache_hit=cached_result is not None,
    hit_type="exact" if cached_result and cached_result.get("exact_match") else "semantic",
    similarity_score=cached_result.get("similarity_score") if cached_result else None,
    source_document_count=len(source_document_ids),
    lookup_time_ms=lookup_time_ms
)

# In store_cache
logger.info(
    "cache_store",
    workspace_id=workspace_id,
    intent=intent,
    ttl_seconds=ttl,
    source_document_count=len(source_document_ids),
    composite_key=composite_key[:16] + "...",
    cache_entry_id=point_id
)

# In invalidation
logger.info(
    "cache_invalidation",
    workspace_id=workspace_id,
    invalidation_type="document",
    document_id=document_id,
    entries_invalidated=count
)
```

## Best Practices

### 1. Similarity Threshold Tuning

- Start with conservative thresholds (0.92+)
- Monitor false positive rate (wrong cached answers)
- Adjust per query type based on observed accuracy
- Remember: Document matching is always exact, only question matching uses similarity

### 2. TTL Strategy

| Data Volatility | Recommended TTL |
|-----------------|-----------------|
| Static (documentation) | 7-30 days |
| Semi-static (reports) | 24 hours |
| Dynamic (live data) | 1-6 hours |
| Real-time | Don't cache or use partial caching |

### 3. Document ID Consistency

Ensure document IDs are consistent across your system:
- Use the same ID format everywhere (UUID, path, etc.)
- Sort document IDs before hashing for consistent composite keys
- Don't include temporary or session-specific IDs

### 4. Cache Warming

For frequently asked questions, pre-populate the cache:

```python
async def warm_cache(
    self,
    workspace_id: str,
    common_questions: List[Dict]
):
    """
    Pre-populate cache with common questions.

    Each entry must include source_document_ids.
    """
    for q in common_questions:
        # Generate answer through normal pipeline
        result = await self.process_question(
            q["question"],
            workspace_id,
            source_document_ids=q["source_document_ids"]
        )

        # Store in cache with extended TTL
        await self.cache_manager.store_cache(
            question=q["question"],
            answer=result["answer"],
            workspace_id=workspace_id,
            source_document_ids=q["source_document_ids"],
            intent=result["intent"],
            ttl_override=2592000  # 30 days
        )
```

### 5. Graceful Degradation

```python
async def search_cache_safe(
    self,
    question: str,
    workspace_id: str,
    source_document_ids: List[str],
    **kwargs
):
    """Cache search with graceful fallback."""
    try:
        return await self.search_cache(
            question,
            workspace_id,
            source_document_ids,
            **kwargs
        )
    except Exception as e:
        logger.warning("cache_search_failed", error=str(e))
        return None  # Proceed without cache
```

## Summary

This solution provides:

### Core Features

1. **Security-First Design**: Permission validation ensures users only receive cached answers from documents they can access
2. **Semantic Question Similarity**: Uses Qdrant vector search to find similar questions (high cache hit rate)
3. **Permission-Based Validation**: Current user must have READ access to ALL source documents in cached answer
4. **Cross-User Cache Sharing**: Users with overlapping document access can benefit from shared cache entries
5. **Information Leakage Prevention**: Users cannot access cached answers derived from confidential documents they lack permission to read
6. **Non-Blocking Cache Storage**: Cache writes happen in background, never blocking user responses

### Intelligent Caching (Optimized Single LLM Call)

7. **Unified Pre-Cache Analysis**: Single LLM call handles ALL pre-cache analysis (dissatisfaction + question analysis + enhancement + cache decision)
8. **Performance Optimized**: Reduced from 3 LLM calls to 1, saving 400-800ms latency and 66% token cost
9. **Cache Worthiness Analysis**: LLM determines if a question-answer pair should be cached
10. **Question Enhancement**: Context-dependent questions are enhanced to self-contained versions for better cache matching
11. **Dissatisfaction Detection**: System detects when users are unhappy with answers and bypasses cache for fresh queries
12. **Confidence Scoring**: Cache entries have confidence scores that decrease on negative feedback
13. **Automatic Invalidation**: Cache entries with repeated negative feedback are auto-invalidated

### Infrastructure

14. **Flexible TTL**: Configurable expiration per query type
15. **Event-Driven Invalidation**: Automatic cache refresh when documents change
16. **Workspace Isolation**: Separate cache per workspace
17. **Performance Monitoring**: Built-in stats and logging
18. **Graceful Degradation**: System works even if cache fails

### Key Difference from Simple Caching

| Aspect | Simple Cache | Intelligent Permission-Based Cache |
|--------|--------------|-----------------------------------|
| Question matching | Exact or similar | Similar (semantic) with enhancement |
| Context handling | None | LLM enhances context-dependent questions |
| Document validation | None or exact match | Permission check at query time |
| Security | May leak information | Prevents information leakage |
| Cross-user sharing | No | Yes (if permissions allow) |
| User feedback | Ignored | Detects dissatisfaction, bypasses cache |
| Cache quality | Static | Dynamic confidence scoring |

### Intelligent Caching Decision Flow (Optimized)

```
User Message + Chat History + Previous Response
                    │
                    ▼
    ┌───────────────────────────────────────┐
    │     SINGLE UNIFIED LLM CALL           │
    │                                       │
    │  Analyzes ALL at once:                │
    │  • Dissatisfaction detection          │
    │  • Self-contained check               │
    │  • Question enhancement               │
    │  • Cache worthiness decision          │
    │                                       │
    │  Output: unified_analysis JSON        │
    └───────────────────────────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
┌───────────┐ ┌───────────┐ ┌───────────┐
│DISSATISFIED│ │ CACHEABLE │ │NOT CACHE- │
│            │ │           │ │   ABLE    │
│→ Invalidate│ │→ Cache    │ │           │
│  old cache │ │  lookup   │ │→ Fresh    │
│→ Fresh     │ │  (with    │ │  query    │
│  query     │ │  enhanced │ │→ Don't    │
│            │ │  question)│ │  store    │
└───────────┘ └─────┬─────┘ └───────────┘
                    │
                    ▼
          ┌─────────────────┐
          │ Permission check│
          │ for each        │
          │ candidate       │
          └────────┬────────┘
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
    ┌───────────┐    ┌───────────┐
    │ Cache HIT │    │Cache MISS │
    │ Return    │    │Fresh query│
    │ cached    │    │+ store    │
    │ answer    │    │ (async)   │
    └───────────┘    └───────────┘
```

**Old vs New: LLM Call Comparison**
```
OLD DESIGN (Up to 3 LLM calls):
  LLM Call 1: Dissatisfaction detection  → 200-400ms
  LLM Call 2: Cache worthiness analysis  → 200-400ms
  LLM Call 3: Question enhancement       → 200-400ms
  ─────────────────────────────────────────────────
  Total: 600-1200ms before cache lookup

NEW DESIGN (Single LLM call):
  LLM Call 1: Unified analysis           → 200-400ms
  ─────────────────────────────────────────────────
  Total: 200-400ms before cache lookup

  SAVINGS: 400-800ms latency, 66% token cost reduction
```

### Security Guarantee

```
For any cache hit to occur:
  ∀ doc ∈ cached_entry.source_documents:
    current_user.can_read(doc) == TRUE

If ANY source document is not accessible to the current user,
the cache entry is SKIPPED and the next candidate is evaluated.
```

### Cache Bypass Triggers

| Trigger | Detection Method | Action |
|---------|------------------|--------|
| User says "wrong" | LLM dissatisfaction detection | Bypass + fresh query |
| User says "refresh" | LLM dissatisfaction detection | Bypass + fresh query |
| User says "check again" | LLM dissatisfaction detection | Bypass + invalidate + fresh |
| Context-dependent question | LLM analysis | Enhance or skip cache |
| Pronoun-heavy question | LLM analysis | Enhance or skip cache |

### Expected Performance (Optimized)

| Scenario | Response Time | Notes |
|----------|---------------|-------|
| **Pre-cache analysis** | 200-400ms | Single unified LLM call (was 600-1200ms with 3 calls) |
| Cache hit (permission granted) | 215-460ms | Unified analysis + semantic search + permission check |
| Cache hit with enhanced question | 215-460ms | Same (enhancement included in unified call) |
| Cache miss (cacheable) | 2-5s | Full pipeline + background cache store |
| Cache bypass (dissatisfaction) | 2-5s | Fresh query, no cache lookup |
| Non-cacheable question | 2-5s | Full pipeline, no cache store |
| Background cache storage | 0ms added to user | Runs asynchronously |

**Performance Improvement from Optimization:**
| Metric | Old (3 LLM calls) | New (1 LLM call) | Improvement |
|--------|-------------------|------------------|-------------|
| Pre-cache latency | 600-1200ms | 200-400ms | **66% faster** |
| Token cost | 3x | 1x | **66% cheaper** |
| API calls | 3 | 1 | **66% fewer** |
