# LLM-Driven Adaptive Chunking Design

## Overview

This document describes a simplified, intelligent approach to document chunking that replaces the current domain-based pattern matching system with a single LLM call per uploaded file.

## Problem Statement

### Current Approach Issues

1. **Hardcoded patterns are fragile** - False positives like "rx" in "arXiv" causing academic papers to be classified as medical prescriptions
2. **Domain boundaries are artificial** - A medical research paper is both academic AND medical
3. **Too many strategies** - 8+ domain-specific strategies that mostly share the same core goal
4. **Maintenance burden** - Every new document type requires new patterns
5. **Pattern explosion** - 800+ lines of regex patterns across multiple files

### Example of Current Failure

```
Input: Research paper bibliography with arXiv references
  - "Medical graph rag: Towards safe medical large language model..."
  - URL: https://arxiv.org/abs/2408.04187

Current Classification:
  - CPT code pattern matches "04187" (5-digit number) → MEDICAL domain
  - Prescription pattern matches "rx" in "arXiv" → prescription type
  - Result: academic content chunked as medical prescription ❌
```

## Solution: LLM-Driven Structure Analysis

### Core Principle

Instead of classifying documents into predefined domains, **ask the LLM to analyze the content structure directly** and provide chunking guidance based on semantic understanding.

### Key Benefits

| Aspect | Old (Pattern-Based) | New (LLM-Driven) |
|--------|---------------------|------------------|
| Maintenance | Add patterns for each doc type | Zero maintenance |
| Accuracy | Fails on edge cases | Understands context |
| Flexibility | Rigid domain categories | Adapts to any content |
| Complexity | 800+ lines of patterns | ~100 lines + 1 LLM call |
| Cost | Zero | 1 small LLM call per file |

## Document Source Types

The system handles two distinct document processing scenarios:

### Scenario A: OCR-Processed Documents (Multi-Page)

**Source formats:** PDF, scanned images, photos

**Processing result:** Multiple per-page markdown files
```
document.pdf (20 pages)
    ↓ OCR
├── page_1.md
├── page_2.md
├── ...
└── page_20.md
```

### Scenario B: Direct Conversion Documents (Single File)

**Source formats:** Word (.docx), Excel (.xlsx), CSV, plain text, log files

**Processing result:** Single markdown file (can be very large)
```
document.docx (50 pages equivalent)
    ↓ Conversion
└── document.md (single file, ~100,000 chars)
```

## Architecture

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   User Uploads File                         │
└─────────────────────────────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────────┐
│   Scenario A: OCR     │       │  Scenario B: Conversion   │
│                       │       │                           │
│  PDF, Images          │       │  Word, Excel, CSV, Text   │
│       ↓               │       │         ↓                 │
│  Multiple .md files   │       │  Single .md file          │
│  (page_1, page_2...)  │       │  (one large file)         │
└───────────────────────┘       └───────────────────────────┘
            │                               │
            └───────────────┬───────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│          Structure Analysis (1 LLM Call per file)           │
│                                                             │
│  Unified sampling strategy:                                 │
│    - First 3000 chars                                       │
│    - Middle 1000 chars                                      │
│    - Last 1000 chars                                        │
│                                                             │
│  Output: Structure guidance JSON                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│           Smart Chunking Engine (No LLM calls)              │
│                                                             │
│  Scenario A: Apply guidance to each page file               │
│  Scenario B: Apply guidance to single large file            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Final Chunks                           │
└─────────────────────────────────────────────────────────────┘
```

## Scenario Detection

### Output Folder Structure

After OCR or document conversion completes, output files are stored in:

```
/output/{username}/{workspace}/{filename_as_folder}/*_nohf.md

Examples:
  /output/john/project_alpha/research_paper/page_1_nohf.md
  /output/john/project_alpha/research_paper/page_2_nohf.md
  ...

  /output/mary/finance_docs/quarterly_report/quarterly_report_nohf.md
```

### How to Determine Which Scenario

Check the output folder for `*_nohf.md` files count:

```
Output folder: /output/{username}/{workspace}/{filename_as_folder}/

Check: Count of *_nohf.md files

If count == 1:
  → Scenario B (Single File)
  → Use character-based sampling

If count > 1:
  → Scenario A (Multi-Page)
  → Use page-based sampling
```

### Detection Logic

```python
def detect_scenario(output_folder: str) -> str:
    """
    Detect processing scenario based on output file count.

    Args:
        output_folder: Path like /output/{username}/{workspace}/{filename}/

    Returns: "multi_page" or "single_file"
    """
    nohf_files = glob.glob(f"{output_folder}/*_nohf.md")

    if len(nohf_files) == 1:
        return "single_file"
    else:
        return "multi_page"
```

## Content Sampling Strategy

### Why Sample Three Regions

| Region | What It Reveals |
|--------|-----------------|
| **First** | Document type, title, headers, TOC, introduction |
| **Middle** | Recurring body patterns, typical content structure |
| **Last** | References, appendix, signatures, footer patterns |

### Scenario A: Multi-Page Document Sampling (Page-Based)

**No file combining needed** - simply read specific page files directly.

```
Example: 20-page PDF → 20 markdown files
         page_1_nohf.md, page_2_nohf.md, ... page_20_nohf.md

Sampling Strategy:
┌────────────────────────────────────────────────────────────┐
│ FIRST PAGES: Read page 1, 2, 3                             │
│   Files: page_1_nohf.md, page_2_nohf.md, page_3_nohf.md    │
│   Contains: title, abstract, introduction, TOC             │
├────────────────────────────────────────────────────────────┤
│                        ...                                 │
├────────────────────────────────────────────────────────────┤
│ MIDDLE PAGES: Read page 10, 11                             │
│   Files: page_10_nohf.md, page_11_nohf.md                  │
│   Calculated: pages at 50% position                        │
│   Contains: typical body content, recurring patterns       │
├────────────────────────────────────────────────────────────┤
│                        ...                                 │
├────────────────────────────────────────────────────────────┤
│ LAST PAGES: Read page 19, 20                               │
│   Files: page_19_nohf.md, page_20_nohf.md                  │
│   Contains: conclusion, references, bibliography, footer   │
└────────────────────────────────────────────────────────────┘

Total: 7 page files read (not all 20)
```

### Page Selection Logic for Multi-Page

```
Given: N total pages

FIRST pages:  page 1, 2, 3 (always first 3)
MIDDLE pages: page N/2, N/2+1 (2 pages at middle)
LAST pages:   page N-1, N (always last 2)

Examples:
  N = 5 pages:  First [1,2,3], Middle [2,3], Last [4,5]
                (overlap is fine for small docs)

  N = 10 pages: First [1,2,3], Middle [5,6], Last [9,10]

  N = 20 pages: First [1,2,3], Middle [10,11], Last [19,20]

  N = 100 pages: First [1,2,3], Middle [50,51], Last [99,100]

Special cases:
  N = 1 page:   Use entire content (Scenario B applies)
  N = 2 pages:  Read both pages
  N = 3 pages:  Read all 3 pages
  N = 4 pages:  First [1,2], Middle [2,3], Last [3,4]
```

### Scenario B: Single File Sampling (Character-Based)

```
Example: Word document → 1 markdown file (~100,000 chars)
         document_nohf.md

Sampling Strategy:
┌────────────────────────────────────────────────────────────┐
│ FIRST 3000 chars (position 0 - 3000)                       │
│   Contains: title, headers, introduction                   │
├────────────────────────────────────────────────────────────┤
│                        ...                                 │
├────────────────────────────────────────────────────────────┤
│ MIDDLE 1000 chars (position ~49500 - 50500)                │
│   Calculated: (total_length / 2) - 500 to + 500            │
│   Contains: typical body content patterns                  │
├────────────────────────────────────────────────────────────┤
│                        ...                                 │
├────────────────────────────────────────────────────────────┤
│ LAST 1000 chars (position 99000 - 100000)                  │
│   Contains: conclusion, appendix, references               │
└────────────────────────────────────────────────────────────┘

Short document (< 5000 chars):
  → Use entire content, no sampling needed
```

### Sampling Implementation Summary

| Scenario | Detection | Sampling Method | What to Read |
|----------|-----------|-----------------|--------------|
| **A: Multi-Page** | *_nohf.md count > 1 | Page-based | First 3 + Middle 2 + Last 2 pages |
| **B: Single File** | *_nohf.md count == 1 | Character-based | First 3000 + Middle 1000 + Last 1000 chars |

### Benefits of Page-Based Sampling (Scenario A)

| Aspect | Old (Combine All) | New (Page-Based) |
|--------|-------------------|------------------|
| **Memory** | Load all 20 files | Load only 7 files |
| **I/O** | Read 40,000 chars | Read ~14,000 chars |
| **Speed** | Slower | Faster |
| **Complexity** | Combine then slice | Direct file read |

## LLM Structure Analysis

### Approach: Predefined Strategy Library

Instead of having the LLM generate custom patterns (which may be unreliable), the LLM **selects from predefined chunking strategies**. This ensures:

1. **Reliability** - No risk of invalid regex from LLM
2. **Testability** - Each strategy can be unit tested
3. **Predictability** - Same strategy = same behavior
4. **Simplicity** - LLM picks a strategy, doesn't generate code

### Predefined Strategy Library

| Strategy | Split At | Preserve | Best For |
|----------|----------|----------|----------|
| `header_based` | Markdown headers (# ## ###) | Tables, code blocks, equations | Structured documents with clear sections |
| `paragraph_based` | Double newlines (\n\n) | Lists, tables, code blocks | Narrative text, articles, essays |
| `sentence_based` | Sentence boundaries (. ! ?) | Paragraphs within size limit | Dense text without structure |
| `list_item_based` | List item boundaries (- * 1.) | Nested list groups | TOCs, bullet lists, numbered lists |
| `table_row_based` | Table row boundaries | Header row (repeated in chunks) | Tabular data, spreadsheets |
| `citation_based` | Between citation entries | Complete citation (author→URL) | Bibliographies, references |
| `log_entry_based` | Timestamp boundaries | Stack traces, multi-line entries | Log files, event logs |
| `clause_based` | Numbered clauses (1.1, 1.1.1) | Sub-clauses, definitions | Legal docs, specs, contracts |

### Strategy Details

#### `header_based`
```
Split pattern:   ^#{1,6}\s+
Preserve:        Tables, code blocks, equations
Default overlap: 10%
Use when:        Document has markdown headers
```

#### `paragraph_based`
```
Split pattern:   \n\n+
Preserve:        Lists, tables, code blocks
Default overlap: 15%
Use when:        Narrative text without clear headers
```

#### `sentence_based`
```
Split pattern:   (?<=[.!?])\s+
Preserve:        Keep sentences together
Default overlap: 20%
Use when:        Dense text, no structure markers
```

#### `list_item_based`
```
Split pattern:   ^[\s]*[-*•]\s+|^[\s]*\d+\.\s+
Preserve:        Nested list items together
Default overlap: 5%
Use when:        Lists, TOCs, enumerated content
```

#### `table_row_based`
```
Split pattern:   ^\|.*\|$
Preserve:        Header row (include in each chunk)
Default overlap: 0% (but repeat header)
Use when:        Markdown tables, CSV-like content
```

#### `citation_based`
```
Split pattern:   (?=^[A-Z][a-z]+,?\s+[A-Z])|(?=^\[\d+\])
Preserve:        Complete citation entry
Default overlap: 5%
Use when:        Bibliography, references section
```

#### `log_entry_based`
```
Split pattern:   ^\d{4}-\d{2}-\d{2}|^\[[\d:]+\]
Preserve:        Stack traces, multi-line entries
Default overlap: 10%
Use when:        Log files, timestamps present
```

#### `clause_based`
```
Split pattern:   ^[\s]*\d+(\.\d+)*[\s]+
Preserve:        Sub-clauses with parent
Default overlap: 5%
Use when:        Legal, specs, numbered sections
```

### Prompt Template

```
Analyze this document and select the best chunking strategy.

=== DOCUMENT START ===
{first_content}

=== DOCUMENT MIDDLE ===
{middle_content}

=== DOCUMENT END ===
{last_content}

Available strategies:
1. header_based - Split at markdown headers (# ## ###), best for structured docs with sections
2. paragraph_based - Split at paragraph breaks, best for narrative text and articles
3. sentence_based - Split at sentences, best for dense text without structure
4. list_item_based - Split at list items, best for bullet points, numbered lists, TOCs
5. table_row_based - Split at table rows, best for spreadsheets and tabular data
6. citation_based - Split between citations, best for bibliographies and references
7. log_entry_based - Split at timestamps, best for log files and event logs
8. clause_based - Split at numbered clauses (1.1, 1.1.1), best for legal docs and specs

Respond in JSON only:
{
  "selected_strategy": "<strategy_name>",
  "chunk_size": <512-1024>,
  "overlap_percent": <5-20>,
  "preserve_elements": ["tables", "code_blocks", "equations", "lists"],
  "reasoning": "Brief explanation of why this strategy fits"
}
```

### Example LLM Responses

**Academic Research Paper:**
```json
{
  "selected_strategy": "header_based",
  "chunk_size": 768,
  "overlap_percent": 15,
  "preserve_elements": ["equations", "tables", "code_blocks", "citations"],
  "reasoning": "Document has clear section headers (Abstract, Introduction, Methods, etc.) and academic structure"
}
```

**Legal Contract:**
```json
{
  "selected_strategy": "clause_based",
  "chunk_size": 1024,
  "overlap_percent": 5,
  "preserve_elements": ["tables", "definitions", "signature_blocks"],
  "reasoning": "Document uses numbered clause structure (1.1, 1.1.1) typical of legal contracts"
}
```

**Spreadsheet/CSV:**
```json
{
  "selected_strategy": "table_row_based",
  "chunk_size": 1024,
  "overlap_percent": 0,
  "preserve_elements": ["header_row"],
  "reasoning": "Document is primarily tabular data that should be split at row boundaries"
}
```

**Bibliography/References:**
```json
{
  "selected_strategy": "citation_based",
  "chunk_size": 512,
  "overlap_percent": 5,
  "preserve_elements": ["complete_citations"],
  "reasoning": "Document contains academic citations that should not be split mid-entry"
}
```

## Smart Chunking Engine

### Strategy Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│                 LLM Response                                │
│  {                                                          │
│    "selected_strategy": "header_based",                     │
│    "chunk_size": 768,                                       │
│    "overlap_percent": 15,                                   │
│    "preserve_elements": ["equations", "tables"]             │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Load Strategy Definition                       │
│                                                             │
│  STRATEGIES["header_based"] = {                             │
│    "split_pattern": r"^#{1,6}\s+",                          │
│    "preserve_patterns": [TABLE_REGEX, CODE_REGEX, ...],     │
│    "default_chunk_size": 768,                               │
│    "default_overlap": 10                                    │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Apply Universal Rules First                    │
│                                                             │
│  1. Identify atomic elements (never split)                  │
│  2. Mark preserve boundaries                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Apply Strategy Split Pattern                   │
│                                                             │
│  1. Find all split points using strategy pattern            │
│  2. Filter out splits inside atomic elements                │
│  3. Merge small chunks below minimum size                   │
│  4. Split large chunks at secondary boundaries              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Apply Size Constraints                         │
│                                                             │
│  chunk_size: from LLM response (768)                        │
│  overlap: from LLM response (15%)                           │
│  min_size: 256 tokens                                       │
│  max_size: 1536 tokens                                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Final Chunks                             │
└─────────────────────────────────────────────────────────────┘
```

### Universal Rules (Applied to All Strategies)

These rules apply regardless of which strategy is selected:

```
1. NEVER split mid-sentence
   - Detect sentence boundaries (., !, ?)
   - Complete the sentence before splitting

2. NEVER split atomic elements
   - Markdown tables: |...|...|
   - Code blocks: ```...```
   - LaTeX equations: $$...$$
   - HTML blocks: <table>...</table>

3. SIZE constraints
   - Minimum: 256 tokens (merge if smaller)
   - Maximum: 1536 tokens (force split if larger)
   - Target: from LLM response (512-1024)
   - Overlap: from LLM response (5-20%)
```

### How Strategy + Universal Rules Work Together

```
Selected Strategy: clause_based
Universal Rules: Always applied

Step 1: Mark atomic elements
  → Tables at lines 50-65: PROTECTED
  → Code block at lines 120-140: PROTECTED

Step 2: Find split points using clause_based pattern
  → ^[\s]*\d+(\.\d+)*[\s]+
  → Matches: line 10 (1.0), line 25 (1.1), line 45 (2.0), ...

Step 3: Filter invalid splits
  → Remove split at line 55 (inside protected table)

Step 4: Create chunks with overlap
  → Chunk 1: lines 1-24 (section 1.0)
  → Chunk 2: lines 20-44 (section 1.1, with 15% overlap)
  → Chunk 3: lines 40-70 (section 2.0, includes protected table)
  → ...

Step 5: Verify size constraints
  → Chunk 3 is 1800 tokens (too large)
  → Force split at paragraph boundary within section 2.0
```

### Strategy Executor Pseudocode

```python
def execute_strategy(content: str, llm_response: dict) -> List[Chunk]:
    # Load strategy definition
    strategy = STRATEGIES[llm_response["selected_strategy"]]

    # Step 1: Find and protect atomic elements
    protected_ranges = find_atomic_elements(content)

    # Step 2: Find split points using strategy pattern
    split_points = find_matches(content, strategy["split_pattern"])

    # Step 3: Filter splits inside protected ranges
    valid_splits = [p for p in split_points if not in_protected_range(p, protected_ranges)]

    # Step 4: Create initial chunks
    chunks = create_chunks_at_splits(content, valid_splits)

    # Step 5: Apply size constraints
    final_chunks = []
    for chunk in chunks:
        if len(chunk) < MIN_SIZE:
            # Merge with previous chunk
            merge_with_previous(final_chunks, chunk)
        elif len(chunk) > MAX_SIZE:
            # Force split at secondary boundaries
            sub_chunks = force_split(chunk, secondary_pattern=r"\n\n")
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)

    # Step 6: Add overlap
    return add_overlap(final_chunks, llm_response["overlap_percent"])
```

## Fallback Strategy

When LLM is unavailable or fails, use default strategy:

```json
{
  "selected_strategy": "paragraph_based",
  "chunk_size": 768,
  "overlap_percent": 10,
  "preserve_elements": ["tables", "code_blocks", "lists"],
  "reasoning": "Default fallback - paragraph-based splitting works for most documents"
}
```

### Fallback Logic

```python
def get_chunking_config(content: str, output_folder: str) -> dict:
    try:
        # Try LLM-based strategy selection
        sampled_content = sample_content(content, output_folder)
        return call_llm_for_strategy(sampled_content)
    except Exception as e:
        logger.warning(f"LLM strategy selection failed: {e}, using fallback")
        return DEFAULT_STRATEGY

DEFAULT_STRATEGY = {
    "selected_strategy": "paragraph_based",
    "chunk_size": 768,
    "overlap_percent": 10,
    "preserve_elements": ["tables", "code_blocks", "lists"]
}
```

This provides reasonable defaults that work for most documents.

## Components to Remove

| File/Component | Action |
|----------------|--------|
| `domain_patterns.py` → `DOMAIN_CONFIGS` | Remove |
| `domain_patterns.py` → `MEDICAL_PATTERNS` | Remove |
| `domain_patterns.py` → `ACADEMIC_PATTERNS` | Remove |
| `domain_patterns.py` → `LEGAL_PATTERNS` | Remove |
| `domain_patterns.py` → `ENGINEERING_PATTERNS` | Remove |
| `domain_patterns.py` → `EDUCATION_PATTERNS` | Remove |
| `domain_patterns.py` → `FINANCIAL_PATTERNS` | Remove |
| `domain_patterns.py` → `GOVERNMENT_PATTERNS` | Remove |
| `domain_patterns.py` → `PROFESSIONAL_PATTERNS` | Remove |
| `domain_patterns.py` → `detect_domain_from_content()` | Remove |
| `document_classifier.py` → Domain-based classification | Remove |
| `document_type_classifier.py` → `_classify_by_patterns()` | Remove |

## Components to Keep/Modify

| Component | Action |
|-----------|--------|
| Universal atomic patterns (tables, code, equations) | Keep |
| Size constraints and overlap logic | Keep |
| Basic sentence boundary detection | Keep |
| Markdown header detection | Keep |

## New Components to Add

| Component | Description |
|-----------|-------------|
| `structure_analyzer.py` | Single LLM call for structure analysis |
| `content_sampler.py` | First/middle/last content sampling |
| `adaptive_chunker.py` | Chunking engine using LLM guidance |

## Cost Analysis

| Metric | Value |
|--------|-------|
| LLM calls per uploaded file | **1** |
| LLM calls per page | **0** |
| Input tokens per call | ~1,500 (5,000 chars) |
| Output tokens per call | ~100-200 |
| Estimated latency | 1-2 seconds |
| Fallback (no LLM) | 0 calls, use defaults |

### Cost Comparison

```
Old Approach:
  - 0 LLM calls
  - But: Frequent misclassification, manual pattern maintenance

New Approach:
  - 1 LLM call per file
  - But: Accurate classification, zero maintenance, handles any document
```

## Implementation Phases

### Phase 1: Core Implementation
1. Create `structure_analyzer.py` with LLM prompt
2. Create `content_sampler.py` for first/middle/last sampling
3. Modify `adaptive_chunker.py` to use LLM guidance
4. Add fallback defaults

### Phase 2: Integration
1. Integrate with existing chunking pipeline
2. Ensure structure guidance is shared across all pages
3. Add caching for structure analysis results

### Phase 3: Cleanup
1. Remove domain pattern files
2. Remove pattern-based classification
3. Update tests

## Success Metrics

| Metric | Target |
|--------|--------|
| Misclassification rate | < 5% (vs current ~15-20%) |
| LLM calls per file | 1 |
| Chunking latency overhead | < 2 seconds |
| Code reduction | > 500 lines removed |

## Appendix: Example End-to-End Flows

### Example A: OCR Multi-Page Document (PDF)

**Input:**
- 20-page research paper PDF about RAG and LLMs
- OCR generates: page_1_nohf.md, page_2_nohf.md, ... page_20_nohf.md
- Contains: abstract, methodology, results, 100+ bibliography entries

**Scenario Detection:**
```
Output folder: /output/john/ml_research/graph_r1/
Count of *_nohf.md files: 20
→ Scenario A (Multi-Page) → Use page-based sampling
```

**Sampling (7 pages read, not 20):**
- First pages [1,2,3]: title, abstract, introduction
- Middle pages [10,11]: methodology section with equations
- Last pages [19,20]: bibliography with arXiv citations

**LLM Response:**
```json
{
  "selected_strategy": "header_based",
  "chunk_size": 768,
  "overlap_percent": 15,
  "preserve_elements": ["equations", "tables", "citations", "code_blocks"],
  "reasoning": "Document has clear academic structure with section headers and contains equations and citations"
}
```

**Strategy Execution:**
1. Load `header_based` strategy: split at `^#{1,6}\s+`
2. Protect equations, tables, citations
3. Find headers across all 20 pages
4. Create chunks at header boundaries
5. Apply 768 token size limit with 15% overlap

**Chunking Result:**
- Strategy applied consistently to each of 20 page files
- Bibliography entries kept intact (citation_based would be used for refs-only pages)
- Equations preserved with surrounding context
- **No false "prescription" classification**

---

### Example B: Conversion Single File (Word Document)

**Input:**
- 80-page technical specification Word document
- Conversion generates: single document_nohf.md (~150,000 chars)
- Contains: requirements, diagrams, appendices, revision history

**Scenario Detection:**
```
Output folder: /output/mary/engineering/system_requirements/
Count of *_nohf.md files: 1
→ Scenario B (Single File) → Use character-based sampling
```

**Sampling (character positions):**
- First 3000: chars 0-3000 (title page, TOC, introduction)
- Middle 1000: chars 74,500-75,500 (middle of requirements section)
- Last 1000: chars 149,000-150,000 (revision history, signatures)

**LLM Response:**
```json
{
  "selected_strategy": "clause_based",
  "chunk_size": 1024,
  "overlap_percent": 5,
  "preserve_elements": ["tables", "code_blocks", "definitions"],
  "reasoning": "Document uses numbered clause structure (1.1, 1.1.1) typical of technical specifications"
}
```

**Strategy Execution:**
1. Load `clause_based` strategy: split at `^[\s]*\d+(\.\d+)*[\s]+`
2. Protect tables and definition blocks
3. Find all numbered clauses in 150,000 char file
4. Create chunks at clause boundaries
5. Apply 1024 token size limit with 5% overlap

**Chunking Result:**
- Requirements kept with their sub-clauses
- Tables preserved intact
- Consistent chunking throughout 150,000 char document

---

### Example C: Conversion Single File (CSV/Excel)

**Input:**
- Large inventory spreadsheet
- Conversion generates: single inventory_nohf.md (~50,000 chars)
- Contains: 2000 rows of product data in markdown table format

**Scenario Detection:**
```
Output folder: /output/alex/warehouse/product_inventory/
Count of *_nohf.md files: 1
→ Scenario B (Single File) → Use character-based sampling
```

**Sampling (character positions):**
- First 3000: chars 0-3000 (headers, first ~30 rows)
- Middle 1000: chars 24,500-25,500 (middle rows)
- Last 1000: chars 49,000-50,000 (last rows, totals)

**LLM Response:**
```json
{
  "selected_strategy": "table_row_based",
  "chunk_size": 1024,
  "overlap_percent": 0,
  "preserve_elements": ["header_row", "summary_rows"],
  "reasoning": "Document is tabular data that should be split at row boundaries with header repeated"
}
```

**Strategy Execution:**
1. Load `table_row_based` strategy: split at `^\|.*\|$`
2. Identify header row (first row with column names)
3. Split at row boundaries, respecting chunk size
4. Prepend header row to each chunk for context
5. No overlap needed (header provides context)

**Chunking Result:**
- Table header included in each chunk for context
- Rows grouped logically (~50 rows per chunk)
- Summary rows kept together at end

---

### Example D: Conversion Single File (Log File)

**Input:**
- Application server log file
- Conversion generates: single server_nohf.md (~200,000 chars)
- Contains: timestamped log entries, stack traces, metrics

**Scenario Detection:**
```
Output folder: /output/devops/production/app_server_2024/
Count of *_nohf.md files: 1
→ Scenario B (Single File) → Use character-based sampling
```

**Sampling (character positions):**
- First 3000: chars 0-3000 (startup logs, config info)
- Middle 1000: chars 99,500-100,500 (typical runtime logs)
- Last 1000: chars 199,000-200,000 (recent errors/shutdown)

**LLM Response:**
```json
{
  "selected_strategy": "log_entry_based",
  "chunk_size": 1024,
  "overlap_percent": 10,
  "preserve_elements": ["stack_traces", "multi_line_entries"],
  "reasoning": "Document contains timestamped log entries with stack traces that should not be split"
}
```

**Strategy Execution:**
1. Load `log_entry_based` strategy: split at `^\d{4}-\d{2}-\d{2}|^\[[\d:]+\]`
2. Identify and protect stack traces (Exception...at...at...at)
3. Split at timestamp boundaries
4. Merge small entries, split large ones at secondary boundaries
5. Apply 10% overlap for context continuity

**Chunking Result:**
- Stack traces never split mid-trace
- Related log entries grouped together
- Timestamps provide natural boundaries
