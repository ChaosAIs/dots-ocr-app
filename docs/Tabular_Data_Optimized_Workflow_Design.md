# Tabular Data Optimized Workflow Design

## Document Information
- **Version:** 2.0
- **Created:** 2025-12-29
- **Updated:** 2025-12-29
- **Status:** Design Document (Revised)
- **Author:** System Architecture Team

---

## 1. Executive Summary

### 1.1 Problem Statement

The current document processing pipeline applies the same workflow to all document types:
1. **Conversion** -> Markdown
2. **Chunking** -> Split into semantic chunks
3. **Vector Indexing** -> Embed chunks to Qdrant
4. **GraphRAG Indexing** -> Extract entities to Neo4j
5. **Data Extraction** -> Parse structured data to PostgreSQL

For **tabular/dataset-style documents** (Excel, CSV, invoices, receipts, bank statements), this creates:
- **Redundant storage**: Same data in Qdrant (chunks) AND PostgreSQL (line_items)
- **Wasted processing**: Chunking breaks table context, embeddings have limited value
- **Slower processing**: Unnecessary vector indexing adds ~5-10 seconds per document
- **Storage overhead**: Qdrant stores chunked table data that's rarely queried semantically

### 1.2 Proposed Solution (Revised)

Create an **integrated optimized pathway** within the existing chunking flow that:
- **Uses the existing classification system** to detect tabular documents (no file extension dependency)
- **Decides at chunking time** whether to skip row-level chunking based on comprehensive analysis
- **Generates summary/metadata chunks** for document discovery even when skipping row chunking
- **Continues normal indexing** for the summary chunks to enable vector retrieval
- **Handles hybrid documents** where tabular data is only part of the content

### 1.3 Key Design Principle: Classification-Driven, Not Extension-Driven

**Previous Approach (v1.0):**
```
Upload -> Check File Extension -> Route to Tabular/Standard Path
```

**New Approach (v2.0):**
```
Upload -> Convert -> Chunk Classification -> Analyze Strategy -> Decide Skip -> Generate Summary -> Index
```

The key insight is that **tabular detection should happen inside the chunking process**, after document classification has already analyzed the content structure.

### 1.4 Key Insight: Two-Stage Query Process

For tabular data queries, we need a **two-stage process**:

```
Stage 1: Document Discovery (Vector Search on Summaries)
   "Which documents contain sales data for Q4?"
   -> Returns: [doc_id_1, doc_id_2, doc_id_3]

Stage 2: Data Analytics (SQL Query)
   "Total sales by category from these documents"
   -> Executes SQL on documents_data for doc_id_1, doc_id_2, doc_id_3
```

**Why we still need vector indexing for summaries:**
- Users may have 100+ Excel files in a workspace
- Need to find WHICH files are relevant before querying
- Metadata embeddings enable semantic document discovery
- "Find spreadsheets about inventory" -> vector search on summaries

### 1.5 Benefits

| Metric | Current | Optimized | Improvement |
|--------|---------|-----------|-------------|
| Processing Time | ~15-20s | ~8-12s | **40% faster** |
| Qdrant Storage | Full row chunks (100+) | Summary + metadata only (1-3) | **95% reduction** |
| Query Accuracy | Mixed | SQL-based | **More precise** |
| Document Discovery | Full text | Semantic summary | **Preserved** |
| Detection Accuracy | Extension-based only | Content + Classification | **More accurate** |

---

## 2. Revised Architecture: Classification-Driven Detection

### 2.1 Why Not File Extension Detection?

The previous design relied on file extensions at upload time:

**Problems with Extension-Based Detection:**
1. **PDF invoices/receipts** - Extension is `.pdf` but content is tabular
2. **Scanned spreadsheets** - Extension is `.jpg/.png` but OCR produces tables
3. **Hybrid documents** - Research papers with data tables should NOT skip all chunking
4. **False positives** - A `.csv` with only 3 rows might not benefit from tabular optimization

**New Approach: Use Existing Classification System:**
- The `AdaptiveChunker` already classifies documents
- It already detects `document_type` and `recommended_strategy`
- When `recommended_strategy = table_preserving`, we have a candidate for tabular optimization
- Additional analysis decides whether to actually skip chunking

### 2.2 Document Types That May Use Tabular Optimization

These document types, when detected by the classification system, are **candidates** for tabular optimization:

#### By Document Type (Content-Based Classification)
```
spreadsheet         - Generic spreadsheet data
invoice             - Purchase/sales invoices
receipt             - Transaction receipts
bank_statement      - Bank account statements
credit_card_statement - Credit card statements
expense_report      - Expense tracking documents
inventory_report    - Stock/inventory listings
purchase_order      - PO documents with line items
sales_order         - SO documents with line items
payroll             - Salary/wage documents
financial_statement - Balance sheets, P&L
price_list          - Product pricing tables
```

#### By Recommended Strategy
```
table_preserving    - Documents classified with this strategy are candidates
```

#### Exclusions (Keep Full Chunking Even If Tables Present)
```
research_paper      - Tables are supplementary, narrative is primary
academic_article    - Same as above
contract            - Tables may exist but legal text is primary
technical_spec      - Tables exist but requirements text is primary
user_manual         - Mixed content, keep full chunking
```

### 2.3 New Decision Flow (Inside Adaptive Chunker)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ADAPTIVE CHUNKER (REVISED FLOW)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  STEP 1: Pre-processing content                                             │
│           │                                                                  │
│           ▼                                                                  │
│  STEP 2: Document Classification                                            │
│           │                                                                  │
│           ├── document_type: "inventory_report"                             │
│           ├── document_domain: "professional"                               │
│           ├── recommended_strategy: "table_preserving"                      │
│           └── chunk_size: 1024                                              │
│           │                                                                  │
│           ▼                                                                  │
│  STEP 3: ★ NEW ★ TABULAR SKIP ANALYSIS                                      │
│           │                                                                  │
│           ├── Is recommended_strategy == "table_preserving"?                │
│           │    └── NO  ──────────────────────────────────────► STEP 5       │
│           │    └── YES ─► Continue analysis                                 │
│           │                                                                  │
│           ├── Check document_type in TABULAR_DOCUMENT_TYPES?                │
│           │    └── invoice, receipt, bank_statement, etc. ─► High skip      │
│           │    └── research_paper, contract, etc. ─► Low skip               │
│           │                                                                  │
│           ├── Calculate table_content_ratio                                 │
│           │    └── > 80% tables ─► High skip                                │
│           │    └── 50-80% tables ─► Medium (check doc type)                 │
│           │    └── < 50% tables ─► Low skip (hybrid document)               │
│           │                                                                  │
│           ├── Check row_count vs total_content                              │
│           │    └── Many rows, little narrative ─► High skip                 │
│           │    └── Few rows, much narrative ─► Low skip                     │
│           │                                                                  │
│           └── DECISION: skip_row_chunking = True/False                      │
│           │                                                                  │
│           ▼                                                                  │
│  STEP 4: ★ NEW ★ TABULAR SUMMARY GENERATION (if skip_row_chunking=True)    │
│           │                                                                  │
│           ├── Generate document summary chunk (LLM-driven)                  │
│           ├── Generate schema description chunk                             │
│           ├── Generate business context chunk (optional)                    │
│           └── Return summary_chunks (1-3 chunks) instead of row chunks      │
│           │                                                                  │
│           ▼                                                                  │
│  STEP 5: Execute Chunking Strategy                                          │
│           │                                                                  │
│           ├── If skip_row_chunking=True: chunks = summary_chunks            │
│           └── If skip_row_chunking=False: chunks = strategy.chunk(content)  │
│           │                                                                  │
│           ▼                                                                  │
│  STEP 6: Post-processing (metadata enhancement)                             │
│           │                                                                  │
│           ▼                                                                  │
│  RETURN: ChunkingResult(chunks, profile, skip_info)                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Tabular Skip Analysis Logic

### 3.1 Skip Decision Algorithm

```python
def should_skip_row_chunking(
    profile: ChunkingProfile,
    content: str,
    document_type: str,
    file_extension: str = None
) -> Tuple[bool, str, Dict]:
    """
    Determine if row-level chunking should be skipped for tabular documents.

    This function is called AFTER document classification, when we already
    know the recommended_strategy is "table_preserving".

    Returns:
        Tuple of (should_skip, reason, analysis_details)
    """

    # Only consider if strategy is table_preserving
    if profile.recommended_strategy != "table_preserving":
        return False, "strategy_not_table_preserving", {}

    analysis = {
        "document_type": document_type,
        "table_content_ratio": 0.0,
        "row_count_estimate": 0,
        "narrative_ratio": 0.0,
        "skip_score": 0.0
    }

    # ========== FACTOR 1: Document Type (Weight: 40%) ==========
    # Some document types are inherently tabular and benefit from skip
    STRONG_TABULAR_TYPES = {
        'spreadsheet', 'invoice', 'receipt', 'bank_statement',
        'credit_card_statement', 'expense_report', 'inventory_report',
        'purchase_order', 'sales_order', 'payroll', 'financial_statement',
        'price_list', 'data_export', 'transaction_log', 'shipping_manifest'
    }

    # Document types that should NEVER skip chunking even with tables
    NEVER_SKIP_TYPES = {
        'research_paper', 'academic_article', 'thesis', 'dissertation',
        'contract', 'agreement', 'legal_document', 'terms_of_service',
        'technical_spec', 'user_manual', 'documentation', 'tutorial',
        'news_article', 'blog_post', 'report'  # General reports have narrative
    }

    doc_type_lower = document_type.lower() if document_type else ""

    if doc_type_lower in NEVER_SKIP_TYPES:
        return False, f"document_type_excluded:{doc_type_lower}", analysis

    doc_type_score = 0.4 if doc_type_lower in STRONG_TABULAR_TYPES else 0.1

    # ========== FACTOR 2: Table Content Ratio (Weight: 35%) ==========
    table_ratio = _calculate_table_content_ratio(content)
    analysis["table_content_ratio"] = table_ratio

    # If tables are less than 50% of content, this is likely a hybrid document
    if table_ratio < 0.5:
        table_score = 0.0
    elif table_ratio < 0.7:
        table_score = 0.15
    elif table_ratio < 0.85:
        table_score = 0.25
    else:
        table_score = 0.35

    # ========== FACTOR 3: Row Density Analysis (Weight: 25%) ==========
    row_count, avg_row_length = _analyze_table_rows(content)
    analysis["row_count_estimate"] = row_count

    # Many short rows = typical dataset (high skip benefit)
    # Few long rows = formatted tables in document (low skip benefit)
    if row_count > 50 and avg_row_length < 200:
        row_score = 0.25  # Looks like a dataset
    elif row_count > 20 and avg_row_length < 300:
        row_score = 0.15
    elif row_count > 10:
        row_score = 0.1
    else:
        row_score = 0.0  # Too few rows, probably not a dataset

    # ========== CALCULATE FINAL SCORE ==========
    total_score = doc_type_score + table_score + row_score
    analysis["skip_score"] = total_score
    analysis["narrative_ratio"] = 1.0 - table_ratio

    # Threshold for skipping
    SKIP_THRESHOLD = 0.6

    should_skip = total_score >= SKIP_THRESHOLD

    reason = f"score:{total_score:.2f}|doc_type:{doc_type_score:.2f}|table:{table_score:.2f}|rows:{row_score:.2f}"

    return should_skip, reason, analysis


def _calculate_table_content_ratio(content: str) -> float:
    """Calculate what percentage of content is markdown tables."""
    if not content:
        return 0.0

    lines = content.split('\n')
    total_chars = len(content)
    table_chars = 0

    for line in lines:
        # Markdown table line: | col1 | col2 | col3 |
        if '|' in line and line.strip().startswith('|'):
            table_chars += len(line)
        # Table separator: |---|---|---|
        elif '|' in line and '-' in line and line.count('-') > 5:
            table_chars += len(line)

    return table_chars / max(total_chars, 1)


def _analyze_table_rows(content: str) -> Tuple[int, float]:
    """Count table rows and calculate average row length."""
    lines = content.split('\n')
    table_rows = []

    for line in lines:
        if '|' in line and line.strip().startswith('|'):
            # Skip separator lines
            if not (line.count('-') > 5 and '|' in line):
                table_rows.append(line)

    if not table_rows:
        return 0, 0.0

    avg_length = sum(len(row) for row in table_rows) / len(table_rows)
    return len(table_rows), avg_length
```

### 3.2 Decision Matrix

| Document Type | Table Ratio | Row Count | Skip Decision | Reason |
|---------------|-------------|-----------|---------------|--------|
| inventory_report | >80% | 500+ | **SKIP** | Pure tabular dataset |
| invoice | >70% | 10-50 | **SKIP** | Tabular type, high table ratio |
| receipt | >60% | 5-20 | **SKIP** | Tabular type |
| research_paper | 30% | 20 | **NO SKIP** | Excluded document type |
| contract | 40% | 15 | **NO SKIP** | Excluded document type |
| spreadsheet | >90% | 1000+ | **SKIP** | Pure dataset |
| technical_spec | 50% | 30 | **NO SKIP** | Excluded type, hybrid content |
| bank_statement | >85% | 100+ | **SKIP** | Tabular type, high ratio |
| unknown | >85% | 200+ | **SKIP** | High scores on content analysis |
| unknown | 40% | 50 | **NO SKIP** | Mixed content, low score |

### 3.3 Handling Hybrid Documents

For documents that have significant tabular content but ALSO significant narrative:

```python
# Example: Research paper with 30% data tables

profile.recommended_strategy = "table_preserving"  # Due to table presence
document_type = "research_paper"  # Detected type

# Analysis:
# - doc_type_score = 0.0 (research_paper is in NEVER_SKIP_TYPES)
# - Result: NO SKIP - use full table_preserving chunking

# The table_preserving strategy will:
# - Keep tables intact within chunks
# - But still create chunks for all content
# - Narrative text gets chunked normally
```

---

## 4. Summary Chunk Generation (When Skip = True)

### 4.1 What Gets Created Instead of Row Chunks

When `skip_row_chunking = True`, instead of creating 100+ row chunks, we create **1-3 summary chunks**:

#### Chunk 1: Document Summary (Required)
```
"Sales Report Q4 2024 - Contains 1,500 transactions from October to December 2024.
Includes sales data for 45 products across 12 categories. Total revenue: $2.3M.
Key customers: Acme Corp, GlobalTech, MegaStore."
```

#### Chunk 2: Schema/Structure Description (Required)
```
"Spreadsheet with columns: Transaction Date, Customer Name, Product, Category,
Quantity, Unit Price, Total Amount, Payment Method, Sales Rep.
Numeric fields: Quantity (sum), Unit Price, Total Amount (sum).
Categorical fields: Category (12 unique), Payment Method (4 types).
Date range: 2024-10-01 to 2024-12-31."
```

#### Chunk 3: Business Context & Sample Data (Optional)
```
"Sample transactions: Electronics category with products like Laptop Pro,
Wireless Mouse. Customers include enterprise and retail segments.
Payment methods: Credit Card, Wire Transfer, Invoice, Cash.
Business domain: Retail sales tracking with customer relationship data."
```

### 4.2 Summary Generation Methods

**Primary: LLM-Driven Generation (Recommended)**
- Analyzes sample rows and column headers
- Infers business domain and key entities
- Generates natural language descriptions optimized for semantic search

**Fallback: Rule-Based Generation**
- Uses filename keywords and column names
- Groups fields by semantic type (amount, date, category)
- Creates structured descriptions

### 4.3 Metadata for Summary Chunks

Summary chunks must include the same classification metadata as standard chunks for consistency in retrieval and filtering:

```python
{
    # === Standard chunk metadata (same as regular chunks) ===
    "document_id": "uuid-xxx",
    "source": "Q4_Sales_Report.xlsx",
    "document_type": "inventory_report",  # IMPORTANT: Keep for filtering/routing
    "document_domain": "professional",    # IMPORTANT: Keep for filtering/routing

    # === Tabular-specific metadata ===
    "chunk_type": "tabular_summary",  # or "tabular_schema", "tabular_context"
    "is_tabular": True,
    "is_summary_chunk": True,  # Indicates this is a summary, not row data
    "skip_row_chunking": True,  # Indicates row chunking was skipped

    # === Schema information ===
    "schema_type": "spreadsheet",
    "row_count": 1500,
    "column_count": 9,
    "columns": ["Transaction Date", "Customer Name", "Product", ...],
    "amount_fields": ["Total Amount", "Unit Price"],
    "category_fields": ["Category", "Payment Method"],
    "date_fields": ["Transaction Date"],
    "date_range_start": "2024-10-01",
    "date_range_end": "2024-12-31",

    # === Processing information ===
    "summary_method": "llm",  # or "rule_based"
    "recommended_strategy": "table_preserving",  # Original strategy from classification
    "skip_score": 0.85,  # Score that triggered the skip decision
    "skip_reason": "score:0.85|doc_type:0.4|table:0.35|rows:0.1"
}
```

**Why keep `document_type` and `document_domain`:**
1. **Consistency**: Same metadata structure as standard chunks enables unified retrieval
2. **Filtering**: Queries can filter by document_type regardless of chunk type
3. **Routing**: Intent classifier can use document_type to route queries appropriately
4. **Analytics**: Enables analysis of document types across the workspace

---

## 5. Revised Workflow Diagram

### 5.1 Complete Flow (Integrated with Existing System)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DOCUMENT PROCESSING FLOW (v2.0)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────┐                                                          │
│  │   UPLOAD      │  No special handling at upload time                      │
│  │   /upload     │  (Remove extension-based detection from upload)          │
│  └───────┬───────┘                                                          │
│          │                                                                   │
│          ▼                                                                   │
│  ┌───────────────┐                                                          │
│  │   CONVERT     │  Convert to Markdown (unchanged)                         │
│  │   OCR/Doc     │  Excel/CSV -> Markdown tables                            │
│  └───────┬───────┘                                                          │
│          │                                                                   │
│          ▼                                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                    ADAPTIVE CHUNKER (ENHANCED)                         │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │ STEP 1-2: Classification (existing)                              │  │ │
│  │  │   document_type = "inventory_report"                             │  │ │
│  │  │   recommended_strategy = "table_preserving"                      │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │                              │                                          │ │
│  │                              ▼                                          │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │ STEP 3: ★ NEW ★ Tabular Skip Analysis                           │  │ │
│  │  │                                                                   │  │ │
│  │  │   if recommended_strategy == "table_preserving":                 │  │ │
│  │  │       should_skip, reason, analysis = should_skip_row_chunking() │  │ │
│  │  │                                                                   │  │ │
│  │  │   Factors considered:                                            │  │ │
│  │  │   - document_type in TABULAR_TYPES? (40%)                        │  │ │
│  │  │   - table_content_ratio > 80%? (35%)                             │  │ │
│  │  │   - row_count > 50 with short rows? (25%)                        │  │ │
│  │  │   - document_type NOT in NEVER_SKIP_TYPES?                       │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │                              │                                          │ │
│  │              ┌───────────────┴───────────────┐                          │ │
│  │              │                               │                          │ │
│  │              ▼                               ▼                          │ │
│  │  ┌─────────────────────┐       ┌─────────────────────────────────┐     │ │
│  │  │ skip_row_chunking   │       │ skip_row_chunking = False       │     │ │
│  │  │ = True              │       │                                   │     │ │
│  │  │                     │       │ STEP 5: Normal Chunking           │     │ │
│  │  │ STEP 4: Generate    │       │   - table_preserving strategy     │     │ │
│  │  │ Summary Chunks      │       │   - Creates 100+ row chunks       │     │ │
│  │  │   - 1-3 chunks      │       │   - Full content indexed          │     │ │
│  │  │   - LLM summaries   │       │                                   │     │ │
│  │  └─────────┬───────────┘       └─────────────────┬───────────────┘     │ │
│  │            │                                     │                      │ │
│  │            └─────────────┬───────────────────────┘                      │ │
│  │                          │                                              │ │
│  │                          ▼                                              │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │ STEP 6: Post-processing & Return ChunkingResult                  │  │ │
│  │  │   chunks: [summary_chunks] OR [row_chunks]                       │  │ │
│  │  │   profile: classification info                                   │  │ │
│  │  │   skip_info: {skipped: true/false, reason: "...", analysis: {}} │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│          │                                                                   │
│          ▼                                                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         VECTOR INDEXING                                │ │
│  │                                                                         │ │
│  │  Same flow for both paths - index whatever chunks were created:        │ │
│  │                                                                         │ │
│  │  If skip_row_chunking=True:                                            │ │
│  │    - Index 1-3 summary chunks                                          │ │
│  │    - Enables document discovery via vector search                      │ │
│  │    - indexed_chunks = 1-3                                              │ │
│  │                                                                         │ │
│  │  If skip_row_chunking=False:                                           │ │
│  │    - Index all row chunks (100+)                                       │ │
│  │    - Full content searchable                                           │ │
│  │    - indexed_chunks = 100+                                             │ │
│  │                                                                         │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│          │                                                                   │
│          ▼                                                                   │
│  ┌───────────────┐                                                          │
│  │  GraphRAG     │  Skip for tabular documents (existing logic)             │
│  │  (if enabled) │  Based on document_type and file_type checks             │
│  └───────────────┘                                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Data Flow Comparison

| Stage | Standard Document | Tabular (Skip=False) | Tabular (Skip=True) |
|-------|-------------------|----------------------|---------------------|
| Upload | Normal | Normal | Normal |
| Convert | Markdown | Markdown (tables) | Markdown (tables) |
| Classify | semantic_header | table_preserving | table_preserving |
| Skip Analysis | N/A | score < 0.6 | score >= 0.6 |
| Chunking | 20-50 chunks | 100+ row chunks | 1-3 summary chunks |
| Vector Index | 20-50 embeddings | 100+ embeddings | 1-3 embeddings |
| GraphRAG | Yes | Skip | Skip |
| PostgreSQL | Extract if applicable | Extract rows | Extract rows |

---

## 6. Implementation Details

### 6.1 Changes to AdaptiveChunker

**File:** `backend/rag_service/chunking/adaptive_chunker.py`

```python
class AdaptiveChunker:
    """
    Adaptive document chunker with tabular skip optimization.
    """

    def chunk_content(
        self,
        content: str,
        source_name: str,
        file_path: str = "",
        force_strategy: str = None,
        skip_classification: bool = False,
    ) -> AdaptiveChunkingResult:
        """
        Chunk content with optional tabular skip optimization.
        """
        # ... existing STEP 1-2 code ...

        # STEP 2: Document classification
        profile = self.classifier.classify(content=content, filename=source_name)

        logger.info(f"[Chunker] Classification result:")
        logger.info(f"[Chunker]   - Document Type: {profile.document_type}")
        logger.info(f"[Chunker]   - Recommended Strategy: {profile.recommended_strategy}")

        # ========== NEW STEP 3: TABULAR SKIP ANALYSIS ==========
        skip_info = {
            "skip_row_chunking": False,
            "reason": "not_analyzed",
            "analysis": {}
        }

        if profile.recommended_strategy == "table_preserving":
            logger.info("-" * 80)
            logger.info("[Chunker] STEP 3: Tabular Skip Analysis...")

            should_skip, reason, analysis = self._analyze_tabular_skip(
                profile=profile,
                content=content,
                document_type=profile.document_type,
                source_name=source_name
            )

            skip_info = {
                "skip_row_chunking": should_skip,
                "reason": reason,
                "analysis": analysis
            }

            logger.info(f"[Chunker] Skip Analysis Result:")
            logger.info(f"[Chunker]   - Should Skip: {should_skip}")
            logger.info(f"[Chunker]   - Reason: {reason}")
            logger.info(f"[Chunker]   - Score: {analysis.get('skip_score', 0):.2f}")

            if should_skip:
                # ========== NEW STEP 4: GENERATE SUMMARY CHUNKS ==========
                logger.info("-" * 80)
                logger.info("[Chunker] STEP 4: Generating Summary Chunks...")

                chunks = self._generate_tabular_summary_chunks(
                    content=content,
                    source_name=source_name,
                    profile=profile,
                    analysis=analysis
                )

                logger.info(f"[Chunker] Generated {len(chunks)} summary chunks (skipped row chunking)")

                # Skip to post-processing
                return self._finalize_result(chunks, profile, skip_info)

        # ========== STEP 5: Execute normal chunking strategy ==========
        logger.info("-" * 80)
        logger.info("[Chunker] STEP 5: Executing chunking...")

        strategy = get_strategy_for_profile(profile)
        chunks = strategy.chunk(content=content, source_name=source_name)

        # ========== STEP 6: Post-processing ==========
        return self._finalize_result(chunks, profile, skip_info)

    def _analyze_tabular_skip(
        self,
        profile: ChunkingProfile,
        content: str,
        document_type: str,
        source_name: str
    ) -> Tuple[bool, str, Dict]:
        """
        Analyze whether to skip row-level chunking for tabular document.

        See Section 3.1 for full algorithm details.
        """
        # Implementation as described in Section 3.1
        pass

    def _generate_tabular_summary_chunks(
        self,
        content: str,
        source_name: str,
        profile: ChunkingProfile,
        analysis: Dict
    ) -> List[Document]:
        """
        Generate 1-3 summary chunks for tabular document discovery.

        See Section 4 for chunk specifications.
        """
        # Implementation as described in Section 4
        pass
```

### 6.2 ChunkingResult Enhancement

```python
@dataclass
class AdaptiveChunkingResult:
    """Result from adaptive chunking with skip information."""
    chunks: List[Document]
    profile: ChunkingProfile
    skip_info: Dict = field(default_factory=lambda: {
        "skip_row_chunking": False,
        "reason": "not_analyzed",
        "analysis": {}
    })

    @property
    def stats(self) -> Dict:
        return {
            "total_chunks": len(self.chunks),
            "strategy_used": self.profile.recommended_strategy,
            "document_type": self.profile.document_type,
            "skip_row_chunking": self.skip_info.get("skip_row_chunking", False),
            "skip_reason": self.skip_info.get("reason", ""),
        }
```

### 6.3 Remove Upload-Time Detection

**File:** `backend/main.py`

Remove or simplify the tabular detection at upload time:

```python
@app.post("/upload")
async def upload_file(...):
    # ... existing code ...

    # REMOVED: Extension-based tabular detection at upload
    # The decision now happens in AdaptiveChunker during chunking

    # Keep only basic flags for future use
    doc = Document(
        filename=file.filename,
        # ... other fields ...

        # These will be set by the chunker after classification
        is_tabular_data=False,  # Will be updated after chunking
        processing_path="standard",  # Will be updated after chunking
    )
```

### 6.4 Update Document After Chunking

**File:** `backend/rag_service/markdown_chunker.py`

```python
def chunk_markdown_with_summaries(...) -> ChunkingResult:
    """Chunk markdown and update document metadata."""

    result = chunk_markdown_adaptive(md_path, source_name)

    # ★ NEW: Update document with skip info
    if result.skip_info.get("skip_row_chunking"):
        _update_document_tabular_flags(
            source_name=source_name,
            is_tabular=True,
            processing_path="tabular",
            skip_reason=result.skip_info.get("reason")
        )

    return result


def _update_document_tabular_flags(
    source_name: str,
    is_tabular: bool,
    processing_path: str,
    skip_reason: str
):
    """Update document record with tabular processing info."""
    with get_db_session() as db:
        doc = db.query(Document).filter(
            Document.filename.like(f"{source_name}.%")
        ).first()

        if doc:
            doc.is_tabular_data = is_tabular
            doc.processing_path = processing_path

            # Store skip analysis in indexing_details
            if not doc.indexing_details:
                doc.indexing_details = {}
            doc.indexing_details["tabular_skip"] = {
                "skipped": is_tabular,
                "reason": skip_reason
            }

            db.commit()
```

---

## 7. Database Schema (Minimal Changes)

### 7.1 Document Model

The existing fields are sufficient with minor additions:

```python
class Document(Base):
    # EXISTING - Keep as-is
    convert_status = Column(Enum(ConvertStatus))
    index_status = Column(Enum(IndexStatus))
    indexed_chunks = Column(Integer)  # Will be 1-3 for tabular
    indexing_details = Column(JSONB)
    skip_graphrag = Column(Boolean)
    skip_graphrag_reason = Column(String)

    # EXISTING - Already added in v1.0
    is_tabular_data = Column(Boolean, default=False)
    processing_path = Column(String(50), default="standard")
    summary_chunk_ids = Column(ARRAY(String), nullable=True)
```

### 7.2 indexing_details Structure

```json
{
    "vector_indexing": {
        "status": "completed",
        "chunks_indexed": 3
    },
    "tabular_skip": {
        "skipped": true,
        "reason": "score:0.85|doc_type:0.4|table:0.35|rows:0.1",
        "analysis": {
            "document_type": "inventory_report",
            "table_content_ratio": 0.92,
            "row_count_estimate": 500,
            "skip_score": 0.85
        }
    },
    "graphrag_indexing": {
        "status": "skipped",
        "reason": "tabular_document"
    }
}
```

---

## 8. Query Flow (Unchanged from v1.0)

The two-stage query process remains the same:

1. **Stage 1: Document Discovery** - Vector search on summary chunks
2. **Stage 2: Data Analytics** - SQL query on PostgreSQL

See Section 6 of v1.0 design for details.

---

## 9. Configuration

### 9.1 Environment Variables

```bash
# Tabular skip analysis configuration
TABULAR_SKIP_ENABLED=true
TABULAR_SKIP_THRESHOLD=0.6  # Score threshold for skipping

# Content analysis thresholds
TABULAR_TABLE_RATIO_HIGH=0.85
TABULAR_TABLE_RATIO_MEDIUM=0.70
TABULAR_TABLE_RATIO_LOW=0.50
TABULAR_MIN_ROW_COUNT=50

# Summary generation
TABULAR_SUMMARY_USE_LLM=true
TABULAR_SUMMARY_MAX_SAMPLE_ROWS=20
```

### 9.2 Document Type Configuration

```python
# Document types that strongly indicate tabular data
STRONG_TABULAR_TYPES = {
    'spreadsheet', 'invoice', 'receipt', 'bank_statement',
    'credit_card_statement', 'expense_report', 'inventory_report',
    'purchase_order', 'sales_order', 'payroll', 'financial_statement',
    'price_list', 'data_export', 'transaction_log', 'shipping_manifest'
}

# Document types that should NEVER skip chunking
NEVER_SKIP_TYPES = {
    'research_paper', 'academic_article', 'thesis', 'dissertation',
    'contract', 'agreement', 'legal_document', 'terms_of_service',
    'technical_spec', 'user_manual', 'documentation', 'tutorial',
    'news_article', 'blog_post', 'report'
}
```

---

## 10. Testing Strategy

### 10.1 Unit Tests

```python
def test_inventory_report_skips_chunking():
    """Inventory report with high table ratio should skip."""
    content = generate_table_content(rows=500, cols=10)
    profile = ChunkingProfile(
        document_type="inventory_report",
        recommended_strategy="table_preserving"
    )

    should_skip, reason, analysis = should_skip_row_chunking(profile, content)

    assert should_skip == True
    assert analysis["skip_score"] >= 0.6

def test_research_paper_never_skips():
    """Research paper should never skip even with tables."""
    content = generate_mixed_content(table_ratio=0.4)
    profile = ChunkingProfile(
        document_type="research_paper",
        recommended_strategy="table_preserving"
    )

    should_skip, reason, analysis = should_skip_row_chunking(profile, content)

    assert should_skip == False
    assert "document_type_excluded" in reason

def test_hybrid_document_no_skip():
    """Document with 40% tables should not skip."""
    content = generate_mixed_content(table_ratio=0.4)
    profile = ChunkingProfile(
        document_type="unknown",
        recommended_strategy="table_preserving"
    )

    should_skip, reason, analysis = should_skip_row_chunking(profile, content)

    assert should_skip == False
    assert analysis["skip_score"] < 0.6
```

### 10.2 Integration Tests

```python
def test_csv_upload_generates_summary_chunks():
    """CSV file should generate summary chunks instead of row chunks."""
    response = client.post("/upload", files={"file": large_csv_file})
    doc_id = response.json()["document_id"]

    # Wait for processing
    wait_for_indexing(doc_id)

    # Verify chunking result
    doc = get_document(doc_id)
    assert doc.is_tabular_data == True
    assert doc.indexed_chunks <= 3  # Summary chunks only

    # Verify summary chunks exist in Qdrant
    results = vectorstore.search(filter={"document_id": doc_id})
    assert len(results) <= 3
    assert all(r.metadata.get("is_summary_chunk") for r in results)

def test_pdf_invoice_detected_and_skipped():
    """PDF invoice should be detected as tabular and skip row chunking."""
    response = client.post("/upload", files={"file": invoice_pdf_file})
    doc_id = response.json()["document_id"]

    wait_for_indexing(doc_id)

    doc = get_document(doc_id)
    # PDF detected as invoice by content classification
    assert doc.is_tabular_data == True
    assert doc.indexed_chunks <= 3
```

---

## 11. Migration from v1.0

### 11.1 Changes Required

| Component | v1.0 Behavior | v2.0 Behavior | Migration |
|-----------|---------------|---------------|-----------|
| Upload detection | Extension-based | None (moved to chunker) | Remove detection code |
| Task queue check | Check `is_tabular_data` flag | Remove check | Flag set after chunking |
| AdaptiveChunker | No skip analysis | New skip analysis step | Add new methods |
| ChunkingResult | No skip info | Includes skip_info | Extend dataclass |

### 11.2 Backward Compatibility

- Documents already processed keep their current state
- New uploads use the v2.0 flow
- Existing `is_tabular_data` flags remain valid

---

## 12. Summary

### 12.1 Key Changes from v1.0

| Aspect | v1.0 | v2.0 |
|--------|------|------|
| **Detection Location** | Upload endpoint | Inside AdaptiveChunker |
| **Detection Method** | File extension | Classification + Content Analysis |
| **Decision Point** | Before conversion | After classification |
| **Hybrid Support** | None | Yes (score-based) |
| **Accuracy** | Extension-limited | Content-aware |

### 12.2 Benefits of v2.0 Approach

1. **More Accurate Detection** - Uses content classification, not just extension
2. **Handles PDF Invoices** - Detects tabular content regardless of file type
3. **Hybrid Document Support** - Research papers with tables handled correctly
4. **Unified Flow** - All documents go through same chunking entry point
5. **Simpler Upload** - No special handling needed at upload time
6. **Better Logging** - Decision made with full classification context

### 12.3 Files to Modify

| File | Change |
|------|--------|
| `backend/rag_service/chunking/adaptive_chunker.py` | Add skip analysis and summary generation |
| `backend/rag_service/markdown_chunker.py` | Update document flags after chunking |
| `backend/main.py` | Remove upload-time tabular detection |
| `backend/services/task_queue_service.py` | Remove pre-chunking tabular check |

---

*End of Design Document v2.0*
