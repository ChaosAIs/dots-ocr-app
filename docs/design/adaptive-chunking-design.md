# Adaptive Chunking System Design

> **Version**: 1.0
> **Last Updated**: 2025-12-20
> **Status**: Implemented

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Workflow](#3-workflow)
4. [Components](#4-components)
5. [Domain & Document Taxonomy](#5-domain--document-taxonomy)
6. [Chunking Strategies](#6-chunking-strategies)
7. [Configuration](#7-configuration)
8. [Metadata Schema](#8-metadata-schema)
9. [Usage Examples](#9-usage-examples)
10. [Performance Considerations](#10-performance-considerations)

---

## 1. Overview

### 1.1 Problem Statement

Traditional chunking approaches use a one-size-fits-all strategy (e.g., split by headers with fixed chunk size). This fails to preserve semantic integrity for specialized documents:

| Document Type | Problem with Generic Chunking |
|---------------|-------------------------------|
| Legal contracts | Clauses split mid-sentence, losing legal context |
| Research papers | Citations separated from context, abstracts fragmented |
| Medical records | SOAP sections mixed, vital signs scattered |
| Receipts/Invoices | Line items separated from totals |
| Technical specs | Requirements split from rationale |

### 1.2 Solution

The **Adaptive Chunking System** provides:

- **Pre-classification**: Analyze document before chunking to determine optimal strategy
- **Domain-aware splitting**: 9 specialized strategies for different document types
- **Rich metadata**: Domain-specific metadata extraction (ICD codes, citations, clause numbers, etc.)
- **Configurable**: Environment-based configuration with sensible defaults

### 1.3 Key Features

| Feature | Description |
|---------|-------------|
| 9 Document Domains | Legal, Academic, Medical, Engineering, Education, Financial, Government, Professional, General |
| 72 Document Types | Comprehensive taxonomy from receipts to research papers |
| 9 Chunking Strategies | Optimized splitting logic per domain |
| Rule-based Classification | Fast pattern matching (no LLM required) |
| Optional LLM Classification | Higher accuracy when needed |
| Backward Compatible | Opt-in via environment variable |

---

## 2. Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ENTRY POINTS                                    │
│  markdown_chunker.py                                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  chunk_markdown_with_summaries()  ─┐                                        │
│  chunk_markdown_adaptive()         ├──► Unified interface                   │
│  chunk_content_adaptive()         ─┘                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ADAPTIVE CHUNKER                                   │
│  chunking/adaptive_chunker.py                                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  AdaptiveChunker                                                             │
│  ├── chunk_content()      Main orchestration                                │
│  ├── _preprocess_content()  Remove base64, clean content                    │
│  ├── _postprocess_chunks()  Add metadata, dates, importance                 │
│  └── _calculate_importance()  Score chunk relevance                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│  DOCUMENT CLASSIFIER │ │  CHUNKING STRATEGIES │ │   DOMAIN PATTERNS    │
│  document_classifier │ │  chunking_strategies │ │   domain_patterns    │
│  .py                 │ │  .py                 │ │   .py                │
├──────────────────────┤ ├──────────────────────┤ ├──────────────────────┤
│ • LLM classification │ │ • 9 strategy classes │ │ • Pattern definitions│
│ • Rule-based classif.│ │ • Strategy factory   │ │ • Domain configs     │
│ • Marker detection   │ │ • Base abstractions  │ │ • Type mappings      │
└──────────────────────┘ └──────────────────────┘ └──────────────────────┘
```

### 2.2 File Structure

```
backend/rag_service/
├── markdown_chunker.py          # Entry points (existing, modified)
└── chunking/                    # New module
    ├── __init__.py              # Module exports
    ├── domain_patterns.py       # Domain/type taxonomy & patterns
    ├── chunk_metadata.py        # Metadata dataclasses
    ├── document_classifier.py   # Pre-chunking classification
    ├── chunking_strategies.py   # 9 strategy implementations
    └── adaptive_chunker.py      # Main orchestrator
```

---

## 3. Workflow

### 3.1 Complete Processing Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           DOCUMENT INPUT                                      │
│                    (Markdown file or content string)                         │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: PRE-PROCESSING                                                       │
│ adaptive_chunker.py:277-283                                                  │
│ ┌──────────────────────────────────────────────────────────────────────────┐ │
│ │ • Remove base64 images → replaced with [image]                           │ │
│ │ • Clean base64-like patterns                                             │ │
│ └──────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: DOCUMENT CLASSIFICATION                                              │
│ document_classifier.py:133-411                                               │
│ ┌──────────────────────────────────────────────────────────────────────────┐ │
│ │ 2a. Detect Structural Markers (regex)                                    │ │
│ │     has_tables, has_headers, has_citations, has_clauses...              │ │
│ │                                                                          │ │
│ │ 2b. Calculate Size: tokens = chars / 4                                   │ │
│ │     is_small (< 1000), is_large (> 10000)                               │ │
│ │                                                                          │ │
│ │ 2c. Classify:                                                            │ │
│ │     ├── LLM Mode: Send to LLM with UNIVERSAL_PRE_CHUNKING_PROMPT        │ │
│ │     └── Rule Mode: Pattern match → Infer type → Override domain         │ │
│ └──────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│ OUTPUT: ChunkingProfile                                                      │
│ ┌──────────────────────────────────────────────────────────────────────────┐ │
│ │ document_type: "research_paper"                                          │ │
│ │ document_domain: "academic"                                              │ │
│ │ recommended_strategy: "academic_structure"                               │ │
│ │ recommended_chunk_size: 768                                              │ │
│ │ recommended_overlap_percent: 15                                          │ │
│ └──────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: STRATEGY SELECTION                                                   │
│ chunking_strategies.py:1179-1203                                             │
│ ┌──────────────────────────────────────────────────────────────────────────┐ │
│ │ get_strategy_for_profile(profile)                                        │ │
│ │                                                                          │ │
│ │ Domain → Strategy Mapping:                                               │ │
│ │ ┌────────────────┬──────────────────────┬──────────┬─────────┐          │ │
│ │ │ Domain         │ Strategy             │ Chunk    │ Overlap │          │ │
│ │ ├────────────────┼──────────────────────┼──────────┼─────────┤          │ │
│ │ │ financial      │ whole_document       │ 0 (all)  │ 0%      │          │ │
│ │ │ legal          │ clause_preserving    │ 1024     │ 5%      │          │ │
│ │ │ academic       │ academic_structure   │ 768      │ 15%     │          │ │
│ │ │ medical        │ medical_section      │ 1024     │ 5%      │          │ │
│ │ │ engineering    │ requirement_based    │ 1024     │ 5%      │          │ │
│ │ │ education      │ educational_unit     │ 768      │ 10%     │          │ │
│ │ │ general        │ semantic_header      │ 512      │ 10%     │          │ │
│ │ └────────────────┴──────────────────────┴──────────┴─────────┘          │ │
│ └──────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: CHUNKING EXECUTION                                                   │
│ chunking_strategies.py (each strategy class)                                 │
│ ┌──────────────────────────────────────────────────────────────────────────┐ │
│ │ strategy.chunk(content, source_name, file_path)                          │ │
│ │                                                                          │ │
│ │ Each strategy has specialized splitting logic:                           │ │
│ │ • WholeDocumentStrategy: Keep as single chunk                           │ │
│ │ • SemanticHeaderStrategy: Split by markdown headers                     │ │
│ │ • ClausePreservingStrategy: Split by legal clauses                      │ │
│ │ • AcademicStructureStrategy: Preserve abstract, citations               │ │
│ │ • MedicalSectionStrategy: Split by SOAP sections                        │ │
│ │ • RequirementBasedStrategy: Preserve requirement IDs                    │ │
│ │ • EducationalUnitStrategy: Preserve learning objectives                 │ │
│ │ • TablePreservingStrategy: Keep tables intact                           │ │
│ │ • ParagraphStrategy: Split by paragraph boundaries                      │ │
│ └──────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: POST-PROCESSING                                                      │
│ adaptive_chunker.py:285-318                                                  │
│ ┌──────────────────────────────────────────────────────────────────────────┐ │
│ │ 1. Add document classification to metadata:                              │ │
│ │    document_type, document_domain, content_density, structure_type      │ │
│ │                                                                          │ │
│ │ 2. Date Normalization (if available):                                    │ │
│ │    dates_normalized, primary_date, date_year, date_month, date_day      │ │
│ │                                                                          │ │
│ │ 3. Calculate Importance Score:                                           │ │
│ │    • Base: 0.5                                                           │ │
│ │    • +0.3 if is_key_section                                              │ │
│ │    • +0.2 if abstract/summary/conclusion                                 │ │
│ │    • +0.1 if first chunk                                                 │ │
│ │    • Domain-specific bonuses                                             │ │
│ └──────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                            OUTPUT                                             │
│                    List[Document] with rich metadata                         │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Classification Decision Tree

```
                         ┌─────────────────┐
                         │  Document Input │
                         └────────┬────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │  Detect Structural Markers │
                    └─────────────┬─────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
     ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
     │ tokens < 1000  │  │ 1000-10000     │  │ tokens > 10000 │
     │ (Small Doc)    │  │ (Normal Doc)   │  │ (Large Doc)    │
     └───────┬────────┘  └───────┬────────┘  └───────┬────────┘
             │                   │                   │
             ▼                   ▼                   ▼
     ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
     │ Has Headers?   │  │ Domain Pattern │  │ Aggressive     │
     │ No → whole_doc │  │ Matching       │  │ Splitting      │
     │ Yes → semantic │  │                │  │                │
     └────────────────┘  └───────┬────────┘  └────────────────┘
                                 │
         ┌───────────┬───────────┼───────────┬───────────┐
         ▼           ▼           ▼           ▼           ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
    │ Legal   │ │Academic │ │Medical  │ │Engineer │ │Education│
    │ clause_ │ │academic_│ │medical_ │ │require_ │ │education│
    │ preserv │ │structure│ │section  │ │ment_    │ │al_unit  │
    └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
```

---

## 4. Components

### 4.1 Document Classifier

**File**: `chunking/document_classifier.py`

**Purpose**: Analyze document content before chunking to determine optimal strategy.

**Key Features**:
- Structural marker detection (regex-based)
- Domain pattern matching
- Document type inference
- LLM-based classification (optional)

**Structural Markers Detected**:

| Marker | Detection Pattern | Example |
|--------|-------------------|---------|
| `has_tables` | `<table>` or `\|...\|` | Markdown tables |
| `has_headers` | `^#{1,6}\s+` | `## Section` |
| `has_lists` | `^\s*[-*•]\s+` or `\d+\.` | Bullet points |
| `has_code` | ` ``` ` or `<code>` | Code blocks |
| `has_equations` | `$$` or `\begin{equation}` | LaTeX math |
| `has_numbered_clauses` | `^\s*\d+\.\d+` | `1.1 Definitions` |
| `has_citations` | `\[\d+\]` or `(Author, 2024)` | Academic citations |
| `has_form_fields` | `:\s*_{3,}` or `[\s*]` | Form blanks |
| `has_dates` | `\d{1,2}[/-]\d{1,2}[/-]\d{2,4}` | Date formats |
| `has_amounts` | `[$€£¥]\s*[\d,]+` | Currency amounts |

### 4.2 Chunking Strategies

**File**: `chunking/chunking_strategies.py`

**Base Class**: `ChunkingStrategy` (abstract)

```python
class ChunkingStrategy(ABC):
    def __init__(self, chunk_size, chunk_overlap, profile):
        ...

    @abstractmethod
    def chunk(self, content, source_name, file_path, existing_metadata) -> List[Document]:
        ...
```

**Strategy Implementations**:

| Strategy | Target Domain | Key Behavior |
|----------|---------------|--------------|
| `WholeDocumentStrategy` | Financial (small) | No splitting, atomic unit |
| `SemanticHeaderStrategy` | General | Split by markdown headers |
| `ClausePreservingStrategy` | Legal | Never split within clauses |
| `AcademicStructureStrategy` | Academic | Preserve abstract, citations |
| `MedicalSectionStrategy` | Medical | Split by SOAP sections |
| `RequirementBasedStrategy` | Engineering | Preserve REQ-IDs |
| `EducationalUnitStrategy` | Education | Preserve learning objectives |
| `TablePreservingStrategy` | Financial | Keep tables intact |
| `ParagraphStrategy` | General | Split at paragraph boundaries |

### 4.3 Domain Patterns

**File**: `chunking/domain_patterns.py`

**Purpose**: Define domain-specific regex patterns and configurations.

**Pattern Categories**:

```python
LEGAL_PATTERNS = {
    "clause_number": r"^\s*(\d+\.)+\d*\s+",
    "article": r"^Article\s+[IVXLCDM\d]+",
    "whereas": r"^WHEREAS",
    "definition": r'"[^"]+" means|shall mean',
}

ACADEMIC_PATTERNS = {
    "abstract": r"^Abstract",
    "citation_bracket": r"\[[\d,\s\-]+\]",
    "citation_author_year": r"\(Author, \d{4}\)",
    "equation": r"\$\$.*\$\$",
}

MEDICAL_PATTERNS = {
    "soap_section": r"Chief Complaint|HPI|Assessment|Plan",
    "icd_code": r"[A-Z]\d{2}(\.\d{1,2})?",
    "vital_signs": r"BP|HR|RR|Temp|SpO2",
}
```

### 4.4 Chunk Metadata

**File**: `chunking/chunk_metadata.py`

**Purpose**: Define metadata structures for chunks.

**Classes**:

```
ChunkingProfile          ← Pre-classification results
UniversalChunkMetadata   ← Base metadata for all chunks
├── LegalChunkMetadata       ← clause_number, party_names
├── AcademicChunkMetadata    ← citations, is_abstract, doi
├── MedicalChunkMetadata     ← icd_codes, medications
├── EngineeringChunkMetadata ← requirement_ids, priority_level
├── EducationChunkMetadata   ← learning_objectives, exercises
├── FinancialChunkMetadata   ← amounts, currency, totals
└── GovernmentChunkMetadata  ← form_id, legal_references
```

---

## 5. Domain & Document Taxonomy

### 5.1 Document Domains (9)

| Domain | Description | Recommended Strategy |
|--------|-------------|---------------------|
| `legal` | Contracts, agreements, ToS | `clause_preserving` |
| `academic` | Research papers, theses | `academic_structure` |
| `medical` | Clinical notes, records | `medical_section` |
| `engineering` | Technical specs, API docs | `requirement_based` |
| `education` | Textbooks, course materials | `educational_unit` |
| `financial` | Receipts, invoices | `whole_document` / `table_preserving` |
| `government` | Forms, permits, legislation | `semantic_header` |
| `professional` | Resumes, job descriptions | `semantic_header` |
| `general` | Reports, articles, memos | `semantic_header` |

### 5.2 Document Types (72)

**Business & Finance**:
- receipt, invoice, financial_report, bank_statement, tax_document, expense_report, purchase_order, quotation

**Legal & Compliance**:
- contract, agreement, legal_brief, court_filing, patent, regulatory_filing, terms_of_service, privacy_policy, compliance_doc

**Technical & Engineering**:
- technical_spec, api_documentation, user_manual, installation_guide, architecture_doc, datasheet, schematic, code_documentation

**Academic & Research**:
- research_paper, thesis, academic_article, literature_review, case_study, lab_report, conference_paper, grant_proposal

**Medical & Healthcare**:
- medical_record, clinical_report, prescription, lab_result, insurance_claim, patient_summary, discharge_summary

**Education & Training**:
- textbook, course_material, syllabus, lecture_notes, exam, tutorial, certification_material

**Government & Public**:
- government_form, permit, license, official_notice, policy_document, legislation

**Professional & HR**:
- resume, cover_letter, job_description, performance_review, employee_handbook, training_material

**General**:
- report, memo, letter, email, meeting_notes, presentation, article, blog_post, other

---

## 6. Chunking Strategies

### 6.1 WholeDocumentStrategy

**Use Case**: Small atomic documents that should not be split

**Examples**: Single-page receipts, short invoices, memos, emails

**Behavior**:
```
Input: Document < 1000 tokens
Output: 1 chunk with is_atomic=True
```

**Configuration**:
- Chunk size: 0 (entire document)
- Overlap: 0%

### 6.2 SemanticHeaderStrategy

**Use Case**: Well-structured markdown documents with clear headers

**Examples**: Reports, manuals, resumes, structured documents

**Behavior**:
```
1. Split by markdown headers (#, ##, ###, etc.)
2. If section > max_chunk_size, apply recursive splitting
3. Preserve header hierarchy in metadata
```

**Configuration**:
- Chunk size: 512
- Overlap: 10%
- Max chunk size: 1024

### 6.3 ClausePreservingStrategy

**Use Case**: Legal documents with numbered clauses

**Examples**: Contracts, agreements, terms of service

**Behavior**:
```
1. Extract clauses using patterns:
   - Numbered: 1.1, 1.1.1, etc.
   - Articles: Article I, Article 2
   - Sections: Section 1
2. Group clauses that fit within chunk_size
3. NEVER split individual clauses
```

**Extracted Metadata**:
- `clause_number`: "1.1"
- `clause_range`: "1.1 - 1.3"
- `is_definition`: true/false

**Configuration**:
- Chunk size: 1024
- Overlap: 5%

### 6.4 AcademicStructureStrategy

**Use Case**: Research papers, theses, academic articles

**Behavior**:
```
1. Extract abstract as separate high-importance chunk
2. Split remaining by headers
3. Extract citations and figure references
4. Detect section types (intro, methods, results...)
```

**Extracted Metadata**:
- `section_type`: "abstract", "methods", "results"
- `citations_in_chunk`: ["[1]", "[2]"]
- `figures_referenced`: ["Figure 1", "Table 2"]
- `is_abstract`: true/false
- `is_bibliography`: true/false

**Configuration**:
- Chunk size: 768
- Overlap: 15%

### 6.5 MedicalSectionStrategy

**Use Case**: Medical records, clinical notes, discharge summaries

**Behavior**:
```
1. Split by SOAP/medical section headers:
   Chief Complaint, HPI, PMH, Medications, Assessment, Plan, etc.
2. Extract ICD codes, medications, vital signs
3. Mark Assessment/Plan as key sections
```

**Section Headers Recognized**:
- Chief Complaint, CC
- History of Present Illness, HPI
- Past Medical History, PMH
- Medications, Allergies
- Physical Exam, PE
- Assessment, Plan
- Vital Signs, Labs

**Extracted Metadata**:
- `section_type`: "HPI", "Assessment"
- `icd_codes`: ["J06.9", "I10"]
- `medications_mentioned`: ["Lisinopril 10mg"]
- `vitals_present`: true/false

**Configuration**:
- Chunk size: 1024
- Overlap: 5%

### 6.6 RequirementBasedStrategy

**Use Case**: Technical specifications, requirements documents

**Behavior**:
```
1. Use semantic header as base
2. Extract requirement IDs (REQ-001, SPEC-002)
3. Detect priority (shall/must/should/may)
4. Extract diagram references
```

**Extracted Metadata**:
- `requirement_ids`: ["REQ-001", "REQ-002"]
- `priority_level`: "shall", "should", "may"
- `version`: "1.2.0"
- `diagrams_referenced`: ["Figure 3"]

**Configuration**:
- Chunk size: 1024
- Overlap: 5%

### 6.7 EducationalUnitStrategy

**Use Case**: Textbooks, course materials, tutorials

**Behavior**:
```
1. Use semantic header as base
2. Detect chapters, learning objectives
3. Identify exercises, examples, definitions
```

**Extracted Metadata**:
- `chapter`: "Chapter 5"
- `learning_objectives`: ["Understand X", "Apply Y"]
- `contains_exercise`: true/false
- `key_terms`: ["term1", "term2"]
- `is_example`: true/false

**Configuration**:
- Chunk size: 768
- Overlap: 10%

### 6.8 TablePreservingStrategy

**Use Case**: Documents with significant tabular data

**Examples**: Bank statements, invoices with line items

**Behavior**:
```
1. Extract table boundaries (HTML or Markdown)
2. Keep tables intact
3. Split large tables by rows (preserve headers)
4. Use semantic header for non-table content
```

**Extracted Metadata**:
- `table_type`: "html", "markdown"
- `table_part`: 1
- `table_total_parts`: 3
- `contains_table`: true

**Configuration**:
- Chunk size: 1024
- Overlap: 0%

### 6.9 ParagraphStrategy

**Use Case**: Narrative documents without strong structure

**Examples**: Articles, blog posts, narrative text

**Behavior**:
```
1. Split on paragraph breaks (\n\n)
2. Combine small paragraphs up to chunk_size
3. Add overlap between chunks
```

**Configuration**:
- Chunk size: 768
- Overlap: ~13%

---

## 7. Configuration

### 7.1 Environment Variables

```bash
# ==============================================================================
# ADAPTIVE CHUNKING CONFIGURATION
# ==============================================================================

# Enable adaptive chunking (recommended: true)
ADAPTIVE_CHUNKING_ENABLED=true

# Use LLM for classification (recommended: false for speed)
ADAPTIVE_CHUNKING_USE_LLM=false

# Domain-specific chunk sizes
CHUNKING_LEGAL_SIZE=1024
CHUNKING_LEGAL_OVERLAP_PERCENT=5

CHUNKING_ACADEMIC_SIZE=768
CHUNKING_ACADEMIC_OVERLAP_PERCENT=15

CHUNKING_MEDICAL_SIZE=1024
CHUNKING_MEDICAL_OVERLAP_PERCENT=5

CHUNKING_ENGINEERING_SIZE=1024
CHUNKING_ENGINEERING_OVERLAP_PERCENT=5

CHUNKING_EDUCATION_SIZE=768
CHUNKING_EDUCATION_OVERLAP_PERCENT=10

CHUNKING_FINANCIAL_SIZE=1024
CHUNKING_FINANCIAL_OVERLAP_PERCENT=15

CHUNKING_GENERAL_SIZE=512
CHUNKING_GENERAL_OVERLAP_PERCENT=10

# Size thresholds
CHUNKING_SMALL_DOC_TOKENS=1000    # Below → may keep whole
CHUNKING_LARGE_DOC_TOKENS=10000   # Above → aggressive splitting
```

### 7.2 Configuration Comparison

| Setting | Value | Effect |
|---------|-------|--------|
| `ENABLED=false, USE_LLM=*` | Original chunking | Basic header-based splitting |
| `ENABLED=true, USE_LLM=false` | **Recommended** | Fast rule-based adaptive chunking |
| `ENABLED=true, USE_LLM=true` | Full adaptive | LLM-based classification (slower, more accurate) |

---

## 8. Metadata Schema

### 8.1 Universal Metadata (All Chunks)

```json
{
  "chunk_id": "uuid",
  "source": "filename",
  "file_path": "/path/to/file.md",
  "chunk_index": 0,
  "total_chunks": 7,
  "heading_path": "Section → Subsection",

  "document_type": "research_paper",
  "document_domain": "academic",
  "content_density": "normal",
  "structure_type": "hierarchical",

  "chunking_strategy": "academic_structure",
  "chunk_size_used": 768,
  "overlap_applied": 115,
  "is_atomic": false,

  "contains_table": false,
  "contains_list": true,
  "contains_code": false,
  "contains_image_ref": false,
  "contains_equation": true,

  "dates_normalized": ["2024-01-15"],
  "primary_date": "2024-01-15",
  "date_year": 2024,
  "date_month": 1,
  "date_day": 15,

  "importance_score": 0.8,
  "is_key_section": true
}
```

### 8.2 Domain-Specific Metadata

**Legal**:
```json
{
  "clause_number": "1.1",
  "clause_hierarchy": ["Article I", "Section 1", "1.1"],
  "is_definition": true,
  "party_names": ["Company A", "Company B"],
  "effective_date": "2024-01-01"
}
```

**Academic**:
```json
{
  "section_type": "methods",
  "citations_in_chunk": ["[1]", "[2]", "(Smith, 2024)"],
  "figures_referenced": ["Figure 1", "Table 2"],
  "is_abstract": false,
  "is_bibliography": false,
  "doi": "10.1234/example"
}
```

**Medical**:
```json
{
  "section_type": "Assessment",
  "icd_codes": ["J06.9", "I10"],
  "medications_mentioned": ["Lisinopril 10mg"],
  "vitals_present": true,
  "is_phi_redacted": false
}
```

**Engineering**:
```json
{
  "requirement_ids": ["REQ-001", "REQ-002"],
  "priority_level": "shall",
  "version": "1.2.0",
  "diagrams_referenced": ["Figure 3"],
  "traces_to": ["SRS-100"]
}
```

**Education**:
```json
{
  "chapter": "Chapter 5",
  "learning_objectives": ["Understand X"],
  "contains_exercise": true,
  "key_terms": ["machine learning", "neural network"],
  "difficulty_level": "intermediate"
}
```

**Financial**:
```json
{
  "amounts_detected": [3.99, 2.49, 12.39],
  "currency": "USD",
  "total_amount": 12.39,
  "tax_amount": 0.92,
  "vendor_name": "ABC Grocery",
  "payment_method": "Credit Card"
}
```

---

## 9. Usage Examples

### 9.1 Basic Usage (Default Configuration)

```python
from rag_service.markdown_chunker import chunk_markdown_with_summaries

# With ADAPTIVE_CHUNKING_ENABLED=true in .env
result = chunk_markdown_with_summaries("/path/to/document.md")

for chunk in result.chunks:
    print(f"Chunk {chunk.metadata['chunk_index']}: {chunk.metadata['heading_path']}")
    print(f"  Domain: {chunk.metadata['document_domain']}")
    print(f"  Strategy: {chunk.metadata['chunking_strategy']}")
```

### 9.2 Explicit Adaptive Chunking

```python
from rag_service.markdown_chunker import chunk_markdown_adaptive

# Force adaptive chunking with specific options
result = chunk_markdown_adaptive(
    md_path="/path/to/contract.md",
    source_name="NDA_2024",
    use_llm=False,  # Fast rule-based
    force_strategy=None  # Auto-detect
)
```

### 9.3 Force Specific Strategy

```python
from rag_service.markdown_chunker import chunk_markdown_adaptive

# Force legal strategy regardless of content
result = chunk_markdown_adaptive(
    md_path="/path/to/document.md",
    force_strategy="clause_preserving"
)
```

### 9.4 Content-Based Chunking (No File)

```python
from rag_service.markdown_chunker import chunk_content_adaptive

content = """
## Abstract
This research investigates...

## 1. Introduction
Climate change has been identified...
"""

result = chunk_content_adaptive(
    content=content,
    source_name="climate_paper",
    use_llm=False
)
```

### 9.5 Get Classification Without Chunking

```python
from rag_service.markdown_chunker import get_chunking_profile

profile = get_chunking_profile(
    content="...",
    filename="contract.md",
    use_llm=False
)

print(f"Type: {profile['document_type']}")
print(f"Domain: {profile['document_domain']}")
print(f"Strategy: {profile['recommended_strategy']}")
```

---

## 10. Performance Considerations

### 10.1 Classification Mode Comparison

| Mode | Speed | Accuracy | Cost | Use Case |
|------|-------|----------|------|----------|
| Rule-based | ~1ms | 85-90% | Free | Production (recommended) |
| LLM-based | ~500-2000ms | 95%+ | Token cost | Complex/unusual documents |

### 10.2 Strategy Performance

| Strategy | Complexity | Typical Chunks | Notes |
|----------|------------|----------------|-------|
| `whole_document` | O(1) | 1 | No splitting |
| `semantic_header` | O(n) | 5-20 | Header-based |
| `clause_preserving` | O(n) | 10-50 | Regex extraction |
| `academic_structure` | O(n) | 7-15 | Abstract + headers |
| `medical_section` | O(n) | 8-20 | Section detection |
| `table_preserving` | O(n) | Variable | Table extraction |

### 10.3 Memory Considerations

- Documents are processed in-memory
- Base64 images are stripped to reduce size
- Large documents (>10000 tokens) use aggressive splitting
- Chunk metadata adds ~500 bytes per chunk

### 10.4 Fallback Behavior

If any step fails, the system falls back gracefully:

```
Adaptive Chunking
      │
      ├── Classification fails → Use semantic_header
      │
      ├── Strategy fails → Use semantic_header
      │
      └── Entire adaptive fails → Use original chunking
```

---

## Appendix A: File Reference

| File | Lines | Description |
|------|-------|-------------|
| `markdown_chunker.py` | ~700 | Entry points, legacy + adaptive |
| `chunking/__init__.py` | ~95 | Module exports |
| `chunking/domain_patterns.py` | ~875 | 9 domains, 72 types, patterns |
| `chunking/chunk_metadata.py` | ~485 | Metadata dataclasses |
| `chunking/document_classifier.py` | ~575 | Classification logic |
| `chunking/chunking_strategies.py` | ~1235 | 9 strategy implementations |
| `chunking/adaptive_chunker.py` | ~410 | Orchestrator |

---

## Appendix B: Pattern Reference

### Legal Patterns
- `clause_number`: `^\s*(\d+\.)+\d*\s+`
- `article`: `^Article\s+[IVXLCDM\d]+`
- `whereas`: `^WHEREAS`
- `definition`: `"[^"]+" means|shall mean`

### Academic Patterns
- `abstract`: `^Abstract`
- `citation_bracket`: `\[[\d,\s\-]+\]`
- `citation_author_year`: `\([A-Z][a-z]+,?\s*\d{4}\)`
- `equation`: `\$\$.*\$\$`

### Medical Patterns
- `soap_section`: `Chief Complaint|HPI|Assessment|Plan`
- `icd_code`: `[A-Z]\d{2}(\.\d{1,2})?`
- `medication`: `\d+\s*(mg|mcg|ml|units)`
- `vital_signs`: `BP|HR|RR|Temp|SpO2`

### Engineering Patterns
- `requirement_id`: `(REQ|SPEC|SRS)[-_]?\d+`
- `shall_requirement`: `shall|must|should|may`
- `version`: `[Vv]ersion?\s*[\d\.]+`

### Education Patterns
- `chapter`: `^Chapter|Unit|Module\s+\d+`
- `learning_objective`: `Learning\s+Objectives?`
- `exercise`: `^Exercise|Problem\s+\d+`

---

*End of Design Document*
