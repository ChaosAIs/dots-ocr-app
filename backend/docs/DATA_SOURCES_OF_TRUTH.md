# Data Sources of Truth

This document defines the authoritative sources for key data fields across the system.

## Overview

The system uses multiple storage backends:
- **PostgreSQL**: Primary relational database (authoritative source)
- **Qdrant**: Vector database for semantic search (read-only cache)
- **Neo4j**: Graph database for entity relationships (optional)

## Field Definitions

### schema_type
**Authoritative Source**: `documents_data.schema_type` (PostgreSQL column)

| Location | Type | Purpose | Notes |
|----------|------|---------|-------|
| `documents_data.schema_type` | VARCHAR(64) | Primary storage | Set during extraction |
| Qdrant chunk metadata | String | Search cache | Copied during indexing |

**DO NOT** store schema_type in `documents.document_metadata` JSONB.

---

### document_types
**Authoritative Source**: `documents.document_metadata['document_types']` (PostgreSQL JSONB)

| Location | Type | Purpose | Notes |
|----------|------|---------|-------|
| `documents.document_metadata['document_types']` | Array[String] | Classification | Set during early routing |
| Qdrant chunk metadata (document_type) | String | Chunk classification | Single type per chunk |

**Note**: PostgreSQL stores an array, Qdrant chunks use single string per chunk.

---

### vendor_name / customer_name
**Authoritative Source**: `documents_data.header_data` (PostgreSQL JSONB)

| Location | Type | Purpose | Notes |
|----------|------|---------|-------|
| `documents_data.header_data['vendor_name']` | String | Extracted value | Primary source |
| `documents.document_metadata['entities']` | Object | NER results | Secondary/enriched |
| Qdrant tabular_summary metadata | String | Search cache | Copied during indexing |

---

### invoice_number / invoice_date / total_amount
**Authoritative Source**: `documents_data.header_data` (PostgreSQL JSONB)

| Location | Type | Purpose | Notes |
|----------|------|---------|-------|
| `documents_data.header_data` | JSONB | Extracted values | Primary source |
| `documents_data.summary_data` | JSONB | Aggregated values | For totals |
| Qdrant tabular_summary metadata | Various | Search cache | Copied during indexing |

---

### Extraction Status
**Authoritative Source**: `documents.indexing_details` (PostgreSQL JSONB) for detailed status

| Location | Type | Purpose | Notes |
|----------|------|---------|-------|
| `documents.extraction_status` | VARCHAR(20) | Quick filtering | Convenience column |
| `documents.indexing_details['metadata_extraction']` | JSONB | Detailed status | Full state tracking |
| `document_status_log` | Table | Audit trail | Historical changes |

**Note**: Use `extraction_status` column for queries, `indexing_details` for full state.

---

## Data Flow

```
Document Upload
    │
    ▼
Early Classification
    │
    ├─► documents.document_metadata['document_types'] = ["invoice"]
    │
    ▼
Data Extraction (if eligible)
    │
    ├─► documents_data.schema_type = "invoice"
    ├─► documents_data.header_data = {"vendor_name": "...", ...}
    ├─► documents_data.summary_data = {"total_amount": ...}
    │
    ▼
Vector Indexing
    │
    └─► Qdrant tabular_summary chunk
        └─► metadata = {schema_type, vendor_name, ...}  (CACHE)
```

## Guidelines

1. **Always write to PostgreSQL first** - It's the source of truth
2. **Qdrant is a cache** - Metadata in Qdrant should match PostgreSQL
3. **Don't duplicate across JSONB columns** - Pick one location per field
4. **Use columns for filtering, JSONB for details** - Balance query performance with flexibility

## Updating Data

When updating extracted data:
1. Update `documents_data` in PostgreSQL
2. Re-index affected chunks to Qdrant (to sync cache)
3. Don't update Qdrant directly - always via re-indexing

## Migration Notes

- `extraction_schema_type` column was removed (use `documents_data.schema_type`)
- `metadatas` Qdrant collection was removed (use `documents` collection with `tabular_summary` chunks)
- `schema_type` was removed from `document_metadata` JSONB (use `documents_data.schema_type`)
