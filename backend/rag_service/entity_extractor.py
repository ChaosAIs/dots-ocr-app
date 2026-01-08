"""
Entity Extraction and Normalization Module

This module provides NER-based entity extraction and normalization for documents.
It enhances metadata quality by:
1. Extracting entities (organizations, persons, products) using SpaCy NER
2. Normalizing entity names for consistent matching (e.g., "amazon.com" -> "Amazon")
3. Applying regex patterns for common invoice/receipt vendors
4. Using fuzzy matching for entity deduplication

Used in:
- Document ingestion (metadata_extractor.py)
- Structured data extraction (extraction_service.py)
- Document routing (document_router.py)
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from functools import lru_cache

logger = logging.getLogger(__name__)

# Configuration
SPACY_MODEL = os.getenv("SPACY_NER_MODEL", "en_core_web_sm")
FUZZY_MATCH_THRESHOLD = int(os.getenv("ENTITY_FUZZY_THRESHOLD", "85"))

# Lazy load SpaCy to avoid import overhead when not needed
_nlp = None
_spacy_available = None


def _get_spacy_nlp():
    """Lazily load SpaCy NLP model."""
    global _nlp, _spacy_available

    if _spacy_available is False:
        return None

    if _nlp is not None:
        return _nlp

    try:
        import spacy
        _nlp = spacy.load(SPACY_MODEL)
        _spacy_available = True
        logger.info(f"[EntityExtractor] Loaded SpaCy model: {SPACY_MODEL}")
        return _nlp
    except ImportError:
        logger.warning("[EntityExtractor] SpaCy not installed. Install with: pip install spacy && python -m spacy download en_core_web_sm")
        _spacy_available = False
        return None
    except OSError:
        logger.warning(f"[EntityExtractor] SpaCy model '{SPACY_MODEL}' not found. Download with: python -m spacy download {SPACY_MODEL}")
        _spacy_available = False
        return None


def _get_fuzzy_matcher():
    """Get fuzzy matching function (rapidfuzz preferred, fallback to fuzzywuzzy)."""
    try:
        from rapidfuzz import fuzz
        return fuzz.ratio
    except ImportError:
        try:
            from fuzzywuzzy import fuzz
            return fuzz.ratio
        except ImportError:
            logger.debug("[EntityExtractor] No fuzzy matching library available")
            return None


@dataclass
class ExtractedEntity:
    """Represents an extracted entity with metadata."""
    name: str
    normalized_name: str
    entity_type: str  # "organization", "person", "product", "location"
    confidence: float = 1.0
    source: str = "spacy"  # "spacy", "regex", "header_data", "llm"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "normalized_name": self.normalized_name,
            "type": self.entity_type,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass
class EntityExtractionResult:
    """Result of entity extraction from a document."""
    organizations: List[ExtractedEntity] = field(default_factory=list)
    persons: List[ExtractedEntity] = field(default_factory=list)
    products: List[ExtractedEntity] = field(default_factory=list)
    locations: List[ExtractedEntity] = field(default_factory=list)

    # Normalized versions for quick lookup
    organizations_normalized: List[str] = field(default_factory=list)
    persons_normalized: List[str] = field(default_factory=list)

    # Primary entities (most likely vendor/customer)
    primary_vendor: Optional[ExtractedEntity] = None
    primary_customer: Optional[ExtractedEntity] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "organizations": [e.to_dict() for e in self.organizations],
            "persons": [e.to_dict() for e in self.persons],
            "products": [e.to_dict() for e in self.products],
            "locations": [e.to_dict() for e in self.locations],
            "organizations_normalized": self.organizations_normalized,
            "persons_normalized": self.persons_normalized,
            "primary_vendor": self.primary_vendor.to_dict() if self.primary_vendor else None,
            "primary_customer": self.primary_customer.to_dict() if self.primary_customer else None,
        }

    def get_all_normalized_entities(self) -> List[str]:
        """Get all normalized entity names for search/matching."""
        return self.organizations_normalized + self.persons_normalized


# ============================================================================
# ENTITY NORMALIZATION
# ============================================================================

# Common company suffixes to remove for normalization
COMPANY_SUFFIXES = [
    r'\s+inc\.?$', r'\s+incorporated$', r'\s+corp\.?$', r'\s+corporation$',
    r'\s+llc\.?$', r'\s+ltd\.?$', r'\s+limited$', r'\s+co\.?$', r'\s+company$',
    r'\s+plc\.?$', r'\s+gmbh$', r'\s+ag$', r'\s+sa$', r'\s+srl$',
    r'\s+pty\.?\s*ltd\.?$', r'\s+pte\.?\s*ltd\.?$',
]

# Known vendor aliases for normalization
VENDOR_ALIASES: Dict[str, str] = {
    # Amazon variants
    "amazon.com": "Amazon",
    "amazon.ca": "Amazon",
    "amazon.co.uk": "Amazon",
    "amzn": "Amazon",
    "amazon web services": "AWS",
    "aws": "AWS",

    # Google variants
    "google llc": "Google",
    "google inc": "Google",
    "google cloud": "Google Cloud",
    "gcp": "Google Cloud",

    # Microsoft variants
    "microsoft corporation": "Microsoft",
    "microsoft corp": "Microsoft",
    "msft": "Microsoft",
    "azure": "Microsoft Azure",

    # Apple variants
    "apple inc": "Apple",
    "apple computer": "Apple",

    # Common services
    "github inc": "GitHub",
    "github": "GitHub",
    "slack technologies": "Slack",
    "zoom video communications": "Zoom",
    "dropbox inc": "Dropbox",
    "atlassian": "Atlassian",
    "jira": "Atlassian",
    "confluence": "Atlassian",

    # Payment processors
    "stripe inc": "Stripe",
    "paypal inc": "PayPal",
    "paypal": "PayPal",
    "square inc": "Square",

    # Shipping
    "fedex": "FedEx",
    "federal express": "FedEx",
    "ups": "UPS",
    "united parcel service": "UPS",
    "usps": "USPS",
    "dhl": "DHL",
}


def normalize_entity_name(name: str, entity_type: str = "organization") -> str:
    """
    Normalize an entity name for consistent matching.

    Args:
        name: Original entity name
        entity_type: Type of entity ("organization", "person", etc.)

    Returns:
        Normalized lowercase name with common suffixes removed
    """
    if not name:
        return ""

    # Basic normalization
    normalized = name.strip().lower()

    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized)

    # Check known aliases first
    if normalized in VENDOR_ALIASES:
        return VENDOR_ALIASES[normalized].lower()

    # Remove company suffixes for organizations
    if entity_type == "organization":
        for suffix_pattern in COMPANY_SUFFIXES:
            normalized = re.sub(suffix_pattern, '', normalized, flags=re.IGNORECASE)

    # Remove common punctuation
    normalized = re.sub(r'[,\.\-\'\"]+$', '', normalized)

    # Check aliases again after suffix removal
    if normalized in VENDOR_ALIASES:
        return VENDOR_ALIASES[normalized].lower()

    return normalized.strip()


def are_entities_similar(name1: str, name2: str, threshold: int = None) -> bool:
    """
    Check if two entity names are similar using fuzzy matching.

    Args:
        name1: First entity name
        name2: Second entity name
        threshold: Similarity threshold (0-100), defaults to FUZZY_MATCH_THRESHOLD

    Returns:
        True if entities are considered similar
    """
    if not name1 or not name2:
        return False

    threshold = threshold or FUZZY_MATCH_THRESHOLD

    # Normalize both names
    norm1 = normalize_entity_name(name1)
    norm2 = normalize_entity_name(name2)

    # Exact match after normalization
    if norm1 == norm2:
        return True

    # Substring match
    if norm1 in norm2 or norm2 in norm1:
        return True

    # Fuzzy match
    fuzzy_ratio = _get_fuzzy_matcher()
    if fuzzy_ratio:
        score = fuzzy_ratio(norm1, norm2)
        return score >= threshold

    return False


# ============================================================================
# REGEX-BASED EXTRACTION (for invoices/receipts)
# ============================================================================

# Patterns for extracting vendor/seller from invoice text
VENDOR_PATTERNS = [
    # "From: Company Name" or "Seller: Company Name"
    r'(?:from|seller|vendor|supplier|billed\s+by|sold\s+by|merchant)[\s:]+([A-Z][A-Za-z0-9\s\.,&\'-]+?)(?:\n|$|,)',
    # "Company Name Invoice" at start of document
    r'^([A-Z][A-Za-z0-9\s\.,&\'-]{2,50})\s+(?:invoice|receipt|statement|bill)',
    # Email domain as fallback (e.g., noreply@amazon.com)
    r'@([a-z0-9\-]+)\.(?:com|org|net|io|co)',
]

# Patterns for extracting customer/buyer
CUSTOMER_PATTERNS = [
    # "To: Customer Name" or "Bill To: Customer Name"
    r'(?:to|customer|buyer|bill\s+to|ship\s+to|sold\s+to|client)[\s:]+([A-Z][A-Za-z0-9\s\.,&\'-]+?)(?:\n|$|,)',
    # "Dear Customer Name,"
    r'dear\s+([A-Z][A-Za-z\s]+?)[,\n]',
]


def extract_entities_with_regex(text: str) -> EntityExtractionResult:
    """
    Extract entities using regex patterns.

    Best for structured documents like invoices where format is predictable.

    Args:
        text: Document text content

    Returns:
        EntityExtractionResult with extracted entities
    """
    result = EntityExtractionResult()

    if not text:
        return result

    # Extract vendors
    for pattern in VENDOR_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            name = match.strip()
            if len(name) >= 2 and len(name) <= 100:
                normalized = normalize_entity_name(name, "organization")
                if normalized and normalized not in result.organizations_normalized:
                    entity = ExtractedEntity(
                        name=name,
                        normalized_name=normalized,
                        entity_type="organization",
                        confidence=0.7,
                        source="regex"
                    )
                    result.organizations.append(entity)
                    result.organizations_normalized.append(normalized)

    # Extract customers
    for pattern in CUSTOMER_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        for match in matches:
            name = match.strip()
            if len(name) >= 2 and len(name) <= 100:
                normalized = normalize_entity_name(name, "person")
                if normalized and normalized not in result.persons_normalized:
                    entity = ExtractedEntity(
                        name=name,
                        normalized_name=normalized,
                        entity_type="person",
                        confidence=0.7,
                        source="regex"
                    )
                    result.persons.append(entity)
                    result.persons_normalized.append(normalized)

    return result


# ============================================================================
# SPACY NER EXTRACTION
# ============================================================================

# SpaCy entity label mappings
SPACY_LABEL_MAP = {
    "ORG": "organization",
    "COMPANY": "organization",
    "GPE": "location",
    "LOC": "location",
    "PERSON": "person",
    "PER": "person",
    "PRODUCT": "product",
    "WORK_OF_ART": "product",
}


def extract_entities_with_spacy(text: str, max_length: int = 100000) -> EntityExtractionResult:
    """
    Extract entities using SpaCy NER.

    Args:
        text: Document text content
        max_length: Maximum text length to process (SpaCy has limits)

    Returns:
        EntityExtractionResult with extracted entities
    """
    result = EntityExtractionResult()

    nlp = _get_spacy_nlp()
    if not nlp or not text:
        return result

    # Truncate very long texts
    if len(text) > max_length:
        text = text[:max_length]
        logger.debug(f"[EntityExtractor] Truncated text to {max_length} chars for SpaCy")

    try:
        doc = nlp(text)

        seen_normalized: Set[str] = set()

        for ent in doc.ents:
            entity_type = SPACY_LABEL_MAP.get(ent.label_)
            if not entity_type:
                continue

            name = ent.text.strip()
            if len(name) < 2 or len(name) > 100:
                continue

            normalized = normalize_entity_name(name, entity_type)
            if not normalized or normalized in seen_normalized:
                continue

            seen_normalized.add(normalized)

            entity = ExtractedEntity(
                name=name,
                normalized_name=normalized,
                entity_type=entity_type,
                confidence=0.85,
                source="spacy"
            )

            if entity_type == "organization":
                result.organizations.append(entity)
                result.organizations_normalized.append(normalized)
            elif entity_type == "person":
                result.persons.append(entity)
                result.persons_normalized.append(normalized)
            elif entity_type == "product":
                result.products.append(entity)
            elif entity_type == "location":
                result.locations.append(entity)

        logger.debug(
            f"[EntityExtractor] SpaCy extracted: {len(result.organizations)} orgs, "
            f"{len(result.persons)} persons, {len(result.products)} products"
        )

    except Exception as e:
        logger.error(f"[EntityExtractor] SpaCy extraction failed: {e}")

    return result


# ============================================================================
# HEADER DATA EXTRACTION
# ============================================================================

def extract_entities_from_header_data(header_data: Dict[str, Any]) -> EntityExtractionResult:
    """
    Extract entities from structured header_data (from extraction service).

    This is the most reliable source as it comes from structured extraction.

    Args:
        header_data: Structured data with fields like vendor_name, customer_name

    Returns:
        EntityExtractionResult with extracted entities
    """
    result = EntityExtractionResult()

    if not header_data:
        return result

    # Vendor/seller fields
    vendor_fields = ['vendor_name', 'seller_name', 'merchant_name', 'supplier_name', 'company_name', 'store_name']
    for field in vendor_fields:
        value = header_data.get(field)
        if value and isinstance(value, str) and len(value.strip()) >= 2:
            name = value.strip()
            normalized = normalize_entity_name(name, "organization")
            if normalized and normalized not in result.organizations_normalized:
                entity = ExtractedEntity(
                    name=name,
                    normalized_name=normalized,
                    entity_type="organization",
                    confidence=0.95,  # High confidence from structured data
                    source="header_data"
                )
                result.organizations.append(entity)
                result.organizations_normalized.append(normalized)

                # First vendor found is primary vendor
                if result.primary_vendor is None:
                    result.primary_vendor = entity

    # Customer/buyer fields
    customer_fields = ['customer_name', 'buyer_name', 'client_name', 'bill_to', 'ship_to']
    for field in customer_fields:
        value = header_data.get(field)
        if value and isinstance(value, str) and len(value.strip()) >= 2:
            name = value.strip()
            # Could be person or organization
            normalized = normalize_entity_name(name, "person")
            if normalized and normalized not in result.persons_normalized:
                entity = ExtractedEntity(
                    name=name,
                    normalized_name=normalized,
                    entity_type="person",  # Default to person, can be org too
                    confidence=0.95,
                    source="header_data"
                )
                result.persons.append(entity)
                result.persons_normalized.append(normalized)

                # First customer found is primary customer
                if result.primary_customer is None:
                    result.primary_customer = entity

    return result


# ============================================================================
# COMBINED EXTRACTION
# ============================================================================

def _deduplicate_entities_optimized(
    entities: List[ExtractedEntity],
    entity_type: str = "organization"
) -> Tuple[List[ExtractedEntity], List[str]]:
    """
    Deduplicate entities using exact-match-first strategy.

    This is an O(n + groups²) optimization over naive O(n²) fuzzy matching:
    1. Group entities by exact normalized name (O(n))
    2. Only fuzzy match between different groups (O(groups²))

    For documents with many repeated entities, this significantly reduces comparisons.

    Args:
        entities: List of extracted entities (may have duplicates)
        entity_type: Type for logging ("organization" or "person")

    Returns:
        Tuple of (deduplicated entities, normalized names list)
    """
    if not entities:
        return [], []

    # Step 1: Group by exact normalized name (O(n))
    groups: Dict[str, List[ExtractedEntity]] = {}
    for entity in entities:
        key = entity.normalized_name.lower()
        if key not in groups:
            groups[key] = []
        groups[key].append(entity)

    # Step 2: For each group, keep the highest confidence entity
    representatives: List[ExtractedEntity] = []
    for key, group in groups.items():
        # Sort by confidence and take the best one
        group.sort(key=lambda e: e.confidence, reverse=True)
        representatives.append(group[0])

    # Step 3: Fuzzy match between groups only if we have fuzzy matcher
    # This is O(groups²) which is much smaller than O(n²)
    fuzzy_ratio = _get_fuzzy_matcher()
    if fuzzy_ratio and len(representatives) > 1:
        # Track which entities to merge
        merged_indices: Set[int] = set()
        final_entities: List[ExtractedEntity] = []

        for i, entity_i in enumerate(representatives):
            if i in merged_indices:
                continue

            # Check for fuzzy matches with remaining entities
            for j in range(i + 1, len(representatives)):
                if j in merged_indices:
                    continue

                entity_j = representatives[j]
                score = fuzzy_ratio(entity_i.normalized_name, entity_j.normalized_name)

                if score >= FUZZY_MATCH_THRESHOLD:
                    # Merge j into i (keep higher confidence)
                    merged_indices.add(j)
                    if entity_j.confidence > entity_i.confidence:
                        entity_i = entity_j

            final_entities.append(entity_i)

        representatives = final_entities

    # Build normalized names list
    normalized_names = [e.normalized_name for e in representatives]

    # Log optimization stats
    original_count = len(entities)
    final_count = len(representatives)
    if original_count > final_count:
        logger.debug(
            f"[EntityExtractor] Dedup {entity_type}: {original_count} → {final_count} "
            f"({len(groups)} unique groups)"
        )

    return representatives, normalized_names


def extract_all_entities(
    text: str = None,
    header_data: Dict[str, Any] = None,
    use_spacy: bool = True,
    use_regex: bool = True,
) -> EntityExtractionResult:
    """
    Extract entities using all available methods and merge results.

    Priority (highest to lowest confidence):
    1. header_data (structured extraction) - 0.95 confidence
    2. SpaCy NER - 0.85 confidence
    3. Regex patterns - 0.70 confidence

    OPTIMIZATION: Uses exact-match-first deduplication to reduce O(n²) to O(groups²)

    Args:
        text: Document text content
        header_data: Structured header data from extraction
        use_spacy: Whether to use SpaCy NER
        use_regex: Whether to use regex patterns

    Returns:
        Merged EntityExtractionResult with deduplicated entities
    """
    # Collect all entities from all sources first
    all_orgs: List[ExtractedEntity] = []
    all_persons: List[ExtractedEntity] = []
    all_products: List[ExtractedEntity] = []
    all_locations: List[ExtractedEntity] = []
    primary_vendor = None
    primary_customer = None

    def collect_entities(source_result: EntityExtractionResult):
        """Collect entities from source for batch deduplication."""
        nonlocal primary_vendor, primary_customer

        all_orgs.extend(source_result.organizations)
        all_persons.extend(source_result.persons)
        all_products.extend(source_result.products)
        all_locations.extend(source_result.locations)

        # Set primary vendor/customer from first source that has them
        if primary_vendor is None and source_result.primary_vendor:
            primary_vendor = source_result.primary_vendor
        if primary_customer is None and source_result.primary_customer:
            primary_customer = source_result.primary_customer

    # Priority 1: Header data (highest confidence)
    if header_data:
        header_result = extract_entities_from_header_data(header_data)
        collect_entities(header_result)
        logger.debug(f"[EntityExtractor] Header data: {len(header_result.organizations)} orgs")

    # Priority 2: SpaCy NER
    if use_spacy and text:
        spacy_result = extract_entities_with_spacy(text)
        collect_entities(spacy_result)

    # Priority 3: Regex patterns
    if use_regex and text:
        regex_result = extract_entities_with_regex(text)
        collect_entities(regex_result)

    # OPTIMIZATION: Batch deduplicate using exact-match-first strategy
    # This is O(n + groups²) instead of O(n²)
    result = EntityExtractionResult()
    result.organizations, result.organizations_normalized = _deduplicate_entities_optimized(
        all_orgs, "organization"
    )
    result.persons, result.persons_normalized = _deduplicate_entities_optimized(
        all_persons, "person"
    )
    result.products = all_products  # Products typically don't need dedup
    result.locations = all_locations  # Locations typically don't need dedup

    # Sort by confidence (highest first)
    result.organizations.sort(key=lambda e: e.confidence, reverse=True)
    result.persons.sort(key=lambda e: e.confidence, reverse=True)

    # Set primary vendor/customer
    result.primary_vendor = primary_vendor
    result.primary_customer = primary_customer

    # Infer primary vendor/customer if not set
    if result.primary_vendor is None and result.organizations:
        result.primary_vendor = result.organizations[0]
    if result.primary_customer is None and result.persons:
        result.primary_customer = result.persons[0]

    logger.info(
        f"[EntityExtractor] Combined extraction: {len(result.organizations)} orgs, "
        f"{len(result.persons)} persons, "
        f"vendor={result.primary_vendor.name if result.primary_vendor else 'None'}, "
        f"customer={result.primary_customer.name if result.primary_customer else 'None'}"
    )

    return result


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_normalized_metadata(
    extraction_result: EntityExtractionResult
) -> Dict[str, Any]:
    """
    Create normalized metadata dict for storage in Qdrant/PostgreSQL.

    Returns dict with:
    - vendor_normalized: Normalized primary vendor name
    - customer_normalized: Normalized primary customer name
    - all_entities_normalized: List of all normalized entity names
    - entities_detail: Full extraction result
    """
    return {
        "vendor_normalized": extraction_result.primary_vendor.normalized_name if extraction_result.primary_vendor else None,
        "customer_normalized": extraction_result.primary_customer.normalized_name if extraction_result.primary_customer else None,
        "vendor_name": extraction_result.primary_vendor.name if extraction_result.primary_vendor else None,
        "customer_name": extraction_result.primary_customer.name if extraction_result.primary_customer else None,
        "all_entities_normalized": extraction_result.get_all_normalized_entities(),
        "organizations_normalized": extraction_result.organizations_normalized,
        "persons_normalized": extraction_result.persons_normalized,
    }


def entity_matches_query(
    entity_name: str,
    query_entities: List[str],
    threshold: int = None
) -> Tuple[bool, float]:
    """
    Check if an entity name matches any query entities.

    Args:
        entity_name: Entity name to check
        query_entities: List of query entity names
        threshold: Fuzzy match threshold

    Returns:
        Tuple of (matches: bool, best_score: float)
    """
    if not entity_name or not query_entities:
        return False, 0.0

    normalized = normalize_entity_name(entity_name)
    best_score = 0.0

    for query_entity in query_entities:
        query_normalized = normalize_entity_name(query_entity)

        # Exact match
        if normalized == query_normalized:
            return True, 1.0

        # Substring match
        if query_normalized in normalized or normalized in query_normalized:
            return True, 0.9

        # Fuzzy match
        fuzzy_ratio = _get_fuzzy_matcher()
        if fuzzy_ratio:
            score = fuzzy_ratio(normalized, query_normalized) / 100.0
            best_score = max(best_score, score)
            if score >= (threshold or FUZZY_MATCH_THRESHOLD) / 100.0:
                return True, score

    return False, best_score
