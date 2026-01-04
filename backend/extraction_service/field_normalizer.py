"""
Field Normalizer Module

Normalizes extracted field names to canonical schema names using alias matching
and LLM-based semantic mapping as fallback.

This ensures consistent field naming across all documents regardless of how
the original data named the fields (e.g., "Item" → "description", "Qty" → "quantity").

Supports grouped field mappings:
- header_mappings: Document-level fields (vendor_name, invoice_date, etc.)
- line_item_mappings: Row-level fields (description, quantity, amount, etc.)
- summary_mappings: Aggregate fields (subtotal, tax_amount, total_amount, etc.)

Includes bilingual/label detection to prevent false matches on fields like
"Item subtotal /Sous-total del'article" which contain "/" separators.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)


# LLM prompt for semantic field mapping
FIELD_MAPPING_PROMPT = """You are a field name mapper. Map extracted field names to canonical schema field names.

## Canonical Schema Fields (target names with their aliases):
{canonical_fields_json}

## Extracted Fields to Map:
{extracted_fields}

## Rules:
1. Map each extracted field to the most appropriate canonical field name
2. Use semantic meaning - "Item" should map to "description", "Qty" to "quantity", etc.
3. If a field doesn't match any canonical field, use null
4. Preserve "row_number" as-is (it's a system field)
5. Be case-insensitive when matching
6. IMPORTANT: Fields containing "/" are likely bilingual labels - map to "description" or null

Return ONLY a JSON object mapping extracted names to canonical names:
{{"ExtractedName1": "canonical_name1", "ExtractedName2": "canonical_name2", ...}}

Example:
Input fields: ["Item", "Qty", "Amount", "row_number"]
Schema has: description (aliases: Item, Product), quantity (aliases: Qty), amount (aliases: Amount, Total)
Output: {{"Item": "description", "Qty": "quantity", "Amount": "amount", "row_number": "row_number"}}
"""


# =============================================================================
# DEFAULT FIELD MAPPINGS BY SCHEMA TYPE
# =============================================================================
# These are used when a schema doesn't have explicit field_mappings defined.
# Each mapping includes:
#   - canonical: The standard field name to normalize to
#   - data_type: Expected data type (string, number, datetime)
#   - patterns: Keywords to match in field names (case-insensitive)
#   - exclude_patterns: Keywords that indicate this is NOT the field (e.g., "/" for bilingual labels)
#   - required: Whether field is required (optional, default False)

DEFAULT_INVOICE_MAPPINGS = {
    "header_mappings": [
        {"canonical": "vendor_name", "data_type": "string", "patterns": ["vendor", "supplier", "seller", "from", "fournisseur", "lieferant"], "exclude_patterns": ["/"]},
        {"canonical": "customer_name", "data_type": "string", "patterns": ["customer", "client", "buyer", "bill to", "sold to", "client"], "exclude_patterns": ["/"]},
        {"canonical": "vendor_address", "data_type": "string", "patterns": ["vendor address", "seller address", "from address"], "exclude_patterns": ["/"]},
        {"canonical": "customer_address", "data_type": "string", "patterns": ["customer address", "bill to address", "billing address"], "exclude_patterns": ["/"]},
        {"canonical": "invoice_number", "data_type": "string", "patterns": ["invoice #", "invoice no", "invoice number", "inv #", "facture no", "rechnungsnummer"]},
        {"canonical": "invoice_date", "data_type": "datetime", "patterns": ["invoice date", "date", "dated", "issue date"]},
        {"canonical": "due_date", "data_type": "datetime", "patterns": ["due date", "payment due", "due by", "échéance"]},
        {"canonical": "currency", "data_type": "string", "patterns": ["currency", "devise", "währung"]},
        {"canonical": "payment_terms", "data_type": "string", "patterns": ["terms", "payment terms", "net 30", "net 60"]},
        {"canonical": "po_number", "data_type": "string", "patterns": ["po #", "po number", "purchase order", "order #"]},
    ],
    "line_item_mappings": [
        {"canonical": "description", "data_type": "string", "patterns": ["description", "item", "product", "service", "article", "désignation", "beschreibung", "name"], "exclude_patterns": ["/"]},
        {"canonical": "quantity", "data_type": "number", "patterns": ["qty", "quantity", "units", "quantité", "menge", "count"], "exclude_patterns": ["/"]},
        {"canonical": "unit_price", "data_type": "number", "patterns": ["unit price", "price", "rate", "prix unitaire", "einzelpreis", "each"], "exclude_patterns": ["/"]},
        {"canonical": "amount", "data_type": "number", "patterns": ["amount", "total", "extended", "line total", "montant", "betrag"], "exclude_patterns": ["/", "subtotal", "tax", "grand total", "total amount"]},
        {"canonical": "sku", "data_type": "string", "patterns": ["sku", "item code", "product code", "part #", "article #", "code"]},
        {"canonical": "category", "data_type": "string", "patterns": ["category", "type", "catégorie", "kategorie"]},
    ],
    "summary_mappings": [
        {"canonical": "subtotal", "data_type": "number", "patterns": ["subtotal", "sub-total", "sous-total", "net amount", "zwischensumme"], "exclude_patterns": ["/"]},
        {"canonical": "tax_amount", "data_type": "number", "patterns": ["tax", "vat", "gst", "hst", "pst", "sales tax", "tva", "taxe", "mwst"], "exclude_patterns": ["/"]},
        {"canonical": "discount_amount", "data_type": "number", "patterns": ["discount", "reduction", "rabais", "remise", "rabatt"], "exclude_patterns": ["/"]},
        {"canonical": "shipping_amount", "data_type": "number", "patterns": ["shipping", "freight", "delivery", "livraison", "versand"], "exclude_patterns": ["/"]},
        {"canonical": "total_amount", "data_type": "number", "patterns": ["total", "grand total", "amount due", "balance due", "total à payer", "gesamtbetrag"], "exclude_patterns": ["/"]},
    ]
}

DEFAULT_RECEIPT_MAPPINGS = {
    "header_mappings": [
        {"canonical": "store_name", "data_type": "string", "patterns": ["store", "merchant", "restaurant", "shop", "retailer", "magasin"], "exclude_patterns": ["/"]},
        {"canonical": "store_address", "data_type": "string", "patterns": ["address", "location", "store address"], "exclude_patterns": ["/"]},
        {"canonical": "transaction_date", "data_type": "datetime", "patterns": ["date", "transaction date", "purchase date", "time"]},
        {"canonical": "receipt_number", "data_type": "string", "patterns": ["receipt #", "transaction #", "order #", "ticket #", "check #"]},
        {"canonical": "payment_method", "data_type": "string", "patterns": ["payment", "paid by", "card", "cash", "credit", "debit"]},
        {"canonical": "cashier", "data_type": "string", "patterns": ["cashier", "server", "employee", "served by"]},
    ],
    "line_item_mappings": [
        {"canonical": "description", "data_type": "string", "patterns": ["description", "item", "product", "article", "name"], "exclude_patterns": ["/"]},
        {"canonical": "quantity", "data_type": "number", "patterns": ["qty", "quantity", "x", "units"], "exclude_patterns": ["/"]},
        {"canonical": "unit_price", "data_type": "number", "patterns": ["unit price", "each", "@", "price"], "exclude_patterns": ["/"]},
        {"canonical": "amount", "data_type": "number", "patterns": ["amount", "price", "total", "cost"], "exclude_patterns": ["/", "subtotal", "tax", "grand total", "tip"]},
    ],
    "summary_mappings": [
        {"canonical": "subtotal", "data_type": "number", "patterns": ["subtotal", "sub-total", "food total", "items total"], "exclude_patterns": ["/"]},
        {"canonical": "tax_amount", "data_type": "number", "patterns": ["tax", "hst", "gst", "pst", "sales tax", "vat"], "exclude_patterns": ["/"]},
        {"canonical": "tip_amount", "data_type": "number", "patterns": ["tip", "gratuity", "service charge", "pourboire"], "exclude_patterns": ["/"]},
        {"canonical": "total_amount", "data_type": "number", "patterns": ["total", "grand total", "amount", "balance"], "exclude_patterns": ["/"]},
    ]
}

DEFAULT_SPREADSHEET_MAPPINGS = {
    "header_mappings": [
        {"canonical": "title", "data_type": "string", "patterns": ["title", "name", "sheet name"]},
        {"canonical": "date", "data_type": "datetime", "patterns": ["date", "created", "modified"]},
    ],
    "line_item_mappings": [
        # Spreadsheets have dynamic columns, so we use generic patterns
        {"canonical": "description", "data_type": "string", "patterns": ["description", "name", "item", "product", "title"], "exclude_patterns": ["/"]},
        {"canonical": "quantity", "data_type": "number", "patterns": ["qty", "quantity", "count", "units", "stock"], "exclude_patterns": ["/"]},
        {"canonical": "amount", "data_type": "number", "patterns": ["amount", "total", "value", "price", "cost", "sales"], "exclude_patterns": ["/", "subtotal"]},
        {"canonical": "date", "data_type": "datetime", "patterns": ["date", "time", "created", "updated"]},
        {"canonical": "category", "data_type": "string", "patterns": ["category", "type", "group", "class"]},
        {"canonical": "status", "data_type": "string", "patterns": ["status", "state", "condition"]},
    ],
    "summary_mappings": [
        {"canonical": "total_amount", "data_type": "number", "patterns": ["total", "sum", "grand total"], "exclude_patterns": ["/"]},
    ]
}

DEFAULT_BANK_STATEMENT_MAPPINGS = {
    "header_mappings": [
        {"canonical": "bank_name", "data_type": "string", "patterns": ["bank", "institution", "financial institution"]},
        {"canonical": "account_number", "data_type": "string", "patterns": ["account", "account #", "account number"]},
        {"canonical": "statement_date", "data_type": "datetime", "patterns": ["statement date", "date", "period end"]},
        {"canonical": "period_start", "data_type": "datetime", "patterns": ["period start", "from", "start date"]},
        {"canonical": "period_end", "data_type": "datetime", "patterns": ["period end", "to", "end date"]},
    ],
    "line_item_mappings": [
        {"canonical": "transaction_date", "data_type": "datetime", "patterns": ["date", "transaction date", "post date"]},
        {"canonical": "description", "data_type": "string", "patterns": ["description", "transaction", "details", "memo"], "exclude_patterns": ["/"]},
        {"canonical": "debit", "data_type": "number", "patterns": ["debit", "withdrawal", "payment", "out"], "exclude_patterns": ["/"]},
        {"canonical": "credit", "data_type": "number", "patterns": ["credit", "deposit", "in"], "exclude_patterns": ["/"]},
        {"canonical": "balance", "data_type": "number", "patterns": ["balance", "running balance"], "exclude_patterns": ["/"]},
        {"canonical": "reference", "data_type": "string", "patterns": ["reference", "ref #", "check #", "confirmation"]},
    ],
    "summary_mappings": [
        {"canonical": "opening_balance", "data_type": "number", "patterns": ["opening balance", "beginning balance", "previous balance"], "exclude_patterns": ["/"]},
        {"canonical": "total_debits", "data_type": "number", "patterns": ["total debits", "total withdrawals", "total payments"], "exclude_patterns": ["/"]},
        {"canonical": "total_credits", "data_type": "number", "patterns": ["total credits", "total deposits"], "exclude_patterns": ["/"]},
        {"canonical": "closing_balance", "data_type": "number", "patterns": ["closing balance", "ending balance", "current balance"], "exclude_patterns": ["/"]},
    ]
}

# Mapping of schema types to their default field mappings
DEFAULT_FIELD_MAPPINGS_BY_SCHEMA = {
    "invoice": DEFAULT_INVOICE_MAPPINGS,
    "receipt": DEFAULT_RECEIPT_MAPPINGS,
    "spreadsheet": DEFAULT_SPREADSHEET_MAPPINGS,
    "bank_statement": DEFAULT_BANK_STATEMENT_MAPPINGS,
}


class FieldNormalizer:
    """
    Normalizes extracted field names to canonical schema names.

    Uses a multi-tier approach:
    1. Bilingual/label detection (skip fields that are labels, not data)
    2. Pattern matching with exclude patterns (fast, keyword-based)
    3. Alias matching (exact/case-insensitive)
    4. LLM semantic mapping (fallback for unmatched fields)

    Supports grouped field mappings:
    - header_mappings: Document-level fields
    - line_item_mappings: Row-level fields
    - summary_mappings: Aggregate fields
    """

    # System fields to preserve as-is
    SYSTEM_FIELDS = {'row_number', 'line_number', 'id', '_id'}

    # Indicators that a field name is a bilingual label, not a data field
    LABEL_INDICATORS = ['/']  # Language separator: "English / French"
    MAX_FIELD_NAME_LENGTH = 50  # Fields longer than this are likely labels
    MAX_WORD_COUNT = 5  # Fields with more words are likely labels

    def __init__(self, llm_client=None):
        """
        Initialize the field normalizer.

        Args:
            llm_client: Optional LLM client for semantic matching fallback
        """
        self.llm_client = llm_client

    def _is_bilingual_label(self, field_name: str) -> bool:
        """
        Detect if a field name is a bilingual label, not a data field.

        Bilingual labels like "Item subtotal /Sous-total del'article" should NOT
        be classified as numeric fields even though they contain keywords like "subtotal".

        Args:
            field_name: The field name to check

        Returns:
            True if this appears to be a bilingual label
        """
        # Check for language separators
        for indicator in self.LABEL_INDICATORS:
            if indicator in field_name:
                logger.debug(f"[FieldNormalizer] Bilingual label detected (contains '{indicator}'): {field_name}")
                return True

        # Check for excessively long field names
        if len(field_name) > self.MAX_FIELD_NAME_LENGTH:
            logger.debug(f"[FieldNormalizer] Bilingual label detected (too long): {field_name}")
            return True

        # Check for too many words
        if field_name.count(' ') > self.MAX_WORD_COUNT:
            logger.debug(f"[FieldNormalizer] Bilingual label detected (too many words): {field_name}")
            return True

        return False

    def get_default_mappings(self, schema_type: str) -> Dict[str, List[Dict]]:
        """
        Get default field mappings for a schema type.

        Args:
            schema_type: The schema type (invoice, receipt, spreadsheet, etc.)

        Returns:
            Dict with header_mappings, line_item_mappings, summary_mappings
        """
        # Try exact match first
        if schema_type in DEFAULT_FIELD_MAPPINGS_BY_SCHEMA:
            return DEFAULT_FIELD_MAPPINGS_BY_SCHEMA[schema_type]

        # Try to find a matching base type
        schema_lower = schema_type.lower()
        for base_type, mappings in DEFAULT_FIELD_MAPPINGS_BY_SCHEMA.items():
            if base_type in schema_lower or schema_lower in base_type:
                logger.info(f"[FieldNormalizer] Using {base_type} mappings for schema {schema_type}")
                return mappings

        # Default to invoice mappings (most common)
        logger.info(f"[FieldNormalizer] No specific mappings for {schema_type}, using invoice defaults")
        return DEFAULT_INVOICE_MAPPINGS

    def _matches_mapping_rule(self, field_name: str, mapping_rule: Dict) -> bool:
        """
        Check if a field name matches a mapping rule.

        Args:
            field_name: The field name to check
            mapping_rule: Dict with 'patterns' and optional 'exclude_patterns'

        Returns:
            True if field matches patterns and doesn't match exclude_patterns
        """
        field_lower = field_name.lower()

        # Check exclude patterns first - if any match, this is NOT the field
        exclude_patterns = mapping_rule.get('exclude_patterns', [])
        for exclude in exclude_patterns:
            if exclude.lower() in field_lower:
                return False

        # Check include patterns
        patterns = mapping_rule.get('patterns', [])
        for pattern in patterns:
            if pattern.lower() in field_lower:
                return True

        return False

    def normalize_line_items(
        self,
        line_items: List[Dict[str, Any]],
        schema_field_mappings: Dict[str, Dict],
        use_llm: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Normalize line item field names to canonical names.

        Args:
            line_items: Raw extracted line items with original field names
            schema_field_mappings: field_mappings['line_item_fields'] from data_schemas
            use_llm: Whether to use LLM for unmatched fields (default True)

        Returns:
            Tuple of:
                - normalized_line_items: List with canonical field names
                - mapping_used: Dict mapping original → canonical names
        """
        if not line_items:
            return [], {}

        if not schema_field_mappings:
            logger.warning("[FieldNormalizer] No schema field mappings provided, returning items unchanged")
            return line_items, {}

        # Get all unique field names from line items
        extracted_fields = set()
        for item in line_items:
            extracted_fields.update(item.keys())
        extracted_fields = list(extracted_fields)

        logger.info(f"[FieldNormalizer] Extracted fields: {extracted_fields}")
        logger.info(f"[FieldNormalizer] Canonical fields: {list(schema_field_mappings.keys())}")

        # Step 1: Build mapping using alias matching
        mapping = self._build_mapping_with_aliases(extracted_fields, schema_field_mappings)

        # Step 2: Check for unmatched fields
        unmatched = [f for f in extracted_fields if f not in mapping and f not in self.SYSTEM_FIELDS]

        # Step 3: Use LLM for unmatched fields if available
        if unmatched and use_llm and self.llm_client:
            logger.info(f"[FieldNormalizer] Using LLM for unmatched fields: {unmatched}")
            llm_mapping = self._build_mapping_with_llm(unmatched, schema_field_mappings)
            mapping.update(llm_mapping)

        # Log the final mapping
        logger.info(f"[FieldNormalizer] Final field mapping: {mapping}")

        # Step 4: Apply mapping to all line items
        normalized_items = []
        for item in line_items:
            normalized_item = {}
            for key, value in item.items():
                if key in self.SYSTEM_FIELDS:
                    # Preserve system fields
                    normalized_item[key] = value
                elif key in mapping and mapping[key]:
                    # Use canonical name
                    normalized_item[mapping[key]] = value
                else:
                    # Keep original if no mapping (shouldn't happen often)
                    normalized_item[key] = value
            normalized_items.append(normalized_item)

        return normalized_items, mapping

    def _build_mapping_with_aliases(
        self,
        extracted_fields: List[str],
        canonical_fields: Dict[str, Dict]
    ) -> Dict[str, str]:
        """
        Build field mapping using alias matching.

        Tries exact match first, then case-insensitive match.

        Args:
            extracted_fields: List of field names from extracted data
            canonical_fields: Dict of canonical field definitions with aliases

        Returns:
            Dict mapping extracted_name → canonical_name
        """
        mapping = {}

        # Build alias lookup: alias → canonical_name
        alias_lookup = {}  # exact match
        alias_lookup_lower = {}  # case-insensitive

        for canonical_name, field_def in canonical_fields.items():
            # Add the canonical name itself as an alias
            alias_lookup[canonical_name] = canonical_name
            alias_lookup_lower[canonical_name.lower()] = canonical_name

            # Add defined aliases
            aliases = field_def.get('aliases', [])
            if isinstance(aliases, list):
                for alias in aliases:
                    alias_lookup[alias] = canonical_name
                    alias_lookup_lower[alias.lower()] = canonical_name

        # Map extracted fields
        for field in extracted_fields:
            if field in self.SYSTEM_FIELDS:
                continue

            # Try exact match first
            if field in alias_lookup:
                mapping[field] = alias_lookup[field]
                logger.debug(f"[FieldNormalizer] Exact match: {field} → {alias_lookup[field]}")
            # Try case-insensitive match
            elif field.lower() in alias_lookup_lower:
                mapping[field] = alias_lookup_lower[field.lower()]
                logger.debug(f"[FieldNormalizer] Case-insensitive match: {field} → {alias_lookup_lower[field.lower()]}")

        return mapping

    def _build_mapping_with_llm(
        self,
        unmatched_fields: List[str],
        canonical_fields: Dict[str, Dict]
    ) -> Dict[str, str]:
        """
        Use LLM to map fields that couldn't be matched by aliases.

        Args:
            unmatched_fields: Fields that need LLM mapping
            canonical_fields: Dict of canonical field definitions

        Returns:
            Dict mapping unmatched_field → canonical_name (or None)
        """
        if not self.llm_client:
            return {}

        try:
            # Format canonical fields for prompt
            canonical_for_prompt = {}
            for name, field_def in canonical_fields.items():
                canonical_for_prompt[name] = {
                    'semantic_type': field_def.get('semantic_type', 'unknown'),
                    'aliases': field_def.get('aliases', [])
                }

            prompt = FIELD_MAPPING_PROMPT.format(
                canonical_fields_json=json.dumps(canonical_for_prompt, indent=2),
                extracted_fields=json.dumps(unmatched_fields)
            )

            response = self.llm_client.generate(prompt)

            # Parse JSON response
            mapping = self._parse_json_response(response)
            if mapping:
                # Validate that mapped values are actual canonical fields
                valid_mapping = {}
                for orig, canonical in mapping.items():
                    if canonical in canonical_fields or canonical is None:
                        valid_mapping[orig] = canonical
                    else:
                        logger.warning(f"[FieldNormalizer] LLM suggested invalid canonical field: {canonical}")
                return valid_mapping

        except Exception as e:
            logger.error(f"[FieldNormalizer] LLM mapping failed: {e}")

        return {}

    def _parse_json_response(self, response: str) -> Optional[Dict[str, str]]:
        """Parse JSON from LLM response."""
        try:
            # Try to extract JSON from response
            text = response.strip()

            # Handle markdown code blocks
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                text = text.split('```')[1].split('```')[0]

            text = text.strip()

            # Find JSON object boundaries
            if not text.startswith('{'):
                start = text.find('{')
                end = text.rfind('}') + 1
                if start != -1 and end > start:
                    text = text[start:end]

            return json.loads(text)

        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"[FieldNormalizer] Failed to parse LLM response: {e}")
            return None

    def normalize_header_data(
        self,
        header_data: Dict[str, Any],
        schema_field_mappings: Dict[str, Dict],
        use_llm: bool = True
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Normalize header data field names to canonical names.

        Args:
            header_data: Raw header data with original field names
            schema_field_mappings: field_mappings['header_fields'] from data_schemas
            use_llm: Whether to use LLM for unmatched fields

        Returns:
            Tuple of:
                - normalized_header: Dict with canonical field names
                - mapping_used: Dict mapping original → canonical names
        """
        if not header_data or not schema_field_mappings:
            return header_data or {}, {}

        # Extract field names
        extracted_fields = list(header_data.keys())

        # Build mapping
        mapping = self._build_mapping_with_aliases(extracted_fields, schema_field_mappings)

        # Check for unmatched
        unmatched = [f for f in extracted_fields if f not in mapping and f not in self.SYSTEM_FIELDS]

        # Use LLM if needed
        if unmatched and use_llm and self.llm_client:
            llm_mapping = self._build_mapping_with_llm(unmatched, schema_field_mappings)
            mapping.update(llm_mapping)

        # Apply mapping
        normalized = {}
        for key, value in header_data.items():
            if key in mapping and mapping[key]:
                normalized[mapping[key]] = value
            else:
                normalized[key] = value

        return normalized, mapping

    # =========================================================================
    # NEW GROUPED NORMALIZATION METHODS
    # =========================================================================

    def normalize_with_grouped_mappings(
        self,
        header_data: Dict[str, Any],
        line_items: List[Dict[str, Any]],
        summary_data: Dict[str, Any],
        grouped_mappings: Dict[str, List[Dict]],
        schema_type: str = "invoice",
        use_llm: bool = True
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any], Dict[str, Dict[str, str]]]:
        """
        Normalize all extracted data using grouped field mappings.

        This is the main entry point for the new grouped normalization approach.
        It processes header_data, line_items, and summary_data separately using
        their respective mapping rules.

        Args:
            header_data: Raw header data with original field names
            line_items: Raw line items with original field names
            summary_data: Raw summary data with original field names
            grouped_mappings: Dict with 'header_mappings', 'line_item_mappings', 'summary_mappings'
            schema_type: Schema type for fallback to defaults
            use_llm: Whether to use LLM for unmatched fields

        Returns:
            Tuple of:
                - normalized_header: Dict with canonical field names
                - normalized_line_items: List with canonical field names
                - normalized_summary: Dict with canonical field names
                - all_mappings: Dict with mappings used for each group
        """
        # Get mappings, falling back to defaults if not provided
        if not grouped_mappings or not any(grouped_mappings.get(k) for k in ['header_mappings', 'line_item_mappings', 'summary_mappings']):
            logger.info(f"[FieldNormalizer] No grouped mappings provided, using defaults for {schema_type}")
            grouped_mappings = self.get_default_mappings(schema_type)

        all_mappings = {}

        # Normalize header data
        header_mappings = grouped_mappings.get('header_mappings', [])
        normalized_header, header_map = self._normalize_data_with_rules(
            header_data or {},
            header_mappings,
            'header',
            use_llm
        )
        all_mappings['header'] = header_map

        # Normalize line items
        line_item_mappings = grouped_mappings.get('line_item_mappings', [])
        normalized_line_items, line_item_map = self._normalize_line_items_with_rules(
            line_items or [],
            line_item_mappings,
            use_llm
        )
        all_mappings['line_item'] = line_item_map

        # Normalize summary data
        summary_mappings = grouped_mappings.get('summary_mappings', [])
        normalized_summary, summary_map = self._normalize_data_with_rules(
            summary_data or {},
            summary_mappings,
            'summary',
            use_llm
        )
        all_mappings['summary'] = summary_map

        logger.info(f"[FieldNormalizer] Grouped normalization complete:")
        logger.info(f"[FieldNormalizer]   Header: {len(header_map)} mappings")
        logger.info(f"[FieldNormalizer]   Line items: {len(line_item_map)} mappings")
        logger.info(f"[FieldNormalizer]   Summary: {len(summary_map)} mappings")

        return normalized_header, normalized_line_items, normalized_summary, all_mappings

    def _normalize_data_with_rules(
        self,
        data: Dict[str, Any],
        mapping_rules: List[Dict],
        group_name: str,
        use_llm: bool = True
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Normalize a dict of data using pattern-based mapping rules.

        Args:
            data: Raw data with original field names
            mapping_rules: List of mapping rules with 'canonical', 'patterns', 'exclude_patterns'
            group_name: Name of the group for logging (header, summary)
            use_llm: Whether to use LLM for unmatched fields

        Returns:
            Tuple of (normalized_data, mapping_used)
        """
        if not data:
            return {}, {}

        mapping = {}
        normalized = {}

        # Get all field names
        extracted_fields = list(data.keys())

        for field_name in extracted_fields:
            # Skip system fields
            if field_name in self.SYSTEM_FIELDS:
                normalized[field_name] = data[field_name]
                continue

            # Check if field is a bilingual label
            if self._is_bilingual_label(field_name):
                # Keep bilingual labels as-is, don't try to classify them
                logger.info(f"[FieldNormalizer] [{group_name}] Bilingual label kept as-is: {field_name}")
                normalized[field_name] = data[field_name]
                continue

            # Try to match against mapping rules
            matched_canonical = None
            for rule in mapping_rules:
                if self._matches_mapping_rule(field_name, rule):
                    matched_canonical = rule.get('canonical')
                    logger.debug(f"[FieldNormalizer] [{group_name}] Pattern match: {field_name} → {matched_canonical}")
                    break

            if matched_canonical:
                mapping[field_name] = matched_canonical
                normalized[matched_canonical] = data[field_name]
            else:
                # Keep original field name if no match
                normalized[field_name] = data[field_name]

        return normalized, mapping

    def _normalize_line_items_with_rules(
        self,
        line_items: List[Dict[str, Any]],
        mapping_rules: List[Dict],
        use_llm: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        Normalize line items using pattern-based mapping rules.

        Args:
            line_items: Raw line items with original field names
            mapping_rules: List of mapping rules for line items
            use_llm: Whether to use LLM for unmatched fields

        Returns:
            Tuple of (normalized_line_items, mapping_used)
        """
        if not line_items:
            return [], {}

        # Get all unique field names from line items
        extracted_fields = set()
        for item in line_items:
            extracted_fields.update(item.keys())

        # Build mapping using pattern rules
        mapping = {}
        unmatched_fields = []
        for field_name in extracted_fields:
            if field_name in self.SYSTEM_FIELDS:
                continue

            # Check if field is a bilingual label
            if self._is_bilingual_label(field_name):
                logger.info(f"[FieldNormalizer] [line_item] Bilingual label skipped: {field_name}")
                continue

            # Try to match against mapping rules
            matched = False
            for rule in mapping_rules:
                if self._matches_mapping_rule(field_name, rule):
                    canonical = rule.get('canonical')
                    mapping[field_name] = canonical
                    logger.debug(f"[FieldNormalizer] [line_item] Pattern match: {field_name} → {canonical}")
                    matched = True
                    break

            if not matched:
                unmatched_fields.append(field_name)

        # Use LLM for unmatched fields if available and requested
        if use_llm and unmatched_fields and self.llm_client:
            logger.info(f"[FieldNormalizer] [line_item] Using LLM for {len(unmatched_fields)} unmatched fields: {unmatched_fields}")
            llm_mapping = self._build_line_item_mapping_with_llm(unmatched_fields, mapping_rules)
            if llm_mapping:
                for field_name, canonical in llm_mapping.items():
                    if canonical:  # Only add if LLM found a valid mapping
                        mapping[field_name] = canonical
                        logger.info(f"[FieldNormalizer] [line_item] LLM match: {field_name} → {canonical}")

        # Apply mapping to all line items
        normalized_items = []
        for item in line_items:
            normalized_item = {}
            for key, value in item.items():
                if key in self.SYSTEM_FIELDS:
                    normalized_item[key] = value
                elif key in mapping:
                    normalized_item[mapping[key]] = value
                elif not self._is_bilingual_label(key):
                    # Keep non-bilingual fields that didn't match
                    normalized_item[key] = value
                # Skip bilingual labels entirely

            normalized_items.append(normalized_item)

        logger.info(f"[FieldNormalizer] [line_item] Normalized {len(line_items)} items with {len(mapping)} field mappings")
        return normalized_items, mapping

    def _build_line_item_mapping_with_llm(
        self,
        unmatched_fields: List[str],
        mapping_rules: List[Dict]
    ) -> Dict[str, Optional[str]]:
        """
        Use LLM to map line item fields that couldn't be matched by patterns.

        This handles cases where OCR extracts table column headers that don't match
        standard field name patterns (e.g., "600 x 1 User Messages" should map to "description").

        Args:
            unmatched_fields: Fields that need LLM mapping
            mapping_rules: List of canonical field rules with patterns

        Returns:
            Dict mapping unmatched_field → canonical_name (or None)
        """
        if not self.llm_client:
            return {}

        try:
            # Build canonical field info for prompt
            canonical_fields = {}
            for rule in mapping_rules:
                canonical = rule.get('canonical')
                if canonical:
                    canonical_fields[canonical] = {
                        'data_type': rule.get('data_type', 'string'),
                        'patterns': rule.get('patterns', [])
                    }

            prompt = f"""You are analyzing extracted table column headers from a document.
Map each extracted column name to the most appropriate canonical field name.

## Canonical Fields Available:
{json.dumps(canonical_fields, indent=2)}

## Extracted Column Names to Map:
{json.dumps(unmatched_fields)}

## Rules:
1. Map each extracted field to the most appropriate canonical field name
2. Use semantic meaning - look at what the column likely represents
3. If a field contains "User Messages" or similar product/service text, map to "description"
4. If a field looks like a quantity pattern (numbers like "600 x 1"), consider "quantity" or "description"
5. If a field doesn't match any canonical field well, use null
6. Be case-insensitive when analyzing

Return ONLY a JSON object mapping extracted names to canonical names:
{{"ExtractedName1": "canonical_name1", "ExtractedName2": null, ...}}
"""

            response = self.llm_client.generate(prompt)

            # Parse JSON response
            mapping = self._parse_json_response(response)
            if mapping:
                # Validate that mapped values are actual canonical fields
                valid_mapping = {}
                for orig, canonical in mapping.items():
                    if canonical is None or canonical in canonical_fields:
                        valid_mapping[orig] = canonical
                    else:
                        logger.warning(f"[FieldNormalizer] LLM suggested invalid canonical field: {canonical}")
                return valid_mapping

        except Exception as e:
            logger.error(f"[FieldNormalizer] LLM line item mapping failed: {e}")

        return {}

    def convert_legacy_mappings_to_grouped(
        self,
        legacy_mappings: Dict[str, Dict],
        schema_type: str = "invoice"
    ) -> Dict[str, List[Dict]]:
        """
        Convert legacy flat field_mappings to new grouped format.

        This is a helper for migrating existing schemas.

        Args:
            legacy_mappings: Old flat format {field_name: {semantic_type, aliases, ...}}
            schema_type: Schema type for determining default groupings

        Returns:
            New grouped format with header_mappings, line_item_mappings, summary_mappings
        """
        if not legacy_mappings:
            return self.get_default_mappings(schema_type)

        # Try to classify legacy fields into groups based on semantic type
        header_rules = []
        line_item_rules = []
        summary_rules = []

        # Semantic types that belong to each group
        HEADER_TYPES = {'entity', 'date', 'identifier', 'method', 'region'}
        SUMMARY_TYPES = {'subtotal', 'total', 'tax'}
        LINE_ITEM_TYPES = {'product', 'quantity', 'amount', 'category', 'description'}

        for field_name, field_def in legacy_mappings.items():
            semantic_type = field_def.get('semantic_type', 'unknown')
            data_type = field_def.get('data_type', 'string')
            aliases = field_def.get('aliases', [])

            # Build patterns from field name and aliases
            patterns = [field_name.lower()] + [a.lower() for a in aliases]

            rule = {
                'canonical': field_name,
                'data_type': data_type,
                'patterns': patterns,
                'exclude_patterns': ['/'] if data_type == 'number' else []
            }

            # Classify into group
            if semantic_type in HEADER_TYPES:
                header_rules.append(rule)
            elif semantic_type in SUMMARY_TYPES or 'total' in field_name.lower():
                summary_rules.append(rule)
            else:
                line_item_rules.append(rule)

        return {
            'header_mappings': header_rules,
            'line_item_mappings': line_item_rules,
            'summary_mappings': summary_rules
        }
