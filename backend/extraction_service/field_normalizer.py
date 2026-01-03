"""
Field Normalizer Module

Normalizes extracted field names to canonical schema names using alias matching
and LLM-based semantic mapping as fallback.

This ensures consistent field naming across all documents regardless of how
the original data named the fields (e.g., "Item" → "description", "Qty" → "quantity").
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

Return ONLY a JSON object mapping extracted names to canonical names:
{{"ExtractedName1": "canonical_name1", "ExtractedName2": "canonical_name2", ...}}

Example:
Input fields: ["Item", "Qty", "Amount", "row_number"]
Schema has: description (aliases: Item, Product), quantity (aliases: Qty), amount (aliases: Amount, Total)
Output: {{"Item": "description", "Qty": "quantity", "Amount": "amount", "row_number": "row_number"}}
"""


class FieldNormalizer:
    """
    Normalizes extracted field names to canonical schema names.

    Uses a two-tier approach:
    1. Alias matching (fast, exact match)
    2. LLM semantic mapping (fallback for unmatched fields)
    """

    # System fields to preserve as-is
    SYSTEM_FIELDS = {'row_number', 'line_number', 'id', '_id'}

    def __init__(self, llm_client=None):
        """
        Initialize the field normalizer.

        Args:
            llm_client: Optional LLM client for semantic matching fallback
        """
        self.llm_client = llm_client

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
