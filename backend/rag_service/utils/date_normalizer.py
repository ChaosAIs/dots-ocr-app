"""
Date normalization utilities for OCR pipeline.

Converts dates to ISO 8601 (YYYY-MM-DD) format while preserving raw text.
Handles fuzzy parsing, locale detection, and ambiguity resolution.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import dateparser

logger = logging.getLogger(__name__)


@dataclass
class DateEntity:
    """Normalized date entity with raw and ISO formats."""
    raw: str                        # Original: "10/14/2025 08:36:15 PM"
    normalized: str                 # ISO 8601: "2025-10-14"
    time: Optional[str] = None      # 24-hour: "20:36:15"
    format_detected: str = ""       # "MM/DD/YYYY HH:MM:SS AM/PM"
    locale: str = "en_US"           # "en_US" or "en_GB"
    confidence: float = 1.0         # 0.0-1.0
    year: Optional[int] = None      # 2025
    month: Optional[int] = None     # 10
    day: Optional[int] = None       # 14
    position: Tuple[int, int] = (0, 0)  # Character position in text
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


def infer_locale_from_context(text: str, date_str: str) -> str:
    """
    Infer locale from document context.
    
    Heuristics:
    - Check for country names, addresses, phone formats
    - Check for currency symbols ($, £, €)
    - Check for other dates with unambiguous day >12
    - Default to en_US if uncertain
    
    Args:
        text: Full document text
        date_str: The date string being parsed
    
    Returns:
        Locale code: "en_US", "en_GB", "en_CA", etc.
    """
    text_lower = text.lower()
    
    # Check for UK/EU indicators
    uk_indicators = ['uk', 'united kingdom', 'london', 'manchester', 'birmingham', '£']
    eu_indicators = ['€', 'europe', 'paris', 'berlin', 'madrid']
    
    if any(ind in text_lower for ind in uk_indicators):
        return "en_GB"
    
    if any(ind in text_lower for ind in eu_indicators):
        return "en_GB"  # Use DD/MM format
    
    # Check for Canada (uses DD/MM in some contexts)
    canada_indicators = ['canada', 'toronto', 'vancouver', 'montreal', 'ontario']
    if any(ind in text_lower for ind in canada_indicators):
        # Check for specific Canadian date patterns
        # For now, default to US format as Canada often uses both
        return "en_US"
    
    # Check for unambiguous dates in text (day > 12)
    # If we find dates like "13/10/2025", it must be DD/MM
    ambiguous_pattern = r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b'
    for match in re.finditer(ambiguous_pattern, text):
        first, second, _ = match.groups()
        if int(first) > 12:
            return "en_GB"  # First number > 12, must be DD/MM
    
    # Default to US format
    return "en_US"


def disambiguate_ambiguous_date(date_str: str, locale: str, text_context: str = "") -> Optional[datetime]:
    """
    Handle MM/DD vs DD/MM ambiguity.
    
    Rules:
    - If day >12: Must be DD/MM (e.g., "13/10" = Oct 13)
    - If locale is en_US: Prefer MM/DD
    - If locale is en_GB/en_CA: Prefer DD/MM
    - If uncertain: Use dateparser with locale settings
    
    Args:
        date_str: Date string to parse
        locale: Locale hint ("en_US", "en_GB", etc.)
        text_context: Surrounding text for additional context
    
    Returns:
        Parsed datetime object or None
    """
    # Check for unambiguous case: day > 12
    slash_pattern = r'^(\d{1,2})/(\d{1,2})/(\d{4})$'
    match = re.match(slash_pattern, date_str.strip())
    
    if match:
        first, second, year = match.groups()
        first_num, second_num = int(first), int(second)
        
        # Unambiguous: first > 12 means DD/MM
        if first_num > 12:
            try:
                return datetime(int(year), second_num, first_num)
            except ValueError:
                pass
        
        # Unambiguous: second > 12 means MM/DD
        if second_num > 12:
            try:
                return datetime(int(year), first_num, second_num)
            except ValueError:
                pass
    
    # Use dateparser with locale-specific settings
    date_order = 'MDY' if locale == 'en_US' else 'DMY'

    # Convert locale format for dateparser (en_US -> en, en_GB -> en)
    dateparser_locale = locale.split('_')[0] if '_' in locale else locale

    parsed = dateparser.parse(
        date_str,
        settings={
            'DATE_ORDER': date_order,
            'PREFER_DATES_FROM': 'past',
            'STRICT_PARSING': False,
        }
        # Don't specify locales parameter - let dateparser auto-detect
    )

    return parsed


def detect_date_format(date_str: str) -> str:
    """
    Detect the format of a date string.

    Returns:
        Format string like "MM/DD/YYYY", "YYYY-MM-DD", etc.
    """
    patterns = {
        r'^\d{1,2}/\d{1,2}/\d{4}': 'MM/DD/YYYY',
        r'^\d{4}-\d{2}-\d{2}': 'YYYY-MM-DD',
        r'^\d{1,2}-\d{1,2}-\d{4}': 'MM-DD-YYYY',
        r'^\w+ \d{1,2},? \d{4}': 'Month DD, YYYY',
        r'^\d{1,2} \w+ \d{4}': 'DD Month YYYY',
    }

    for pattern, format_name in patterns.items():
        if re.match(pattern, date_str.strip()):
            return format_name

    return "Unknown"


def normalize_date(date_str: str, text_context: str = "", locale_hint: str = "en_US") -> Optional[DateEntity]:
    """
    Normalize a date string to ISO 8601 (YYYY-MM-DD).

    Args:
        date_str: Raw date string (e.g., "10/14/2025", "Oct 14, 2025")
        text_context: Surrounding text for locale detection
        locale_hint: Locale hint for ambiguous dates

    Returns:
        DateEntity with normalized date or None if parsing fails

    Examples:
        >>> normalize_date("10/14/2025")
        DateEntity(raw="10/14/2025", normalized="2025-10-14", ...)
        >>> normalize_date("October 14, 2025")
        DateEntity(raw="October 14, 2025", normalized="2025-10-14", ...)
    """
    if not date_str or not date_str.strip():
        return None

    date_str = date_str.strip()

    # Infer locale from context if available
    locale = infer_locale_from_context(text_context, date_str) if text_context else locale_hint

    # Try to parse the date
    parsed = disambiguate_ambiguous_date(date_str, locale, text_context)

    if not parsed:
        logger.debug(f"Failed to parse date: {date_str}")
        return None

    # Extract time if present
    time_str = None
    time_pattern = r'(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM)?'
    time_match = re.search(time_pattern, date_str, re.IGNORECASE)

    if time_match:
        hour, minute, second, meridiem = time_match.groups()
        hour = int(hour)
        minute = int(minute)
        second = int(second) if second else 0

        # Convert to 24-hour format
        if meridiem:
            if meridiem.upper() == 'PM' and hour != 12:
                hour += 12
            elif meridiem.upper() == 'AM' and hour == 12:
                hour = 0

        time_str = f"{hour:02d}:{minute:02d}:{second:02d}"

    # Detect format
    format_detected = detect_date_format(date_str)

    # Create DateEntity
    return DateEntity(
        raw=date_str,
        normalized=parsed.strftime('%Y-%m-%d'),
        time=time_str,
        format_detected=format_detected,
        locale=locale,
        confidence=0.95,  # High confidence for successfully parsed dates
        year=parsed.year,
        month=parsed.month,
        day=parsed.day,
        position=(0, 0)  # Will be set by find_and_normalize_dates
    )


def find_and_normalize_dates(text: str, locale_hint: str = "en_US") -> List[DateEntity]:
    """
    Find all dates in text and normalize them.

    Args:
        text: Input text containing dates
        locale_hint: Default locale for ambiguous dates

    Returns:
        List of DateEntity objects with normalized dates

    Example:
        >>> find_and_normalize_dates("Date: 10/14/2025 08:36:15 PM")
        [DateEntity(raw="10/14/2025 08:36:15 PM", normalized="2025-10-14", ...)]
    """
    dates = []

    # Common date patterns (ordered by specificity)
    patterns = [
        # Date with time: "10/14/2025 08:36:15 PM"
        r'\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?',
        # ISO format: "2025-10-14"
        r'\d{4}-\d{2}-\d{2}',
        # Slash format: "10/14/2025"
        r'\d{1,2}/\d{1,2}/\d{4}',
        # Month name with comma: "October 14, 2025"
        r'\w+\s+\d{1,2},\s+\d{4}',
        # Month name without comma: "October 14 2025"
        r'\w+\s+\d{1,2}\s+\d{4}',
        # Day month year: "14 October 2025"
        r'\d{1,2}\s+\w+\s+\d{4}',
    ]

    seen_positions = set()

    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            start, end = match.span()

            # Skip if we've already found a date at this position
            if any(start >= s and start < e for s, e in seen_positions):
                continue

            raw = match.group(0)
            date_entity = normalize_date(raw, text_context=text, locale_hint=locale_hint)

            if date_entity:
                date_entity.position = (start, end)
                dates.append(date_entity)
                seen_positions.add((start, end))

    return dates


def normalize_query_dates(query: str) -> str:
    """
    Normalize month names and date expressions in query to ISO format.

    Uses dateparser to detect and normalize ALL date formats in the query,
    including month names, numeric dates, and partial dates.

    This helps LLM match user queries with normalized dates in documents.

    Args:
        query: User query (e.g., "do we have meal in 2025 Oct?")

    Returns:
        Query with normalized dates (e.g., "do we have meal in October 2025 (2025-10)?")

    Examples:
        >>> normalize_query_dates("meals in Oct 2025")
        "meals in October 2025 (2025-10)"
        >>> normalize_query_dates("receipts from October 2025")
        "receipts from October 2025 (2025-10)"
        >>> normalize_query_dates("do we have meal in 2025 Oct?")
        "do we have meal in October 2025 (2025-10)?"
    """
    # Month name to number mapping (for expansion)
    month_map = {
        'jan': 'January', 'january': 'January',
        'feb': 'February', 'february': 'February',
        'mar': 'March', 'march': 'March',
        'apr': 'April', 'april': 'April',
        'may': 'May',
        'jun': 'June', 'june': 'June',
        'jul': 'July', 'july': 'July',
        'aug': 'August', 'august': 'August',
        'sep': 'September', 'sept': 'September', 'september': 'September',
        'oct': 'October', 'october': 'October',
        'nov': 'November', 'november': 'November',
        'dec': 'December', 'december': 'December',
    }

    normalized = query

    # Month abbreviation to number mapping
    month_num_map = {
        'jan': '01', 'january': '01',
        'feb': '02', 'february': '02',
        'mar': '03', 'march': '03',
        'apr': '04', 'april': '04',
        'may': '05',
        'jun': '06', 'june': '06',
        'jul': '07', 'july': '07',
        'aug': '08', 'august': '08',
        'sep': '09', 'sept': '09', 'september': '09',
        'oct': '10', 'october': '10',
        'nov': '11', 'november': '11',
        'dec': '12', 'december': '12',
    }

    # Pattern 1: "YYYY Month" or "Month YYYY" (e.g., "2025 Oct", "Oct 2025")
    # Match year (4 digits) followed by month name, or month name followed by year
    year_month_pattern = r'\b(\d{4})\s+([a-zA-Z]+)\b|\b([a-zA-Z]+)\s+(\d{4})\b'
    matches = list(re.finditer(year_month_pattern, query, re.IGNORECASE))

    # Process matches in reverse order to preserve positions
    replacements = []
    for match in reversed(matches):
        original = match.group(0)
        start, end = match.span()

        # Extract year and month from the match
        if match.group(1):  # "YYYY Month" format
            year = match.group(1)
            month_str = match.group(2).lower()
        else:  # "Month YYYY" format
            month_str = match.group(3).lower()
            year = match.group(4)

        # Check if month_str is a valid month name
        if month_str in month_num_map:
            month_num = month_num_map[month_str]
            month_full = month_map.get(month_str, month_str.capitalize())

            # Create normalized format: "October 2025 (2025-10)"
            normalized_date = f"{month_full} {year} ({year}-{month_num})"
            replacements.append((start, end, normalized_date))

    # Apply replacements in reverse order
    for start, end, replacement in replacements:
        normalized = normalized[:start] + replacement + normalized[end:]

    # Pattern 2: Standalone month names (e.g., "Oct", "October")
    # Expand abbreviated month names to full names for better matching
    # Only do this if not already processed in Pattern 1
    words = normalized.split()
    for i, word in enumerate(words):
        word_lower = word.lower().strip('.,!?;:')
        if word_lower in month_map and word_lower not in ['may']:  # Skip if already full name
            # Check if this word was already processed (has parentheses nearby)
            if i > 0 and '(' in words[i-1]:
                continue
            if i < len(words) - 1 and '(' in words[i+1]:
                continue

            # Replace with full month name
            full_month = month_map[word_lower]
            # Preserve original punctuation
            suffix = ''.join(c for c in word if not c.isalnum())
            words[i] = full_month + suffix

    normalized = ' '.join(words)

    return normalized


def augment_text_with_dates(text: str, dates: List[DateEntity]) -> str:
    """
    Augment text with normalized dates for better vector search.

    This adds normalized date formats alongside raw dates to improve
    vector search matching.

    Args:
        text: Original text
        dates: List of DateEntity objects found in text

    Returns:
        Augmented text with normalized dates

    Example:
        Input:  "Date: 10/14/2025 08:36:15 PM"
        Output: "Date: 10/14/2025 08:36:15 PM (2025-10-14, October 2025)"
    """
    if not dates:
        return text

    # Sort dates by position (reverse order to maintain positions)
    sorted_dates = sorted(dates, key=lambda d: d.position[0], reverse=True)

    augmented = text

    for date in sorted_dates:
        start, end = date.position

        # Create augmentation string
        # Include: ISO format, year-month, month name
        month_names = {
            1: 'January', 2: 'February', 3: 'March', 4: 'April',
            5: 'May', 6: 'June', 7: 'July', 8: 'August',
            9: 'September', 10: 'October', 11: 'November', 12: 'December'
        }

        month_name = month_names.get(date.month, '')
        year_month = f"{date.year}-{date.month:02d}"

        augmentation = f" ({date.normalized}, {month_name} {date.year})"

        # Insert augmentation after the raw date
        augmented = augmented[:end] + augmentation + augmented[end:]

    return augmented


def extract_primary_date(dates: List[DateEntity], document_type: str = "") -> Optional[DateEntity]:
    """
    Extract the primary/most important date from a list of dates.

    Heuristics:
    - For receipts/invoices: First date is usually transaction date
    - For resumes: Latest date is usually most relevant
    - For reports: First date is usually report date

    Args:
        dates: List of DateEntity objects
        document_type: Type of document (receipt, resume, report, etc.)

    Returns:
        Primary DateEntity or None
    """
    if not dates:
        return None

    if len(dates) == 1:
        return dates[0]

    # For receipts/invoices, first date is usually transaction date
    if document_type in ["receipt", "invoice"]:
        return dates[0]

    # For resumes, latest date is usually most relevant
    if document_type == "resume":
        return max(dates, key=lambda d: d.normalized)

    # Default: return first date
    return dates[0]

