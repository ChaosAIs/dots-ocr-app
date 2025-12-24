import os
import sys
import json
import re
from html.parser import HTMLParser

from PIL import Image
from .image_utils import PILimage_to_base64


def has_latex_markdown(text: str) -> bool:
    """
    Checks if a string contains LaTeX markdown patterns.
    
    Args:
        text (str): The string to check.
        
    Returns:
        bool: True if LaTeX markdown is found, otherwise False.
    """
    if not isinstance(text, str):
        return False
    
    # Define regular expression patterns for LaTeX markdown
    latex_patterns = [
        r'\$\$.*?\$\$',           # Block-level math formula $$...$$
        r'\$[^$\n]+?\$',          # Inline math formula $...$
        r'\\begin\{.*?\}.*?\\end\{.*?\}',  # LaTeX environment \begin{...}...\end{...}
        r'\\[a-zA-Z]+\{.*?\}',    # LaTeX command \command{...}
        r'\\[a-zA-Z]+',           # Simple LaTeX command \command
        r'\\\[.*?\\\]',           # Display math formula \[...\]
        r'\\\(.*?\\\)',           # Inline math formula \(...\)
    ]
    
    # Check if any of the patterns match
    for pattern in latex_patterns:
        if re.search(pattern, text, re.DOTALL):
            return True
    
    return False


def clean_latex_preamble(latex_text: str) -> str:
    """
    Removes LaTeX preamble commands like document class and package imports.
    
    Args:
        latex_text (str): The original LaTeX text.

    Returns:
        str: The cleaned LaTeX text without preamble commands.
    """
    # Define patterns to be removed
    patterns = [
        r'\\documentclass\{[^}]+\}',  # \documentclass{...}
        r'\\usepackage\{[^}]+\}',    # \usepackage{...}
        r'\\usepackage\[[^\]]*\]\{[^}]+\}',  # \usepackage[options]{...}
        r'\\begin\{document\}',       # \begin{document}
        r'\\end\{document\}',         # \end{document}
    ]
    
    # Apply each pattern to clean the text
    cleaned_text = latex_text
    for pattern in patterns:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
    
    return cleaned_text
    

def get_formula_in_markdown(text: str) -> str:
    """
    Formats a string containing a formula into a standard Markdown block.
    
    Args:
        text (str): The input string, potentially containing a formula.

    Returns:
        str: The formatted string, ready for Markdown rendering.
    """
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Check if it's already enclosed in $$
    if text.startswith('$$') and text.endswith('$$'):
        text_new = text[2:-2].strip()
        if not '$' in text_new:
            return f"$$\n{text_new}\n$$"
        else:
            return text

    # Handle \[...\] format, convert to $$...$$
    if text.startswith('\\[') and text.endswith('\\]'):
        inner_content = text[2:-2].strip()
        return f"$$\n{inner_content}\n$$"
        
    # Check if it's enclosed in \[ \]
    if len(re.findall(r'.*\\\[.*\\\].*', text)) > 0:
        return text

    # Handle inline formulas ($...$)
    pattern = r'\$([^$]+)\$'
    matches = re.findall(pattern, text)
    if len(matches) > 0:
        # It's an inline formula, return it as is
        return text  

    # If no LaTeX markdown syntax is present, return directly
    if not has_latex_markdown(text):  
        return text

    # Handle unnecessary LaTeX formatting like \usepackage
    if 'usepackage' in text:
        text = clean_latex_preamble(text)

    if text[0] == '`' and text[-1] == '`':
        text = text[1:-1]

    # Enclose the final text in a $$ block with newlines
    text = f"$$\n{text}\n$$"
    return text 


def clean_text(text: str) -> str:
    """
    Cleans text by removing extra whitespace.

    Args:
        text: The original text.

    Returns:
        str: The cleaned text.
    """
    if not text:
        return ""

    # Remove leading and trailing whitespace
    text = text.strip()

    # Replace multiple consecutive whitespace characters with a single space
    text = re.sub(r'\s+', ' ', text)

    return text


class HTMLTableParser(HTMLParser):
    """
    Parses HTML tables and converts them to Markdown format.
    """
    def __init__(self):
        super().__init__()
        self.rows = []
        self.current_row = []
        self.current_cell = []
        self.in_table = False
        self.in_thead = False
        self.in_tbody = False
        self.in_tr = False
        self.in_th = False
        self.in_td = False

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.in_table = True
            self.rows = []
        elif tag == 'thead':
            self.in_thead = True
        elif tag == 'tbody':
            self.in_tbody = True
        elif tag == 'tr':
            self.in_tr = True
            self.current_row = []
        elif tag == 'th':
            self.in_th = True
            self.current_cell = []
        elif tag == 'td':
            self.in_td = True
            self.current_cell = []

    def handle_endtag(self, tag):
        if tag == 'table':
            self.in_table = False
        elif tag == 'thead':
            self.in_thead = False
        elif tag == 'tbody':
            self.in_tbody = False
        elif tag == 'tr':
            self.in_tr = False
            if self.current_row:
                self.rows.append({
                    'cells': self.current_row.copy(),
                    'is_header': self.in_thead
                })
        elif tag == 'th':
            self.in_th = False
            cell_text = ''.join(self.current_cell).strip()
            self.current_row.append(cell_text)
        elif tag == 'td':
            self.in_td = False
            cell_text = ''.join(self.current_cell).strip()
            self.current_row.append(cell_text)

    def handle_data(self, data):
        if self.in_th or self.in_td:
            self.current_cell.append(data)


def html_table_to_markdown(html: str) -> str:
    """
    Converts an HTML table to Markdown table format.

    Args:
        html: HTML string containing a table.

    Returns:
        str: Markdown formatted table.
    """
    if not html or not isinstance(html, str):
        return html

    # Check if it's actually an HTML table
    if '<table' not in html.lower():
        return html

    parser = HTMLTableParser()
    try:
        parser.feed(html)
    except Exception as e:
        print(f"Error parsing HTML table: {e}")
        return html

    if not parser.rows:
        return html

    # Build markdown table
    markdown_lines = []

    # Determine the maximum number of columns
    max_cols = max(len(row['cells']) for row in parser.rows) if parser.rows else 0

    if max_cols == 0:
        return html

    # Process rows
    has_header = False
    for i, row in enumerate(parser.rows):
        cells = row['cells']
        is_header = row['is_header']

        # Pad cells to match max_cols
        while len(cells) < max_cols:
            cells.append('')

        # Escape pipe characters in cell content
        escaped_cells = [cell.replace('|', '\\|') for cell in cells]

        # Create markdown row
        markdown_row = '| ' + ' | '.join(escaped_cells) + ' |'
        markdown_lines.append(markdown_row)

        # Add separator after header row
        if is_header or (i == 0 and not has_header):
            separator = '| ' + ' | '.join(['---'] * max_cols) + ' |'
            markdown_lines.append(separator)
            has_header = True

    # If no header was found, add separator after first row
    if not has_header and len(markdown_lines) > 0:
        separator = '| ' + ' | '.join(['---'] * max_cols) + ' |'
        markdown_lines.insert(1, separator)

    return '\n'.join(markdown_lines)


def normalize_markdown_table(text: str) -> str:
    """
    Normalize a markdown table by ensuring each row is on a separate line.
    Handles tables that are on a single line or have improper formatting.

    Args:
        text: Markdown table text (may be single-line or multi-line)

    Returns:
        str: Properly formatted markdown table with each row on a separate line
    """
    if not text or '|' not in text:
        return text

    # If the table already has newlines, check if it's properly formatted
    if '\n' in text:
        lines = text.split('\n')
        # If we have multiple lines with pipes, it's likely already formatted
        pipe_lines = [line for line in lines if '|' in line]
        if len(pipe_lines) > 1:
            return text

    # Find the separator row pattern
    separator_match = re.search(r'\|[\s\-:]+\|', text)
    if not separator_match:
        return text

    # Find the complete separator by extending from the match
    sep_start = separator_match.start()
    sep_end = separator_match.end()

    # Extend forward to capture all separator cells
    while sep_end < len(text) and text[sep_end] in '|-: \t':
        sep_end += 1

    separator = text[sep_start:sep_end].strip()

    # Count columns by counting the number of dash groups
    num_cols = len(re.findall(r'-+', separator))

    # Extract header and data
    header_text = text[:sep_start].strip()
    data_text = text[sep_end:].strip()

    # Parse header cells
    header_cells = [c.strip() for c in header_text.split('|')]
    # Remove empty strings from start/end (artifacts of split)
    while header_cells and header_cells[0] == '':
        header_cells.pop(0)
    while header_cells and header_cells[-1] == '':
        header_cells.pop()

    # Parse data cells
    data_cells = [c.strip() for c in data_text.split('|')]
    # Remove empty strings from start/end
    while data_cells and data_cells[0] == '':
        data_cells.pop(0)
    while data_cells and data_cells[-1] == '':
        data_cells.pop()

    # Build result
    result = []

    # Add header
    if header_cells:
        header_row = '| ' + ' | '.join(header_cells) + ' |'
        result.append(header_row)

    # Add separator - create a clean separator based on num_cols
    clean_separator = '| ' + ' | '.join(['---'] * num_cols) + ' |'
    result.append(clean_separator)

    # Group data cells into rows
    # The pattern is: num_cols cells, then 1 empty cell (row boundary marker)
    # So we process in chunks of (num_cols + 1) and take only the first num_cols
    if data_cells:
        i = 0
        while i < len(data_cells):
            # Take num_cols cells for this row
            row_cells = []
            for j in range(num_cols):
                if i < len(data_cells):
                    row_cells.append(data_cells[i])
                    i += 1
                else:
                    row_cells.append('')  # Pad if needed

            # Skip the next cell if it's empty (row boundary marker)
            if i < len(data_cells) and data_cells[i] == '':
                i += 1

            # Create the row
            row_str = '| ' + ' | '.join(row_cells) + ' |'
            result.append(row_str)

    return '\n'.join(result) if len(result) > 1 else text


def is_markdown_table(text: str) -> bool:
    """
    Check if text contains a markdown table.
    A markdown table has rows with pipes and at least one separator row with dashes.
    Handles both multi-line tables and single-line tables.

    Args:
        text: Text to check

    Returns:
        bool: True if text appears to be a markdown table
    """
    if not text or '|' not in text:
        return False

    # Check for separator pattern (contains --- and |)
    # This works for both single-line and multi-line tables
    if '---' in text or '--' in text:
        # Look for the pattern |--| or | --- | which indicates a separator row
        if re.search(r'\|[\s\-:]+\|', text):
            return True

    # Fallback: Split by newlines and check line by line
    lines = text.split('\n')

    # Check if there's at least one line with pipes and one separator line
    has_pipes = any('|' in line for line in lines)

    # Check for separator line (contains ---, may have : for alignment, and only | and spaces)
    has_separator = any(
        '---' in line and '|' in line and
        all(c in '|-: \t' for c in line.strip())
        for line in lines
    )

    return has_pipes and has_separator


def layoutjson2md(image: Image.Image, cells: list, text_key: str = 'text', no_page_hf: bool = False) -> str:
    """
    Converts a layout JSON format to Markdown.

    In the layout JSON, formulas are LaTeX, tables are plain text, and text is Markdown.

    Args:
        image: A PIL Image object.
        cells: A list of dictionaries, each representing a layout cell.
        text_key: The key for the text field in the cell dictionary.
        no_page_header_footer: If True, skips page headers and footers.

    Returns:
        str: The text in Markdown format.
    """
    text_items = []

    for i, cell in enumerate(cells):
        x1, y1, x2, y2 = [int(coord) for coord in cell['bbox']]
        text = cell.get(text_key, "")

        if no_page_hf and cell['category'] in ['Page-header', 'Page-footer']:
            continue

        if cell['category'] == 'Picture':
            image_crop = image.crop((x1, y1, x2, y2))
            image_base64 = PILimage_to_base64(image_crop)
            text_items.append(f"![]({image_base64})")
        elif cell['category'] == 'Formula':
            text_items.append(get_formula_in_markdown(text))
        elif cell['category'] == 'Table':
            # Tables are now extracted as plain text
            # Check if it looks like HTML (legacy format) and convert if needed
            if text.strip().startswith('<table'):
                # Convert HTML table to Markdown table (for backward compatibility)
                markdown_table = html_table_to_markdown(text)
                normalized_table = normalize_markdown_table(markdown_table)
                text_items.append(normalized_table)
            else:
                # Plain text table - wrap in code block to preserve formatting
                text_items.append(f"```\n{text}\n```")
        else:
            # Check if the text contains a markdown table (even if category is not 'Table')
            # This handles cases where OCR returns markdown tables as plain text
            if is_markdown_table(text):
                # Normalize the table to ensure proper formatting
                normalized_table = normalize_markdown_table(text)
                text_items.append(normalized_table)
            else:
                text = clean_text(text)
                text_items.append(f"{text}")

    markdown_text = '\n\n'.join(text_items)
    return markdown_text


def fix_streamlit_formulas(md: str) -> str:
    """
    Fixes the format of formulas in Markdown to ensure they display correctly in Streamlit.
    It adds a newline after the opening $$ and before the closing $$ if they don't already exist.
    
    Args:
        md_text (str): The Markdown text to fix.
        
    Returns:
        str: The fixed Markdown text.
    """
    
    # This inner function will be used by re.sub to perform the replacement
    def replace_formula(match):
        content = match.group(1)
        # If the content already has surrounding newlines, don't add more.
        if content.startswith('\n'):
            content = content[1:]
        if content.endswith('\n'):
            content = content[:-1]
        return f'$$\n{content}\n$$'
    
    # Use regex to find all $$....$$ patterns and replace them using the helper function.
    return re.sub(r'\$\$(.*?)\$\$', replace_formula, md, flags=re.DOTALL)
