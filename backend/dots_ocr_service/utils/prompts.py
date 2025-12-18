dict_promptmode_to_prompt = {
    # prompt_layout_all_en: parse all layout info in json format.
    "prompt_layout_all_en": """CRITICAL FIRST STEP - MULTI-DOCUMENT DETECTION:
Before processing any text, you MUST analyze if this image contains multiple separate documents.

Look for these indicators of multiple documents:
- Two or more distinct receipts, invoices, forms, business cards, letters, or contracts
- Separate regions with clear spatial gaps or borders
- Different company names, headers, or logos
- Different document types or purposes
- Visual separation (white space, borders, different backgrounds)
- Documents placed side-by-side (left/right), stacked (top/bottom), overlapping, or at different angles

IF YOU DETECT MULTIPLE DOCUMENTS (e.g., 2 receipts side-by-side):
You MUST structure your output as follows:

Step 1: Start with a Section-header for the first document:
{"category": "Section-header", "text": "## Document 1 (Left)", "bbox": [x1, y1, x2, y2]}

Step 2: Add ALL layout elements for Document 1 in reading order

Step 3: Add a separator:
{"category": "Section-header", "text": "---", "bbox": [x_mid, y1, x_mid+10, y2]}

Step 4: Add a Section-header for the second document:
{"category": "Section-header", "text": "## Document 2 (Right)", "bbox": [x1, y1, x2, y2]}

Step 5: Add ALL layout elements for Document 2 in reading order

CRITICAL: Do NOT mix content from different documents. Process each document completely before moving to the next.

Example for 2 side-by-side receipts:
[
  {"category": "Section-header", "text": "## Document 1 (Left)", "bbox": [0, 0, 600, 1400]},
  {"category": "Text", "text": "Dragon Legend", "bbox": [235, 237, 392, 272]},
  {"category": "Text", "text": "25 Lanark Rd.", "bbox": [234, 269, 387, 300]},
  ... (all elements from left receipt)
  {"category": "Section-header", "text": "---", "bbox": [600, 0, 610, 1400]},
  {"category": "Section-header", "text": "## Document 2 (Right)", "bbox": [610, 0, 1200, 1400]},
  {"category": "Text", "text": "Providential9", "bbox": [777, 130, 1075, 265]},
  ... (all elements from right receipt)
]

---

Now, output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as Markdown table. For tables containing code, JSON, or API data, preserve the exact formatting including braces, quotes, and special characters.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Special Content Handling:
    - Code/JSON/API Content: When tables or text contain programming code, JSON objects, API responses, or technical documentation:
      * Preserve exact syntax including curly braces {}, square brackets [], quotation marks "", colons :, and commas.
      * Preserve key-value pairs exactly as shown (e.g., "Rel": "getquote", "Method": "GET").
      * URLs must be transcribed character-by-character with full accuracy (e.g., http://example.com/api/quote/BA6F3961).
      * Pay special attention to alphanumeric identifiers like BA6F3961, BA6F4062, etc. - each character matters.
    - HTTP Methods: Recognize common HTTP methods (GET, POST, PUT, DELETE, PATCH) and preserve them in uppercase.

5. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order within each document.
    - For technical content, accuracy of every character is critical.
    - For multi-document images, ensure clear separation between documents with Section-header elements.

6. Final Output: The entire output must be a single JSON array containing all layout elements.
""",

    # prompt_layout_only_en: layout detection
    "prompt_layout_only_en": """Please output the layout information from this PDF image, including each layout's bbox and its category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format.""",

    # prompt_ocr: parse ocr text from the image
    "prompt_ocr": """CRITICAL FIRST STEP - MULTI-DOCUMENT DETECTION:
Before extracting any text, you MUST analyze if this image contains multiple separate documents.

Look for these indicators:
- Two or more distinct receipts, invoices, forms, business cards, letters, or contracts
- Separate regions with clear spatial gaps or borders
- Different company names, headers, or logos
- Visual separation (white space, borders, different backgrounds)
- Documents placed side-by-side (left/right), stacked (top/bottom), or overlapping

IF YOU DETECT MULTIPLE DOCUMENTS:
You MUST structure your output as follows:

## Document 1 (Left)
[Extract ALL text from the first document]

---

## Document 2 (Right)
[Extract ALL text from the second document]

CRITICAL: Do NOT mix content from different documents. Process each document completely before moving to the next.

Example for 2 side-by-side receipts:
## Document 1 (Left)
Dragon Legend
25 Lanark Rd.
Markham, ON L3R 9Y7
Phone: (905)940-1811
...
Total: 222.78

---

## Document 2 (Right)
Providential9
Unit A, 8425 Woodbine Ave
Markham, ON L3R 2P4
(905) 305-1338
...
TOTAL $201.75

---

Now, extract the text content from this image.

Special handling for technical content:
- For code, JSON, or API responses: preserve exact syntax including {}, [], "", :, and commas.
- For URLs: transcribe character-by-character with full accuracy.
- For alphanumeric identifiers (e.g., BA6F3961): each character matters.
- For HTTP methods: preserve in uppercase (GET, POST, PUT, DELETE, PATCH).
- For key-value pairs: preserve exactly as shown (e.g., "Rel": "getquote").""",

    # prompt_grounding_ocr: extract text content in the given bounding box
    "prompt_grounding_ocr": """Extract text from the given bounding box on the image.

Special handling for technical content:
- For code, JSON, or API responses: preserve exact syntax including {}, [], "", :, and commas.
- For URLs: transcribe character-by-character with full accuracy.
- For alphanumeric identifiers (e.g., BA6F3961): each character matters.
- For HTTP methods: preserve in uppercase (GET, POST, PUT, DELETE, PATCH).

Bounding Box (format: [x1, y1, x2, y2]):
""",

    # "prompt_table_html": """Convert the table in this image to HTML.""",
    # "prompt_table_latex": """Convert the table in this image to LaTeX.""",
    # "prompt_formula_latex": """Convert the formula in this image to LaTeX.""",
    "prompt_flow_chart": """Convert the content in this image to ASCII flow diagram.""",
}
