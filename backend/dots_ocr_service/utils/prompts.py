dict_promptmode_to_prompt = {
    # prompt_layout_all_en: parse all layout info in json format.
    "prompt_layout_all_en": """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

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
    - All layout elements must be sorted according to human reading order.
    - For technical content, accuracy of every character is critical.

6. Final Output: The entire output must be a single JSON object.
""",

    # prompt_layout_only_en: layout detection
    "prompt_layout_only_en": """Please output the layout information from this PDF image, including each layout's bbox and its category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format.""",

    # prompt_ocr: parse ocr text from the image
    "prompt_ocr": """Extract the text content from this image.

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
