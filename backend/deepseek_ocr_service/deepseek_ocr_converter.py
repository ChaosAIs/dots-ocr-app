"""
DeepSeek OCR Converter

This module provides functionality to convert images to markdown using DeepSeek OCR API.

The DeepSeek-OCR model is served via vLLM and provides an OpenAI-compatible API endpoint.
This converter communicates with the vLLM server to perform OCR on images.

Reference Implementation:
    Based on DeepSeek-OCR-Web-main official sample code:
    - Main script: DeepSeek-OCR-Web-main/backend/run_dpsk_ocr_image.py
    - Image processing: DeepSeek-OCR-Web-main/backend/process/image_process.py
    - Configuration: DeepSeek-OCR-Web-main/backend/config.py
    - Model definition: DeepSeek-OCR-Web-main/backend/deepseek_ocr.py

    Key improvements from sample code:
    - Grounding mode prompts for better OCR accuracy
    - Proper bounding box and grounding tag removal
    - Recommended prompt formats for different image types

Default Prompt Format:
    The default prompt is: "<|grounding|>Convert the document to markdown."

    Note: When using the OpenAI-compatible API endpoint (/v1/chat/completions),
    the <image> token is NOT included in the prompt. The image is sent separately
    via the image_url field, and vLLM handles the image placement internally.

    For direct vLLM API calls (llm.generate()), the <image> token would be required.

    Auto-Detection Feature:
    The converter can automatically detect image types based on filename keywords and
    apply intelligent, context-aware prompts for better understanding and explanation.

    Supported image types:
    - Receipts/Invoices: Extracts transaction details and explains the purchase
    - Forms: Identifies form purpose, fields, and requirements
    - Flowcharts: Analyzes process logic, decision points, and workflow
    - Diagrams: Understands architecture, components, and relationships
    - Charts/Graphs: Extracts data and explains trends and insights
    - Dashboards/Reports: Extracts data from all visualizations, ignores placeholders, provides insights
    - Infographics: Identifies key messages, statistics, and narrative flow
    - Tables: Extracts data with context and highlights patterns
    - Screenshots: Analyzes UI elements and application context
    - Documents: Converts with structure awareness and content understanding

    IMPORTANT: Prompts are kept SHORT and SIMPLE for best results.
    Testing showed that complex multi-step prompts cause the model to output
    repetitive generic text instead of actual data extraction.

    Simple, direct prompts work best:
    - "Extract all text from X. Convert to markdown."
    - "Extract text and data from this dashboard. Convert to markdown."
    - Short, clear instructions produce better OCR results

Usage:
    converter = DeepSeekOCRConverter(
        api_base="http://localhost:8005/v1",
        model_name="deepseek-ai/DeepSeek-OCR"
    )

    # Convert with auto-detection (recommended for flowcharts/diagrams)
    markdown = converter.convert_image_to_markdown(Path("flowchart.png"))
    # Will auto-detect 'flowchart' from filename and use optimized prompt

    # Convert with default document prompt (disable auto-detection)
    markdown = converter.convert_image_to_markdown(
        Path("image.png"),
        auto_detect_type=False
    )

    # Convert with custom prompt (no <image> token needed)
    markdown = converter.convert_image_to_markdown(
        Path("image.png"),
        prompt="Free OCR."
    )
"""

import logging
import base64
from pathlib import Path
from openai import OpenAI
from PIL import Image


class DeepSeekOCRConverter:
    """
    Converter class for converting images to markdown using DeepSeek OCR API.

    This converter uses the OpenAI-compatible API provided by vLLM serving DeepSeek-OCR model.
    The DeepSeek-OCR model is a vision-language model specifically designed for OCR tasks,
    supporting document layout analysis, table recognition, and markdown conversion.

    Key Features:
        - High-quality OCR with layout preservation
        - Table recognition and markdown formatting
        - Support for various image formats (PNG, JPG, GIF, BMP, TIFF, WEBP)
        - Customizable prompts for different OCR tasks
        - Grounding mode for element localization

    Attributes:
        SUPPORTED_EXTENSIONS: List of supported image file extensions
        client: OpenAI client for API communication
        model_name: Name of the DeepSeek-OCR model
        temperature: Sampling temperature (0.0 for deterministic output)
        max_tokens: Maximum tokens to generate
    """
    
    # Supported image extensions
    SUPPORTED_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']
    
    def __init__(
        self,
        api_base: str = "http://localhost:8005/v1",
        model_name: str = "deepseek-ai/DeepSeek-OCR",
        temperature: float = 0.0,
        max_tokens: int = 8192,
        ngram_size: int = 30,
        window_size: int = 90,
        whitelist_token_ids: list = None,
        timeout: int = 3600,
        use_vllm_xargs: bool = False
    ):
        """
        Initialize DeepSeek OCR Converter.

        Args:
            api_base: Base URL for the DeepSeek OCR API (vLLM server)
            model_name: Model name to use
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens to generate
            ngram_size: N-gram size for logits processor
            window_size: Window size for logits processor
            whitelist_token_ids: List of token IDs to whitelist (e.g., for <td>, </td>)
            timeout: Request timeout in seconds
            use_vllm_xargs: Whether to use vllm_xargs in extra_body (default: False)
        """
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or [128821, 128822]
        self.use_vllm_xargs = use_vllm_xargs
        
        # Initialize OpenAI client with vLLM endpoint
        self.client = OpenAI(
            api_key="EMPTY",  # vLLM doesn't require API key
            base_url=api_base,
            timeout=timeout
        )
        
        self.logger.info(f"DeepSeek OCR Converter initialized with API base: {api_base}")
        self.logger.info(f"Model: {model_name}")
    
    def is_supported_file(self, file_path: Path) -> bool:
        """
        Check if the file is a supported image format.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is supported, False otherwise
        """
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS
    
    def _encode_image_to_base64(self, image_path: Path) -> str:
        """
        Encode image to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string with data URI prefix
        """
        try:
            # Open and convert image to RGB
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save to bytes
                import io
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                buffer.seek(0)
                
                # Encode to base64
                image_bytes = buffer.read()
                base64_image = base64.b64encode(image_bytes).decode('utf-8')
                
                return f"data:image/png;base64,{base64_image}"
        except Exception as e:
            self.logger.error(f"Error encoding image {image_path}: {e}")
            raise
    
    def _detect_image_type(self, image_path: Path) -> str:
        """
        Detect the type of image content to determine the best prompt.

        Args:
            image_path: Path to the image file

        Returns:
            Image type: 'flowchart', 'diagram', 'chart', 'infographic', 'form',
            'receipt', 'invoice', 'table', 'screenshot', or 'document'
        """
        try:
            # Simple heuristic based on filename
            filename_lower = image_path.name.lower()

            # Check for receipt/invoice keywords (high priority - specific business documents)
            receipt_keywords = ['receipt', 'invoice', 'bill', 'payment']
            if any(keyword in filename_lower for keyword in receipt_keywords):
                return 'receipt'

            # Check for form keywords
            form_keywords = ['form', 'application', 'survey', 'questionnaire']
            if any(keyword in filename_lower for keyword in form_keywords):
                return 'form'

            # Check for flowchart/diagram keywords
            flowchart_keywords = ['flow', 'flowchart', 'workflow', 'process']
            if any(keyword in filename_lower for keyword in flowchart_keywords):
                return 'flowchart'

            # Check for diagram keywords (broader than flowchart)
            diagram_keywords = ['diagram', 'architecture', 'schema', 'blueprint', 'uml']
            if any(keyword in filename_lower for keyword in diagram_keywords):
                return 'diagram'

            # Check for chart keywords (data visualization)
            chart_keywords = ['chart', 'graph', 'plot', 'bar', 'pie', 'line']
            if any(keyword in filename_lower for keyword in chart_keywords):
                return 'chart'

            # Check for dashboard/report with data visualizations (high priority)
            dashboard_keywords = ['dashboard', 'report', 'financial', 'business', 'analytics', 'metrics', 'kpi']
            if any(keyword in filename_lower for keyword in dashboard_keywords):
                return 'dashboard'

            # Check for infographic keywords
            infographic_keywords = ['infographic', 'visual', 'poster', 'presentation', 'summary', 'analysis']
            if any(keyword in filename_lower for keyword in infographic_keywords):
                return 'infographic'

            # Check for table keywords
            table_keywords = ['table', 'grid', 'spreadsheet']
            if any(keyword in filename_lower for keyword in table_keywords):
                return 'table'

            # Check for screenshot keywords
            screenshot_keywords = ['screenshot', 'screen', 'capture', 'snap']
            if any(keyword in filename_lower for keyword in screenshot_keywords):
                return 'screenshot'

            # Default to document
            return 'document'

        except Exception as e:
            self.logger.warning(f"Error detecting image type: {e}, defaulting to 'document'")
            return 'document'

    def _get_optimized_prompt(self, image_type: str) -> str:
        """
        Get optimized prompt based on image type.

        Based on DeepSeek-OCR-Web-main/backend/config.py recommended prompts.
        Uses grounding mode for better OCR accuracy.

        Args:
            image_type: Type of image ('flowchart', 'diagram', 'chart', 'infographic',
                       'form', 'receipt', 'invoice', 'table', 'screenshot', 'document')

        Returns:
            Optimized prompt string
        """
        # Recommended prompts from sample code:
        # document: <|grounding|>Convert the document to markdown.
        # other image: <|grounding|>OCR this image.
        # without layouts: Free OCR.
        # figures in document: Parse the figure.
        # general: Describe this image in detail.

        if image_type == 'receipt':
            # Receipt/invoice - use grounding mode for precise extraction
            return "<|grounding|>OCR this receipt."

        elif image_type == 'form':
            # Form - use grounding mode
            return "<|grounding|>OCR this form."

        elif image_type == 'flowchart':
            # Flowchart - parse as figure
            return "Parse the figure."

        elif image_type == 'diagram':
            # Diagram - parse as figure
            return "Parse the figure."

        elif image_type == 'chart':
            # Chart - parse as figure
            return "Parse the figure."

        elif image_type == 'dashboard':
            # Dashboard/Business Report - use grounding mode
            return "<|grounding|>OCR this image."

        elif image_type == 'infographic':
            # Infographic - use grounding mode
            return "<|grounding|>OCR this image."

        elif image_type == 'table':
            # Table - use grounding mode for accurate table extraction
            return "<|grounding|>Convert the document to markdown."

        elif image_type == 'screenshot':
            # Screenshot - use grounding mode
            return "<|grounding|>OCR this image."

        else:  # document (default)
            # Document - recommended default prompt from sample code
            return "<|grounding|>Convert the document to markdown."

    def _remove_bounding_boxes(self, content: str) -> str:
        """Post-process grounding output using spatial layout.

        The raw grounding output has the form:

            <|ref|>TEXT<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>

        where the coordinates describe the position of TEXT on the page.

        Instead of just stripping tags, we use the coordinates to build a
        more structured text where numeric values (percentages, $ amounts,
        etc.) are linked to their nearest textual labels. This greatly
        improves dashboards/infographics like the sample "One Pager Business
        Monthly" report.

        NOTE: We intentionally avoid hard-coding specific label strings and
        rely purely on spatial layout + simple numeric detection.
        """
        import re
        from collections import defaultdict
        from statistics import median

        # Normalise LaTeX-style equality symbols (matches official code)
        content = content.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")

        # Regex capturing <|ref|>TEXT</|ref|><|det|>[[x1, y1, x2, y2]]</|det|>
        pattern = (
            r"<\|ref\|>(.*?)<\|/ref\|>"  # TEXT
            r"<\|det\|>\[\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\]<\|/det\|>"
        )
        matches = list(re.finditer(pattern, content, re.DOTALL))

        # If we cannot parse coordinates, fall back to simple tag stripping
        if not matches:
            content_simple = re.sub(
                r"<\|ref\|>image<\|/ref\|><\|det\|>\[\[[^\]]*\]\]<\|/det\|>",
                "",
                content,
                flags=re.DOTALL,
            )
            content_simple = re.sub(
                r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>\[\[[^\]]*\]\]<\|/det\|>",
                r"\1\n",
                content_simple,
                flags=re.DOTALL,
            )
            content_simple = re.sub(r"\[\[\d+,\s*\d+,\s*\d+,\s*\d+\]\]", "", content_simple)
            content_simple = re.sub(r"<\|/?ref\|>", "", content_simple)
            content_simple = re.sub(r"<\|/?det\|>", "", content_simple)
            lines = [line.strip() for line in content_simple.split("\n")]
            lines = [line for line in lines if line]
            return "\n".join(lines)

        # Parse items with coordinates
        items = []
        for m in matches:
            text = m.group(1).strip()
            if not text:
                continue
            if text.lower() == "image":
                # Skip pure image refs; they don't carry textual information
                continue
            x1, y1, x2, y2 = map(int, m.groups()[1:])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            items.append(
                {
                    "text": text,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": cx,
                    "cy": cy,
                }
            )

        if not items:
            return ""

        def is_numeric_token(t: str) -> bool:
            """Heuristic: token with digits and relatively few letters."""
            has_digit = any(ch.isdigit() for ch in t)
            if not has_digit:
                return False
            letters = sum(ch.isalpha() for ch in t)
            digits = sum(ch.isdigit() for ch in t)
            return digits >= letters

        label_items = []
        numeric_items = []
        for it in items:
            if is_numeric_token(it["text"]):
                numeric_items.append(it)
            else:
                label_items.append(it)

        # If we don't have both labels and numbers, fall back to simple cleanup
        if not numeric_items or not label_items:
            content_simple = re.sub(
                r"<\|ref\|>image<\|/ref\|><\|det\|>\[\[[^\]]*\]\]<\|/det\|>",
                "",
                content,
                flags=re.DOTALL,
            )
            content_simple = re.sub(
                r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>\[\[[^\]]*\]\]<\|/det\|>",
                r"\1\n",
                content_simple,
                flags=re.DOTALL,
            )
            content_simple = re.sub(r"\[\[\d+,\s*\d+,\s*\d+,\s*\d+\]\]", "", content_simple)
            content_simple = re.sub(r"<\|/?ref\|>", "", content_simple)
            content_simple = re.sub(r"<\|/?det\|>", "", content_simple)
            lines = [line.strip() for line in content_simple.split("\n")]
            lines = [line for line in lines if line]
            return "\n".join(lines)

        # Approximate page dimensions to set distance thresholds.
        min_x = min(it["x1"] for it in items)
        max_x = max(it["x2"] for it in items)
        min_y = min(it["y1"] for it in items)
        max_y = max(it["y2"] for it in items)
        page_width = max(max_x - min_x, 1)
        page_height = max(max_y - min_y, 1)

        # Use relative text height to distinguish headings / subheadings from labels.
        label_heights = [it["y2"] - it["y1"] for it in label_items]
        base_height = median(label_heights) if label_heights else 1.0

        heading_height_threshold = 1.6 * base_height
        subheading_height_threshold = 1.25 * base_height

        heading_items = []
        subheading_items = []
        value_label_items = []
        for it in label_items:
            height = it["y2"] - it["y1"]
            if height >= heading_height_threshold:
                heading_items.append(it)
            elif height >= subheading_height_threshold:
                subheading_items.append(it)
            else:
                value_label_items.append(it)

        if not value_label_items:
            # If everything looks like a heading, treat them all as labels.
            value_label_items = label_items
            heading_items = []
            subheading_items = []

        # Map each numeric item to its nearest label using spatial distance,
        # but only if it is reasonably close in both x and y.
        def score(num, lab):
            dx = abs(num["cx"] - lab["cx"])
            dy = abs(num["cy"] - lab["cy"])
            # Prefer labels that are vertically close and roughly aligned in x,
            # and (for dashboards like the sample) slightly below the number.
            penalty = 0.0
            if lab["cy"] >= num["cy"]:
                penalty -= 0.3 * dy
            return dy + 0.5 * dx + penalty

        max_vertical_dist = page_height * 0.10
        max_horizontal_dist = page_width * 0.30

        label_to_values = defaultdict(list)
        mapped_numeric_ids = set()

        for num in numeric_items:
            candidates = [
                lab
                for lab in value_label_items
                if abs(num["cy"] - lab["cy"]) <= max_vertical_dist
                and abs(num["cx"] - lab["cx"]) <= max_horizontal_dist
            ]
            if not candidates:
                continue
            best_label = min(candidates, key=lambda lab: score(num, lab))
            label_to_values[best_label["text"]].append((num["cy"], num["text"]))
            mapped_numeric_ids.add(id(num))

        # Build markdown: headings, subheadings, and labels in reading order.
        nodes = (
            [("heading", it) for it in heading_items]
            + [("subheading", it) for it in subheading_items]
            + [("label", it) for it in value_label_items]
        )
        nodes.sort(key=lambda pair: (pair[1]["cy"], pair[1]["cx"]))

        lines = []
        for idx, (kind, it) in enumerate(nodes):
            text = it["text"]
            if kind == "heading":
                level = 1 if idx == 0 else 2
                lines.append(f"{'#' * level} {text}")
            elif kind == "subheading":
                lines.append(f"### {text}")
            else:
                values = label_to_values.get(text)
                if values:
                    values_sorted = [v for _, v in sorted(values)]
                    lines.append(f"- {text}: {', '.join(values_sorted)}")
                else:
                    lines.append(f"- {text}")

        # Add any numeric tokens that did not get mapped (rare)
        for num in sorted(numeric_items, key=lambda v: (v["cy"], v["cx"])):
            if id(num) not in mapped_numeric_ids:
                lines.append(f"- {num['text']}")

        return "\n".join(lines)

    def _detect_hallucination_loop(self, content: str) -> bool:
        """
        Detect if the model is stuck in a hallucination loop.

        Detects repetitive patterns that indicate the model is generating
        the same content over and over.

        Args:
            content: Generated content

        Returns:
            True if hallucination loop detected
        """
        # Split into lines
        lines = content.split('\n')

        # Count consecutive identical or very similar lines
        consecutive_similar = 0
        max_consecutive = 0
        prev_line = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this line is very similar to the previous one
            if line == prev_line or (len(line) > 5 and len(prev_line) > 5 and
                                     line[:10] == prev_line[:10]):
                consecutive_similar += 1
                max_consecutive = max(max_consecutive, consecutive_similar)
            else:
                consecutive_similar = 0

            prev_line = line

        # If we see more than 20 consecutive similar lines, it's likely a loop
        if max_consecutive > 20:
            self.logger.warning(f"Hallucination loop detected: {max_consecutive} consecutive similar lines")
            return True

        return False

    def _fix_markdown_tables(self, markdown_content: str) -> str:
        """
        Fix incomplete markdown tables by adding missing separator lines.

        DeepSeek OCR sometimes generates table headers without the required separator line.
        This function detects such cases and adds the missing separator.

        Args:
            markdown_content: Raw markdown content from OCR

        Returns:
            Fixed markdown content with proper table formatting
        """
        if not markdown_content or not markdown_content.strip():
            return markdown_content

        lines = markdown_content.split('\n')
        fixed_lines = []
        i = 0
        tables_fixed = 0

        while i < len(lines):
            current_line = lines[i]
            fixed_lines.append(current_line)

            # Check if current line looks like a table row (has pipes)
            if '|' in current_line and current_line.strip().startswith('|'):
                # Check if next line exists
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()

                    # If next line is also a table row (has pipes) but NOT a separator (no '---')
                    if next_line and '|' in next_line and '---' not in next_line:
                        # Check if we're at the start of a new table section
                        # This happens when:
                        # 1. Previous line is empty or a heading
                        # 2. Current line is the first row after a heading

                        is_table_start = False
                        if i == 0:
                            is_table_start = True
                        elif i >= 1:
                            prev_line = lines[i - 1].strip()
                            # Table starts after empty line or heading
                            if not prev_line or prev_line.startswith('#'):
                                is_table_start = True

                        # Also check if this looks like a header row
                        # Header rows often have column names or are mostly empty except first column
                        cells = [cell.strip() for cell in current_line.split('|')]
                        cells = [c for c in cells if c]  # Remove empty strings from split

                        # If we're at table start, add separator
                        if is_table_start and len(cells) > 0:
                            # Count the number of columns (number of pipes - 1)
                            num_pipes = current_line.count('|')
                            # For a proper table row like "| A | B | C |", we have 4 pipes
                            # We need 3 columns, so columns = pipes - 1
                            columns = num_pipes - 1

                            if columns > 0:
                                # Create separator line with same number of columns
                                separator = '|' + '---|' * columns
                                fixed_lines.append(separator)
                                tables_fixed += 1
                                self.logger.debug(f"Added table separator after line {i+1}: {separator}")

            i += 1

        fixed_content = '\n'.join(fixed_lines)

        # Log if we made any changes
        if tables_fixed > 0:
            self.logger.info(f"Fixed {tables_fixed} incomplete markdown table(s) by adding separator lines")

        return fixed_content

    def convert_image_to_markdown(
        self,
        image_path: Path,
        prompt: str = None,
        auto_detect_type: bool = True
    ) -> str:
        """
        Convert an image to markdown using DeepSeek OCR API.

        Args:
            image_path: Path to the image file
            prompt: Prompt to use for OCR. If None, uses default prompt or auto-detected prompt.
                   Note: Do NOT include <image> token when using OpenAI-compatible API.
                   The image is sent separately via image_url field.
            auto_detect_type: If True and prompt is None, automatically detect image type
                            and use optimized prompt (default: True)

        Returns:
            Markdown content as string

        Raises:
            ValueError: If file is not supported
            Exception: If API call fails
        """
        if not self.is_supported_file(image_path):
            raise ValueError(
                f"Unsupported file type: {image_path.suffix}. "
                f"Supported types: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Use default prompt if none provided
        # Note: When using OpenAI-compatible API, we don't include <image> token
        # The API handles image placement automatically via the image_url field
        # For direct vLLM API, <image> token would be required in the prompt
        if prompt is None:
            if auto_detect_type:
                # Detect image type and use optimized prompt
                image_type = self._detect_image_type(image_path)
                prompt = self._get_optimized_prompt(image_type)
                self.logger.info(f"Auto-detected image type: {image_type}")
            else:
                # Default instruction for document conversion
                prompt = "<|grounding|>Convert the document to markdown."

        # Remove <image> token if present (not needed for OpenAI-compatible API)
        # The image is sent separately via image_url field
        if "<image>" in prompt:
            self.logger.info("Removing <image> token from prompt (handled by API)")
            prompt = prompt.replace("<image>", "").strip()

        self.logger.info(f"Converting image to markdown: {image_path}")
        self.logger.info(f"Using prompt: {prompt[:100]}...")  # Log first 100 chars

        try:
            # Encode image to base64
            base64_image = self._encode_image_to_base64(image_path)

            # Prepare messages for OpenAI-compatible API
            # The image is sent via image_url, and the prompt is sent as text
            # vLLM will internally handle the image processing and merge with the prompt
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]

            # Call DeepSeek OCR API
            self.logger.info(f"Calling DeepSeek OCR API for {image_path.name}...")

            # Prepare extra_body parameters
            extra_body = {"skip_special_tokens": False}
            if self.use_vllm_xargs:
                extra_body["vllm_xargs"] = {
                    "ngram_size": self.ngram_size,
                    "window_size": self.window_size,
                    "whitelist_token_ids": self.whitelist_token_ids,
                }

            # Calculate appropriate max_tokens to avoid exceeding context length
            # The model has a max context of 8192 tokens total (input + output)
            # We need to leave room for the image and prompt tokens
            # Estimate: large images can take 200-500 tokens, prompt ~50-200 tokens
            # Use lower max_tokens (2048) to prevent hallucination loops
            safe_max_tokens = min(self.max_tokens, 2048)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=safe_max_tokens,
                temperature=self.temperature,
                top_p=0.95,  # Nucleus sampling for more focused output
                extra_body=extra_body,
            )

            # Debug: Log the raw response
            self.logger.debug(f"API Response: {response}")
            self.logger.debug(f"Response choices: {response.choices}")
            if response.choices:
                self.logger.debug(f"First choice message: {response.choices[0].message}")
                self.logger.debug(f"First choice content: {response.choices[0].message.content}")

            # Extract markdown content
            markdown_content = response.choices[0].message.content

            # Clean up the result - remove end-of-sentence tokens if present
            # Based on sample code's output cleaning approach
            if '<｜end▁of▁sentence｜>' in markdown_content:
                markdown_content = markdown_content.replace('<｜end▁of▁sentence｜>', '')
                self.logger.info("Removed end-of-sentence tokens from output")

            # Save original output for debugging (like sample code does)
            self.logger.debug(f"Original OCR output length: {len(markdown_content)} characters")

            # Remove bounding boxes and grounding tags from grounding mode output
            # This matches the sample code's post-processing in run_dpsk_ocr_image.py
            if ('[[' in markdown_content and ']]' in markdown_content) or '<|ref|>' in markdown_content:
                self.logger.info("Detected grounding mode output, cleaning bounding boxes and tags")
                markdown_content = self._remove_bounding_boxes(markdown_content)
                self.logger.info("Removed bounding box coordinates/grounding tags from output")

            # Detect hallucination loops (just log warning, don't truncate)
            if self._detect_hallucination_loop(markdown_content):
                self.logger.warning("Hallucination loop detected - output may contain repetitive content")

            # Fix incomplete markdown tables
            markdown_content = self._fix_markdown_tables(markdown_content)

            self.logger.info(f"Successfully converted {image_path.name} to markdown")
            self.logger.info(f"Output length: {len(markdown_content)} characters")

            return markdown_content

        except Exception as e:
            self.logger.error(f"Error converting image {image_path}: {e}")
            raise
    
    def convert_file(
        self,
        input_path: Path,
        output_path: Path,
        prompt: str = None,
        auto_detect_type: bool = True
    ) -> bool:
        """
        Convert an image file to markdown and save to output path.

        Args:
            input_path: Path to input image file
            output_path: Path to output markdown file
            prompt: Prompt to use for OCR. If None, uses default prompt or auto-detected prompt.
                   Note: Do NOT include <image> token when using OpenAI-compatible API.
            auto_detect_type: If True and prompt is None, automatically detect image type
                            and use optimized prompt (default: True)

        Returns:
            True if conversion successful, False otherwise
        """
        try:
            # Convert image to markdown
            markdown_content = self.convert_image_to_markdown(
                input_path,
                prompt=prompt,
                auto_detect_type=auto_detect_type
            )

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write markdown to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            self.logger.info(f"Markdown saved to: {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to convert {input_path}: {e}")
            return False
    
    def get_supported_extensions(self) -> list:
        """
        Get list of supported file extensions.
        
        Returns:
            List of supported extensions
        """
        return self.SUPPORTED_EXTENSIONS.copy()

