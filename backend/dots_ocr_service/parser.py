import os
import json
import logging
import re

from tqdm import tqdm
from multiprocessing.pool import ThreadPool, Pool
import argparse
from dotenv import load_dotenv
import time
import threading


from .model.inference import inference_with_vllm
from .utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS
from .utils.image_utils import get_image_by_fitz_doc, fetch_image, smart_resize
from .utils.doc_utils import fitz_doc_to_image, load_images_from_pdf
from .utils.prompts import dict_promptmode_to_prompt
from .utils.layout_utils import post_process_output, draw_layout_on_image, pre_process_bboxes
from .utils.format_transformer import layoutjson2md
from gemma_ocr_service.gemma3_ocr_converter import Gemma3OCRConverter
from qwen_ocr_service.qwen3_ocr_converter import Qwen3OCRConverter


def _to_relative_output_path(absolute_path: str) -> str:
    """
    Convert an absolute output path to a relative path for database storage.
    Strips the OUTPUT_DIR prefix from the path.
    """
    if not absolute_path:
        return absolute_path

    # If already relative, return as-is
    if not os.path.isabs(absolute_path):
        return absolute_path

    # Get OUTPUT_DIR from environment or use default
    output_dir = os.getenv("OUTPUT_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "output"))

    # Strip the OUTPUT_DIR prefix
    if absolute_path.startswith(output_dir + os.sep):
        return absolute_path[len(output_dir) + 1:]
    elif absolute_path.startswith(output_dir):
        return absolute_path[len(output_dir):].lstrip(os.sep)

    return absolute_path


class DotsOCRParser:
    """
    parse image or pdf file
    """

    def __init__(self,
            ip=None,
            port=None,
            model_name=None,
            temperature=None,
            top_p=None,
            max_completion_tokens=None,
            num_thread=None,
            dpi=None,
            output_dir=None,
            min_pixels=None,
            max_pixels=None,
            progress_callback=None,
        ):
        # Load environment variables from .env file
        load_dotenv()

        # Set defaults from environment variables or use hardcoded defaults
        # DOTS_OCR_VLLM_* for main document conversion to markdown with layout detection
        self.ip = ip or os.getenv('DOTS_OCR_VLLM_HOST', 'localhost')
        self.port = port or int(os.getenv('DOTS_OCR_VLLM_PORT', 8001))
        self.model_name = model_name or os.getenv('DOTS_OCR_VLLM_MODEL', 'dots_ocr')
        self.temperature = temperature or float(os.getenv('TEMPERATURE', 0.1))
        self.top_p = top_p or float(os.getenv('TOP_P', 1.0))
        self.max_completion_tokens = max_completion_tokens or int(os.getenv('MAX_COMPLETION_TOKENS', 16384))
        self.num_thread = num_thread or int(os.getenv('NUM_THREAD', 64))
        self.dpi = dpi or int(os.getenv('DPI', 200))
        self.output_dir = output_dir or os.getenv('OUTPUT_DIR', './output')
        self.progress_callback = progress_callback

        # Context for OCR status tracking
        self.current_filename = None  # Original filename for database lookup
        self.current_page_number = None  # Current page being processed

        # Handle None values for min_pixels and max_pixels
        min_pixels_env = os.getenv('MIN_PIXELS', 'None')
        max_pixels_env = os.getenv('MAX_PIXELS', 'None')
        self.min_pixels = min_pixels if min_pixels is not None else (None if min_pixels_env == 'None' else int(min_pixels_env))
        self.max_pixels = max_pixels if max_pixels is not None else (None if max_pixels_env == 'None' else int(max_pixels_env))

        assert self.min_pixels is None or self.min_pixels >= MIN_PIXELS
        assert self.max_pixels is None or self.max_pixels <= MAX_PIXELS

        # Minimum image size threshold for Dots OCR processing
        # Images smaller than this will be skipped entirely to avoid poor results on dense/complex content
        # Default: 100000 pixels (e.g., 316x316, 250x400, etc.)
        # This prevents processing of small technical diagrams with many dense objects
        self.min_image_size_threshold = int(os.getenv('DOTS_MIN_IMAGE_SIZE_THRESHOLD', '100000'))

        # Log initialization configuration
        import logging
        init_logger = logging.getLogger(__name__)
        init_logger.info(f"DotsOCRParser initialized:")
        init_logger.info(f"  DOTS_OCR_VLLM_HOST: {self.ip}")
        init_logger.info(f"  DOTS_OCR_VLLM_PORT: {self.port}")
        init_logger.info(f"  DOTS_OCR_VLLM_MODEL: {self.model_name}")
        init_logger.info(f"  vLLM endpoint: http://{self.ip}:{self.port}/v1")

        # Initialize OCR image analysis converters (Gemma3 and Qwen3 via Ollama).
        # These are optional helpers; if initialization fails, core OCR still works.
        self.gemma3_converter = None
        self.qwen3_converter = None

        try:
            self.gemma3_converter = Gemma3OCRConverter()
        except Exception:
            # If initialization fails, continue without Gemma3 so core OCR still works
            self.gemma3_converter = None

        try:
            self.qwen3_converter = Qwen3OCRConverter()
        except Exception:
            # If initialization fails, continue without Qwen3 so core OCR still works
            self.qwen3_converter = None



    def _inference_with_vllm(self, image, prompt, smooth_progress=True):
        """Inference with optional smooth progress updates during the blocking call

        Args:
            image: Image to process
            prompt: Prompt for inference
            smooth_progress: If True, send smooth progress updates (for single images).
                           If False, don't send updates (for PDF pages processed in parallel).
        """
        import logging
        logger = logging.getLogger(__name__)

        # Log image details before inference
        image_width, image_height = image.size
        image_pixels = image_width * image_height
        logger.info(f"🔍 Starting inference wrapper for image: {image_width}x{image_height} = {image_pixels:,} pixels")
        logger.info(f"   Smooth progress: {smooth_progress}")
        logger.info(f"   Prompt length: {len(prompt)} chars")

        response = [None]  # Use list to allow modification in thread
        exception = [None]
        start_time = [time.time()]

        def run_inference():
            try:
                logger.info(f"🚀 Calling inference_with_vllm() in thread...")
                response[0] = inference_with_vllm(
                    image,
                    prompt,
                    model_name=self.model_name,
                    ip=self.ip,
                    port=self.port,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_completion_tokens=self.max_completion_tokens,
                )
                logger.info(f"✅ inference_with_vllm() completed successfully")
            except Exception as e:
                logger.error(f"❌ inference_with_vllm() failed: {e}")
                logger.error(f"   Error type: {type(e).__name__}")
                logger.exception("Full stack trace:")
                exception[0] = e

        # Start inference in a thread
        inference_thread = threading.Thread(target=run_inference, daemon=True)
        inference_thread.start()

        if smooth_progress:
            # Send smooth progress updates while waiting for inference to complete
            # Progress from 55% to 75% during inference
            start_progress = 55
            end_progress = 75
            update_interval = 0.2  # Update every 200ms for smooth animation

            while inference_thread.is_alive():
                elapsed = time.time() - start_time[0]
                # Calculate progress based on elapsed time (smooth curve)
                progress_ratio = min(elapsed / 30.0, 1.0)  # Assume inference takes ~30 seconds max
                current_progress = int(start_progress + (end_progress - start_progress) * progress_ratio)

                if self.progress_callback:
                    self.progress_callback(
                        progress=current_progress,
                        message=f"Running model inference... ({current_progress}%)"
                    )

                # Wait a bit before next update
                inference_thread.join(timeout=update_interval)
        else:
            # For PDF pages, just wait without sending progress updates
            # Page-level progress will be handled by parse_pdf()
            inference_thread.join()

        # Wait for inference to complete
        inference_thread.join()

        if exception[0]:
            raise exception[0]

        return response[0]

    def _get_image_analysis_converter(self):
        # Determine which image analysis backend to use (Gemma3 or Qwen3)
        backend = (os.getenv("IMAGE_ANALYSIS_BACKEND", "gemma3") or "").strip().lower()
        logger = logging.getLogger(__name__)

        gemma = getattr(self, "gemma3_converter", None)
        qwen = getattr(self, "qwen3_converter", None)

        if backend == "qwen3":
            if qwen is not None:
                return qwen, "qwen3"
            if gemma is not None:
                logger.warning(
                    "IMAGE_ANALYSIS_BACKEND=qwen3 but Qwen3 converter is not available; "
                    "falling back to Gemma3.",
                )
                return gemma, "gemma3"
            logger.info(
                "IMAGE_ANALYSIS_BACKEND=qwen3 but no image analysis converter is "
                "available; skipping image analysis.",
            )
            return None, backend or "qwen3"

        # Default: gemma3
        if gemma is not None:
            return gemma, "gemma3"
        if qwen is not None:
            logger.warning(
                "IMAGE_ANALYSIS_BACKEND=%s but Gemma3 converter is not available; "
                "falling back to Qwen3.",
                backend or "gemma3",
            )
            return qwen, "qwen3"

        logger.info("No image analysis converter is available; skipping image analysis.")
        return None, backend or "gemma3"

    def get_prompt(self, prompt_mode, bbox=None, origin_image=None, image=None, min_pixels=None, max_pixels=None):
        prompt = dict_promptmode_to_prompt[prompt_mode]
        if prompt_mode == 'prompt_grounding_ocr':
            assert bbox is not None
            bboxes = [bbox]
            bbox = pre_process_bboxes(origin_image, bboxes, input_width=image.width, input_height=image.height, min_pixels=min_pixels, max_pixels=max_pixels)[0]
            prompt = prompt + str(bbox)
        return prompt

    def _add_image_analysis_to_markdown(self, md_content: str) -> str:
        """Insert image analysis sections before embedded base64 images.

        This scans for markdown image tags that use a base64 data URL, for example:
            ![](data:image/png;base64,AAAA...)

        The active backend (Gemma3 or Qwen3, selected via IMAGE_ANALYSIS_BACKEND)
        is called to analyze each image, and the resulting markdown is inserted
        immediately before the corresponding image tag.
        """
        converter, backend_name = self._get_image_analysis_converter()
        if converter is None:
            # No image analysis backend configured or available
            return md_content

        logger = logging.getLogger(__name__)

        # Match markdown images with data URLs
        image_pattern = re.compile(r"!\[(.*?)\]\((data:image/[^)]+)\)")

        # Track image position for status tracking
        image_position = 0

        def _analyze_and_replace(match: re.Match) -> str:  # type: ignore[name-defined]
            nonlocal image_position
            current_image_position = image_position
            image_position += 1

            alt_text = match.group(1)
            data_url = match.group(2)

            # Build a *small* local snippet of the markdown around this image to
            # give Gemma context about the document language and style without
            # sending the entire file content. We only keep a limited window
            # around the image location as sample data.
            window_chars = 600
            start = max(0, match.start() - window_chars)
            end = min(len(md_content), match.end() + window_chars)
            local_context = md_content[start:end].strip()
            if len(local_context) > 800:
                head = local_context[:400]
                tail = local_context[-400:]
                local_context = f"{head}\n...\n{tail}"

            # Extract raw base64 string from data URL if present
            base64_str = data_url
            if "base64," in data_url:
                base64_str = data_url.split("base64,", 1)[1]

            # Get image size for tracking
            image_size_pixels = None
            try:
                if hasattr(converter, '_get_image_dimensions'):
                    width, height = converter._get_image_dimensions(base64_str)
                    image_size_pixels = width * height
            except:
                pass

            try:
                # Let the converter choose the appropriate prompt based on image size
                # (simple for small images, complex for large images)
                # Then we'll get that prompt and append language instructions

                # Determine which prompt to use based on image size
                if hasattr(converter, '_is_simple_image') and converter._is_simple_image(base64_str):
                    base_prompt = converter._build_simple_prompt()
                else:
                    base_prompt = converter._build_complex_prompt() if hasattr(converter, '_build_complex_prompt') else converter._build_default_prompt()

                language_instruction = (
                    "\n\nLanguage and style requirements:\n"
                    "- The image belongs to a larger document whose surrounding Markdown "
                    "content is shown below between triple backticks.\n"
                    "```markdown\n"
                    f"{local_context}\n"
                    "```\n"
                    "- Detect the primary human language used in that content and in "
                    "any visible text in the image (for example, Chinese, English, French, etc.).\n"
                    "- Write your entire analysis and any transcribed text in the same "
                    "language as that content.\n"
                    "- Do not translate the document into another language; preserve "
                    "the original language.\n"
                )

                prompt = base_prompt + language_instruction
                analysis_markdown = converter.convert_image_base64_to_markdown(
                    base64_str,
                    prompt=prompt,
                )

                # Check if image was skipped (empty result means too small)
                if not analysis_markdown or not analysis_markdown.strip():
                    # Track skipped embedded image OCR
                    if self.current_filename and self.current_page_number is not None:
                        self._track_embedded_image_ocr_skipped(
                            filename=self.current_filename,
                            page_number=self.current_page_number,
                            image_position=current_image_position,
                            ocr_backend=backend_name,
                            skip_reason="Image too small for OCR",
                            image_size_pixels=image_size_pixels
                        )
                else:
                    # Track successful embedded image OCR
                    if self.current_filename and self.current_page_number is not None:
                        self._track_embedded_image_ocr_success(
                            filename=self.current_filename,
                            page_number=self.current_page_number,
                            image_position=current_image_position,
                            ocr_backend=backend_name,
                            image_size_pixels=image_size_pixels
                        )

            except Exception as e:  # pragma: no cover - safety net
                logger.error("Error during %s image analysis: %s", backend_name, e)

                # Track failed embedded image OCR
                if self.current_filename and self.current_page_number is not None:
                    self._track_embedded_image_ocr_failure(
                        filename=self.current_filename,
                        page_number=self.current_page_number,
                        image_position=current_image_position,
                        ocr_backend=backend_name,
                        error=str(e),
                        image_size_pixels=image_size_pixels
                    )

                return match.group(0)

            if not analysis_markdown:
                return match.group(0)

            analysis_markdown = analysis_markdown.strip()

            # Normalize headings within the analysis so that the first non-empty
            # line starts with a level-4 heading (####), and any other headings
            # are at least level 4 as well.
            normalized_lines: list[str] = []
            first_content_emitted = False
            for line in analysis_markdown.splitlines():
                stripped = line.strip()
                if not first_content_emitted and stripped:
                    first_content_emitted = True
                    heading_match = re.match(r"^(\s{0,3})(#{1,6})(\s+)(.*)", line)
                    if heading_match:
                        indent, hashes, space, rest = heading_match.groups()
                        if len(hashes) < 4:
                            hashes = "#" * 4
                        normalized_lines.append(f"{indent}{hashes}{space}{rest}")
                    else:
                        # Force the first content line to be a level-4 heading so that
                        # the injected block always begins with "####".
                        leading_ws = line[: len(line) - len(line.lstrip(" "))]
                        normalized_lines.append(f"{leading_ws}#### {stripped}")
                    continue

                heading_match = re.match(r"^(\s{0,3})(#{1,6})(\s+)(.*)", line)
                if heading_match:
                    indent, hashes, space, rest = heading_match.groups()
                    if len(hashes) < 4:
                        hashes = "#" * 4
                    normalized_lines.append(f"{indent}{hashes}{space}{rest}")
                else:
                    normalized_lines.append(line)

            analysis_markdown = "\n".join(normalized_lines)

            # Determine whether we actually need to insert horizontal rules before/after.
            # If the original markdown already has a `---` just before or just after this
            # image, we avoid inserting a duplicate separator.
            include_top_rule = True
            include_bottom_rule = True

            # Look at the last non-empty line before the image in the original content.
            before_text = md_content[: match.start()]
            before_lines = before_text.rstrip("\n").splitlines()
            if before_lines:
                last_line = before_lines[-1].strip()
                if last_line == "---":
                    include_top_rule = False

            # Look at the first non-empty line after the image in the original content.
            after_text = md_content[match.end() :]
            for line in after_text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped == "---":
                    include_bottom_rule = False
                break

            # Build the injected section: ensure clear separation from main content.
            # We add a horizontal rule before and after the whole section so the
            # image+analysis block is visually isolated, but only if there isn't already
            # a separator there.
            analysis_section_lines = []

            # Optional separator before the analysis+image block
            analysis_section_lines.append("")  # blank line before block
            if include_top_rule:
                analysis_section_lines.append("---")
                analysis_section_lines.append("")

            # Analysis body first (so it appears before the image tag in the markdown)
            analysis_section_lines.append(analysis_markdown)
            analysis_section_lines.append("")  # blank line between analysis and image

            # Image itself
            analysis_section_lines.append(match.group(0))

            # Optional separator after the analysis block
            if include_bottom_rule:
                analysis_section_lines.append("")  # blank line before closing separator
                analysis_section_lines.append("---")

            analysis_section_lines.append("")  # blank line after entire block

            return "\n".join(analysis_section_lines)

        # Apply replacement across the entire markdown content
        processed = image_pattern.sub(_analyze_and_replace, md_content)

        # Normalize duplicate horizontal rules that may appear when multiple
        # image+analysis blocks are adjacent. If there are two or more '---'
        # separators with only blank lines between them, collapse them into a
        # single separator so the UI doesn't show double lines.
        lines = processed.splitlines()
        normalized_lines = []
        for line in lines:
            if line.strip() == "---":
                # Look back to the last non-empty line in the normalized output
                # to see if we've just emitted another '---'. If so, skip this
                # one to avoid duplicate separators.
                j = len(normalized_lines) - 1
                while j >= 0 and not normalized_lines[j].strip():
                    j -= 1
                if j >= 0 and normalized_lines[j].strip() == "---":
                    continue
            normalized_lines.append(line)

        return "\n".join(normalized_lines)


    # def post_process_results(self, response, prompt_mode, save_dir, save_name, origin_image, image, min_pixels, max_pixels)
    def _parse_single_image(
        self,
        origin_image,
        prompt_mode,
        save_dir,
        save_name,
        source="image",
        page_idx=0,
        bbox=None,
        fitz_preprocess=False,
        ):
        # Check if image is too small for reliable Dots OCR processing
        # Small images with dense content (technical diagrams, engineering drawings) produce poor results
        image_width, image_height = origin_image.size
        image_pixels = image_width * image_height

        if image_pixels < self.min_image_size_threshold:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"⚠️ Image/page too small for Dots OCR ({image_width}x{image_height}={image_pixels:,} pixels < {self.min_image_size_threshold:,}) "
                f"- skipping {source} page {page_idx} to avoid poor results on dense/complex content"
            )

            # Return empty result indicating the image was skipped
            return {
                'page_no': page_idx,
                'skipped': True,
                'skip_reason': f'Image too small: {image_width}x{image_height}={image_pixels:,} pixels < {self.min_image_size_threshold:,} pixels',
                'markdown': '',
                'input_height': image_height,
                'input_width': image_width,
            }

        min_pixels, max_pixels = self.min_pixels, self.max_pixels
        if prompt_mode == "prompt_grounding_ocr":
            min_pixels = min_pixels or MIN_PIXELS  # preprocess image to the final input
            max_pixels = max_pixels or MAX_PIXELS
        if min_pixels is not None: assert min_pixels >= MIN_PIXELS, f"min_pixels should >= {MIN_PIXELS}"
        if max_pixels is not None: assert max_pixels <= MAX_PIXELS, f"max_pixels should <+ {MAX_PIXELS}"

        if source == 'image' and fitz_preprocess:
            image = get_image_by_fitz_doc(origin_image, target_dpi=self.dpi)
            image = fetch_image(image, min_pixels=min_pixels, max_pixels=max_pixels)
        else:
            image = fetch_image(origin_image, min_pixels=min_pixels, max_pixels=max_pixels)
        input_height, input_width = smart_resize(image.height, image.width)
        prompt = self.get_prompt(prompt_mode, bbox, origin_image, image, min_pixels=min_pixels, max_pixels=max_pixels)

        # Run inference (progress updates are handled inside _inference_with_vllm)
        # For PDF pages, use smooth_progress=False to avoid blocking page-level progress updates
        response = self._inference_with_vllm(image, prompt, smooth_progress=(source == "image"))
        result = {'page_no': page_idx,
            "input_height": input_height,
            "input_width": input_width
        }
        if source == 'pdf':
            save_name = f"{save_name}_page_{page_idx}"
        if prompt_mode in ['prompt_layout_all_en', 'prompt_layout_only_en', 'prompt_grounding_ocr']:
            cells, filtered = post_process_output(
                response,
                prompt_mode,
                origin_image,
                image,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                )
            if filtered and prompt_mode != 'prompt_layout_only_en':  # model output json failed, use filtered process
                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w') as w:
                    json.dump(response, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                origin_image.save(image_layout_path)
                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                })

                md_file_path = os.path.join(save_dir, f"{save_name}.md")
                with open(md_file_path, "w", encoding="utf-8") as md_file:
                    md_file.write(cells)
                result.update({
                    'md_content_path': md_file_path
                })
                result.update({
                    'filtered': True
                })
            else:
                try:
                    image_with_layout = draw_layout_on_image(origin_image, cells)
                except Exception as e:
                    print(f"Error drawing layout on image: {e}")
                    image_with_layout = origin_image

                json_file_path = os.path.join(save_dir, f"{save_name}.json")
                with open(json_file_path, 'w') as w:
                    json.dump(cells, w, ensure_ascii=False)

                image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
                image_with_layout.save(image_layout_path)
                result.update({
                    'layout_info_path': json_file_path,
                    'layout_image_path': image_layout_path,
                })
                if prompt_mode != "prompt_layout_only_en":  # no text md when detection only
                    md_content = layoutjson2md(origin_image, cells, text_key='text')
                    md_content_no_hf = layoutjson2md(origin_image, cells, text_key='text', no_page_hf=True)  # used for clean output or metric of omnidocbench、olmbench

                    # Set context for embedded image OCR tracking
                    # save_name format: "filename_page_0" for PDFs, "filename" for images
                    if source == "pdf":
                        self.current_filename = save_name.rsplit('_page_', 1)[0] if '_page_' in save_name else save_name
                        self.current_page_number = page_idx
                    else:
                        self.current_filename = save_name
                        self.current_page_number = 0

                    # Post-process the no-header/footer markdown to insert Gemma3 image analysis
                    md_content_no_hf = self._add_image_analysis_to_markdown(md_content_no_hf)

                    # Clear context after processing
                    self.current_filename = None
                    self.current_page_number = None

                    md_file_path = os.path.join(save_dir, f"{save_name}.md")
                    with open(md_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content)
                    md_nohf_file_path = os.path.join(save_dir, f"{save_name}_nohf.md")
                    with open(md_nohf_file_path, "w", encoding="utf-8") as md_file:
                        md_file.write(md_content_no_hf)
                    result.update({
                        'md_content_path': md_file_path,
                        'md_content_nohf_path': md_nohf_file_path,
                    })
        else:
            image_layout_path = os.path.join(save_dir, f"{save_name}.jpg")
            origin_image.save(image_layout_path)
            result.update({
                'layout_image_path': image_layout_path,
            })

            md_content = response
            md_file_path = os.path.join(save_dir, f"{save_name}.md")
            with open(md_file_path, "w", encoding="utf-8") as md_file:
                md_file.write(md_content)
            result.update({
                'md_content_path': md_file_path,
            })

        return result

    def parse_image(self, input_path, filename, prompt_mode, save_dir, bbox=None, fitz_preprocess=False):
        # Report progress: loading image
        if self.progress_callback:
            self.progress_callback(progress=5, message="Loading image...")

        origin_image = fetch_image(input_path)

        # Check if image is too small for reliable Dots OCR processing
        # Small images with dense content (technical diagrams, engineering drawings) produce poor results
        image_width, image_height = origin_image.size
        image_pixels = image_width * image_height

        if image_pixels < self.min_image_size_threshold:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"⚠️ Image too small for Dots OCR ({image_width}x{image_height}={image_pixels:,} pixels < {self.min_image_size_threshold:,}) "
                f"- skipping processing to avoid poor results on dense/complex content"
            )

            # Report progress: skipped
            if self.progress_callback:
                self.progress_callback(
                    progress=100,
                    message=f"Image too small ({image_width}x{image_height}={image_pixels:,} pixels) - skipped"
                )

            # Return empty result indicating the image was skipped
            return [{
                'file_path': input_path,
                'page_no': 0,
                'skipped': True,
                'skip_reason': f'Image too small: {image_width}x{image_height}={image_pixels:,} pixels < {self.min_image_size_threshold:,} pixels',
                'markdown': '',
            }]

        # Report progress: image loaded, preparing
        if self.progress_callback:
            self.progress_callback(progress=15, message="Image loaded, preparing for processing...")

        # Report progress: processing image
        if self.progress_callback:
            self.progress_callback(progress=25, message="Preprocessing image...")

        result = self._parse_single_image(origin_image, prompt_mode, save_dir, filename, source="image", bbox=bbox, fitz_preprocess=fitz_preprocess)
        result['file_path'] = input_path

        # Report progress: post-processing
        if self.progress_callback:
            self.progress_callback(progress=80, message="Post-processing results...")

        # Report progress: finalizing
        if self.progress_callback:
            self.progress_callback(progress=95, message="Finalizing conversion...")

        return [result]

    def parse_pdf(self, input_path, filename, prompt_mode, save_dir):
        print(f"loading pdf: {input_path}")
        images_origin = load_images_from_pdf(input_path)
        total_pages = len(images_origin)

        # Report progress: PDF loaded
        if self.progress_callback:
            self.progress_callback(progress=10, message=f"PDF loaded with {total_pages} pages")

        # Initialize OCR tracking in database
        self._init_ocr_tracking(filename, total_pages)

        # Check which pages already have markdown files (skip already converted pages)
        pages_to_skip = set()
        existing_results = []

        for page_idx in range(total_pages):
            # Check if _nohf.md file exists (this is the main output file)
            page_md_nohf = os.path.join(save_dir, f"{filename}_page_{page_idx}_nohf.md")
            page_json = os.path.join(save_dir, f"{filename}_page_{page_idx}.json")

            # Skip if the markdown file exists
            if os.path.exists(page_md_nohf):
                pages_to_skip.add(page_idx)
                print(f"⏭️  Skipping page {page_idx + 1}/{total_pages} (already converted)")

                # Try to load existing result from JSON if available
                if os.path.exists(page_json):
                    try:
                        with open(page_json, 'r', encoding='utf-8') as f:
                            json_data = json.load(f)

                            # Handle different JSON formats
                            if isinstance(json_data, list):
                                # JSON is a list of layout objects (old format)
                                existing_result = {
                                    'page_no': page_idx,
                                    'file_path': input_path,
                                    'md_content': '',
                                    'layout_dets': json_data
                                }
                            elif isinstance(json_data, dict):
                                # JSON is a dict with layout_dets key (new format)
                                existing_result = json_data
                                existing_result['page_no'] = page_idx
                                existing_result['file_path'] = input_path
                            else:
                                # Unknown format, create minimal result
                                existing_result = {
                                    'page_no': page_idx,
                                    'file_path': input_path,
                                    'md_content': '',
                                    'layout_dets': []
                                }

                            existing_results.append(existing_result)
                    except Exception as e:
                        print(f"⚠️  Warning: Could not load existing JSON for page {page_idx}: {e}")
                        # Create a minimal result entry even if JSON is missing
                        existing_results.append({
                            'page_no': page_idx,
                            'file_path': input_path,
                            'md_content': '',
                            'layout_dets': []
                        })
                else:
                    # JSON doesn't exist but markdown does - create minimal result
                    print(f"ℹ️  Page {page_idx + 1} has markdown but no JSON, creating minimal result")
                    existing_results.append({
                        'page_no': page_idx,
                        'file_path': input_path,
                        'md_content': '',
                        'layout_dets': []
                    })

        # Create tasks only for pages that need conversion
        tasks = [
            {
                "origin_image": image,
                "prompt_mode": prompt_mode,
                "save_dir": save_dir,
                "save_name": filename,
                "source":"pdf",
                "page_idx": i,
            } for i, image in enumerate(images_origin) if i not in pages_to_skip
        ]

        pages_to_convert = len(tasks)
        pages_skipped = len(pages_to_skip)

        if pages_skipped > 0:
            print(f"📊 Conversion summary: {pages_to_convert} pages to convert, {pages_skipped} pages already converted")
            if self.progress_callback:
                self.progress_callback(
                    progress=10,
                    message=f"Resuming: {pages_to_convert} pages to convert, {pages_skipped} already done"
                )

        def _execute_task(task_args):
            page_idx = task_args.get('page_idx', 0)
            try:
                result = self._parse_single_image(**task_args)

                # Track successful page OCR
                self._track_page_ocr_success(
                    filename=filename,
                    page_number=page_idx,
                    result=result,
                    save_dir=save_dir
                )

                return result
            except Exception as e:
                # Track failed page OCR
                self._track_page_ocr_failure(
                    filename=filename,
                    page_number=page_idx,
                    error=str(e)
                )
                raise

        results = existing_results.copy()  # Start with existing results

        if pages_to_convert > 0:
            num_thread = min(pages_to_convert, self.num_thread)
            print(f"Parsing PDF with {pages_to_convert} pages using {num_thread} threads...")

            with ThreadPool(num_thread) as pool:
                with tqdm(total=pages_to_convert, desc="Processing PDF pages") as pbar:
                    for result in pool.imap_unordered(_execute_task, tasks):
                        results.append(result)
                        pbar.update(1)

                        # Report progress: pages processed (including skipped pages)
                        pages_processed = len(results)
                        progress = 10 + int((pages_processed / total_pages) * 80)  # 10-90% for processing
                        if self.progress_callback:
                            self.progress_callback(
                                progress=progress,
                                message=f"Processing pages: {pages_processed}/{total_pages}"
                            )
        else:
            print(f"✅ All {total_pages} pages already converted, skipping conversion")
            if self.progress_callback:
                self.progress_callback(
                    progress=90,
                    message=f"All {total_pages} pages already converted"
                )

        results.sort(key=lambda x: x["page_no"])
        for i in range(len(results)):
            results[i]['file_path'] = input_path

        # Report progress: finalizing
        if self.progress_callback:
            self.progress_callback(progress=95, message="Finalizing conversion...")

        return results

    def parse_file(self,
        input_path,
        output_dir="",
        prompt_mode="prompt_layout_all_en",
        bbox=None,
        fitz_preprocess=False,
        progress_callback=None
        ):
        output_dir = output_dir or self.output_dir
        output_dir = os.path.abspath(output_dir)
        filename, file_ext = os.path.splitext(os.path.basename(input_path))
        # Convert extension to lowercase for case-insensitive comparison
        file_ext = file_ext.lower()
        save_dir = os.path.join(output_dir, filename)
        os.makedirs(save_dir, exist_ok=True)

        # Set progress callback if provided
        if progress_callback is not None:
            self.progress_callback = progress_callback

        if file_ext == '.pdf':
            results = self.parse_pdf(input_path, filename, prompt_mode, save_dir)
        elif file_ext in image_extensions:
            results = self.parse_image(input_path, filename, prompt_mode, save_dir, bbox=bbox, fitz_preprocess=fitz_preprocess)
        else:
            raise ValueError(f"file extension {file_ext} not supported, supported extensions are {image_extensions} and pdf")

        print(f"Parsing finished, results saving to {save_dir}")
        with open(os.path.join(output_dir, os.path.basename(filename)+'.jsonl'), 'w') as w:
            for result in results:
                w.write(json.dumps(result, ensure_ascii=False) + '\n')

        # Report final completion
        if self.progress_callback:
            self.progress_callback(progress=100, message="Conversion completed successfully")

        return results

    # ========== OCR Status Tracking Helper Methods ==========

    def _init_ocr_tracking(self, filename: str, total_pages: int):
        """Initialize OCR tracking in database for a document."""
        try:
            from db.database import get_db_session
            from db.document_repository import DocumentRepository

            with get_db_session() as db:
                repo = DocumentRepository(db)
                doc = repo.get_by_filename(filename)
                if doc:
                    repo.init_ocr_details(doc)
                    repo.set_total_pages_for_ocr(doc, total_pages)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not initialize OCR tracking for {filename}: {e}")

    def _track_page_ocr_success(self, filename: str, page_number: int, result: dict, save_dir: str):
        """Track successful page OCR in database."""
        try:
            from db.database import get_db_session
            from db.document_repository import DocumentRepository

            # Determine page file path (store as relative path for portability)
            absolute_page_path = os.path.join(save_dir, f"{filename}_page_{page_number}_nohf.md")
            page_file_path = _to_relative_output_path(absolute_page_path)

            # Count embedded images in the result
            embedded_images_count = 0
            if 'md_content' in result:
                # Count markdown image tags in the content
                import re
                image_pattern = re.compile(r"!\[.*?\]\(data:image/[^)]+\)")
                embedded_images_count = len(image_pattern.findall(result['md_content']))

            with get_db_session() as db:
                repo = DocumentRepository(db)
                doc = repo.get_by_filename(filename)
                if doc:
                    repo.update_page_ocr_status(
                        doc,
                        page_number=page_number,
                        page_file_path=page_file_path,
                        status="success",
                        embedded_images_count=embedded_images_count
                    )
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not track page OCR success for {filename} page {page_number}: {e}")

    def _track_page_ocr_failure(self, filename: str, page_number: int, error: str):
        """Track failed page OCR in database."""
        try:
            from db.database import get_db_session
            from db.document_repository import DocumentRepository

            with get_db_session() as db:
                repo = DocumentRepository(db)
                doc = repo.get_by_filename(filename)
                if doc:
                    repo.update_page_ocr_status(
                        doc,
                        page_number=page_number,
                        page_file_path=None,
                        status="failed",
                        error=error
                    )
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Could not track page OCR failure for {filename} page {page_number}: {e}")

    def _track_embedded_image_ocr_success(
        self,
        filename: str,
        page_number: int,
        image_position: int,
        ocr_backend: str,
        image_size_pixels: int = None
    ):
        """Track successful embedded image OCR in database."""
        try:
            from db.database import get_db_session
            from db.document_repository import DocumentRepository

            with get_db_session() as db:
                repo = DocumentRepository(db)
                doc = repo.get_by_filename(filename)
                if doc:
                    repo.update_embedded_image_ocr_status(
                        doc,
                        page_number=page_number,
                        image_position=image_position,
                        status="success",
                        ocr_backend=ocr_backend,
                        image_size_pixels=image_size_pixels
                    )
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Could not track embedded image OCR success for {filename} "
                f"page {page_number} image {image_position}: {e}"
            )

    def _track_embedded_image_ocr_failure(
        self,
        filename: str,
        page_number: int,
        image_position: int,
        ocr_backend: str,
        error: str,
        image_size_pixels: int = None
    ):
        """Track failed embedded image OCR in database."""
        try:
            from db.database import get_db_session
            from db.document_repository import DocumentRepository

            with get_db_session() as db:
                repo = DocumentRepository(db)
                doc = repo.get_by_filename(filename)
                if doc:
                    repo.update_embedded_image_ocr_status(
                        doc,
                        page_number=page_number,
                        image_position=image_position,
                        status="failed",
                        ocr_backend=ocr_backend,
                        error=error,
                        image_size_pixels=image_size_pixels
                    )
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Could not track embedded image OCR failure for {filename} "
                f"page {page_number} image {image_position}: {e}"
            )

    def _track_embedded_image_ocr_skipped(
        self,
        filename: str,
        page_number: int,
        image_position: int,
        ocr_backend: str,
        skip_reason: str,
        image_size_pixels: int = None
    ):
        """Track skipped embedded image OCR in database."""
        try:
            from db.database import get_db_session
            from db.document_repository import DocumentRepository

            with get_db_session() as db:
                repo = DocumentRepository(db)
                doc = repo.get_by_filename(filename)
                if doc:
                    repo.update_embedded_image_ocr_status(
                        doc,
                        page_number=page_number,
                        image_position=image_position,
                        status="skipped",
                        ocr_backend=ocr_backend,
                        skip_reason=skip_reason,
                        image_size_pixels=image_size_pixels
                    )
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Could not track embedded image OCR skip for {filename} "
                f"page {page_number} image {image_position}: {e}"
            )


def main():
    prompts = list(dict_promptmode_to_prompt.keys())
    parser = argparse.ArgumentParser(
        description="dots.ocr Multilingual Document Layout Parser",
    )

    parser.add_argument(
        "input_path", type=str,
        help="Input PDF/image file path"
    )

    parser.add_argument(
        "--output", type=str, default="./output",
        help="Output directory (default: ./output)"
    )

    parser.add_argument(
        "--prompt", choices=prompts, type=str, default="prompt_layout_all_en",
        help="prompt to query the model, different prompts for different tasks"
    )
    parser.add_argument(
        '--bbox',
        type=int,
        nargs=4,
        metavar=('x1', 'y1', 'x2', 'y2'),
        help='should give this argument if you want to prompt_grounding_ocr'
    )
    parser.add_argument(
        "--ip", type=str, default="localhost",
        help=""
    )
    parser.add_argument(
        "--port", type=int, default=8001,
        help=""
    )
    parser.add_argument(
        "--model_name", type=str, default="dots_ocr",
        help=""
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help=""
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0,
        help=""
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help=""
    )
    parser.add_argument(
        "--max_completion_tokens", type=int, default=16384,
        help=""
    )
    parser.add_argument(
        "--num_thread", type=int, default=16,
        help=""
    )
    # parser.add_argument(
    #     "--fitz_preprocess", type=bool, default=False,
    #     help="False will use tikz dpi upsample pipeline, good for images which has been render with low dpi, but maybe result in higher computational costs"
    # )
    parser.add_argument(
        "--min_pixels", type=int, default=None,
        help=""
    )
    parser.add_argument(
        "--max_pixels", type=int, default=None,
        help=""
    )
    args = parser.parse_args()

    dots_ocr_parser = DotsOCRParser(
        ip=args.ip,
        port=args.port,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_tokens=args.max_completion_tokens,
        num_thread=args.num_thread,
        dpi=args.dpi,
        output_dir=args.output,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    result = dots_ocr_parser.parse_file(
        args.input_path,
        prompt_mode=args.prompt,
        bbox=args.bbox,
        )



if __name__ == "__main__":
    main()