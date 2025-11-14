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


from dots_ocr.model.inference import inference_with_vllm
from dots_ocr.utils.consts import image_extensions, MIN_PIXELS, MAX_PIXELS
from dots_ocr.utils.image_utils import get_image_by_fitz_doc, fetch_image, smart_resize
from dots_ocr.utils.doc_utils import fitz_doc_to_image, load_images_from_pdf
from dots_ocr.utils.prompts import dict_promptmode_to_prompt
from dots_ocr.utils.layout_utils import post_process_output, draw_layout_on_image, pre_process_bboxes
from dots_ocr.utils.format_transformer import layoutjson2md
from gemma_ocr_service.gemma3_ocr_converter import Gemma3OCRConverter



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
        self.ip = ip or os.getenv('VLLM_IP', 'localhost')
        self.port = port or int(os.getenv('VLLM_PORT', 8001))
        self.model_name = model_name or os.getenv('VLLM_MODEL_NAME', 'dots_ocr')
        self.temperature = temperature or float(os.getenv('TEMPERATURE', 0.1))
        self.top_p = top_p or float(os.getenv('TOP_P', 1.0))
        self.max_completion_tokens = max_completion_tokens or int(os.getenv('MAX_COMPLETION_TOKENS', 16384))
        self.num_thread = num_thread or int(os.getenv('NUM_THREAD', 64))
        self.dpi = dpi or int(os.getenv('DPI', 200))
        self.output_dir = output_dir or os.getenv('OUTPUT_DIR', './output')
        self.progress_callback = progress_callback

        # Handle None values for min_pixels and max_pixels
        min_pixels_env = os.getenv('MIN_PIXELS', 'None')
        max_pixels_env = os.getenv('MAX_PIXELS', 'None')
        self.min_pixels = min_pixels if min_pixels is not None else (None if min_pixels_env == 'None' else int(min_pixels_env))
        self.max_pixels = max_pixels if max_pixels is not None else (None if max_pixels_env == 'None' else int(max_pixels_env))

        assert self.min_pixels is None or self.min_pixels >= MIN_PIXELS
        assert self.max_pixels is None or self.max_pixels <= MAX_PIXELS
        # Initialize Gemma3 OCR converter for optional image analysis
        try:
            self.gemma3_converter = Gemma3OCRConverter()
        except Exception:
            # If initialization fails, continue without Gemma3 so core OCR still works
            self.gemma3_converter = None



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

    def get_prompt(self, prompt_mode, bbox=None, origin_image=None, image=None, min_pixels=None, max_pixels=None):
        prompt = dict_promptmode_to_prompt[prompt_mode]
        if prompt_mode == 'prompt_grounding_ocr':
            assert bbox is not None
            bboxes = [bbox]
            bbox = pre_process_bboxes(origin_image, bboxes, input_width=image.width, input_height=image.height, min_pixels=min_pixels, max_pixels=max_pixels)[0]
            prompt = prompt + str(bbox)
        return prompt
    def _add_image_analysis_to_markdown(self, md_content: str) -> str:
        """Insert Gemma3 image analysis sections before embedded base64 images.

        This scans for markdown image tags that use a base64 data URL, for example:
            ![](data:image/png;base64,AAAA...)

        For each image found, it calls Gemma3 via the Gemma3OCRConverter and inserts
        an "Image Analysis" section immediately before the image tag.
        """
        # If Gemma3 converter is not available, return content unchanged
        if not getattr(self, "gemma3_converter", None):
            return md_content

        logger = logging.getLogger(__name__)

        # Match markdown images with data URLs
        image_pattern = re.compile(r"!\[(.*?)\]\((data:image/[^)]+)\)")

        def _analyze_and_replace(match: re.Match) -> str:  # type: ignore[name-defined]
            alt_text = match.group(1)
            data_url = match.group(2)

            # Extract raw base64 string from data URL if present
            base64_str = data_url
            if "base64," in data_url:
                base64_str = data_url.split("base64,", 1)[1]

            try:
                analysis_markdown = self.gemma3_converter.convert_image_base64_to_markdown(base64_str)
            except Exception as e:  # pragma: no cover - safety net
                logger.error(f"Error during Gemma3 image analysis: {e}")
                return match.group(0)

            if not analysis_markdown:
                return match.group(0)

            analysis_markdown = analysis_markdown.strip()

            # Build the injected section: image first, then analysis under it
            analysis_section_lines = [
                match.group(0),
                "",
                "### Image Analysis",
                "",
                analysis_markdown,
            ]
            return "\n".join(analysis_section_lines)

        # Apply replacement across the entire markdown content
        return image_pattern.sub(_analyze_and_replace, md_content)


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

                    # Post-process the no-header/footer markdown to insert Gemma3 image analysis
                    md_content_no_hf = self._add_image_analysis_to_markdown(md_content_no_hf)

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

        tasks = [
            {
                "origin_image": image,
                "prompt_mode": prompt_mode,
                "save_dir": save_dir,
                "save_name": filename,
                "source":"pdf",
                "page_idx": i,
            } for i, image in enumerate(images_origin)
        ]

        def _execute_task(task_args):
            return self._parse_single_image(**task_args)

        num_thread = min(total_pages, self.num_thread)
        print(f"Parsing PDF with {total_pages} pages using {num_thread} threads...")

        results = []
        with ThreadPool(num_thread) as pool:
            with tqdm(total=total_pages, desc="Processing PDF pages") as pbar:
                for result in pool.imap_unordered(_execute_task, tasks):
                    results.append(result)
                    pbar.update(1)

                    # Report progress: pages processed
                    pages_processed = len(results)
                    progress = 10 + int((pages_processed / total_pages) * 80)  # 10-90% for processing
                    if self.progress_callback:
                        self.progress_callback(
                            progress=progress,
                            message=f"Processing pages: {pages_processed}/{total_pages}"
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