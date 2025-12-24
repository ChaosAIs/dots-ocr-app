"""
Document service for file path resolution, markdown operations, and document management.
"""
import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List
from uuid import UUID

from sqlalchemy.orm import Session

from db.models import Document, ConvertStatus, IndexStatus
from db.document_repository import DocumentRepository

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Service for document-related operations including:
    - File path resolution (relative <-> absolute)
    - Markdown file operations
    - Document status management
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str
    ):
        """
        Initialize document service.

        Args:
            input_dir: Base directory for input files
            output_dir: Base directory for output files
        """
        self.input_dir = input_dir
        self.output_dir = output_dir

        # Ensure directories exist
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    def resolve_file_path(self, relative_or_absolute_path: str, base_dir: str = None) -> str:
        """
        Resolve a file path to absolute path.

        If the path is already absolute and exists, return it as-is.
        If the path is relative (e.g., "fyang/my_documents/file.png"),
        prepend the base_dir (defaults to INPUT_DIR).

        Args:
            relative_or_absolute_path: The path from database (relative or legacy absolute)
            base_dir: Base directory to prepend for relative paths (defaults to input_dir)

        Returns:
            Absolute path to the file
        """
        if base_dir is None:
            base_dir = self.input_dir

        # If path is already absolute and exists, use it
        if os.path.isabs(relative_or_absolute_path):
            if os.path.exists(relative_or_absolute_path):
                return relative_or_absolute_path
            # If absolute path doesn't exist, try extracting relative part
            # This handles legacy paths like "/home/fy/.../input/fyang/file.png"
            for known_base in [self.input_dir, self.output_dir]:
                if known_base in relative_or_absolute_path:
                    relative_part = relative_or_absolute_path.split(known_base + os.sep)[-1]
                    candidate = os.path.join(base_dir, relative_part)
                    if os.path.exists(candidate):
                        return candidate

        # Relative path - prepend base directory
        return os.path.join(base_dir, relative_or_absolute_path)

    def resolve_output_path(self, relative_or_absolute_path: str) -> str:
        """
        Resolve output path to absolute path.
        Uses output_dir as the base directory.
        """
        return self.resolve_file_path(relative_or_absolute_path, self.output_dir)

    def to_relative_path(self, absolute_path: str, base_dir: str = None) -> str:
        """
        Convert an absolute path to a relative path for database storage.

        Args:
            absolute_path: The absolute path to convert
            base_dir: Base directory to remove (defaults to input_dir, also tries output_dir)

        Returns:
            Relative path (e.g., "username/workspace/filename")
        """
        if not absolute_path:
            return absolute_path

        # If already relative, return as-is
        if not os.path.isabs(absolute_path):
            return absolute_path

        # Try to strip known base directories
        for known_base in [self.input_dir, self.output_dir]:
            if absolute_path.startswith(known_base + os.sep):
                return absolute_path[len(known_base) + 1:]  # +1 for the separator
            elif absolute_path.startswith(known_base):
                return absolute_path[len(known_base):].lstrip(os.sep)

        # If custom base_dir provided, try that too
        if base_dir:
            if absolute_path.startswith(base_dir + os.sep):
                return absolute_path[len(base_dir) + 1:]
            elif absolute_path.startswith(base_dir):
                return absolute_path[len(base_dir):].lstrip(os.sep)

        # Couldn't convert - return original (should log a warning)
        logger.warning(f"Could not convert absolute path to relative: {absolute_path}")
        return absolute_path

    def to_relative_output_path(self, absolute_path: str) -> str:
        """
        Convert an absolute output path to a relative path for database storage.
        """
        return self.to_relative_path(absolute_path, self.output_dir)

    def check_markdown_exists_at_path(self, output_dir: str) -> Tuple[bool, Optional[str], bool, int]:
        """
        Check if markdown file exists at a specific output directory path.
        Handles both single markdown files and multi-page markdown files.

        Args:
            output_dir: Full path to the output directory

        Returns:
            (markdown_exists, markdown_path, is_multipage, converted_pages)
        """
        if not os.path.exists(output_dir):
            return False, None, False, 0

        base_name = os.path.basename(output_dir)

        # Check for single markdown file (for single-page documents, images, Word, Excel, TXT)
        markdown_path_nohf = os.path.join(output_dir, f"{base_name}_nohf.md")
        if os.path.exists(markdown_path_nohf):
            return True, markdown_path_nohf, False, 1

        # Check for multi-page markdown files (for PDFs with multiple pages)
        page_files = []
        for file in os.listdir(output_dir):
            if file.endswith("_nohf.md") and "_page_" in file:
                page_files.append(file)

        if page_files:
            # Sort by page number
            page_files.sort(key=lambda x: int(x.split("_page_")[1].split("_")[0]))
            return True, os.path.join(output_dir, page_files[0]), True, len(page_files)

        return False, None, False, 0

    def check_markdown_exists(self, file_name_without_ext: str) -> Tuple[bool, Optional[str], bool, int]:
        """
        Check if markdown file exists for a document (legacy function using default output_dir).

        Returns:
            (markdown_exists, markdown_path, is_multipage, converted_pages)
        """
        output_dir = os.path.join(self.output_dir, file_name_without_ext)
        return self.check_markdown_exists_at_path(output_dir)

    def get_pdf_page_count(self, file_path: str) -> int:
        """
        Get the total number of pages in a PDF file.

        Returns:
            Total page count, or 0 if not a PDF or error occurs
        """
        try:
            if not file_path.lower().endswith('.pdf'):
                return 0

            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            page_count = len(doc)
            doc.close()
            return page_count
        except Exception as e:
            logger.error(f"Error getting PDF page count for {file_path}: {str(e)}")
            return 0

    def get_markdown_content(
        self,
        filename: str,
        page_no: Optional[int] = None,
        db: Session = None
    ) -> Tuple[str, Optional[int]]:
        """
        Get the markdown content of a converted document.

        Args:
            filename: The name of the file (without extension)
            page_no: Optional page number for multi-page documents
            db: Database session for looking up document paths

        Returns:
            Tuple of (content, page_no)
        """
        # Determine output directory
        output_dir = None
        if db:
            repo = DocumentRepository(db)
            doc = repo.get_by_filename(filename)
            if not doc:
                # Try common extensions
                for ext in ['.pdf', '.docx', '.xlsx', '.png', '.jpg', '.jpeg']:
                    doc = repo.get_by_filename(filename + ext)
                    if doc:
                        break

            if doc and doc.output_path:
                output_dir = self.resolve_output_path(doc.output_path)
            elif doc and doc.file_path:
                rel_dir = os.path.dirname(doc.file_path)
                output_dir = os.path.join(self.output_dir, rel_dir, filename) if rel_dir else os.path.join(self.output_dir, filename)

        # Fallback to legacy path
        if not output_dir:
            output_dir = os.path.join(self.output_dir, filename)

        # Determine which markdown file to read
        if page_no is not None:
            markdown_path = os.path.join(output_dir, f"{filename}_page_{page_no}_nohf.md")
        else:
            markdown_path = os.path.join(output_dir, f"{filename}_nohf.md")

        if not os.path.exists(markdown_path):
            raise FileNotFoundError(f"Markdown file not found for: {filename}")

        # Read and return the markdown content
        with open(markdown_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Normalize any single-line tables in the content
        from dots_ocr_service.utils.format_transformer import normalize_markdown_table, is_markdown_table

        lines = content.split('\n')
        normalized_lines = []
        for line in lines:
            if is_markdown_table(line):
                normalized_table = normalize_markdown_table(line)
                normalized_lines.append(normalized_table)
            else:
                normalized_lines.append(line)

        content = '\n'.join(normalized_lines)
        return content, page_no

    def update_markdown_content(
        self,
        filename: str,
        content: str,
        page_no: Optional[int] = None
    ) -> bool:
        """
        Update the markdown content of a converted document.

        Args:
            filename: The name of the file (without extension)
            content: New markdown content
            page_no: Optional page number for multi-page documents

        Returns:
            True if successful
        """
        # Determine which markdown file to write
        if page_no is not None:
            markdown_path = os.path.join(self.output_dir, filename, f"{filename}_page_{page_no}_nohf.md")
        else:
            markdown_path = os.path.join(self.output_dir, filename, f"{filename}_nohf.md")

        if not os.path.exists(markdown_path):
            raise FileNotFoundError(f"Markdown file not found for: {filename}")

        # Write the updated content
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(content)

        return True

    def list_markdown_files(
        self,
        filename: str,
        db: Session = None
    ) -> List[dict]:
        """
        List all markdown files associated with a document.

        Args:
            filename: The name of the file (without extension)
            db: Database session for looking up document paths

        Returns:
            List of markdown file info dicts
        """
        # Look up document in database to get the correct output path
        output_dir = None
        if db:
            repo = DocumentRepository(db)
            doc = repo.get_by_filename(filename)
            if not doc:
                for ext in ['.pdf', '.docx', '.xlsx', '.png', '.jpg', '.jpeg']:
                    doc = repo.get_by_filename(filename + ext)
                    if doc:
                        break

            if doc and doc.output_path:
                output_dir = self.resolve_output_path(doc.output_path)
            elif doc and doc.file_path:
                rel_dir = os.path.dirname(doc.file_path)
                output_dir = os.path.join(self.output_dir, rel_dir, filename) if rel_dir else os.path.join(self.output_dir, filename)

        if not output_dir:
            output_dir = os.path.join(self.output_dir, filename)

        if not os.path.exists(output_dir):
            raise FileNotFoundError(f"No output directory for: {filename}")

        markdown_files = []

        # Check for single markdown file
        single_md_path = os.path.join(output_dir, f"{filename}_nohf.md")
        if os.path.exists(single_md_path):
            markdown_files.append({
                "filename": f"{filename}_nohf.md",
                "path": single_md_path,
                "page_no": None,
                "is_multipage": False,
            })

        # Check for multi-page markdown files
        page_files = []
        for file in os.listdir(output_dir):
            if file.endswith("_nohf.md") and "_page_" in file:
                page_no = int(file.split("_page_")[1].split("_")[0])
                page_files.append({
                    "filename": file,
                    "path": os.path.join(output_dir, file),
                    "page_no": page_no,
                    "is_multipage": True,
                })

        # Sort by page number
        page_files.sort(key=lambda x: x["page_no"])
        markdown_files.extend(page_files)

        return markdown_files

    def resize_image_if_needed(self, file_path: str, max_pixels: int = None) -> dict:
        """
        Resize image if it exceeds maximum pixel dimensions.

        Args:
            file_path: Path to the image file
            max_pixels: Maximum number of pixels allowed

        Returns:
            dict with resize info
        """
        from PIL import Image
        import math

        # Get max_pixels from environment variable if not specified
        if max_pixels is None:
            max_pixels_env = os.getenv('MAX_PIXELS', '8000000')
            if max_pixels_env.lower() == 'none':
                max_pixels = 11289600
            else:
                max_pixels = int(max_pixels_env)

        # Apply a safety margin (90% of max_pixels)
        safe_max_pixels = int(max_pixels * 0.9)

        # Check if file is an image
        file_ext = os.path.splitext(file_path)[1].lower()
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}

        if file_ext not in image_extensions:
            return {
                "resized": False,
                "original_size": None,
                "new_size": None,
                "message": "Not an image file"
            }

        try:
            with Image.open(file_path) as img:
                original_width, original_height = img.size
                original_pixels = original_width * original_height

                if original_pixels <= safe_max_pixels:
                    return {
                        "resized": False,
                        "original_size": (original_width, original_height),
                        "new_size": None,
                        "message": f"Image size OK ({original_width}x{original_height} = {original_pixels:,} pixels)"
                    }

                # Calculate new dimensions maintaining aspect ratio
                scale_factor = math.sqrt(safe_max_pixels / original_pixels)
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)

                # Ensure dimensions are divisible by 28 (IMAGE_FACTOR)
                new_width = (new_width // 28) * 28
                new_height = (new_height // 28) * 28

                # Ensure minimum size
                if new_width < 28:
                    new_width = 28
                if new_height < 28:
                    new_height = 28

                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = rgb_img
                elif img.mode != 'RGB':
                    img = img.convert('RGB')

                # Resize image using high-quality LANCZOS resampling
                resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Save resized image
                if file_ext in {'.jpg', '.jpeg'}:
                    resized_img.save(file_path, 'JPEG', quality=90, optimize=True)
                elif file_ext == '.png':
                    new_file_path = os.path.splitext(file_path)[0] + '.jpg'
                    resized_img.save(new_file_path, 'JPEG', quality=90, optimize=True)
                    if new_file_path != file_path:
                        os.remove(file_path)
                        os.rename(new_file_path, file_path)
                else:
                    resized_img.save(file_path, 'JPEG', quality=90, optimize=True)

                new_pixels = new_width * new_height
                return {
                    "resized": True,
                    "original_size": (original_width, original_height),
                    "new_size": (new_width, new_height),
                    "message": f"Image resized from {original_width}x{original_height} to {new_width}x{new_height}"
                }

        except Exception as e:
            logger.error(f"Error resizing image {file_path}: {e}")
            return {
                "resized": False,
                "original_size": None,
                "new_size": None,
                "message": f"Error resizing image: {str(e)}"
            }

    def get_uncompleted_pages(self, filename: str) -> List[dict]:
        """
        Get list of uncompleted pages for a document.

        An uncompleted page is one that has a JPG file but no corresponding _nohf.md file.

        Args:
            filename: The document filename (without extension)

        Returns:
            List of dicts with page_no, jpg_file, expected_md_file
        """
        file_name_without_ext = os.path.splitext(filename)[0]
        output_dir = os.path.join(self.output_dir, file_name_without_ext)

        if not os.path.exists(output_dir):
            raise FileNotFoundError(f"Output directory not found for: {filename}")

        uncompleted_pages = []

        for file in os.listdir(output_dir):
            if file.endswith(".jpg") and "_page_" in file:
                try:
                    page_part = file.split("_page_")[1]
                    page_no = int(page_part.split(".")[0])

                    md_filename = f"{file_name_without_ext}_page_{page_no}_nohf.md"
                    md_path = os.path.join(output_dir, md_filename)

                    if not os.path.exists(md_path):
                        uncompleted_pages.append({
                            "page_no": page_no,
                            "jpg_file": file,
                            "expected_md_file": md_filename
                        })
                except (ValueError, IndexError):
                    continue

        uncompleted_pages.sort(key=lambda x: x["page_no"])
        return uncompleted_pages
