#!/usr/bin/env python3
"""
Document Converter Manager

This module orchestrates different document converters and provides a unified interface
for converting various document types to Markdown format.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from .base_converter import BaseDocumentConverter
from .word_converter import WordDocumentConverter
from .pdf_converter import PDFDocumentConverter
from .excel_converter import ExcelDocumentConverter
from .image_converter import ImageDocumentConverter
from .plain_text_converter import PlainTextConverter


class DocumentConverterManager:
    """Manages multiple document converters and provides unified conversion interface."""
    
    def __init__(self, input_folder: str = "input", output_folder: str = "output", add_section_numbers: bool = True):
        """
        Initialize the document converter manager.

        Args:
            input_folder: Path to folder containing documents to convert
            output_folder: Path to folder where Markdown files will be saved
            add_section_numbers: Whether to add automatic section numbering
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.add_section_numbers = add_section_numbers

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize converters
        self.converters: List[BaseDocumentConverter] = [
            WordDocumentConverter(),
            PDFDocumentConverter(),
            ExcelDocumentConverter(),
            ImageDocumentConverter(),
            PlainTextConverter()
        ]
        
        # Create folders if they don't exist
        self._create_folders()
        
        # Log available converters
        self._log_available_converters()
    
    def _create_folders(self) -> None:
        """Create input and output folders if they don't exist."""
        self.input_folder.mkdir(exist_ok=True)
        self.output_folder.mkdir(exist_ok=True)
        self.logger.info(f"Input folder: {self.input_folder.absolute()}")
        self.logger.info(f"Output folder: {self.output_folder.absolute()}")
    
    def _log_available_converters(self) -> None:
        """Log information about available converters."""
        self.logger.info("Available converters:")
        for converter in self.converters:
            info = converter.get_converter_info()
            extensions = ", ".join(info["supported_extensions"])
            self.logger.info(f"  - {info['name']}: {extensions}")
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get all supported file extensions across all converters.
        
        Returns:
            List of all supported file extensions
        """
        extensions = []
        for converter in self.converters:
            extensions.extend(converter.get_supported_extensions())
        return list(set(extensions))  # Remove duplicates
    
    def find_converter_for_file(self, file_path: Path) -> Optional[BaseDocumentConverter]:
        """
        Find the appropriate converter for a given file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Converter that can handle the file, or None if no converter found
        """
        for converter in self.converters:
            if converter.can_convert(file_path):
                return converter
        return None
    
    def get_convertible_files(self) -> List[Path]:
        """
        Get all files in the input folder that can be converted.
        
        Returns:
            List of file paths that can be converted
        """
        convertible_files = []
        supported_extensions = self.get_supported_extensions()
        
        for file_path in self.input_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                convertible_files.append(file_path)
        
        return convertible_files
    
    def convert_file(self, input_file: Path) -> bool:
        """
        Convert a single file to Markdown.
        
        Args:
            input_file: Path to the input file
            
        Returns:
            True if conversion successful, False otherwise
        """
        # Find appropriate converter
        converter = self.find_converter_for_file(input_file)
        if not converter:
            self.logger.error(f"No converter found for file type: {input_file.suffix}")
            return False
        
        # Create output filename
        output_filename = input_file.stem + ".md"
        output_path = self.output_folder / output_filename
        
        # Convert the file
        return converter.convert_file(input_file, output_path)
    
    def convert_all(self) -> Dict[str, Any]:
        """
        Convert all supported documents in the input folder.
        
        Returns:
            Dictionary with conversion results and statistics
        """
        convertible_files = self.get_convertible_files()
        
        if not convertible_files:
            self.logger.warning(f"No convertible documents found in {self.input_folder}")
            return {
                "total_files": 0,
                "successful_conversions": 0,
                "failed_conversions": 0,
                "results": []
            }
        
        self.logger.info(f"Found {len(convertible_files)} convertible document(s)")
        
        successful_conversions = 0
        failed_conversions = 0
        results = []
        
        for file_path in convertible_files:
            converter = self.find_converter_for_file(file_path)
            converter_name = converter.__class__.__name__ if converter else "Unknown"
            
            if self.convert_file(file_path):
                successful_conversions += 1
                results.append({
                    "file": file_path.name,
                    "status": "success",
                    "converter": converter_name
                })
            else:
                failed_conversions += 1
                results.append({
                    "file": file_path.name,
                    "status": "failed",
                    "converter": converter_name
                })
        
        summary = {
            "total_files": len(convertible_files),
            "successful_conversions": successful_conversions,
            "failed_conversions": failed_conversions,
            "results": results
        }
        
        self.logger.info(f"Conversion complete: {successful_conversions} successful, {failed_conversions} failed")
        return summary
    
    def get_conversion_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about available converters and supported file types.
        
        Returns:
            Dictionary with converter statistics
        """
        stats = {
            "total_converters": len(self.converters),
            "supported_extensions": self.get_supported_extensions(),
            "converters": []
        }
        
        for converter in self.converters:
            info = converter.get_converter_info()
            stats["converters"].append(info)
        
        return stats
    
    def add_converter(self, converter: BaseDocumentConverter) -> None:
        """
        Add a new converter to the manager.
        
        Args:
            converter: Converter instance to add
        """
        if not isinstance(converter, BaseDocumentConverter):
            raise ValueError("Converter must inherit from BaseDocumentConverter")
        
        self.converters.append(converter)
        self.logger.info(f"Added converter: {converter.__class__.__name__}")
    
    def remove_converter(self, converter_class: type) -> bool:
        """
        Remove a converter by class type.
        
        Args:
            converter_class: Class of the converter to remove
            
        Returns:
            True if converter was removed, False if not found
        """
        for i, converter in enumerate(self.converters):
            if isinstance(converter, converter_class):
                removed_converter = self.converters.pop(i)
                self.logger.info(f"Removed converter: {removed_converter.__class__.__name__}")
                return True
        
        return False
