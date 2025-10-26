#!/usr/bin/env python3
"""
Text Chunking Utilities

This module provides utilities for chunking large text and CSV content to handle
AI service token limitations while maintaining content coherence and structure.
"""

import re
import csv
import io
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ChunkInfo:
    """Information about a content chunk."""
    content: str
    chunk_index: int
    total_chunks: int
    content_type: str  # 'text', 'csv', 'header'
    estimated_tokens: int
    metadata: Dict[str, Any] = None


class TokenEstimator:
    """Estimates token count for different AI services."""
    
    # Rough token estimation: 1 token ≈ 4 characters for most models
    CHARS_PER_TOKEN = 4
    
    # Conservative token limits for different services (leaving room for prompts and responses)
    # These limits account for both context windows and rate limits
    TOKEN_LIMITS = {
        'openai': {
            'gpt-4o': 25000,   # Conservative limit considering rate limits and response space
            'gpt-4': 7000,     # 8k context
            'gpt-3.5-turbo': 15000  # 16k context
        },
        'ollama': {
            'default': 3000,   # Conservative limit for local models
            'llava:latest': 3000,
            'llama2': 3000
        }
    }
    
    @classmethod
    def estimate_tokens(cls, text: str) -> int:
        """
        Estimate the number of tokens in a text string.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Simple estimation: character count / 4
        # This is conservative and works reasonably well for most content
        return max(1, len(text) // cls.CHARS_PER_TOKEN)
    
    @classmethod
    def get_max_tokens(cls, service_type: str, model_name: str = None) -> int:
        """
        Get maximum token limit for a service and model.
        
        Args:
            service_type: Type of AI service ('openai', 'ollama')
            model_name: Specific model name
            
        Returns:
            Maximum token limit for content
        """
        service_limits = cls.TOKEN_LIMITS.get(service_type.lower(), {})
        
        if model_name and model_name in service_limits:
            return service_limits[model_name]
        elif 'default' in service_limits:
            return service_limits['default']
        else:
            # Fallback to most conservative limit
            return 3000


class TextChunker:
    """Handles chunking of plain text content."""
    
    def __init__(self, max_tokens: int = 3000):
        """
        Initialize text chunker.
        
        Args:
            max_tokens: Maximum tokens per chunk
        """
        self.max_tokens = max_tokens
        self.max_chars = max_tokens * TokenEstimator.CHARS_PER_TOKEN
    
    def chunk_text(self, text: str) -> List[ChunkInfo]:
        """
        Split text into chunks at natural boundaries.
        
        Args:
            text: Text content to chunk
            
        Returns:
            List of ChunkInfo objects
        """
        if TokenEstimator.estimate_tokens(text) <= self.max_tokens:
            return [ChunkInfo(
                content=text,
                chunk_index=0,
                total_chunks=1,
                content_type='text',
                estimated_tokens=TokenEstimator.estimate_tokens(text)
            )]
        
        chunks = []
        
        # Try to split at paragraph boundaries first
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed the limit
            test_chunk = current_chunk + ("\n\n" if current_chunk else "") + paragraph
            
            if TokenEstimator.estimate_tokens(test_chunk) <= self.max_tokens:
                current_chunk = test_chunk
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # If single paragraph is too large, split it further
                if TokenEstimator.estimate_tokens(paragraph) > self.max_tokens:
                    sub_chunks = self._split_large_paragraph(paragraph)
                    chunks.extend(sub_chunks)
                else:
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # Convert to ChunkInfo objects
        total_chunks = len(chunks)
        return [
            ChunkInfo(
                content=chunk,
                chunk_index=i,
                total_chunks=total_chunks,
                content_type='text',
                estimated_tokens=TokenEstimator.estimate_tokens(chunk)
            )
            for i, chunk in enumerate(chunks)
        ]
    
    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        """Split a large paragraph at sentence boundaries."""
        # Split at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if TokenEstimator.estimate_tokens(test_chunk) <= self.max_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = sentence
                else:
                    # Single sentence is too large, split by character limit
                    chunks.extend(self._split_by_chars(sentence))
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_chars(self, text: str) -> List[str]:
        """Split text by character limit as last resort."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.max_chars
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to find a good break point (space, newline)
            break_point = end
            for i in range(end - 100, end):
                if i > start and text[i] in ' \n\t':
                    break_point = i
                    break
            
            chunks.append(text[start:break_point])
            start = break_point
        
        return chunks


class CSVChunker:
    """Handles chunking of CSV content while preserving table structure."""
    
    def __init__(self, max_tokens: int = 3000):
        """
        Initialize CSV chunker.
        
        Args:
            max_tokens: Maximum tokens per chunk
        """
        self.max_tokens = max_tokens
    
    def chunk_csv(self, csv_content: str, delimiter: str = ',') -> List[ChunkInfo]:
        """
        Split CSV content into chunks while preserving headers.
        
        Args:
            csv_content: CSV content as string
            delimiter: CSV delimiter character
            
        Returns:
            List of ChunkInfo objects
        """
        try:
            # Parse CSV content
            csv_reader = csv.reader(io.StringIO(csv_content), delimiter=delimiter)
            rows = list(csv_reader)
            
            if not rows:
                return [ChunkInfo(
                    content=csv_content,
                    chunk_index=0,
                    total_chunks=1,
                    content_type='csv',
                    estimated_tokens=TokenEstimator.estimate_tokens(csv_content)
                )]
            
            headers = rows[0]
            data_rows = rows[1:]
            
            # Estimate tokens for header
            header_line = delimiter.join(headers)
            header_tokens = TokenEstimator.estimate_tokens(header_line)
            
            # Calculate how many rows we can fit per chunk
            available_tokens = self.max_tokens - header_tokens - 100  # Leave room for formatting
            
            if available_tokens <= 0:
                # Headers alone exceed limit, return as single chunk
                return [ChunkInfo(
                    content=csv_content,
                    chunk_index=0,
                    total_chunks=1,
                    content_type='csv',
                    estimated_tokens=TokenEstimator.estimate_tokens(csv_content),
                    metadata={'warning': 'Headers exceed token limit'}
                )]
            
            # Estimate average tokens per row
            if data_rows:
                sample_rows = data_rows[:min(10, len(data_rows))]
                avg_row_tokens = sum(
                    TokenEstimator.estimate_tokens(delimiter.join(str(cell) for cell in row))
                    for row in sample_rows
                ) / len(sample_rows)
                
                rows_per_chunk = max(1, int(available_tokens / avg_row_tokens))
            else:
                rows_per_chunk = len(data_rows)
            
            # Create chunks
            chunks = []
            total_chunks = max(1, (len(data_rows) + rows_per_chunk - 1) // rows_per_chunk)
            
            for i in range(0, len(data_rows), rows_per_chunk):
                chunk_rows = [headers] + data_rows[i:i + rows_per_chunk]
                chunk_content = '\n'.join(delimiter.join(str(cell) for cell in row) for row in chunk_rows)
                
                chunks.append(ChunkInfo(
                    content=chunk_content,
                    chunk_index=len(chunks),
                    total_chunks=total_chunks,
                    content_type='csv',
                    estimated_tokens=TokenEstimator.estimate_tokens(chunk_content),
                    metadata={
                        'headers': headers,
                        'row_start': i + 1,  # +1 because we skip header
                        'row_end': min(i + rows_per_chunk, len(data_rows)),
                        'total_rows': len(data_rows)
                    }
                ))
            
            return chunks
            
        except Exception as e:
            # If CSV parsing fails, treat as plain text
            text_chunker = TextChunker(self.max_tokens)
            text_chunks = text_chunker.chunk_text(csv_content)
            
            # Convert to CSV chunks with error metadata
            return [
                ChunkInfo(
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    total_chunks=chunk.total_chunks,
                    content_type='csv',
                    estimated_tokens=chunk.estimated_tokens,
                    metadata={'parsing_error': str(e), 'treated_as_text': True}
                )
                for chunk in text_chunks
            ]


def create_chunker(content_type: str, max_tokens: int = 3000) -> object:
    """
    Factory function to create appropriate chunker.
    
    Args:
        content_type: Type of content ('text' or 'csv')
        max_tokens: Maximum tokens per chunk
        
    Returns:
        Appropriate chunker instance
    """
    if content_type.lower() == 'csv':
        return CSVChunker(max_tokens)
    else:
        return TextChunker(max_tokens)
