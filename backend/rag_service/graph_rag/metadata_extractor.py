"""
Document metadata extraction using hierarchical summarization.

This module extracts structured metadata from documents by:
1. Level 1: Summarizing chunks in batches
2. Level 2: Creating a meta-summary from batch summaries
3. Level 3: Extracting structured metadata from the meta-summary
"""
import logging
import json
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..llm_service import get_llm_service
from .prompts import (
    METADATA_BATCH_SUMMARY_PROMPT,
    METADATA_META_SUMMARY_PROMPT,
    METADATA_STRUCTURED_EXTRACTION_PROMPT,
)

logger = logging.getLogger(__name__)


class HierarchicalMetadataExtractor:
    """Extract document metadata using hierarchical summarization."""

    def __init__(self):
        llm_service = get_llm_service()
        self.llm = llm_service.get_chat_model(temperature=0.2, num_ctx=8192)
        self.str_parser = StrOutputParser()
    
    async def extract_metadata(
        self,
        chunks: List[Dict[str, Any]],
        source_name: str,
        batch_size: int = 10,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Extract document metadata using hierarchical summarization.
        
        Args:
            chunks: List of chunk dicts with 'page_content' and 'metadata'
            source_name: Document source name (e.g., "Felix Yang- Resume - 2025")
            batch_size: Number of chunks per batch for level-1 summarization
            progress_callback: Optional callback for progress updates
            
        Returns:
            Metadata dict with extraction results
        """
        start_time = datetime.now()
        
        try:
            # Filter out empty chunks
            valid_chunks = [c for c in chunks if c.get("page_content", "").strip()]
            
            if not valid_chunks:
                logger.warning(f"No valid chunks for metadata extraction: {source_name}")
                return self._create_empty_metadata(source_name)
            
            logger.info(f"[Metadata] Starting extraction for {source_name}: {len(valid_chunks)} chunks")
            
            # Level 1: Batch summarization
            if progress_callback:
                progress_callback("Summarizing document sections...")
            
            level1_summaries = await self._level1_summarize_batches(
                valid_chunks, batch_size, progress_callback
            )
            
            # Level 2: Meta-summarization
            if progress_callback:
                progress_callback("Creating document overview...")
            
            meta_summary = await self._level2_meta_summarize(level1_summaries)
            
            # Level 3: Structured extraction
            if progress_callback:
                progress_callback("Extracting structured metadata...")
            
            structured_metadata = await self._level3_extract_structured_metadata(
                meta_summary, source_name, valid_chunks
            )
            
            # Build final metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            
            metadata = {
                "extraction_version": "1.0",
                "extracted_at": datetime.now().isoformat(),
                "extraction_method": "hierarchical_summarization",
                **structured_metadata,
                "hierarchical_summary": {
                    "level1_summaries": level1_summaries,
                    "meta_summary": meta_summary,
                },
                "processing_stats": {
                    "total_chunks": len(chunks),
                    "processed_chunks": len(valid_chunks),
                    "llm_calls": len(level1_summaries) + 2,  # batches + meta + structured
                    "processing_time_seconds": round(processing_time, 2),
                },
            }
            
            logger.info(
                f"[Metadata] Extraction complete for {source_name}: "
                f"{structured_metadata.get('document_type', 'unknown')} | "
                f"{processing_time:.1f}s"
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"[Metadata] Extraction failed for {source_name}: {e}", exc_info=True)
            return self._create_error_metadata(source_name, str(e))
    
    async def _level1_summarize_batches(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> List[str]:
        """Level 1: Summarize chunks in batches."""
        summaries = []
        num_batches = (len(chunks) + batch_size - 1) // batch_size
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            # Combine batch chunks into single text
            batch_text = "\n\n---\n\n".join([
                c.get("page_content", "") for c in batch
            ])

            # Truncate if too long (max ~4000 chars per batch)
            if len(batch_text) > 4000:
                batch_text = batch_text[:4000] + "\n...[truncated]"

            if progress_callback:
                progress_callback(f"Summarizing section {batch_num}/{num_batches}...")

            # Create prompt and invoke LLM
            prompt = ChatPromptTemplate.from_template(METADATA_BATCH_SUMMARY_PROMPT)
            chain = prompt | self.llm | self.str_parser

            try:
                summary = await chain.ainvoke({"chunk_text": batch_text})
                summaries.append(summary.strip())
                logger.debug(f"[Metadata] Batch {batch_num}/{num_batches} summarized")
            except Exception as e:
                logger.error(f"[Metadata] Failed to summarize batch {batch_num}: {e}")
                summaries.append(f"[Summary failed for batch {batch_num}]")

        return summaries

    async def _level2_meta_summarize(self, level1_summaries: List[str]) -> str:
        """Level 2: Create meta-summary from batch summaries."""
        # Combine all level-1 summaries
        combined_summaries = "\n\n".join([
            f"Section {i+1}: {summary}"
            for i, summary in enumerate(level1_summaries)
        ])

        # Create prompt and invoke LLM
        prompt = ChatPromptTemplate.from_template(METADATA_META_SUMMARY_PROMPT)
        chain = prompt | self.llm | self.str_parser

        try:
            meta_summary = await chain.ainvoke({"summaries": combined_summaries})
            logger.debug(f"[Metadata] Meta-summary created ({len(meta_summary)} chars)")
            return meta_summary.strip()
        except Exception as e:
            logger.error(f"[Metadata] Failed to create meta-summary: {e}")
            return "[Meta-summary generation failed]"

    async def _level3_extract_structured_metadata(
        self,
        meta_summary: str,
        source_name: str,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Level 3: Extract structured metadata from meta-summary."""
        # Get first chunk for additional context
        first_chunk = chunks[0].get("page_content", "")[:1000] if chunks else ""

        # Create prompt and invoke LLM
        prompt = ChatPromptTemplate.from_template(METADATA_STRUCTURED_EXTRACTION_PROMPT)
        chain = prompt | self.llm | self.str_parser

        try:
            response = await chain.ainvoke({
                "source_name": source_name,
                "meta_summary": meta_summary,
                "first_chunk": first_chunk,
            })

            # Parse JSON response
            # Remove markdown code blocks if present
            response_clean = response.strip()
            if response_clean.startswith("```"):
                # Remove ```json or ``` prefix
                lines = response_clean.split("\n")
                response_clean = "\n".join(lines[1:-1]) if len(lines) > 2 else response_clean

            metadata = json.loads(response_clean)
            logger.debug(f"[Metadata] Structured metadata extracted: {metadata.get('document_type')}")
            return metadata

        except json.JSONDecodeError as e:
            logger.error(f"[Metadata] Failed to parse JSON response: {e}\nResponse: {response[:200]}")
            return self._create_fallback_metadata(source_name, meta_summary)
        except Exception as e:
            logger.error(f"[Metadata] Failed to extract structured metadata: {e}")
            return self._create_fallback_metadata(source_name, meta_summary)

    def _create_empty_metadata(self, source_name: str) -> Dict[str, Any]:
        """Create empty metadata for documents with no valid chunks."""
        return {
            "extraction_version": "1.0",
            "extracted_at": datetime.now().isoformat(),
            "extraction_method": "hierarchical_summarization",
            "document_type": "unknown",
            "subject_name": None,
            "subject_type": None,
            "title": source_name,
            "author": None,
            "summary": "No content available for analysis",
            "topics": [],
            "key_entities": [],
            "confidence": 0.0,
            "processing_stats": {
                "total_chunks": 0,
                "processed_chunks": 0,
                "llm_calls": 0,
            },
        }

    def _create_error_metadata(self, source_name: str, error: str) -> Dict[str, Any]:
        """Create error metadata when extraction fails."""
        return {
            "extraction_version": "1.0",
            "extracted_at": datetime.now().isoformat(),
            "extraction_method": "hierarchical_summarization",
            "document_type": "unknown",
            "subject_name": None,
            "subject_type": None,
            "title": source_name,
            "author": None,
            "summary": "Metadata extraction failed",
            "topics": [],
            "key_entities": [],
            "confidence": 0.0,
            "error": error,
        }

    def _create_fallback_metadata(self, source_name: str, meta_summary: str) -> Dict[str, Any]:
        """Create fallback metadata when structured extraction fails."""
        return {
            "document_type": "other",
            "subject_name": None,
            "subject_type": None,
            "title": source_name,
            "author": None,
            "summary": meta_summary[:200] if meta_summary else "Summary unavailable",
            "topics": [],
            "key_entities": [],
            "confidence": 0.3,
        }

