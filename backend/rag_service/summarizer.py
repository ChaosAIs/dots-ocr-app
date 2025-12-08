"""
LLM-based summarization for RAG chunks and documents.
Uses Ollama with a fast model for generating summaries.
"""

import os
import logging
from typing import List, Optional
from dataclasses import dataclass

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
# Use OLLAMA_QUERY_MODEL for summarization (same model used for query analysis)
# Falls back to OLLAMA_MODEL if OLLAMA_QUERY_MODEL is not set
OLLAMA_SUMMARY_MODEL = os.getenv("OLLAMA_QUERY_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:latest"))
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"


@dataclass
class ChunkSummary:
    """Summary of a single chunk."""
    chunk_index: int
    summary: str
    heading_path: str = ""


CHUNK_SUMMARY_PROMPT = """Summarize the following text in no more than {max_words} words.
Focus on the key information, main topics, and important concepts.
Output ONLY the summary, nothing else.

Text:
{content}"""

FILE_SUMMARY_PROMPT = """Based on the following chunk summaries from a document, generate a comprehensive file summary.

Document filename: {filename}

Chunk summaries:
{chunk_summaries}

Generate a summary that includes:
1. Main topic/title of the document
2. Category of the document (e.g., Technical Documentation, Research Paper, Legal Document, Financial Report, User Manual, Policy Document, Meeting Notes, Tutorial, Reference Guide, etc.)
3. Purpose of the document
4. Key themes and sections covered
5. Brief description (2-3 sentences)

Format your response as:
Title: [document title or main topic]
Category: [document category]
Purpose: [purpose of the document]
Topics: [comma-separated list of main topics]
Description: [2-3 sentence description]

Output ONLY the formatted summary, nothing else."""


def _create_summary_llm():
    """Create the Ollama LLM instance for summarization."""
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_SUMMARY_MODEL,
        temperature=0.1,  # Low temperature for consistent summaries
        num_ctx=4096,
        num_predict=256,  # Limit output tokens
    )


def summarize_chunk(content: str, max_words: int = 200) -> str:
    """
    Summarize a single chunk of text.

    Args:
        content: The text content to summarize.
        max_words: Maximum words for the summary.

    Returns:
        Summary string.
    """
    if not content or len(content.strip()) < 50:
        # Skip very short content
        return content.strip()[:200] if content else ""

    # Adjust max_words based on content length
    content_words = len(content.split())
    if content_words < max_words:
        # Content is already short, just return it trimmed
        return content.strip()[:500]

    try:
        llm = _create_summary_llm()
        prompt = CHUNK_SUMMARY_PROMPT.format(
            max_words=max_words,
            content=content[:4000]  # Limit input size
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        summary = response.content.strip()

        # Clean up any thinking tags if present
        if "</think>" in summary:
            summary = summary.split("</think>")[-1].strip()

        return summary[:500]  # Safety limit

    except Exception as e:
        logger.warning(f"Chunk summarization failed: {e}")
        # Fallback: return first 200 chars
        return content.strip()[:200]


def generate_file_summary(
    filename: str,
    chunk_summaries: List[ChunkSummary]
) -> str:
    """
    Generate a comprehensive file summary from chunk summaries.

    Args:
        filename: The document filename.
        chunk_summaries: List of ChunkSummary objects.

    Returns:
        Formatted file summary string.
    """
    if not chunk_summaries:
        return f"Document: {filename}\nNo content available for summarization."

    try:
        llm = _create_summary_llm()

        # Format chunk summaries for the prompt
        formatted_summaries = []
        for cs in chunk_summaries:
            if cs.heading_path:
                formatted_summaries.append(f"[{cs.heading_path}]: {cs.summary}")
            else:
                formatted_summaries.append(f"[Chunk {cs.chunk_index}]: {cs.summary}")

        prompt = FILE_SUMMARY_PROMPT.format(
            filename=filename,
            chunk_summaries="\n".join(formatted_summaries)
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        summary = response.content.strip()

        # Clean up any thinking tags if present
        if "</think>" in summary:
            summary = summary.split("</think>")[-1].strip()

        return summary

    except Exception as e:
        logger.error(f"File summary generation failed for {filename}: {e}")
        # Fallback: combine first few chunk summaries
        fallback_parts = [f"Document: {filename}"]
        for cs in chunk_summaries[:5]:
            fallback_parts.append(cs.summary)
        return "\n".join(fallback_parts)

