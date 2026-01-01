"""
Timing Metrics - Centralized timing measurement for RAG pipeline.

Provides comprehensive timing breakdown from query input to final output,
tracking each step in the retrieval and generation pipeline.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading

logger = logging.getLogger(__name__)

# Global timing metrics storage (shared across threads for async/tool execution)
# Uses a dictionary keyed by a context ID to support multiple concurrent requests
_global_metrics: Dict[str, "TimingMetrics"] = {}
_global_metrics_lock = threading.Lock()

# Current active metrics ID (set per-request)
_current_metrics_id: Optional[str] = None


@dataclass
class TimingStep:
    """A single timing step in the pipeline."""
    name: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(self):
        """Mark this step as complete."""
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000


@dataclass
class TimingMetrics:
    """
    Collects and reports timing metrics for a complete query pipeline.

    Usage:
        metrics = TimingMetrics("query-123")

        with metrics.measure("preprocessing"):
            # do preprocessing

        with metrics.measure("vector_search"):
            # do search

        metrics.log_summary()
    """
    query_id: str
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float = 0.0
    steps: List[TimingStep] = field(default_factory=list)
    _current_step: Optional[TimingStep] = field(default=None, repr=False)
    query_preview: str = ""

    def __post_init__(self):
        self.start_time = time.perf_counter()

    @contextmanager
    def measure(self, step_name: str, **metadata):
        """
        Context manager to measure a step's duration.

        Args:
            step_name: Name of the step being measured
            **metadata: Additional metadata to attach to the step
        """
        step = TimingStep(
            name=step_name,
            start_time=time.perf_counter(),
            metadata=metadata
        )
        self._current_step = step
        try:
            yield step
        finally:
            step.complete()
            self.steps.append(step)
            self._current_step = None
            logger.debug(f"[Timing] {step_name}: {step.duration_ms:.2f}ms")

    def record(self, step_name: str, duration_ms: float, **metadata):
        """
        Record a pre-measured step duration.

        Args:
            step_name: Name of the step
            duration_ms: Duration in milliseconds
            **metadata: Additional metadata
        """
        now = time.perf_counter()
        step = TimingStep(
            name=step_name,
            start_time=now - (duration_ms / 1000),
            end_time=now,
            duration_ms=duration_ms,
            metadata=metadata
        )
        self.steps.append(step)
        logger.debug(f"[Timing] {step_name}: {duration_ms:.2f}ms (recorded)")

    def complete(self):
        """Mark the entire pipeline as complete."""
        self.end_time = time.perf_counter()

    @property
    def total_duration_ms(self) -> float:
        """Get total pipeline duration in milliseconds."""
        end = self.end_time if self.end_time > 0 else time.perf_counter()
        return (end - self.start_time) * 1000

    @property
    def steps_duration_ms(self) -> float:
        """Get sum of all measured steps in milliseconds."""
        return sum(s.duration_ms for s in self.steps)

    def get_step(self, name: str) -> Optional[TimingStep]:
        """Get a specific step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def get_breakdown(self) -> Dict[str, float]:
        """Get timing breakdown as a dictionary."""
        breakdown = {}
        for step in self.steps:
            breakdown[step.name] = step.duration_ms
        breakdown["_total"] = self.total_duration_ms
        breakdown["_measured"] = self.steps_duration_ms
        breakdown["_overhead"] = self.total_duration_ms - self.steps_duration_ms
        return breakdown

    def log_summary(self):
        """Log a comprehensive timing summary with hierarchical breakdown."""
        self.complete()

        logger.info("=" * 80)
        logger.info("[TIMING] ========== PIPELINE TIMING SUMMARY ==========")
        logger.info("=" * 80)
        logger.info(f"[TIMING] Query ID: {self.query_id}")
        if self.query_preview:
            logger.info(f"[TIMING] Query: '{self.query_preview[:80]}...'")
        logger.info("-" * 80)

        # Define hierarchy of metrics (parent -> children)
        # Children are sub-steps that are already included in parent timing
        hierarchy = {
            "rag_pipeline_total": ["search_documents_total", "llm_generation", "rag_query_classification"],
            "search_documents_total": ["query_analysis", "document_routing", "iterative_retrieval", "context_building"],
            "iterative_retrieval": ["iterative_reasoning"],
            "iterative_reasoning": ["ir_query_generation", "ir_retrieval_step_1", "ir_retrieval_step_2",
                                   "ir_retrieval_step_3", "ir_thinking_llm_step_1", "ir_thinking_llm_step_2",
                                   "ir_thinking_llm_step_3", "ir_final_answer_generation"],
            "ir_retrieval_step_1": ["unified_retrieval", "vector_search"],
            "ir_retrieval_step_2": ["unified_retrieval", "vector_search"],
            "ir_retrieval_step_3": ["unified_retrieval", "vector_search"],
            "unified_preprocessing": ["unified_preprocessing_llm"],
        }

        # Find all children (to mark as nested)
        all_children = set()
        for children in hierarchy.values():
            all_children.update(children)

        # Get top-level metrics (not children of anything)
        top_level_order = [
            "unified_preprocessing",
            "access_control",
            "cache_lookup",
            "rag_pipeline_total",
        ]

        # Build step lookup
        step_lookup = {s.name: s for s in self.steps}

        def log_step(name: str, indent: int = 0):
            """Log a step with proper indentation."""
            step = step_lookup.get(name)
            if not step:
                return

            prefix = "  " * indent
            indent_char = "├─ " if indent > 0 else ""

            # Calculate bar based on total time
            bar_len = int((step.duration_ms / self.total_duration_ms) * 30) if self.total_duration_ms > 0 else 0
            bar = "█" * bar_len + "░" * (30 - bar_len)
            pct = (step.duration_ms / self.total_duration_ms * 100) if self.total_duration_ms > 0 else 0

            display_name = f"{prefix}{indent_char}{name}"
            logger.info(f"[TIMING] {display_name:<40} {bar} {step.duration_ms:>8.2f}ms ({pct:>5.1f}%)")

            # Log children
            if name in hierarchy:
                for child_name in hierarchy[name]:
                    if child_name in step_lookup:
                        log_step(child_name, indent + 1)

        # Log hierarchical view
        logger.info("[TIMING] === HIERARCHICAL BREAKDOWN ===")
        for name in top_level_order:
            if name in step_lookup:
                log_step(name)

        # Log any steps not in hierarchy (miscellaneous)
        logged_steps = set()
        def collect_logged(name):
            logged_steps.add(name)
            if name in hierarchy:
                for child in hierarchy[name]:
                    collect_logged(child)
        for name in top_level_order:
            collect_logged(name)

        other_steps = [s for s in self.steps if s.name not in logged_steps]
        if other_steps:
            logger.info("[TIMING] === OTHER METRICS ===")
            for step in other_steps:
                bar_len = int((step.duration_ms / self.total_duration_ms) * 30) if self.total_duration_ms > 0 else 0
                bar = "█" * bar_len + "░" * (30 - bar_len)
                pct = (step.duration_ms / self.total_duration_ms * 100) if self.total_duration_ms > 0 else 0
                logger.info(f"[TIMING] {step.name:<40} {bar} {step.duration_ms:>8.2f}ms ({pct:>5.1f}%)")

        logger.info("-" * 80)

        # Calculate actual time breakdown (non-overlapping)
        # Top-level non-overlapping steps
        non_overlapping = ["unified_preprocessing", "access_control", "cache_lookup", "rag_pipeline_total"]
        actual_measured = sum(step_lookup[n].duration_ms for n in non_overlapping if n in step_lookup)

        logger.info(f"[TIMING] {'Top-level steps total':<40} {'':>30} {actual_measured:>8.2f}ms")
        logger.info(f"[TIMING] {'Overhead/unmeasured':<40} {'':>30} {self.total_duration_ms - actual_measured:>8.2f}ms")
        logger.info("=" * 80)
        logger.info(f"[TIMING] {'TOTAL PIPELINE TIME':<40} {'':>30} {self.total_duration_ms:>8.2f}ms")
        logger.info("=" * 80)

        # Identify leaf-level bottlenecks (steps without children, >10% of total)
        leaf_steps = [s for s in self.steps if s.name not in hierarchy]
        bottlenecks = [s for s in leaf_steps if s.duration_ms > self.total_duration_ms * 0.1]
        if bottlenecks:
            logger.info("[TIMING] BOTTLENECKS (leaf steps >10% of total):")
            for step in sorted(bottlenecks, key=lambda x: x.duration_ms, reverse=True):
                pct = step.duration_ms / self.total_duration_ms * 100
                logger.info(f"[TIMING]   ⚠ {step.name}: {step.duration_ms:.2f}ms ({pct:.1f}%)")
            logger.info("=" * 80)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            "query_id": self.query_id,
            "query_preview": self.query_preview,
            "total_ms": self.total_duration_ms,
            "steps": [
                {
                    "name": s.name,
                    "duration_ms": s.duration_ms,
                    "metadata": s.metadata
                }
                for s in self.steps
            ],
            "breakdown": self.get_breakdown()
        }


def get_current_metrics() -> Optional[TimingMetrics]:
    """
    Get the current timing metrics context.

    Uses global storage to work across threads (for LangGraph tool execution).
    """
    global _current_metrics_id, _global_metrics
    if _current_metrics_id is None:
        return None
    with _global_metrics_lock:
        return _global_metrics.get(_current_metrics_id)


def set_current_metrics(metrics: TimingMetrics):
    """
    Set the current timing metrics context.

    Stores in global dictionary to allow cross-thread access.
    """
    global _current_metrics_id, _global_metrics
    _current_metrics_id = metrics.query_id
    with _global_metrics_lock:
        _global_metrics[metrics.query_id] = metrics


def clear_current_metrics():
    """Clear the current timing metrics context."""
    global _current_metrics_id, _global_metrics
    if _current_metrics_id is not None:
        with _global_metrics_lock:
            if _current_metrics_id in _global_metrics:
                del _global_metrics[_current_metrics_id]
        _current_metrics_id = None


@contextmanager
def timing_context(query_id: str, query_preview: str = ""):
    """
    Context manager that creates and manages a TimingMetrics instance.

    Usage:
        with timing_context("query-123", "What is...") as metrics:
            with metrics.measure("step1"):
                # do work
            # metrics.log_summary() is called automatically
    """
    metrics = TimingMetrics(query_id=query_id, query_preview=query_preview)
    set_current_metrics(metrics)
    try:
        yield metrics
    finally:
        metrics.log_summary()
        clear_current_metrics()


def measure_step(step_name: str, **metadata):
    """
    Decorator/context manager to measure a step using the current metrics context.

    Can be used as a context manager:
        with measure_step("my_step"):
            # do work

    Or record manually:
        record_step("my_step", duration_ms)
    """
    metrics = get_current_metrics()
    if metrics:
        return metrics.measure(step_name, **metadata)
    else:
        # Return a no-op context manager if no metrics context
        from contextlib import nullcontext
        return nullcontext()


def record_step(step_name: str, duration_ms: float, **metadata):
    """Record a pre-measured step to the current metrics context."""
    metrics = get_current_metrics()
    if metrics:
        metrics.record(step_name, duration_ms, **metadata)
