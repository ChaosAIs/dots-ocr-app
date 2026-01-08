"""
Pydantic models for the Agentic AI Flow system.

These models define the structured data types used throughout the
multi-agent workflow for type safety and validation.
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class AgentType(str, Enum):
    """Enumeration of available agent types."""
    SQL_AGENT = "sql_agent"
    VECTOR_AGENT = "vector_agent"
    GRAPH_AGENT = "graph_agent"
    GENERIC_DOC_AGENT = "generic_doc_agent"


class TaskStatus(str, Enum):
    """Status of a sub-task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ReviewDecisionType(str, Enum):
    """Type of review decision."""
    APPROVE = "approve"
    REFINE = "refine"
    ESCALATE = "escalate"


class AggregationType(str, Enum):
    """Type of aggregation operation."""
    SUM = "sum"
    COUNT = "count"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    GROUP_BY = "group_by"
    COMBINED = "combined"  # Multiple aggregations in one query (e.g., sum + count + avg)


class DataLocation(str, Enum):
    """Location of data within a document."""
    HEADER = "header"
    LINE_ITEMS = "line_items"
    SUMMARY = "summary"


# =============================================================================
# DOCUMENT MODELS
# =============================================================================

class DocumentSource(BaseModel):
    """
    Represents a document with its schema metadata.

    This model captures all relevant information about a document
    that is needed for intelligent routing and processing.
    """
    document_id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Original filename")
    schema_type: str = Field(..., description="Type of document schema (e.g., 'tabular', 'invoice')")
    schema_definition: Dict[str, Any] = Field(
        default_factory=dict,
        description="Field definitions from DataSchema table"
    )
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Relevance score for this document to the query"
    )
    data_location: DataLocation = Field(
        default=DataLocation.HEADER,
        description="Where the primary data is located in the document"
    )
    workspace_id: Optional[str] = Field(None, description="Workspace this document belongs to")
    created_at: Optional[datetime] = Field(None, description="Document creation timestamp")

    class Config:
        use_enum_values = True


class SchemaGroup(BaseModel):
    """
    Groups documents with compatible schemas for efficient processing.

    Documents in the same group can potentially be processed together
    in a single query if they share common fields.
    """
    group_id: str = Field(..., description="Unique identifier for this schema group")
    schema_type: str = Field(..., description="Schema type for this group")
    common_fields: List[str] = Field(
        default_factory=list,
        description="Fields shared across all documents in this group"
    )
    documents: List[DocumentSource] = Field(
        default_factory=list,
        description="Documents belonging to this group"
    )
    can_combine: bool = Field(
        default=False,
        description="Whether documents can be processed together in single query"
    )
    document_count: int = Field(
        default=0,
        description="Number of documents in this group"
    )

    def __init__(self, **data):
        super().__init__(**data)
        if self.document_count == 0:
            self.document_count = len(self.documents)


# =============================================================================
# TASK MODELS
# =============================================================================

class SubTask(BaseModel):
    """
    Individual task for retrieval agents.

    Each sub-task represents a specific question or calculation
    that needs to be processed by one of the retrieval agents.
    """
    task_id: str = Field(..., description="Unique identifier for this task")
    description: str = Field(..., description="Human-readable description of the task")
    original_query_part: str = Field(
        default="",
        description="The part of the original query this task addresses"
    )
    target_agent: AgentType = Field(
        ...,
        description="Which agent should handle this task"
    )

    # Document source context
    document_ids: List[str] = Field(
        default_factory=list,
        description="Documents to process for this task"
    )
    schema_group_id: Optional[str] = Field(
        None,
        description="Reference to the schema group for this task"
    )
    schema_type: str = Field(
        default="unknown",
        description="Schema type for the documents"
    )

    # Execution hints
    aggregation_type: Optional[AggregationType] = Field(
        None,
        description="Type of aggregation if applicable (use 'combined' for multiple)"
    )
    aggregation_types: List[str] = Field(
        default_factory=list,
        description="List of aggregations when aggregation_type='combined' (e.g., ['sum', 'count', 'avg'])"
    )
    target_fields: List[str] = Field(
        default_factory=list,
        description="Specific fields to target in the query"
    )
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Filters to apply to the query"
    )

    # Dependencies and status
    dependencies: List[str] = Field(
        default_factory=list,
        description="Task IDs that must complete before this task"
    )
    status: TaskStatus = Field(
        default=TaskStatus.PENDING,
        description="Current status of the task"
    )
    priority: int = Field(
        default=0,
        description="Priority for execution ordering (higher = more important)"
    )

    class Config:
        use_enum_values = True


class ExecutionPlan(BaseModel):
    """
    Complete execution plan generated by the Planner Agent.

    Contains all sub-tasks, schema groups, and metadata needed
    to execute the multi-agent workflow.
    """
    sub_tasks: List[SubTask] = Field(
        default_factory=list,
        description="List of sub-tasks to execute"
    )
    schema_groups: List[SchemaGroup] = Field(
        default_factory=list,
        description="Schema groups for document organization"
    )
    execution_strategy: Literal["parallel", "sequential", "mixed"] = Field(
        default="parallel",
        description="How tasks should be executed"
    )

    # Document routing metadata
    total_documents: int = Field(
        default=0,
        description="Total number of documents involved"
    )
    documents_by_schema: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of documents by schema type"
    )

    # Planning metadata
    reasoning: str = Field(
        default="",
        description="Explanation of planning decisions"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the plan was created"
    )

    def get_tasks_by_agent(self, agent_type: AgentType) -> List[SubTask]:
        """Get all tasks assigned to a specific agent."""
        return [t for t in self.sub_tasks if t.target_agent == agent_type]

    def get_pending_tasks(self) -> List[SubTask]:
        """Get all tasks that haven't been completed yet."""
        return [t for t in self.sub_tasks if t.status == TaskStatus.PENDING]

    def get_ready_tasks(self) -> List[SubTask]:
        """Get tasks that are ready to execute (dependencies satisfied)."""
        completed_ids = {t.task_id for t in self.sub_tasks if t.status == TaskStatus.COMPLETED}
        return [
            t for t in self.sub_tasks
            if t.status == TaskStatus.PENDING and all(d in completed_ids for d in t.dependencies)
        ]


# =============================================================================
# OUTPUT MODELS
# =============================================================================

class AgentOutput(BaseModel):
    """
    Output from any retrieval agent.

    Standardized format for results from SQL, Vector, Graph,
    or Generic Doc agents.
    """
    task_id: str = Field(..., description="ID of the task this output is for")
    agent_name: str = Field(..., description="Name of the agent that produced this output")
    status: Literal["success", "partial", "failed"] = Field(
        ...,
        description="Status of the task execution"
    )
    data: Any = Field(
        default=None,
        description="The actual result data"
    )

    # Source tracking
    documents_used: List[str] = Field(
        default_factory=list,
        description="Document IDs that were used"
    )
    schema_type: str = Field(
        default="unknown",
        description="Schema type of the processed documents"
    )
    query_executed: Optional[str] = Field(
        None,
        description="SQL or Cypher query that was executed"
    )

    # Quality metrics
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the results"
    )
    row_count: Optional[int] = Field(
        None,
        description="Number of rows returned (for SQL queries)"
    )

    # Issues and suggestions
    issues: List[str] = Field(
        default_factory=list,
        description="Any issues encountered during processing"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Suggestions for improvement"
    )

    # Metadata
    execution_time_ms: Optional[int] = Field(
        None,
        description="Time taken to execute in milliseconds"
    )
    method_used: Optional[str] = Field(
        None,
        description="Specific method used (for generic agent)"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this output was created"
    )


# =============================================================================
# REVIEW MODELS
# =============================================================================

class RefinementRequest(BaseModel):
    """
    Request to refine a specific task's results.
    """
    task_id: str = Field(..., description="ID of the task to refine")
    target_agent: AgentType = Field(..., description="Agent to handle refinement")
    issue: str = Field(..., description="What issue needs to be addressed")
    guidance: str = Field(..., description="Guidance for refinement")
    new_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="New parameters to try"
    )

    class Config:
        use_enum_values = True


class ReviewDecision(BaseModel):
    """
    Decision from the Reviewer Agent.

    Determines whether results are approved, need refinement,
    or should be escalated to the user.
    """
    decision: ReviewDecisionType = Field(
        ...,
        description="The review decision"
    )
    approved_task_ids: List[str] = Field(
        default_factory=list,
        description="Task IDs that passed review"
    )
    refinement_requests: List[RefinementRequest] = Field(
        default_factory=list,
        description="Requests for refinement"
    )
    quality_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Quality score per task"
    )
    reasoning: str = Field(
        default="",
        description="Explanation of the decision"
    )
    escalation_reason: Optional[str] = Field(
        None,
        description="Reason for escalation (if decision is escalate)"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this decision was made"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# SUMMARY MODELS
# =============================================================================

class DataSourceInfo(BaseModel):
    """Information about a data source used in the response."""
    document_id: str
    filename: str
    schema_type: str
    contribution: str = Field(
        default="",
        description="What this source contributed to the answer"
    )


class FinalResponse(BaseModel):
    """
    Final response to be returned to the user.
    """
    summary: str = Field(..., description="Markdown formatted summary")
    data_sources: List[DataSourceInfo] = Field(
        default_factory=list,
        description="Sources used in the response"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence score"
    )
    caveats: List[str] = Field(
        default_factory=list,
        description="Any caveats or limitations"
    )
    structured_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured data for programmatic access"
    )
    execution_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the execution"
    )
