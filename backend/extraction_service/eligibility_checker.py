"""
Extraction Eligibility Checker

Determines if a document is eligible for structured data extraction.
Uses the centralized DocumentTypeClassifier for LLM-driven analysis.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session as DBSession

from db.models import Document
from common.document_type_classifier import (
    DocumentTypeClassifier,
    ClassificationResult,
)
from .extraction_config import (
    determine_extraction_strategy,
    ExtractionStrategy,
)

logger = logging.getLogger(__name__)


class ExtractionEligibilityChecker:
    """
    Checks if documents are eligible for structured data extraction.

    Uses the centralized DocumentTypeClassifier for LLM-driven
    document type classification.

    Criteria:
    - Document type is in extractable list (from DocumentTypeClassifier)
    - Document has been successfully processed (OCR/conversion complete)
    - Document content is structured (tables, line items, etc.)
    """

    def __init__(self, db: DBSession, llm_client=None):
        """
        Initialize the eligibility checker.

        Args:
            db: SQLAlchemy database session
            llm_client: Optional LLM client for document type inference
        """
        self.db = db
        self.llm_client = llm_client
        self.extraction_enabled = os.getenv("DATA_EXTRACTION_ENABLED", "true").lower() == "true"
        # Use the centralized document type classifier
        self.type_classifier = DocumentTypeClassifier(db=db, llm_client=llm_client)

    def check_eligibility(
        self,
        document_id: UUID
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if a document is eligible for extraction.

        Uses the centralized DocumentTypeClassifier for LLM-driven
        document type classification.

        Args:
            document_id: Document UUID

        Returns:
            Tuple of (is_eligible, schema_type, reason)
        """
        if not self.extraction_enabled:
            return False, None, "Data extraction is disabled"

        # Get document
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if not document:
            return False, None, "Document not found"

        # Check document status
        if document.convert_status and document.convert_status.value not in ["converted", "partial"]:
            return False, None, f"Document not ready: convert_status={document.convert_status.value}"

        # Use centralized classifier for document type classification
        classification = self._classify_document(document)

        if not classification:
            return False, None, "Could not classify document type"

        # Check if extractable using the classifier's result
        if not classification.is_extractable:
            return False, None, f"Document type '{classification.document_type}' is not extractable"

        # Get schema type from classification
        schema_type = classification.schema_type
        if not schema_type:
            return False, None, f"No schema defined for document type '{classification.document_type}'"

        logger.info(
            f"[Eligibility] Document {document_id} classified as '{classification.document_type}' "
            f"(schema: {schema_type}, confidence: {classification.confidence:.2f})"
        )

        return True, schema_type, f"Eligible: {classification.reasoning}"

    def _classify_document(self, document: Document) -> Optional[ClassificationResult]:
        """
        Classify document using the centralized DocumentTypeClassifier.

        Args:
            document: Document model instance

        Returns:
            ClassificationResult or None
        """
        if not document.filename:
            return None

        # Build metadata dict for classifier
        doc_metadata = document.document_metadata or {}

        try:
            # Use centralized classifier
            result = self.type_classifier.classify(
                filename=document.filename,
                metadata=doc_metadata,
                content_preview=None  # Content preview can be added if needed
            )
            return result

        except Exception as e:
            logger.error(f"[Eligibility] Classification failed: {e}")
            return None

    def determine_strategy(
        self,
        document_id: UUID
    ) -> Tuple[ExtractionStrategy, Dict[str, Any]]:
        """
        Determine the extraction strategy for a document.

        Args:
            document_id: Document UUID

        Returns:
            Tuple of (strategy, metadata)
        """
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if not document:
            return ExtractionStrategy.LLM_DIRECT, {"error": "Document not found"}

        # Get source type from filename
        source_type = self._get_source_type(document)

        # Estimate row count
        estimated_rows = self._estimate_row_count(document)

        # Determine strategy
        strategy = determine_extraction_strategy(source_type, estimated_rows)

        metadata = {
            "source_type": source_type,
            "estimated_rows": estimated_rows,
            "strategy": strategy.value,
            "total_pages": document.total_pages,
        }

        logger.info(f"Document {document_id}: strategy={strategy.value}, rows~{estimated_rows}")
        return strategy, metadata

    def _get_source_type(self, document: Document) -> str:
        """
        Get source file type from document.

        Args:
            document: Document model instance

        Returns:
            Source type string
        """
        if not document.filename:
            return "unknown"

        filename_lower = document.filename.lower()

        if filename_lower.endswith('.pdf'):
            return 'pdf'
        elif filename_lower.endswith('.xlsx'):
            return 'xlsx'
        elif filename_lower.endswith('.xls'):
            return 'xls'
        elif filename_lower.endswith('.csv'):
            return 'csv'
        elif filename_lower.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            return 'image'
        else:
            return 'unknown'

    def _estimate_row_count(self, document: Document) -> int:
        """
        Estimate the number of data rows in a document.

        Args:
            document: Document model instance

        Returns:
            Estimated row count
        """
        # For spreadsheets, we might have actual row count in metadata
        doc_metadata = document.document_metadata or {}
        if 'row_count' in doc_metadata:
            return doc_metadata['row_count']

        # Estimate based on page count and document type
        total_pages = document.total_pages or 1
        document_type = doc_metadata.get("document_type", "").lower()

        # Different document types have different row densities
        if document_type in ['invoice', 'receipt']:
            # Typically 5-20 line items per page
            return total_pages * 10
        elif document_type in ['bank_statement']:
            # Typically 20-40 transactions per page
            return total_pages * 30
        elif document_type in ['inventory_report', 'spreadsheet']:
            # Can be very dense
            return total_pages * 50
        else:
            # Default estimate
            return total_pages * 20

    def update_document_extraction_status(
        self,
        document_id: UUID,
        eligible: bool,
        schema_type: Optional[str],
        status: str = "pending"
    ):
        """
        Update document with extraction eligibility status.

        Args:
            document_id: Document UUID
            eligible: Whether document is eligible
            schema_type: Schema type if eligible
            status: Extraction status
        """
        document = self.db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.extraction_eligible = eligible
            document.extraction_schema_type = schema_type
            document.extraction_status = status if eligible else "skipped"
            self.db.commit()
            logger.info(
                f"Document {document_id}: eligible={eligible}, schema={schema_type}, status={status}"
            )

    def get_pending_extractions(self, limit: int = 100) -> list:
        """
        Get documents pending extraction.

        Args:
            limit: Maximum number of documents to return

        Returns:
            List of document IDs pending extraction
        """
        documents = self.db.query(Document).filter(
            Document.extraction_eligible == True,
            Document.extraction_status == "pending"
        ).limit(limit).all()

        return [doc.id for doc in documents]
