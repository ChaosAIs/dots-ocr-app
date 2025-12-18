import React, { useState, useEffect, useRef, useCallback, forwardRef, useImperativeHandle } from "react";
import { DataTable } from "primereact/datatable";
import { Column } from "primereact/column";
import { Button } from "primereact/button";
import { ProgressSpinner } from "primereact/progressspinner";
import { Dialog } from "primereact/dialog";
import { useTranslation } from "react-i18next";
import documentService from "../../services/documentService";
import { messageService } from "../../core/message/messageService";
import MarkdownViewer from "./markdownViewer";
import "./documentList.scss";

export const DocumentList = forwardRef(({ refreshTrigger }, ref) => {
  const { t, i18n, ready } = useTranslation();
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [showMarkdownViewer, setShowMarkdownViewer] = useState(false);
  const [batchIndexStatus, setBatchIndexStatus] = useState(null); // batch indexing status
  const [showStatusLogs, setShowStatusLogs] = useState(false); // status logs dialog
  const [statusLogs, setStatusLogs] = useState([]); // status logs data
  const [statusLogsLoading, setStatusLogsLoading] = useState(false);
  const [tableKey, setTableKey] = useState(0); // Force DataTable re-render
  // Ref to store the latest loadDocuments function for WebSocket handler
  const loadDocumentsRef = useRef(null);
  // Ref to store WebSocket instance for reconnection
  const wsRef = useRef(null);
  // Ref to track reconnection attempts
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;
  // Ref to store reconnection timeout
  const reconnectTimeoutRef = useRef(null);
  // Ref to store currently subscribed document IDs
  const subscribedDocIdsRef = useRef(new Set());

  // Debug: Log whenever documents state changes
  useEffect(() => {
    console.log("🔄 Documents state changed! New count:", documents.length);
    console.log("📊 Document statuses:", documents.map(d => ({
      filename: d.filename,
      index_status: d.index_status
    })));
  }, [documents]);

  // Load documents on component mount and when refreshTrigger changes
  useEffect(() => {
    loadDocuments();
    checkBatchIndexStatus(); // Check if batch indexing is in progress on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshTrigger]);

  // Poll for batch index status when running
  useEffect(() => {
    let intervalId = null;
    if (batchIndexStatus?.status === "running") {
      intervalId = setInterval(() => {
        checkBatchIndexStatus();
      }, 2000); // Poll every 2 seconds
    }
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [batchIndexStatus?.status]);

  // Subscribe to document updates via WebSocket
  const subscribeToDocuments = useCallback((documentIds) => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      console.warn("⚠️ Cannot subscribe - WebSocket not connected");
      return;
    }

    if (!documentIds || documentIds.length === 0) {
      console.log("📭 No documents to subscribe to");
      return;
    }

    // Filter out already subscribed documents
    const newDocIds = documentIds.filter(id => !subscribedDocIdsRef.current.has(id));

    if (newDocIds.length === 0) {
      console.log("✅ Already subscribed to all requested documents");
      return;
    }

    console.log(`📬 Subscribing to ${newDocIds.length} documents:`, newDocIds);

    wsRef.current.send(JSON.stringify({
      action: "subscribe",
      document_ids: newDocIds
    }));

    // Update subscribed set
    newDocIds.forEach(id => subscribedDocIdsRef.current.add(id));
  }, []);

  // Expose subscribeToDocuments method to parent component via ref
  useImperativeHandle(ref, () => ({
    subscribeToNewDocuments: (documentIds) => {
      console.log(`📨 DocumentList: Received request to subscribe to ${documentIds?.length || 0} new documents:`, documentIds);
      subscribeToDocuments(documentIds);
    }
  }), [subscribeToDocuments]);

  // WebSocket connection with automatic reconnection and subscription support
  const connectWebSocket = useCallback(() => {
    // Don't reconnect if already connected or connecting
    if (wsRef.current?.readyState === WebSocket.OPEN || wsRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    // Check if we've exceeded max reconnection attempts
    if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
      console.error("❌ Max WebSocket reconnection attempts reached");
      return;
    }

    try {
      const wsUrl = `${documentService.getWebSocketUrl()}/ws/document-status`;
      console.log(`🔌 Connecting to WebSocket (attempt ${reconnectAttemptsRef.current + 1}/${maxReconnectAttempts})...`);

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = async () => {
        console.log("✅ Connected to centralized document status WebSocket");
        reconnectAttemptsRef.current = 0; // Reset counter on successful connection

        // Check for in-progress documents and subscribe to them
        try {
          const response = await documentService.getInProgressDocuments();

          if (response.status === "success" && response.documents.length > 0) {
            const docIds = response.documents.map(d => d.id);
            console.log(`📋 Found ${docIds.length} in-progress documents on connect - subscribing...`);

            // Subscribe to in-progress documents
            subscribeToDocuments(docIds);
          } else {
            console.log("✅ No in-progress documents found on connect");
          }
        } catch (error) {
          console.error("Error checking in-progress documents:", error);
        }

        // Reload documents immediately to get latest status from database
        if (loadDocumentsRef.current) {
          loadDocumentsRef.current();
        }
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log("📨 Document status update:", data);

          // Handle different event types
          if (data.event_type === "connected") {
            console.log("✅ WebSocket connection confirmed");
            return;
          }

          if (data.event_type === "subscribed") {
            console.log(`✅ Subscribed to ${data.count} documents`);
            return;
          }

          if (data.event_type === "unsubscribed") {
            console.log(`✅ Unsubscribed from ${data.count} documents`);
            return;
          }

          // Update specific document in state when status changes
          if (data.event_type && data.document_id) {
            console.log(`🔄 Received status update for document ${data.document_id}: ${data.event_type}`, data);

            // Always reload documents to get the latest status from backend
            console.log(`🔄 Reloading all documents due to ${data.event_type} for document ${data.document_id}`);
            if (loadDocumentsRef.current) {
              loadDocumentsRef.current();
            }

            // If indexing/conversion completed, remove from subscriptions
            if (data.event_type === "indexing_completed" || data.event_type === "ocr_completed") {
              subscribedDocIdsRef.current.delete(data.document_id);
              console.log(`📭 Removed ${data.document_id} from subscriptions (completed)`);
            }
          }
        } catch (error) {
          console.error("Error parsing WebSocket message:", error);
        }
      };

      ws.onerror = (error) => {
        console.error("❌ WebSocket error:", error);
      };

      ws.onclose = (event) => {
        console.log("🔌 WebSocket connection closed", event.code, event.reason);
        wsRef.current = null;

        // Don't reconnect on normal closure or policy violation
        if (event.code === 1000 || event.code === 1008) {
          console.log("WebSocket closed normally - not reconnecting");
          return;
        }

        // Attempt to reconnect after delay if we have subscribed documents
        if (subscribedDocIdsRef.current.size > 0 && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current += 1;
          const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current - 1), 10000); // Exponential backoff, max 10s
          console.log(`🔄 Scheduling reconnection in ${delay}ms (attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts})`);

          reconnectTimeoutRef.current = setTimeout(() => {
            connectWebSocket();
          }, delay);
        } else {
          console.log("⏸️ No subscribed documents - not reconnecting");
        }
      };
    } catch (error) {
      console.error("Error creating WebSocket connection:", error);
    }
  }, [subscribeToDocuments]);

  // Centralized WebSocket connection for all document status updates
  useEffect(() => {
    connectWebSocket();

    // Cleanup function
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
        wsRef.current.close();
      }
    };
  }, [connectWebSocket]);

  // Force re-render when language changes to update translations
  useEffect(() => {
    // This effect ensures the component re-renders when language changes
    // The i18n.language property will trigger a re-render
  }, [i18n.language]);

  const checkBatchIndexStatus = async () => {
    try {
      const status = await documentService.getIndexStatus();
      setBatchIndexStatus(status);
    } catch (error) {
      console.error("Error checking batch index status:", error);
    }
  };

  const loadDocuments = useCallback(async () => {
    console.log("🔄 loadDocuments() called - fetching documents from API...");
    try {
      setLoading(true);
      const response = await documentService.getDocuments();
      console.log("📦 API response received:", response);
      if (response.status === "success") {
        // Sort documents by upload_time in descending order (newest first)
        const sortedDocuments = (response.documents || []).sort((a, b) => {
          const dateA = new Date(a.upload_time);
          const dateB = new Date(b.upload_time);
          return dateB - dateA; // Descending order
        });
        console.log("✅ Setting documents state with", sortedDocuments.length, "documents");
        console.log("📄 Documents:", sortedDocuments.map(d => ({
          filename: d.filename,
          index_status: d.index_status,
          markdown_exists: d.markdown_exists
        })));
        // Force new array reference to trigger DataTable re-render
        setDocuments([...sortedDocuments]);
        // Force DataTable to re-render by changing key
        setTableKey(prev => prev + 1);
        console.log("🔄 Forced DataTable re-render");

        // Find in-progress documents and subscribe to them
        const inProgressDocs = sortedDocuments.filter(doc =>
          doc.convert_status === "converting" || doc.index_status === "indexing"
        );

        if (inProgressDocs.length > 0) {
          const docIds = inProgressDocs.map(d => d.id);
          console.log(`📊 Found ${inProgressDocs.length} in-progress documents - subscribing...`);
          subscribeToDocuments(docIds);
        } else {
          console.log("✅ No in-progress documents found");
        }
      } else {
        console.warn("⚠️ API response status is not 'success':", response.status);
      }
    } catch (error) {
      messageService.errorToast(t("DocumentList.FailedToLoad"));
      console.error("❌ Error loading documents:", error);
    } finally {
      setLoading(false);
      console.log("✅ loadDocuments() completed");
    }
  }, [t, subscribeToDocuments]);

  // Update ref whenever loadDocuments changes
  useEffect(() => {
    loadDocumentsRef.current = loadDocuments;
  }, [loadDocuments]);

  // Monitor documents array changes for debugging
  useEffect(() => {
    console.log("📊 Documents array changed! Count:", documents.length);
    console.log("📊 Document statuses:", documents.map(d => ({
      id: d.id,
      filename: d.filename,
      index_status: d.index_status,
      convert_status: d.convert_status,
      markdown_exists: d.markdown_exists
    })));
    console.log("📊 TableKey:", tableKey);
  }, [documents, tableKey]);

  const handleViewMarkdown = async (document) => {
    try {
      setSelectedDocument(document);
      setShowMarkdownViewer(true);
    } catch (error) {
      messageService.errorToast(t("DocumentList.FailedToView"));
      console.error("Error viewing markdown:", error);
    }
  };

  const handleDelete = async (document) => {
    try {
      // Show confirmation dialog
      messageService.confirmDeletionDialog(
        t("DocumentList.ConfirmDelete"),
        async (confirmed) => {
          if (confirmed) {
            try {
              await documentService.deleteDocument(document.filename);
              messageService.successToast(t("DocumentList.DeleteSuccess"));
              // Reload documents list
              loadDocuments();
            } catch (error) {
              messageService.errorToast(t("DocumentList.DeleteFailed"));
              console.error("Error deleting document:", error);
            }
          }
        }
      );
    } catch (error) {
      messageService.errorToast(t("DocumentList.DeleteFailed"));
      console.error("Error deleting document:", error);
    }
  };

  // Manual trigger functions removed - system is now fully automated
  // OCR and indexing are triggered automatically on file upload



  const handleViewStatusLogs = async (document) => {
    try {
      setStatusLogsLoading(true);
      setSelectedDocument(document);
      setShowStatusLogs(true);

      const response = await documentService.getDocumentStatusLogs(document.filename);
      if (response.status === "success") {
        setStatusLogs(response.logs || []);
      } else {
        setStatusLogs([]);
      }
    } catch (error) {
      console.error("Error fetching status logs:", error);
      setStatusLogs([]);
      messageService.errorToast(t("DocumentList.FailedToLoadLogs"));
    } finally {
      setStatusLogsLoading(false);
    }
  };





  const actionBodyTemplate = (rowData) => {
    const {
      markdown_exists,
      convert_status,
      total_pages = 0,
      converted_pages = 0
    } = rowData;

    // Check if conversion is fully completed with no errors
    const conversionFailed = convert_status === "failed";
    const conversionPartial = convert_status === "partial" ||
                              (total_pages > 0 && converted_pages > 0 && converted_pages < total_pages);
    const conversionConverting = convert_status === "converting";
    const conversionFullyCompleted = markdown_exists && !conversionFailed && !conversionPartial && !conversionConverting;

    return (
      <div className="action-buttons">
        {/* Show view button only if conversion is fully completed with no errors */}
        {markdown_exists && (
          <Button
            icon="pi pi-eye"
            className="p-button-rounded p-button-success"
            onClick={() => handleViewMarkdown(rowData)}
            disabled={!conversionFullyCompleted}
            tooltip={
              conversionFullyCompleted
                ? t("DocumentList.ViewMarkdown")
                : conversionFailed
                  ? t("DocumentList.ConversionFailedCannotView")
                  : conversionPartial
                    ? t("DocumentList.PartialConversionCannotView")
                    : t("DocumentList.ConversionInProgressCannotView")
            }
            tooltipPosition="top"
          />
        )}

        {/* Show status logs button if document has database record */}
        {rowData.document_id && (
          <Button
            icon="pi pi-history"
            className="p-button-rounded p-button-secondary"
            onClick={() => handleViewStatusLogs(rowData)}
            tooltip={t("DocumentList.ViewStatusLogs")}
            tooltipPosition="top"
          />
        )}

        <Button
          icon="pi pi-trash"
          className="p-button-rounded p-button-danger"
          onClick={() => handleDelete(rowData)}
          tooltip={t("DocumentList.DeleteDocument")}
          tooltipPosition="top"
        />
      </div>
    );
  };

  const fileSizeBodyTemplate = (rowData) => {
    return documentService.formatFileSize(rowData.file_size);
  };

  const uploadTimeBodyTemplate = (rowData) => {
    return documentService.formatDate(rowData.upload_time);
  };

  const statusBodyTemplate = (rowData) => {
    // Force re-render when language changes by using i18n.language
    const {
      markdown_exists,
      total_pages = 0,
      converted_pages = 0,
      convert_status,
      indexing_details
    } = rowData;

    // Check indexing phases
    const vectorStatus = indexing_details?.vector_indexing?.status;
    const metadataStatus = indexing_details?.metadata_extraction?.status;
    const graphragStatus = indexing_details?.graphrag_indexing?.status;

    const vectorComplete = vectorStatus === "completed";
    const metadataComplete = metadataStatus === "completed";
    const graphragComplete = graphragStatus === "completed";

    const vectorProcessing = vectorStatus === "processing";
    const metadataProcessing = metadataStatus === "processing";
    const graphragProcessing = graphragStatus === "processing";

    const vectorPending = vectorStatus === "pending" || !vectorStatus;
    const metadataPending = metadataStatus === "pending" || !metadataStatus;
    const graphragPending = graphragStatus === "pending" || !graphragStatus;

    const vectorFailed = vectorStatus === "failed";
    const metadataFailed = metadataStatus === "failed";
    const graphragFailed = graphragStatus === "failed";

    const anyIndexingProcessing = vectorProcessing || metadataProcessing || graphragProcessing;
    const anyIndexingPending = (vectorPending || metadataPending || graphragPending) && markdown_exists;

    // Determine status based on all phases
    let statusText, statusClass;

    // Check conversion status first
    if (!markdown_exists && convert_status === "converting") {
      // OCR conversion in progress
      statusText = t("DocumentList.Indexing");
      statusClass = "indexing";
    } else if (!markdown_exists) {
      // Just uploaded, not converted yet
      statusText = t("DocumentList.NoIndex");
      statusClass = "no-index";
    } else if (convert_status === "failed") {
      statusText = t("DocumentList.ConversionFailed");
      statusClass = "failed";
    } else if (total_pages > 0 && converted_pages > 0 && converted_pages < total_pages) {
      // Only show partial if conversion was started (converted_pages > 0)
      statusText = `${t("DocumentList.Partial")} (${converted_pages}/${total_pages})`;
      statusClass = "partial";
    } else if (vectorFailed || metadataFailed || graphragFailed) {
      // Any indexing phase failed
      statusText = t("DocumentList.IndexingFailed");
      statusClass = "failed";
    } else if (vectorComplete && metadataComplete && graphragComplete) {
      // All phases complete
      statusText = t("DocumentList.FullyIndexed");
      statusClass = "indexed";
    } else if (anyIndexingProcessing) {
      // Any indexing phase is currently processing
      if (graphragProcessing) {
        statusText = t("DocumentList.IndexingGraphRAG");
      } else if (metadataProcessing) {
        statusText = t("DocumentList.IndexingMetadata");
      } else {
        statusText = t("DocumentList.Indexing");
      }
      statusClass = "indexing";
    } else if (vectorComplete && metadataComplete && graphragPending) {
      // Vector + Metadata done, GraphRAG pending
      statusText = t("DocumentList.PartiallyIndexed");
      statusClass = "partial-indexed";
    } else if (anyIndexingPending) {
      // Converted but indexing not started or pending
      statusText = t("DocumentList.Indexing");
      statusClass = "indexing";
    } else if (markdown_exists) {
      // Converted but not indexed yet (fallback - shouldn't normally reach here)
      statusText = t("DocumentList.Converted");
      statusClass = "converted";
    } else {
      statusText = t("DocumentList.NoIndex");
      statusClass = "no-index";
    }

    return (
      <span className={`status-badge ${statusClass}`}>
        {statusText}
      </span>
    );
  };

  // Show loading spinner while translations are loading or documents are loading
  if (!ready || (loading && documents.length === 0)) {
    return (
      <div className="document-list-loading">
        <ProgressSpinner />
      </div>
    );
  }

  const renderBatchIndexStatus = () => {
    if (!batchIndexStatus || batchIndexStatus.status === "idle") {
      return null;
    }

    const { status, indexed_documents, total_documents, current_index, message } = batchIndexStatus;

    if (status === "running") {
      // Calculate progress based on current_index (0-based) if available
      // When processing document i of n, show progress as (i + 0.5) / n to indicate in-progress
      // Use indexed_documents / total_documents as fallback (for completed documents)
      let progress = 0;
      if (total_documents > 0) {
        if (current_index !== undefined && current_index !== null) {
          // Show progress as midpoint of current document (e.g., processing doc 0 of 2 = 25%)
          progress = Math.round(((current_index + 0.5) / total_documents) * 100);
        } else {
          // Fallback to completed documents ratio
          progress = Math.round((indexed_documents / total_documents) * 100);
        }
      }
      return (
        <div className="batch-index-status running">
          <ProgressSpinner style={{ width: '20px', height: '20px' }} strokeWidth="4" />
          <span className="status-message">
            {message || `${t("DocumentList.Indexing")} ${indexed_documents}/${total_documents}`}
          </span>
          <span className="status-progress">({progress}%)</span>
        </div>
      );
    }

    if (status === "completed") {
      return (
        <div className="batch-index-status completed">
          <i className="pi pi-check-circle" />
          <span className="status-message">{message || t("DocumentList.IndexCompleted")}</span>
        </div>
      );
    }

    if (status === "error") {
      return (
        <div className="batch-index-status error">
          <i className="pi pi-times-circle" />
          <span className="status-message">{message || t("DocumentList.IndexError")}</span>
        </div>
      );
    }

    return null;
  };

  return (
    <div className="document-list-container">
      <div className="document-list-header">
        <h2>{t("DocumentList.Title")}</h2>
        <div className="header-actions">
          {/* Auto-processing status indicator */}
          {renderBatchIndexStatus()}
          <Button
            icon="pi pi-refresh"
            className="p-button-rounded p-button-text"
            onClick={loadDocuments}
            loading={loading}
            tooltip={t("DocumentList.Refresh")}
            tooltipPosition="top"
          />
        </div>
      </div>

      {documents.length === 0 ? (
        <div className="no-documents">
          <p>{t("DocumentList.NoDocuments")}</p>
        </div>
      ) : (
        <DataTable
          key={tableKey}
          value={documents}
          dataKey="id"
          className="p-datatable-striped"
          responsiveLayout="scroll"
          paginator
          rows={10}
          rowsPerPageOptions={[5, 10, 20]}
          paginatorTemplate="RowsPerPageDropdown FirstPageLink PrevPageLink CurrentPageReport NextPageLink LastPageLink"
          currentPageReportTemplate={t("DocumentList.PaginatorTemplate")}
        >
          <Column field="filename" header={t("DocumentList.Filename")} style={{ width: "30%" }} />
          <Column
            field="file_size"
            header={t("DocumentList.Size")}
            body={fileSizeBodyTemplate}
            style={{ width: "15%" }}
          />
          <Column
            field="upload_time"
            header={t("DocumentList.UploadTime")}
            body={uploadTimeBodyTemplate}
            style={{ width: "25%" }}
          />
          <Column
            field="markdown_exists"
            header={t("DocumentList.Status")}
            body={statusBodyTemplate}
            style={{ width: "18%" }}
          />
          <Column
            body={actionBodyTemplate}
            header={t("DocumentList.Actions")}
            style={{ width: "15%" }}
            bodyStyle={{ textAlign: "center" }}
          />
        </DataTable>
      )}

      {showMarkdownViewer && selectedDocument && (
        <MarkdownViewer
          document={selectedDocument}
          visible={showMarkdownViewer}
          onHide={() => {
            setShowMarkdownViewer(false);
            setSelectedDocument(null);
          }}
        />
      )}

      {/* Status Logs Dialog */}
      <Dialog
        header={`${t("DocumentList.StatusLogs")} - ${selectedDocument?.filename || ""}`}
        visible={showStatusLogs}
        style={{ width: "50vw" }}
        onHide={() => {
          setShowStatusLogs(false);
          setStatusLogs([]);
        }}
      >
        {statusLogsLoading ? (
          <div className="status-logs-loading">
            <ProgressSpinner style={{ width: "30px", height: "30px" }} />
          </div>
        ) : statusLogs.length === 0 ? (
          <p>{t("DocumentList.NoStatusLogs")}</p>
        ) : (
          <DataTable value={statusLogs} size="small" stripedRows>
            <Column field="created_at" header={t("DocumentList.Time")}
              body={(rowData) => documentService.formatDate(rowData.created_at)} />
            <Column field="status_type" header={t("DocumentList.Type")} />
            <Column field="old_status" header={t("DocumentList.OldStatus")} />
            <Column field="new_status" header={t("DocumentList.NewStatus")} />
            <Column field="message" header={t("DocumentList.Message")} />
          </DataTable>
        )}
      </Dialog>
    </div>
  );
});
