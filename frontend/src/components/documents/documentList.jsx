import React, { useState, useEffect } from "react";
import { DataTable } from "primereact/datatable";
import { Column } from "primereact/column";
import { Button } from "primereact/button";
import { ProgressSpinner } from "primereact/progressspinner";
import { ProgressBar } from "primereact/progressbar";
import { Dialog } from "primereact/dialog";
import { useTranslation } from "react-i18next";
import documentService from "../../services/documentService";
import { messageService } from "../../core/message/messageService";
import MarkdownViewer from "./markdownViewer";
import "./documentList.scss";

export const DocumentList = ({ refreshTrigger }) => {
  const { t, i18n, ready } = useTranslation();
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [showMarkdownViewer, setShowMarkdownViewer] = useState(false);
  const [converting, setConverting] = useState(null);
  const [webSockets, setWebSockets] = useState({}); // conversion_id -> websocket
  const [indexing, setIndexing] = useState(null); // filename being indexed
  const [batchIndexStatus, setBatchIndexStatus] = useState(null); // batch indexing status
  const [showStatusLogs, setShowStatusLogs] = useState(false); // status logs dialog
  const [statusLogs, setStatusLogs] = useState([]); // status logs data
  const [statusLogsLoading, setStatusLogsLoading] = useState(false);

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

  // Force re-render when language changes to update translations
  useEffect(() => {
    // This effect ensures the component re-renders when language changes
    // The i18n.language property will trigger a re-render
  }, [i18n.language]);

  // Cleanup WebSockets on unmount
  useEffect(() => {
    return () => {
      Object.values(webSockets).forEach((ws) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.close();
        }
      });
    };
  }, [webSockets]);

  const checkBatchIndexStatus = async () => {
    try {
      const status = await documentService.getIndexStatus();
      setBatchIndexStatus(status);
    } catch (error) {
      console.error("Error checking batch index status:", error);
    }
  };

  const loadDocuments = async () => {
    try {
      setLoading(true);
      const response = await documentService.getDocuments();
      if (response.status === "success") {
        // Sort documents by upload_time in descending order (newest first)
        const sortedDocuments = (response.documents || []).sort((a, b) => {
          const dateA = new Date(a.upload_time);
          const dateB = new Date(b.upload_time);
          return dateB - dateA; // Descending order
        });
        setDocuments(sortedDocuments);
      }
    } catch (error) {
      messageService.errorToast(t("DocumentList.FailedToLoad"));
      console.error("Error loading documents:", error);
    } finally {
      setLoading(false);
    }
  };

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

  const handleIndex = async (document) => {
    try {
      setIndexing(document.filename);
      messageService.infoToast(t("DocumentList.StartingIndex"));

      const response = await documentService.indexDocument(document.filename);

      if (response.status === "success") {
        messageService.successToast(
          `${t("DocumentList.IndexSuccess")} (${response.chunks_indexed} chunks)`
        );
        // Reload documents to update indexed status
        loadDocuments();
      } else {
        messageService.warnToast(t("DocumentList.IndexWarning"));
      }
    } catch (error) {
      messageService.errorToast(t("DocumentList.IndexFailed"));
      console.error("Error indexing document:", error);
    } finally {
      setIndexing(null);
    }
  };

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

  const handleIndexAll = async () => {
    try {
      const response = await documentService.indexAllDocuments();

      if (response.status === "accepted") {
        messageService.infoToast(t("DocumentList.BatchIndexStarted"));
        // Start polling for status
        checkBatchIndexStatus();
      } else if (response.status === "conflict") {
        messageService.warnToast(t("DocumentList.BatchIndexInProgress"));
      }
    } catch (error) {
      messageService.errorToast(t("DocumentList.BatchIndexFailed"));
      console.error("Error starting batch indexing:", error);
    }
  };

  const handleConvert = async (document) => {
    const { total_pages = 0, converted_pages = 0 } = document;
    const isPartiallyConverted = total_pages > 0 && converted_pages < total_pages;

    try {
      setConverting(document.filename);

      // Update document with conversion status
      setDocuments((prev) =>
        prev.map((doc) =>
          doc.filename === document.filename
            ? { ...doc, conversionProgress: 0, conversionStatus: "converting" }
            : doc
        )
      );

      // For partially converted documents, use direct reconversion (skip dots_ocr_service)
      if (isPartiallyConverted) {
        messageService.infoToast(t("DocumentList.ReconvertingUncompleted"));
        console.log(`Using direct reconversion for partial document: ${document.filename}`);

        const response = await documentService.reconvertUncompletedPages(document.filename);

        // Handle "accepted" status - this means the file was new and full conversion was started
        // We need to connect to WebSocket for progress tracking
        if (response.status === "accepted" && response.conversion_id) {
          const conversionId = response.conversion_id;
          messageService.infoToast(t("DocumentList.StartingConversion"));

          // Connect to WebSocket for progress updates (same as fresh conversion flow)
          const ws = documentService.connectToConversionProgress(
            conversionId,
            (progressData) => {
              setDocuments((prev) =>
                prev.map((doc) =>
                  doc.filename === document.filename
                    ? { ...doc, conversionProgress: progressData.progress || 0 }
                    : doc
                )
              );

              if (progressData.status === "completed") {
                messageService.successToast(t("DocumentList.ConversionSuccess"));
                setDocuments((prev) =>
                  prev.map((doc) =>
                    doc.filename === document.filename
                      ? { ...doc, conversionStatus: "completed", conversionProgress: 100 }
                      : doc
                  )
                );
                setConverting(null);
                // Note: Don't close WebSocket yet - wait for indexing to complete
              }

              // Handle indexing status updates
              if (progressData.status === "indexing") {
                messageService.infoToast(t("DocumentList.IndexingDocument") || "Indexing document...");
              }

              if (progressData.status === "indexed") {
                messageService.successToast(
                  `${t("DocumentList.IndexSuccess")} (${progressData.chunks_indexed || 0} chunks)`
                );
                // Reload documents to update status after indexing completes
                loadDocuments();
              }

              if (progressData.status === "index_error") {
                messageService.warnToast(progressData.message || t("DocumentList.IndexFailed"));
                // Still reload to show current status
                loadDocuments();
              }

              if (progressData.status === "warning") {
                messageService.warnToast(progressData.message || t("DocumentList.ConversionWarning"));
                setDocuments((prev) =>
                  prev.map((doc) =>
                    doc.filename === document.filename
                      ? { ...doc, conversionStatus: "warning", conversionProgress: 100 }
                      : doc
                  )
                );
                setConverting(null);
              }

              if (progressData.status === "error") {
                messageService.errorToast(progressData.message || t("DocumentList.ConversionFailed"));
                setDocuments((prev) =>
                  prev.map((doc) =>
                    doc.filename === document.filename
                      ? { ...doc, conversionStatus: "error" }
                      : doc
                  )
                );
                setConverting(null);
              }
            },
            (error) => {
              console.error("WebSocket error:", error);
              messageService.errorToast(t("DocumentList.ConnectionError"));
              setDocuments((prev) =>
                prev.map((doc) =>
                  doc.filename === document.filename
                    ? { ...doc, conversionStatus: "error" }
                    : doc
                )
              );
              setConverting(null);
            }
          );

          setWebSockets((prev) => ({
            ...prev,
            [conversionId]: ws,
          }));
          return;
        }

        if (response.status === "success" || response.status === "partial") {
          const successCount = response.converted_count || 0;
          const failedCount = response.failed_count || 0;

          if (failedCount === 0) {
            messageService.successToast(
              `${t("DocumentList.ConversionSuccess")} (${successCount} pages)`
            );
          } else {
            messageService.warnToast(
              `Converted ${successCount} pages, ${failedCount} failed`
            );
          }

          setDocuments((prev) =>
            prev.map((doc) =>
              doc.filename === document.filename
                ? { ...doc, conversionStatus: "completed", conversionProgress: 100 }
                : doc
            )
          );
          setConverting(null);
          // Reload documents to update status
          setTimeout(() => loadDocuments(), 1000);
        } else {
          throw new Error(response.message || "Reconversion failed");
        }
        return;
      }

      // For fresh conversions, use the normal flow with WebSocket progress
      messageService.infoToast(t("DocumentList.StartingConversion"));

      // Use auto-detection to route to the appropriate converter
      // - doc_service for Word/Excel/Text files
      // - dots_ocr_service for PDF and images
      const converterType = "auto";

      // Start conversion (returns immediately with conversion_id)
      const response = await documentService.convertDocument(
        document.filename,
        "prompt_layout_all_en",
        converterType
      );

      if (response.status === "accepted" && response.conversion_id) {
        const conversionId = response.conversion_id;

        // Connect to WebSocket for progress updates
        const ws = documentService.connectToConversionProgress(
          conversionId,
          (progressData) => {
            // Update document with progress
            setDocuments((prev) =>
              prev.map((doc) =>
                doc.filename === document.filename
                  ? { ...doc, conversionProgress: progressData.progress || 0 }
                  : doc
              )
            );

            // Handle completion
            if (progressData.status === "completed") {
              messageService.successToast(t("DocumentList.ConversionSuccess"));
              setDocuments((prev) =>
                prev.map((doc) =>
                  doc.filename === document.filename
                    ? { ...doc, conversionStatus: "completed", conversionProgress: 100 }
                    : doc
                )
              );
              setConverting(null);
              // Note: Don't reload yet - wait for indexing to complete
            }

            // Handle indexing status updates
            if (progressData.status === "indexing") {
              messageService.infoToast(t("DocumentList.IndexingDocument") || "Indexing document...");
            }

            if (progressData.status === "indexed") {
              messageService.successToast(
                `${t("DocumentList.IndexSuccess")} (${progressData.chunks_indexed || 0} chunks)`
              );
              // Reload documents to update status after indexing completes
              loadDocuments();
            }

            if (progressData.status === "index_error") {
              messageService.warnToast(progressData.message || t("DocumentList.IndexFailed"));
              // Still reload to show current status
              loadDocuments();
            }

            // Handle warnings (e.g., image skipped due to size)
            if (progressData.status === "warning") {
              messageService.warnToast(
                progressData.message || t("DocumentList.ConversionWarning")
              );
              setDocuments((prev) =>
                prev.map((doc) =>
                  doc.filename === document.filename
                    ? { ...doc, conversionStatus: "warning", conversionProgress: 100 }
                    : doc
                )
              );
              setConverting(null);
            }

            // Handle errors
            if (progressData.status === "error") {
              messageService.errorToast(
                progressData.message || t("DocumentList.ConversionFailed")
              );
              setDocuments((prev) =>
                prev.map((doc) =>
                  doc.filename === document.filename
                    ? { ...doc, conversionStatus: "error" }
                    : doc
                )
              );
              setConverting(null);
            }
          },
          (error) => {
            console.error("WebSocket error:", error);
            messageService.errorToast(t("DocumentList.ConnectionError"));
            setDocuments((prev) =>
              prev.map((doc) =>
                doc.filename === document.filename
                  ? { ...doc, conversionStatus: "error" }
                  : doc
              )
            );
            setConverting(null);
          }
        );

        // Store WebSocket reference for cleanup
        setWebSockets((prev) => ({
          ...prev,
          [conversionId]: ws,
        }));
      }
    } catch (error) {
      messageService.errorToast(t("DocumentList.FailedToConvert"));
      console.error("Error starting conversion:", error);
      setDocuments((prev) =>
        prev.map((doc) =>
          doc.filename === document.filename
            ? { ...doc, conversionStatus: "error" }
            : doc
        )
      );
      setConverting(null);
    }
  };

  const actionBodyTemplate = (rowData) => {
    const { markdown_exists, total_pages = 0, converted_pages = 0 } = rowData;
    const isPartiallyConverted = total_pages > 0 && converted_pages < total_pages;

    return (
      <div className="action-buttons">
        {/* Always show convert button */}
        <Button
          icon="pi pi-refresh"
          className={`p-button-rounded ${isPartiallyConverted ? "p-button-info" : "p-button-warning"}`}
          onClick={() => handleConvert(rowData)}
          loading={converting === rowData.filename}
          disabled={converting !== null}
          tooltip={
            isPartiallyConverted
              ? t("DocumentList.ResumeConversion")
              : markdown_exists
                ? t("DocumentList.ReconvertDocument")
                : t("DocumentList.ConvertToMarkdown")
          }
          tooltipPosition="top"
        />

        {/* Show view button only if markdown exists */}
        {markdown_exists && (
          <Button
            icon="pi pi-eye"
            className="p-button-rounded p-button-success"
            onClick={() => handleViewMarkdown(rowData)}
            tooltip={t("DocumentList.ViewMarkdown")}
            tooltipPosition="top"
          />
        )}

        {/* Show index button only if markdown exists */}
        {markdown_exists && (
          <Button
            icon="pi pi-database"
            className="p-button-rounded p-button-help"
            onClick={() => handleIndex(rowData)}
            loading={indexing === rowData.filename}
            disabled={indexing !== null || batchIndexStatus?.status === "running"}
            tooltip={t("DocumentList.IndexDocument")}
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
    const { markdown_exists, total_pages = 0, converted_pages = 0, indexed } = rowData;

    // Determine status based on page counts and indexing
    let statusText, statusClass;

    if (!markdown_exists) {
      // No conversion done yet
      statusText = t("DocumentList.Pending");
      statusClass = "pending";
    } else if (total_pages > 0 && converted_pages < total_pages) {
      // Partial conversion (some pages converted but not all)
      statusText = `${t("DocumentList.Partial")} (${converted_pages}/${total_pages})`;
      statusClass = "partial";
    } else if (indexed) {
      // Fully converted and indexed
      statusText = t("DocumentList.Indexed");
      statusClass = "indexed";
    } else {
      // Fully converted but not indexed
      statusText = t("DocumentList.Converted");
      statusClass = "converted";
    }

    return (
      <span className={`status-badge ${statusClass}`}>
        {statusText}
      </span>
    );
  };

  const progressBodyTemplate = (rowData) => {
    // Get the conversion status from rowData (same as Status column uses rowData.markdown_exists)
    const status = rowData.conversionStatus;

    // If not converting, return null
    if (status !== "converting") {
      return null;
    }

    // Get the progress from rowData
    const progress = rowData.conversionProgress || 0;

    return (
      <div className="progress-container">
        <ProgressBar
          value={progress}
          showValue={true}
          displayValueTemplate={() => `${Math.round(progress)}%`}
        />
      </div>
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
          <Button
            icon="pi pi-database"
            label={t("DocumentList.IndexAll")}
            className="p-button-outlined p-button-help"
            onClick={handleIndexAll}
            loading={batchIndexStatus?.status === "running"}
            disabled={batchIndexStatus?.status === "running"}
            tooltip={t("DocumentList.IndexAllTooltip")}
            tooltipPosition="top"
          />
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
          key={i18n.language}
          value={documents}
          dataKey="filename"
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
            style={{ width: "15%" }}
          />
          <Column
            header={t("DocumentList.Progress")}
            body={progressBodyTemplate}
            style={{ width: "20%" }}
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
};
