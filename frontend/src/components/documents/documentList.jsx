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

  /**
   * Unified Index handler - combines convert + index operations
   * Handles first-time indexing and retry scenarios
   */
  const handleUnifiedIndex = async (document) => {
    const { markdown_exists, indexing_details, convert_status } = document;

    // Determine if this is a retry scenario
    const needsConversion = !markdown_exists || convert_status === "failed" || convert_status === "partial";
    const needsVectorIndex = !indexing_details ||
                             indexing_details?.vector_indexing?.status === "failed" ||
                             indexing_details?.vector_indexing?.status === "pending";
    const needsMetadata = !indexing_details ||
                          indexing_details?.metadata_extraction?.status === "failed" ||
                          indexing_details?.metadata_extraction?.status === "pending";
    const needsGraphRAG = !indexing_details ||
                          indexing_details?.graphrag_indexing?.status === "failed" ||
                          indexing_details?.graphrag_indexing?.status === "pending";

    try {
      setIndexing(document.filename);

      // Step 1: Convert to markdown if needed
      if (needsConversion) {
        await handleConvertForUnifiedIndex(document);
      }

      // Step 2: Index document (vector + metadata + GraphRAG)
      // The backend /index endpoint handles all three phases with WebSocket progress
      if (needsVectorIndex || needsMetadata || needsGraphRAG) {
        messageService.infoToast(t("DocumentList.StartingIndex"));
        const response = await documentService.indexDocument(document.filename);

        // Backend now returns conversion_id for WebSocket tracking
        if (response.status === "accepted" && response.conversion_id) {
          const conversionId = response.conversion_id;

          // Connect to WebSocket for real-time indexing progress
          const ws = documentService.connectToConversionProgress(
            conversionId,
            (progressData) => {
              // Handle indexing status updates
              if (progressData.status === "indexing") {
                messageService.infoToast(progressData.message || t("DocumentList.IndexingDocument"));
              }

              if (progressData.status === "indexed") {
                messageService.successToast(
                  `${t("DocumentList.IndexSuccess")} (${progressData.chunks_indexed || 0} chunks)`
                );
                // Reload documents to update status
                loadDocuments();
              }

              if (progressData.status === "extracting_metadata") {
                messageService.infoToast(progressData.message || "Extracting metadata...");
              }

              if (progressData.status === "metadata_extracted") {
                messageService.infoToast(
                  progressData.message || "Metadata extracted successfully"
                );
                // Reload to show updated metadata
                loadDocuments();
              }

              if (progressData.status === "graphrag_indexing") {
                messageService.infoToast(progressData.message || "Building knowledge graph...");
              }

              if (progressData.status === "graphrag_indexed") {
                messageService.successToast(
                  progressData.message || "Knowledge graph built successfully"
                );
                // Final reload to show complete status
                loadDocuments();
                setIndexing(null);
                // Close WebSocket after all phases complete
                if (ws && ws.readyState === WebSocket.OPEN) {
                  ws.close();
                }
                // Remove from webSockets state
                setWebSockets((prev) => {
                  const updated = { ...prev };
                  delete updated[conversionId];
                  return updated;
                });
              }

              if (progressData.status === "index_error") {
                messageService.errorToast(progressData.message || t("DocumentList.IndexFailed"));
                loadDocuments();
                setIndexing(null);
                // Close WebSocket on error
                if (ws && ws.readyState === WebSocket.OPEN) {
                  ws.close();
                }
                setWebSockets((prev) => {
                  const updated = { ...prev };
                  delete updated[conversionId];
                  return updated;
                });
              }
            },
            (error) => {
              console.error("WebSocket error during indexing:", error);
              messageService.errorToast(t("DocumentList.ConnectionError"));
              setIndexing(null);
            }
          );

          // Store WebSocket reference
          setWebSockets((prev) => ({
            ...prev,
            [conversionId]: ws,
          }));
        } else if (response.status === "success") {
          // Fallback for old synchronous response (backward compatibility)
          messageService.successToast(
            `${t("DocumentList.IndexSuccess")} (${response.chunks_indexed} chunks)`
          );
          loadDocuments();
        } else {
          messageService.warnToast(t("DocumentList.IndexWarning"));
        }
      }
    } catch (error) {
      messageService.errorToast(t("DocumentList.IndexFailed"));
      console.error("Error in unified index:", error);
    } finally {
      setIndexing(null);
    }
  };

  /**
   * Helper function to handle conversion as part of unified index
   * Returns a promise that resolves when conversion is complete
   */
  const handleConvertForUnifiedIndex = async (document) => {
    const { total_pages = 0, converted_pages = 0 } = document;
    const isPartiallyConverted = total_pages > 0 && converted_pages < total_pages;

    return new Promise(async (resolve, reject) => {
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

        // For partially converted documents, use direct reconversion
        if (isPartiallyConverted) {
          messageService.infoToast(t("DocumentList.ReconvertingUncompleted"));
          const response = await documentService.reconvertUncompletedPages(document.filename);

          if (response.status === "success" || response.status === "partial") {
            setDocuments((prev) =>
              prev.map((doc) =>
                doc.filename === document.filename
                  ? { ...doc, conversionStatus: "completed", conversionProgress: 100 }
                  : doc
              )
            );
            setConverting(null);
            await loadDocuments();
            resolve();
          } else {
            throw new Error(response.message || "Reconversion failed");
          }
          return;
        }

        // For fresh conversions, use the normal flow with WebSocket
        messageService.infoToast(t("DocumentList.StartingConversion"));

        const response = await documentService.convertDocument(
          document.filename,
          "prompt_layout_all_en",
          "auto"
        );

        if (response.status === "accepted" && response.conversion_id) {
          const conversionId = response.conversion_id;

          // Connect to WebSocket for progress updates
          const ws = documentService.connectToConversionProgress(
            conversionId,
            (progressData) => {
              // Update progress
              if (progressData.progress !== undefined) {
                setDocuments((prev) =>
                  prev.map((doc) =>
                    doc.filename === document.filename
                      ? { ...doc, conversionProgress: progressData.progress }
                      : doc
                  )
                );
              }

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
                loadDocuments().then(resolve);
              }

              // Handle errors
              if (progressData.status === "error" || progressData.status === "failed") {
                messageService.errorToast(progressData.message || t("DocumentList.ConversionFailed"));
                setDocuments((prev) =>
                  prev.map((doc) =>
                    doc.filename === document.filename
                      ? { ...doc, conversionStatus: "error" }
                      : doc
                  )
                );
                setConverting(null);
                reject(new Error(progressData.message || "Conversion failed"));
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
              reject(error);
            }
          );

          // Store WebSocket reference
          setWebSockets((prev) => ({
            ...prev,
            [conversionId]: ws,
          }));
        }
      } catch (error) {
        messageService.errorToast(t("DocumentList.FailedToConvert"));
        console.error("Error in conversion:", error);
        setDocuments((prev) =>
          prev.map((doc) =>
            doc.filename === document.filename
              ? { ...doc, conversionStatus: "error" }
              : doc
          )
        );
        setConverting(null);
        reject(error);
      }
    });
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

  /**
   * Determine the state of the unified Index button
   */
  const getUnifiedIndexButtonState = (rowData) => {
    const {
      markdown_exists,
      convert_status,
      indexing_details,
      total_pages = 0,
      converted_pages = 0
    } = rowData;

    // Check if currently processing
    const isProcessing = converting === rowData.filename || indexing === rowData.filename;

    // Check conversion status
    const conversionFailed = convert_status === "failed";
    // Partial means: started conversion (converted_pages > 0) but not finished (< total_pages)
    // Don't treat fresh uploads (converted_pages = 0) as partial
    const conversionPartial = convert_status === "partial" ||
                              (total_pages > 0 && converted_pages > 0 && converted_pages < total_pages);
    const conversionComplete = markdown_exists && !conversionFailed && !conversionPartial;

    // Check indexing status
    const vectorStatus = indexing_details?.vector_indexing?.status;
    const metadataStatus = indexing_details?.metadata_extraction?.status;
    const graphragStatus = indexing_details?.graphrag_indexing?.status;

    const vectorComplete = vectorStatus === "completed";
    const metadataComplete = metadataStatus === "completed";
    const graphragComplete = graphragStatus === "completed";
    const graphragProcessing = graphragStatus === "processing";

    const vectorFailed = vectorStatus === "failed";
    const metadataFailed = metadataStatus === "failed";
    const graphragFailed = graphragStatus === "failed";

    const anyIndexingFailed = vectorFailed || metadataFailed || graphragFailed;
    const anyIndexingProcessing = vectorStatus === "processing" || metadataStatus === "processing" || graphragProcessing;

    // Determine button state
    if (isProcessing || anyIndexingProcessing) {
      return {
        type: "indexing",
        icon: "pi pi-spin pi-spinner",
        className: "p-button-rounded p-button-primary",
        tooltip: t("DocumentList.Indexing"),
        disabled: true,
        loading: true
      };
    }

    // All complete (including GraphRAG)
    if (conversionComplete && vectorComplete && metadataComplete && graphragComplete) {
      return {
        type: "complete",
        icon: "pi pi-check",
        className: "p-button-rounded p-button-success",
        tooltip: t("DocumentList.AllIndexingComplete"),
        disabled: true,
        loading: false
      };
    }

    // Conversion + Vector + Metadata complete, GraphRAG still pending/processing
    if (conversionComplete && vectorComplete && metadataComplete && !graphragComplete && !graphragFailed) {
      return {
        type: "partial_complete",
        icon: "pi pi-clock",
        className: "p-button-rounded p-button-secondary",
        tooltip: t("DocumentList.GraphRAGPending"),
        disabled: true,
        loading: false
      };
    }

    // Any failures - show retry
    if (conversionFailed || conversionPartial || anyIndexingFailed) {
      return {
        type: "retry",
        icon: "pi pi-refresh",
        className: "p-button-rounded p-button-warning",
        tooltip: t("DocumentList.RetryIndex"),
        disabled: false,
        loading: false
      };
    }

    // First time - show index
    return {
      type: "index",
      icon: "pi pi-database",
      className: "p-button-rounded p-button-primary",
      tooltip: t("DocumentList.IndexDocument"),
      disabled: false,
      loading: false
    };
  };

  const actionBodyTemplate = (rowData) => {
    const {
      markdown_exists,
      convert_status,
      total_pages = 0,
      converted_pages = 0
    } = rowData;
    const buttonState = getUnifiedIndexButtonState(rowData);

    // Check if conversion is fully completed with no errors
    const conversionFailed = convert_status === "failed";
    const conversionPartial = convert_status === "partial" ||
                              (total_pages > 0 && converted_pages > 0 && converted_pages < total_pages);
    const conversionConverting = convert_status === "converting";
    const conversionFullyCompleted = markdown_exists && !conversionFailed && !conversionPartial && !conversionConverting;

    return (
      <div className="action-buttons">
        {/* Unified Index button */}
        <Button
          icon={buttonState.icon}
          className={buttonState.className}
          onClick={() => handleUnifiedIndex(rowData)}
          loading={buttonState.loading}
          disabled={buttonState.disabled || batchIndexStatus?.status === "running"}
          tooltip={buttonState.tooltip}
          tooltipPosition="top"
        />

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

  const progressBodyTemplate = (rowData) => {
    const { indexing_details } = rowData;

    // Check if conversion is in progress
    const conversionStatus = rowData.conversionStatus;
    if (conversionStatus === "converting") {
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
    }

    // Check if GraphRAG indexing is in progress
    const graphragStatus = indexing_details?.graphrag_indexing?.status;
    const graphragProcessing = graphragStatus === "processing";

    if (graphragProcessing) {
      const totalChunks = indexing_details?.graphrag_indexing?.total_chunks || 0;
      const processedChunks = indexing_details?.graphrag_indexing?.processed_chunks || 0;
      const progress = totalChunks > 0 ? Math.round((processedChunks / totalChunks) * 100) : 0;

      return (
        <div className="progress-container">
          <ProgressBar
            value={progress}
            showValue={true}
            displayValueTemplate={() => `GraphRAG: ${Math.round(progress)}%`}
            className="graphrag-progress"
          />
        </div>
      );
    }

    // Check if vector or metadata indexing is in progress
    const vectorStatus = indexing_details?.vector_indexing?.status;
    const metadataStatus = indexing_details?.metadata_extraction?.status;

    if (vectorStatus === "processing" || metadataStatus === "processing") {
      return (
        <div className="progress-container">
          <ProgressSpinner style={{ width: '30px', height: '30px' }} strokeWidth="4" />
          <span style={{ marginLeft: '8px' }}>
            {vectorStatus === "processing" ? "Indexing..." : "Extracting metadata..."}
          </span>
        </div>
      );
    }

    return null;
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
            style={{ width: "18%" }}
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
