import React, { useState, useEffect, useCallback } from "react";
import { DataTable } from "primereact/datatable";
import { Column } from "primereact/column";
import { Button } from "primereact/button";
import { ProgressSpinner } from "primereact/progressspinner";
import { ProgressBar } from "primereact/progressbar";
import { Dropdown } from "primereact/dropdown";
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
  const [converterTypes, setConverterTypes] = useState({}); // filename -> converter type

  // Load documents on component mount and when refreshTrigger changes
  useEffect(() => {
    loadDocuments();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshTrigger]);

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

  const handleConvert = async (document) => {
    try {
      setConverting(document.filename);
      messageService.infoToast(t("DocumentList.StartingConversion"));

      // Update document with conversion status
      setDocuments((prev) =>
        prev.map((doc) =>
          doc.filename === document.filename
            ? { ...doc, conversionProgress: 0, conversionStatus: "converting" }
            : doc
        )
      );

      // Get the selected converter type for this document (default to "auto")
      const converterType = converterTypes[document.filename] || "auto";

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
              // Reload documents to update status
              setTimeout(() => loadDocuments(), 1000);
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

  // Get converter options based on file type
  const getConverterOptions = (filename) => {
    const isImage = documentService.isImageFile(filename);
    const isDocService = documentService.isDocServiceFile(filename);

    const options = [
      { label: t("DocumentList.ConverterAuto"), value: "auto" }
    ];

    if (isDocService) {
      options.push({ label: t("DocumentList.ConverterDocService"), value: "doc_service" });
    }

    if (isImage) {
      options.push({ label: t("DocumentList.ConverterDeepSeekOCR"), value: "deepseek_ocr" });
      options.push({ label: t("DocumentList.ConverterOCRService"), value: "ocr_service" });
    } else if (!isDocService) {
      // PDF files
      options.push({ label: t("DocumentList.ConverterOCRService"), value: "ocr_service" });
    }

    return options;
  };

  const actionBodyTemplate = (rowData) => {
    const converterOptions = getConverterOptions(rowData.filename);
    const selectedConverter = converterTypes[rowData.filename] || "auto";

    return (
      <div className="action-buttons">
        {rowData.markdown_exists ? (
          <Button
            icon="pi pi-eye"
            className="p-button-rounded p-button-success"
            onClick={() => handleViewMarkdown(rowData)}
            tooltip={t("DocumentList.ViewMarkdown")}
            tooltipPosition="top"
          />
        ) : (
          <>
            {converterOptions.length > 1 && (
              <Dropdown
                value={selectedConverter}
                options={converterOptions}
                onChange={(e) => {
                  const newValue = e.value;
                  setConverterTypes(prev => ({ ...prev, [rowData.filename]: newValue }));
                }}
                placeholder={t("DocumentList.SelectConverter")}
                className="converter-dropdown"
                disabled={converting === rowData.filename}
              />
            )}
            <Button
              icon="pi pi-refresh"
              className="p-button-rounded p-button-warning"
              onClick={() => handleConvert(rowData)}
              loading={converting === rowData.filename}
              disabled={converting !== null}
              tooltip={t("DocumentList.ConvertToMarkdown")}
              tooltipPosition="top"
            />
          </>
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
    const statusText = rowData.markdown_exists
      ? t("DocumentList.Converted")
      : t("DocumentList.Pending");

    return (
      <span className={`status-badge ${rowData.markdown_exists ? "converted" : "pending"}`}>
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

  return (
    <div className="document-list-container">
      <div className="document-list-header">
        <h2>{t("DocumentList.Title")}</h2>
        <Button
          icon="pi pi-refresh"
          className="p-button-rounded p-button-text"
          onClick={loadDocuments}
          loading={loading}
          tooltip={t("DocumentList.Refresh")}
          tooltipPosition="top"
        />
      </div>

      {documents.length === 0 ? (
        <div className="no-documents">
          <p>{t("DocumentList.NoDocuments")}</p>
        </div>
      ) : (
        <DataTable
          key={`${i18n.language}-${JSON.stringify(converterTypes)}`}
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
    </div>
  );
};
