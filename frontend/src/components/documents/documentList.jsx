import React, { useState, useEffect } from "react";
import { DataTable } from "primereact/datatable";
import { Column } from "primereact/column";
import { Button } from "primereact/button";
import { ProgressSpinner } from "primereact/progressspinner";
import { ProgressBar } from "primereact/progressbar";
import documentService from "../../services/documentService";
import { messageService } from "../../core/message/messageService";
import MarkdownViewer from "./markdownViewer";
import "./documentList.scss";

export const DocumentList = ({ refreshTrigger }) => {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedDocument, setSelectedDocument] = useState(null);
  const [showMarkdownViewer, setShowMarkdownViewer] = useState(false);
  const [converting, setConverting] = useState(null);
  const [webSockets, setWebSockets] = useState({}); // conversion_id -> websocket

  // Load documents on component mount and when refreshTrigger changes
  useEffect(() => {
    loadDocuments();
  }, [refreshTrigger]);

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
      messageService.errorToast("Failed to load documents");
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
      messageService.errorToast("Failed to view markdown");
      console.error("Error viewing markdown:", error);
    }
  };

  const handleConvert = async (document) => {
    try {
      setConverting(document.filename);
      messageService.infoToast("Starting conversion...");

      // Update document with conversion status
      setDocuments((prev) =>
        prev.map((doc) =>
          doc.filename === document.filename
            ? { ...doc, conversionProgress: 0, conversionStatus: "converting" }
            : doc
        )
      );

      // Start conversion (returns immediately with conversion_id)
      const response = await documentService.convertDocument(
        document.filename,
        "prompt_layout_all_en"
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
              messageService.successToast("Document converted successfully");
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
                progressData.message || "Conversion failed"
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
            messageService.errorToast("Connection error during conversion");
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
      messageService.errorToast("Failed to start conversion");
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
    return (
      <div className="action-buttons">
        {rowData.markdown_exists ? (
          <Button
            icon="pi pi-eye"
            className="p-button-rounded p-button-success"
            onClick={() => handleViewMarkdown(rowData)}
            tooltip="View Markdown"
            tooltipPosition="top"
          />
        ) : (
          <Button
            icon="pi pi-refresh"
            className="p-button-rounded p-button-warning"
            onClick={() => handleConvert(rowData)}
            loading={converting === rowData.filename}
            disabled={converting !== null}
            tooltip="Convert to Markdown"
            tooltipPosition="top"
          />
        )}
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
    return (
      <span className={`status-badge ${rowData.markdown_exists ? "converted" : "pending"}`}>
        {rowData.markdown_exists ? "Converted" : "Pending"}
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

  if (loading && documents.length === 0) {
    return (
      <div className="document-list-loading">
        <ProgressSpinner />
      </div>
    );
  }

  return (
    <div className="document-list-container">
      <div className="document-list-header">
        <h2>Uploaded Documents</h2>
        <Button
          icon="pi pi-refresh"
          className="p-button-rounded p-button-text"
          onClick={loadDocuments}
          loading={loading}
          tooltip="Refresh"
          tooltipPosition="top"
        />
      </div>

      {documents.length === 0 ? (
        <div className="no-documents">
          <p>No documents uploaded yet. Upload a document to get started.</p>
        </div>
      ) : (
        <DataTable
          value={documents}
          className="p-datatable-striped"
          responsiveLayout="scroll"
          paginator
          rows={10}
          rowsPerPageOptions={[5, 10, 20]}
          paginatorTemplate="RowsPerPageDropdown FirstPageLink PrevPageLink CurrentPageReport NextPageLink LastPageLink"
          currentPageReportTemplate="{first} to {last} of {totalRecords}"
        >
          <Column field="filename" header="Filename" style={{ width: "30%" }} />
          <Column
            field="file_size"
            header="Size"
            body={fileSizeBodyTemplate}
            style={{ width: "15%" }}
          />
          <Column
            field="upload_time"
            header="Upload Time"
            body={uploadTimeBodyTemplate}
            style={{ width: "25%" }}
          />
          <Column
            field="markdown_exists"
            header="Status"
            body={statusBodyTemplate}
            style={{ width: "15%" }}
          />
          <Column
            header="Progress"
            body={progressBodyTemplate}
            style={{ width: "20%" }}
          />
          <Column
            body={actionBodyTemplate}
            header="Actions"
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
