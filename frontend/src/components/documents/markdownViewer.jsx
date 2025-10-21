import React, { useState, useEffect } from "react";
import { Dialog } from "primereact/dialog";
import { Button } from "primereact/button";
import { ProgressSpinner } from "primereact/progressspinner";
import ReactMarkdown from "react-markdown";
import documentService from "../../services/documentService";
import { messageService } from "../../core/message/messageService";
import "./markdownViewer.scss";

const MarkdownViewer = ({ document, visible, onHide }) => {
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (visible && document) {
      loadMarkdownContent();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, document]);

  const loadMarkdownContent = async () => {
    try {
      setLoading(true);
      const filename = document.filename.split(".")[0]; // Remove extension
      const response = await documentService.getMarkdownContent(filename);
      if (response.status === "success") {
        setContent(response.content);
      }
    } catch (error) {
      messageService.errorToast("Failed to load markdown content");
      console.error("Error loading markdown:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    const element = document.createElement("a");
    const file = new Blob([content], { type: "text/markdown" });
    element.href = URL.createObjectURL(file);
    element.download = `${document.filename.split(".")[0]}.md`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const headerTemplate = (
    <div className="markdown-viewer-header">
      <span>{document?.filename}</span>
      <Button
        icon="pi pi-download"
        className="p-button-rounded p-button-text"
        onClick={handleDownload}
        tooltip="Download Markdown"
        tooltipPosition="left"
      />
    </div>
  );

  return (
    <Dialog
      header={headerTemplate}
      visible={visible}
      onHide={onHide}
      modal
      maximizable
      style={{ width: "90vw", height: "90vh" }}
      className="markdown-viewer-dialog"
    >
      <div className="markdown-viewer-content">
        {loading ? (
          <div className="markdown-loading">
            <ProgressSpinner />
          </div>
        ) : (
          <div className="markdown-body">
            <ReactMarkdown>{content}</ReactMarkdown>
          </div>
        )}
      </div>
    </Dialog>
  );
};

export default MarkdownViewer;

