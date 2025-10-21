import React, { useState, useEffect, useMemo } from "react";
import { Dialog } from "primereact/dialog";
import { Button } from "primereact/button";
import { ProgressSpinner } from "primereact/progressspinner";
import { InputText } from "primereact/inputtext";
import { Sidebar } from "primereact/sidebar";
import ReactMarkdown from "react-markdown";
import SyntaxHighlighter from "react-syntax-highlighter";
import { atomOneDark } from "react-syntax-highlighter/dist/esm/styles/hljs";
import Lightbox from "yet-another-react-lightbox";
import "yet-another-react-lightbox/styles.css";
import documentService from "../../services/documentService";
import { messageService } from "../../core/message/messageService";
import "./markdownViewer.scss";

const MarkdownViewer = ({ document, visible, onHide }) => {
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");
  const [showTOC, setShowTOC] = useState(true);
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const [lightboxIndex, setLightboxIndex] = useState(0);
  const [images, setImages] = useState([]);

  useEffect(() => {
    if (visible && document) {
      loadMarkdownContent();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, document]);

  // Extract table of contents from markdown
  const tableOfContents = useMemo(() => {
    const headings = [];
    const lines = content.split("\n");
    lines.forEach((line, index) => {
      const match = line.match(/^(#{1,6})\s+(.+)$/);
      if (match) {
        const level = match[1].length;
        const title = match[2];
        const id = `heading-${index}`;
        headings.push({ level, title, id });
      }
    });
    return headings;
  }, [content]);

  // Extract images from markdown
  const extractImages = (markdown) => {
    const imageRegex = /!\[.*?\]\((.*?)\)/g;
    const imgs = [];
    let match;
    while ((match = imageRegex.exec(markdown)) !== null) {
      imgs.push({ src: match[1], alt: match[0] });
    }
    setImages(imgs);
  };

  const loadMarkdownContent = async () => {
    try {
      setLoading(true);
      const filename = document.filename.split(".")[0]; // Remove extension
      const response = await documentService.getMarkdownContent(filename);
      if (response.status === "success") {
        setContent(response.content);
        extractImages(response.content);
      }
    } catch (error) {
      messageService.errorToast("Failed to load markdown content");
      console.error("Error loading markdown:", error);
    } finally {
      setLoading(false);
    }
  };

  // Highlight search term in content
  const highlightedContent = useMemo(() => {
    if (!searchTerm) return content;
    const regex = new RegExp(`(${searchTerm})`, "gi");
    return content.replace(regex, `<mark>$1</mark>`);
  }, [content, searchTerm]);

  const handleDownload = () => {
    const element = document.createElement("a");
    const file = new Blob([content], { type: "text/markdown" });
    element.href = URL.createObjectURL(file);
    element.download = `${document.filename.split(".")[0]}.md`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  const handleCopyCode = (code) => {
    navigator.clipboard.writeText(code);
    messageService.successToast("Code copied to clipboard");
  };

  const handleImageClick = (index) => {
    setLightboxIndex(index);
    setLightboxOpen(true);
  };

  const scrollToHeading = (id) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth" });
    }
  };

  // Custom markdown components
  const markdownComponents = {
    code({ node, inline, className, children, ...props }) {
      const match = /language-(\w+)/.exec(className || "");
      const language = match ? match[1] : "text";
      const code = String(children).replace(/\n$/, "");

      if (inline) {
        return (
          <code className="inline-code" {...props}>
            {children}
          </code>
        );
      }

      return (
        <div className="code-block-wrapper">
          <div className="code-block-header">
            <span className="language-label">{language}</span>
            <Button
              icon="pi pi-copy"
              className="p-button-rounded p-button-text p-button-sm"
              onClick={() => handleCopyCode(code)}
              tooltip="Copy code"
              tooltipPosition="left"
            />
          </div>
          <SyntaxHighlighter
            language={language}
            style={atomOneDark}
            className="code-block"
            {...props}
          >
            {code}
          </SyntaxHighlighter>
        </div>
      );
    },
    img({ src, alt, ...props }) {
      const imageIndex = images.findIndex((img) => img.src === src);
      return (
        <img
          src={src}
          alt={alt}
          className="markdown-image"
          onClick={() => handleImageClick(imageIndex)}
          {...props}
        />
      );
    },
    h1({ children, ...props }) {
      const id = `heading-${children}`;
      return (
        <h1 id={id} {...props}>
          {children}
        </h1>
      );
    },
    h2({ children, ...props }) {
      const id = `heading-${children}`;
      return (
        <h2 id={id} {...props}>
          {children}
        </h2>
      );
    },
    h3({ children, ...props }) {
      const id = `heading-${children}`;
      return (
        <h3 id={id} {...props}>
          {children}
        </h3>
      );
    },
  };

  const headerTemplate = (
    <div className="markdown-viewer-header">
      <span>{document?.filename}</span>
      <div className="header-actions">
        <Button
          icon={showTOC ? "pi pi-times" : "pi pi-bars"}
          className="p-button-rounded p-button-text"
          onClick={() => setShowTOC(!showTOC)}
          tooltip={showTOC ? "Hide TOC" : "Show TOC"}
          tooltipPosition="left"
        />
        <Button
          icon="pi pi-download"
          className="p-button-rounded p-button-text"
          onClick={handleDownload}
          tooltip="Download Markdown"
          tooltipPosition="left"
        />
      </div>
    </div>
  );

  return (
    <>
      <Dialog
        header={headerTemplate}
        visible={visible}
        onHide={onHide}
        modal
        maximizable
        style={{ width: "95vw", height: "95vh" }}
        className="markdown-viewer-dialog"
      >
        <div className="markdown-viewer-wrapper">
          {/* Table of Contents Sidebar */}
          <Sidebar
            visible={showTOC && tableOfContents.length > 0}
            onHide={() => setShowTOC(false)}
            position="left"
            className="markdown-toc-sidebar"
            style={{ width: "250px" }}
          >
            <div className="toc-header">
              <h3>Table of Contents</h3>
            </div>
            <div className="toc-content">
              {tableOfContents.map((heading, index) => (
                <div
                  key={index}
                  className={`toc-item toc-level-${heading.level}`}
                  onClick={() => scrollToHeading(heading.id)}
                >
                  {heading.title}
                </div>
              ))}
            </div>
          </Sidebar>

          {/* Main Content */}
          <div className="markdown-viewer-content">
            {/* Search Bar */}
            <div className="search-bar">
              <InputText
                placeholder="Search in document..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="search-input"
              />
              {searchTerm && (
                <Button
                  icon="pi pi-times"
                  className="p-button-rounded p-button-text p-button-sm"
                  onClick={() => setSearchTerm("")}
                />
              )}
            </div>

            {/* Content */}
            {loading ? (
              <div className="markdown-loading">
                <ProgressSpinner />
              </div>
            ) : (
              <div className="markdown-body">
                <ReactMarkdown components={markdownComponents}>
                  {highlightedContent}
                </ReactMarkdown>
              </div>
            )}
          </div>
        </div>
      </Dialog>

      {/* Image Lightbox */}
      <Lightbox
        open={lightboxOpen}
        close={() => setLightboxOpen(false)}
        slides={images.map((img) => ({ src: img.src }))}
        index={lightboxIndex}
      />
    </>
  );
};

export default MarkdownViewer;

