import { useState, useEffect, useMemo, useRef } from "react";
import { Dialog } from "primereact/dialog";
import { Button } from "primereact/button";
import { ProgressSpinner } from "primereact/progressspinner";
import { InputText } from "primereact/inputtext";
import { TabView, TabPanel } from "primereact/tabview";
import { InputTextarea } from "primereact/inputtextarea";
import { useTranslation } from "react-i18next";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";
import rehypeRaw from "rehype-raw";
import "katex/dist/katex.min.css";
import SyntaxHighlighter from "react-syntax-highlighter";
import { atomOneDark } from "react-syntax-highlighter/dist/esm/styles/hljs";
import Lightbox from "yet-another-react-lightbox";
import "yet-another-react-lightbox/styles.css";
import documentService from "../../services/documentService";
import { messageService } from "../../core/message/messageService";
import "./markdownViewer.scss";

const MarkdownViewer = ({ document, visible, onHide }) => {
  const { t } = useTranslation();
  const [content, setContent] = useState("");
  const [editedContent, setEditedContent] = useState("");
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [activeTabIndex, setActiveTabIndex] = useState(0);
  const [searchTerm, setSearchTerm] = useState("");
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const [lightboxIndex, setLightboxIndex] = useState(0);
  const [images, setImages] = useState([]);
  const [pageImages, setPageImages] = useState([]); // Array of {pageNo, imageUrl}
  const [pageImageModalOpen, setPageImageModalOpen] = useState(false);
  const [selectedPageImage, setSelectedPageImage] = useState(null);
  const contentRef = useRef(null);
  const imagePanelRef = useRef(null);

  useEffect(() => {
    if (visible && document) {
      loadMarkdownContent();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, document]);

  // Extract table of contents from markdown with page numbers
  const tableOfContents = useMemo(() => {
    const headings = [];
    const lines = content.split("\n");
    let currentPage = pageImages.length > 0 ? pageImages[0].pageNo : 1;

    lines.forEach((line) => {
      // Check if this line indicates a page separator (matches "*Page 18*" or "*Page18*")
      const pageMatch = line.match(/^\*.*?(\d+)\*$/);
      if (pageMatch && line.toLowerCase().includes('page')) {
        currentPage = parseInt(pageMatch[1], 10);
        return;
      }

      // Check for heading
      const match = line.match(/^(#{1,6})\s+(.+)$/);
      if (match) {
        const level = match[1].length;
        const title = match[2];
        const id = `heading-${title}`;
        headings.push({ level, title, id, pageNo: currentPage });
      }
    });
    return headings;
  }, [content, pageImages]);

  // Extract images from markdown
  const extractImages = (markdown) => {
    const imageRegex = /!\[([^\]]*)\]\(([^)]+)\)/g;
    const imgs = [];
    let match;

    while ((match = imageRegex.exec(markdown)) !== null) {
      const alt = match[1];
      const src = match[2];
      // Only add non-base64 images to lightbox (base64 images are too large)
      if (!src.startsWith('data:')) {
        imgs.push({ src, alt });
      }
    }
    setImages(imgs);
  };

  const loadMarkdownContent = async () => {
    try {
      setLoading(true);
      const filename = document.filename.split(".")[0];

      const filesResponse = await documentService.getMarkdownFiles(filename);
      if (filesResponse.status === "success") {
        let combinedContent = "";
        const imageUrls = [];

        for (const file of filesResponse.markdown_files) {
          try {
            const contentResponse = await documentService.getMarkdownContent(
              filename,
              file.page_no
            );
            if (contentResponse.status === "success") {
              // Debug logging to see what content we're actually receiving
              console.log(`Page ${file.page_no} content preview:`, contentResponse.content.substring(0, 200));
              console.log(`Page ${file.page_no} has base64 images:`, contentResponse.content.includes('data:image/'));

              if (filesResponse.markdown_files.length > 1) {
                combinedContent += `\n\n---\n\n`;
              }
              combinedContent += contentResponse.content;

              if (filesResponse.markdown_files.length > 1) {
                const pageLabel = file.page_no !== null ? `${t("MarkdownViewer.Page")} ${file.page_no}` : t("MarkdownViewer.Combined");
                combinedContent += `\n\n---\n\n*${pageLabel}*`;
              }

              // Get the image URL for this page
              const imageUrl = documentService.getImageUrl(filename, file.page_no);
              imageUrls.push({
                pageNo: file.page_no,
                imageUrl: imageUrl
              });
            }
          } catch (error) {
            console.error(`Error loading page ${file.page_no}:`, error);
          }
        }

        if (combinedContent) {
          // Debug logging for final combined content
          console.log('Final combined content preview:', combinedContent.substring(0, 500));
          console.log('Final content has base64 images:', combinedContent.includes('data:image/'));
          console.log('Final content has relative image paths:', /!\[.*?\]\([^)]*(?<!data:image\/[^)]*)\)/g.test(combinedContent));

          setContent(combinedContent);
          setEditedContent(combinedContent);
          setPageImages(imageUrls);
          extractImages(combinedContent);
          setHasUnsavedChanges(false);
        }
      }
    } catch (error) {
      messageService.errorToast(t("MarkdownViewer.FailedToLoad"));
      console.error("Error loading markdown:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveMarkdown = async () => {
    try {
      setSaving(true);
      const filename = document.filename.split(".")[0];

      // For multi-page documents, we need to save each page separately
      const filesResponse = await documentService.getMarkdownFiles(filename);
      if (filesResponse.status === "success" && filesResponse.markdown_files.length > 1) {
        // Split the edited content by the pattern we use when combining pages
        // The pattern is: \n\n---\n\n[content]\n\n---\n\n*Page X*\n\n---\n\n[content]\n\n---\n\n*Page Y*
        // We need to extract just the content parts, not the page labels

        // First, split by the separator
        const sections = editedContent.split(/\n\n---\n\n/);

        // Extract page contents by filtering out empty sections and page labels
        const pageContents = [];
        for (const section of sections) {
          const trimmed = section.trim();
          // Skip empty sections and page label sections (like "*Page 1*" or "*page 1*")
          if (trimmed && !trimmed.match(/^\*(?:Page|page)\s*\d+\*$/i)) {
            pageContents.push(trimmed);
          }
        }

        console.log('Saving multi-page document:', {
          totalSections: sections.length,
          pageContents: pageContents.length,
          expectedPages: filesResponse.markdown_files.length,
          firstPagePreview: pageContents[0]?.substring(0, 100)
        });

        // Save each page
        if (pageContents.length !== filesResponse.markdown_files.length) {
          console.warn(`Mismatch: Found ${pageContents.length} page contents but expected ${filesResponse.markdown_files.length} pages`);
        }

        for (let i = 0; i < filesResponse.markdown_files.length && i < pageContents.length; i++) {
          const file = filesResponse.markdown_files[i];
          const pageContent = pageContents[i];
          console.log(`Saving page ${file.page_no}, content length: ${pageContent.length}`);
          await documentService.saveMarkdownContent(filename, pageContent, file.page_no);
        }
      } else {
        // Single page document
        await documentService.saveMarkdownContent(filename, editedContent);
      }

      setContent(editedContent);
      setHasUnsavedChanges(false);
      messageService.successToast(t("MarkdownViewer.SaveSuccess"));

      // Reload the content to ensure we're in sync
      await loadMarkdownContent();
    } catch (error) {
      messageService.errorToast(t("MarkdownViewer.SaveFailed"));
      console.error("Error saving markdown:", error);
    } finally {
      setSaving(false);
    }
  };

  const handleEditorChange = (e) => {
    setEditedContent(e.target.value);
    setHasUnsavedChanges(true);
  };

  const handleDialogHide = () => {
    if (hasUnsavedChanges) {
      if (window.confirm(t("MarkdownViewer.UnsavedChanges"))) {
        setHasUnsavedChanges(false);
        onHide();
      }
    } else {
      onHide();
    }
  };

  const highlightedContent = useMemo(() => {
    return content;
  }, [content]);

  const handleDownload = () => {
    const element = window.document.createElement("a");
    const file = new Blob([content], { type: "text/markdown" });
    element.href = URL.createObjectURL(file);
    element.download = `${document.filename.split(".")[0]}.md`;
    window.document.body.appendChild(element);
    element.click();
    window.document.body.removeChild(element);
  };

  const handleCopyCode = (code) => {
    navigator.clipboard.writeText(code);
    messageService.successToast(t("MarkdownViewer.CodeCopied"));
  };

  const handleImageClick = (index) => {
    setLightboxIndex(index);
    setLightboxOpen(true);
  };

  const handlePageImageClick = (pageImage) => {
    setSelectedPageImage(pageImage);
    setPageImageModalOpen(true);
  };

  const handlePageImageModalClose = () => {
    setPageImageModalOpen(false);
    setSelectedPageImage(null);
  };

  const scrollToHeading = (id, pageNo) => {
    // Scroll markdown content to the heading
    if (contentRef.current) {
      const element = contentRef.current.querySelector(`[id="${id}"]`);
      if (element) {
        element.scrollIntoView({ behavior: "smooth" });
      }
    }

    // Scroll image panel to the corresponding page
    if (imagePanelRef.current && pageNo) {
      const pageImageContainer = imagePanelRef.current.querySelector(
        `[data-page-no="${pageNo}"]`
      );
      if (pageImageContainer) {
        pageImageContainer.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }
  };

  const markdownComponents = {
    code({ inline, children, ...props }) {
      if (inline) {
        return (
          <code className="inline-code" {...props}>
            {children}
          </code>
        );
      }
      return null;
    },
    pre({ children, ...props }) {
      const codeElement = children?.[0];
      const codeContent = codeElement?.props?.children || "";
      const className = codeElement?.props?.className || "";
      const match = /language-(\w+)/.exec(className);
      const language = match ? match[1] : "text";
      const code = String(codeContent).replace(/\n$/, "");

      return (
        <div className="code-block-wrapper">
          <div className="code-block-header">
            <span className="language-label">{language}</span>
            <Button
              icon="pi pi-copy"
              className="p-button-rounded p-button-text p-button-sm"
              onClick={() => handleCopyCode(code)}
              tooltip={t("MarkdownViewer.CopyCode")}
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
      if (!src) {
        console.warn('Image component received empty src');
        return null;
      }

      const imageIndex = images.findIndex((img) => img.src === src);
      const isBase64 = src.startsWith('data:');

      // Debug logging for image rendering
      console.log('Rendering image:', {
        isBase64,
        srcPreview: src.substring(0, 100),
        alt,
        imageIndex
      });

      return (
        <img
          src={src}
          alt={alt}
          className="markdown-image"
          onClick={() => !isBase64 && imageIndex >= 0 && handleImageClick(imageIndex)}
          style={{ cursor: isBase64 ? 'default' : 'pointer', maxWidth: '100%' }}
          onError={(e) => {
            console.error('Image failed to load:', {
              src: src.substring(0, 100),
              error: e
            });
          }}
          onLoad={() => {
            console.log('Image loaded successfully:', src.substring(0, 50));
          }}
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
    table({ children, ...props }) {
      return (
        <table className="markdown-table" {...props}>
          {children}
        </table>
      );
    },
    thead({ children, ...props }) {
      return <thead {...props}>{children}</thead>;
    },
    tbody({ children, ...props }) {
      return <tbody {...props}>{children}</tbody>;
    },
    tr({ children, ...props }) {
      return <tr {...props}>{children}</tr>;
    },
    th({ children, ...props }) {
      return <th {...props}>{children}</th>;
    },
    td({ children, ...props }) {
      return <td {...props}>{children}</td>;
    },
  };

  const headerTemplate = (
    <div className="markdown-viewer-header">
      <span>{document?.filename}</span>
      <div className="header-actions">
        <Button
          icon="pi pi-download"
          className="p-button-rounded p-button-text"
          onClick={handleDownload}
          tooltip={t("MarkdownViewer.DownloadMarkdown")}
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
        onHide={handleDialogHide}
        modal
        maximizable
        style={{ width: "95vw", height: "95vh" }}
        className="markdown-viewer-dialog"
      >
        <div className="markdown-viewer-wrapper">
          {/* Table of Contents - Only for multi-page documents */}
          {pageImages.length > 1 && tableOfContents.length > 0 && (
            <div className="markdown-toc-panel">
              <div className="toc-panel-header">
                <h3>{t("MarkdownViewer.TableOfContents")}</h3>
              </div>
              <div className="toc-panel-content">
                {tableOfContents.map((heading, index) => (
                  <div
                    key={index}
                    className={`toc-item toc-level-${heading.level}`}
                    onClick={() => scrollToHeading(heading.id, heading.pageNo)}
                  >
                    <span className="toc-title">{heading.title}</span>
                    {heading.pageNo && (
                      <span className="toc-page-badge">{heading.pageNo}</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Image Panel in the Middle */}
          {pageImages.length > 0 && (
            <div className="markdown-image-panel">
              <div className="image-panel-header">
                <h3>{t("MarkdownViewer.DocumentImages")}</h3>
              </div>
              <div className="image-panel-content" ref={imagePanelRef}>
                {pageImages.map((pageImage, index) => (
                  <div
                    key={index}
                    className="page-image-container"
                    data-page-no={pageImage.pageNo}
                  >
                    {pageImages.length > 1 && (
                      <div className="page-label">
                        {t("MarkdownViewer.Page")} {pageImage.pageNo}
                      </div>
                    )}
                    <img
                      src={pageImage.imageUrl}
                      alt={`Page ${pageImage.pageNo}`}
                      className="page-image clickable"
                      onClick={() => handlePageImageClick(pageImage)}
                      onError={(e) => {
                        console.error(`Failed to load image for page ${pageImage.pageNo}`);
                        e.target.style.display = 'none';
                      }}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Markdown Content on the Right */}
          <div className="markdown-viewer-content">
            {loading ? (
              <div className="markdown-loading">
                <ProgressSpinner />
              </div>
            ) : (
              <TabView activeIndex={activeTabIndex} onTabChange={(e) => setActiveTabIndex(e.index)}>
                {/* Preview Tab */}
                <TabPanel header={t("MarkdownViewer.PreviewTab")} leftIcon="pi pi-eye">
                  <div className="search-bar">
                    <InputText
                      placeholder={t("MarkdownViewer.SearchPlaceholder")}
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
                  <div className="markdown-body" ref={contentRef}>
                    <ReactMarkdown
                      components={markdownComponents}
                      remarkPlugins={[remarkGfm, remarkMath]}
                      rehypePlugins={[rehypeRaw, rehypeKatex]}
                      urlTransform={(url) => {
                        // Preserve all URLs including base64 data URLs
                        console.log('URL transform:', url?.substring(0, 100));
                        return url;
                      }}
                    >
                      {highlightedContent}
                    </ReactMarkdown>
                  </div>
                </TabPanel>

                {/* Edit Tab */}
                <TabPanel header={t("MarkdownViewer.EditTab")} leftIcon="pi pi-pencil">
                  <div className="editor-toolbar">
                    <Button
                      label={saving ? t("MarkdownViewer.SavingChanges") : t("MarkdownViewer.SaveChanges")}
                      icon={saving ? "pi pi-spin pi-spinner" : "pi pi-save"}
                      onClick={handleSaveMarkdown}
                      disabled={!hasUnsavedChanges || saving}
                      className="p-button-success"
                    />
                    {hasUnsavedChanges && (
                      <span className="unsaved-indicator">
                        <i className="pi pi-exclamation-circle"></i> Unsaved changes
                      </span>
                    )}
                  </div>
                  <div className="editor-container">
                    <InputTextarea
                      value={editedContent}
                      onChange={handleEditorChange}
                      className="markdown-editor"
                      rows={30}
                      autoResize={false}
                    />
                  </div>
                </TabPanel>
              </TabView>
            )}
          </div>
        </div>
      </Dialog>

      <Lightbox
        open={lightboxOpen}
        close={() => setLightboxOpen(false)}
        slides={images.map((img) => ({ src: img.src }))}
        index={lightboxIndex}
      />

      {/* Page Image Preview Modal */}
      <Dialog
        header={
          selectedPageImage
            ? `${t("MarkdownViewer.Page")} ${selectedPageImage.pageNo}`
            : t("MarkdownViewer.ImagePreview")
        }
        visible={pageImageModalOpen}
        onHide={handlePageImageModalClose}
        modal
        maximizable
        style={{ width: "90vw" }}
        className="page-image-preview-dialog"
      >
        {selectedPageImage && (
          <div className="page-image-preview-container">
            <img
              src={selectedPageImage.imageUrl}
              alt={`Page ${selectedPageImage.pageNo}`}
              className="page-image-preview"
              onError={() => {
                console.error(`Failed to load preview image for page ${selectedPageImage.pageNo}`);
              }}
            />
          </div>
        )}
      </Dialog>
    </>
  );
};

export default MarkdownViewer;