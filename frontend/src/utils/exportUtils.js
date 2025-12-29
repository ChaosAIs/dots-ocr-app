/**
 * Export utilities for converting markdown content to various formats
 */

/**
 * Download content as a file
 * @param {string} content - The content to download
 * @param {string} filename - The filename
 * @param {string} mimeType - The MIME type
 */
export const downloadFile = (content, filename, mimeType) => {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

/**
 * Export markdown content as .md file
 * @param {string} content - The markdown content
 * @param {string} filename - Optional filename (without extension)
 */
export const exportAsMarkdown = (content, filename = "export") => {
  downloadFile(content, `${filename}.md`, "text/markdown;charset=utf-8");
};

/**
 * Convert markdown to HTML for PDF/Word generation
 * @param {string} markdown - The markdown content
 * @returns {string} - HTML content
 */
const markdownToHtml = (markdown) => {
  // Basic markdown to HTML conversion
  let html = markdown
    // Headers
    .replace(/^### (.*$)/gim, "<h3>$1</h3>")
    .replace(/^## (.*$)/gim, "<h2>$1</h2>")
    .replace(/^# (.*$)/gim, "<h1>$1</h1>")
    // Bold
    .replace(/\*\*(.*?)\*\*/gim, "<strong>$1</strong>")
    .replace(/__(.*?)__/gim, "<strong>$1</strong>")
    // Italic
    .replace(/\*(.*?)\*/gim, "<em>$1</em>")
    .replace(/_(.*?)_/gim, "<em>$1</em>")
    // Code blocks
    .replace(/```(\w*)\n([\s\S]*?)```/gim, "<pre><code>$2</code></pre>")
    // Inline code
    .replace(/`(.*?)`/gim, "<code>$1</code>")
    // Links
    .replace(/\[(.*?)\]\((.*?)\)/gim, '<a href="$2">$1</a>')
    // Unordered lists
    .replace(/^\s*[-*+]\s+(.*$)/gim, "<li>$1</li>")
    // Ordered lists
    .replace(/^\s*\d+\.\s+(.*$)/gim, "<li>$1</li>")
    // Line breaks
    .replace(/\n\n/gim, "</p><p>")
    .replace(/\n/gim, "<br>");

  // Wrap in paragraphs
  html = `<p>${html}</p>`;

  // Wrap consecutive li elements in ul
  html = html.replace(/(<li>.*?<\/li>)+/gim, (match) => `<ul>${match}</ul>`);

  return html;
};

/**
 * Export markdown content as PDF
 * Uses browser print functionality for cross-browser compatibility
 * @param {string} content - The markdown content
 * @param {string} filename - Optional filename (without extension)
 */
export const exportAsPdf = (content, filename = "export") => {
  const html = markdownToHtml(content);

  // Create a styled print window
  const printWindow = window.open("", "_blank");
  printWindow.document.write(`
    <!DOCTYPE html>
    <html>
    <head>
      <title>${filename}</title>
      <style>
        body {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
          line-height: 1.6;
          max-width: 800px;
          margin: 0 auto;
          padding: 40px 20px;
          color: #333;
        }
        h1, h2, h3 { color: #222; margin-top: 1.5em; }
        h1 { font-size: 2em; border-bottom: 2px solid #eee; padding-bottom: 0.3em; }
        h2 { font-size: 1.5em; border-bottom: 1px solid #eee; padding-bottom: 0.3em; }
        h3 { font-size: 1.25em; }
        code {
          background: #f4f4f4;
          padding: 0.2em 0.4em;
          border-radius: 3px;
          font-family: 'Consolas', 'Monaco', monospace;
        }
        pre {
          background: #f4f4f4;
          padding: 1em;
          border-radius: 5px;
          overflow-x: auto;
        }
        pre code { background: none; padding: 0; }
        ul, ol { padding-left: 2em; }
        li { margin: 0.5em 0; }
        a { color: #0066cc; }
        table { border-collapse: collapse; width: 100%; margin: 1em 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background: #f4f4f4; }
        blockquote {
          border-left: 4px solid #ddd;
          margin: 1em 0;
          padding-left: 1em;
          color: #666;
        }
        @media print {
          body { padding: 0; }
        }
      </style>
    </head>
    <body>
      ${html}
    </body>
    </html>
  `);
  printWindow.document.close();

  // Wait for content to load, then trigger print
  printWindow.onload = () => {
    printWindow.print();
  };
};

/**
 * Export markdown content as Word document (.doc)
 * Uses HTML format that Word can open
 * @param {string} content - The markdown content
 * @param {string} filename - Optional filename (without extension)
 */
export const exportAsWord = (content, filename = "export") => {
  const html = markdownToHtml(content);

  // Create Word-compatible HTML
  const wordContent = `
    <!DOCTYPE html>
    <html xmlns:o="urn:schemas-microsoft-com:office:office"
          xmlns:w="urn:schemas-microsoft-com:office:word">
    <head>
      <meta charset="utf-8">
      <title>${filename}</title>
      <style>
        body { font-family: 'Calibri', sans-serif; line-height: 1.6; }
        h1 { font-size: 24pt; color: #2c3e50; }
        h2 { font-size: 18pt; color: #2c3e50; }
        h3 { font-size: 14pt; color: #2c3e50; }
        code { font-family: 'Consolas', monospace; background: #f4f4f4; }
        pre { background: #f4f4f4; padding: 10px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #000; padding: 8px; }
      </style>
    </head>
    <body>
      ${html}
    </body>
    </html>
  `;

  const blob = new Blob([wordContent], {
    type: "application/msword;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${filename}.doc`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

/**
 * Parse markdown tables and convert to array data
 * @param {string} markdown - The markdown content
 * @returns {Array<Array<string>>} - 2D array of table data
 */
const parseMarkdownTables = (markdown) => {
  const rows = [];
  const lines = markdown.split("\n");

  for (const line of lines) {
    // Skip separator lines (|---|---|)
    if (/^\|[\s-:|]+\|$/.test(line.trim())) continue;

    // Parse table rows
    if (line.trim().startsWith("|") && line.trim().endsWith("|")) {
      const cells = line
        .split("|")
        .slice(1, -1) // Remove first and last empty elements
        .map((cell) => cell.trim());
      rows.push(cells);
    }
  }

  return rows;
};

/**
 * Export markdown content as Excel file (.csv)
 * Extracts tables from markdown or creates a simple text export
 * @param {string} content - The markdown content
 * @param {string} filename - Optional filename (without extension)
 */
export const exportAsExcel = (content, filename = "export") => {
  // Try to extract tables from markdown
  const tableData = parseMarkdownTables(content);

  let csvContent;

  if (tableData.length > 0) {
    // Export table data as CSV
    csvContent = tableData
      .map((row) =>
        row
          .map((cell) => {
            // Escape quotes and wrap in quotes if contains comma
            const escaped = cell.replace(/"/g, '""');
            return cell.includes(",") || cell.includes('"')
              ? `"${escaped}"`
              : escaped;
          })
          .join(",")
      )
      .join("\n");
  } else {
    // No tables found, export as simple text in single column
    csvContent = "Content\n" + content.replace(/"/g, '""').replace(/\n/g, " ");
  }

  // Add BOM for Excel to recognize UTF-8
  const bom = "\uFEFF";
  const blob = new Blob([bom + csvContent], {
    type: "text/csv;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${filename}.csv`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

/**
 * Generate a filename from content
 * @param {string} content - The content to generate filename from
 * @param {number} maxLength - Maximum length of filename
 * @returns {string} - Generated filename
 */
export const generateFilename = (content, maxLength = 50) => {
  // Take first line or first N characters
  const firstLine = content.split("\n")[0] || "export";

  // Remove markdown formatting
  let filename = firstLine
    .replace(/^#+\s*/, "") // Remove heading markers
    .replace(/\*\*/g, "") // Remove bold
    .replace(/\*/g, "") // Remove italic
    .replace(/`/g, "") // Remove code
    .trim();

  // Replace invalid filename characters
  filename = filename.replace(/[<>:"/\\|?*]/g, "_");

  // Truncate if too long
  if (filename.length > maxLength) {
    filename = filename.substring(0, maxLength);
  }

  // Add timestamp if filename is empty or too short
  if (filename.length < 3) {
    filename = `export_${new Date().toISOString().slice(0, 10)}`;
  }

  return filename;
};
