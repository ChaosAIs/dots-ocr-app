import http, { createFileUploadClient } from "../core/http/httpClient";
import APP_CONFIG from "../core/config/appConfig";

/**
 * Document Service - handles all document-related API calls
 */
class DocumentService {
  constructor() {
    this.apiDomain = APP_CONFIG.apiDomain || "http://localhost:8080";
  }

  /**
   * Upload a document file
   * @param {File} file - The file to upload
   * @returns {Promise} - Upload response
   */
  async uploadDocument(file) {
    try {
      const formData = new FormData();
      formData.append("file", file);

      const fileUploadClient = createFileUploadClient();
      const response = await fileUploadClient.post(
        `${this.apiDomain}/upload`,
        formData
      );

      return response.data;
    } catch (error) {
      console.error("Error uploading document:", error);
      throw error;
    }
  }

  /**
   * Get list of all uploaded documents
   * @returns {Promise} - List of documents with their status
   */
  async getDocuments() {
    try {
      const response = await http.get(`${this.apiDomain}/documents`);
      return response.data;
    } catch (error) {
      console.error("Error fetching documents:", error);
      throw error;
    }
  }

  /**
   * Detect if a file should use doc_service converter
   * @param {string} filename - The filename to check
   * @returns {boolean} - True if file should use doc_service
   */
  isDocServiceFile(filename) {
    const docServiceExtensions = [
      '.docx', '.doc',           // Word documents
      '.xlsx', '.xlsm', '.xls',  // Excel spreadsheets
      '.txt', '.csv', '.tsv',    // Plain text files
      '.log', '.text'            // Other text formats
    ];

    const extension = '.' + filename.split('.').pop().toLowerCase();
    return docServiceExtensions.includes(extension);
  }

  /**
   * Detect if a file is an image that can use DeepSeek OCR
   * @param {string} filename - The filename to check
   * @returns {boolean} - True if file is an image
   */
  isImageFile(filename) {
    const imageExtensions = [
      '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'
    ];

    const extension = '.' + filename.split('.').pop().toLowerCase();
    return imageExtensions.includes(extension);
  }

  /**
   * Start document conversion using doc_service (non-blocking)
   * For Word, Excel, and text files
   * @param {string} filename - The filename to convert
   * @returns {Promise} - Response with conversion_id
   */
  async convertDocumentWithDocService(filename) {
    try {
      const formData = new FormData();
      formData.append("filename", filename);

      const response = await http.post(
        `${this.apiDomain}/convert-doc`,
        formData
      );

      return response.data;
    } catch (error) {
      console.error("Error starting doc_service conversion:", error);
      throw error;
    }
  }

  /**
   * Start document conversion using OCR parser (non-blocking)
   * For PDF and image files
   * @param {string} filename - The filename to convert
   * @param {string} promptMode - The prompt mode to use
   * @returns {Promise} - Response with conversion_id
   */
  async convertDocumentWithOCR(filename, promptMode = "prompt_layout_all_en") {
    try {
      const formData = new FormData();
      formData.append("filename", filename);
      formData.append("prompt_mode", promptMode);

      const response = await http.post(
        `${this.apiDomain}/convert`,
        formData
      );

      return response.data;
    } catch (error) {
      console.error("Error starting OCR conversion:", error);
      throw error;
    }
  }

  /**
   * Start image conversion using DeepSeek OCR service (non-blocking)
   * For image files only
   * @param {string} filename - The filename to convert
   * @returns {Promise} - Response with conversion_id
   */
  async convertDocumentWithDeepSeekOCR(filename) {
    try {
      const formData = new FormData();
      formData.append("filename", filename);

      const response = await http.post(
        `${this.apiDomain}/convert-deepseek`,
        formData
      );

      return response.data;
    } catch (error) {
      console.error("Error starting DeepSeek OCR conversion:", error);
      throw error;
    }
  }

  /**
   * Start document conversion (non-blocking)
   * Automatically detects file type and routes to appropriate converter
   * @param {string} filename - The filename to convert
   * @param {string} promptMode - The prompt mode to use (only for OCR)
   * @param {string} converterType - Optional: 'auto', 'doc_service', 'dots_ocr_service', or 'deepseek_ocr'
   * @returns {Promise} - Response with conversion_id
   */
  async convertDocument(filename, promptMode = "prompt_layout_all_en", converterType = "auto") {
    try {
      // If converter type is explicitly specified, use it
      if (converterType === "deepseek_ocr") {
        console.log(`Using DeepSeek OCR converter for: ${filename}`);
        return await this.convertDocumentWithDeepSeekOCR(filename);
      } else if (converterType === "doc_service") {
        console.log(`Using doc_service converter for: ${filename}`);
        return await this.convertDocumentWithDocService(filename);
      } else if (converterType === "dots_ocr_service") {
        console.log(`Using DOTS OCR parser for: ${filename}`);
        return await this.convertDocumentWithOCR(filename, promptMode);
      }

      // Auto-detect file type and route to appropriate endpoint
      if (this.isDocServiceFile(filename)) {
        console.log(`Auto-detected: Using doc_service converter for: ${filename}`);
        return await this.convertDocumentWithDocService(filename);
      } else {
        console.log(`Auto-detected: Using OCR parser for: ${filename}`);
        return await this.convertDocumentWithOCR(filename, promptMode);
      }
    } catch (error) {
      console.error("Error starting conversion:", error);
      throw error;
    }
  }

  /**
   * Get conversion status
   * @param {string} conversionId - The conversion task ID
   * @returns {Promise} - Conversion status
   */
  async getConversionStatus(conversionId) {
    try {
      const response = await http.get(
        `${this.apiDomain}/conversion-status/${conversionId}`
      );
      return response.data;
    } catch (error) {
      console.error("Error getting conversion status:", error);
      throw error;
    }
  }

  /**
   * Connect to WebSocket for real-time progress updates
   * @param {string} conversionId - The conversion task ID
   * @param {Function} onMessage - Callback for progress updates
   * @param {Function} onError - Callback for errors
   * @returns {WebSocket} - WebSocket connection
   */
  connectToConversionProgress(conversionId, onMessage, onError) {
    try {
      const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${wsProtocol}//${this.apiDomain.replace(/^https?:\/\//, "")}/ws/conversion/${conversionId}`;

      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        console.log("WebSocket connected for conversion:", conversionId);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (onMessage) {
            onMessage(data);
          }
        } catch (error) {
          console.error("Error parsing WebSocket message:", error);
        }
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        if (onError) {
          onError(error);
        }
      };

      ws.onclose = () => {
        console.log("WebSocket closed for conversion:", conversionId);
      };

      return ws;
    } catch (error) {
      console.error("Error connecting to WebSocket:", error);
      throw error;
    }
  }

  /**
   * Get list of markdown files for a document
   * @param {string} filename - The filename (without extension)
   * @returns {Promise} - List of markdown files
   */
  async getMarkdownFiles(filename) {
    try {
      const response = await http.get(
        `${this.apiDomain}/markdown-files/${filename}`
      );
      return response.data;
    } catch (error) {
      console.error("Error fetching markdown files:", error);
      throw error;
    }
  }

  /**
   * Get markdown content of a converted document
   * @param {string} filename - The filename (without extension)
   * @param {number} pageNo - Optional page number for multi-page documents
   * @returns {Promise} - Markdown content
   */
  async getMarkdownContent(filename, pageNo = null) {
    try {
      let url = `${this.apiDomain}/markdown/${filename}`;
      if (pageNo !== null) {
        url += `?page_no=${pageNo}`;
      }
      const response = await http.get(url);
      return response.data;
    } catch (error) {
      console.error("Error fetching markdown content:", error);
      throw error;
    }
  }

  /**
   * Update markdown content of a converted document
   * @param {string} filename - The filename (without extension)
   * @param {string} content - The new markdown content
   * @param {number} pageNo - Optional page number for multi-page documents
   * @returns {Promise} - Update response
   */
  async saveMarkdownContent(filename, content, pageNo = null) {
    try {
      let url = `${this.apiDomain}/markdown/${filename}`;
      if (pageNo !== null) {
        url += `?page_no=${pageNo}`;
      }
      const response = await http.put(url, { content });
      return response.data;
    } catch (error) {
      console.error("Error saving markdown content:", error);
      throw error;
    }
  }

  /**
   * Get the URL for the JPG image of a converted document
   * @param {string} filename - The filename (without extension)
   * @param {number} pageNo - Optional page number for multi-page documents
   * @returns {string} - Image URL
   */
  getImageUrl(filename, pageNo = null) {
    let url = `${this.apiDomain}/image/${filename}`;
    if (pageNo !== null) {
      url += `?page_no=${pageNo}`;
    }
    return url;
  }

  /**
   * Format file size for display
   * @param {number} bytes - File size in bytes
   * @returns {string} - Formatted file size
   */
  formatFileSize(bytes) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + " " + sizes[i];
  }

  /**
   * Delete a document and all its associated files
   * @param {string} filename - The filename to delete (with extension)
   * @returns {Promise} - Delete response
   */
  async deleteDocument(filename) {
    try {
      const response = await http.delete(
        `${this.apiDomain}/documents/${filename}`
      );
      return response.data;
    } catch (error) {
      console.error("Error deleting document:", error);
      throw error;
    }
  }

  /**
   * Format date for display
   * @param {string} isoString - ISO date string
   * @returns {string} - Formatted date
   */
  formatDate(isoString) {
    const date = new Date(isoString);
    return date.toLocaleString();
  }
}

const documentService = new DocumentService();
export default documentService;

