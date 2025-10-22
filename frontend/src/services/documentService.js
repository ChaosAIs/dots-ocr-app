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
   * Start document conversion (non-blocking)
   * @param {string} filename - The filename to convert
   * @param {string} promptMode - The prompt mode to use
   * @returns {Promise} - Response with conversion_id
   */
  async convertDocument(filename, promptMode = "prompt_layout_all_en") {
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

