import http from "../core/http/httpClient";
import APP_CONFIG from "../core/config/appConfig";

/**
 * Workspace Service - handles all workspace-related API calls
 */
class WorkspaceService {
  constructor() {
    this.apiDomain = APP_CONFIG.apiDomain || "http://localhost:8080";
    this.baseUrl = `${this.apiDomain}/api/workspaces`;
  }

  /**
   * Get all workspaces for the current user
   * @param {boolean} includeSystem - Include system workspaces like "Shared With Me"
   * @returns {Promise} - List of workspaces
   */
  async getWorkspaces(includeSystem = true) {
    try {
      const response = await http.get(this.baseUrl, {
        params: { include_system: includeSystem }
      });
      return response.data;
    } catch (error) {
      console.error("Error fetching workspaces:", error);
      throw error;
    }
  }

  /**
   * Create a new workspace
   * @param {Object} workspace - Workspace data
   * @param {string} workspace.name - Display name
   * @param {string} workspace.description - Optional description
   * @param {string} workspace.color - Hex color code
   * @param {string} workspace.icon - Icon identifier
   * @returns {Promise} - Created workspace
   */
  async createWorkspace(workspace) {
    try {
      const response = await http.post(this.baseUrl, workspace);
      return response.data;
    } catch (error) {
      console.error("Error creating workspace:", error);
      throw error;
    }
  }

  /**
   * Get the user's default workspace (creates one if not exists)
   * @returns {Promise} - Default workspace
   */
  async getDefaultWorkspace() {
    try {
      const response = await http.get(`${this.baseUrl}/default`);
      return response.data;
    } catch (error) {
      console.error("Error fetching default workspace:", error);
      throw error;
    }
  }

  /**
   * Get a workspace with its documents
   * @param {string} workspaceId - Workspace ID
   * @param {number} limit - Max documents to return
   * @param {number} offset - Documents to skip
   * @returns {Promise} - Workspace with documents
   */
  async getWorkspace(workspaceId, limit = 100, offset = 0) {
    try {
      const response = await http.get(`${this.baseUrl}/${workspaceId}`, {
        params: { limit, offset }
      });
      return response.data;
    } catch (error) {
      console.error("Error fetching workspace:", error);
      throw error;
    }
  }

  /**
   * Update workspace metadata
   * @param {string} workspaceId - Workspace ID
   * @param {Object} updates - Fields to update
   * @returns {Promise} - Updated workspace
   */
  async updateWorkspace(workspaceId, updates) {
    try {
      const response = await http.put(`${this.baseUrl}/${workspaceId}`, updates);
      return response.data;
    } catch (error) {
      console.error("Error updating workspace:", error);
      throw error;
    }
  }

  /**
   * Rename a workspace (updates display_name only, folder unchanged)
   * @param {string} workspaceId - Workspace ID
   * @param {string} newDisplayName - New display name (any language)
   * @returns {Promise} - Updated workspace
   */
  async renameWorkspace(workspaceId, newDisplayName) {
    try {
      const response = await http.post(`${this.baseUrl}/${workspaceId}/rename`, {
        display_name: newDisplayName
      });
      return response.data;
    } catch (error) {
      console.error("Error renaming workspace:", error);
      throw error;
    }
  }

  /**
   * Delete a workspace
   * @param {string} workspaceId - Workspace ID
   * @param {boolean} deleteDocuments - If true, delete all documents. If false, move to default.
   * @returns {Promise} - Delete response
   */
  async deleteWorkspace(workspaceId, deleteDocuments = false) {
    try {
      const response = await http.delete(`${this.baseUrl}/${workspaceId}`, {
        params: { delete_documents: deleteDocuments }
      });
      return response.data;
    } catch (error) {
      console.error("Error deleting workspace:", error);
      throw error;
    }
  }

  /**
   * Set a workspace as the user's default
   * @param {string} workspaceId - Workspace ID
   * @returns {Promise} - Response
   */
  async setDefaultWorkspace(workspaceId) {
    try {
      const response = await http.post(`${this.baseUrl}/${workspaceId}/set-default`);
      return response.data;
    } catch (error) {
      console.error("Error setting default workspace:", error);
      throw error;
    }
  }

  /**
   * Update display order for workspaces
   * @param {Array} orders - Array of {workspace_id, order}
   * @returns {Promise} - Response
   */
  async updateDisplayOrder(orders) {
    try {
      const response = await http.post(`${this.baseUrl}/reorder`, {
        orders
      });
      return response.data;
    } catch (error) {
      console.error("Error updating workspace order:", error);
      throw error;
    }
  }

  /**
   * Move a document to another workspace
   * @param {string} documentId - Document ID
   * @param {string} targetWorkspaceId - Target workspace ID
   * @returns {Promise} - Response
   */
  async moveDocument(documentId, targetWorkspaceId) {
    try {
      const response = await http.post(`${this.baseUrl}/move-document`, {
        document_id: documentId,
        target_workspace_id: targetWorkspaceId
      });
      return response.data;
    } catch (error) {
      console.error("Error moving document:", error);
      throw error;
    }
  }

  /**
   * Sync workspace folders with database
   * @returns {Promise} - Sync results
   */
  async syncFolders() {
    try {
      const response = await http.post(`${this.baseUrl}/sync`);
      return response.data;
    } catch (error) {
      console.error("Error syncing workspace folders:", error);
      throw error;
    }
  }

  /**
   * Get workspace color options for UI
   * @returns {Array} - Color options
   */
  getColorOptions() {
    return [
      { name: "Indigo", value: "#6366f1" },
      { name: "Blue", value: "#3b82f6" },
      { name: "Green", value: "#22c55e" },
      { name: "Yellow", value: "#eab308" },
      { name: "Orange", value: "#f97316" },
      { name: "Red", value: "#ef4444" },
      { name: "Purple", value: "#a855f7" },
      { name: "Pink", value: "#ec4899" },
      { name: "Teal", value: "#14b8a6" },
      { name: "Gray", value: "#6b7280" }
    ];
  }

  /**
   * Get workspace icon options for UI
   * @returns {Array} - Icon options
   */
  getIconOptions() {
    return [
      { name: "Folder", value: "folder", icon: "pi pi-folder" },
      { name: "Briefcase", value: "briefcase", icon: "pi pi-briefcase" },
      { name: "Book", value: "book", icon: "pi pi-book" },
      { name: "Star", value: "star", icon: "pi pi-star" },
      { name: "Heart", value: "heart", icon: "pi pi-heart" },
      { name: "Flag", value: "flag", icon: "pi pi-flag" },
      { name: "Box", value: "box", icon: "pi pi-box" },
      { name: "Database", value: "database", icon: "pi pi-database" },
      { name: "Image", value: "image", icon: "pi pi-image" },
      { name: "File", value: "file", icon: "pi pi-file" }
    ];
  }

  /**
   * Save markdown content to a workspace
   * @param {string} content - The markdown content to save
   * @param {string} filename - The filename (without extension)
   * @param {string} workspaceId - The target workspace ID
   * @returns {Promise} - Save response with document_id and filename
   */
  async saveMarkdownToWorkspace(content, filename, workspaceId) {
    try {
      const response = await http.post(`${this.baseUrl}/save-markdown`, {
        content,
        filename,
        workspace_id: workspaceId
      });
      return response.data;
    } catch (error) {
      console.error("Error saving markdown to workspace:", error);
      throw error;
    }
  }
}

const workspaceService = new WorkspaceService();
export default workspaceService;
