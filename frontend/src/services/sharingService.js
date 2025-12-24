import http from "../core/http/httpClient";
import APP_CONFIG from "../core/config/appConfig";

/**
 * Sharing Service - handles document sharing and permission API calls
 */
class SharingService {
  constructor() {
    this.apiDomain = APP_CONFIG.apiDomain || "http://localhost:8080";
    this.baseUrl = `${this.apiDomain}/api/sharing`;
  }

  /**
   * Share a document with one or more users by ID
   * @param {string} documentId - Document ID to share
   * @param {Array<string>} userIds - User IDs to share with
   * @param {Array<string>} permissions - Permissions to grant (read, update, delete, share, full)
   * @param {string} message - Optional message to recipients
   * @param {string} expiresAt - Optional expiration date (ISO string)
   * @returns {Promise} - Share response
   */
  async shareDocument(documentId, userIds, permissions = ["read"], message = null, expiresAt = null) {
    try {
      const response = await http.post(`${this.baseUrl}/share`, {
        document_id: documentId,
        user_ids: userIds,
        permissions,
        message,
        expires_at: expiresAt
      });
      return response.data;
    } catch (error) {
      console.error("Error sharing document:", error);
      throw error;
    }
  }

  /**
   * Share a document with users by username or email
   * @param {string} documentId - Document ID to share
   * @param {Array<string>} usernames - Usernames or emails to share with
   * @param {Array<string>} permissions - Permissions to grant
   * @param {string} message - Optional message
   * @param {string} expiresAt - Optional expiration
   * @returns {Promise} - Share response
   */
  async shareDocumentByUsername(documentId, usernames, permissions = ["read"], message = null, expiresAt = null) {
    try {
      const response = await http.post(`${this.baseUrl}/share-by-username`, {
        document_id: documentId,
        usernames,
        permissions,
        message,
        expires_at: expiresAt
      });
      return response.data;
    } catch (error) {
      console.error("Error sharing document by username:", error);
      throw error;
    }
  }

  /**
   * Get documents shared with the current user
   * @param {number} limit - Max documents to return
   * @param {number} offset - Documents to skip
   * @returns {Promise} - List of shared documents
   */
  async getSharedWithMe(limit = 100, offset = 0) {
    try {
      const response = await http.get(`${this.baseUrl}/shared-with-me`, {
        params: { limit, offset }
      });
      return response.data;
    } catch (error) {
      console.error("Error fetching shared documents:", error);
      throw error;
    }
  }

  /**
   * Get new (unviewed) shared documents
   * @returns {Promise} - New shares with count
   */
  async getNewShares() {
    try {
      const response = await http.get(`${this.baseUrl}/new-shares`);
      return response.data;
    } catch (error) {
      console.error("Error fetching new shares:", error);
      throw error;
    }
  }

  /**
   * Mark a shared document as viewed
   * @param {string} documentId - Document ID
   * @returns {Promise} - Response
   */
  async markAsViewed(documentId) {
    try {
      const response = await http.post(`${this.baseUrl}/mark-viewed/${documentId}`);
      return response.data;
    } catch (error) {
      console.error("Error marking document as viewed:", error);
      throw error;
    }
  }

  /**
   * Mark all shared documents as viewed
   * @returns {Promise} - Response with count
   */
  async markAllAsViewed() {
    try {
      const response = await http.post(`${this.baseUrl}/mark-all-viewed`);
      return response.data;
    } catch (error) {
      console.error("Error marking all as viewed:", error);
      throw error;
    }
  }

  /**
   * Get all users with access to a document
   * @param {string} documentId - Document ID
   * @returns {Promise} - Document shares info
   */
  async getDocumentShares(documentId) {
    try {
      const response = await http.get(`${this.baseUrl}/document/${documentId}/shares`);
      return response.data;
    } catch (error) {
      console.error("Error fetching document shares:", error);
      throw error;
    }
  }

  /**
   * Update a user's permissions on a document
   * @param {string} documentId - Document ID
   * @param {string} userId - User ID
   * @param {Array<string>} permissions - New permissions
   * @returns {Promise} - Updated permission record
   */
  async updatePermissions(documentId, userId, permissions) {
    try {
      const response = await http.put(`${this.baseUrl}/update-permissions`, {
        document_id: documentId,
        user_id: userId,
        permissions
      });
      return response.data;
    } catch (error) {
      console.error("Error updating permissions:", error);
      throw error;
    }
  }

  /**
   * Revoke a user's access to a document
   * @param {string} documentId - Document ID
   * @param {string} userId - User ID to revoke
   * @returns {Promise} - Response
   */
  async revokeAccess(documentId, userId) {
    try {
      const response = await http.post(`${this.baseUrl}/revoke`, {
        document_id: documentId,
        user_id: userId
      });
      return response.data;
    } catch (error) {
      console.error("Error revoking access:", error);
      throw error;
    }
  }

  /**
   * Transfer document ownership to another user
   * @param {string} documentId - Document ID
   * @param {string} newOwnerId - New owner's user ID
   * @returns {Promise} - Response
   */
  async transferOwnership(documentId, newOwnerId) {
    try {
      const response = await http.post(`${this.baseUrl}/transfer-ownership`, {
        document_id: documentId,
        new_owner_id: newOwnerId
      });
      return response.data;
    } catch (error) {
      console.error("Error transferring ownership:", error);
      throw error;
    }
  }

  /**
   * Check current user's access to a document
   * @param {string} documentId - Document ID
   * @returns {Promise} - Access info
   */
  async checkAccess(documentId) {
    try {
      const response = await http.get(`${this.baseUrl}/check-access/${documentId}`);
      return response.data;
    } catch (error) {
      console.error("Error checking access:", error);
      throw error;
    }
  }

  /**
   * Get permission options for UI
   * @returns {Array} - Permission options
   */
  getPermissionOptions() {
    return [
      { label: "View", value: "read", description: "Can view document content" },
      { label: "Edit", value: "update", description: "Can modify document" },
      { label: "Delete", value: "delete", description: "Can delete document" },
      { label: "Share", value: "share", description: "Can share with others" },
      { label: "Full Access", value: "full", description: "All permissions" }
    ];
  }
}

const sharingService = new SharingService();
export default sharingService;
