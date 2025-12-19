import http from "../core/http/httpClient";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8080";

class ChatService {
  /**
   * Create a new chat session
   * @param {string} sessionName - Optional name for the session
   * @returns {Promise<Object>} Session object with id, session_name, etc.
   */
  async createSession(sessionName = null) {
    try {
      const response = await http.post(
        `${API_BASE_URL}/api/chat/sessions`,
        { session_name: sessionName }
      );
      return response.data;
    } catch (error) {
      console.error("Error creating chat session:", error);
      throw error;
    }
  }

  /**
   * Get all chat sessions for the current authenticated user
   * @param {number} limit - Maximum number of sessions to retrieve
   * @returns {Promise<Array>} Array of session objects
   */
  async getSessions(limit = 50) {
    try {
      console.log(`[ChatService] getSessions called with limit=${limit}`);
      const response = await http.get(
        `${API_BASE_URL}/api/chat/sessions`,
        {
          params: { limit }
        }
      );
      console.log(`[ChatService] getSessions response:`, response.data);
      console.log(`[ChatService] Number of sessions:`, response.data.length);
      return response.data;
    } catch (error) {
      console.error("[ChatService] Error fetching chat sessions:", error);
      throw error;
    }
  }

  /**
   * Get a specific chat session by ID
   * @param {string} sessionId - Session UUID
   * @returns {Promise<Object>} Session object
   */
  async getSession(sessionId) {
    try {
      const response = await http.get(
        `${API_BASE_URL}/api/chat/sessions/${sessionId}`
      );
      return response.data;
    } catch (error) {
      console.error("Error fetching chat session:", error);
      throw error;
    }
  }

  /**
   * Get all messages for a chat session
   * @param {string} sessionId - Session UUID
   * @param {number} limit - Optional limit on number of messages
   * @returns {Promise<Array>} Array of message objects
   */
  async getSessionMessages(sessionId, limit = null) {
    try {
      const params = limit ? { limit } : {};
      const response = await http.get(
        `${API_BASE_URL}/api/chat/sessions/${sessionId}/messages`,
        { params }
      );
      return response.data;
    } catch (error) {
      console.error("Error fetching session messages:", error);
      throw error;
    }
  }

  /**
   * Update a chat session
   * @param {string} sessionId - Session UUID
   * @param {Object} updates - Object with session_name and/or is_active
   * @returns {Promise<Object>} Updated session object
   */
  async updateSession(sessionId, updates) {
    try {
      const response = await http.patch(
        `${API_BASE_URL}/api/chat/sessions/${sessionId}`,
        updates
      );
      return response.data;
    } catch (error) {
      console.error("Error updating chat session:", error);
      throw error;
    }
  }

  /**
   * Delete a chat session
   * @param {string} sessionId - Session UUID
   * @returns {Promise<Object>} Success message
   */
  async deleteSession(sessionId) {
    try {
      const response = await http.delete(
        `${API_BASE_URL}/api/chat/sessions/${sessionId}`
      );
      return response.data;
    } catch (error) {
      console.error("Error deleting chat session:", error);
      throw error;
    }
  }

  /**
   * Delete all messages after a specific message (including the message itself)
   * Used for retry functionality
   * @param {string} sessionId - Session UUID
   * @param {string} messageId - Message UUID to delete from (inclusive)
   * @returns {Promise<Object>} Object with deleted_count and message
   */
  async deleteMessagesAfter(sessionId, messageId) {
    try {
      console.log(`[ChatService] deleteMessagesAfter called: sessionId=${sessionId}, messageId=${messageId}`);
      const response = await http.delete(
        `${API_BASE_URL}/api/chat/sessions/${sessionId}/messages/after/${messageId}`
      );
      console.log(`[ChatService] deleteMessagesAfter response:`, response.data);
      return response.data;
    } catch (error) {
      console.error("[ChatService] Error deleting messages:", error);
      throw error;
    }
  }

  /**
   * Clean up empty chat sessions (sessions with 0 messages)
   * @returns {Promise<Object>} Object with deleted_count and message
   */
  async cleanupEmptySessions() {
    try {
      const response = await http.post(
        `${API_BASE_URL}/api/chat/sessions/cleanup-empty`
      );
      return response.data;
    } catch (error) {
      console.error("Error cleaning up empty sessions:", error);
      throw error;
    }
  }

  /**
   * Regenerate title for a specific session
   * @param {string} sessionId - Session UUID
   * @returns {Promise<Object>} Object with new_title and message
   */
  async regenerateSessionTitle(sessionId) {
    try {
      const response = await http.post(
        `${API_BASE_URL}/api/chat/sessions/${sessionId}/regenerate-title`
      );
      return response.data;
    } catch (error) {
      console.error("Error regenerating session title:", error);
      throw error;
    }
  }

  /**
   * Regenerate titles for all sessions with generic titles
   * @returns {Promise<Object>} Object with regenerated_count and message
   */
  async regenerateAllTitles() {
    try {
      const response = await http.post(
        `${API_BASE_URL}/api/chat/sessions/regenerate-all-titles`
      );
      return response.data;
    } catch (error) {
      console.error("Error regenerating all titles:", error);
      throw error;
    }
  }

  /**
   * Format date for display
   * @param {string} dateString - ISO date string
   * @returns {string} Formatted date string
   */
  formatDate(dateString) {
    if (!dateString) return "";
    const date = new Date(dateString);
    return date.toLocaleString();
  }
}

const chatServiceInstance = new ChatService();
export default chatServiceInstance;

