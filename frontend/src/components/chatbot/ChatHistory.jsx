import { useState, useEffect, useRef, useCallback, forwardRef, useImperativeHandle } from "react";
import { Button } from "primereact/button";
import { ProgressSpinner } from "primereact/progressspinner";
import { Toast } from "primereact/toast";
import chatService from "../../services/chatService";
import { messageService } from "../../core/message/messageService";
import "./ChatHistory.scss";

export const ChatHistory = forwardRef(({ currentSessionId, onSessionSelect }, ref) => {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(false);
  const toast = useRef(null);

  const loadSessions = useCallback(async () => {
    console.log("[ChatHistory] loadSessions called");
    try {
      setLoading(true);
      const data = await chatService.getSessions();
      console.log("[ChatHistory] Received sessions data:", data);
      console.log("[ChatHistory] Sessions count:", data.length);
      if (data.length > 0) {
        console.log("[ChatHistory] First session:", {
          id: data[0].id,
          name: data[0].session_name,
          message_count: data[0].message_count
        });
      }
      setSessions(data);
      console.log("[ChatHistory] State updated with new sessions");
    } catch (error) {
      console.error("[ChatHistory] Error loading sessions:", error);
      toast.current?.show({
        severity: "error",
        summary: "Error",
        detail: "Failed to load chat history. Please make sure you are logged in.",
        life: 3000,
      });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  // Update a specific session's metadata in local state (without re-fetching all)
  const updateSessionMetadata = useCallback((sessionId, metadata) => {
    console.log("[ChatHistory] updateSessionMetadata called:", { sessionId, metadata });
    setSessions(prevSessions =>
      prevSessions.map(session =>
        session.id === sessionId
          ? { ...session, session_metadata: { ...session.session_metadata, ...metadata } }
          : session
      )
    );
  }, []);

  // Update a specific session in local state (for title updates, etc.)
  const updateSession = useCallback((sessionId, updates) => {
    console.log("[ChatHistory] updateSession called:", { sessionId, updates });
    setSessions(prevSessions =>
      prevSessions.map(session =>
        session.id === sessionId
          ? { ...session, ...updates }
          : session
      )
    );
  }, []);

  // Expose methods to parent component
  useImperativeHandle(ref, () => ({
    loadSessions,
    updateSessionMetadata,
    updateSession
  }));

  const handleDeleteSession = (sessionId, sessionName) => {
    messageService.confirmDeletionDialog(
      `Are you sure you want to delete "${sessionName}"?`,
      async (confirmed) => {
        if (confirmed) {
          try {
            await chatService.deleteSession(sessionId);
            setSessions(sessions.filter((s) => s.id !== sessionId));
            toast.current?.show({
              severity: "success",
              summary: "Deleted",
              detail: "Chat session deleted",
              life: 3000,
            });
          } catch (error) {
            console.error("Error deleting session:", error);
            toast.current?.show({
              severity: "error",
              summary: "Error",
              detail: "Failed to delete session",
              life: 3000,
            });
          }
        }
      }
    );
  };

  const handleCleanupEmpty = async () => {
    try {
      const result = await chatService.cleanupEmptySessions();
      toast.current?.show({
        severity: "success",
        summary: "Cleanup Complete",
        detail: result.message,
        life: 3000,
      });
      loadSessions();
    } catch (error) {
      toast.current?.show({
        severity: "error",
        summary: "Error",
        detail: "Failed to cleanup empty sessions",
        life: 5000,
      });
    }
  };

  const handleRegenerateAllTitles = async () => {
    try {
      toast.current?.show({
        severity: "info",
        summary: "Processing",
        detail: "Regenerating titles with LLM...",
        life: 2000,
      });

      const result = await chatService.regenerateAllTitles();

      toast.current?.show({
        severity: "success",
        summary: "Titles Regenerated",
        detail: result.message,
        life: 3000,
      });

      loadSessions();
    } catch (error) {
      toast.current?.show({
        severity: "error",
        summary: "Error",
        detail: "Failed to regenerate titles",
        life: 5000,
      });
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return "";
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString();
  };

  const getSessionDisplayName = (session) => {
    // If session has a custom name (not the default datetime format), use it
    if (session.session_name && !session.session_name.startsWith("Chat ")) {
      return session.session_name;
    }
    // Otherwise, show "New Chat" for sessions without a title
    return "New Chat";
  };

  return (
    <div className="chat-history-panel">
      <Toast ref={toast} />

      <div className="chat-history-header">
        <h3>Chat History</h3>
        <div className="header-actions">
          <Button
            icon="pi pi-sparkles"
            text
            rounded
            severity="warning"
            onClick={handleRegenerateAllTitles}
            tooltip="Regenerate all titles with LLM"
            tooltipOptions={{ position: "bottom" }}
            style={{ width: '1.75rem', height: '1.75rem', padding: 0 }}
          />
          <Button
            icon="pi pi-trash"
            text
            rounded
            severity="danger"
            onClick={handleCleanupEmpty}
            tooltip="Clean up empty sessions"
            tooltipOptions={{ position: "bottom" }}
            style={{ width: '1.75rem', height: '1.75rem', padding: 0 }}
          />
          <Button
            icon="pi pi-refresh"
            text
            rounded
            onClick={loadSessions}
            tooltip="Refresh"
            tooltipOptions={{ position: "bottom" }}
            style={{ width: '1.75rem', height: '1.75rem', padding: 0 }}
          />
        </div>
      </div>

      {loading ? (
        <div className="loading-container">
          <ProgressSpinner />
        </div>
      ) : (
        <div className="sessions-list">
          {sessions.length === 0 ? (
            <div className="empty-state">
              <i className="pi pi-comments" />
              <p>No chat history yet</p>
            </div>
          ) : (
            sessions.map((session) => (
              <div
                key={session.id}
                className={`session-item ${session.id === currentSessionId ? "active" : ""}`}
                onClick={() => {
                  onSessionSelect(session.id);
                }}
              >
                <i className="pi pi-comments session-icon" />
                <div className="session-info">
                  <div className="session-name">{getSessionDisplayName(session)}</div>
                  <div className="session-meta">
                    <span className="message-count">
                      <i className="pi pi-comment" />
                      {session.message_count}
                    </span>
                    <span className="session-date">{formatDate(session.updated_at)}</span>
                  </div>
                </div>
                <Button
                  icon="pi pi-trash"
                  text
                  rounded
                  severity="danger"
                  className="delete-btn"
                  style={{ width: '1.75rem', height: '1.75rem', padding: 0 }}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteSession(session.id, session.session_name);
                  }}
                  tooltip="Delete"
                  tooltipOptions={{ position: "top" }}
                />
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
});

