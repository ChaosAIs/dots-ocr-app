import { useState, useRef, useEffect, useCallback } from "react";
import { InputTextarea } from "primereact/inputtextarea";
import { InputText } from "primereact/inputtext";
import { Button } from "primereact/button";
import { Card } from "primereact/card";
import { ProgressSpinner } from "primereact/progressspinner";
import { Toast } from "primereact/toast";
import { Dialog } from "primereact/dialog";
import { RadioButton } from "primereact/radiobutton";
import { Dropdown } from "primereact/dropdown";
import { Checkbox } from "primereact/checkbox";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import chatService from "../../services/chatService";
import authService from "../../services/authService";
import workspaceService from "../../services/workspaceService";
import { ChatHistory } from "./ChatHistory";
import { WorkspaceBrowser } from "./WorkspaceBrowser";
import { useTranslation } from "react-i18next";
import { useAuth } from "../../core/auth/components/authProvider";
import {
  exportAsMarkdown,
  exportAsPdf,
  exportAsWord,
  exportAsExcel,
  generateFilename,
} from "../../utils/exportUtils";
import { useConnectionStatus } from "../../contexts/ConnectionStatusContext";
import "./AgenticChatBot.scss";

// Get WebSocket URL from environment or use default
const WS_BASE_URL = process.env.REACT_APP_WS_URL || "ws://localhost:8080";

export const AgenticChatBot = () => {
  const { t } = useTranslation();
  const { user } = useAuth();
  const { updateChatStatus } = useConnectionStatus();
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [isInitializing, setIsInitializing] = useState(true);
  const [sessionContext, setSessionContext] = useState(null);
  const [editingMessageIndex, setEditingMessageIndex] = useState(null);
  const [editingContent, setEditingContent] = useState("");
  const [isSavingCorrection, setIsSavingCorrection] = useState(false);

  // Markdown editor dialog state
  const [markdownEditorVisible, setMarkdownEditorVisible] = useState(false);
  const [markdownEditorContent, setMarkdownEditorContent] = useState("");
  const [markdownEditorMessageIndex, setMarkdownEditorMessageIndex] = useState(null);
  const [markdownEditorPreview, setMarkdownEditorPreview] = useState(false);

  // Export/Download dialog state
  const [exportDialogVisible, setExportDialogVisible] = useState(false);
  const [exportContent, setExportContent] = useState("");
  const [exportOption, setExportOption] = useState("markdown");
  const [exportFilename, setExportFilename] = useState("");
  const [workspaces, setWorkspaces] = useState([]);
  const [selectedWorkspaceForExport, setSelectedWorkspaceForExport] = useState(null);
  const [isExporting, setIsExporting] = useState(false);
  const [loadingWorkspaces, setLoadingWorkspaces] = useState(false);

  // Create new workspace state (within export dialog)
  const [showCreateWorkspace, setShowCreateWorkspace] = useState(false);
  const [newWorkspaceName, setNewWorkspaceName] = useState("");
  const [isCreatingWorkspace, setIsCreatingWorkspace] = useState(false);

  // Progress step tracking for better UX
  const [progressStep, setProgressStep] = useState(null);

  // Workspace browser state
  const [selectedWorkspaceIds, setSelectedWorkspaceIds] = useState([]);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState([]);
  const [workspaceBrowserCollapsed, setWorkspaceBrowserCollapsed] = useState(false);
  const [mobileWorkspaceDrawerOpen, setMobileWorkspaceDrawerOpen] = useState(false);

  // Graph RAG knowledge reasoning toggle
  const [graphRagEnabled, setGraphRagEnabled] = useState(true); // Default to true, will be updated from config

  const wsRef = useRef(null);
  const workspaceBrowserRef = useRef(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const toast = useRef(null);
  const streamingContentRef = useRef("");
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 10; // Increased for better resilience
  const chatHistoryRef = useRef(null);
  const messageCountRef = useRef(0);
  const pingIntervalRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const isManualCloseRef = useRef(false); // Track if close was intentional

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingContent, scrollToBottom]);

  // Initialize - no session created until first message
  useEffect(() => {
    setIsInitializing(false);
  }, []);

  // Load chat preferences (selected workspaces and documents) on mount
  useEffect(() => {
    const loadChatPreferences = async () => {
      try {
        const result = await authService.getChatPreferences();
        if (result.success && result.chat) {
          if (result.chat.selectedWorkspaceIds) {
            console.log("[AgenticChatBot] Loaded workspace preferences:", result.chat.selectedWorkspaceIds);
            setSelectedWorkspaceIds(result.chat.selectedWorkspaceIds);
          }
          if (result.chat.selectedDocumentIds) {
            console.log("[AgenticChatBot] Loaded document preferences:", result.chat.selectedDocumentIds);
            setSelectedDocumentIds(result.chat.selectedDocumentIds);
          }
        }
      } catch (error) {
        console.error("[AgenticChatBot] Error loading chat preferences:", error);
      }
    };

    if (user) {
      loadChatPreferences();
    }
  }, [user]);

  // Load chat config (graph RAG enabled setting) on mount
  useEffect(() => {
    const loadChatConfig = async () => {
      try {
        const config = await chatService.getChatConfig();
        if (config && typeof config.graph_rag_query_enabled === 'boolean') {
          console.log("[AgenticChatBot] Loaded chat config, graph_rag_query_enabled:", config.graph_rag_query_enabled);
          setGraphRagEnabled(config.graph_rag_query_enabled);
        }
      } catch (error) {
        console.error("[AgenticChatBot] Error loading chat config:", error);
        // Keep default value on error
      }
    };

    loadChatConfig();
  }, []);

  // Track if selection was initialized (to avoid saving on initial load)
  const selectionInitializedRef = useRef(false);
  const lastSavedSelectionRef = useRef({ workspaceIds: [], documentIds: [] });

  // Save workspace/document selection to current session metadata when it changes
  // This ensures the selection is persisted and restored when reopening the session
  useEffect(() => {
    const saveSelectionToSession = async () => {
      // Only save if there's an active session
      if (!sessionId) return;

      // Skip saving on initial load - wait until selection is initialized
      if (!selectionInitializedRef.current) {
        // Mark as initialized after first render with a session
        selectionInitializedRef.current = true;
        lastSavedSelectionRef.current = {
          workspaceIds: [...selectedWorkspaceIds],
          documentIds: [...selectedDocumentIds]
        };
        return;
      }

      // Check if selection actually changed
      const workspacesSame =
        selectedWorkspaceIds.length === lastSavedSelectionRef.current.workspaceIds.length &&
        selectedWorkspaceIds.every(id => lastSavedSelectionRef.current.workspaceIds.includes(id));
      const documentsSame =
        selectedDocumentIds.length === lastSavedSelectionRef.current.documentIds.length &&
        selectedDocumentIds.every(id => lastSavedSelectionRef.current.documentIds.includes(id));

      if (workspacesSame && documentsSame) {
        return; // No change, skip save
      }

      try {
        await chatService.updateSessionMetadata(
          sessionId,
          selectedWorkspaceIds,
          selectedDocumentIds
        );
        lastSavedSelectionRef.current = {
          workspaceIds: [...selectedWorkspaceIds],
          documentIds: [...selectedDocumentIds]
        };
        console.log("[AgenticChatBot] Saved selection to session metadata:", {
          sessionId,
          workspaceIds: selectedWorkspaceIds,
          documentIds: selectedDocumentIds
        });
      } catch (error) {
        // Silently fail - this is not critical
        console.error("[AgenticChatBot] Error saving selection to session:", error);
      }
    };

    saveSelectionToSession();
  }, [sessionId, selectedWorkspaceIds, selectedDocumentIds]);

  // Calculate exponential backoff delay for reconnection
  const getReconnectDelay = useCallback((attempt) => {
    // Exponential backoff: 1s, 2s, 4s, 8s, 16s, max 30s
    const baseDelay = 1000;
    const maxDelay = 30000;
    const delay = Math.min(baseDelay * Math.pow(2, attempt), maxDelay);
    // Add jitter (±20%) to prevent thundering herd
    const jitter = delay * 0.2 * (Math.random() - 0.5);
    return Math.round(delay + jitter);
  }, []);

  // Start heartbeat to keep connection alive
  const startHeartbeat = useCallback((ws) => {
    // Clear any existing heartbeat
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
    }

    // Send ping every 25 seconds (most servers timeout at 30-60s)
    pingIntervalRef.current = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(JSON.stringify({ type: "ping" }));
          console.log("[WebSocket] Ping sent");
        } catch (e) {
          console.error("[WebSocket] Failed to send ping:", e);
        }
      }
    }, 25000);
  }, []);

  // Stop heartbeat
  const stopHeartbeat = useCallback(() => {
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }
  }, []);

  const connectWebSocket = useCallback(() => {
    // Don't connect if no session ID or already connected
    if (!sessionId || wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    // Don't connect if closing/connecting - wait for close to complete
    if (wsRef.current?.readyState === WebSocket.CONNECTING) {
      console.log("[WebSocket] Already connecting, skipping...");
      return;
    }

    // Don't connect if closing - wait for close to complete
    if (wsRef.current?.readyState === WebSocket.CLOSING) {
      console.log("[WebSocket] Closing in progress, will reconnect after close...");
      return;
    }

    // Check if we've exceeded max reconnection attempts
    if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
      console.error("Max WebSocket reconnection attempts reached");
      toast.current?.show({
        severity: "error",
        summary: "Connection Failed",
        detail: "Unable to connect to chat server. Please refresh the page.",
        life: 5000,
      });
      return;
    }

    // Clear any pending reconnect timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    console.log(`[WebSocket] Connecting with session ID: ${sessionId} (attempt ${reconnectAttemptsRef.current + 1}/${maxReconnectAttempts})`);
    const ws = new WebSocket(`${WS_BASE_URL}/api/chat/ws/chat/${sessionId}`);

    ws.onopen = () => {
      setIsConnected(true);
      updateChatStatus(true, true, false); // connected, session active, not reconnecting
      reconnectAttemptsRef.current = 0; // Reset counter on successful connection
      isManualCloseRef.current = false;
      console.log("[WebSocket] Connected successfully");

      // Start heartbeat to keep connection alive
      startHeartbeat(ws);
    };

    ws.onclose = (event) => {
      setIsConnected(false);
      updateChatStatus(false, !!sessionId, false); // disconnected, session may still be active
      stopHeartbeat();
      console.log(`[WebSocket] Disconnected - code: ${event.code}, reason: ${event.reason || 'none'}, wasManualClose: ${isManualCloseRef.current}`);

      // Clear the WebSocket reference since it's now closed
      if (wsRef.current === ws) {
        wsRef.current = null;
      }

      // Don't reconnect if manually closed or auth errors
      if (isManualCloseRef.current) {
        console.log("[WebSocket] Manually closed - not reconnecting");
        return;
      }

      // Don't reconnect on authentication/authorization errors (403, 401)
      // or if the session was explicitly closed (1000)
      if (event.code === 1008 || event.code === 1000) {
        console.log("[WebSocket] Closed normally or due to policy violation - not reconnecting");
        return;
      }

      // Schedule reconnection with exponential backoff
      if (reconnectAttemptsRef.current < maxReconnectAttempts) {
        const delay = getReconnectDelay(reconnectAttemptsRef.current);
        reconnectAttemptsRef.current += 1;
        console.log(`[WebSocket] Scheduling reconnection in ${delay}ms (attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts})`);

        updateChatStatus(false, true, true); // disconnected, session active, reconnecting
        reconnectTimeoutRef.current = setTimeout(connectWebSocket, delay);
      } else {
        toast.current?.show({
          severity: "error",
          summary: "Connection Lost",
          detail: "Unable to reconnect. Please refresh the page.",
          life: 10000,
        });
      }
    };

    ws.onerror = (error) => {
      console.error("[WebSocket] Error:", error);
      // Don't show toast on every error - onclose will handle reconnection
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        // Handle pong response (heartbeat acknowledgment)
        if (data.type === "pong") {
          console.log("[WebSocket] Pong received");
          return;
        }

        if (data.type === "progress") {
          // Display progress message (e.g., "Analyzing query...", "Routing to documents...")
          const progressMsg = data.message || "Processing...";
          console.log("[WebSocket Progress] Received progress message:", progressMsg);

          // Update progress step for UI display (without percentage for cleaner UX)
          setProgressStep(progressMsg);
          // Don't set streaming content for progress - we'll show a separate progress indicator
        } else if (data.type === "token") {
          // Log when token streaming starts (only first token to avoid spam)
          if (!streamingContentRef.current) {
            console.log("[WebSocket] Token streaming started, clearing progress step");
          }
          // Clear progress step when actual content starts streaming
          setProgressStep(null);
          streamingContentRef.current += data.content;
          setStreamingContent(streamingContentRef.current);
        } else if (data.type === "end") {
          // Complete the message with the accumulated content
          let finalContent = streamingContentRef.current;

          // If no content received, provide a friendly fallback message
          // This ensures the AI assistant always responds
          if (!finalContent.trim()) {
            finalContent = "I couldn't generate a response for your question. This might be because:\n\n• No relevant information was found in the selected documents\n• The question might need more context\n• Please try rephrasing your question or selecting different documents.";
          }

          // Always add the assistant message
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: finalContent,
            },
          ]);

          streamingContentRef.current = "";
          setStreamingContent("");
          setProgressStep(null);
          setIsLoading(false);

          // Increment message count
          messageCountRef.current += 2; // user + assistant

          // Refresh chat history after each assistant response (to show auto-generated/updated title)
          // Backend now checks and updates title on every response if needed
          console.log("[AgenticChatBot] Assistant response completed, scheduling sidebar refresh");
          if (chatHistoryRef.current?.loadSessions) {
            setTimeout(() => {
              console.log("[AgenticChatBot] Calling chatHistoryRef.loadSessions() after assistant response");
              chatHistoryRef.current.loadSessions();
            }, 500); // Small delay to ensure backend has updated the title
          } else {
            console.warn("[AgenticChatBot] chatHistoryRef.current.loadSessions not available");
          }

          // Refresh session context after each message exchange
          if (sessionId) {
            setTimeout(async () => {
              try {
                const session = await chatService.getSession(sessionId);
                setSessionContext(session.session_metadata || {});
              } catch (error) {
                console.error("Error refreshing session context:", error);
              }
            }, 500);
          }
        } else if (data.type === "error") {
          // Log the error but don't show toast - we'll show a friendly assistant message instead
          console.warn("[WebSocket] Received error:", data.message);

          // Create a user-friendly error response
          const errorContent = streamingContentRef.current.trim() ||
            "I'm sorry, but I encountered an issue while processing your request. Please try again, or rephrase your question.\n\nIf the problem persists, try refreshing the page.";

          // Add the error as an assistant message so the chat flow continues
          setMessages((prev) => [
            ...prev,
            {
              role: "assistant",
              content: errorContent,
              isError: true,  // Flag to style differently if needed
            },
          ]);

          // Clear streaming content
          streamingContentRef.current = "";
          setStreamingContent("");
          setProgressStep(null);
          setIsLoading(false);

          // Increment message count
          messageCountRef.current += 2; // user + assistant
        }
      } catch (e) {
        console.error("Error parsing message:", e);
        // If we're in loading state, add a fallback response to avoid blocking
        if (setIsLoading) {
          setMessages((prev) => {
            // Check if the last message is from user (waiting for response)
            if (prev.length > 0 && prev[prev.length - 1].role === "user") {
              return [
                ...prev,
                {
                  role: "assistant",
                  content: "I encountered an issue processing the response. Please try your question again.",
                  isError: true,
                },
              ];
            }
            return prev;
          });
          streamingContentRef.current = "";
          setStreamingContent("");
          setProgressStep(null);
          setIsLoading(false);
        }
      }
    };

    wsRef.current = ws;
  }, [sessionId, startHeartbeat, stopHeartbeat, getReconnectDelay, updateChatStatus]);

  // Handle visibility change - reconnect when tab becomes visible
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible' && sessionId) {
        // Tab became visible - check if we need to reconnect
        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
          console.log("[WebSocket] Tab became visible - attempting to reconnect");
          reconnectAttemptsRef.current = 0; // Reset attempts when user returns
          connectWebSocket();
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [sessionId, connectWebSocket]);

  // Connect WebSocket when session ID is available
  useEffect(() => {
    if (sessionId) {
      connectWebSocket();
    }
    return () => {
      // Cleanup on unmount
      isManualCloseRef.current = true;
      stopHeartbeat();
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWebSocket, sessionId, stopHeartbeat]);

  // Load chat history when session is created
  useEffect(() => {
    const loadHistory = async () => {
      if (!sessionId) return;

      try {
        const history = await chatService.getSessionMessages(sessionId);
        if (history && history.length > 0) {
          setMessages(
            history.map((msg) => ({
              id: msg.id,  // Include message ID for retry functionality
              role: msg.role,
              content: msg.content,
            }))
          );
        }
      } catch (error) {
        console.error("Error loading chat history:", error);
      }
    };

    loadHistory();
  }, [sessionId]);

  // Load session context (metadata) when session changes
  useEffect(() => {
    const loadSessionContext = async () => {
      if (!sessionId) {
        setSessionContext(null);
        return;
      }

      try {
        const session = await chatService.getSession(sessionId);
        setSessionContext(session.session_metadata || {});
      } catch (error) {
        console.error("Error loading session context:", error);
        setSessionContext(null);
      }
    };

    loadSessionContext();
  }, [sessionId]);

  const sendMessage = useCallback(async () => {
    if (!inputValue.trim() || isLoading) return;

    const messageContent = inputValue.trim();

    // Create session if it doesn't exist (lazy creation)
    let currentSessionId = sessionId;
    let needsConnection = false;

    if (!currentSessionId) {
      try {
        const session = await chatService.createSession();
        currentSessionId = session.id;
        setSessionId(session.id);
        messageCountRef.current = 0;
        needsConnection = true;
        console.log("Chat session created:", session.id);
      } catch (error) {
        console.error("Error creating session:", error);
        toast.current?.show({
          severity: "error",
          summary: "Session Error",
          detail: "Failed to create chat session. Please make sure you are logged in.",
          life: 5000,
        });
        return;
      }
    }

    const userMessage = {
      role: "user",
      content: messageContent,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);
    streamingContentRef.current = "";
    setStreamingContent("");

    // If we just created a new session, wait for WebSocket to connect
    if (needsConnection) {
      console.log("Waiting for WebSocket connection...");

      // Wait up to 2 seconds for WebSocket to connect
      let retries = 0;
      const maxRetries = 20; // 20 * 100ms = 2 seconds

      while (wsRef.current?.readyState !== WebSocket.OPEN && retries < maxRetries) {
        await new Promise(resolve => setTimeout(resolve, 100));
        retries++;
      }

      if (wsRef.current?.readyState !== WebSocket.OPEN) {
        console.error("WebSocket failed to connect after session creation");
        toast.current?.show({
          severity: "error",
          summary: "Connection Failed",
          detail: "Could not connect to chat server. Please try again.",
          life: 5000,
        });
        setIsLoading(false);
        // Restore the message so user can try again
        setMessages((prev) => prev.slice(0, -1));
        setInputValue(messageContent);
        return;
      }

      console.log("WebSocket connected successfully");
    }

    // Send to WebSocket (history is now managed by backend)
    // Only send document_ids - backend filters by selected documents directly
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          message: userMessage.content,
          user_id: user?.id,
          document_ids: selectedDocumentIds.length > 0 ? selectedDocumentIds : [],
          graph_rag_enabled: graphRagEnabled,
        })
      );
    } else {
      toast.current?.show({
        severity: "warn",
        summary: "Not Connected",
        detail: "Reconnecting to server...",
        life: 3000,
      });
      setIsLoading(false);
      connectWebSocket();
    }
  }, [inputValue, isLoading, sessionId, connectWebSocket, selectedDocumentIds, graphRagEnabled]);

  const handleRetry = useCallback(async (msg, msgIndex) => {
    if (isLoading || !sessionId) return;

    console.log(`[AgenticChatBot] handleRetry called for message at index ${msgIndex}`);
    try {
      // If the message has an ID (from database), delete all messages after it
      if (msg.id) {
        console.log(`[AgenticChatBot] Deleting messages after message ${msg.id} in session ${sessionId}`);
        await chatService.deleteMessagesAfter(sessionId, msg.id);
        console.log(`[AgenticChatBot] Delete request completed, waiting 500ms for DB commit`);

        // Wait to ensure the delete operation is fully committed in the database
        // This prevents race conditions where the backend might load old history
        await new Promise(resolve => setTimeout(resolve, 500));
        console.log(`[AgenticChatBot] DB commit wait completed`);

        // Note: We don't call loadSessions() here because it will be called
        // automatically when the WebSocket response completes (in the "end" handler)
        // This prevents duplicate HTTP requests
      }

      // Remove all messages after this one from the UI (including this message)
      console.log(`[AgenticChatBot] Removing messages from UI, keeping first ${msgIndex} messages`);
      setMessages((prev) => prev.slice(0, msgIndex));

      // Resend the message
      const messageContent = msg.content;

      // Add the user message back to UI
      const userMessage = {
        role: "user",
        content: messageContent,
      };

      setMessages((prev) => [...prev, userMessage]);
      setIsLoading(true);
      streamingContentRef.current = "";
      setStreamingContent("");

      // Ensure WebSocket is connected - initiate connection if needed
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        console.log("[Retry] WebSocket not ready, initiating connection...");

        // Reset manual close flag since we want to connect
        isManualCloseRef.current = false;
        reconnectAttemptsRef.current = 0;

        // Initiate connection
        connectWebSocket();

        // Wait for connection with timeout
        let retries = 0;
        const maxRetries = 30; // 3 seconds total

        while ((!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) && retries < maxRetries) {
          await new Promise(resolve => setTimeout(resolve, 100));
          retries++;
        }

        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
          console.error("[Retry] WebSocket failed to connect after 3 seconds");
          toast.current?.show({
            severity: "error",
            summary: "Connection Failed",
            detail: "Could not connect to chat server. Please try again.",
            life: 5000,
          });
          setIsLoading(false);
          return;
        }

        console.log("[Retry] WebSocket connected successfully");
      }

      // Send to WebSocket with is_retry flag to force new document search
      // Only send document_ids - backend filters by selected documents directly
      wsRef.current.send(
        JSON.stringify({
          message: messageContent,
          user_id: user?.id,
          is_retry: true,
          document_ids: selectedDocumentIds.length > 0 ? selectedDocumentIds : [],
          graph_rag_enabled: graphRagEnabled,
        })
      );

      console.log(`[Retry] Sending message with is_retry=true, user_id=${user?.id}, document_ids=${selectedDocumentIds.length}, graph_rag_enabled=${graphRagEnabled}: ${messageContent}`);
    } catch (error) {
      console.error("Error retrying message:", error);
      toast.current?.show({
        severity: "error",
        summary: "Retry Failed",
        detail: "Failed to retry message. Please try again.",
        life: 5000,
      });
      setIsLoading(false);
    }
  }, [isLoading, sessionId, selectedDocumentIds, connectWebSocket, graphRagEnabled, user?.id]);

  // Start editing a message
  const handleStartEdit = useCallback((msgIndex, content) => {
    setEditingMessageIndex(msgIndex);
    setEditingContent(content);
  }, []);

  // Cancel editing
  const handleCancelEdit = useCallback(() => {
    setEditingMessageIndex(null);
    setEditingContent("");
  }, []);

  // Handle editing content change
  const handleEditContentChange = useCallback((newContent) => {
    setEditingContent(newContent);
  }, []);

  // Open markdown editor dialog for assistant messages
  const handleOpenMarkdownEditor = useCallback((msgIndex, content) => {
    setMarkdownEditorMessageIndex(msgIndex);
    setMarkdownEditorContent(content);
    setMarkdownEditorPreview(false);
    setMarkdownEditorVisible(true);
  }, []);

  // Close markdown editor dialog
  const handleCloseMarkdownEditor = useCallback(() => {
    setMarkdownEditorVisible(false);
    setMarkdownEditorContent("");
    setMarkdownEditorMessageIndex(null);
    setMarkdownEditorPreview(false);
  }, []);

  // Open export dialog for assistant message
  const handleOpenExportDialog = useCallback(async (content) => {
    setExportContent(content);
    setExportOption("markdown");
    setSelectedWorkspaceForExport(null);
    setShowCreateWorkspace(false);
    setNewWorkspaceName("");
    // Generate default filename from content
    setExportFilename(generateFilename(content));
    setExportDialogVisible(true);

    // Load workspaces for "Copy to workspace" option
    setLoadingWorkspaces(true);
    try {
      const result = await workspaceService.getWorkspaces(false);
      // The API returns a list directly, not wrapped in {workspaces: []}
      setWorkspaces(Array.isArray(result) ? result : (result.workspaces || []));
    } catch (error) {
      console.error("Error loading workspaces:", error);
      setWorkspaces([]);
    } finally {
      setLoadingWorkspaces(false);
    }
  }, []);

  // Close export dialog
  const handleCloseExportDialog = useCallback(() => {
    setExportDialogVisible(false);
    setExportContent("");
    setExportOption("markdown");
    setExportFilename("");
    setSelectedWorkspaceForExport(null);
    setShowCreateWorkspace(false);
    setNewWorkspaceName("");
  }, []);

  // Create new workspace from export dialog
  const handleCreateWorkspaceForExport = useCallback(async () => {
    if (!newWorkspaceName.trim()) {
      toast.current?.show({
        severity: "warn",
        summary: t("Workspace.NameRequired"),
        detail: t("Workspace.NameRequired"),
        life: 3000,
      });
      return;
    }

    setIsCreatingWorkspace(true);
    try {
      const newWorkspace = await workspaceService.createWorkspace({
        name: newWorkspaceName.trim(),
        color: "#6366f1",
        icon: "folder"
      });

      // Add new workspace to the list and select it
      setWorkspaces(prev => [...prev, newWorkspace]);
      setSelectedWorkspaceForExport(newWorkspace.id);
      setShowCreateWorkspace(false);
      setNewWorkspaceName("");

      toast.current?.show({
        severity: "success",
        summary: t("Workspace.CreateSuccess"),
        detail: t("Workspace.CreateSuccess"),
        life: 3000,
      });
    } catch (error) {
      console.error("Error creating workspace:", error);
      toast.current?.show({
        severity: "error",
        summary: t("Workspace.CreateFailed"),
        detail: error.response?.data?.detail || t("Workspace.CreateFailed"),
        life: 5000,
      });
    } finally {
      setIsCreatingWorkspace(false);
    }
  }, [newWorkspaceName, t]);

  // Handle export action
  const handleExport = useCallback(async () => {
    if (!exportContent) return;

    // Use the user-specified filename or fallback to generated one
    const filename = exportFilename.trim() || generateFilename(exportContent);

    if (exportOption === "markdown") {
      exportAsMarkdown(exportContent, filename);
      handleCloseExportDialog();
      toast.current?.show({
        severity: "success",
        summary: t("Chat.ExportSuccess"),
        detail: t("Chat.ExportedAsMarkdown"),
        life: 3000,
      });
    } else if (exportOption === "pdf") {
      exportAsPdf(exportContent, filename);
      handleCloseExportDialog();
      toast.current?.show({
        severity: "success",
        summary: t("Chat.ExportSuccess"),
        detail: t("Chat.ExportedAsPdf"),
        life: 3000,
      });
    } else if (exportOption === "word") {
      exportAsWord(exportContent, filename);
      handleCloseExportDialog();
      toast.current?.show({
        severity: "success",
        summary: t("Chat.ExportSuccess"),
        detail: t("Chat.ExportedAsWord"),
        life: 3000,
      });
    } else if (exportOption === "excel") {
      exportAsExcel(exportContent, filename);
      handleCloseExportDialog();
      toast.current?.show({
        severity: "success",
        summary: t("Chat.ExportSuccess"),
        detail: t("Chat.ExportedAsExcel"),
        life: 3000,
      });
    } else if (exportOption === "workspace") {
      if (!selectedWorkspaceForExport) {
        toast.current?.show({
          severity: "warn",
          summary: t("Chat.SelectWorkspace"),
          detail: t("Chat.PleaseSelectWorkspace"),
          life: 3000,
        });
        return;
      }

      setIsExporting(true);
      try {
        // Use the new backend API to save markdown to workspace
        await workspaceService.saveMarkdownToWorkspace(
          exportContent,
          filename,
          selectedWorkspaceForExport
        );

        handleCloseExportDialog();
        toast.current?.show({
          severity: "success",
          summary: t("Chat.ExportSuccess"),
          detail: t("Chat.SavedToWorkspace"),
          life: 3000,
        });
      } catch (error) {
        console.error("Error saving to workspace:", error);
        toast.current?.show({
          severity: "error",
          summary: t("Chat.ExportFailed"),
          detail: error.response?.data?.detail || t("Chat.FailedToSaveToWorkspace"),
          life: 5000,
        });
      } finally {
        setIsExporting(false);
      }
    }
  }, [exportContent, exportOption, exportFilename, selectedWorkspaceForExport, handleCloseExportDialog, t]);

  // Save markdown editor content
  const handleSaveMarkdownEditor = useCallback(async () => {
    if (markdownEditorMessageIndex === null || !sessionId || isSavingCorrection) return;

    const msg = messages[markdownEditorMessageIndex];
    if (!msg || !msg.id) return;

    const newContent = markdownEditorContent.trim();
    if (!newContent || newContent === msg.content) {
      handleCloseMarkdownEditor();
      return;
    }

    setIsSavingCorrection(true);
    try {
      await chatService.updateMessage(sessionId, msg.id, newContent);

      // Update the message in UI
      setMessages((prev) => {
        const updated = [...prev];
        updated[markdownEditorMessageIndex] = {
          ...updated[markdownEditorMessageIndex],
          content: newContent,
          isEdited: true
        };
        return updated;
      });

      toast.current?.show({
        severity: "success",
        summary: t("Chat.CorrectionSaved"),
        detail: t("Chat.CorrectionSavedDetail"),
        life: 3000,
      });

      handleCloseMarkdownEditor();
    } catch (error) {
      console.error("Error saving correction:", error);
      toast.current?.show({
        severity: "error",
        summary: t("Chat.CorrectionFailed"),
        detail: t("Chat.CorrectionFailedDetail"),
        life: 5000,
      });
    } finally {
      setIsSavingCorrection(false);
    }
  }, [markdownEditorMessageIndex, markdownEditorContent, sessionId, messages, isSavingCorrection, handleCloseMarkdownEditor, t]);

  // Submit correction for assistant message (just updates the database)
  const handleSubmitCorrection = useCallback(async (msg, msgIndex) => {
    if (!sessionId || !msg.id || isSavingCorrection) return;

    const newContent = editingContent.trim();
    if (!newContent || newContent === msg.content) {
      handleCancelEdit();
      return;
    }

    setIsSavingCorrection(true);
    try {
      await chatService.updateMessage(sessionId, msg.id, newContent);

      // Update the message in UI
      setMessages((prev) => {
        const updated = [...prev];
        updated[msgIndex] = {
          ...updated[msgIndex],
          content: newContent,
          isEdited: true
        };
        return updated;
      });

      toast.current?.show({
        severity: "success",
        summary: t("Chat.CorrectionSaved"),
        detail: t("Chat.CorrectionSavedDetail"),
        life: 3000,
      });

      handleCancelEdit();
    } catch (error) {
      console.error("Error saving correction:", error);
      toast.current?.show({
        severity: "error",
        summary: t("Chat.CorrectionFailed"),
        detail: t("Chat.CorrectionFailedDetail"),
        life: 5000,
      });
    } finally {
      setIsSavingCorrection(false);
    }
  }, [sessionId, editingContent, isSavingCorrection, handleCancelEdit, t]);

  // Handle retry with edited user message
  const handleRetryWithEdit = useCallback(async (msg, msgIndex) => {
    if (isLoading || !sessionId) return;

    const newContent = editingContent.trim();
    if (!newContent) {
      handleCancelEdit();
      return;
    }

    // If content hasn't changed, just do a normal retry
    if (newContent === msg.content) {
      handleCancelEdit();
      handleRetry(msg, msgIndex);
      return;
    }

    console.log(`[AgenticChatBot] handleRetryWithEdit called for message at index ${msgIndex}`);
    try {
      // If the message has an ID (from database), update it and delete messages after it
      if (msg.id) {
        // First update the message content
        await chatService.updateMessage(sessionId, msg.id, newContent);
        console.log(`[AgenticChatBot] Updated message ${msg.id} with new content`);

        // Then delete all messages after this one
        await chatService.deleteMessagesAfter(sessionId, msg.id);
        console.log(`[AgenticChatBot] Deleted messages after ${msg.id}`);

        // Wait for DB commit
        await new Promise(resolve => setTimeout(resolve, 500));

        // Note: We don't call loadSessions() here because it will be called
        // automatically when the WebSocket response completes (in the "end" handler)
        // This prevents duplicate HTTP requests
      }

      // Cancel edit mode
      handleCancelEdit();

      // Remove all messages after this one from the UI (including this message)
      setMessages((prev) => prev.slice(0, msgIndex));

      // Add the updated user message back to UI
      const userMessage = {
        id: msg.id,
        role: "user",
        content: newContent,
      };

      setMessages((prev) => [...prev, userMessage]);
      setIsLoading(true);
      streamingContentRef.current = "";
      setStreamingContent("");

      // Ensure WebSocket is connected - initiate connection if needed
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        console.log("[RetryWithEdit] WebSocket not ready, initiating connection...");

        // Reset manual close flag since we want to connect
        isManualCloseRef.current = false;
        reconnectAttemptsRef.current = 0;

        // Initiate connection
        connectWebSocket();

        // Wait for connection with timeout
        let retries = 0;
        const maxRetries = 30; // 3 seconds total

        while ((!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) && retries < maxRetries) {
          await new Promise(resolve => setTimeout(resolve, 100));
          retries++;
        }

        if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
          console.error("[RetryWithEdit] WebSocket failed to connect after 3 seconds");
          toast.current?.show({
            severity: "error",
            summary: "Connection Failed",
            detail: "Could not connect to chat server. Please try again.",
            life: 5000,
          });
          setIsLoading(false);
          return;
        }

        console.log("[RetryWithEdit] WebSocket connected successfully");
      }

      // Send to WebSocket with is_retry flag to force new document search
      // Only send document_ids - backend filters by selected documents directly
      wsRef.current.send(
        JSON.stringify({
          message: newContent,
          user_id: user?.id,
          is_retry: true,
          document_ids: selectedDocumentIds.length > 0 ? selectedDocumentIds : [],
          graph_rag_enabled: graphRagEnabled,
        })
      );

      console.log(`[Retry] Sending edited message with is_retry=true, user_id=${user?.id}, document_ids=${selectedDocumentIds.length}, graph_rag_enabled=${graphRagEnabled}: ${newContent}`);
    } catch (error) {
      console.error("Error retrying with edit:", error);
      toast.current?.show({
        severity: "error",
        summary: "Retry Failed",
        detail: "Failed to retry message. Please try again.",
        life: 5000,
      });
      setIsLoading(false);
    }
  }, [isLoading, sessionId, editingContent, handleCancelEdit, handleRetry, selectedDocumentIds, connectWebSocket, graphRagEnabled, user?.id]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = async () => {
    // Clear current session and messages
    // New session will be created when user sends first message

    // Reset reconnection counter and message count
    reconnectAttemptsRef.current = 0;
    messageCountRef.current = 0;

    // Mark as manual close BEFORE closing to prevent reconnection attempts
    isManualCloseRef.current = true;

    // Stop heartbeat
    stopHeartbeat();

    // Clear any pending reconnect timeout
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Close WebSocket and clear reference
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    // Now clear session state (after WebSocket is closed)
    setSessionId(null);
    setMessages([]);
    streamingContentRef.current = "";
    setStreamingContent("");
    setIsConnected(false);
    updateChatStatus(false, false, false); // disconnected, no session, not reconnecting

    // Reset workspace/document selection to user preferences for new chat
    // This ensures new chat starts with user's default document sources
    try {
      const result = await authService.getChatPreferences();
      if (result.success && result.chat) {
        const prefWorkspaceIds = result.chat.selectedWorkspaceIds || [];
        const prefDocumentIds = result.chat.selectedDocumentIds || [];
        console.log("[AgenticChatBot] New chat - restoring user preferences:", prefWorkspaceIds);
        setSelectedWorkspaceIds(prefWorkspaceIds);
        setSelectedDocumentIds(prefDocumentIds);
      }
    } catch (error) {
      console.error("[AgenticChatBot] Error loading chat preferences for new chat:", error);
      // On error, keep current selection rather than clearing it
    }

    // Reload chat history to show the new session will be created
    if (chatHistoryRef.current) {
      chatHistoryRef.current.loadSessions();
    }

    toast.current?.show({
      severity: "success",
      summary: "New Chat",
      detail: "Ready to start a new conversation",
      life: 3000,
    });
  };

  const handleSessionSelect = async (newSessionId) => {
    if (newSessionId === sessionId) return;

    try {
      // Load messages and session details in parallel
      const [history, sessionDetails] = await Promise.all([
        chatService.getSessionMessages(newSessionId),
        chatService.getSession(newSessionId)
      ]);

      const loadedMessages = history.map((msg) => ({
        id: msg.id,  // Include message ID for retry functionality
        role: msg.role,
        content: msg.content,
      }));

      // IMPORTANT: Close existing WebSocket BEFORE changing session ID
      // This prevents race conditions with reconnection attempts

      // Mark as manual close to prevent reconnection to old session
      isManualCloseRef.current = true;

      // Stop heartbeat
      stopHeartbeat();

      // Clear any pending reconnect timeout
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }

      // Close old WebSocket and clear reference
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }

      setIsConnected(false);

      // Reset reconnection counter and set message count
      reconnectAttemptsRef.current = 0;
      messageCountRef.current = loadedMessages.length;

      // Now update session ID - this will trigger useEffect to create new connection
      setMessages(loadedMessages);
      setSessionId(newSessionId);

      // Restore workspace/document selection from session metadata (if available)
      // This allows reopening a chat with the same document context it was using
      // Note: This does NOT update user preferences - only applies to this session
      console.log("[AgenticChatBot] Session metadata:", sessionDetails?.session_metadata);

      if (sessionDetails?.session_metadata) {
        const metadata = sessionDetails.session_metadata;
        const sessionWorkspaceIds = metadata._prev_workspace_ids || [];
        const sessionDocumentIds = metadata._prev_document_ids || [];

        console.log("[AgenticChatBot] Restoring selection from session:", {
          workspaceIds: sessionWorkspaceIds,
          documentIds: sessionDocumentIds
        });

        // Always set the selection from session metadata (even if empty)
        // This ensures the WorkspaceBrowser reflects the session's state
        setSelectedWorkspaceIds(sessionWorkspaceIds);
        setSelectedDocumentIds(sessionDocumentIds);

        // Update the last saved selection to match what we just restored
        // This prevents the save effect from immediately re-saving the same data
        lastSavedSelectionRef.current = {
          workspaceIds: [...sessionWorkspaceIds],
          documentIds: [...sessionDocumentIds]
        };
        // Mark as initialized since we have valid data from session
        selectionInitializedRef.current = true;
      } else {
        // Session has no metadata - clear the selection to start fresh
        console.log("[AgenticChatBot] Session has no metadata, clearing selection");
        setSelectedWorkspaceIds([]);
        setSelectedDocumentIds([]);
        lastSavedSelectionRef.current = { workspaceIds: [], documentIds: [] };
        selectionInitializedRef.current = true;
      }

      // Check if title needs regeneration (do this in background)
      chatService.regenerateSessionTitle(newSessionId)
        .then((result) => {
          if (result.new_title && result.new_title !== "New Chat") {
            // Refresh chat history to show new title
            chatHistoryRef.current?.loadSessions();
          }
        })
        .catch((error) => {
          // Silently fail - title regeneration is not critical
          console.log("Title regeneration skipped or failed:", error);
        });

      toast.current?.show({
        severity: "success",
        summary: "Session Loaded",
        detail: "Chat history loaded",
        life: 2000,
      });
    } catch (error) {
      console.error("Error loading session:", error);
      toast.current?.show({
        severity: "error",
        summary: "Error",
        detail: "Failed to load chat session",
        life: 5000,
      });
    }
  };

  return (
    <div className="chatbot-container">
      <Toast ref={toast} />

      {/* Left Sidebar - Chat History */}
      <div className="chatbot-sidebar">
        <ChatHistory
          ref={chatHistoryRef}
          currentSessionId={sessionId}
          onSessionSelect={handleSessionSelect}
        />
      </div>

      {/* Main Chat Area */}
      <div className="chatbot-main">
        {isInitializing ? (
          <div className="chatbot-loading">
            <ProgressSpinner />
            <p>Initializing chat session...</p>
          </div>
        ) : (
          <>
            {/* Header */}
            <div className="chatbot-header">
              <div className="header-title">
                <i className="pi pi-comments" />
                <span>Document Chat Assistant</span>
              </div>
              <div className="header-actions">
                {/* Mobile workspace browser toggle */}
                <Button
                  icon="pi pi-database"
                  className="p-button-text p-button-secondary mobile-workspace-toggle"
                  onClick={() => setMobileWorkspaceDrawerOpen(true)}
                  tooltip="Knowledge Sources"
                  tooltipOptions={{ position: "left" }}
                  badge={selectedWorkspaceIds.length > 0 ? String(selectedWorkspaceIds.length) : null}
                  badgeClassName="workspace-badge"
                />
                <Button
                  icon="pi pi-plus"
                  className="p-button-text p-button-secondary"
                  onClick={clearChat}
                  tooltip="New chat"
                  tooltipOptions={{ position: "left" }}
                />
              </div>
            </div>

          {/* Messages Area */}
          <div className="chatbot-messages">
            {messages.length === 0 && !streamingContent && (
              <div className="empty-state">
                <i className="pi pi-file-pdf" />
                <h3>Welcome to Document Chat</h3>
                <p>Ask questions about your uploaded documents.</p>
                <p className="hint">The assistant will search through indexed documents to find relevant information.</p>
              </div>
            )}

        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role} ${msg.hasError ? 'has-error' : ''} ${editingMessageIndex === idx ? 'editing' : ''}`}>
            <div className="message-avatar">
              <i className={`pi ${msg.role === "user" ? "pi-user" : "pi-android"}`} />
            </div>
            <div className="message-content-wrapper">
              {editingMessageIndex === idx ? (
                // Editing mode
                <div className="message-edit-container">
                  <InputTextarea
                    value={editingContent}
                    onChange={(e) => handleEditContentChange(e.target.value)}
                    rows={3}
                    autoResize
                    className="message-edit-textarea"
                    disabled={isLoading || isSavingCorrection}
                  />
                  <div className="message-edit-actions">
                    {msg.role === "user" ? (
                      // User message: Retry button to resend with edited content
                      <Button
                        icon={isLoading ? "pi pi-spin pi-spinner" : "pi pi-refresh"}
                        label={t("Chat.RetryWithEdit")}
                        className="p-button-sm p-button-primary"
                        onClick={() => handleRetryWithEdit(msg, idx)}
                        disabled={isLoading || !editingContent.trim()}
                        tooltip={t("Chat.RetryWithEditTooltip")}
                        tooltipOptions={{ position: "top" }}
                      />
                    ) : (
                      // Assistant message: Submit Correction button
                      <Button
                        icon={isSavingCorrection ? "pi pi-spin pi-spinner" : "pi pi-check"}
                        label={t("Chat.SubmitCorrection")}
                        className="p-button-sm p-button-success"
                        onClick={() => handleSubmitCorrection(msg, idx)}
                        disabled={isSavingCorrection || !editingContent.trim()}
                        tooltip={t("Chat.SubmitCorrectionTooltip")}
                        tooltipOptions={{ position: "top" }}
                      />
                    )}
                    <Button
                      icon="pi pi-times"
                      label={t("Chat.Cancel")}
                      className="p-button-sm p-button-secondary p-button-text"
                      onClick={handleCancelEdit}
                      disabled={isLoading || isSavingCorrection}
                    />
                  </div>
                </div>
              ) : (
                // Display mode
                <>
                  <Card className={`message-content ${msg.isEdited ? 'edited' : ''}`}>
                    {msg.role === "assistant" ? (
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                    ) : (
                      <p>{msg.content}</p>
                    )}
                    {msg.isEdited && (
                      <span className="edited-indicator">{t("Chat.Edited")}</span>
                    )}
                  </Card>
                  <div className="message-actions">
                    {msg.role === "user" ? (
                      // User message actions
                      <>
                        <Button
                          icon="pi pi-pencil"
                          className="p-button-text p-button-sm edit-button"
                          onClick={() => handleStartEdit(idx, msg.content)}
                          tooltip={t("Chat.EditMessage")}
                          tooltipOptions={{ position: "top" }}
                          disabled={isLoading || editingMessageIndex !== null}
                        />
                        <Button
                          icon="pi pi-refresh"
                          className="p-button-text p-button-sm retry-button"
                          onClick={() => handleRetry(msg, idx)}
                          tooltip={t("Chat.RetryMessage")}
                          tooltipOptions={{ position: "top" }}
                          disabled={isLoading || editingMessageIndex !== null}
                        />
                      </>
                    ) : (
                      // Assistant message actions - edit and download
                      <>
                        <Button
                          icon="pi pi-pencil"
                          className="p-button-text p-button-sm edit-button"
                          onClick={() => handleOpenMarkdownEditor(idx, msg.content)}
                          tooltip={t("Chat.EditResponse")}
                          tooltipOptions={{ position: "top" }}
                          disabled={isLoading || markdownEditorVisible || !msg.id}
                        />
                        <Button
                          icon="pi pi-download"
                          className="p-button-text p-button-sm download-button"
                          onClick={() => handleOpenExportDialog(msg.content)}
                          tooltip={t("Chat.DownloadResponse")}
                          tooltipOptions={{ position: "top" }}
                          disabled={isLoading || exportDialogVisible}
                        />
                      </>
                    )}
                  </div>
                </>
              )}
              {msg.hasError && (
                <div className="error-indicator">
                  <i className="pi pi-exclamation-triangle" />
                  <span>{t("Chat.FailedToSend")}</span>
                </div>
              )}
            </div>
          </div>
        ))}

        {streamingContent && (
          <div className="message assistant">
            <div className="message-avatar">
              <i className="pi pi-android" />
            </div>
            <Card className="message-content streaming">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{streamingContent}</ReactMarkdown>
              <span className="typing-cursor"></span>
            </Card>
          </div>
        )}

        {isLoading && !streamingContent && (
          <div className="message assistant loading">
            <div className="message-avatar">
              <i className="pi pi-android" />
            </div>
            <Card className="message-content progress-card">
              <div className="progress-indicator">
                <div className="progress-dots">
                  <span className="dot"></span>
                  <span className="dot"></span>
                  <span className="dot"></span>
                </div>
                <span className="progress-text">{progressStep || t("Chat.ProcessingRequest")}</span>
              </div>
            </Card>
          </div>
        )}

            <div ref={messagesEndRef} />
          </div>

          {/* Context Indicators - Full Width Topics Display */}
          {sessionContext && sessionContext.topics?.length > 0 && (
            <div className="context-topics-bar">
              <i className="pi pi-tag" />
              <span className="topics-label">Topics:</span>
              <div className="topics-list">
                {sessionContext.topics.map((topic, index) => (
                  <span key={index} className="topic-tag">{topic}</span>
                ))}
              </div>
            </div>
          )}

          {/* Input Area with Graph RAG Toggle */}
          <div className="chatbot-input-wrapper">
            {/* Graph RAG Toggle */}
            <div className="graph-rag-toggle">
              <Checkbox
                inputId="graphRagEnabled"
                checked={graphRagEnabled}
                onChange={(e) => setGraphRagEnabled(e.checked)}
                disabled={isLoading}
              />
              <label htmlFor="graphRagEnabled" className="graph-rag-label">
                <i className="pi pi-sitemap" />
                {t("Chat.EnableGraphKnowledgeReasoning", "Enable Graph Knowledge Reasoning")}
              </label>
            </div>
            <div className="chatbot-input">
              <InputTextarea
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question about your documents..."
                rows={2}
                autoResize
                disabled={isLoading || (sessionId && !isConnected)}
              />
              <Button
                icon={isLoading ? "pi pi-spin pi-spinner" : "pi pi-send"}
                onClick={sendMessage}
                disabled={!inputValue.trim() || isLoading || (sessionId && !isConnected)}
                className="send-button"
              />
            </div>
          </div>
          </>
        )}
      </div>

      {/* Right Sidebar - Workspace Browser (Desktop) */}
      <WorkspaceBrowser
        ref={workspaceBrowserRef}
        selectedWorkspaceIds={selectedWorkspaceIds}
        selectedDocumentIds={selectedDocumentIds}
        onSelectionChange={setSelectedWorkspaceIds}
        onDocumentSelectionChange={setSelectedDocumentIds}
        collapsed={workspaceBrowserCollapsed}
        onToggleCollapse={() => setWorkspaceBrowserCollapsed(!workspaceBrowserCollapsed)}
      />

      {/* Mobile Workspace Browser Drawer */}
      {mobileWorkspaceDrawerOpen && (
        <div className="mobile-workspace-overlay" onClick={() => setMobileWorkspaceDrawerOpen(false)}>
          <div className="mobile-workspace-drawer" onClick={(e) => e.stopPropagation()}>
            <div className="mobile-drawer-header">
              <span>Knowledge Sources</span>
              <Button
                icon="pi pi-times"
                className="p-button-text p-button-secondary"
                onClick={() => setMobileWorkspaceDrawerOpen(false)}
              />
            </div>
            <WorkspaceBrowser
              selectedWorkspaceIds={selectedWorkspaceIds}
              selectedDocumentIds={selectedDocumentIds}
              onSelectionChange={setSelectedWorkspaceIds}
              onDocumentSelectionChange={setSelectedDocumentIds}
              collapsed={false}
              onToggleCollapse={() => setMobileWorkspaceDrawerOpen(false)}
            />
          </div>
        </div>
      )}

      {/* Markdown Editor Dialog for Assistant Messages */}
      <Dialog
        visible={markdownEditorVisible}
        onHide={handleCloseMarkdownEditor}
        header={t("Chat.EditMarkdownTitle")}
        className="markdown-editor-dialog"
        style={{ width: "90vw", maxWidth: "1200px" }}
        modal
        draggable={false}
        resizable={false}
        footer={
          <div className="markdown-editor-footer">
            <Button
              label={t("Chat.Cancel")}
              icon="pi pi-times"
              className="p-button-text p-button-secondary"
              onClick={handleCloseMarkdownEditor}
              disabled={isSavingCorrection}
            />
            <Button
              label={markdownEditorPreview ? t("Chat.EditMode") : t("Chat.PreviewMode")}
              icon={markdownEditorPreview ? "pi pi-pencil" : "pi pi-eye"}
              className="p-button-text"
              onClick={() => setMarkdownEditorPreview(!markdownEditorPreview)}
              disabled={isSavingCorrection}
            />
            <Button
              label={t("Chat.SaveChanges")}
              icon={isSavingCorrection ? "pi pi-spin pi-spinner" : "pi pi-check"}
              className="p-button-primary"
              onClick={handleSaveMarkdownEditor}
              disabled={isSavingCorrection || !markdownEditorContent.trim()}
            />
          </div>
        }
      >
        <div className="markdown-editor-content">
          {markdownEditorPreview ? (
            <div className="markdown-preview">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{markdownEditorContent}</ReactMarkdown>
            </div>
          ) : (
            <InputTextarea
              value={markdownEditorContent}
              onChange={(e) => setMarkdownEditorContent(e.target.value)}
              className="markdown-textarea"
              placeholder={t("Chat.MarkdownPlaceholder")}
              disabled={isSavingCorrection}
            />
          )}
        </div>
      </Dialog>

      {/* Export/Download Dialog */}
      <Dialog
        visible={exportDialogVisible}
        onHide={handleCloseExportDialog}
        header={t("Chat.ExportDialogTitle")}
        className="export-dialog"
        style={{ width: "450px" }}
        modal
        draggable={false}
        resizable={false}
        footer={
          <div className="export-dialog-footer">
            <Button
              label={t("Chat.Cancel")}
              icon="pi pi-times"
              className="p-button-text p-button-secondary"
              onClick={handleCloseExportDialog}
              disabled={isExporting}
            />
            <Button
              label={exportOption === "workspace" ? t("Chat.SaveToWorkspace") : t("Chat.Download")}
              icon={isExporting ? "pi pi-spin pi-spinner" : (exportOption === "workspace" ? "pi pi-upload" : "pi pi-download")}
              className="p-button-primary"
              onClick={handleExport}
              disabled={isExporting || (exportOption === "workspace" && !selectedWorkspaceForExport)}
            />
          </div>
        }
      >
        <div className="export-dialog-content">
          {/* Filename input */}
          <div className="export-filename-section">
            <label htmlFor="export-filename" className="export-filename-label">
              {t("Chat.Filename")}
            </label>
            <InputText
              id="export-filename"
              value={exportFilename}
              onChange={(e) => setExportFilename(e.target.value)}
              placeholder={t("Chat.FilenamePlaceholder")}
              className="export-filename-input"
            />
          </div>

          <p className="export-description">{t("Chat.ExportDescription")}</p>

          <div className="export-options">
            <div className="export-option" onClick={() => setExportOption("markdown")}>
              <RadioButton
                inputId="export-markdown"
                name="exportOption"
                value="markdown"
                checked={exportOption === "markdown"}
                onChange={(e) => setExportOption(e.value)}
              />
              <label htmlFor="export-markdown" className="export-option-label">
                <i className="pi pi-file" />
                <span>{t("Chat.ExportMarkdown")}</span>
              </label>
            </div>

            <div className="export-option" onClick={() => setExportOption("pdf")}>
              <RadioButton
                inputId="export-pdf"
                name="exportOption"
                value="pdf"
                checked={exportOption === "pdf"}
                onChange={(e) => setExportOption(e.value)}
              />
              <label htmlFor="export-pdf" className="export-option-label">
                <i className="pi pi-file-pdf" />
                <span>{t("Chat.ExportPdf")}</span>
              </label>
            </div>

            <div className="export-option" onClick={() => setExportOption("word")}>
              <RadioButton
                inputId="export-word"
                name="exportOption"
                value="word"
                checked={exportOption === "word"}
                onChange={(e) => setExportOption(e.value)}
              />
              <label htmlFor="export-word" className="export-option-label">
                <i className="pi pi-file-word" />
                <span>{t("Chat.ExportWord")}</span>
              </label>
            </div>

            <div className="export-option" onClick={() => setExportOption("excel")}>
              <RadioButton
                inputId="export-excel"
                name="exportOption"
                value="excel"
                checked={exportOption === "excel"}
                onChange={(e) => setExportOption(e.value)}
              />
              <label htmlFor="export-excel" className="export-option-label">
                <i className="pi pi-file-excel" />
                <span>{t("Chat.ExportExcel")}</span>
              </label>
            </div>

            <div className="export-option" onClick={() => setExportOption("workspace")}>
              <RadioButton
                inputId="export-workspace"
                name="exportOption"
                value="workspace"
                checked={exportOption === "workspace"}
                onChange={(e) => setExportOption(e.value)}
              />
              <label htmlFor="export-workspace" className="export-option-label">
                <i className="pi pi-folder" />
                <span>{t("Chat.ExportToWorkspace")}</span>
              </label>
            </div>
          </div>

          {exportOption === "workspace" && (
            <div className="workspace-selector">
              {loadingWorkspaces ? (
                <div className="loading-workspaces">
                  <ProgressSpinner style={{ width: "30px", height: "30px" }} />
                  <span>{t("Chat.LoadingWorkspaces")}</span>
                </div>
              ) : (
                <>
                  {workspaces.length > 0 && (
                    <Dropdown
                      value={selectedWorkspaceForExport}
                      options={workspaces}
                      onChange={(e) => setSelectedWorkspaceForExport(e.value)}
                      optionLabel="name"
                      optionValue="id"
                      placeholder={t("Chat.SelectWorkspacePlaceholder")}
                      className="workspace-dropdown"
                      disabled={showCreateWorkspace}
                      itemTemplate={(option) => (
                        <div className="workspace-dropdown-item">
                          <i className="pi pi-folder" style={{ color: option.color }} />
                          <span>{option.name}</span>
                        </div>
                      )}
                      valueTemplate={(option) => {
                        if (!option) return <span>{t("Chat.SelectWorkspacePlaceholder")}</span>;
                        return (
                          <div className="workspace-dropdown-item">
                            <i className="pi pi-folder" style={{ color: option.color }} />
                            <span>{option.name}</span>
                          </div>
                        );
                      }}
                    />
                  )}

                  {workspaces.length === 0 && !showCreateWorkspace && (
                    <div className="no-workspaces">
                      <i className="pi pi-info-circle" />
                      <span>{t("Chat.NoWorkspacesAvailable")}</span>
                    </div>
                  )}

                  {/* Create New Workspace Section */}
                  {showCreateWorkspace ? (
                    <div className="create-workspace-form">
                      <div className="create-workspace-input-group">
                        <InputText
                          value={newWorkspaceName}
                          onChange={(e) => setNewWorkspaceName(e.target.value)}
                          placeholder={t("Chat.NewWorkspaceName")}
                          className="create-workspace-input"
                          disabled={isCreatingWorkspace}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" && !isCreatingWorkspace) {
                              handleCreateWorkspaceForExport();
                            }
                          }}
                        />
                        <Button
                          icon={isCreatingWorkspace ? "pi pi-spin pi-spinner" : "pi pi-check"}
                          className="p-button-success p-button-sm"
                          onClick={handleCreateWorkspaceForExport}
                          disabled={isCreatingWorkspace || !newWorkspaceName.trim()}
                          tooltip={t("Chat.CreateWorkspace")}
                          tooltipOptions={{ position: "top" }}
                        />
                        <Button
                          icon="pi pi-times"
                          className="p-button-secondary p-button-sm"
                          onClick={() => {
                            setShowCreateWorkspace(false);
                            setNewWorkspaceName("");
                          }}
                          disabled={isCreatingWorkspace}
                          tooltip={t("Chat.Cancel")}
                          tooltipOptions={{ position: "top" }}
                        />
                      </div>
                    </div>
                  ) : (
                    <Button
                      label={t("Chat.CreateNewWorkspace")}
                      icon="pi pi-plus"
                      className="p-button-text p-button-sm create-workspace-button"
                      onClick={() => setShowCreateWorkspace(true)}
                    />
                  )}
                </>
              )}
            </div>
          )}
        </div>
      </Dialog>
    </div>
  );
};

