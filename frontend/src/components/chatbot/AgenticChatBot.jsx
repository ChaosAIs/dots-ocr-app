import { useState, useRef, useEffect, useCallback } from "react";
import { InputTextarea } from "primereact/inputtextarea";
import { Button } from "primereact/button";
import { Card } from "primereact/card";
import { ProgressSpinner } from "primereact/progressspinner";
import { Toast } from "primereact/toast";
import { Dialog } from "primereact/dialog";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import chatService from "../../services/chatService";
import authService from "../../services/authService";
import { ChatHistory } from "./ChatHistory";
import { WorkspaceBrowser } from "./WorkspaceBrowser";
import { useTranslation } from "react-i18next";
import { useAuth } from "../../core/auth/components/authProvider";
import "./AgenticChatBot.scss";

// Get WebSocket URL from environment or use default
const WS_BASE_URL = process.env.REACT_APP_WS_URL || "ws://localhost:8080";

export const AgenticChatBot = () => {
  const { t } = useTranslation();
  const { user } = useAuth();
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

  // Workspace browser state
  const [selectedWorkspaceIds, setSelectedWorkspaceIds] = useState([]);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState([]);
  const [workspaceBrowserCollapsed, setWorkspaceBrowserCollapsed] = useState(false);
  const [mobileWorkspaceDrawerOpen, setMobileWorkspaceDrawerOpen] = useState(false);

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
      reconnectAttemptsRef.current = 0; // Reset counter on successful connection
      isManualCloseRef.current = false;
      console.log("[WebSocket] Connected successfully");

      // Start heartbeat to keep connection alive
      startHeartbeat(ws);
    };

    ws.onclose = (event) => {
      setIsConnected(false);
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
          const progressPercent = data.percent;

          // Update streaming content to show progress
          const progressText = progressPercent
            ? `🔄 ${progressMsg} (${progressPercent}%)`
            : `🔄 ${progressMsg}`;

          setStreamingContent(progressText);
        } else if (data.type === "token") {
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
          setIsLoading(false);
        }
      }
    };

    wsRef.current = ws;
  }, [sessionId, startHeartbeat, stopHeartbeat, getReconnectDelay]);

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
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          message: userMessage.content,
          user_id: user?.id,
          workspace_ids: selectedWorkspaceIds.length > 0 ? selectedWorkspaceIds : [],
          document_ids: selectedDocumentIds.length > 0 ? selectedDocumentIds : [],
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
  }, [inputValue, isLoading, sessionId, connectWebSocket, selectedWorkspaceIds, selectedDocumentIds]);

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

        // Refresh the chat history sidebar to show updated message count
        if (chatHistoryRef.current?.loadSessions) {
          console.log(`[AgenticChatBot] Refreshing chat history sidebar after message deletion`);
          chatHistoryRef.current.loadSessions();
        }
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
      wsRef.current.send(
        JSON.stringify({
          message: messageContent,
          user_id: user?.id,
          is_retry: true,
          workspace_ids: selectedWorkspaceIds.length > 0 ? selectedWorkspaceIds : [],
          document_ids: selectedDocumentIds.length > 0 ? selectedDocumentIds : [],
        })
      );

      console.log(`[Retry] Sending message with is_retry=true, user_id=${user?.id}, workspace_ids=${selectedWorkspaceIds.length}, document_ids=${selectedDocumentIds.length}: ${messageContent}`);
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
  }, [isLoading, sessionId, selectedWorkspaceIds, selectedDocumentIds, connectWebSocket]);

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

        // Refresh the chat history sidebar to show updated message count
        if (chatHistoryRef.current?.loadSessions) {
          console.log(`[AgenticChatBot] Refreshing chat history sidebar after message deletion`);
          chatHistoryRef.current.loadSessions();
        }
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
      wsRef.current.send(
        JSON.stringify({
          message: newContent,
          user_id: user?.id,
          is_retry: true,
          workspace_ids: selectedWorkspaceIds.length > 0 ? selectedWorkspaceIds : [],
          document_ids: selectedDocumentIds.length > 0 ? selectedDocumentIds : [],
        })
      );

      console.log(`[Retry] Sending edited message with is_retry=true, user_id=${user?.id}, workspace_ids=${selectedWorkspaceIds.length}, document_ids=${selectedDocumentIds.length}: ${newContent}`);
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
  }, [isLoading, sessionId, editingContent, handleCancelEdit, handleRetry, selectedWorkspaceIds, selectedDocumentIds, connectWebSocket]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
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
      // Load messages for the selected session
      const history = await chatService.getSessionMessages(newSessionId);
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
                {/* Only show connection status when there's an active session */}
                {sessionId && (
                  <span className={`connection-status ${isConnected ? "connected" : "disconnected"}`}>
                    <i className={`pi ${isConnected ? "pi-check-circle" : "pi-exclamation-triangle"}`} />
                    {isConnected ? "Ready" : "Reconnecting..."}
                  </span>
                )}
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
                      // Assistant message actions - open markdown editor popup
                      <Button
                        icon="pi pi-pencil"
                        className="p-button-text p-button-sm edit-button"
                        onClick={() => handleOpenMarkdownEditor(idx, msg.content)}
                        tooltip={t("Chat.EditResponse")}
                        tooltipOptions={{ position: "top" }}
                        disabled={isLoading || markdownEditorVisible || !msg.id}
                      />
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
              <span className="typing-indicator">▊</span>
            </Card>
          </div>
        )}

        {isLoading && !streamingContent && (
          <div className="message assistant loading">
            <div className="message-avatar">
              <i className="pi pi-android" />
            </div>
            <Card className="message-content">
              <ProgressSpinner style={{ width: "24px", height: "24px" }} strokeWidth="4" />
              <span>Searching documents...</span>
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

          {/* Input Area */}
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
    </div>
  );
};

