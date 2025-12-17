import { useState, useRef, useEffect, useCallback } from "react";
import { InputTextarea } from "primereact/inputtextarea";
import { Button } from "primereact/button";
import { Card } from "primereact/card";
import { ProgressSpinner } from "primereact/progressspinner";
import { Toast } from "primereact/toast";
import ReactMarkdown from "react-markdown";
import chatService from "../../services/chatService";
import { ChatHistory } from "./ChatHistory";
import "./AgenticChatBot.scss";

// Get WebSocket URL from environment or use default
const WS_BASE_URL = process.env.REACT_APP_WS_URL || "ws://localhost:8080";

export const AgenticChatBot = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [isInitializing, setIsInitializing] = useState(true);

  const wsRef = useRef(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const toast = useRef(null);
  const streamingContentRef = useRef("");
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;
  const chatHistoryRef = useRef(null);
  const messageCountRef = useRef(0);

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

  const connectWebSocket = useCallback(() => {
    // Don't connect if no session ID or already connected
    if (!sessionId || wsRef.current?.readyState === WebSocket.OPEN) {
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

    console.log(`Connecting to WebSocket with session ID: ${sessionId}`);
    const ws = new WebSocket(`${WS_BASE_URL}/api/chat/ws/chat/${sessionId}`);

    ws.onopen = () => {
      setIsConnected(true);
      reconnectAttemptsRef.current = 0; // Reset counter on successful connection
      console.log("WebSocket connected");
    };

    ws.onclose = (event) => {
      setIsConnected(false);
      console.log("WebSocket disconnected", event.code, event.reason);

      // Don't reconnect on authentication/authorization errors (403, 401)
      // or if the session was explicitly closed (1000)
      if (event.code === 1008 || event.code === 1000) {
        console.log("WebSocket closed normally or due to policy violation - not reconnecting");
        return;
      }

      // Increment reconnection attempts
      reconnectAttemptsRef.current += 1;

      // Attempt to reconnect after 3 seconds if we haven't exceeded max attempts
      if (reconnectAttemptsRef.current < maxReconnectAttempts) {
        console.log(`Scheduling reconnection attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts}`);
        setTimeout(connectWebSocket, 3000);
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      toast.current?.show({
        severity: "error",
        summary: "Connection Error",
        detail: "Failed to connect to chat server",
        life: 3000,
      });
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        if (data.type === "token") {
          streamingContentRef.current += data.content;
          setStreamingContent(streamingContentRef.current);
        } else if (data.type === "end") {
          // Complete the message with the accumulated content
          const finalContent = streamingContentRef.current;
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
          if (chatHistoryRef.current?.loadSessions) {
            setTimeout(() => {
              chatHistoryRef.current.loadSessions();
            }, 500); // Small delay to ensure backend has updated the title
          }
        } else if (data.type === "error") {
          toast.current?.show({
            severity: "error",
            summary: "Error",
            detail: data.message,
            life: 5000,
          });
          streamingContentRef.current = "";
          setStreamingContent("");
          setIsLoading(false);
        }
      } catch (e) {
        console.error("Error parsing message:", e);
      }
    };

    wsRef.current = ws;
  }, [sessionId]);

  useEffect(() => {
    if (sessionId) {
      connectWebSocket();
    }
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWebSocket, sessionId]);

  // Load chat history when session is created
  useEffect(() => {
    const loadHistory = async () => {
      if (!sessionId) return;

      try {
        const history = await chatService.getSessionMessages(sessionId);
        if (history && history.length > 0) {
          setMessages(
            history.map((msg) => ({
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

  const sendMessage = useCallback(async () => {
    if (!inputValue.trim() || isLoading) return;

    // Create session if it doesn't exist (lazy creation)
    let currentSessionId = sessionId;
    if (!currentSessionId) {
      try {
        const session = await chatService.createSession();
        currentSessionId = session.id;
        setSessionId(session.id);
        messageCountRef.current = 0;
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
      content: inputValue.trim(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);
    streamingContentRef.current = "";
    setStreamingContent("");

    // Send to WebSocket (history is now managed by backend)
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          message: userMessage.content,
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
  }, [inputValue, isLoading, sessionId, connectWebSocket]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    // Clear current session and messages
    // New session will be created when user sends first message
    setSessionId(null);
    setMessages([]);
    streamingContentRef.current = "";
    setStreamingContent("");

    // Reset reconnection counter and message count
    reconnectAttemptsRef.current = 0;
    messageCountRef.current = 0;

    // Close WebSocket
    if (wsRef.current) {
      wsRef.current.close();
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
      // Load messages for the selected session
      const history = await chatService.getSessionMessages(newSessionId);
      const loadedMessages = history.map((msg) => ({
        role: msg.role,
        content: msg.content,
      }));
      setMessages(loadedMessages);

      // Update session ID and reconnect WebSocket
      setSessionId(newSessionId);

      // Reset reconnection counter and set message count
      reconnectAttemptsRef.current = 0;
      messageCountRef.current = loadedMessages.length;

      if (wsRef.current) {
        wsRef.current.close();
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
                {/* Only show connection status when there's an active session */}
                {sessionId && (
                  <span className={`connection-status ${isConnected ? "connected" : "disconnected"}`}>
                    <i className={`pi ${isConnected ? "pi-check-circle" : "pi-exclamation-triangle"}`} />
                    {isConnected ? "Ready" : "Reconnecting..."}
                  </span>
                )}
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
          <div key={idx} className={`message ${msg.role}`}>
            <div className="message-avatar">
              <i className={`pi ${msg.role === "user" ? "pi-user" : "pi-android"}`} />
            </div>
            <Card className="message-content">
              {msg.role === "assistant" ? (
                <ReactMarkdown>{msg.content}</ReactMarkdown>
              ) : (
                <p>{msg.content}</p>
              )}
            </Card>
          </div>
        ))}

        {streamingContent && (
          <div className="message assistant">
            <div className="message-avatar">
              <i className="pi pi-android" />
            </div>
            <Card className="message-content streaming">
              <ReactMarkdown>{streamingContent}</ReactMarkdown>
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
    </div>
  );
};

