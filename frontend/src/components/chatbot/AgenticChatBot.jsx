import { useState, useRef, useEffect, useCallback } from "react";
import { InputTextarea } from "primereact/inputtextarea";
import { Button } from "primereact/button";
import { Card } from "primereact/card";
import { ProgressSpinner } from "primereact/progressspinner";
import { Toast } from "primereact/toast";
import ReactMarkdown from "react-markdown";
import "./AgenticChatBot.scss";

// Get WebSocket URL from environment or use default
const WS_BASE_URL = process.env.REACT_APP_WS_URL || "ws://localhost:8080";

export const AgenticChatBot = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");

  const wsRef = useRef(null);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const toast = useRef(null);
  const streamingContentRef = useRef("");

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, streamingContent, scrollToBottom]);

  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    const ws = new WebSocket(`${WS_BASE_URL}/api/chat/ws/chat`);

    ws.onopen = () => {
      setIsConnected(true);
      console.log("WebSocket connected");
    };

    ws.onclose = () => {
      setIsConnected(false);
      console.log("WebSocket disconnected");
      // Attempt to reconnect after 3 seconds
      setTimeout(connectWebSocket, 3000);
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
  }, []);

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connectWebSocket]);

  const sendMessage = useCallback(() => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      role: "user",
      content: inputValue.trim(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);
    streamingContentRef.current = "";
    setStreamingContent("");

    // Send to WebSocket
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({
          message: userMessage.content,
          history: messages.map((m) => ({
            role: m.role,
            content: m.content,
          })),
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
  }, [inputValue, isLoading, messages, connectWebSocket]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    streamingContentRef.current = "";
    setStreamingContent("");
  };

  return (
    <div className="chatbot-container">
      <Toast ref={toast} />

      {/* Header */}
      <div className="chatbot-header">
        <div className="header-title">
          <i className="pi pi-comments" />
          <span>Document Chat Assistant</span>
        </div>
        <div className="header-actions">
          <span className={`connection-status ${isConnected ? "connected" : "disconnected"}`}>
            <i className={`pi ${isConnected ? "pi-check-circle" : "pi-times-circle"}`} />
            {isConnected ? "Connected" : "Disconnected"}
          </span>
          <Button
            icon="pi pi-trash"
            className="p-button-text p-button-secondary"
            onClick={clearChat}
            tooltip="Clear chat"
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
          disabled={isLoading || !isConnected}
        />
        <Button
          icon={isLoading ? "pi pi-spin pi-spinner" : "pi pi-send"}
          onClick={sendMessage}
          disabled={!inputValue.trim() || isLoading || !isConnected}
          className="send-button"
        />
      </div>
    </div>
  );
};

