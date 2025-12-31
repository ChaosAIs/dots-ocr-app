import React, { createContext, useContext, useState, useCallback, useEffect, useRef } from "react";
import APP_CONFIG from "../core/config/appConfig";

/**
 * ConnectionStatusContext - Global connection status management
 *
 * This context provides a centralized way to track and display
 * connection status across the application (e.g., backend server connectivity).
 */

const ConnectionStatusContext = createContext(null);

export const ConnectionStatusProvider = ({ children }) => {
  // Server connection state
  const [serverConnected, setServerConnected] = useState(false);
  const [checking, setChecking] = useState(true);
  const checkIntervalRef = useRef(null);

  // Chat WebSocket states (for future use)
  const [chatConnected, setChatConnected] = useState(false);
  const [chatSessionActive, setChatSessionActive] = useState(false);
  const [reconnecting, setReconnecting] = useState(false);

  // Check server connectivity
  const checkServerConnection = useCallback(async () => {
    try {
      const apiDomain = APP_CONFIG.apiDomain || "http://localhost:8080";
      const response = await fetch(`${apiDomain}/health`, {
        method: "GET",
        headers: { "Content-Type": "application/json" },
      });
      setServerConnected(response.ok);
    } catch (error) {
      setServerConnected(false);
    } finally {
      setChecking(false);
    }
  }, []);

  // Check server connection on mount and periodically
  useEffect(() => {
    checkServerConnection();

    // Check every 30 seconds
    checkIntervalRef.current = setInterval(checkServerConnection, 30000);

    return () => {
      if (checkIntervalRef.current) {
        clearInterval(checkIntervalRef.current);
      }
    };
  }, [checkServerConnection]);

  // Update chat connection status
  const updateChatStatus = useCallback((connected, sessionActive = false, isReconnecting = false) => {
    setChatConnected(connected);
    setChatSessionActive(sessionActive);
    setReconnecting(isReconnecting);
  }, []);

  // Get overall connection status (prioritizes server connection)
  const getConnectionStatus = useCallback(() => {
    if (checking) {
      return { status: "checking", message: "Checking..." };
    }
    if (!serverConnected) {
      return { status: "disconnected", message: "Disconnected" };
    }
    // Server is connected
    if (chatSessionActive) {
      if (reconnecting) {
        return { status: "reconnecting", message: "Reconnecting..." };
      }
      if (chatConnected) {
        return { status: "connected", message: "Connected" };
      }
      return { status: "disconnected", message: "Disconnected" };
    }
    // Server connected, no active chat session
    return { status: "connected", message: "Ready" };
  }, [serverConnected, checking, chatConnected, chatSessionActive, reconnecting]);

  const value = {
    serverConnected,
    chatConnected,
    chatSessionActive,
    reconnecting,
    updateChatStatus,
    getConnectionStatus,
    checkServerConnection,
  };

  return (
    <ConnectionStatusContext.Provider value={value}>
      {children}
    </ConnectionStatusContext.Provider>
  );
};

export const useConnectionStatus = () => {
  const context = useContext(ConnectionStatusContext);
  if (!context) {
    throw new Error("useConnectionStatus must be used within a ConnectionStatusProvider");
  }
  return context;
};

export default ConnectionStatusContext;
