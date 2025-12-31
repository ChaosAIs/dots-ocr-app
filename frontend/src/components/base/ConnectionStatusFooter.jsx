import React from "react";
import { useConnectionStatus } from "../../contexts/ConnectionStatusContext";
import { useTranslation } from "react-i18next";
import "./ConnectionStatusFooter.scss";

/**
 * ConnectionStatusFooter - Displays connection status in the page footer
 *
 * Shows the current connection status (connected, disconnected, reconnecting)
 * in a minimal footer bar at the bottom of the page.
 */
// Status color mapping
const STATUS_COLORS = {
  connected: "#22c55e",    // green
  reconnecting: "#facc15", // yellow
  disconnected: "#f87171", // red
  checking: "#facc15",     // yellow
  idle: "#a0a0b0",         // gray
};

export const ConnectionStatusFooter = () => {
  const { t } = useTranslation();
  const { getConnectionStatus } = useConnectionStatus();

  const { status } = getConnectionStatus();

  // Get color based on status
  const getColor = () => STATUS_COLORS[status] || STATUS_COLORS.idle;

  // Get appropriate icon based on status
  const getIcon = () => {
    switch (status) {
      case "connected":
        return "pi pi-check-circle";
      case "reconnecting":
        return "pi pi-spin pi-spinner";
      case "disconnected":
        return "pi pi-exclamation-triangle";
      default:
        return "pi pi-circle-on";
    }
  };

  // Get translated message
  const getTranslatedMessage = () => {
    switch (status) {
      case "connected":
        return t("Connection.Connected", "Connected");
      case "reconnecting":
        return t("Connection.Reconnecting", "Reconnecting...");
      case "disconnected":
        return t("Connection.Disconnected", "Disconnected");
      default:
        return t("Connection.Ready", "Ready");
    }
  };

  const color = getColor();
  const currentYear = new Date().getFullYear();

  return (
    <footer className="connection-status-footer">
      <div className="footer-content">
        <div className="footer-left">
          <span className="copyright">&copy; {currentYear} Private Knowledge AI Assistant</span>
          <span className="version-separator">|</span>
          <span className="app-version">v1.0.0</span>
        </div>
        <div className="connection-indicator">
          <span style={{ color, fontSize: "0.85rem" }}>&#x25CF;</span>
          <span style={{ color }}>{getTranslatedMessage()}</span>
        </div>
      </div>
    </footer>
  );
};

export default ConnectionStatusFooter;
