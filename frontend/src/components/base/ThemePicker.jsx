import React, { useState, useRef, useContext, useEffect } from "react";
import { Button } from "primereact/button";
import { OverlayPanel } from "primereact/overlaypanel";
import { AuthContext } from "../../core/auth/components/authProvider";
import authService from "../../services/authService";
import APP_CONFIG from "../../core/config/appConfig";
import "./ThemePicker.scss";

// Available PrimeReact themes
const THEMES = [
  { name: "Saga Blue", value: "saga-blue", color: "#2196F3" },
  { name: "Saga Green", value: "saga-green", color: "#4CAF50" },
  { name: "Saga Orange", value: "saga-orange", color: "#FFC107" },
  { name: "Saga Purple", value: "saga-purple", color: "#9C27B0" },
  { name: "Vela Blue", value: "vela-blue", color: "#64B5F6", dark: true },
  { name: "Vela Green", value: "vela-green", color: "#81C784", dark: true },
  { name: "Vela Orange", value: "vela-orange", color: "#FFD54F", dark: true },
  { name: "Vela Purple", value: "vela-purple", color: "#BA68C8", dark: true },
  { name: "Arya Blue", value: "arya-blue", color: "#42A5F5", dark: true },
  { name: "Arya Green", value: "arya-green", color: "#66BB6A", dark: true },
  { name: "Arya Orange", value: "arya-orange", color: "#FFCA28", dark: true },
  { name: "Arya Purple", value: "arya-purple", color: "#AB47BC", dark: true },
  { name: "Lara Light Blue", value: "lara-light-blue", color: "#3B82F6" },
  { name: "Lara Light Indigo", value: "lara-light-indigo", color: "#6366F1" },
  { name: "Lara Light Purple", value: "lara-light-purple", color: "#8B5CF6" },
  { name: "Lara Light Teal", value: "lara-light-teal", color: "#14B8A6" },
  { name: "Lara Dark Blue", value: "lara-dark-blue", color: "#60A5FA", dark: true },
  { name: "Lara Dark Indigo", value: "lara-dark-indigo", color: "#818CF8", dark: true },
  { name: "Lara Dark Purple", value: "lara-dark-purple", color: "#A78BFA", dark: true },
  { name: "Lara Dark Teal", value: "lara-dark-teal", color: "#2DD4BF", dark: true },
];

const DEFAULT_THEME = "saga-blue";

// Get the base path for theme URLs
const getThemeBasePath = () => {
  // Use APP_CONFIG.basePath if available, otherwise empty string
  const basePath = APP_CONFIG?.basePath || "";
  // Ensure no double slashes
  return basePath.endsWith("/") ? basePath.slice(0, -1) : basePath;
};

// Function to change theme dynamically
const changeTheme = (newTheme) => {
  const linkElement = document.getElementById("theme-link");
  const basePath = getThemeBasePath();
  const themeUrl = `${basePath}/themes/${newTheme}/theme.css`;

  if (linkElement) {
    // Update existing link
    linkElement.href = themeUrl;
  } else {
    // Create new link element if it doesn't exist
    const newLink = document.createElement("link");
    newLink.id = "theme-link";
    newLink.rel = "stylesheet";
    newLink.href = themeUrl;
    document.head.appendChild(newLink);
  }

  // Store in localStorage as fallback
  localStorage.setItem("app-theme", newTheme);
};

// Function to get current theme
export const getCurrentTheme = () => {
  return localStorage.getItem("app-theme") || DEFAULT_THEME;
};

// Function to initialize theme on app load
export const initializeTheme = (theme) => {
  const themeToUse = theme || getCurrentTheme();
  changeTheme(themeToUse);
  return themeToUse;
};

export const ThemePicker = () => {
  const overlayRef = useRef(null);
  const authContext = useContext(AuthContext);
  const user = authContext?.getUser?.();
  const [currentTheme, setCurrentTheme] = useState(getCurrentTheme());
  const [loading, setLoading] = useState(false);

  // Load theme preference on mount
  useEffect(() => {
    const loadThemePreference = async () => {
      if (user) {
        try {
          const result = await authService.getThemePreference();
          if (result.success && result.theme) {
            setCurrentTheme(result.theme);
            changeTheme(result.theme);
          }
        } catch (error) {
          console.error("Error loading theme preference:", error);
        }
      }
    };

    loadThemePreference();
  }, [user]);

  const handleThemeChange = async (theme) => {
    setLoading(true);
    setCurrentTheme(theme.value);
    changeTheme(theme.value);

    // Save to backend if user is logged in
    if (user) {
      try {
        await authService.updateThemePreference(theme.value);
      } catch (error) {
        console.error("Error saving theme preference:", error);
      }
    }

    setLoading(false);
    overlayRef.current?.hide();
  };

  const currentThemeObj = THEMES.find((t) => t.value === currentTheme) || THEMES[0];

  return (
    <div className="theme-picker">
      <Button
        icon="pi pi-palette"
        className="p-button-text p-button-rounded theme-picker-btn"
        onClick={(e) => overlayRef.current?.toggle(e)}
        tooltip="Change Theme"
        tooltipOptions={{ position: "bottom" }}
        disabled={loading}
      />

      <OverlayPanel ref={overlayRef} className="theme-picker-panel">
        <div className="theme-picker-header">
          <i className="pi pi-palette"></i>
          <span>Select Theme</span>
        </div>

        <div className="theme-categories">
          {/* Light Themes */}
          <div className="theme-category">
            <div className="category-label">Light Themes</div>
            <div className="theme-grid">
              {THEMES.filter((t) => !t.dark).map((theme) => (
                <div
                  key={theme.value}
                  className={`theme-option ${currentTheme === theme.value ? "selected" : ""}`}
                  onClick={() => handleThemeChange(theme)}
                  title={theme.name}
                >
                  <div
                    className="theme-color"
                    style={{ backgroundColor: theme.color }}
                  >
                    {currentTheme === theme.value && (
                      <i className="pi pi-check"></i>
                    )}
                  </div>
                  <span className="theme-name">{theme.name}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Dark Themes */}
          <div className="theme-category">
            <div className="category-label">Dark Themes</div>
            <div className="theme-grid">
              {THEMES.filter((t) => t.dark).map((theme) => (
                <div
                  key={theme.value}
                  className={`theme-option ${currentTheme === theme.value ? "selected" : ""}`}
                  onClick={() => handleThemeChange(theme)}
                  title={theme.name}
                >
                  <div
                    className="theme-color dark"
                    style={{ backgroundColor: theme.color }}
                  >
                    {currentTheme === theme.value && (
                      <i className="pi pi-check"></i>
                    )}
                  </div>
                  <span className="theme-name">{theme.name}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="theme-picker-footer">
          <span className="current-theme">
            Current: <strong>{currentThemeObj.name}</strong>
          </span>
        </div>
      </OverlayPanel>
    </div>
  );
};

export default ThemePicker;
