import React from "react";
// Theme CSS is loaded dynamically via index.html and ThemePicker
import "primereact/resources/primereact.min.css";
import "primeicons/primeicons.css";
import "./styling/primeicons.css";
import "primeflex/primeflex.css";
import "./styling/App.scss";
import "./styling/colors.css";
import "./App.css";

import { PrimeReactProvider } from "primereact/api";
import MessageBox from "./core/message/components/messageBox";
import MessageToast from "./core/message/components/messageToast";
import AppProgress from "./core/loading/components/appProgress";
import { ConfirmDialog } from "primereact/confirmdialog";

import { AuthProvider } from "./core/auth/components/authProvider";
import { WorkspaceProvider } from "./contexts/WorkspaceContext";
import { ConnectionStatusProvider } from "./contexts/ConnectionStatusContext";
import { ConnectionStatusFooter } from "./components/base/ConnectionStatusFooter";
import { BrowserRouter, useLocation } from "react-router-dom";
import { AppRoutes } from "./routes/routes";
import { NavBar } from "./components/base/navbar";
import APP_CONFIG from "./core/config/appConfig";
import MessageDialog from "./core/message/components/messageDialog";
import { initializeTheme } from "./components/base/ThemePicker";

// Import debug auth utilities to make them globally available
import "./core/auth/debugAuth";

// Initialize theme from localStorage on app load
const savedTheme = localStorage.getItem("app-theme");
if (savedTheme) {
  initializeTheme(savedTheme);
}

// Component to conditionally render navbar based on current route
const ConditionalNavBar = () => {
  const location = useLocation();
  // Check if current path is login or register (without basePath prefix since BrowserRouter handles it)
  const hideNavbarRoutes = ['/login', '/register'];
  const shouldHideNavbar = hideNavbarRoutes.some(route => location.pathname === route || location.pathname.endsWith(route));

  return shouldHideNavbar ? null : <NavBar />;
};

function App() {

  return (
    <PrimeReactProvider>
      <div className="app">
        <MessageBox />
        <MessageToast />
        <MessageDialog />
        <ConfirmDialog />
        <AppProgress />
        <AuthProvider>
        {/*  When authentication process updated, nexsted route
                and route's refered component will got re-render automatically.
                Note: Any nested components inside "AuthProvider",
                it can call useContext(AuthContext) to get reference of authService.
            */}
        <ConnectionStatusProvider>
          {/* ConnectionStatusProvider tracks global connection status */}
          <WorkspaceProvider>
            {/* WorkspaceProvider provides centralized workspace state management */}
            {/* As for "basename", it is the base URL for all locations.
                  If your app is served from a sub-directory on your server,
                  you'll want to set this to the sub-directory.
                  A properly formatted basename should have a leading slash,
                  but no trailing slash.
                */}
            <BrowserRouter basename={`${APP_CONFIG.basePath.length > 0 ? APP_CONFIG.basePath : "/"}`}>
              <ConditionalNavBar />
              <div className="body-area">
                <AppRoutes />
              </div>
              <ConnectionStatusFooter />
            </BrowserRouter>
          </WorkspaceProvider>
        </ConnectionStatusProvider>
      </AuthProvider>
      </div>
    </PrimeReactProvider>
  );
}

export default App;
