//import * as React from "react";
import { Route, Routes } from "react-router-dom";
import { Callback } from "../core/auth/components/callback";
import { Logout } from "../core/auth/components/logout";
import { LogoutCallback } from "../core/auth/components/logoutCallback";
import { PrivateRoute } from "./privateRoute";
import { SilentRenew } from "../core/auth/components/silentRenew";
import { UserProfile } from "../pages/userprofile";
import { Role } from "../core/enumertions/role";
import { UserManagement } from "../pages/usermanagement";
import { NotFound } from "../pages/notfound";
import { Home } from "../pages/home";
import { Chat } from "../pages/chat";
import { Login } from "../components/auth/Login";
import { Register } from "../components/auth/Register";

/** Global routes definition.
 *  Link specified url to specified component.
 *  Note: Here no need to add APP_CONFIG.basePath since BrowserRouter "baseName" already be assigned by APP_CONFIG.basePath.
 */
export const AppRoutes = () => (
  <Routes>
    <Route path="/auth-callback" element={<Callback />} />
    <Route path="/silent-auth-callback" element={<SilentRenew />} />
    <Route path="/logout" element={<Logout />} />
    <Route path="/logout/callback" element={<LogoutCallback />} />
    <Route path="/userprofile" element={<PrivateRoute component={UserProfile} />} />
    <Route path="/usermanagement" element={<PrivateRoute component={UserManagement} roles={[Role.PPAdministrator]} />} />

    {/* Authentication routes */}
    <Route path="/login" element={<Login />} />
    <Route path="/register" element={<Register />} />

    {/* Protected routes - require authentication */}
    <Route path="/home" element={<PrivateRoute component={Home} />} />
    <Route path="/chat" element={<PrivateRoute component={Chat} />} />
    <Route path="/" element={<PrivateRoute component={Home} />} />
    {/* <Route path="/sample" element={<Sample />} /> */}
    <Route path="*" element={<NotFound />} />
  </Routes>
);
