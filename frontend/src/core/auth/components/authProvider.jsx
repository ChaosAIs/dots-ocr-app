import React, { useState, useEffect } from "react";
// import AuthService from "../authService"; // Temporarily disabled IAM SSO
// import TempAuthService from "../tempAuthService"; // Using temporary auth service
import authService from "../../../services/authService"; // Using real auth service

/**
 * Declare a Authentication context.
 * Note: Provider refered to authService.
 *
 * Creates a Context object. When React renders a component that subscribes
 * to this Context object it will read the current context value
 * from the closest matching Provider above it in the tree.
 *
 * Note: This looks like define an service interface.
 */
export const AuthContext = React.createContext({});

/**
 * Consume the interfaces/properties defined in Authentication context.
 * Try to expose the Authentication Context defined properties to the nested components.
 *
 * Note: This looks like define a reference of the service interface.
 * Note: AuthConsumer can be replaced by useContext(AuthContext).
 */
export const AuthConsumer = AuthContext.Consumer;

/**
 * Note: This looks as register implementation service instance (with detail business logic codes) into
 * the service interface.
 *
 * Reference: https://stackoverflow.com/questions/58197800/set-the-data-in-react-context-from-asynchronous-api-call
 */
export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user is already authenticated on mount
    const currentUser = authService.getUser();
    if (currentUser) {
      setUser(currentUser);
    }
    setLoading(false);
  }, []);

  const login = async (username, password) => {
    const result = await authService.login(username, password);
    if (result.success) {
      setUser(result.user);
    }
    return result;
  };

  const logout = async () => {
    await authService.logout();
    setUser(null);
  };

  const register = async (username, email, password, fullName) => {
    const result = await authService.register(username, email, password, fullName);
    if (result.success) {
      setUser(result.user);
    }
    return result;
  };

  const contextValue = {
    user,
    loading,
    isAuthenticated: () => authService.isAuthenticated(),
    isAdmin: () => authService.isAdmin(),
    getUser: () => authService.getUser(),
    login,
    logout,
    register,
    getAuthHeaders: () => authService.getAuthHeaders(),
  };

  // The Provider component accepts a value prop to be passed to consuming components
  // that are descendants of this Provider.
  // One Provider can be connected to many consumers.
  // Providers can be nested to override values deeper within the tree.
  // All consumers that are descendants of a Provider will re-render
  // whenever the Provider’s value prop changes.
  return (
    <AuthContext.Provider value={contextValue}>
      {/* "children" represents all nested children components. */}
      {!loading && children}
    </AuthContext.Provider>
  );
};
