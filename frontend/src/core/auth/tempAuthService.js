/**
 * Temporary Authentication Service
 * Bypasses IAM SSO for development/testing purposes
 * Provides basic user authentication and authorization checks
 */

import { Role } from "../enumertions/role";

export default class TempAuthService {
  constructor() {
    this.TEMP_USER_KEY = "temp_auth_user";
    this.TEMP_TOKEN_KEY = "temp_auth_token";
    this.initializeUser();
  }

  /**
   * Initialize a temporary user if not already authenticated
   */
  initializeUser() {
    const existingUser = this.getStoredUser();
    if (!existingUser) {
      // Create a default temporary user
      const tempUser = {
        id: "temp-user-001",
        userName: "tempuser",
        displayName: "Temporary User",
        email: "temp@example.com",
        roles: [Role.Administrator, Role.User],
        permissions: ["read", "write", "delete"],
        authProviderId: "TEMP",
        language: "en",
        accessToken: this.generateTempToken(),
      };
      this.storeUser(tempUser);
    }
  }

  /**
   * Generate a temporary JWT-like token
   */
  generateTempToken() {
    const header = btoa(JSON.stringify({ alg: "HS256", typ: "JWT" }));
    const payload = btoa(
      JSON.stringify({
        sub: "temp-user-001",
        name: "Temporary User",
        iat: Math.floor(Date.now() / 1000),
        exp: Math.floor(Date.now() / 1000) + 86400 * 7, // 7 days
      })
    );
    const signature = btoa("temp-signature");
    return `${header}.${payload}.${signature}`;
  }

  /**
   * Store user in session storage
   */
  storeUser(user) {
    sessionStorage.setItem(this.TEMP_USER_KEY, JSON.stringify(user));
    sessionStorage.setItem(this.TEMP_TOKEN_KEY, user.accessToken);
  }

  /**
   * Get stored user from session storage
   */
  getStoredUser() {
    const userJson = sessionStorage.getItem(this.TEMP_USER_KEY);
    return userJson ? JSON.parse(userJson) : null;
  }

  /**
   * Get current authenticated user
   */
  getUser() {
    return this.getStoredUser();
  }

  /**
   * Check if user is authenticated
   */
  isAuthenticated() {
    const user = this.getStoredUser();
    return !!user && !!user.accessToken;
  }

  /**
   * Sign in with temporary credentials
   * @param {string} username - Username (optional, uses default if not provided)
   * @param {string} password - Password (optional, not validated)
   */
  signin(username = "tempuser") {
    return Promise.resolve({
      id: "temp-user-001",
      userName: username,
      displayName: username,
      email: `${username}@example.com`,
      roles: [Role.Administrator, Role.User],
      permissions: ["read", "write", "delete"],
      authProviderId: "TEMP",
      language: "en",
      accessToken: this.generateTempToken(),
    });
  }

  /**
   * Sign in redirect (no-op for temporary auth)
   */
  signinRedirect() {
    console.log("Temporary auth: signinRedirect called (no-op)");
    return Promise.resolve();
  }

  /**
   * Sign in redirect callback (no-op for temporary auth)
   */
  signinRedirectCallback() {
    console.log("Temporary auth: signinRedirectCallback called (no-op)");
    return Promise.resolve();
  }

  /**
   * Sign out
   */
  signout() {
    // Clear temporary auth tokens
    sessionStorage.removeItem(this.TEMP_USER_KEY);
    sessionStorage.removeItem(this.TEMP_TOKEN_KEY);

    // Clear real auth tokens (from authService)
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user_data');

    // Clear any other auth-related items
    localStorage.removeItem('redirectUri');

    console.log('All authentication data cleared (temp and real auth)');
    window.location.replace("/");
  }

  /**
   * Logout (alias for signout)
   */
  logout() {
    this.signout();
  }

  /**
   * Sign out redirect callback (no-op for temporary auth)
   */
  signoutRedirectCallback() {
    this.signout();
  }

  /**
   * Navigate to home
   */
  navigateToHome() {
    window.location.replace("/");
  }

  /**
   * Clear auth state
   */
  clearAuthState() {
    // Clear temporary auth tokens
    sessionStorage.removeItem(this.TEMP_USER_KEY);
    sessionStorage.removeItem(this.TEMP_TOKEN_KEY);

    // Clear real auth tokens (from authService)
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
    localStorage.removeItem('user_data');

    console.log('Auth state cleared');
  }

  /**
   * Debug auth state
   */
  debugAuthState() {
    const user = this.getStoredUser();
    console.log("🔍 Temporary Auth State:", {
      isAuthenticated: this.isAuthenticated(),
      user: user,
    });
  }

  /**
   * Check if user has a specific role
   */
  hasRole(role) {
    const user = this.getStoredUser();
    if (!user || !user.roles) return false;
    return user.roles.includes(role);
  }

  /**
   * Check if user has a specific permission
   */
  hasPermission(permission) {
    const user = this.getStoredUser();
    if (!user || !user.permissions) return false;
    return user.permissions.includes(permission);
  }

  /**
   * Create signin request (no-op for temporary auth)
   */
  createSigninRequest() {
    return Promise.resolve({});
  }

  /**
   * Silent signin (no-op for temporary auth)
   */
  signinSilent() {
    console.log("Temporary auth: signinSilent called (no-op)");
    return Promise.resolve();
  }

  /**
   * Silent signin callback (no-op for temporary auth)
   */
  signinSilentCallback() {
    console.log("Temporary auth: signinSilentCallback called (no-op)");
  }
}

