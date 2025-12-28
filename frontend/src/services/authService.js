/**
 * Authentication service for user login, registration, and token management.
 * Replaces tempAuthService with real JWT-based authentication.
 */

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8080';
const AUTH_API_URL = `${API_BASE_URL}/api/auth`;

// Storage keys
const TOKEN_KEY = 'access_token';
const REFRESH_TOKEN_KEY = 'refresh_token';
const USER_KEY = 'user_data';

class AuthService {
    /**
     * Register a new user
     */
    async register(username, email, password, fullName = null) {
        try {
            const response = await fetch(`${AUTH_API_URL}/register`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    username,
                    email,
                    password,
                    full_name: fullName,
                }),
            });

            const data = await response.json();

            if (!response.ok) {
                // Handle validation errors (array of error objects)
                if (Array.isArray(data.detail)) {
                    const errorMessages = data.detail.map(err => {
                        const field = err.loc && err.loc.length > 1 ? err.loc[err.loc.length - 1] : 'field';
                        return `${field}: ${err.msg}`;
                    }).join(', ');
                    return { success: false, error: errorMessages };
                }
                // Handle string error messages
                return { success: false, error: data.detail || 'Registration failed' };
            }

            // Store tokens and user data
            this.storeAuthData(data.access_token, data.refresh_token, data.user);

            return { success: true, user: data.user };
        } catch (error) {
            console.error('Registration error:', error);
            return { success: false, error: error.message || 'Network error' };
        }
    }

    /**
     * Login user
     */
    async login(username, password) {
        try {
            const response = await fetch(`${AUTH_API_URL}/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    username,
                    password,
                }),
            });

            const data = await response.json();

            if (!response.ok) {
                // Handle validation errors (array of error objects)
                if (Array.isArray(data.detail)) {
                    const errorMessages = data.detail.map(err => {
                        const field = err.loc && err.loc.length > 1 ? err.loc[err.loc.length - 1] : 'field';
                        return `${field}: ${err.msg}`;
                    }).join(', ');
                    return { success: false, error: errorMessages };
                }
                // Handle string error messages
                return { success: false, error: data.detail || 'Login failed' };
            }

            // Store tokens and user data
            this.storeAuthData(data.access_token, data.refresh_token, data.user);

            return { success: true, user: data.user };
        } catch (error) {
            console.error('Login error:', error);
            return { success: false, error: error.message || 'Network error' };
        }
    }

    /**
     * Logout user
     */
    async logout() {
        const refreshToken = localStorage.getItem(REFRESH_TOKEN_KEY);

        if (refreshToken) {
            try {
                await fetch(`${AUTH_API_URL}/logout`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        ...this.getAuthHeaders(),
                    },
                    body: JSON.stringify({
                        refresh_token: refreshToken,
                    }),
                });
            } catch (error) {
                console.error('Logout error:', error);
            }
        }

        // Clear local storage
        this.clearAuthData();
    }

    /**
     * Refresh access token
     */
    async refreshToken() {
        const refreshToken = localStorage.getItem(REFRESH_TOKEN_KEY);

        if (!refreshToken) {
            return { success: false, error: 'No refresh token' };
        }

        try {
            const response = await fetch(`${AUTH_API_URL}/refresh`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    refresh_token: refreshToken,
                }),
            });

            const data = await response.json();

            if (!response.ok) {
                this.clearAuthData();
                // Handle validation errors (array of error objects)
                if (Array.isArray(data.detail)) {
                    const errorMessages = data.detail.map(err => {
                        const field = err.loc && err.loc.length > 1 ? err.loc[err.loc.length - 1] : 'field';
                        return `${field}: ${err.msg}`;
                    }).join(', ');
                    return { success: false, error: errorMessages };
                }
                // Handle string error messages
                return { success: false, error: data.detail || 'Token refresh failed' };
            }

            // Store new tokens
            this.storeAuthData(data.access_token, data.refresh_token, data.user);

            return { success: true, user: data.user };
        } catch (error) {
            console.error('Token refresh error:', error);
            this.clearAuthData();
            return { success: false, error: error.message || 'Network error' };
        }
    }

    /**
     * Get current user
     */
    getUser() {
        const userData = localStorage.getItem(USER_KEY);
        if (userData) {
            try {
                return JSON.parse(userData);
            } catch (error) {
                console.error('Error parsing user data:', error);
                return null;
            }
        }
        return null;
    }

    /**
     * Check if user is authenticated
     */
    isAuthenticated() {
        const token = localStorage.getItem(TOKEN_KEY);
        const user = this.getUser();
        return !!(token && user);
    }

    /**
     * Check if user is admin
     */
    isAdmin() {
        const user = this.getUser();
        return user && user.role === 'admin';
    }

    /**
     * Get authorization headers
     */
    getAuthHeaders() {
        const token = localStorage.getItem(TOKEN_KEY);
        if (token) {
            return {
                'Authorization': `Bearer ${token}`,
            };
        }
        return {};
    }

    /**
     * Store authentication data
     */
    storeAuthData(accessToken, refreshToken, user) {
        localStorage.setItem(TOKEN_KEY, accessToken);
        localStorage.setItem(REFRESH_TOKEN_KEY, refreshToken);
        localStorage.setItem(USER_KEY, JSON.stringify(user));
    }

    /**
     * Clear authentication data
     * Clears both real auth tokens and temporary auth tokens
     */
    clearAuthData() {
        // Clear real authentication tokens
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(REFRESH_TOKEN_KEY);
        localStorage.removeItem(USER_KEY);

        // Clear temporary auth tokens (from tempAuthService)
        sessionStorage.removeItem('temp_auth_user');
        sessionStorage.removeItem('temp_auth_token');

        // Clear any other auth-related items
        localStorage.removeItem('redirectUri');

        console.log('All authentication data cleared from localStorage and sessionStorage');
    }

    /**
     * Change password
     */
    async changePassword(oldPassword, newPassword) {
        try {
            const response = await fetch(`${AUTH_API_URL}/change-password`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    ...this.getAuthHeaders(),
                },
                body: JSON.stringify({
                    old_password: oldPassword,
                    new_password: newPassword,
                }),
            });

            const data = await response.json();

            if (!response.ok) {
                // Handle validation errors (array of error objects)
                if (Array.isArray(data.detail)) {
                    const errorMessages = data.detail.map(err => {
                        const field = err.loc && err.loc.length > 1 ? err.loc[err.loc.length - 1] : 'field';
                        return `${field}: ${err.msg}`;
                    }).join(', ');
                    return { success: false, error: errorMessages };
                }
                // Handle string error messages
                return { success: false, error: data.detail || 'Password change failed' };
            }

            return { success: true, message: data.message };
        } catch (error) {
            console.error('Password change error:', error);
            return { success: false, error: error.message || 'Network error' };
        }
    }

    /**
     * Get current user info from server
     */
    async getCurrentUser() {
        try {
            const response = await fetch(`${AUTH_API_URL}/me`, {
                method: 'GET',
                headers: {
                    ...this.getAuthHeaders(),
                },
            });

            if (!response.ok) {
                return { success: false, error: 'Failed to get user info' };
            }

            const user = await response.json();

            // Update stored user data
            localStorage.setItem(USER_KEY, JSON.stringify(user));

            return { success: true, user };
        } catch (error) {
            console.error('Get user error:', error);
            return { success: false, error: error.message || 'Network error' };
        }
    }

    // ===== User Preferences Methods =====

    /**
     * Get all user preferences
     */
    async getPreferences() {
        try {
            const response = await fetch(`${AUTH_API_URL}/preferences`, {
                method: 'GET',
                headers: {
                    ...this.getAuthHeaders(),
                },
            });

            if (!response.ok) {
                return { success: false, error: 'Failed to get preferences' };
            }

            const data = await response.json();
            return { success: true, preferences: data.preferences };
        } catch (error) {
            console.error('Get preferences error:', error);
            return { success: false, error: error.message || 'Network error' };
        }
    }

    /**
     * Update all user preferences
     */
    async updatePreferences(preferences) {
        try {
            const response = await fetch(`${AUTH_API_URL}/preferences`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                    ...this.getAuthHeaders(),
                },
                body: JSON.stringify({ preferences }),
            });

            if (!response.ok) {
                return { success: false, error: 'Failed to update preferences' };
            }

            const data = await response.json();
            return { success: true, preferences: data.preferences };
        } catch (error) {
            console.error('Update preferences error:', error);
            return { success: false, error: error.message || 'Network error' };
        }
    }

    /**
     * Get chat-specific preferences
     */
    async getChatPreferences() {
        try {
            const response = await fetch(`${AUTH_API_URL}/preferences/chat`, {
                method: 'GET',
                headers: {
                    ...this.getAuthHeaders(),
                },
            });

            if (!response.ok) {
                return { success: false, error: 'Failed to get chat preferences' };
            }

            const data = await response.json();
            return { success: true, chat: data.chat };
        } catch (error) {
            console.error('Get chat preferences error:', error);
            return { success: false, error: error.message || 'Network error' };
        }
    }

    /**
     * Update chat-specific preferences (workspace and document selections)
     */
    async updateChatPreferences(selectedWorkspaceIds, selectedDocumentIds = []) {
        try {
            const response = await fetch(`${AUTH_API_URL}/preferences/chat`, {
                method: 'PATCH',
                headers: {
                    'Content-Type': 'application/json',
                    ...this.getAuthHeaders(),
                },
                body: JSON.stringify({ selectedWorkspaceIds, selectedDocumentIds }),
            });

            if (!response.ok) {
                return { success: false, error: 'Failed to update chat preferences' };
            }

            const data = await response.json();
            return { success: true, preferences: data.preferences };
        } catch (error) {
            console.error('Update chat preferences error:', error);
            return { success: false, error: error.message || 'Network error' };
        }
    }

    /**
     * Get theme preference
     */
    async getThemePreference() {
        try {
            const response = await fetch(`${AUTH_API_URL}/preferences/theme`, {
                method: 'GET',
                headers: {
                    ...this.getAuthHeaders(),
                },
            });

            if (!response.ok) {
                return { success: false, error: 'Failed to get theme preference' };
            }

            const data = await response.json();
            return { success: true, theme: data.theme };
        } catch (error) {
            console.error('Get theme preference error:', error);
            return { success: false, error: error.message || 'Network error' };
        }
    }

    /**
     * Update theme preference
     */
    async updateThemePreference(theme) {
        try {
            const response = await fetch(`${AUTH_API_URL}/preferences/theme`, {
                method: 'PATCH',
                headers: {
                    'Content-Type': 'application/json',
                    ...this.getAuthHeaders(),
                },
                body: JSON.stringify({ theme }),
            });

            if (!response.ok) {
                return { success: false, error: 'Failed to update theme preference' };
            }

            const data = await response.json();
            return { success: true, preferences: data.preferences };
        } catch (error) {
            console.error('Update theme preference error:', error);
            return { success: false, error: error.message || 'Network error' };
        }
    }
}

// Export singleton instance
const authService = new AuthService();
export default authService;
