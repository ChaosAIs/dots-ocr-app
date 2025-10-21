import { UserManager, WebStorageStateStore, Log } from "oidc-client";
import APP_CONFIG from "../config/appConfig";
import { IDENTITY_CLIENT_CONFIG } from "../config/identityClientConfig";
import { IDENTITY_META_CONFIG } from "../config/identityMetaConfig";

/**
 * Authentication/Authorization service.
 * Corporate with Identity Server 4.
 *
 * Reference: https://medium.com/@franciscopa91/how-to-implement-oidc-authentication-with-react-context-api-and-react-router-205e13f2d49
 */
export default class AuthService {
  userManager;

  constructor() {
    // Create state store without explicit prefix - let library handle it
    const stateStore = new WebStorageStateStore({
      store: window.sessionStorage,
    });

    const userStore = new WebStorageStateStore({
      store: window.sessionStorage,
    });

    this.userManager = new UserManager({
      ...IDENTITY_CLIENT_CONFIG,
      userStore: userStore,
      stateStore: stateStore,
      metadata: {
        ...IDENTITY_META_CONFIG,
      },
    });
    //
    // Register OIDC client Logger
    // Exposed OIDC-Client library debugging information into console log.
    //
    Log.logger = console;
    Log.level = Log.WARN; // Enable logging to help debug state issues

    this.userManager.events.addUserLoaded((user) => {
      console.log("🔄 User loaded event triggered");
      if (user) {
        // Debug log the user that was loaded
        const appUser = this.toAppUser(user);
        console.log("🔄 User loaded with roles:", appUser?.roles);
        console.log("🔄 User loaded with permissions:", appUser?.permissions);
      }

      if (window.location.href.indexOf("auth-callback") !== -1) {
        this.navigateToHome();
      }
    });

    this.userManager.events.addSilentRenewError((e) => {
      console.log("silent renew error", e.message);
      // If silent renewal fails, redirect to login
      console.log("Silent renewal failed, redirecting to login");
      this.signinRedirect();
    });

    this.userManager.events.addAccessTokenExpired(() => {
      console.log("token expired");
      this.signinSilent();
    });

    // Clear stale state on initialization
    this.userManager.clearStaleState();
  }

  signinRedirectCallback = () => {
    console.log("SigninRedirectCallback called");
    console.log("Current URL:", window.location.href);

    // Log all session storage keys for debugging
    const sessionKeys = [];
    for (let i = 0; i < sessionStorage.length; i++) {
      const key = sessionStorage.key(i);
      if (key && key.includes("oidc")) {
        sessionKeys.push(key);
      }
    }
    console.log("OIDC SessionStorage keys before callback:", sessionKeys);

    return this.userManager
      .signinRedirectCallback()
      .then((user) => {
        console.log("SigninRedirectCallback success:", user);

        // Debug log user roles immediately after successful authentication
        if (user) {
          console.log("🎉 Authentication successful! Logging user roles...");
          const appUser = this.toAppUser(user);
          this.debugLogUserRoles(appUser, user);
        }

        const redirectUrl = localStorage.getItem("redirectUri");

        if (redirectUrl && redirectUrl.length > 0) {
          window.location.replace(redirectUrl);
        } else {
          this.navigateToHome();
        }
        return user;
      })
      .catch((error) => {
        console.error("SigninRedirectCallback error:", error);
        console.error("Error details:", error.message, error.stack);

        // Log session storage state when error occurs
        const sessionKeysOnError = [];
        for (let i = 0; i < sessionStorage.length; i++) {
          const key = sessionStorage.key(i);
          if (key && key.includes("oidc")) {
            sessionKeysOnError.push(key);
          }
        }
        console.log("OIDC SessionStorage keys on error:", sessionKeysOnError);

        // Clear any stale state and redirect to home
        this.userManager.clearStaleState();
        localStorage.removeItem("redirectUri");
        throw error; // Re-throw to allow component to handle
      });
  };

  /**
   *  Get logon user object if system has. Otherwises, return null.
   *  Note: Here only try to return user profile info. It does not require system login.
   *  If no login user, then, return null.
   *  Work for UI dispolay user profile information purpose.
   *  For example, as for home page which is public page,
   *  system also will try to publish logon user info on the menu
   *  if there is any user login success.
   *
   *
   * Return User object json format:
   * {
   *   id,
   *   userName,
   *   displayName,
   *   firstName,
   *   lastName,
   *   email,
   *   language,
   *   roles,
   *   permissions,
   *   authProviderId
   * }
   *
   */
  getUser = () => {
    const oidcStorage = JSON.parse(
      /**
       * Note: It is OIDC-Client library default storage.
       * The key format is defined by oidc-client library.
       * Note: Developer no needs to modify the session key.
       * */
      sessionStorage.getItem(`oidc.user:${APP_CONFIG.iamDomain}:${APP_CONFIG.clientId}`)
    );

    if (oidcStorage && oidcStorage.access_token && !oidcStorage.expired) {
      const user = this.toAppUser(oidcStorage);

      // Debug logging for user roles and permissions
      this.debugLogUserRoles(user, oidcStorage);

      return user;
    } else {
      return null;
    }
  };

  /**
   * Convert ID4 user to DTO user. Only called by getUser.
   * @param user ID4 returned user claims, refer to OIDC-Client library.
   */
  toAppUser(user) {
    const result = {};
    if (user && !user.expired) {
      // console.debug("toAppUser", "OIDC user: " + JSON.stringify(user.profile));

      result.id = user.sub;
      result.userName = user.profile.name ?? "";
      result.email = user.profile.email ?? "";
      result.firstName = user.profile.given_name ?? "";
      result.lastName = user.profile.family_name ?? "";

      // Construct display name: use profile displayname if available and not just username,
      // otherwise construct from firstName and lastName
      let displayName = user.profile.displayname ?? "";

      // If displayName is empty or appears to be just a username (same as userName),
      // construct it from firstName and lastName
      if (!displayName || displayName === result.userName) {
        const firstName = result.firstName.trim();
        const lastName = result.lastName.trim();

        if (firstName && lastName) {
          displayName = `${firstName} ${lastName}`;
        } else if (firstName) {
          displayName = firstName;
        } else if (lastName) {
          displayName = lastName;
        } else {
          // Fallback to userName if no name components are available
          displayName = result.userName;
        }
      }

      result.displayName = displayName;
      result.language = user.profile.locale;
      result.accessToken = user.access_token;
      result.authProviderId = user.profile.idp === "local" ? "APP" : user.profile.idp; // Database driving login is "local" for Identity Server. Convert it to "APP".
      const strRoles = user.profile.role;
      const strPermissions = user.profile.permissions;

      // Debug log raw role and permission strings
      console.log("🔍 Raw role string from token:", strRoles);
      console.log("🔍 Raw permissions string from token:", strPermissions);

      if (strRoles) {
        const roles = [];
        strRoles.split(",").forEach((r) => {
          const trimmedRole = r.trim();
          if (trimmedRole) {
            roles.push(trimmedRole);
          }
        });
        result.roles = roles;
        console.log("🔍 Parsed roles array:", roles);
      }

      if (strPermissions) {
        const permissions = [];
        strPermissions.split(",").forEach((p) => {
          const trimmedPermission = p.trim();
          if (trimmedPermission) {
            permissions.push(trimmedPermission);
          }
        });

        result.permissions = permissions;
        console.log("🔍 Parsed permissions array:", permissions);
      }
    }

    return result;
  }

  parseJwt = (token) => {
    const base64Url = token.split(".")[1];
    const base64 = base64Url.replace("-", "+").replace("_", "/");
    return JSON.parse(window.atob(base64));
  };

  signin = () => {
    this.signinRedirect();
  };

  /** Redirect to Identity Server 4 login page for login requirement. */
  signinRedirect = () => {
    console.log("SigninRedirect called from:", window.location.href);

    // Clear any stale state before starting new authentication
    this.userManager.clearStaleState();
    localStorage.setItem("redirectUri", window.location.pathname);

    console.log("Redirect URI set to:", window.location.pathname);
    console.log("Starting signin redirect...");

    return this.userManager
      .signinRedirect({})
      .then(() => {
        console.log("SigninRedirect initiated successfully");
      })
      .catch((error) => {
        console.error("SigninRedirect error:", error);
        throw error;
      });
  };

  /** Redirect to Home page which is public page. */
  navigateToHome = () => {
    window.location.replace(APP_CONFIG.basePath);
  };

  isAuthenticated = () => {
    const oidcStorage = JSON.parse(
      /**
       * Note: It is OIDC-Client library default storage.
       * The key format is defined by oidc-client library.
       * Note: Developer no needs to modify the session key.
       * */
      sessionStorage.getItem(`oidc.user:${APP_CONFIG.iamDomain}:${APP_CONFIG.clientId}`)
    );
    return !!oidcStorage && !!oidcStorage.access_token && !oidcStorage.expired;
  };

  signinSilent = () => {
    console.log("signinSilent called");
    this.userManager
      .signinSilent()
      .then((user) => {
        console.log("silent signed in success", user);
      })
      .catch((err) => {
        console.log("silent signed in error", err);
        // If silent signin fails, redirect to login
        console.log("Silent signin failed, redirecting to login");
        this.signinRedirect();
      });
  };

  signinSilentCallback = () => {
    console.log("SigninSilentCallback called");
    this.userManager.signinSilentCallback();
  };

  createSigninRequest = () => {
    return this.userManager.createSigninRequest();
  };

  signout = () => {
    this.userManager.signoutRedirect({
      id_token_hint: localStorage.getItem("id_token"),
    });
    this.userManager.clearStaleState();
  };

  signoutRedirectCallback = () => {
    this.userManager.signoutRedirectCallback().then(() => {
      localStorage.clear();
      // Go to home page.
      window.location.replace("/");
    });
    this.userManager.clearStaleState();
  };

  /** Clear stale authentication state */
  clearAuthState = () => {
    this.userManager.clearStaleState();
    localStorage.removeItem("redirectUri");

    // Clear all OIDC-related storage
    const keysToRemove = [];
    for (let i = 0; i < sessionStorage.length; i++) {
      const key = sessionStorage.key(i);
      if (key && key.includes("oidc")) {
        keysToRemove.push(key);
      }
    }
    keysToRemove.forEach((key) => sessionStorage.removeItem(key));

    // Also clear localStorage OIDC keys
    const localKeysToRemove = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.includes("oidc")) {
        localKeysToRemove.push(key);
      }
    }
    localKeysToRemove.forEach((key) => localStorage.removeItem(key));
  };

  /** Debug method to log current authentication state */
  debugAuthState = () => {
    console.log("=== Authentication Debug Info ===");
    console.log("APP_CONFIG:", APP_CONFIG);
    console.log("Current URL:", window.location.href);
    console.log("Redirect URI:", localStorage.getItem("redirectUri"));

    // Parse URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    console.log("URL Parameters:", Object.fromEntries(urlParams));

    // Check sessionStorage for OIDC state
    const stateKeys = [];
    for (let i = 0; i < sessionStorage.length; i++) {
      const key = sessionStorage.key(i);
      if (key && key.includes("oidc")) {
        stateKeys.push({
          key: key,
          value: sessionStorage.getItem(key),
        });
      }
    }
    console.log("OIDC SessionStorage entries:", stateKeys);

    // Check if user is authenticated
    console.log("Is Authenticated:", this.isAuthenticated());
    console.log("Current User:", this.getUser());
    console.log("=== End Debug Info ===");
  };

  /** Manual callback handler for debugging */
  manualCallbackHandler = () => {
    const urlParams = new URLSearchParams(window.location.search);
    const code = urlParams.get("code");
    const state = urlParams.get("state");
    const sessionState = urlParams.get("session_state");

    console.log("Manual callback handler:");
    console.log("Code:", code);
    console.log("State:", state);
    console.log("Session State:", sessionState);

    // Check different possible state key formats
    const possibleStateKeys = [`oidc.state:${state}`, `oidc.${state}`, state, `state:${state}`];

    let foundState = null;
    for (const stateKey of possibleStateKeys) {
      const storedState = sessionStorage.getItem(stateKey);
      console.log(`Checking state key: ${stateKey} -> ${storedState ? "FOUND" : "NOT FOUND"}`);
      if (storedState) {
        foundState = { key: stateKey, value: storedState };
        break;
      }
    }

    if (!foundState) {
      console.error("No matching state found for:", state);
      // List all state-related keys
      const allStateKeys = [];
      for (let i = 0; i < sessionStorage.length; i++) {
        const key = sessionStorage.key(i);
        if (key && (key.includes("oidc") || key.includes("state"))) {
          allStateKeys.push({
            key: key,
            value: sessionStorage.getItem(key),
          });
        }
      }
      console.log("All OIDC/state keys:", allStateKeys);
    } else {
      console.log("Found matching state:", foundState);
    }
  };

  /**
   * Debug logging method to expose logged-in user's roles and permissions
   * @param {Object} user - The processed user object
   * @param {Object} oidcStorage - The raw OIDC storage object
   */
  debugLogUserRoles = (user, oidcStorage) => {
    console.group("🔐 User Authentication Debug Info");

    // Log basic user info
    console.log("👤 User Info:", {
      id: user?.id,
      userName: user?.userName,
      displayName: user?.displayName,
      email: user?.email,
      authProvider: user?.authProviderId,
    });

    // Log roles with detailed information
    if (user?.roles && user.roles.length > 0) {
      console.log("🎭 User Roles:", user.roles);
      console.log("🎭 Roles Details:");
      user.roles.forEach((role, index) => {
        console.log(`  ${index + 1}. Role ID: ${role} (${this.getRoleDescription(role)})`);
      });
    } else {
      console.warn("⚠️ No roles found for user");
    }

    // Log permissions with detailed information
    if (user?.permissions && user.permissions.length > 0) {
      console.log("🔑 User Permissions:", user.permissions);
      console.log("🔑 Permissions Details:");
      user.permissions.forEach((permission, index) => {
        console.log(`  ${index + 1}. Permission ID: ${permission} (${this.getPermissionDescription(permission)})`);
      });
    } else {
      console.warn("⚠️ No permissions found for user");
    }

    // Log raw claims from OIDC token
    if (oidcStorage?.profile) {
      console.log("🎫 Raw Token Claims:");
      console.log("  - Role claim:", oidcStorage.profile.role);
      console.log("  - Permissions claim:", oidcStorage.profile.permissions);
      console.log("  - All profile claims:", oidcStorage.profile);
    }

    // Log token expiration info
    if (oidcStorage) {
      const expiresAt = new Date(oidcStorage.expires_at * 1000);
      const now = new Date();
      const timeUntilExpiry = expiresAt - now;

      console.log("⏰ Token Info:");
      console.log(`  - Expires at: ${expiresAt.toLocaleString()}`);
      console.log(`  - Time until expiry: ${Math.round(timeUntilExpiry / 1000 / 60)} minutes`);
      console.log(`  - Is expired: ${oidcStorage.expired}`);
    }

    console.groupEnd();
  };

  /**
   * Get human-readable description for role ID
   * @param {string} roleId - The role ID
   * @returns {string} Role description
   */
  getRoleDescription = (roleId) => {
    const roleDescriptions = {
      15: "Partner Plans Administrator",
      3: "Active Partner",
      16: "New Partner",
      17: "Executive Leadership/Reviewer",
    };
    return roleDescriptions[roleId] || "Unknown Role";
  };

  /**
   * Get human-readable description for permission ID
   * @param {string} permissionId - The permission ID
   * @returns {string} Permission description
   */
  getPermissionDescription = (permissionId) => {
    const permissionDescriptions = {
      1: "Login",
      2: "Track Own Partner Plan",
      3: "Track All Partner Plans",
      4: "Draft Submit Partner Plan",
      5: "Edit Partner Plans Under Review",
      6: "Partner Plans Final Submission",
      7: "Mid End Year Self Assessment",
      8: "Mid End Year Reviewer Assessment",
      9: "View Submitted Partner Plans",
      10: "Edit Submitted Partner Plans",
      11: "Export Plan Data To Excel",
      12: "Manage Partner Reviewer Relationships",
      13: "Upload KPI Data",
      14: "Edit Publish Input Form",
    };
    return permissionDescriptions[permissionId] || "Unknown Permission";
  };
}
