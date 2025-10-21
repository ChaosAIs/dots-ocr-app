import APP_CONFIG from "../config/appConfig";

/**
 * Debug utility functions for authentication and authorization
 * These functions help developers debug user roles and permissions
 */

/**
 * Log current user's roles and permissions to console
 * Can be called from anywhere in the application for debugging
 */
export const debugCurrentUserRoles = () => {
  console.group("🔍 Debug Current User Roles - Manual Check");
  
  try {
    // Get OIDC storage directly
    const oidcStorage = JSON.parse(
      sessionStorage.getItem(`oidc.user:${APP_CONFIG.iamDomain}:${APP_CONFIG.clientId}`)
    );

    if (!oidcStorage) {
      console.warn("❌ No OIDC storage found - user not authenticated");
      console.groupEnd();
      return null;
    }

    if (oidcStorage.expired) {
      console.warn("⏰ User token has expired");
      console.groupEnd();
      return null;
    }

    // Extract user information
    const profile = oidcStorage.profile || {};
    const userInfo = {
      id: oidcStorage.sub,
      userName: profile.name || "",
      displayName: profile.displayname || profile.name || "",
      email: profile.email || "",
      roles: [],
      permissions: []
    };

    // Parse roles
    if (profile.role) {
      userInfo.roles = profile.role.split(",").map(r => r.trim()).filter(r => r);
    }

    // Parse permissions  
    if (profile.permissions) {
      userInfo.permissions = profile.permissions.split(",").map(p => p.trim()).filter(p => p);
    }

    // Log the information
    console.log("👤 Current User:", {
      id: userInfo.id,
      userName: userInfo.userName,
      displayName: userInfo.displayName,
      email: userInfo.email
    });

    console.log("🎭 Current User Roles:", userInfo.roles);
    if (userInfo.roles.length > 0) {
      userInfo.roles.forEach((role, index) => {
        console.log(`  ${index + 1}. Role ID: ${role} (${getRoleDescription(role)})`);
      });
    } else {
      console.warn("⚠️ No roles assigned to current user");
    }

    console.log("🔑 Current User Permissions:", userInfo.permissions);
    if (userInfo.permissions.length > 0) {
      userInfo.permissions.forEach((permission, index) => {
        console.log(`  ${index + 1}. Permission ID: ${permission} (${getPermissionDescription(permission)})`);
      });
    } else {
      console.warn("⚠️ No permissions assigned to current user");
    }

    // Log raw token data
    console.log("🎫 Raw Token Profile:", profile);

    // Log token expiration
    const expiresAt = new Date(oidcStorage.expires_at * 1000);
    const now = new Date();
    const timeUntilExpiry = expiresAt - now;
    
    console.log("⏰ Token Expiration:");
    console.log(`  - Expires at: ${expiresAt.toLocaleString()}`);
    console.log(`  - Time until expiry: ${Math.round(timeUntilExpiry / 1000 / 60)} minutes`);

    console.groupEnd();
    return userInfo;

  } catch (error) {
    console.error("❌ Error debugging user roles:", error);
    console.groupEnd();
    return null;
  }
};

/**
 * Check if current user has a specific role
 * @param {string|number} roleId - The role ID to check
 * @returns {boolean} True if user has the role
 */
export const debugHasRole = (roleId) => {
  const userInfo = debugCurrentUserRoles();
  if (!userInfo) return false;
  
  const hasRole = userInfo.roles.includes(String(roleId));
  console.log(`🔍 Role Check: User ${hasRole ? 'HAS' : 'DOES NOT HAVE'} role ${roleId} (${getRoleDescription(roleId)})`);
  return hasRole;
};

/**
 * Check if current user has a specific permission
 * @param {string|number} permissionId - The permission ID to check
 * @returns {boolean} True if user has the permission
 */
export const debugHasPermission = (permissionId) => {
  const userInfo = debugCurrentUserRoles();
  if (!userInfo) return false;
  
  const hasPermission = userInfo.permissions.includes(String(permissionId));
  console.log(`🔍 Permission Check: User ${hasPermission ? 'HAS' : 'DOES NOT HAVE'} permission ${permissionId} (${getPermissionDescription(permissionId)})`);
  return hasPermission;
};

/**
 * Get human-readable description for role ID
 */
const getRoleDescription = (roleId) => {
  const roleDescriptions = {
    "15": "Partner Plans Administrator",
    "3": "Active Partner", 
    "16": "New Partner",
    "17": "Executive Leadership/Reviewer"
  };
  return roleDescriptions[String(roleId)] || "Unknown Role";
};

/**
 * Get human-readable description for permission ID
 */
const getPermissionDescription = (permissionId) => {
  const permissionDescriptions = {
    "1": "Login",
    "2": "Track Own Partner Plan",
    "3": "Track All Partner Plans", 
    "4": "Draft Submit Partner Plan",
    "5": "Edit Partner Plans Under Review",
    "6": "Partner Plans Final Submission",
    "7": "Mid End Year Self Assessment",
    "8": "Mid End Year Reviewer Assessment",
    "9": "View Submitted Partner Plans",
    "10": "Edit Submitted Partner Plans",
    "11": "Export Plan Data To Excel",
    "12": "Manage Partner Reviewer Relationships",
    "13": "Upload KPI Data",
    "14": "Edit Publish Input Form"
  };
  return permissionDescriptions[String(permissionId)] || "Unknown Permission";
};

/**
 * Make debug functions available globally for easy console access
 */
if (typeof window !== 'undefined') {
  window.debugAuth = {
    debugCurrentUserRoles,
    debugHasRole,
    debugHasPermission,
    logRoles: debugCurrentUserRoles, // Alias for easier typing
    hasRole: debugHasRole,
    hasPermission: debugHasPermission
  };
  
  console.log("🔧 Debug auth functions available globally:");
  console.log("  - window.debugAuth.debugCurrentUserRoles() or window.debugAuth.logRoles()");
  console.log("  - window.debugAuth.debugHasRole(roleId) or window.debugAuth.hasRole(roleId)");
  console.log("  - window.debugAuth.debugHasPermission(permissionId) or window.debugAuth.hasPermission(permissionId)");
}
