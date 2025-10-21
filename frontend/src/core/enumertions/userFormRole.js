/**
 * User Form Role enumeration definitions.
 * Reference to Enumerations.UserFormRole in server side.
 * Reference to records in table dbo.[FormAccessConfig] column "UserRole".
 * Note: It is different with UserRole in Identity database.
 * Work for user's role in relation to a specific form.
 */
export const UserFormRole = {
  /**
   * User has no access to the form. Value = 0
   */
  NoAccess: 0,

  /**
   * User is the form owner (partner). Value = 1
   */
  FormOwner: 1,

  /**
   * User is the primary reviewer or Secondary reviewer for this form. Value = 2
   */
  Reviewer: 2,

  /**
   * User is an admin with full access. Value = 3
   */
  Admin: 3,

  /**
   * ExecutiveLeadership and he/she is not current process partner plan's reviewer. Value = 4
   */
  ELT: 4,
};

/**
 * Get user form role display name
 * @param {number} roleId - The user form role ID
 * @returns {string} Display name of the user form role
 */
export const getUserFormRoleName = (roleId) => {
  switch (roleId) {
    case UserFormRole.NoAccess:
      return "No Access";
    case UserFormRole.FormOwner:
      return "Form Owner";
    case UserFormRole.Reviewer:
      return "Reviewer";
    case UserFormRole.Admin:
      return "Admin";
    case UserFormRole.ELT:
      return "Executive Leadership";
    default:
      return "Unknown";
  }
};

/**
 * Get user form role CSS class for styling
 * @param {number} roleId - The user form role ID
 * @returns {string} CSS class name for the role
 */
export const getUserFormRoleClass = (roleId) => {
  switch (roleId) {
    case UserFormRole.NoAccess:
      return "role-no-access";
    case UserFormRole.FormOwner:
      return "role-form-owner";
    case UserFormRole.Reviewer:
      return "role-reviewer";
    case UserFormRole.Admin:
      return "role-admin";
    case UserFormRole.ELT:
      return "role-elt";
    default:
      return "role-unknown";
  }
};

/**
 * Check if a user form role has edit permissions
 * @param {number} roleId - The user form role ID
 * @returns {boolean} True if the role has edit permissions
 */
export const canEditForm = (roleId) => {
  return roleId === UserFormRole.FormOwner || roleId === UserFormRole.Admin;
};

/**
 * Check if a user form role has review permissions
 * @param {number} roleId - The user form role ID
 * @returns {boolean} True if the role has review permissions
 */
export const canReviewForm = (roleId) => {
  return roleId === UserFormRole.Reviewer || roleId === UserFormRole.Admin;
};

/**
 * Check if a user form role has admin permissions
 * @param {number} roleId - The user form role ID
 * @returns {boolean} True if the role has admin permissions
 */
export const hasAdminAccess = (roleId) => {
  return roleId === UserFormRole.Admin;
};

/**
 * Check if a user form role has any access to the form
 * @param {number} roleId - The user form role ID
 * @returns {boolean} True if the role has any access
 */
export const hasFormAccess = (roleId) => {
  return roleId !== UserFormRole.NoAccess;
};
