/**
 * Questionnaire Status enumeration definitions.
 * Reference to Enumerations.QuestionnaireStatus in server side.
 * Reference to records in table dbo.[QuestionnaireStatus].
 * Work for questionnaire status management and display.
 */
export const QuestionnaireStatus = {
  /**
   * Draft status - Questionnaire is in draft state
   */
  Draft: 0,

  /**
   * Published status - Questionnaire has been published and is active
   */
  Published: 1,

  /**
   * Closed status - Questionnaire is closed after EndYear cycle published
   */
  Closed: 2
};

/**
 * Get questionnaire status display name
 * @param {number} statusId - The questionnaire status ID
 * @returns {string} Display name of the questionnaire status
 */
export const getQuestionnaireStatusName = (statusId) => {
  switch (statusId) {
    case QuestionnaireStatus.Draft:
      return 'Draft';
    case QuestionnaireStatus.Published:
      return 'Published';
    case QuestionnaireStatus.Closed:
      return 'Closed';
    default:
      return 'Unknown';
  }
};

/**
 * Get questionnaire status CSS class for styling
 * @param {number} statusId - The questionnaire status ID
 * @returns {string} CSS class name for the status
 */
export const getQuestionnaireStatusClass = (statusId) => {
  switch (statusId) {
    case QuestionnaireStatus.Draft:
      return 'status-draft';
    case QuestionnaireStatus.Published:
      return 'status-published';
    case QuestionnaireStatus.Closed:
      return 'status-closed';
    default:
      return 'status-unknown';
  }
};

/**
 * Check if questionnaire status allows editing in the designer. 
 * Note: If the questionnaire is closed, it can still be viewed in the designer, but it is not editable.
 * @param {number} statusId - The questionnaire status ID
 * @returns {boolean} True if questionnaire can be edited
 */
export const isQuestionnaireEditable = (statusId) => {
  return statusId !== QuestionnaireStatus.Closed;
};

/**
 * Check if questionnaire is active (can be used for forms)
 * @param {number} statusId - The questionnaire status ID
 * @returns {boolean} True if questionnaire is active
 */
export const isQuestionnaireActive = (statusId) => {
  return statusId === QuestionnaireStatus.Published;
};

/**
 * Get next possible statuses for a questionnaire
 * @param {number} currentStatusId - The current questionnaire status ID
 * @returns {Array} Array of possible next status IDs
 */
export const getNextPossibleStatuses = (currentStatusId) => {
  switch (currentStatusId) {
    case QuestionnaireStatus.Draft:
      return [QuestionnaireStatus.Published];
    case QuestionnaireStatus.Published:
      return [QuestionnaireStatus.Closed]; // Can close after EndYear cycle published
    case QuestionnaireStatus.Closed:
      return []; // No transitions from closed
    default:
      return [];
  }
};
