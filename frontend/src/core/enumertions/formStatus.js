/**
 * Form Status enumeration definitions.
 * Reference to Enumerations.FormStatus in server side.
 * Reference to records in table dbo.[FormStatus].
 * Work for form status management and display.
 * Updated to match the new workflow-aware status system.
 */
export const FormStatus = {
  // Planning Cycle Statuses (0-4)
  /**
   * Planning - Not Started status - Form is not started yet for Planning cycle.
   */
  PlanningNotStarted: 0,

  /**
   * Planning - Draft status - Form is in draft state for Planning cycle.
   */
  PlanningDraft: 1,

  /**
   * Planning - Reopened status - Form is sent back by reviewer for further edits.
   */
  PlanningReopened: 2,

  /**
   * Planning - Under Review status - Form is submitted for review in Planning cycle.
   */
  PlanningUnderReview: 3,

  /**
   * Planning - Completed status - Form has been completed for Planning cycle.
   */
  PlanningCompleted: 4,

  // Mid Year Review Cycle Statuses (5-9)
  /**
   * Mid Year Review - Not Started status - Form is not started yet for Mid Year Review cycle.
   */
  MidYearReviewNotStarted: 5,

  /**
   * Mid Year Review - Draft status - Form is in draft state for Mid Year Review cycle.
   */
  MidYearReviewDraft: 6,

  /**
   * Mid Year Review - Reopened status - Form is sent back by reviewer for further edits.
   */
  MidYearReviewReopened: 7,

  /**
   * Mid Year Review - Under Review status - Form is submitted for review in Mid Year Review cycle.
   */
  MidYearReviewUnderReview: 8,

  /**
   * Mid Year Review - Completed status - Form has been completed for Mid Year Review cycle.
   */
  MidYearReviewCompleted: 9,

  // Year End Review Cycle Statuses (10-14)
  /**
   * Year End Review - Not Started status - Form is not started yet for Year End Review cycle.
   */
  YearEndReviewNotStarted: 10,

  /**
   * Year End Review - Draft status - Form is in draft state for Year End Review cycle.
   */
  YearEndReviewDraft: 11,

  /**
   * Year End Review - Reopened status - Form is sent back by reviewer for further edits.
   */
  YearEndReviewReopened: 12,

  /**
   * Year End Review - Under Review status - Form is submitted for review in Year End Review cycle.
   */
  YearEndReviewUnderReview: 13,

  /**
   * Year End Review - Completed status - Form has been completed for Year End Review cycle.
   */
  YearEndReviewCompleted: 14,
};

/**
 * Get form status display name
 * @param {number} statusId - The form status ID
 * @returns {string} Display name of the form status
 */
export const getFormStatusName = (statusId) => {
  switch (statusId) {
    // Planning Cycle
    case FormStatus.PlanningNotStarted:
      return "Not Started";
    case FormStatus.PlanningDraft:
      return "Draft";
    case FormStatus.PlanningReopened:
      return "Reopened";
    case FormStatus.PlanningUnderReview:
      return "Under Review";
    case FormStatus.PlanningCompleted:
      return "Completed";

    // Mid Year Review Cycle
    case FormStatus.MidYearReviewNotStarted:
      return "Not Started";
    case FormStatus.MidYearReviewDraft:
      return "Draft";
    case FormStatus.MidYearReviewReopened:
      return "Reopened";
    case FormStatus.MidYearReviewUnderReview:
      return "Under Review";
    case FormStatus.MidYearReviewCompleted:
      return "Completed";

    // Year End Review Cycle
    case FormStatus.YearEndReviewNotStarted:
      return "Not Started";
    case FormStatus.YearEndReviewDraft:
      return "Draft";
    case FormStatus.YearEndReviewReopened:
      return "Reopened";
    case FormStatus.YearEndReviewUnderReview:
      return "Under Review";
    case FormStatus.YearEndReviewCompleted:
      return "Completed";
    default:
      return "Unknown";
  }
};

/**
 * Get form status CSS class for styling
 * @param {number} statusId - The form status ID
 * @returns {string} CSS class name for the status
 */
export const getFormStatusClass = (statusId) => {
  switch (statusId) {
    // Planning Cycle
    case FormStatus.PlanningNotStarted:
      return "status-not-started";
    case FormStatus.PlanningDraft:
      return "status-draft";
    case FormStatus.PlanningReopened:
      return "status-reopened";
    case FormStatus.PlanningUnderReview:
      return "status-under-review";
    case FormStatus.PlanningCompleted:
      return "status-completed";

    // Mid Year Review Cycle
    case FormStatus.MidYearReviewNotStarted:
      return "status-not-started";
    case FormStatus.MidYearReviewDraft:
      return "status-draft";
    case FormStatus.MidYearReviewReopened:
      return "status-reopened";
    case FormStatus.MidYearReviewUnderReview:
      return "status-under-review";
    case FormStatus.MidYearReviewCompleted:
      return "status-completed";

    // Year End Review Cycle
    case FormStatus.YearEndReviewNotStarted:
      return "status-not-started";
    case FormStatus.YearEndReviewDraft:
      return "status-draft";
    case FormStatus.YearEndReviewReopened:
      return "status-reopened";
    case FormStatus.YearEndReviewUnderReview:
      return "status-under-review";
    case FormStatus.YearEndReviewCompleted:
      return "status-completed";

    default:
      return "status-unknown";
  }
};

/**
 * Check if a form status represents a completed state
 * @param {number} statusId - The form status ID
 * @returns {boolean} True if the form is completed
 */
export const isFormCompleted = (statusId) => {
  return (
    statusId === FormStatus.PlanningCompleted || statusId === FormStatus.MidYearReviewCompleted || statusId === FormStatus.YearEndReviewCompleted
  );
};

/**
 * Check if a form status represents a reopened state
 * @param {number} statusId - The form status ID
 * @returns {boolean} True if the form is in reopened state
 */
export const isFormReopened = (statusId) => {
  return statusId === FormStatus.PlanningReopened || statusId === FormStatus.MidYearReviewReopened || statusId === FormStatus.YearEndReviewReopened;
};
