/** Corporate with server side project, Enumerations.PartnerPlanCycle definitions. */
export const PartnerPlanCycle = {
  /** Planning cycle. 0 */
  Planning: 0,
  /** Mid-Year Review cycle. 1 */
  MidYearReview: 1,
  /** Year-End Review cycle. 2 */
  YearEndReview: 2,
};

/** Helper function to get cycle display name */
export const getCycleDisplayName = (cycle) => {
  switch (cycle) {
    case PartnerPlanCycle.Planning:
      return "Planning";
    case PartnerPlanCycle.MidYearReview:
      return "Mid-Year Review";
    case PartnerPlanCycle.YearEndReview:
      return "Year-End Review";
    default:
      return "Unknown";
  }
};

/** Helper function to get cycle from form status */
export const getCycleFromFormStatus = (formStatus) => {
  // Based on FormStatus enumeration: 0-4=Planning, 5-9=MidYear, 10-14=YearEnd
  if (formStatus >= 0 && formStatus <= 4) {
    return PartnerPlanCycle.Planning;
  } else if (formStatus >= 5 && formStatus <= 9) {
    return PartnerPlanCycle.MidYearReview;
  } else if (formStatus >= 10 && formStatus <= 14) {
    return PartnerPlanCycle.YearEndReview;
  }
  // Default fallback
  return PartnerPlanCycle.Planning;
};

/** Helper function to get due date from questionnaire based on cycle */
export const getDueDateForCycle = (questionnaire, cycle) => {
  if (!questionnaire) return null;

  switch (cycle) {
    case PartnerPlanCycle.Planning:
      return questionnaire.planningDueDate;
    case PartnerPlanCycle.MidYearReview:
      return questionnaire.midYearReviewDueDate;
    case PartnerPlanCycle.YearEndReview:
      return questionnaire.endYearReviewDueDate;
    default:
      return null;
  }
};

/** Helper function to format due date for display */
export const formatDueDate = (dueDate) => {
  if (!dueDate) return "No due date";
  const date = new Date(dueDate);
  return date.toLocaleDateString();
};

/** Get cycle options for dropdowns */
export const getCycleOptions = () => [
  { label: "Planning", value: PartnerPlanCycle.Planning },
  { label: "Mid-Year Review", value: PartnerPlanCycle.MidYearReview },
  { label: "Year-End Review", value: PartnerPlanCycle.YearEndReview },
];
