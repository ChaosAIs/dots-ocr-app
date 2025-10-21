/**
 * Reference to Enumerations.Permission in server side.
 * Reference to records in table dbo.[Permission].
 * Work for Authorizatoin check.
 */
export const Permission = {
  //
  // Partner Plan Management Permissions
  //
  Login: "1",
  TrackOwnPartnerPlan: "2",
  TrackAllPartnerPlans: "3",
  DraftSubmitPartnerPlan: "4",
  EditPartnerPlansUnderReview: "5",
  PartnerPlansFinalSubmission: "6",
  MidEndYearSelfAssessment: "7",
  MidEndYearReviewerAssessment: "8",
  ViewSubmittedPartnerPlans: "9",
  EditSubmittedPartnerPlans: "10",
  ExportPlanDataToExcel: "11",
  ManagePartnerReviewerRelationships: "12",
  UploadKPIData: "13",
  EditPublishInputForm: "14",
};
