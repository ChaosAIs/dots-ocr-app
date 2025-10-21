import { Serializer } from "survey-core";
import partnerReferenceDataUploadService from "../../services/partnerReferenceDataUploadService";
import formService from "../../services/formService";
import lookupService from "../../services/lookupService";

/**
 * Register custom properties for SurveyJS runtime and designer
 * This utility consolidates the custom property registration logic used by both
 * PartnerPlanQuestionnaire.jsx and QuestionnaireDesignerCore.jsx
 */

/**
 * Register custom properties for runtime (must be called before creating Survey model)
 * Work for survey form serialization process to recognize added custom properties in SurveyJS Form Designer.
 * Note: If not calling this method, even the form template json file contains the custom properties,
 * the survey model will not contain those properties.
 *
 * @param {Object} options - Configuration options
 * @param {boolean} options.isDesigner - Whether this is for the designer (includes additional properties)
 * @param {number} options.questionnaireYear - The year of the questionnaire (for designer mode)
 * @param {boolean} options.skipDuplicateCheck - Skip checking for existing properties (for designer mode)
 */
export const registerCustomPropertiesForRuntime = async (options = {}) => {
  const { isDesigner = false, questionnaireYear = null, skipDuplicateCheck = false } = options;

  try {
    let columnChoices = [];
    let groupChoices = [];
    let leadershipRoleChoices = [];
    let serviceLinesChoices = [];
    let subServiceLinesChoices = [];

    // For designer mode, fetch dynamic data
    if (isDesigner) {
      columnChoices = await partnerReferenceDataUploadService.getAvailableColumnNamesForMapping(questionnaireYear, true);
      groupChoices = await partnerReferenceDataUploadService.getUniqueGroupNames(questionnaireYear);

      // Fetch leadership roles for the questionnaire year
      try {
        const leadershipRoles = await formService.getUniqueLeadershipRoles(questionnaireYear);
        // Convert Lookup objects to choice format: Key (normalized) as value, Value (original) as text
        leadershipRoleChoices = leadershipRoles.map((lookup) => ({ value: lookup.key, text: lookup.value }));
      } catch (error) {
        console.warn("Failed to fetch leadership roles:", error);
        leadershipRoleChoices = [];
      }

      // Fetch service lines for dropdown choices
      try {
        const serviceLines = await lookupService.getServiceLines(false);
        // Convert Lookup objects to choice format: Key as value, Value as text
        serviceLinesChoices = serviceLines.map((lookup) => ({ value: lookup.key, text: lookup.value }));
      } catch (error) {
        console.warn("Failed to fetch service lines:", error);
        serviceLinesChoices = [];
      }

      // Fetch sub-service lines for dropdown choices
      try {
        const subServiceLines = await lookupService.getSubServiceLines(false);
        // Convert Lookup objects to choice format: Key as value, Value as text
        subServiceLinesChoices = subServiceLines.map((lookup) => ({ value: lookup.key, text: lookup.value }));
      } catch (error) {
        console.warn("Failed to fetch sub-service lines:", error);
        subServiceLinesChoices = [];
      }
    }

    // Register tag property for panels
    if (skipDuplicateCheck || !Serializer.findProperty("panel", "tag")) {
      Serializer.addProperty("panel", {
        name: "tag",
        displayName: "Tag",
        category: "customSettings",
        categoryDisplayName: "Custom Settings",
        type: "dropdown",
        choices: [
          { value: "", text: "-- None --" },
          { value: "PlanPanel", text: "Planning Panel" },
          { value: "MidYearPanel", text: isDesigner ? "Mid-year Panel" : "Mid-Year Panel" },
          { value: "YearEndPanel", text: isDesigner ? "Year-end Panel" : "Year-End Panel" },
        ],
        ...(isDesigner && { visibleIndex: 1 }),
        description: "Select a tag for the panel that contains questions for the Planning, Mid-Year, or Year-End period.",
      });
    }

    // Register LeadershipRoles property for panels
    if (skipDuplicateCheck || !Serializer.findProperty("panel", "leadershipRoles")) {
      Serializer.addProperty("panel", {
        name: "leadershipRoles",
        displayName: "Leadership Roles",
        category: "customSettings",
        categoryDisplayName: "Custom Settings",
        type: isDesigner ? "multiplevalues" : "string",
        ...(isDesigner && {
          choices: leadershipRoleChoices.length
            ? [{ value: "", text: "-- None --" }, ...leadershipRoleChoices]
            : [
                { value: "", text: "-- None --" },
                { value: "", text: "No leadership roles available" },
              ],
          visibleIndex: 2,
        }),
        description: "Select leadership roles that can view this panel. If empty, panel is visible to all users.",
      });
    }

    // Register ServiceLines property for panels
    if (skipDuplicateCheck || !Serializer.findProperty("panel", "serviceLines")) {
      Serializer.addProperty("panel", {
        name: "serviceLines",
        displayName: "Service Lines",
        category: "customSettings",
        categoryDisplayName: "Custom Settings",
        type: isDesigner ? "multiplevalues" : "string",
        ...(isDesigner && {
          choices: serviceLinesChoices.length
            ? [{ value: "", text: "-- None --" }, ...serviceLinesChoices]
            : [
                { value: "", text: "-- None --" },
                { value: "", text: "No service lines available" },
              ],
          visibleIndex: 3,
        }),
        description: "Select service lines that can view this panel. If empty, panel is visible to all users.",
      });
    }

    // Register SubServiceLines property for panels
    if (skipDuplicateCheck || !Serializer.findProperty("panel", "subServiceLines")) {
      Serializer.addProperty("panel", {
        name: "subServiceLines",
        displayName: "Sub-Service Lines",
        category: "customSettings",
        categoryDisplayName: "Custom Settings",
        type: isDesigner ? "multiplevalues" : "string",
        ...(isDesigner && {
          choices: subServiceLinesChoices.length
            ? [{ value: "", text: "-- None --" }, ...subServiceLinesChoices]
            : [
                { value: "", text: "-- None --" },
                { value: "", text: "No sub-service lines available" },
              ],
          visibleIndex: 4,
        }),
        description: "Select sub-service lines that can view this panel. If empty, panel is visible to all users.",
      });
    }

    // Register ExceptServiceLines property for panels
    if (skipDuplicateCheck || !Serializer.findProperty("panel", "exceptServiceLines")) {
      Serializer.addProperty("panel", {
        name: "exceptServiceLines",
        displayName: "Except Service Lines",
        category: "customSettings",
        categoryDisplayName: "Custom Settings",
        type: isDesigner ? "multiplevalues" : "string",
        ...(isDesigner && {
          choices: serviceLinesChoices.length
            ? [{ value: "", text: "-- None --" }, ...serviceLinesChoices]
            : [
                { value: "", text: "-- None --" },
                { value: "", text: "No service lines available" },
              ],
          visibleIndex: 5,
        }),
        description: "Select service lines that are NOT allowed to view this panel. If empty, no service lines are excluded.",
      });
    }

    // Register ExceptSubServiceLines property for panels
    if (skipDuplicateCheck || !Serializer.findProperty("panel", "exceptSubServiceLines")) {
      Serializer.addProperty("panel", {
        name: "exceptSubServiceLines",
        displayName: "Except Sub-Service Lines",
        category: "customSettings",
        categoryDisplayName: "Custom Settings",
        type: isDesigner ? "multiplevalues" : "string",
        ...(isDesigner && {
          choices: subServiceLinesChoices.length
            ? [{ value: "", text: "-- None --" }, ...subServiceLinesChoices]
            : [
                { value: "", text: "-- None --" },
                { value: "", text: "No sub-service lines available" },
              ],
          visibleIndex: 6,
        }),
        description: "Select sub-service lines that are NOT allowed to view this panel. If empty, no sub-service lines are excluded.",
      });
    }

    // Register tag property for questions
    if (skipDuplicateCheck || !Serializer.findProperty("question", "tag")) {
      const questionTagChoices = isDesigner
        ? [
            { value: "", text: "-- None --" },
            { value: "PlanningPartnerQuestion", text: "Planning Partner Question" },
            { value: "PlanningReviewerQuestion", text: "Planning Reviewer Question" },
            { value: "MidyearPartnerQuestion", text: "Mid-year Partner Question" },
            { value: "MidyearReviewerQuestion", text: "Mid-Year Reviewer Question" },
            { value: "YearendPartnerQuestion", text: "Year-end Partner Question" },
            { value: "YearendReviewerQuestion", text: "Year-end Reviewer Question" },
          ]
        : [
            { value: "", text: "-- None --" },
            { value: "PlanningPartnerQuestion", text: "Planning Partner Question" },
            { value: "MidYearPartnerQuestion", text: "Mid-Year Partner Question" },
            { value: "YearEndPartnerQuestion", text: "Year-End Partner Question" },
            { value: "PlanningReviewerQuestion", text: "Planning Reviewer Question" },
            { value: "MidYearReviewerQuestion", text: "Mid-Year Reviewer Question" },
            { value: "YearEndReviewerQuestion", text: "Year-End Reviewer Question" },
          ];

      Serializer.addProperty("question", {
        name: "tag",
        displayName: "Tag",
        category: "customSettings",
        categoryDisplayName: "Custom Settings",
        type: "dropdown",
        choices: questionTagChoices,
        ...(isDesigner && { visibleIndex: 0 }),
        description: isDesigner
          ? "Select a partner reference data column to map this question to"
          : "Select a tag for the question that indicates the period and role.",
      });
    }

    // Register mapFrom property for questions
    if (skipDuplicateCheck || !Serializer.findProperty("question", "mapFrom")) {
      Serializer.addProperty("question", {
        name: "mapFrom",
        displayName: "Map From",
        category: "customSettings",
        categoryDisplayName: "Custom Settings",
        type: isDesigner ? "dropdown" : "string",
        ...(isDesigner && {
          choices: columnChoices.length ? [{ value: "", text: "-- None --" }, ...columnChoices] : [{ value: "", text: "-- None --" }],
        }),
        ...(isDesigner && { visibleIndex: 1 }),
        description: "Select a partner reference data column to map this question to",
      });
    }

    // Register exportColumnName property for questions
    if (skipDuplicateCheck || !Serializer.findProperty("question", "exportColumnName")) {
      Serializer.addProperty("question", {
        name: "exportColumnName",
        displayName: "Export Column Name",
        category: "customSettings",
        categoryDisplayName: "Custom Settings",
        type: "text",
        ...(isDesigner && { visibleIndex: 2 }),
        description: "Specify a custom column name for data export",
      });
    }

    // Register mapFromGroup property for questions
    if (skipDuplicateCheck || !Serializer.findProperty("question", "mapFromGroup")) {
      Serializer.addProperty("question", {
        name: "mapFromGroup",
        displayName: "Partner Reference Group",
        category: "customSettings",
        categoryDisplayName: "Custom Settings",
        type: isDesigner ? "dropdown" : "string",
        ...(isDesigner && {
          choices: groupChoices.length
            ? [{ value: "", text: "-- None --" }, ...groupChoices]
            : [
                { value: "", text: "-- None --" },
                { value: "", text: "No groups available" },
              ],
          visibleIndex: 3,
          visibleIf: (obj) => obj && obj.getType && obj.getType() === "dropdown",
        }),
        description: "Select a partner reference data group to map this question to",
      });
    }

    // Register linkedToGroupPlanning property for questions.
    // Work for rendering planning cycle partner reference data's specified column value.
    if (skipDuplicateCheck || !Serializer.findProperty("question", "linkedToGroupPlanning")) {
      const linkedToGroupPlanningConfig = {
        name: "linkedToGroupPlanning",
        displayName: "Linked Group (Planning)",
        category: "customSettings",
        categoryDisplayName: "Custom Settings",
        type: isDesigner ? "dropdown" : "string",
        description: isDesigner
          ? "Link this text field to a dropdown with mapFromGroup property for auto-population during Planning cycle"
          : "Link this text field to a dropdown for auto-population during Planning cycle",
      };

      if (isDesigner) {
        linkedToGroupPlanningConfig.choices = function (obj) {
          const survey = obj && obj.survey;
          if (!survey) return [{ value: "", text: "-- None --" }];

          const dropdownQuestions = survey
            .getAllQuestions()
            .filter((q) => q.getType() === "dropdown" && q.mapFromGroup)
            .map((q) => ({
              value: q.name,
              text: `${q.title || q.name} (Group: ${q.mapFromGroup})`,
            }));

          return [{ value: "", text: "-- None --" }, ...dropdownQuestions];
        };
        linkedToGroupPlanningConfig.visibleIndex = 2;
        linkedToGroupPlanningConfig.visibleIf = (obj) => obj && obj.getType && obj.getType() === "text";
      }

      Serializer.addProperty("question", linkedToGroupPlanningConfig);
    }

    // Register linkedToGroupMidYear property for questions.
    // Work for rendering Mid year cycle partner reference data's specified column value.
    if (skipDuplicateCheck || !Serializer.findProperty("question", "linkedToGroupMidYear")) {
      const linkedToGroupMidYearConfig = {
        name: "linkedToGroupMidYear",
        displayName: "Linked Group (Mid Year)",
        category: "customSettings",
        categoryDisplayName: "Custom Settings",
        type: isDesigner ? "dropdown" : "string",
        description: isDesigner
          ? "Link this text field to a dropdown with mapFromGroup property for auto-population during Mid Year cycle"
          : "Link this text field to a dropdown for auto-population during Mid Year cycle",
      };

      if (isDesigner) {
        linkedToGroupMidYearConfig.choices = function (obj) {
          const survey = obj && obj.survey;
          if (!survey) return [{ value: "", text: "-- None --" }];

          const dropdownQuestions = survey
            .getAllQuestions()
            .filter((q) => q.getType() === "dropdown" && q.mapFromGroup)
            .map((q) => ({
              value: q.name,
              text: `${q.title || q.name} (Group: ${q.mapFromGroup})`,
            }));

          return [{ value: "", text: "-- None --" }, ...dropdownQuestions];
        };
        linkedToGroupMidYearConfig.visibleIndex = 3;
        linkedToGroupMidYearConfig.visibleIf = (obj) => obj && obj.getType && obj.getType() === "text";
      }

      Serializer.addProperty("question", linkedToGroupMidYearConfig);
    }

    // Register linkedToGroupYearEnd property for questions
    if (skipDuplicateCheck || !Serializer.findProperty("question", "linkedToGroupYearEnd")) {
      const linkedToGroupYearEndConfig = {
        name: "linkedToGroupYearEnd",
        displayName: "Linked Group (Year End)",
        category: "customSettings",
        categoryDisplayName: "Custom Settings",
        type: isDesigner ? "dropdown" : "string",
        description: isDesigner
          ? "Link this text field to a dropdown with mapFromGroup property for auto-population during Year End cycle"
          : "Link this text field to a dropdown for auto-population during Year End cycle",
      };

      if (isDesigner) {
        linkedToGroupYearEndConfig.choices = function (obj) {
          const survey = obj && obj.survey;
          if (!survey) return [{ value: "", text: "-- None --" }];

          const dropdownQuestions = survey
            .getAllQuestions()
            .filter((q) => q.getType() === "dropdown" && q.mapFromGroup)
            .map((q) => ({
              value: q.name,
              text: `${q.title || q.name} (Group: ${q.mapFromGroup})`,
            }));

          return [{ value: "", text: "-- None --" }, ...dropdownQuestions];
        };
        linkedToGroupYearEndConfig.visibleIndex = 4;
        linkedToGroupYearEndConfig.visibleIf = (obj) => obj && obj.getType && obj.getType() === "text";
      }

      Serializer.addProperty("question", linkedToGroupYearEndConfig);
    }
  } catch (error) {
    console.error("Error registering custom properties:", error);
    throw error;
  }
};

/**
 * Legacy function name for backward compatibility
 * @deprecated Use registerCustomPropertiesForRuntime instead
 */
export const setupCustomProperties = registerCustomPropertiesForRuntime;
