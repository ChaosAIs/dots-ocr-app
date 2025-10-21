/**
 * Audit Utilities for parsing and comparing audit data
 * Provides helper functions for processing audit log data and identifying changes
 */

/**
 * Safely parse JSON string, return null if invalid
 * @param {string} jsonString - JSON string to parse
 * @returns {Object|null} Parsed object or null if invalid
 */
export const safeJsonParse = (jsonString) => {
  if (!jsonString || typeof jsonString !== "string") {
    return null;
  }

  try {
    return JSON.parse(jsonString);
  } catch (error) {
    console.warn("Failed to parse JSON:", jsonString, error);
    return null;
  }
};

/**
 * Create a hash of a string value for comparison
 * @param {string} str - String to hash
 * @returns {string} Hash value
 */
const createStringHash = (str) => {
  if (!str) return "";

  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return hash.toString();
};

/**
 * Compress a string by removing all whitespace and normalizing
 * @param {string} value - String value to compress
 * @returns {string} Compressed string
 */
const compressString = (value) => {
  if (typeof value !== "string") return value;

  // Remove all whitespace characters (spaces, tabs, newlines, etc.)
  return value.replace(/\s+/g, "").toLowerCase();
};

/**
 * Normalize a value for comparison with aggressive whitespace removal and hashing
 * @param {any} value - Value to normalize
 * @returns {any} Normalized value
 */
const normalizeValueForComparison = (value) => {
  if (value === null || value === undefined) {
    return value;
  }

  if (typeof value === "string") {
    // Compress string and create hash for comparison
    const compressed = compressString(value);
    return compressed ? createStringHash(compressed) : compressed;
  }

  if (Array.isArray(value)) {
    // For arrays, normalize each element and sort for consistent comparison
    return value.map(normalizeValueForComparison).sort((a, b) => {
      // Sort by string representation for consistent ordering
      const aStr = typeof a === "object" ? JSON.stringify(a) : String(a);
      const bStr = typeof b === "object" ? JSON.stringify(b) : String(b);
      return aStr.localeCompare(bStr);
    });
  }

  if (typeof value === "object") {
    // Recursively normalize object properties
    const normalized = {};
    Object.keys(value)
      .sort()
      .forEach((key) => {
        normalized[key] = normalizeValueForComparison(value[key]);
      });
    return normalized;
  }

  return value;
};

/**
 * Compare two values with normalization for whitespace and formatting
 * @param {any} originalValue - Original value
 * @param {any} currentValue - Current value
 * @returns {boolean} True if values are effectively identical after normalization
 */
const areValuesEffectivelyEqual = (originalValue, currentValue) => {
  // First normalize both values
  const normalizedOriginal = normalizeValueForComparison(originalValue);
  const normalizedCurrent = normalizeValueForComparison(currentValue);

  // Use JSON.stringify for deep comparison after normalization
  return JSON.stringify(normalizedOriginal) === JSON.stringify(normalizedCurrent);
};

/**
 * Compare two arrays item by item and return detailed changes
 * @param {Array} originalArray - Original array
 * @param {Array} currentArray - Current array
 * @param {string} fieldName - Name of the field being compared
 * @returns {Array} Array of detailed change objects
 */
const getArrayItemChanges = (originalArray, currentArray, fieldName) => {
  const changes = [];

  if (!Array.isArray(originalArray) && !Array.isArray(currentArray)) {
    return changes;
  }

  // Handle case where one is not an array
  if (!Array.isArray(originalArray) || !Array.isArray(currentArray)) {
    return [
      {
        field: fieldName,
        original: originalArray,
        current: currentArray,
        changeType: "modified",
      },
    ];
  }

  // If arrays are the same length, try to match items by position first
  if (originalArray.length === currentArray.length) {
    for (let i = 0; i < originalArray.length; i++) {
      const originalItem = originalArray[i];
      const currentItem = currentArray[i];

      if (!areValuesEffectivelyEqual(originalItem, currentItem)) {
        changes.push({
          field: `${fieldName}[${i}]`,
          original: originalItem,
          current: currentItem,
          changeType: "modified",
        });
      }
    }
    return changes;
  }

  // For different length arrays, use hash-based matching
  const originalMap = new Map();
  const currentMap = new Map();
  const usedCurrentIndices = new Set();
  const usedOriginalIndices = new Set();

  // Process original array
  originalArray.forEach((item, index) => {
    const hash = createStringHash(JSON.stringify(normalizeValueForComparison(item)));
    if (!originalMap.has(hash)) {
      originalMap.set(hash, []);
    }
    originalMap.get(hash).push({ item, index });
  });

  // Process current array
  currentArray.forEach((item, index) => {
    const hash = createStringHash(JSON.stringify(normalizeValueForComparison(item)));
    if (!currentMap.has(hash)) {
      currentMap.set(hash, []);
    }
    currentMap.get(hash).push({ item, index });
  });

  // Find matching items (no change)
  originalMap.forEach((originalItems, hash) => {
    if (currentMap.has(hash)) {
      const currentItems = currentMap.get(hash);
      const matchCount = Math.min(originalItems.length, currentItems.length);

      for (let i = 0; i < matchCount; i++) {
        usedOriginalIndices.add(originalItems[i].index);
        usedCurrentIndices.add(currentItems[i].index);
      }
    }
  });

  // Find items that exist in original but not in current (removed)
  originalArray.forEach((item, index) => {
    if (!usedOriginalIndices.has(index)) {
      changes.push({
        field: `${fieldName}[${index}]`,
        original: item,
        current: null,
        changeType: "removed",
      });
    }
  });

  // Find items that exist in current but not in original (added)
  currentArray.forEach((item, index) => {
    if (!usedCurrentIndices.has(index)) {
      changes.push({
        field: `${fieldName}[${index}]`,
        original: null,
        current: item,
        changeType: "added",
      });
    }
  });

  return changes;
};

/**
 * Compare two objects and return only the fields that have changed
 * @param {Object} originalObj - Original object
 * @param {Object} currentObj - Current object
 * @param {Object} fieldTitleMapping - Optional mapping of field names to user-friendly titles
 * @returns {Array} Array of change objects with field, original, and current values
 */
export const getChangedFields = (originalObj, currentObj, fieldTitleMapping = {}) => {
  const changes = [];

  if (!originalObj && !currentObj) {
    return changes;
  }

  // If one is null/undefined, treat as complete change
  if (!originalObj || !currentObj) {
    return [
      {
        field: "Complete Object",
        original: originalObj || "(empty)",
        current: currentObj || "(empty)",
        changeType: !originalObj ? "added" : "removed",
      },
    ];
  }

  // Get all unique keys from both objects
  const allKeys = new Set([...Object.keys(originalObj), ...Object.keys(currentObj)]);

  allKeys.forEach((key) => {
    const originalValue = originalObj[key];
    const currentValue = currentObj[key];

    // Skip if values are effectively identical after normalization
    if (areValuesEffectivelyEqual(originalValue, currentValue)) {
      return;
    }

    // Special handling for arrays - compare item by item
    if (Array.isArray(originalValue) || Array.isArray(currentValue)) {
      const arrayChanges = getArrayItemChanges(originalValue, currentValue, key);
      changes.push(...arrayChanges);
      return;
    }

    // Determine change type
    let changeType = "modified";
    if (originalValue === undefined || originalValue === null) {
      changeType = "added";
    } else if (currentValue === undefined || currentValue === null) {
      changeType = "removed";
    }

    // Use field title if available, otherwise use the original field name
    // Try multiple resolution strategies in order of preference
    let displayField = fieldTitleMapping[key];

    if (!displayField) {
      // Strategy 1: Try without "-Comment" suffix if present
      if (key.endsWith("-Comment")) {
        const baseKey = key.substring(0, key.length - "-Comment".length);
        displayField = fieldTitleMapping[baseKey];
      }
    }

    if (!displayField) {
      // Strategy 2: Handle array notation like "fieldName[0]" or "fieldName[0]-Comment"
      const arrayMatch = key.match(/^(.+)\[\d+\](-Comment)?$/);
      if (arrayMatch) {
        const baseFieldName = arrayMatch[1];
        const hasComment = arrayMatch[2];

        // Try to find mapping for the base field name (could be dynamic panel or template element)
        displayField = fieldTitleMapping[baseFieldName];

        // If not found and it has -Comment, try without -Comment
        if (!displayField && hasComment) {
          displayField = fieldTitleMapping[baseFieldName];
        }

        // Note: Hierarchical lookup for empty/missing titles is handled
        // during the mapping creation phase for better performance
      }
    }

    displayField = displayField || key;

    changes.push({
      field: displayField,
      original: originalValue,
      current: currentValue,
      changeType,
    });
  });

  return changes;
};

/**
 * Process audit detail record to extract meaningful field changes
 * @param {Object} auditDetail - Audit detail record from API
 * @param {Object} fieldTitleMapping - Optional mapping of field names to user-friendly titles
 * @returns {Object} Processed audit detail with parsed changes
 */
export const processAuditDetail = (auditDetail, fieldTitleMapping = {}) => {
  if (!auditDetail) {
    return null;
  }

  const originalJson = safeJsonParse(auditDetail.minCreatedOnOriginalValue);
  const currentJson = safeJsonParse(auditDetail.maxCreatedOnCurrentValue);

  // If both are valid JSON objects, compare them
  if (originalJson && currentJson && typeof originalJson === "object" && typeof currentJson === "object") {
    const changes = getChangedFields(originalJson, currentJson, fieldTitleMapping);

    return {
      ...auditDetail,
      hasJsonChanges: true,
      fieldChanges: changes,
      originalParsed: originalJson,
      currentParsed: currentJson,
    };
  }

  // If not JSON or parsing failed, treat as simple string comparison with normalization
  const hasStringChange = !areValuesEffectivelyEqual(auditDetail.minCreatedOnOriginalValue, auditDetail.maxCreatedOnCurrentValue);

  return {
    ...auditDetail,
    hasJsonChanges: false,
    fieldChanges: hasStringChange
      ? [
          {
            field: auditDetail.fieldTitle || auditDetail.fieldName || "Value",
            original: auditDetail.minCreatedOnOriginalValue,
            current: auditDetail.maxCreatedOnCurrentValue,
            changeType: "modified",
          },
        ]
      : [],
    originalParsed: null,
    currentParsed: null,
  };
};

/**
 * Format field value for display
 * @param {any} value - Value to format
 * @param {number} maxLength - Maximum length before truncation
 * @returns {string} Formatted value
 */
export const formatFieldValue = (value, maxLength = 100) => {
  if (value === null || value === undefined || value === "Loading...") {
    return "";
  }

  if (typeof value === "object") {
    const jsonString = JSON.stringify(value, null, 2);
    return jsonString.length > maxLength ? jsonString.substring(0, maxLength) + "..." : jsonString;
  }

  const stringValue = String(value);
  return stringValue.length > maxLength ? stringValue.substring(0, maxLength) + "..." : stringValue;
};

/**
 * Get change type severity for UI styling
 * @param {string} changeType - Type of change (added, removed, modified)
 * @returns {string} PrimeReact severity class
 */
export const getChangeTypeSeverity = (changeType) => {
  switch (changeType) {
    case "added":
      return "success";
    case "removed":
      return "danger";
    case "modified":
      return "warning";
    default:
      return "info";
  }
};

/**
 * Get change type icon
 * @param {string} changeType - Type of change (added, removed, modified)
 * @returns {string} PrimeReact icon class
 */
export const getChangeTypeIcon = (changeType) => {
  switch (changeType) {
    case "added":
      return "pi pi-plus-circle";
    case "removed":
      return "pi pi-minus-circle";
    case "modified":
      return "pi pi-pencil";
    default:
      return "pi pi-info-circle";
  }
};

/**
 * Replace placeholders in field titles with actual year values
 * @param {string} title - Title that may contain placeholders
 * @param {number} questionnaireYear - Questionnaire year for replacement
 * @returns {string} Title with placeholders replaced
 */
const replaceTitlePlaceholders = (title, questionnaireYear) => {
  if (!title || !questionnaireYear) {
    return title;
  }

  const currentYear = questionnaireYear;
  const previousYear = currentYear - 1;

  return title.replace(/{currentYear}/g, currentYear.toString()).replace(/{previousYear}/g, previousYear.toString());
};

/**
 * Extract field name to title mapping from questionnaire definition JSON
 * @param {Object} questionnaireDefinition - Questionnaire definition JSON object
 * @param {number} questionnaireYear - Questionnaire year for placeholder replacement
 * @returns {Object} Dictionary mapping field names to user-friendly titles
 */
export const extractFieldTitleMapping = (questionnaireDefinition, questionnaireYear = null) => {
  const fieldTitleMapping = {};

  if (!questionnaireDefinition || !questionnaireDefinition.pages) {
    return fieldTitleMapping;
  }

  /**
   * Recursively process Survey.js elements to extract field names and titles
   * @param {Array} elements - Array of Survey.js elements
   * @param {string} parentTitle - Title of parent panel for context
   */
  const processElements = (elements, parentTitle = null) => {
    if (!Array.isArray(elements)) return;

    elements.forEach((element) => {
      const name = element.name;
      const title = element.title;
      const elementType = element.type;

      // Add mapping if both name and title exist
      if (name && title) {
        // Replace placeholders in title if questionnaire year is available
        const processedTitle = replaceTitlePlaceholders(title, questionnaireYear);

        // Include parent panel title for better context if available
        let finalTitle = processedTitle;
        if (parentTitle) {
          finalTitle = combineUniqueTitle(parentTitle, processedTitle);
        }

        // For dynamic panels with empty/meaningless titles, use hierarchical lookup
        if (elementType === "paneldynamic" && (!processedTitle || !processedTitle.trim() || processedTitle.trim() === " ")) {
          const hierarchicalTitle = resolveFieldTitleHierarchically(name, { pages: questionnaireDefinition.pages, year: questionnaireYear });
          if (hierarchicalTitle) {
            finalTitle = hierarchicalTitle;
          }
        }

        // Add mapping for the original field name
        fieldTitleMapping[name] = finalTitle;

        // Also add mapping for field names with "-Comment" suffix
        // This handles cases like "question23-Comment" which should map to "question23"
        const commentFieldName = name + "-Comment";
        fieldTitleMapping[commentFieldName] = finalTitle;
      }

      // Recursively process nested elements in regular panels and pages
      if (element.elements) {
        // For panels, pass the panel title as parent context
        let contextTitle = parentTitle;
        if ((elementType === "panel" || elementType === "paneldynamic") && title) {
          const panelTitle = replaceTitlePlaceholders(title, questionnaireYear);
          contextTitle = parentTitle ? combineUniqueTitle(parentTitle, panelTitle) : panelTitle;
        }

        processElements(element.elements, contextTitle);
      }

      // Recursively process templateElements for paneldynamic elements
      if (elementType === "paneldynamic" && element.templateElements) {
        // For dynamic panels with empty/meaningless titles, use hierarchical lookup for the panel itself
        let panelTitle = replaceTitlePlaceholders(title, questionnaireYear);
        if (!panelTitle || !panelTitle.trim() || panelTitle.trim() === " ") {
          const hierarchicalTitle = resolveFieldTitleHierarchically(name, { pages: questionnaireDefinition.pages, year: questionnaireYear });
          if (hierarchicalTitle) {
            panelTitle = hierarchicalTitle;
          }
        }

        // For template elements, we want to use the parent context (not the dynamic panel's title)
        // because template elements inherit the meaningful parent context
        const contextTitle = parentTitle || panelTitle;

        processElements(element.templateElements, contextTitle);
      }
    });
  };

  // Process all pages
  processElements(questionnaireDefinition.pages);

  return fieldTitleMapping;
};

/**
 * Resolve field title by traversing up the form hierarchy to find meaningful parent titles
 * @param {string} fieldName - Field name to resolve.
 * @param {Object} questionnaireDefinition - The questionnaire definition JSON
 * @returns {string|null} Hierarchical title or null if not found
 */
const resolveFieldTitleHierarchically = (fieldName, questionnaireDefinition) => {
  try {
    if (!questionnaireDefinition || !questionnaireDefinition.pages) {
      return null;
    }

    return findFieldInHierarchy(questionnaireDefinition.pages, fieldName, questionnaireDefinition.year);
  } catch (error) {
    console.error(`Error resolving field title hierarchically for field: ${fieldName}`, error);
    return null;
  }
};

/**
 * Recursively search for a field in the form hierarchy and return its meaningful parent title
 * @param {Array} elements - Array of Survey.js elements
 * @param {string} targetFieldName - Field name to find
 * @param {number} questionnaireYear - Questionnaire year for placeholder replacement
 * @param {Array} parentTitles - Stack of parent titles for context
 * @returns {string|null} Meaningful parent title or null if not found
 */
const findFieldInHierarchy = (elements, targetFieldName, questionnaireYear, parentTitles = []) => {
  if (!elements || !Array.isArray(elements)) return null;

  for (const element of elements) {
    const name = element.name;
    const title = element.title;
    const elementType = element.type;

    // Add current element's title to parent stack if it's meaningful
    const currentParentTitles = [...parentTitles];
    if (title && title.trim() && title.trim() !== " ") {
      const processedTitle = replaceTitlePlaceholders(title, questionnaireYear);
      currentParentTitles.push(processedTitle);
    }

    // Check if this is the target field
    if (name === targetFieldName) {
      // Return the most meaningful parent title (last non-empty one)
      for (let i = currentParentTitles.length - 1; i >= 0; i--) {
        if (currentParentTitles[i] && currentParentTitles[i].trim()) {
          return currentParentTitles[i];
        }
      }
    }

    // Recursively search in nested elements
    if (element.elements) {
      const result = findFieldInHierarchy(element.elements, targetFieldName, questionnaireYear, currentParentTitles);
      if (result) return result;
    }

    // Recursively search in templateElements for paneldynamic
    if (elementType === "paneldynamic" && element.templateElements) {
      const result = findFieldInHierarchy(element.templateElements, targetFieldName, questionnaireYear, currentParentTitles);
      if (result) return result;
    }
  }

  return null;
};

/**
 * Combine parent and child titles while removing duplicate segments
 * @param {string} parentTitle - Parent title (may contain multiple segments separated by " - ")
 * @param {string} childTitle - Child title to append
 * @returns {string} Combined title with unique segments
 */
const combineUniqueTitle = (parentTitle, childTitle) => {
  if (!parentTitle) return childTitle;
  if (!childTitle) return parentTitle;

  // Split both titles into segments
  const parentSegments = parentTitle
    .split(" - ")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);

  const childSegments = childTitle
    .split(" - ")
    .map((s) => s.trim())
    .filter((s) => s.length > 0);

  // Combine and deduplicate segments (case-insensitive comparison)
  const uniqueSegments = [];
  const seenSegments = new Set();

  // Add parent segments first
  parentSegments.forEach((segment) => {
    const lowerSegment = segment.toLowerCase();
    if (!seenSegments.has(lowerSegment)) {
      uniqueSegments.push(segment);
      seenSegments.add(lowerSegment);
    }
  });

  // Add child segments that aren't duplicates
  childSegments.forEach((segment) => {
    const lowerSegment = segment.toLowerCase();
    if (!seenSegments.has(lowerSegment)) {
      uniqueSegments.push(segment);
      seenSegments.add(lowerSegment);
    }
  });

  return uniqueSegments.join(" - ");
};

/**
 * Process all audit details to extract meaningful changes
 * @param {Array} auditDetails - Array of audit detail records
 * @param {Object} fieldTitleMapping - Optional mapping of field names to user-friendly titles
 * @returns {Array} Array of processed audit details with changes
 */
export const processAllAuditDetails = (auditDetails, fieldTitleMapping = {}) => {
  if (!Array.isArray(auditDetails)) {
    return [];
  }

  return auditDetails
    .map((detail) => processAuditDetail(detail, fieldTitleMapping))
    .filter((detail) => detail && detail.fieldChanges && detail.fieldChanges.length > 0);
};
