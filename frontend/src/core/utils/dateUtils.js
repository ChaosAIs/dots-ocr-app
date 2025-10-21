/**
 * Date utility functions for formatting and manipulating dates
 */

/**
 * Parse date value and handle timezone conversion
 * @param {string|Date} dateValue - The date value to parse
 * @returns {Date|null} Parsed date object or null if invalid
 */
const parseDate = (dateValue) => {
  if (!dateValue) return null;
  
  try {
    let date;
    
    if (typeof dateValue === 'string') {
      // If the string doesn't end with 'Z' and doesn't have timezone info, assume it's UTC
      if (!dateValue.includes('Z') && !dateValue.includes('+') && !dateValue.includes('-', 10)) {
        date = new Date(dateValue + 'Z'); // Add Z to indicate UTC
      } else {
        date = new Date(dateValue);
      }
    } else {
      date = new Date(dateValue);
    }
    
    return isNaN(date.getTime()) ? null : date;
  } catch (error) {
    console.error('Error parsing date:', dateValue, error);
    return null;
  }
};

/**
 * Format a date string or Date object to a readable date and time format
 * @param {string|Date} dateValue - The date value to format
 * @param {Object} options - Formatting options
 * @param {boolean} options.includeTime - Whether to include time (default: true)
 * @param {boolean} options.includeSeconds - Whether to include seconds (default: false)
 * @param {string} options.locale - Locale for formatting (default: 'en-US')
 * @returns {string} Formatted date string
 */
export const formatDateTime = (dateValue, options = {}) => {
  if (!dateValue) return '';

  const {
    includeTime = true,
    includeSeconds = false,
    locale = 'en-US'
  } = options;

  const date = parseDate(dateValue);
  if (!date) return 'Invalid Date';

  try {
    const dateOptions = {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone
    };

    if (includeTime) {
      dateOptions.hour = 'numeric';
      dateOptions.minute = '2-digit';
      dateOptions.timeZoneName = 'short';
      
      if (includeSeconds) {
        dateOptions.second = '2-digit';
      }
    }

    return date.toLocaleString(locale, dateOptions);
  } catch (error) {
    console.error('Error formatting date:', dateValue, error);
    return 'Invalid Date';
  }
};

/**
 * Format a date string or Date object to a readable date format (no time)
 * @param {string|Date} dateValue - The date value to format
 * @param {string} locale - Locale for formatting (default: 'en-US')
 * @returns {string} Formatted date string
 */
export const formatDate = (dateValue, locale = 'en-US') => {
  if (!dateValue) return '';

  const date = parseDate(dateValue);
  if (!date) return 'Invalid Date';

  try {
    const dateOptions = {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone
    };

    return date.toLocaleDateString(locale, dateOptions);
  } catch (error) {
    console.error('Error formatting date:', dateValue, error);
    return 'Invalid Date';
  }
};

/**
 * Format a date string or Date object to a readable time format (no date)
 * @param {string|Date} dateValue - The date value to format
 * @param {boolean} includeSeconds - Whether to include seconds (default: false)
 * @param {string} locale - Locale for formatting (default: 'en-US')
 * @returns {string} Formatted time string
 */
export const formatTime = (dateValue, includeSeconds = false, locale = 'en-US') => {
  if (!dateValue) return '';

  const date = parseDate(dateValue);
  if (!date) return 'Invalid Time';

  try {
    const timeOptions = {
      hour: 'numeric',
      minute: '2-digit',
      timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      timeZoneName: 'short'
    };

    if (includeSeconds) {
      timeOptions.second = '2-digit';
    }

    return date.toLocaleTimeString(locale, timeOptions);
  } catch (error) {
    console.error('Error formatting time:', error);
    return 'Invalid Time';
  }
};

/**
 * Get relative time string (e.g., "2 hours ago", "3 days ago")
 * @param {string|Date} dateValue - The date value to compare
 * @returns {string} Relative time string
 */
export const getRelativeTime = (dateValue) => {
  if (!dateValue) return '';

  const date = parseDate(dateValue);
  if (!date) return 'Invalid Date';

  try {
    const now = new Date();
    const diffInSeconds = Math.floor((now - date) / 1000);

    if (diffInSeconds < 60) {
      return 'Just now';
    }

    const diffInMinutes = Math.floor(diffInSeconds / 60);
    if (diffInMinutes < 60) {
      return `${diffInMinutes} minute${diffInMinutes !== 1 ? 's' : ''} ago`;
    }

    const diffInHours = Math.floor(diffInMinutes / 60);
    if (diffInHours < 24) {
      return `${diffInHours} hour${diffInHours !== 1 ? 's' : ''} ago`;
    }

    const diffInDays = Math.floor(diffInHours / 24);
    if (diffInDays < 30) {
      return `${diffInDays} day${diffInDays !== 1 ? 's' : ''} ago`;
    }

    const diffInMonths = Math.floor(diffInDays / 30);
    if (diffInMonths < 12) {
      return `${diffInMonths} month${diffInMonths !== 1 ? 's' : ''} ago`;
    }

    const diffInYears = Math.floor(diffInMonths / 12);
    return `${diffInYears} year${diffInYears !== 1 ? 's' : ''} ago`;
  } catch (error) {
    console.error('Error calculating relative time:', error);
    return '';
  }
};

/**
 * Helper to check if a date matches a reference date in user's timezone
 * @param {Date} date - The date to check
 * @param {Date} referenceDate - The reference date (e.g., today or yesterday)
 * @returns {boolean}
 */
const isSameDayInUserTZ = (date, referenceDate) => {
  try {
    const userTimeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    const dateInUserTZ = new Date(date.toLocaleString('en-US', { timeZone: userTimeZone }));
    const refInUserTZ = new Date(referenceDate.toLocaleString('en-US', { timeZone: userTimeZone }));
    return (
      dateInUserTZ.getDate() === refInUserTZ.getDate() &&
      dateInUserTZ.getMonth() === refInUserTZ.getMonth() &&
      dateInUserTZ.getFullYear() === refInUserTZ.getFullYear()
    );
  } catch (error) {
    console.error('Error comparing dates in user timezone:', error);
    return false;
  }
};

/**
 * Check if a date is today (in user's timezone)
 * @param {string|Date} dateValue
 * @returns {boolean}
 */
export const isToday = (dateValue) => {
  const date = parseDate(dateValue);
  if (!date) return false;
  return isSameDayInUserTZ(date, new Date());
};

/**
 * Check if a date is yesterday (in user's timezone)
 * @param {string|Date} dateValue
 * @returns {boolean}
 */
export const isYesterday = (dateValue) => {
  const date = parseDate(dateValue);
  if (!date) return false;
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  return isSameDayInUserTZ(date, yesterday);
};

/**
 * Format date for audit display with relative time if recent
 * @param {string|Date} dateValue - The date value to format
 * @returns {string} Formatted date string with relative time if applicable
 */
export const formatAuditDate = (dateValue) => {
  if (!dateValue) return 'N/A';

  const date = parseDate(dateValue);
  if (!date) return 'Invalid Date';

  try {
    const now = new Date();
    const diffInHours = Math.floor((now - date) / (1000 * 60 * 60));

    // Show relative time for recent dates (within 24 hours)
    if (diffInHours < 24) {
      const relativeTime = getRelativeTime(dateValue);
      const formattedDate = formatDateTime(dateValue);
      return `${relativeTime} (${formattedDate})`;
    }

    // Show formatted date for older dates
    return formatDateTime(dateValue);
  } catch (error) {
    console.error('Error formatting audit date:', error);
    return 'Invalid Date';
  }
};

/**
 * Reusable DataTable column template for date/time fields.
 * @param {object} rowData - The row data object.
 * @param {string} fieldName - The field name to extract the date from.
 * @param {object} options - Formatting options for formatDateTime.
 * @returns {string}
 */
export const columnDateTimeFormatTemplate = (
  rowData,
  fieldName,
  options = { includeTime: true, includeSeconds: false }
) => {
  const value = rowData && rowData[fieldName];
  if (!value) return "-";
  return formatDateTime(value, options);
};
