import { setLicenseKey } from 'survey-core';

/**
 * Configure Survey.js commercial license key
 * This should be called before initializing any Survey.js components
 */
export const configureSurveyJSLicense = () => {
  const licenseKey = process.env.REACT_APP_SURVEYJS_LICENSE_KEY;
  
  if (licenseKey && licenseKey !== 'YOUR_LICENSE_KEY_HERE') {
    try {
      setLicenseKey(licenseKey);
      console.log('Survey.js commercial license applied successfully');
    } catch (error) {
      console.error('Failed to apply Survey.js license key:', error);
    }
  } else {
    console.warn('Survey.js license key not configured. Please set REACT_APP_SURVEYJS_LICENSE_KEY in your environment file.');
  }
};
