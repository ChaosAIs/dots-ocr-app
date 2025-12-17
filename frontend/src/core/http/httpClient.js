import axios from "axios";
import { loadingService } from "../loading/loadingService";
import APP_CONFIG from "../config/appConfig";
import { messageService } from "../message/messageService";
import i18n from "../config/i18nConfig";
// import AuthService from "../auth/authService"; // Temporarily disabled IAM SSO
import TempAuthService from "../auth/tempAuthService"; // Using temporary auth service
/**
 * It is instance of axios Http client, which works for http call.
 * Corporate with loading.jsx for a global progress bar process.
 * Corporate with loadingReducer.js
 *
 * Note: It is single tone.
 *
 * Reference: https://sumn2u.medium.com/global-progress-bar-on-api-s-call-in-react-5133f818d12a
 *
 */
const http = axios.create({
  timeout: 600000, // 10 minutes (600 seconds) for regular API calls
  maxContentLength: 52428800, // 50MB
  maxBodyLength: 52428800, // 50MB
});

/**
 * Corporate with http const to inject access_token to http request header.
 * Checks for real authentication token first (localStorage), then falls back to temporary auth (sessionStorage).
 * @returns Bearer token header string.
 */
const getAccessTokenHeader = () => {
  // First, check for real authentication token in localStorage
  // Note: authService.js stores the token with key 'access_token'
  const realToken = localStorage.getItem('access_token');
  if (realToken) {
    return `Bearer ${realToken}`;
  }

  // Fall back to temporary auth service
  const tempAuthService = new TempAuthService();
  const user = tempAuthService.getUser();
  if (user && user.accessToken) {
    return `Bearer ${user.accessToken}`;
  }

  return "";
};

/** Get current language codes. */
export const getLanguageCode = () => {
  //
  // corporate with Redux/store.js
  //
  var languageCode = localStorage.getItem(`${APP_CONFIG.clientId}_CurrentLanguage`);

  if (languageCode && languageCode.length > 0) return languageCode;
  else return "en";
};

http.interceptors.request.use(
  (request) => {
    // Notify loading service to show progress spinner.
    loadingService.httpRequestSent();
    const tokenHeader = getAccessTokenHeader();
    if (tokenHeader && tokenHeader.length > 0) {
      request.headers["authorization"] = tokenHeader;
    }
    const languageCode = getLanguageCode();
    //
    // Corporate with web api project middleware called "RequestCultureMiddleware.cs"
    //
    request.headers["currentLanguage"] = languageCode;
    return request;
  },
  (error) => {
    loadingService.error();
    // Notify loading service to hide progress spinner.
    return Promise.reject(error);
  }
);

http.interceptors.response.use(
  (response) => {
    // Notify loding service to hide progress spinner.
    loadingService.httpResponseReceived();
    return response;
  },
  (error) => {
    // Notify loading service to hide progress spinner.
    loadingService.error();

    if (error.response) {
      switch (error.response.status) {
        case 400:
          messageService.errorToast(i18n.t("HttpClient.BadRequest"));
          break;
        case 401:
          messageService.errorToast(i18n.t("HttpClient.Unauthorized"));
          // Auto-redirect to login on unauthorized access
          handleUnauthorizedAccess();
          break;
        //case 404:
        //  messageService.errorToast("Required access endpoint is No Found");
        //  break;
        case 406:
          messageService.errorToast(i18n.t("HttpClient.NotAcceptable"));
          break;
        case 403:
          messageService.errorToast(i18n.t("HttpClient.Forbidden"));
          // Auto-redirect to login on forbidden access
          handleUnauthorizedAccess();
          break;
        case 408:
          messageService.errorToast(i18n.t("HttpClient.Timeout"));
          break;
        default:
          messageService.errorToast(i18n.t("HttpClient.RequestFailed"));
          break;
      }
    }
    return Promise.reject(error);
  }
);

/**
 * Handle unauthorized access by redirecting to login
 * This function ensures consistent behavior across the entire application
 */
const handleUnauthorizedAccess = () => {
  try {
    // Using temporary auth service
    const authService = new TempAuthService();

    // Clear any stale authentication state
    authService.clearAuthState();

    // Store the current location for redirect after login
    const currentPath = window.location.pathname + window.location.search;
    localStorage.setItem("redirectUri", currentPath);

    console.log("Unauthorized access detected. For temporary auth, allowing access...");
    console.log("Current path stored for redirect:", currentPath);

    // For temporary auth, just log and continue (no redirect needed)
    // In production with IAM, this would redirect to login
  } catch (error) {
    console.error("Error handling unauthorized access:", error);
    // Fallback: navigate to home page
    window.location.replace(APP_CONFIG.basePath);
  }
};

/**
 * Create a specialized HTTP client for file upload operations with extended timeout
 * @returns {AxiosInstance} Axios instance configured for file uploads
 */
export const createFileUploadClient = () => {
  const fileUploadClient = axios.create({
    timeout: 600000, // 10 minutes (600 seconds) for file upload operations
    maxContentLength: 52428800, // 50MB
    maxBodyLength: 52428800, // 50MB
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });

  // Apply the same interceptors as the main http client
  fileUploadClient.interceptors.request.use(
    (request) => {
      loadingService.httpRequestSent();
      const tokenHeader = getAccessTokenHeader();
      if (tokenHeader && tokenHeader.length > 0) {
        request.headers["authorization"] = tokenHeader;
      }
      const languageCode = getLanguageCode();
      request.headers["currentLanguage"] = languageCode;
      return request;
    },
    (error) => {
      loadingService.error();
      return Promise.reject(error);
    }
  );

  fileUploadClient.interceptors.response.use(
    (response) => {
      loadingService.httpResponseReceived();
      return response;
    },
    (error) => {
      loadingService.error();
      if (error.response) {
        switch (error.response.status) {
          case 400:
            messageService.errorToast("It is Bad Request");
            break;
          case 401:
            messageService.errorToast("Unauthorized Access. Redirecting to login...");
            handleUnauthorizedAccess();
            break;
          case 406:
            messageService.errorToast("Request Not Acceptable");
            break;
          case 403:
            messageService.errorToast("Access Forbidden");
            break;
          case 500:
            messageService.errorToast("Internal Server Error");
            break;
          case 408:
            messageService.errorToast("Request Timeout - Please try again");
            break;
          default:
            if (error.code === "ECONNABORTED") {
              messageService.errorToast("Request timeout - Please try again with a smaller file or check your connection");
            }
            break;
        }
      } else if (error.code === "ECONNABORTED") {
        messageService.errorToast("Request timeout - Please try again with a smaller file or check your connection");
      }
      return Promise.reject(error);
    }
  );

  return fileUploadClient;
};

export default http;
