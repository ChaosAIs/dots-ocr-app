import axios from "axios";
import { loadingService } from "../loading/loadingService";
import APP_CONFIG from "../config/appConfig";
import { messageService } from "../message/messageService";
import AuthService from "../auth/authService";
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
 * Work for Identity Server 4 single sign on.
 * @returns Bearer token header string.
 */
const getAccessTokenHeader = () => {
  const oidc = JSON.parse(
    /**
     * Note: It is OIDC-Client library default storage.
     * The key format is defined by oidc-client library.
     * Note: Developer no needs to modify the session key.
     * */
    sessionStorage.getItem(`oidc.user:${APP_CONFIG.iamDomain}:${APP_CONFIG.clientId}`)
  );
  if (!!oidc && !!oidc.access_token && !oidc.expired) {
    return `Bearer ${oidc.access_token}`;
  } else return "";
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
          messageService.errorToast("It is Bad Request");
          break;
        case 401:
          messageService.errorToast("Unauthorized Access. Redirecting to login...");
          // Auto-redirect to login on unauthorized access
          handleUnauthorizedAccess();
          break;
        //case 404:
        //  messageService.errorToast("Required access endpoint is No Found");
        //  break;
        case 406:
          messageService.errorToast("Request Not Acceptable");
          break;
        case 403:
          messageService.errorToast("Access Forbidden. Redirecting to login...");
          // Auto-redirect to login on forbidden access
          handleUnauthorizedAccess();
          break;
        case 408:
          messageService.errorToast("Request is Timeout");
          break;
        default:
          messageService.errorToast("Request process failed.");
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
    // Create a new instance of AuthService to handle the redirect
    const authService = new AuthService();

    // Clear any stale authentication state
    authService.userManager.clearStaleState();

    // Store the current location for redirect after login
    const currentPath = window.location.pathname + window.location.search;
    localStorage.setItem("redirectUri", currentPath);

    console.log("Unauthorized access detected. Redirecting to login...");
    console.log("Current path stored for redirect:", currentPath);

    // Redirect to login
    authService.signinRedirect().catch((error) => {
      console.error("Error during signin redirect:", error);
      // Fallback: navigate to home page if signin redirect fails
      window.location.replace(APP_CONFIG.basePath);
    });
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
