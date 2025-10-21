import { Subject } from "rxjs";

/** Work for subscribe http request/response processing status crossing components
 *  Work for global progress spinner.
 */
const loadingSubject = new Subject();
/**
 * Global variable to keep the couting of http request / http respone.
 */
var loadingCount = 0;

/**
 * Global flag to control whether loading interceptor should be active
 * When disabled, HTTP requests won't trigger the loading spinner
 */
var isLoadingInterceptorEnabled = true;

/**
 * Set of disabled contexts - when any of these contexts are active,
 * the loading interceptor will be disabled
 */
const disabledContexts = new Set();

function timmerBeforeSend(inputCount) {
  if (inputCount === 0) {
    loadingCount = 0;
  }

  loadingCount = loadingCount + inputCount;

  if (loadingCount <= 0) {
    //
    // Immediately notify system to hide the progress spinner.
    //
    loadingCount = 0;
    return new Promise((resolve) => {
      resolve("hide");
    });
  } else {
    //
    // Reduced delay to prevent BlockUI issues while still preventing spinner flash
    // for very fast HTTP requests
    //
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve("show");
      }, 1000); 
    });
  }
}

/**
 * Check if loading interceptor should be active
 * @returns {boolean} true if interceptor should be active, false otherwise
 */
function shouldApplyLoadingInterceptor() {
  // If globally disabled, return false
  if (!isLoadingInterceptorEnabled) {
    return false;
  }

  // If any disabled context is active, return false
  if (disabledContexts.size > 0) {
    return false;
  }

  return true;
}

/**
 * Called by LoadingService. Work for show/hide porgress spinner when http request sent and http response received.
 * @param {*} inputCount for http request sent, set inputCount = 1; for http response received, set inputCount = -1; any http error happened, set inputCount = 0;
 */
async function notifyLoading(inputCount) {
  try {
    // Check if loading interceptor should be applied
    if (!shouldApplyLoadingInterceptor()) {
      return; // Skip loading notification if interceptor is disabled
    }

    const result = await timmerBeforeSend(inputCount);
    if (result === "hide") {
      if (loadingCount <= 0) {
        loadingSubject.next(false); // notify to hide spinner.
      }
    } else if (result === "show") {
      if (loadingCount > 0) {
        loadingSubject.next(true); // notify to show spinner.
      }
    }
  } catch (error) {
    console.error("Error in notifyLoading:", error);
    // On error, ensure loading is hidden
    loadingSubject.next(false);
  }
}

/***
 * Note: It is single tone service.
 * Reference: https://jasonwatmore.com/post/2019/02/13/react-rxjs-communicating-between-components-with-observable-subject#:~:text=React%20%2B%20RxJS%20App%20Component%20that,divs%20in%20the%20render%20method.
 */
export const loadingService = {
  /**
   * when http request sent, notify observer.
   */
  httpRequestSent: () => {
    try {
      notifyLoading(1);
    } catch (error) {
      console.error("Error in httpRequestSent:", error);
    }
  },

  /**
   * when http response received, notify observer.
   */
  httpResponseReceived: () => {
    try {
      notifyLoading(-1);
    } catch (error) {
      console.error("Error in httpResponseReceived:", error);
    }
  },

  /** Any error happened in any http request or http response, notify observer */
  error: () => {
    try {
      notifyLoading(0);
    } catch (error) {
      console.error("Error in error handler:", error);
      // Force hide loading on error
      loadingSubject.next(false);
    }
  },

  /**
   * Return observaber subject to receiver.
   * @returns observer.
   */
  get: () => {
    return loadingSubject.asObservable();
  },

  /**
   * Enable or disable the loading interceptor globally
   * @param {boolean} enabled - true to enable, false to disable
   */
  setInterceptorEnabled: (enabled) => {
    isLoadingInterceptorEnabled = enabled;
    console.log(`Loading interceptor ${enabled ? 'enabled' : 'disabled'}`);
  },

  /**
   * Check if the loading interceptor is currently enabled
   * @returns {boolean} true if enabled, false if disabled
   */
  isInterceptorEnabled: () => {
    return isLoadingInterceptorEnabled;
  },

  /**
   * Add a context to the disabled contexts set
   * When any disabled context is active, the loading interceptor will be disabled
   * @param {string} context - context identifier (e.g., 'survey', 'form', etc.)
   */
  disableForContext: (context) => {
    disabledContexts.add(context);
    console.log(`Loading interceptor disabled for context: ${context}`);
  },

  /**
   * Remove a context from the disabled contexts set
   * @param {string} context - context identifier to re-enable
   */
  enableForContext: (context) => {
    disabledContexts.delete(context);
    console.log(`Loading interceptor enabled for context: ${context}`);
  },

  /**
   * Clear all disabled contexts
   */
  clearDisabledContexts: () => {
    disabledContexts.clear();
    console.log('All disabled contexts cleared');
  },

  /**
   * Get the current set of disabled contexts
   * @returns {Set<string>} set of disabled context identifiers
   */
  getDisabledContexts: () => {
    return new Set(disabledContexts);
  },

  /**
   * Check if a specific context is disabled
   * @param {string} context - context identifier to check
   * @returns {boolean} true if context is disabled, false otherwise
   */
  isContextDisabled: (context) => {
    return disabledContexts.has(context);
  }
};
