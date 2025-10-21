/**
 * Request deduplication utility to prevent duplicate API calls
 * Especially useful in React StrictMode where effects run twice
 */

class RequestDeduplication {
  constructor() {
    this.pendingRequests = new Map();
    this.requestTimeouts = new Map();
  }

  /**
   * Create a unique key for a request
   * @param {string} method - HTTP method
   * @param {string} url - Request URL
   * @param {object} params - Request parameters
   * @returns {string} Unique request key
   */
  createRequestKey(method, url, params = {}) {
    const sortedParams = JSON.stringify(params, Object.keys(params).sort());
    return `${method.toUpperCase()}:${url}:${sortedParams}`;
  }

  /**
   * Execute a request with deduplication
   * @param {string} key - Unique request key
   * @param {Function} requestFn - Function that returns a promise for the request
   * @param {number} timeout - Timeout in ms to clear the cache (default: 1000ms)
   * @returns {Promise} The request promise
   */
  async execute(key, requestFn, timeout = 1000) {
    // If request is already pending, return the existing promise
    if (this.pendingRequests.has(key)) {
      return this.pendingRequests.get(key);
    }

    // Create and store the request promise
    const requestPromise = requestFn()
      .finally(() => {
        // Clean up after request completes
        this.pendingRequests.delete(key);
        if (this.requestTimeouts.has(key)) {
          clearTimeout(this.requestTimeouts.get(key));
          this.requestTimeouts.delete(key);
        }
      });

    this.pendingRequests.set(key, requestPromise);

    // Set a timeout to clean up the cache in case of hanging requests
    const timeoutId = setTimeout(() => {
      this.pendingRequests.delete(key);
      this.requestTimeouts.delete(key);
    }, timeout);

    this.requestTimeouts.set(key, timeoutId);

    return requestPromise;
  }

  /**
   * Clear all pending requests (useful for cleanup)
   */
  clear() {
    // Clear all timeouts
    this.requestTimeouts.forEach(timeoutId => clearTimeout(timeoutId));
    
    // Clear maps
    this.pendingRequests.clear();
    this.requestTimeouts.clear();
  }

  /**
   * Check if a request is currently pending
   * @param {string} key - Request key
   * @returns {boolean} True if request is pending
   */
  isPending(key) {
    return this.pendingRequests.has(key);
  }
}

// Export singleton instance
export const requestDeduplication = new RequestDeduplication();
export default requestDeduplication;
