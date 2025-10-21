import { useEffect } from 'react';
import { loadingService } from '../loadingService';

/**
 * Custom hook to control loading interceptor for a specific context
 * @param {string} context - The context identifier (e.g., 'survey', 'form', etc.)
 * @param {boolean} disabled - Whether to disable loading for this context (default: true)
 * @returns {object} Object with methods to control loading interceptor
 */
export const useLoadingControl = (context, disabled = true) => {
  useEffect(() => {
    if (disabled) {
      loadingService.disableForContext(context);
    } else {
      loadingService.enableForContext(context);
    }

    // Cleanup: re-enable when component unmounts
    return () => {
      if (disabled) {
        loadingService.enableForContext(context);
      }
    };
  }, [context, disabled]);

  return {
    // Manual control methods
    disable: () => loadingService.disableForContext(context),
    enable: () => loadingService.enableForContext(context),
    isDisabled: () => loadingService.isContextDisabled(context),
    
    // Global control methods
    setGlobalEnabled: (enabled) => loadingService.setInterceptorEnabled(enabled),
    isGlobalEnabled: () => loadingService.isInterceptorEnabled(),
    
    // Context management
    clearAllDisabled: () => loadingService.clearDisabledContexts(),
    getDisabledContexts: () => loadingService.getDisabledContexts()
  };
};

/**
 * Hook to globally enable/disable loading interceptor
 * @param {boolean} enabled - Whether loading interceptor should be enabled
 * @returns {object} Object with methods to control global loading interceptor
 */
export const useGlobalLoadingControl = (enabled = true) => {
  useEffect(() => {
    loadingService.setInterceptorEnabled(enabled);
    
    // Cleanup: restore to enabled state when component unmounts
    return () => {
      loadingService.setInterceptorEnabled(true);
    };
  }, [enabled]);

  return {
    setEnabled: (enabled) => loadingService.setInterceptorEnabled(enabled),
    isEnabled: () => loadingService.isInterceptorEnabled(),
    disable: () => loadingService.setInterceptorEnabled(false),
    enable: () => loadingService.setInterceptorEnabled(true)
  };
};
