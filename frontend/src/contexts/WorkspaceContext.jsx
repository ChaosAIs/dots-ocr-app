import React, { createContext, useContext, useState, useCallback, useEffect, useRef } from "react";
import workspaceService from "../services/workspaceService";
import { useAuth } from "../core/auth/components/authProvider";

/**
 * WorkspaceContext - Centralized state management for workspaces
 *
 * This context provides:
 * - Current selected workspace
 * - List of all workspaces
 * - Loading states
 * - Methods to switch/create/update/delete workspaces
 * - Event subscription system for workspace changes
 */

const WorkspaceContext = createContext(null);

// Event types for workspace changes
export const WorkspaceEvents = {
  WORKSPACE_SELECTED: "workspace_selected",
  WORKSPACE_CREATED: "workspace_created",
  WORKSPACE_UPDATED: "workspace_updated",
  WORKSPACE_DELETED: "workspace_deleted",
  WORKSPACES_LOADED: "workspaces_loaded",
  DOCUMENTS_REFRESH_NEEDED: "documents_refresh_needed",
};

export const WorkspaceProvider = ({ children }) => {
  const { user, isAuthenticated } = useAuth();

  // State
  const [workspaces, setWorkspaces] = useState([]);
  const [currentWorkspace, setCurrentWorkspace] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Event subscribers
  const subscribersRef = useRef(new Map());
  const subscriberIdRef = useRef(0);

  /**
   * Subscribe to workspace events
   * @param {string} eventType - Event type from WorkspaceEvents
   * @param {Function} callback - Callback function
   * @returns {Function} Unsubscribe function
   */
  const subscribe = useCallback((eventType, callback) => {
    const id = ++subscriberIdRef.current;

    if (!subscribersRef.current.has(eventType)) {
      subscribersRef.current.set(eventType, new Map());
    }

    subscribersRef.current.get(eventType).set(id, callback);
    console.log(`📢 WorkspaceContext: Subscribed to ${eventType} (id: ${id})`);

    // Return unsubscribe function
    return () => {
      const eventSubscribers = subscribersRef.current.get(eventType);
      if (eventSubscribers) {
        eventSubscribers.delete(id);
        console.log(`📢 WorkspaceContext: Unsubscribed from ${eventType} (id: ${id})`);
      }
    };
  }, []);

  /**
   * Emit an event to all subscribers
   * @param {string} eventType - Event type
   * @param {any} data - Event data
   */
  const emit = useCallback((eventType, data) => {
    console.log(`📢 WorkspaceContext: Emitting ${eventType}`, data);
    const eventSubscribers = subscribersRef.current.get(eventType);
    if (eventSubscribers) {
      eventSubscribers.forEach((callback) => {
        try {
          callback(data);
        } catch (err) {
          console.error(`Error in ${eventType} subscriber:`, err);
        }
      });
    }
  }, []);

  /**
   * Load workspaces from API
   */
  const loadWorkspaces = useCallback(async () => {
    if (!isAuthenticated) {
      console.log("📁 WorkspaceContext: Not authenticated, skipping workspace load");
      setWorkspaces([]);
      setCurrentWorkspace(null);
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);
      console.log("📁 WorkspaceContext: Loading workspaces...");

      const data = await workspaceService.getWorkspaces(true);
      console.log("📁 WorkspaceContext: Loaded workspaces:", data);

      if (Array.isArray(data)) {
        setWorkspaces(data);

        // Auto-select default workspace if none selected
        if (!currentWorkspace && data.length > 0) {
          const defaultWs = data.find(ws => ws.is_default) || data[0];
          console.log("📁 WorkspaceContext: Auto-selecting default workspace:", defaultWs?.name);
          setCurrentWorkspace(defaultWs);
          emit(WorkspaceEvents.WORKSPACE_SELECTED, defaultWs);
        } else if (currentWorkspace) {
          // Update current workspace with fresh data
          const updatedCurrent = data.find(ws => ws.id === currentWorkspace.id);
          if (updatedCurrent) {
            setCurrentWorkspace(updatedCurrent);
          }
        }

        emit(WorkspaceEvents.WORKSPACES_LOADED, data);
      }
    } catch (err) {
      console.error("📁 WorkspaceContext: Error loading workspaces:", err);
      setError(err.message || "Failed to load workspaces");
    } finally {
      setLoading(false);
    }
  }, [isAuthenticated, currentWorkspace, emit]);

  /**
   * Select a workspace
   * @param {Object} workspace - Workspace to select
   */
  const selectWorkspace = useCallback((workspace) => {
    console.log("📁 WorkspaceContext: Selecting workspace:", workspace?.name, "ID:", workspace?.id);
    setCurrentWorkspace(workspace);
    emit(WorkspaceEvents.WORKSPACE_SELECTED, workspace);
    emit(WorkspaceEvents.DOCUMENTS_REFRESH_NEEDED, { workspaceId: workspace?.id });
  }, [emit]);

  /**
   * Create a new workspace
   * @param {Object} workspaceData - Workspace data
   * @returns {Object} Created workspace
   */
  const createWorkspace = useCallback(async (workspaceData) => {
    try {
      const newWorkspace = await workspaceService.createWorkspace(workspaceData);
      console.log("📁 WorkspaceContext: Created workspace:", newWorkspace);

      setWorkspaces(prev => [...prev, newWorkspace]);
      emit(WorkspaceEvents.WORKSPACE_CREATED, newWorkspace);

      return newWorkspace;
    } catch (err) {
      console.error("📁 WorkspaceContext: Error creating workspace:", err);
      throw err;
    }
  }, [emit]);

  /**
   * Update a workspace
   * @param {string} workspaceId - Workspace ID
   * @param {Object} updates - Updates to apply
   * @returns {Object} Updated workspace
   */
  const updateWorkspace = useCallback(async (workspaceId, updates) => {
    try {
      const updatedWorkspace = await workspaceService.updateWorkspace(workspaceId, updates);
      console.log("📁 WorkspaceContext: Updated workspace:", updatedWorkspace);

      setWorkspaces(prev => prev.map(ws =>
        ws.id === workspaceId ? updatedWorkspace : ws
      ));

      // Update current workspace if it was updated
      if (currentWorkspace?.id === workspaceId) {
        setCurrentWorkspace(updatedWorkspace);
      }

      emit(WorkspaceEvents.WORKSPACE_UPDATED, updatedWorkspace);

      return updatedWorkspace;
    } catch (err) {
      console.error("📁 WorkspaceContext: Error updating workspace:", err);
      throw err;
    }
  }, [currentWorkspace, emit]);

  /**
   * Delete a workspace
   * @param {string} workspaceId - Workspace ID
   * @param {boolean} deleteDocuments - Whether to delete documents
   */
  const deleteWorkspace = useCallback(async (workspaceId, deleteDocuments = false) => {
    try {
      const deletedWorkspace = workspaces.find(ws => ws.id === workspaceId);
      const wasCurrentWorkspace = currentWorkspace?.id === workspaceId;

      await workspaceService.deleteWorkspace(workspaceId, deleteDocuments);
      console.log("📁 WorkspaceContext: Deleted workspace:", workspaceId);

      // Refresh workspaces list from server to get any newly created default workspace
      const response = await workspaceService.getWorkspaces(true);
      const freshWorkspaces = response.workspaces || [];
      setWorkspaces(freshWorkspaces);
      console.log("📁 WorkspaceContext: Refreshed workspaces after deletion:", freshWorkspaces.length);

      // If deleted workspace was selected, select a new one from the fresh list
      if (wasCurrentWorkspace && freshWorkspaces.length > 0) {
        const defaultWs = freshWorkspaces.find(ws => ws.is_default);
        const newWorkspace = defaultWs || freshWorkspaces[0];
        console.log("📁 WorkspaceContext: Selecting new workspace after deletion:", newWorkspace?.name);
        selectWorkspace(newWorkspace);
      }

      emit(WorkspaceEvents.WORKSPACE_DELETED, deletedWorkspace);
    } catch (err) {
      console.error("📁 WorkspaceContext: Error deleting workspace:", err);
      throw err;
    }
  }, [workspaces, currentWorkspace, selectWorkspace, emit]);

  /**
   * Set a workspace as default
   * @param {string} workspaceId - Workspace ID
   */
  const setDefaultWorkspace = useCallback(async (workspaceId) => {
    try {
      await workspaceService.setDefaultWorkspace(workspaceId);
      console.log("📁 WorkspaceContext: Set default workspace:", workspaceId);

      setWorkspaces(prev => prev.map(ws => ({
        ...ws,
        is_default: ws.id === workspaceId
      })));

      emit(WorkspaceEvents.WORKSPACE_UPDATED, { id: workspaceId, is_default: true });
    } catch (err) {
      console.error("📁 WorkspaceContext: Error setting default workspace:", err);
      throw err;
    }
  }, [emit]);

  /**
   * Request documents refresh (used after upload)
   */
  const refreshDocuments = useCallback(() => {
    console.log("📁 WorkspaceContext: Requesting documents refresh for workspace:", currentWorkspace?.id);
    emit(WorkspaceEvents.DOCUMENTS_REFRESH_NEEDED, { workspaceId: currentWorkspace?.id });
  }, [currentWorkspace, emit]);

  // Load workspaces when authentication changes
  // Use both user and isAuthenticated to ensure we react to login/logout events
  useEffect(() => {
    console.log("📁 WorkspaceContext: Auth state changed - user:", user?.username, "isAuthenticated:", isAuthenticated);
    if (isAuthenticated && user) {
      console.log("📁 WorkspaceContext: User authenticated, loading workspaces...");
      loadWorkspaces();
    } else {
      console.log("📁 WorkspaceContext: User not authenticated, clearing workspaces");
      setWorkspaces([]);
      setCurrentWorkspace(null);
      setLoading(false);
    }
  }, [isAuthenticated, user]); // eslint-disable-line react-hooks/exhaustive-deps

  const value = {
    // State
    workspaces,
    currentWorkspace,
    currentWorkspaceId: currentWorkspace?.id || null,
    loading,
    error,
    user,
    isAuthenticated,

    // Actions
    loadWorkspaces,
    selectWorkspace,
    createWorkspace,
    updateWorkspace,
    deleteWorkspace,
    setDefaultWorkspace,
    refreshDocuments,

    // Event system
    subscribe,
    emit,
    WorkspaceEvents,
  };

  return (
    <WorkspaceContext.Provider value={value}>
      {children}
    </WorkspaceContext.Provider>
  );
};

/**
 * Hook to use workspace context
 * @returns {Object} Workspace context value
 */
export const useWorkspace = () => {
  const context = useContext(WorkspaceContext);
  if (!context) {
    throw new Error("useWorkspace must be used within a WorkspaceProvider");
  }
  return context;
};

/**
 * Hook to subscribe to workspace events
 * @param {string} eventType - Event type to subscribe to
 * @param {Function} callback - Callback function
 */
export const useWorkspaceEvent = (eventType, callback) => {
  const { subscribe } = useWorkspace();

  useEffect(() => {
    const unsubscribe = subscribe(eventType, callback);
    return unsubscribe;
  }, [eventType, callback, subscribe]);
};

export default WorkspaceContext;
