import { useState, useEffect, useCallback, forwardRef, useImperativeHandle } from "react";
import { Button } from "primereact/button";
import { Checkbox } from "primereact/checkbox";
import { Badge } from "primereact/badge";
import { Tooltip } from "primereact/tooltip";
import { ProgressSpinner } from "primereact/progressspinner";
import { useWorkspace, WorkspaceEvents } from "../../contexts/WorkspaceContext";
import workspaceService from "../../services/workspaceService";
import authService from "../../services/authService";
import "./WorkspaceBrowser.scss";

/**
 * WorkspaceBrowser - Right panel component for selecting workspaces and documents to filter chat RAG search
 *
 * Features:
 * - Multi-select workspace checkboxes
 * - Document-level checkboxes within each workspace
 * - When workspace is checked, all its documents are selected by default
 * - When workspace is unchecked, all its documents are deselected
 * - Users can select/deselect individual documents within a workspace
 * - Workspace checkbox reflects partial selection state (indeterminate)
 * - Collapsible panel
 * - Workspace color/icon indicators
 * - Expandable document lists under each workspace
 * - Persists selection to user preferences (both workspace and document IDs)
 * - Shows "All Documents" when no selection
 */
export const WorkspaceBrowser = forwardRef(({
  selectedWorkspaceIds,
  selectedDocumentIds,
  onSelectionChange,
  onDocumentSelectionChange,
  collapsed,
  onToggleCollapse
}, ref) => {
  const { workspaces, loading: workspacesLoading, subscribe } = useWorkspace();
  const [localSelectedIds, setLocalSelectedIds] = useState(selectedWorkspaceIds || []);
  const [localSelectedDocIds, setLocalSelectedDocIds] = useState(selectedDocumentIds || []);
  const [saving, setSaving] = useState(false);
  const [expandedWorkspaces, setExpandedWorkspaces] = useState({});
  const [workspaceDocuments, setWorkspaceDocuments] = useState({});
  const [loadingDocuments, setLoadingDocuments] = useState({});

  // Sync with parent's selectedWorkspaceIds
  useEffect(() => {
    setLocalSelectedIds(selectedWorkspaceIds || []);
  }, [selectedWorkspaceIds]);

  // Sync with parent's selectedDocumentIds
  useEffect(() => {
    setLocalSelectedDocIds(selectedDocumentIds || []);
  }, [selectedDocumentIds]);

  // Validate and clean up stale document IDs when workspace documents are loaded
  // This ensures selectedDocumentIds only contains documents from selected workspaces
  useEffect(() => {
    const validateDocumentSelection = async () => {
      if (localSelectedIds.length === 0) {
        // No workspaces selected - clear document selection if any
        if (localSelectedDocIds.length > 0) {
          console.log("[WorkspaceBrowser] No workspaces selected, clearing stale document IDs");
          setLocalSelectedDocIds([]);
          onDocumentSelectionChange?.([]);
          savePreferences([], []);
        }
        return;
      }

      // Collect all valid document IDs from selected workspaces
      const validDocIds = new Set();
      let needsLoad = false;

      for (const wsId of localSelectedIds) {
        const docs = workspaceDocuments[wsId];
        if (docs) {
          docs.forEach(doc => validDocIds.add(doc.id));
        } else {
          // Documents not loaded yet for this workspace - need to load them
          needsLoad = true;
        }
      }

      // If we have all workspace documents loaded, validate the selection
      if (!needsLoad && localSelectedDocIds.length > 0) {
        const staleIds = localSelectedDocIds.filter(id => !validDocIds.has(id));
        if (staleIds.length > 0) {
          console.log(`[WorkspaceBrowser] Removing ${staleIds.length} stale document IDs not in selected workspaces:`, staleIds);
          const cleanedDocIds = localSelectedDocIds.filter(id => validDocIds.has(id));
          setLocalSelectedDocIds(cleanedDocIds);
          onDocumentSelectionChange?.(cleanedDocIds);
          savePreferences(localSelectedIds, cleanedDocIds);
        }
      }
    };

    validateDocumentSelection();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [localSelectedIds, workspaceDocuments]); // Re-run when workspaces or their documents change (callbacks are stable)

  // Load documents for selected workspaces on initial mount (for validation)
  useEffect(() => {
    const loadSelectedWorkspaceDocs = async () => {
      if (localSelectedIds.length === 0) return;

      // Load documents for any selected workspace that doesn't have docs loaded yet
      for (const wsId of localSelectedIds) {
        if (!workspaceDocuments[wsId] && !loadingDocuments[wsId]) {
          setLoadingDocuments(prev => ({ ...prev, [wsId]: true }));
          try {
            const result = await workspaceService.getWorkspace(wsId, 50, 0);
            setWorkspaceDocuments(prev => ({
              ...prev,
              [wsId]: result.documents || []
            }));
          } catch (error) {
            console.error(`[WorkspaceBrowser] Failed to load documents for workspace ${wsId}:`, error);
            setWorkspaceDocuments(prev => ({
              ...prev,
              [wsId]: []
            }));
          } finally {
            setLoadingDocuments(prev => ({ ...prev, [wsId]: false }));
          }
        }
      }
    };

    loadSelectedWorkspaceDocs();
  }, [localSelectedIds]); // Only trigger when selected workspace IDs change

  // Subscribe to workspace deletion events to clean up selections
  useEffect(() => {
    const unsubscribe = subscribe(WorkspaceEvents.WORKSPACE_DELETED, (deletedWorkspace) => {
      if (deletedWorkspace && localSelectedIds.includes(deletedWorkspace.id)) {
        const newIds = localSelectedIds.filter(id => id !== deletedWorkspace.id);
        setLocalSelectedIds(newIds);
        onSelectionChange?.(newIds);

        // Also remove documents from this workspace from selection
        const workspaceDocs = workspaceDocuments[deletedWorkspace.id] || [];
        const docIdsToRemove = workspaceDocs.map(doc => doc.id);
        const newDocIds = localSelectedDocIds.filter(id => !docIdsToRemove.includes(id));
        setLocalSelectedDocIds(newDocIds);
        onDocumentSelectionChange?.(newDocIds);

        // Save to preferences
        savePreferences(newIds, newDocIds);
      }
    });
    return unsubscribe;
  }, [subscribe, localSelectedIds, localSelectedDocIds, onSelectionChange, onDocumentSelectionChange, workspaceDocuments]);

  // Expose methods to parent
  useImperativeHandle(ref, () => ({
    getSelectedWorkspaceIds: () => localSelectedIds,
    getSelectedDocumentIds: () => localSelectedDocIds,
    setSelectedWorkspaceIds: (ids) => {
      setLocalSelectedIds(ids);
      onSelectionChange?.(ids);
    },
    setSelectedDocumentIds: (ids) => {
      setLocalSelectedDocIds(ids);
      onDocumentSelectionChange?.(ids);
    },
    clearSelection: () => {
      setLocalSelectedIds([]);
      setLocalSelectedDocIds([]);
      onSelectionChange?.([]);
      onDocumentSelectionChange?.([]);
      savePreferences([], []);
    }
  }));

  const savePreferences = useCallback(async (workspaceIds, documentIds) => {
    setSaving(true);
    try {
      await authService.updateChatPreferences(workspaceIds, documentIds);
    } catch (error) {
      console.error("Failed to save workspace/document preferences:", error);
    } finally {
      setSaving(false);
    }
  }, []);

  const handleWorkspaceToggle = useCallback(async (workspaceId, e) => {
    // Prevent expanding when clicking checkbox
    if (e) {
      e.stopPropagation();
    }

    const isCurrentlySelected = localSelectedIds.includes(workspaceId);
    let newWorkspaceIds;
    let newDocIds = [...localSelectedDocIds];

    if (isCurrentlySelected) {
      // Uncheck workspace - remove workspace and all its documents
      newWorkspaceIds = localSelectedIds.filter(id => id !== workspaceId);

      // Remove all documents from this workspace
      const workspaceDocs = workspaceDocuments[workspaceId] || [];
      const docIdsToRemove = workspaceDocs.map(doc => doc.id);
      newDocIds = newDocIds.filter(id => !docIdsToRemove.includes(id));
    } else {
      // Check workspace - add workspace and all its documents
      newWorkspaceIds = [...localSelectedIds, workspaceId];

      // Load documents if not already loaded
      let docs = workspaceDocuments[workspaceId];
      if (!docs) {
        try {
          const result = await workspaceService.getWorkspace(workspaceId, 50, 0);
          docs = result.documents || [];
          setWorkspaceDocuments(prev => ({
            ...prev,
            [workspaceId]: docs
          }));
        } catch (error) {
          console.error(`Failed to load documents for workspace ${workspaceId}:`, error);
          docs = [];
        }
      }

      // Add all documents from this workspace
      const docIdsToAdd = docs.map(doc => doc.id);
      newDocIds = [...new Set([...newDocIds, ...docIdsToAdd])];
    }

    setLocalSelectedIds(newWorkspaceIds);
    setLocalSelectedDocIds(newDocIds);
    onSelectionChange?.(newWorkspaceIds);
    onDocumentSelectionChange?.(newDocIds);
    savePreferences(newWorkspaceIds, newDocIds);
  }, [localSelectedIds, localSelectedDocIds, workspaceDocuments, onSelectionChange, onDocumentSelectionChange, savePreferences]);

  // Toggle workspace expansion and load documents if needed
  const handleToggleExpand = useCallback(async (workspaceId, e) => {
    e.stopPropagation();

    const isCurrentlyExpanded = expandedWorkspaces[workspaceId];

    // Toggle expansion state
    setExpandedWorkspaces(prev => ({
      ...prev,
      [workspaceId]: !isCurrentlyExpanded
    }));

    // Load documents if expanding and not already loaded
    if (!isCurrentlyExpanded && !workspaceDocuments[workspaceId]) {
      setLoadingDocuments(prev => ({ ...prev, [workspaceId]: true }));
      try {
        const result = await workspaceService.getWorkspace(workspaceId, 50, 0);
        setWorkspaceDocuments(prev => ({
          ...prev,
          [workspaceId]: result.documents || []
        }));
      } catch (error) {
        console.error(`Failed to load documents for workspace ${workspaceId}:`, error);
        setWorkspaceDocuments(prev => ({
          ...prev,
          [workspaceId]: []
        }));
      } finally {
        setLoadingDocuments(prev => ({ ...prev, [workspaceId]: false }));
      }
    }
  }, [expandedWorkspaces, workspaceDocuments]);

  const handleSelectAll = useCallback(async () => {
    const allWorkspaceIds = workspaces.filter(ws => !ws.is_system).map(ws => ws.id);
    setLocalSelectedIds(allWorkspaceIds);
    onSelectionChange?.(allWorkspaceIds);

    // Load and select all documents from all workspaces
    const allDocIds = [];
    for (const wsId of allWorkspaceIds) {
      let docs = workspaceDocuments[wsId];
      if (!docs) {
        try {
          const result = await workspaceService.getWorkspace(wsId, 50, 0);
          docs = result.documents || [];
          setWorkspaceDocuments(prev => ({
            ...prev,
            [wsId]: docs
          }));
        } catch (error) {
          console.error(`Failed to load documents for workspace ${wsId}:`, error);
          docs = [];
        }
      }
      docs.forEach(doc => allDocIds.push(doc.id));
    }

    const uniqueDocIds = [...new Set(allDocIds)];
    setLocalSelectedDocIds(uniqueDocIds);
    onDocumentSelectionChange?.(uniqueDocIds);
    savePreferences(allWorkspaceIds, uniqueDocIds);
  }, [workspaces, workspaceDocuments, onSelectionChange, onDocumentSelectionChange, savePreferences]);

  const handleClearSelection = useCallback(() => {
    setLocalSelectedIds([]);
    setLocalSelectedDocIds([]);
    onSelectionChange?.([]);
    onDocumentSelectionChange?.([]);
    savePreferences([], []);
  }, [onSelectionChange, onDocumentSelectionChange, savePreferences]);

  // Handle individual document toggle
  const handleDocumentToggle = useCallback((workspaceId, docId, e) => {
    if (e) {
      e.stopPropagation();
    }

    const isDocSelected = localSelectedDocIds.includes(docId);
    let newDocIds;
    let newWorkspaceIds = [...localSelectedIds];

    if (isDocSelected) {
      // Uncheck document
      newDocIds = localSelectedDocIds.filter(id => id !== docId);
    } else {
      // Check document
      newDocIds = [...localSelectedDocIds, docId];
    }

    // Update workspace selection state based on document selection
    const workspaceDocs = workspaceDocuments[workspaceId] || [];
    const workspaceDocIds = workspaceDocs.map(doc => doc.id);
    const selectedDocsInWorkspace = newDocIds.filter(id => workspaceDocIds.includes(id));

    if (selectedDocsInWorkspace.length === 0) {
      // No documents selected in this workspace - uncheck workspace
      newWorkspaceIds = newWorkspaceIds.filter(id => id !== workspaceId);
    } else if (!newWorkspaceIds.includes(workspaceId)) {
      // Some documents selected but workspace not in list - add it
      newWorkspaceIds = [...newWorkspaceIds, workspaceId];
    }

    setLocalSelectedDocIds(newDocIds);
    setLocalSelectedIds(newWorkspaceIds);
    onDocumentSelectionChange?.(newDocIds);
    onSelectionChange?.(newWorkspaceIds);
    savePreferences(newWorkspaceIds, newDocIds);
  }, [localSelectedDocIds, localSelectedIds, workspaceDocuments, onDocumentSelectionChange, onSelectionChange, savePreferences]);

  // Get workspace checkbox state (checked, unchecked, or indeterminate)
  const getWorkspaceCheckState = useCallback((workspaceId) => {
    const workspaceDocs = workspaceDocuments[workspaceId] || [];
    if (workspaceDocs.length === 0) {
      // No documents loaded yet - use workspace selection state
      return { checked: localSelectedIds.includes(workspaceId), indeterminate: false };
    }

    const workspaceDocIds = workspaceDocs.map(doc => doc.id);
    const selectedDocsInWorkspace = localSelectedDocIds.filter(id => workspaceDocIds.includes(id));

    if (selectedDocsInWorkspace.length === 0) {
      return { checked: false, indeterminate: false };
    } else if (selectedDocsInWorkspace.length === workspaceDocIds.length) {
      return { checked: true, indeterminate: false };
    } else {
      return { checked: true, indeterminate: true };
    }
  }, [workspaceDocuments, localSelectedDocIds, localSelectedIds]);

  // Filter out system workspaces (like "Shared With Me")
  const regularWorkspaces = workspaces.filter(ws => !ws.is_system);
  const selectedCount = localSelectedIds.length;

  // Get workspace icon class
  const getWorkspaceIcon = (iconName) => {
    const iconMap = {
      folder: "pi-folder",
      briefcase: "pi-briefcase",
      book: "pi-book",
      star: "pi-star",
      heart: "pi-heart",
      flag: "pi-flag",
      box: "pi-box",
      database: "pi-database",
      image: "pi-image",
      file: "pi-file"
    };
    return iconMap[iconName] || "pi-folder";
  };

  // Get document icon based on file type
  const getDocumentIcon = (filename) => {
    if (!filename) return "pi-file";
    const ext = filename.split('.').pop()?.toLowerCase();
    const iconMap = {
      pdf: "pi-file-pdf",
      doc: "pi-file-word",
      docx: "pi-file-word",
      xls: "pi-file-excel",
      xlsx: "pi-file-excel",
      ppt: "pi-file",
      pptx: "pi-file",
      txt: "pi-file",
      md: "pi-file",
      jpg: "pi-image",
      jpeg: "pi-image",
      png: "pi-image",
      gif: "pi-image",
      svg: "pi-image"
    };
    return iconMap[ext] || "pi-file";
  };

  // Truncate filename if too long
  const truncateFilename = (filename, maxLength = 25) => {
    if (!filename || filename.length <= maxLength) return filename;
    const ext = filename.split('.').pop();
    const nameWithoutExt = filename.slice(0, filename.lastIndexOf('.'));
    const truncatedName = nameWithoutExt.slice(0, maxLength - ext.length - 4) + '...';
    return `${truncatedName}.${ext}`;
  };

  if (collapsed) {
    return (
      <div className="workspace-browser-collapsed">
        <Button
          icon="pi pi-angle-left"
          text
          rounded
          className="expand-btn"
          onClick={onToggleCollapse}
          tooltip="Expand Knowledge Sources"
          tooltipOptions={{ position: "left" }}
          style={{ width: '1.75rem', height: '1.75rem', padding: 0 }}
        >
          {selectedCount > 0 && (
            <Badge value={selectedCount} severity="info" className="selection-badge" />
          )}
        </Button>
      </div>
    );
  }

  return (
    <div className="workspace-browser-panel">
      <div className="workspace-browser-header">
        <div className="header-title">
          <i className="pi pi-database" />
          <span>Knowledge Sources</span>
          {saving && <i className="pi pi-spin pi-spinner saving-indicator" />}
        </div>
        <Button
          icon="pi pi-angle-right"
          text
          rounded
          className="collapse-btn"
          onClick={onToggleCollapse}
          tooltip="Collapse"
          tooltipOptions={{ position: "left" }}
          style={{ width: '1.75rem', height: '1.75rem', padding: 0 }}
        />
      </div>

      <div className="selection-status">
        {selectedCount === 0 && localSelectedDocIds.length === 0 ? (
          <span className="status-text all-docs">
            <i className="pi pi-globe" />
            Searching all documents
          </span>
        ) : (
          <span className="status-text filtered">
            <i className="pi pi-filter" />
            {localSelectedDocIds.length > 0
              ? `${localSelectedDocIds.length} document${localSelectedDocIds.length !== 1 ? "s" : ""} selected`
              : `${selectedCount} workspace${selectedCount !== 1 ? "s" : ""} selected`}
          </span>
        )}
      </div>

      <div className="action-buttons">
        <Button
          label="Select All"
          icon="pi pi-check-square"
          className="p-button-text p-button-sm"
          onClick={handleSelectAll}
          disabled={regularWorkspaces.length === 0}
        />
        <Button
          label="Clear"
          icon="pi pi-times"
          className="p-button-text p-button-sm p-button-secondary"
          onClick={handleClearSelection}
          disabled={selectedCount === 0}
        />
      </div>

      {workspacesLoading ? (
        <div className="loading-container">
          <i className="pi pi-spin pi-spinner" />
          <span>Loading workspaces...</span>
        </div>
      ) : regularWorkspaces.length === 0 ? (
        <div className="empty-state">
          <i className="pi pi-inbox" />
          <p>No workspaces available</p>
        </div>
      ) : (
        <div className="workspaces-list">
          {regularWorkspaces.map((workspace) => {
            const isExpanded = expandedWorkspaces[workspace.id];
            const documents = workspaceDocuments[workspace.id] || [];
            const isLoadingDocs = loadingDocuments[workspace.id];
            const docCount = workspace.document_count || 0;
            const checkState = getWorkspaceCheckState(workspace.id);

            return (
              <div key={workspace.id} className="workspace-container">
                <div
                  className={`workspace-item ${checkState.checked ? "selected" : ""} ${checkState.indeterminate ? "indeterminate" : ""}`}
                >
                  <Checkbox
                    checked={checkState.checked}
                    onChange={(e) => handleWorkspaceToggle(workspace.id, e)}
                    className={`workspace-checkbox ${checkState.indeterminate ? "p-indeterminate" : ""}`}
                    onClick={(e) => e.stopPropagation()}
                  />
                  <div
                    className="workspace-color-dot"
                    style={{ backgroundColor: workspace.color || "#6366f1" }}
                  />
                  <i className={`workspace-icon pi ${getWorkspaceIcon(workspace.icon)}`} />
                  <div
                    className="workspace-info"
                    onClick={(e) => docCount > 0 && handleToggleExpand(workspace.id, e)}
                  >
                    <span className="workspace-name">{workspace.name}</span>
                    <span className="workspace-doc-count">
                      {docCount} doc{docCount !== 1 ? "s" : ""}
                    </span>
                  </div>
                  {docCount > 0 && (
                    <Button
                      icon={`pi ${isExpanded ? "pi-chevron-up" : "pi-chevron-down"}`}
                      text
                      rounded
                      className="expand-toggle-btn"
                      onClick={(e) => handleToggleExpand(workspace.id, e)}
                      tooltip={isExpanded ? "Collapse documents" : "Show documents"}
                      tooltipOptions={{ position: "left" }}
                      style={{ width: '1.75rem', height: '1.75rem', padding: 0 }}
                    />
                  )}
                </div>

                {/* Documents List */}
                {isExpanded && (
                  <div className="documents-list">
                    {isLoadingDocs ? (
                      <div className="documents-loading">
                        <ProgressSpinner style={{ width: '16px', height: '16px' }} strokeWidth="4" />
                        <span>Loading documents...</span>
                      </div>
                    ) : documents.length === 0 ? (
                      <div className="documents-empty">
                        <i className="pi pi-info-circle" />
                        <span>No documents in this workspace</span>
                      </div>
                    ) : (
                      documents.map((doc) => (
                        <div key={doc.id} className={`document-item ${localSelectedDocIds.includes(doc.id) ? "selected" : ""}`}>
                          <Checkbox
                            checked={localSelectedDocIds.includes(doc.id)}
                            onChange={(e) => handleDocumentToggle(workspace.id, doc.id, e)}
                            className="document-checkbox"
                            onClick={(e) => e.stopPropagation()}
                          />
                          <i className={`document-icon pi ${getDocumentIcon(doc.original_filename || doc.filename)}`} />
                          <span
                            className="document-name"
                            title={doc.original_filename || doc.filename}
                          >
                            {truncateFilename(doc.original_filename || doc.filename)}
                          </span>
                          {doc.ocr_status && (
                            <span className={`document-status status-${doc.ocr_status}`}>
                              {doc.ocr_status === 'completed' ? (
                                <i className="pi pi-check-circle" title="Indexed" />
                              ) : doc.ocr_status === 'processing' || doc.ocr_status === 'converting' ? (
                                <i className="pi pi-spin pi-spinner" title="Processing" />
                              ) : doc.ocr_status === 'failed' ? (
                                <i className="pi pi-exclamation-triangle" title="Failed" />
                              ) : (
                                <i className="pi pi-clock" title="Pending" />
                              )}
                            </span>
                          )}
                        </div>
                      ))
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      <div className="workspace-browser-footer">
        <Tooltip target=".info-icon" position="top" />
        <i
          className="pi pi-info-circle info-icon"
          data-pr-tooltip="Select workspaces to limit your search scope. Leave empty to search all accessible documents."
        />
      </div>
    </div>
  );
});

export default WorkspaceBrowser;
