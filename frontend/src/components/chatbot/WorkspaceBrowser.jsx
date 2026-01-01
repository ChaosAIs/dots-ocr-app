import { useState, useEffect, useCallback, forwardRef, useImperativeHandle } from "react";
import { Button } from "primereact/button";
import { Checkbox } from "primereact/checkbox";
import { Badge } from "primereact/badge";
import { Tooltip } from "primereact/tooltip";
import { ProgressSpinner } from "primereact/progressspinner";
import { useWorkspace, WorkspaceEvents } from "../../contexts/WorkspaceContext";
import workspaceService from "../../services/workspaceService";
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
 * - Selection is managed by parent component (stored in session metadata)
 */
export const WorkspaceBrowser = forwardRef(({
  selectedWorkspaceIds = [],
  selectedDocumentIds = [],
  onSelectionChange,
  onDocumentSelectionChange,
  collapsed,
  onToggleCollapse
}, ref) => {
  const { workspaces, loading: workspacesLoading, subscribe } = useWorkspace();
  const [expandedWorkspaces, setExpandedWorkspaces] = useState({});
  const [workspaceDocuments, setWorkspaceDocuments] = useState({});
  const [loadingDocuments, setLoadingDocuments] = useState({});

  // Load documents for selected workspaces when selection changes
  useEffect(() => {
    const loadSelectedWorkspaceDocs = async () => {
      if (!selectedWorkspaceIds || selectedWorkspaceIds.length === 0) return;

      for (const wsId of selectedWorkspaceIds) {
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
  }, [selectedWorkspaceIds, workspaceDocuments, loadingDocuments]);

  // Find and restore workspace IDs for orphaned document selections
  // This handles the case where session metadata has documents but no workspace IDs
  useEffect(() => {
    const findWorkspacesForDocuments = async () => {
      // Only run if we have documents but no workspaces selected
      if (selectedDocumentIds.length === 0 || selectedWorkspaceIds.length > 0) return;

      console.log("[WorkspaceBrowser] Finding workspaces for orphaned documents:", selectedDocumentIds);

      // Load documents from all workspaces to find which ones contain our selected documents
      const workspacesWithSelectedDocs = [];
      for (const ws of workspaces.filter(w => !w.is_system)) {
        if (!workspaceDocuments[ws.id]) {
          try {
            const result = await workspaceService.getWorkspace(ws.id, 50, 0);
            setWorkspaceDocuments(prev => ({
              ...prev,
              [ws.id]: result.documents || []
            }));
            // Check if this workspace contains any of our selected documents
            const docIds = (result.documents || []).map(d => d.id);
            const hasSelectedDocs = selectedDocumentIds.some(id => docIds.includes(id));
            if (hasSelectedDocs) {
              workspacesWithSelectedDocs.push(ws.id);
            }
          } catch (error) {
            console.error(`[WorkspaceBrowser] Failed to load workspace ${ws.id}:`, error);
          }
        } else {
          // Check already loaded documents
          const docIds = workspaceDocuments[ws.id].map(d => d.id);
          const hasSelectedDocs = selectedDocumentIds.some(id => docIds.includes(id));
          if (hasSelectedDocs) {
            workspacesWithSelectedDocs.push(ws.id);
          }
        }
      }

      if (workspacesWithSelectedDocs.length > 0) {
        console.log("[WorkspaceBrowser] Found workspaces for documents:", workspacesWithSelectedDocs);
        onSelectionChange?.(workspacesWithSelectedDocs);
      }
    };

    findWorkspacesForDocuments();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDocumentIds.length, selectedWorkspaceIds.length, workspaces]);

  // Subscribe to workspace deletion events
  useEffect(() => {
    const unsubscribe = subscribe(WorkspaceEvents.WORKSPACE_DELETED, (deletedWorkspace) => {
      if (deletedWorkspace && selectedWorkspaceIds.includes(deletedWorkspace.id)) {
        const newWorkspaceIds = selectedWorkspaceIds.filter(id => id !== deletedWorkspace.id);
        const workspaceDocs = workspaceDocuments[deletedWorkspace.id] || [];
        const docIdsToRemove = workspaceDocs.map(doc => doc.id);
        const newDocIds = selectedDocumentIds.filter(id => !docIdsToRemove.includes(id));

        onSelectionChange?.(newWorkspaceIds);
        onDocumentSelectionChange?.(newDocIds);
      }
    });
    return unsubscribe;
  }, [subscribe, selectedWorkspaceIds, selectedDocumentIds, workspaceDocuments, onSelectionChange, onDocumentSelectionChange]);

  // Expose methods to parent
  useImperativeHandle(ref, () => ({
    getSelectedWorkspaceIds: () => selectedWorkspaceIds,
    getSelectedDocumentIds: () => selectedDocumentIds,
    clearSelection: () => {
      onSelectionChange?.([]);
      onDocumentSelectionChange?.([]);
    }
  }));

  const handleWorkspaceToggle = useCallback(async (workspaceId, e) => {
    if (e) e.stopPropagation();

    console.log("[WorkspaceBrowser] handleWorkspaceToggle:", { workspaceId });

    const isCurrentlySelected = selectedWorkspaceIds.includes(workspaceId);
    let newWorkspaceIds;
    let newDocIds = [...selectedDocumentIds];

    if (isCurrentlySelected) {
      // Uncheck workspace - remove workspace and all its documents
      newWorkspaceIds = selectedWorkspaceIds.filter(id => id !== workspaceId);
      const workspaceDocs = workspaceDocuments[workspaceId] || [];
      const docIdsToRemove = workspaceDocs.map(doc => doc.id);
      newDocIds = newDocIds.filter(id => !docIdsToRemove.includes(id));
    } else {
      // Check workspace - add workspace and all its documents
      newWorkspaceIds = [...selectedWorkspaceIds, workspaceId];

      // Load documents if not already loaded
      let docs = workspaceDocuments[workspaceId];
      if (!docs) {
        try {
          setLoadingDocuments(prev => ({ ...prev, [workspaceId]: true }));
          const result = await workspaceService.getWorkspace(workspaceId, 50, 0);
          docs = result.documents || [];
          setWorkspaceDocuments(prev => ({
            ...prev,
            [workspaceId]: docs
          }));
        } catch (error) {
          console.error(`Failed to load documents for workspace ${workspaceId}:`, error);
          docs = [];
        } finally {
          setLoadingDocuments(prev => ({ ...prev, [workspaceId]: false }));
        }
      }

      // Add all documents from this workspace
      const docIdsToAdd = docs.map(doc => doc.id);
      newDocIds = [...new Set([...newDocIds, ...docIdsToAdd])];
    }

    console.log("[WorkspaceBrowser] Workspace toggle result:", {
      newWorkspaceIds,
      newDocIds
    });

    onSelectionChange?.(newWorkspaceIds);
    onDocumentSelectionChange?.(newDocIds);
  }, [selectedWorkspaceIds, selectedDocumentIds, workspaceDocuments, onSelectionChange, onDocumentSelectionChange]);

  const handleToggleExpand = useCallback(async (workspaceId, e) => {
    e.stopPropagation();

    const isCurrentlyExpanded = expandedWorkspaces[workspaceId];
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

    onSelectionChange?.(allWorkspaceIds);
    onDocumentSelectionChange?.([...new Set(allDocIds)]);
  }, [workspaces, workspaceDocuments, onSelectionChange, onDocumentSelectionChange]);

  const handleClearSelection = useCallback(() => {
    onSelectionChange?.([]);
    onDocumentSelectionChange?.([]);
  }, [onSelectionChange, onDocumentSelectionChange]);

  const handleDocumentToggle = useCallback((workspaceId, docId, e) => {
    if (e) e.stopPropagation();

    console.log("[WorkspaceBrowser] handleDocumentToggle:", { workspaceId, docId });

    const isDocSelected = selectedDocumentIds.includes(docId);
    let newDocIds;
    let newWorkspaceIds = [...selectedWorkspaceIds];

    if (isDocSelected) {
      newDocIds = selectedDocumentIds.filter(id => id !== docId);
    } else {
      newDocIds = [...selectedDocumentIds, docId];
    }

    // Update workspace selection based on document selection
    const workspaceDocs = workspaceDocuments[workspaceId] || [];
    const workspaceDocIds = workspaceDocs.map(doc => doc.id);
    const selectedDocsInWorkspace = newDocIds.filter(id => workspaceDocIds.includes(id));

    console.log("[WorkspaceBrowser] Document toggle calculation:", {
      workspaceDocs: workspaceDocs.length,
      workspaceDocIds,
      selectedDocsInWorkspace,
      currentWorkspaceIds: selectedWorkspaceIds
    });

    if (selectedDocsInWorkspace.length === 0) {
      newWorkspaceIds = newWorkspaceIds.filter(id => id !== workspaceId);
    } else if (!newWorkspaceIds.includes(workspaceId)) {
      newWorkspaceIds = [...newWorkspaceIds, workspaceId];
    }

    console.log("[WorkspaceBrowser] Calling selection change:", {
      newWorkspaceIds,
      newDocIds
    });

    onSelectionChange?.(newWorkspaceIds);
    onDocumentSelectionChange?.(newDocIds);
  }, [selectedDocumentIds, selectedWorkspaceIds, workspaceDocuments, onSelectionChange, onDocumentSelectionChange]);

  // Get workspace checkbox state
  const getWorkspaceCheckState = useCallback((workspaceId) => {
    const isWorkspaceSelected = selectedWorkspaceIds.includes(workspaceId);
    const workspaceDocs = workspaceDocuments[workspaceId] || [];

    if (workspaceDocs.length === 0) {
      return { checked: isWorkspaceSelected, indeterminate: false };
    }

    const workspaceDocIds = workspaceDocs.map(doc => doc.id);
    const selectedDocsInWorkspace = selectedDocumentIds.filter(id => workspaceDocIds.includes(id));

    if (selectedDocsInWorkspace.length === 0) {
      return { checked: false, indeterminate: false };
    } else if (selectedDocsInWorkspace.length === workspaceDocIds.length) {
      return { checked: true, indeterminate: false };
    } else {
      return { checked: true, indeterminate: true };
    }
  }, [workspaceDocuments, selectedDocumentIds, selectedWorkspaceIds]);

  const regularWorkspaces = workspaces.filter(ws => !ws.is_system);
  const selectedCount = selectedWorkspaceIds.length;

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

  const getDocumentIcon = (filename) => {
    if (!filename) return "pi-file";
    const ext = filename.split('.').pop()?.toLowerCase();
    const iconMap = {
      pdf: "pi-file-pdf",
      doc: "pi-file-word",
      docx: "pi-file-word",
      xls: "pi-file-excel",
      xlsx: "pi-file-excel",
      txt: "pi-file",
      md: "pi-file",
      jpg: "pi-image",
      jpeg: "pi-image",
      png: "pi-image",
      gif: "pi-image",
      csv: "pi-file-excel"
    };
    return iconMap[ext] || "pi-file";
  };

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
        {selectedCount === 0 && selectedDocumentIds.length === 0 ? (
          <span className="status-text all-docs">
            <i className="pi pi-globe" />
            Searching all documents
          </span>
        ) : (
          <span className="status-text filtered">
            <i className="pi pi-filter" />
            {selectedDocumentIds.length > 0
              ? `${selectedDocumentIds.length} document${selectedDocumentIds.length !== 1 ? "s" : ""} selected`
              : `${selectedCount} workspace${selectedCount !== 1 ? "s" : ""} selected`}
          </span>
        )}
      </div>

      <div className="action-buttons">
        <Button
          label="Select All"
          icon={selectedCount === regularWorkspaces.length && regularWorkspaces.length > 0 ? "pi pi-check-square" : "pi pi-stop"}
          className="p-button-text p-button-sm"
          onClick={handleSelectAll}
          disabled={regularWorkspaces.length === 0 || selectedCount === regularWorkspaces.length}
        />
        <Button
          label="Clear"
          icon="pi pi-times"
          className="p-button-text p-button-sm p-button-secondary"
          onClick={handleClearSelection}
          disabled={selectedCount === 0 && selectedDocumentIds.length === 0}
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
                        <div key={doc.id} className={`document-item ${selectedDocumentIds.includes(doc.id) ? "selected" : ""}`}>
                          <Checkbox
                            checked={selectedDocumentIds.includes(doc.id)}
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
