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
 * WorkspaceBrowser - Right panel component for selecting workspaces to filter chat RAG search
 *
 * Features:
 * - Multi-select workspace checkboxes
 * - Collapsible panel
 * - Workspace color/icon indicators
 * - Expandable document lists under each workspace
 * - Persists selection to user preferences
 * - Shows "All Documents" when no selection
 */
export const WorkspaceBrowser = forwardRef(({
  selectedWorkspaceIds,
  onSelectionChange,
  collapsed,
  onToggleCollapse
}, ref) => {
  const { workspaces, loading: workspacesLoading, subscribe } = useWorkspace();
  const [localSelectedIds, setLocalSelectedIds] = useState(selectedWorkspaceIds || []);
  const [saving, setSaving] = useState(false);
  const [expandedWorkspaces, setExpandedWorkspaces] = useState({});
  const [workspaceDocuments, setWorkspaceDocuments] = useState({});
  const [loadingDocuments, setLoadingDocuments] = useState({});

  // Sync with parent's selectedWorkspaceIds
  useEffect(() => {
    setLocalSelectedIds(selectedWorkspaceIds || []);
  }, [selectedWorkspaceIds]);

  // Subscribe to workspace deletion events to clean up selections
  useEffect(() => {
    const unsubscribe = subscribe(WorkspaceEvents.WORKSPACE_DELETED, (deletedWorkspace) => {
      if (deletedWorkspace && localSelectedIds.includes(deletedWorkspace.id)) {
        const newIds = localSelectedIds.filter(id => id !== deletedWorkspace.id);
        setLocalSelectedIds(newIds);
        onSelectionChange?.(newIds);
        // Save to preferences
        savePreferences(newIds);
      }
    });
    return unsubscribe;
  }, [subscribe, localSelectedIds, onSelectionChange]);

  // Expose methods to parent
  useImperativeHandle(ref, () => ({
    getSelectedWorkspaceIds: () => localSelectedIds,
    setSelectedWorkspaceIds: (ids) => {
      setLocalSelectedIds(ids);
      onSelectionChange?.(ids);
    },
    clearSelection: () => {
      setLocalSelectedIds([]);
      onSelectionChange?.([]);
      savePreferences([]);
    }
  }));

  const savePreferences = useCallback(async (ids) => {
    setSaving(true);
    try {
      await authService.updateChatPreferences(ids);
    } catch (error) {
      console.error("Failed to save workspace preferences:", error);
    } finally {
      setSaving(false);
    }
  }, []);

  const handleWorkspaceToggle = useCallback((workspaceId, e) => {
    // Prevent expanding when clicking checkbox
    if (e) {
      e.stopPropagation();
    }
    setLocalSelectedIds(prev => {
      const newIds = prev.includes(workspaceId)
        ? prev.filter(id => id !== workspaceId)
        : [...prev, workspaceId];

      onSelectionChange?.(newIds);
      savePreferences(newIds);
      return newIds;
    });
  }, [onSelectionChange, savePreferences]);

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

  const handleSelectAll = useCallback(() => {
    const allIds = workspaces.filter(ws => !ws.is_system).map(ws => ws.id);
    setLocalSelectedIds(allIds);
    onSelectionChange?.(allIds);
    savePreferences(allIds);
  }, [workspaces, onSelectionChange, savePreferences]);

  const handleClearSelection = useCallback(() => {
    setLocalSelectedIds([]);
    onSelectionChange?.([]);
    savePreferences([]);
  }, [onSelectionChange, savePreferences]);

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
          className="p-button-text expand-btn"
          onClick={onToggleCollapse}
          tooltip="Expand Knowledge Sources"
          tooltipOptions={{ position: "left" }}
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
          className="p-button-text p-button-sm collapse-btn"
          onClick={onToggleCollapse}
          tooltip="Collapse"
          tooltipOptions={{ position: "left" }}
        />
      </div>

      <div className="selection-status">
        {selectedCount === 0 ? (
          <span className="status-text all-docs">
            <i className="pi pi-globe" />
            Searching all documents
          </span>
        ) : (
          <span className="status-text filtered">
            <i className="pi pi-filter" />
            {selectedCount} workspace{selectedCount !== 1 ? "s" : ""} selected
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

            return (
              <div key={workspace.id} className="workspace-container">
                <div
                  className={`workspace-item ${localSelectedIds.includes(workspace.id) ? "selected" : ""}`}
                >
                  <Checkbox
                    checked={localSelectedIds.includes(workspace.id)}
                    onChange={(e) => handleWorkspaceToggle(workspace.id, e)}
                    className="workspace-checkbox"
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
                      className="p-button-text p-button-sm expand-toggle-btn"
                      onClick={(e) => handleToggleExpand(workspace.id, e)}
                      tooltip={isExpanded ? "Collapse documents" : "Show documents"}
                      tooltipOptions={{ position: "left" }}
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
                        <div key={doc.id} className="document-item">
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
