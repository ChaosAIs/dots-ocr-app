import React, { useState, useEffect, useCallback } from "react";
import { Button } from "primereact/button";
import { Dialog } from "primereact/dialog";
import { InputText } from "primereact/inputtext";
import { InputTextarea } from "primereact/inputtextarea";
import { Dropdown } from "primereact/dropdown";
import { Badge } from "primereact/badge";
import { useTranslation } from "react-i18next";
import workspaceService from "../../services/workspaceService";
import sharingService from "../../services/sharingService";
import documentService from "../../services/documentService";
import { messageService } from "../../core/message/messageService";
import { useWorkspace } from "../../contexts/WorkspaceContext";

export const WorkspaceSidebar = ({
  selectedWorkspace,
  onWorkspaceSelect,
  onWorkspaceChange,
  collapsed = false
}) => {
  const { t } = useTranslation();

  // Get workspaces from context
  const {
    workspaces,
    loading,
    createWorkspace: ctxCreateWorkspace,
    updateWorkspace: ctxUpdateWorkspace,
    deleteWorkspace: ctxDeleteWorkspace,
    setDefaultWorkspace: ctxSetDefaultWorkspace,
  } = useWorkspace();

  const [showCreateDialog, setShowCreateDialog] = useState(false);
  const [showEditDialog, setShowEditDialog] = useState(false);
  const [showDeleteOptionsDialog, setShowDeleteOptionsDialog] = useState(false);
  const [workspaceToDelete, setWorkspaceToDelete] = useState(null);
  const [editingWorkspace, setEditingWorkspace] = useState(null);
  const [newSharesCount, setNewSharesCount] = useState(0);
  const [workspacesWithInProgress, setWorkspacesWithInProgress] = useState(new Set());

  // Form state
  const [formData, setFormData] = useState({
    name: "",
    description: "",
    color: "#6366f1",
    icon: "folder"
  });

  const colorOptions = workspaceService.getColorOptions();
  const iconOptions = workspaceService.getIconOptions();

  // Load new shares count
  const loadNewSharesCount = useCallback(async () => {
    try {
      const data = await sharingService.getNewShares();
      setNewSharesCount(data.count);
    } catch (error) {
      console.error("Error loading new shares count:", error);
    }
  }, []);

  // Check for in-progress documents and track which workspaces have them
  const checkInProgressDocuments = useCallback(async () => {
    try {
      const response = await documentService.getInProgressDocuments();
      if (response.status === "success" && response.documents) {
        // Get unique workspace IDs from in-progress documents
        const workspaceIds = new Set(
          response.documents
            .filter(doc => doc.workspace_id)
            .map(doc => doc.workspace_id)
        );
        setWorkspacesWithInProgress(workspaceIds);
      } else {
        setWorkspacesWithInProgress(new Set());
      }
    } catch (error) {
      console.error("Error checking in-progress documents:", error);
    }
  }, []);

  useEffect(() => {
    loadNewSharesCount();
  }, [loadNewSharesCount]);

  // Check for in-progress documents on mount and periodically
  useEffect(() => {
    checkInProgressDocuments();

    // Poll every 5 seconds to keep track of in-progress documents
    const intervalId = setInterval(checkInProgressDocuments, 5000);

    return () => clearInterval(intervalId);
  }, [checkInProgressDocuments]);

  // Create workspace using context
  const handleCreate = async () => {
    if (!formData.name.trim()) {
      messageService.warnToast(t("Workspace.NameRequired"));
      return;
    }

    try {
      const newWorkspace = await ctxCreateWorkspace({
        name: formData.name.trim(),
        description: formData.description.trim(),
        color: formData.color,
        icon: formData.icon
      });

      setShowCreateDialog(false);
      resetForm();
      messageService.successToast(t("Workspace.CreateSuccess"));

      if (onWorkspaceChange) {
        onWorkspaceChange("create", newWorkspace);
      }
    } catch (error) {
      console.error("Error creating workspace:", error);
      messageService.errorToast(error.response?.data?.detail || t("Workspace.CreateError"));
    }
  };

  // Edit workspace using context
  const handleEdit = async () => {
    if (!formData.name.trim()) {
      messageService.warnToast(t("Workspace.NameRequired"));
      return;
    }

    try {
      const updated = await ctxUpdateWorkspace(editingWorkspace.id, {
        name: formData.name.trim(),
        description: formData.description.trim(),
        color: formData.color,
        icon: formData.icon
      });

      setShowEditDialog(false);
      setEditingWorkspace(null);
      resetForm();
      messageService.successToast(t("Workspace.UpdateSuccess"));

      if (onWorkspaceChange) {
        onWorkspaceChange("update", updated);
      }
    } catch (error) {
      console.error("Error updating workspace:", error);
      messageService.errorToast(error.response?.data?.detail || t("Workspace.UpdateError"));
    }
  };

  // Delete workspace - show options dialog
  const handleDelete = (workspace) => {
    if (workspace.is_default || workspace.is_system) {
      messageService.warnToast(t("Workspace.CannotDeleteDefault"));
      return;
    }

    setWorkspaceToDelete(workspace);
    setShowDeleteOptionsDialog(true);
  };

  // Execute workspace deletion with selected option
  const executeDeleteWorkspace = async (deleteDocuments) => {
    if (!workspaceToDelete) return;

    try {
      await ctxDeleteWorkspace(workspaceToDelete.id, deleteDocuments);
      messageService.successToast(t("Workspace.DeleteSuccess"));

      // Select default workspace if deleted workspace was selected
      if (selectedWorkspace?.id === workspaceToDelete.id) {
        const defaultWs = workspaces.find(ws => ws.is_default && ws.id !== workspaceToDelete.id);
        if (defaultWs && onWorkspaceSelect) {
          onWorkspaceSelect(defaultWs);
        }
      }

      if (onWorkspaceChange) {
        onWorkspaceChange("delete", workspaceToDelete);
      }
    } catch (error) {
      console.error("Error deleting workspace:", error);
      messageService.errorToast(error.response?.data?.detail || t("Workspace.DeleteError"));
    } finally {
      setShowDeleteOptionsDialog(false);
      setWorkspaceToDelete(null);
    }
  };

  // Set default workspace using context
  const handleSetDefault = async (workspace) => {
    if (workspace.is_system) return;

    try {
      await ctxSetDefaultWorkspace(workspace.id);
      messageService.successToast(t("Workspace.SetDefaultSuccess"));
    } catch (error) {
      console.error("Error setting default workspace:", error);
      messageService.errorToast(t("Workspace.SetDefaultError"));
    }
  };

  const resetForm = () => {
    setFormData({
      name: "",
      description: "",
      color: "#6366f1",
      icon: "folder"
    });
  };

  const openEditDialog = (workspace) => {
    setEditingWorkspace(workspace);
    setFormData({
      name: workspace.name,
      description: workspace.description || "",
      color: workspace.color,
      icon: workspace.icon
    });
    setShowEditDialog(true);
  };

  const getIconClass = (iconName) => {
    const iconMap = {
      folder: "pi pi-folder",
      briefcase: "pi pi-briefcase",
      book: "pi pi-book",
      star: "pi pi-star",
      heart: "pi pi-heart",
      flag: "pi pi-flag",
      box: "pi pi-box",
      database: "pi pi-database",
      image: "pi pi-image",
      file: "pi pi-file",
      share: "pi pi-share-alt"
    };
    return iconMap[iconName] || "pi pi-folder";
  };

  // Render workspace item
  const renderWorkspaceItem = (workspace) => {
    const isSelected = selectedWorkspace?.id === workspace.id;
    const isSharedWithMe = workspace.is_system && workspace.name === "Shared With Me";
    // Can delete if not default and not system
    const canDelete = !workspace.is_default;

    return (
      <div
        key={workspace.id}
        className="workspace-item"
        style={{
          display: 'flex',
          alignItems: 'center',
          padding: '0.5rem 0.75rem',
          marginBottom: '0.25rem',
          cursor: 'pointer',
          borderLeft: '3px solid',
          borderLeftColor: isSelected ? 'var(--primary-color)' : 'transparent',
          borderRadius: '0 6px 6px 0',
          backgroundColor: isSelected ? 'var(--highlight-bg)' : 'transparent',
          transition: 'all 0.15s',
          opacity: workspace.is_system ? 0.9 : 1
        }}
        onMouseEnter={(e) => {
          if (!isSelected) {
            e.currentTarget.style.backgroundColor = 'var(--surface-hover)';
          }
        }}
        onMouseLeave={(e) => {
          if (!isSelected) {
            e.currentTarget.style.backgroundColor = 'transparent';
          }
        }}
        onClick={() => onWorkspaceSelect && onWorkspaceSelect(workspace)}
      >
        {/* Icon - fixed width */}
        <i
          className={getIconClass(workspace.icon)}
          style={{
            fontSize: '1rem',
            color: workspace.is_system ? 'var(--text-color-secondary)' : workspace.color,
            width: '20px',
            flexShrink: 0,
            marginRight: '0.5rem'
          }}
        />

        {/* Text content - flexible, truncates */}
        <div style={{
          flex: '1 1 0%',
          minWidth: 0,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem'
          }}>
            <span style={{
              fontSize: '0.875rem',
              color: isSelected ? 'var(--primary-color)' : 'var(--text-color)',
              fontWeight: isSelected ? '600' : '500',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis'
            }}>
              {workspace.name}
            </span>
            {workspace.is_default && !workspace.is_system && (
              <i
                className="pi pi-star-fill"
                title={t("Workspace.Default")}
                style={{
                  fontSize: '0.75rem',
                  color: '#EAB308',
                  flexShrink: 0
                }}
              />
            )}
          </div>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            color: 'var(--text-color-secondary)',
            fontSize: '0.75rem'
          }}>
            <span>{workspace.document_count} {t("Workspace.Documents")}</span>
            {isSharedWithMe && newSharesCount > 0 && (
              <Badge value={newSharesCount} severity="danger" />
            )}
          </div>
        </div>

        {/* Action buttons - fixed width with explicit sizing for theme consistency */}
        {!workspace.is_system && !collapsed && (
          <div style={{
            display: 'flex',
            gap: '0.125rem',
            flexShrink: 0,
            marginLeft: '0.25rem'
          }}>
            {!workspace.is_default && (
              <Button
                icon="pi pi-pencil"
                text
                rounded
                onClick={(e) => {
                  e.stopPropagation();
                  openEditDialog(workspace);
                }}
                tooltip={t("Workspace.EditWorkspace")}
                style={{ width: '1.75rem', height: '1.75rem', padding: 0 }}
              />
            )}
            {!workspace.is_default && (
              <Button
                icon="pi pi-star"
                text
                rounded
                onClick={(e) => {
                  e.stopPropagation();
                  handleSetDefault(workspace);
                }}
                tooltip={t("Workspace.SetDefault")}
                style={{ width: '1.75rem', height: '1.75rem', padding: 0 }}
              />
            )}
            {canDelete && (
              <Button
                icon="pi pi-trash"
                text
                rounded
                severity="danger"
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(workspace);
                }}
                tooltip={t("Workspace.Delete")}
                style={{ width: '1.75rem', height: '1.75rem', padding: 0 }}
              />
            )}
          </div>
        )}
      </div>
    );
  };

  // Dialog footer
  const dialogFooter = (onSubmit, onCancel) => (
    <div>
      <Button
        label={t("Workspace.Cancel")}
        icon="pi pi-times"
        onClick={onCancel}
        className="p-button-text"
      />
      <Button
        label={t("Workspace.Save")}
        icon="pi pi-check"
        onClick={onSubmit}
        autoFocus
      />
    </div>
  );

  // Color option template
  const colorOptionTemplate = (option) => (
    <div className="flex align-items-center gap-2 p-1">
      <div className="w-1rem h-1rem border-round border-1 surface-border" style={{ backgroundColor: option.value }} />
      <span>{option.name}</span>
    </div>
  );

  // Icon option template
  const iconOptionTemplate = (option) => (
    <div className="flex align-items-center gap-2 p-1">
      <i className={option.icon} />
      <span>{option.name}</span>
    </div>
  );

  return (
    <div className="flex flex-column h-full">
      {/* Header */}
      <div className="flex justify-content-between align-items-center border-bottom-1 surface-border" style={{ padding: '0.75rem 1rem' }}>
        <span style={{ fontSize: '0.9rem', fontWeight: '600', color: 'var(--text-color)' }}>{t("Workspace.Title")}</span>
        <Button
          icon="pi pi-plus"
          text
          rounded
          onClick={() => {
            resetForm();
            setShowCreateDialog(true);
          }}
          tooltip={t("Workspace.Create")}
          style={{ width: '1.75rem', height: '1.75rem', padding: 0 }}
        />
      </div>

      {/* Workspace list */}
      <div className="flex-1 overflow-auto py-2">
        {loading ? (
          <div className="flex justify-content-center p-4" style={{ color: 'var(--text-color-secondary)' }}>
            <i className="pi pi-spin pi-spinner text-xl" />
          </div>
        ) : (
          <>
            {/* Regular workspaces */}
            {workspaces
              .filter(ws => !ws.is_system)
              .map(renderWorkspaceItem)}

            {/* System workspaces divider */}
            {workspaces.some(ws => ws.is_system) && (
              <div className="border-top-1 surface-border mx-3 my-2" />
            )}

            {/* System workspaces */}
            {workspaces
              .filter(ws => ws.is_system)
              .map(renderWorkspaceItem)}
          </>
        )}
      </div>

      {/* Create Dialog */}
      <Dialog
        visible={showCreateDialog}
        onHide={() => setShowCreateDialog(false)}
        header={t("Workspace.CreateWorkspace")}
        footer={dialogFooter(handleCreate, () => setShowCreateDialog(false))}
        style={{ width: "450px" }}
      >
        <div className="p-fluid">
          <div className="field">
            <label htmlFor="name">{t("Workspace.Name")}</label>
            <InputText
              id="name"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              placeholder={t("Workspace.NamePlaceholder")}
              autoFocus
            />
          </div>
          <div className="field">
            <label htmlFor="description">{t("Workspace.Description")}</label>
            <InputTextarea
              id="description"
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              rows={3}
              placeholder={t("Workspace.DescriptionPlaceholder")}
            />
          </div>
          <div className="field">
            <label htmlFor="color">{t("Workspace.Color")}</label>
            <Dropdown
              id="color"
              value={formData.color}
              options={colorOptions}
              onChange={(e) => setFormData({ ...formData, color: e.value })}
              optionLabel="name"
              optionValue="value"
              itemTemplate={colorOptionTemplate}
              valueTemplate={colorOptionTemplate}
            />
          </div>
          <div className="field">
            <label htmlFor="icon">{t("Workspace.Icon")}</label>
            <Dropdown
              id="icon"
              value={formData.icon}
              options={iconOptions}
              onChange={(e) => setFormData({ ...formData, icon: e.value })}
              optionLabel="name"
              optionValue="value"
              itemTemplate={iconOptionTemplate}
              valueTemplate={iconOptionTemplate}
            />
          </div>
        </div>
      </Dialog>

      {/* Edit Dialog */}
      <Dialog
        visible={showEditDialog}
        onHide={() => {
          setShowEditDialog(false);
          setEditingWorkspace(null);
        }}
        header={t("Workspace.EditWorkspace")}
        footer={dialogFooter(handleEdit, () => {
          setShowEditDialog(false);
          setEditingWorkspace(null);
        })}
        style={{ width: "450px" }}
      >
        <div className="p-fluid">
          <div className="field">
            <label htmlFor="edit-name">{t("Workspace.Name")}</label>
            <InputText
              id="edit-name"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              autoFocus
            />
          </div>
          <div className="field">
            <label htmlFor="edit-description">{t("Workspace.Description")}</label>
            <InputTextarea
              id="edit-description"
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              rows={3}
            />
          </div>
          <div className="field">
            <label htmlFor="edit-color">{t("Workspace.Color")}</label>
            <Dropdown
              id="edit-color"
              value={formData.color}
              options={colorOptions}
              onChange={(e) => setFormData({ ...formData, color: e.value })}
              optionLabel="name"
              optionValue="value"
              itemTemplate={colorOptionTemplate}
              valueTemplate={colorOptionTemplate}
            />
          </div>
          <div className="field">
            <label htmlFor="edit-icon">{t("Workspace.Icon")}</label>
            <Dropdown
              id="edit-icon"
              value={formData.icon}
              options={iconOptions}
              onChange={(e) => setFormData({ ...formData, icon: e.value })}
              optionLabel="name"
              optionValue="value"
              itemTemplate={iconOptionTemplate}
              valueTemplate={iconOptionTemplate}
            />
          </div>
        </div>
      </Dialog>

      {/* Delete Options Dialog */}
      <Dialog
        visible={showDeleteOptionsDialog}
        onHide={() => {
          setShowDeleteOptionsDialog(false);
          setWorkspaceToDelete(null);
        }}
        header={t("Workspace.DeleteTitle")}
        style={{ width: "500px" }}
      >
        <div>
          <div className="flex align-items-start gap-3 mb-4 p-3 bg-yellow-50 border-round border-1 border-yellow-200">
            <i className="pi pi-exclamation-triangle text-xl text-yellow-600" />
            <span className="text-yellow-900">{t("Workspace.DeleteOptionsMessage", { name: workspaceToDelete?.name })}</span>
          </div>
          <div className="flex flex-column gap-3">
            <Button
              label={t("Workspace.DeleteAndRemoveFiles")}
              icon="pi pi-trash"
              severity="danger"
              className="w-full justify-content-start"
              onClick={() => executeDeleteWorkspace(true)}
            />
            <Button
              label={t("Workspace.DeleteAndMoveFiles")}
              icon="pi pi-folder"
              severity="warning"
              className="w-full justify-content-start"
              onClick={() => executeDeleteWorkspace(false)}
            />
            <Button
              label={t("Workspace.Cancel")}
              icon="pi pi-times"
              text
              className="w-full justify-content-start"
              onClick={() => {
                setShowDeleteOptionsDialog(false);
                setWorkspaceToDelete(null);
              }}
            />
          </div>
        </div>
      </Dialog>
    </div>
  );
};

export default WorkspaceSidebar;
