import React, { useState, useEffect, useCallback } from "react";
import { Button } from "primereact/button";
import { Dialog } from "primereact/dialog";
import { InputText } from "primereact/inputtext";
import { InputTextarea } from "primereact/inputtextarea";
import { Dropdown } from "primereact/dropdown";
import { Badge } from "primereact/badge";
import { confirmDialog } from "primereact/confirmdialog";
import { useTranslation } from "react-i18next";
import workspaceService from "../../services/workspaceService";
import sharingService from "../../services/sharingService";
import { messageService } from "../../core/message/messageService";
import { useWorkspace } from "../../contexts/WorkspaceContext";
import "./WorkspaceSidebar.scss";

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
  const [editingWorkspace, setEditingWorkspace] = useState(null);
  const [newSharesCount, setNewSharesCount] = useState(0);

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

  useEffect(() => {
    loadNewSharesCount();
  }, [loadNewSharesCount]);

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

  // Delete workspace using context
  const handleDelete = (workspace) => {
    if (workspace.is_default || workspace.is_system) {
      messageService.warnToast(t("Workspace.CannotDeleteDefault"));
      return;
    }

    confirmDialog({
      message: t("Workspace.DeleteConfirm", { name: workspace.name }),
      header: t("Workspace.DeleteTitle"),
      icon: "pi pi-exclamation-triangle",
      acceptClassName: "p-button-danger",
      accept: async () => {
        try {
          await ctxDeleteWorkspace(workspace.id, false);
          messageService.successToast(t("Workspace.DeleteSuccess"));

          // Select default workspace if deleted workspace was selected
          if (selectedWorkspace?.id === workspace.id) {
            const defaultWs = workspaces.find(ws => ws.is_default && ws.id !== workspace.id);
            if (defaultWs && onWorkspaceSelect) {
              onWorkspaceSelect(defaultWs);
            }
          }

          if (onWorkspaceChange) {
            onWorkspaceChange("delete", workspace);
          }
        } catch (error) {
          console.error("Error deleting workspace:", error);
          messageService.errorToast(error.response?.data?.detail || t("Workspace.DeleteError"));
        }
      }
    });
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

    return (
      <div
        key={workspace.id}
        className={`workspace-item ${isSelected ? "selected" : ""} ${workspace.is_system ? "system" : ""}`}
        onClick={() => onWorkspaceSelect && onWorkspaceSelect(workspace)}
      >
        <div className="workspace-icon" style={{ color: workspace.color }}>
          <i className={getIconClass(workspace.icon)} />
        </div>
        <div className="workspace-info">
          <span className="workspace-name">
            {workspace.name}
            {workspace.is_default && !workspace.is_system && (
              <i className="pi pi-star-fill default-star" title={t("Workspace.Default")} />
            )}
          </span>
          <span className="workspace-count">
            {workspace.document_count} {t("Workspace.Documents")}
            {isSharedWithMe && newSharesCount > 0 && (
              <Badge value={newSharesCount} severity="danger" className="new-badge" />
            )}
          </span>
        </div>
        {!workspace.is_system && !collapsed && (
          <div className="workspace-actions">
            <Button
              icon="pi pi-pencil"
              className="p-button-text p-button-sm"
              onClick={(e) => {
                e.stopPropagation();
                openEditDialog(workspace);
              }}
              tooltip={t("Workspace.EditWorkspace")}
            />
            {!workspace.is_default && (
              <Button
                icon="pi pi-star"
                className="p-button-text p-button-sm"
                onClick={(e) => {
                  e.stopPropagation();
                  handleSetDefault(workspace);
                }}
                tooltip={t("Workspace.SetDefault")}
              />
            )}
            {!workspace.is_default && (
              <Button
                icon="pi pi-trash"
                className="p-button-text p-button-sm p-button-danger"
                onClick={(e) => {
                  e.stopPropagation();
                  handleDelete(workspace);
                }}
                tooltip={t("Workspace.Delete")}
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
    <div className="color-option">
      <div className="color-swatch" style={{ backgroundColor: option.value }} />
      <span>{option.name}</span>
    </div>
  );

  // Icon option template
  const iconOptionTemplate = (option) => (
    <div className="icon-option">
      <i className={option.icon} />
      <span>{option.name}</span>
    </div>
  );

  return (
    <div className={`workspace-sidebar ${collapsed ? "collapsed" : ""}`}>

      {/* Header */}
      <div className="sidebar-header">
        <h3>{t("Workspace.Title")}</h3>
        <Button
          icon="pi pi-plus"
          className="p-button-text p-button-sm"
          onClick={() => {
            resetForm();
            setShowCreateDialog(true);
          }}
          tooltip={t("Workspace.Create")}
        />
      </div>

      {/* Workspace list */}
      <div className="workspace-list">
        {loading ? (
          <div className="loading-indicator">
            <i className="pi pi-spin pi-spinner" />
          </div>
        ) : (
          <>
            {/* Regular workspaces */}
            {workspaces
              .filter(ws => !ws.is_system)
              .map(renderWorkspaceItem)}

            {/* System workspaces divider */}
            {workspaces.some(ws => ws.is_system) && (
              <div className="workspace-divider" />
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
        className="workspace-dialog"
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
        className="workspace-dialog"
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
    </div>
  );
};

export default WorkspaceSidebar;
