import React, { useState, useEffect } from "react";
import { Dialog } from "primereact/dialog";
import { Button } from "primereact/button";
import { ProgressSpinner } from "primereact/progressspinner";
import { useTranslation } from "react-i18next";
import { useWorkspace } from "../../contexts/WorkspaceContext";
import workspaceService from "../../services/workspaceService";
import { messageService } from "../../core/message/messageService";
import "./MoveDocumentDialog.scss";

/**
 * Dialog component for moving a document to another workspace.
 * Displays workspaces as folder icons for user selection.
 */
const MoveDocumentDialog = ({
  visible,
  document,
  onHide,
  onMoveSuccess
}) => {
  const { t } = useTranslation();
  const { currentWorkspaceId, loadWorkspaces } = useWorkspace();
  const [workspaces, setWorkspaces] = useState([]);
  const [loading, setLoading] = useState(false);
  const [moving, setMoving] = useState(false);
  const [selectedWorkspaceId, setSelectedWorkspaceId] = useState(null);

  // Helper function to map workspace icon names to PrimeIcons classes
  const getWorkspaceIconClass = (iconName) => {
    const iconMap = {
      folder: "pi-folder",
      briefcase: "pi-briefcase",
      book: "pi-book",
      home: "pi-home",
      star: "pi-star",
      heart: "pi-heart",
      tag: "pi-tag",
      bookmark: "pi-bookmark",
      flag: "pi-flag",
      inbox: "pi-inbox",
      share: "pi-share-alt",
      box: "pi-box",
      database: "pi-database",
      image: "pi-image",
      file: "pi-file"
    };
    return iconMap[iconName] || "pi-folder";
  };

  const loadWorkspacesData = async () => {
    try {
      setLoading(true);
      const response = await workspaceService.getWorkspaces(false); // Exclude system workspaces
      // Filter out the current workspace (can't move to same workspace)
      const filteredWorkspaces = response.filter(
        ws => ws.id !== currentWorkspaceId && !ws.is_system
      );
      setWorkspaces(filteredWorkspaces);
    } catch (error) {
      console.error("Error loading workspaces:", error);
      messageService.errorToast(t("Workspace.LoadError"));
    } finally {
      setLoading(false);
    }
  };

  // Load workspaces when dialog opens
  useEffect(() => {
    if (visible) {
      loadWorkspacesData();
      setSelectedWorkspaceId(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible, currentWorkspaceId]);

  const handleMove = async () => {
    if (!selectedWorkspaceId || !document) {
      return;
    }

    try {
      setMoving(true);
      await workspaceService.moveDocument(document.id, selectedWorkspaceId);

      messageService.successToast(t("DocumentList.MoveSuccess"));

      // Refresh workspace list to update document counts
      loadWorkspaces();

      // Call success callback to refresh document list
      if (onMoveSuccess) {
        onMoveSuccess();
      }

      onHide();
    } catch (error) {
      console.error("Error moving document:", error);
      messageService.errorToast(t("DocumentList.MoveFailed"));
    } finally {
      setMoving(false);
    }
  };

  const handleWorkspaceSelect = (workspaceId) => {
    setSelectedWorkspaceId(workspaceId);
  };

  const dialogFooter = (
    <div className="move-dialog-footer">
      <Button
        label={t("DocumentList.Cancel")}
        icon="pi pi-times"
        className="p-button-text"
        onClick={onHide}
        disabled={moving}
      />
      <Button
        label={moving ? t("DocumentList.Moving") : t("DocumentList.MoveDocument")}
        icon={moving ? "pi pi-spin pi-spinner" : "pi pi-arrow-right"}
        className="p-button-primary"
        onClick={handleMove}
        disabled={!selectedWorkspaceId || moving}
      />
    </div>
  );

  return (
    <Dialog
      header={t("DocumentList.MoveDocumentTitle")}
      visible={visible}
      style={{ width: "500px", maxWidth: "90vw" }}
      onHide={onHide}
      footer={dialogFooter}
      className="move-document-dialog"
      closable={!moving}
      draggable={false}
    >
      <div className="move-dialog-content">
        {document && (
          <div className="document-info">
            <i className="pi pi-file document-icon" />
            <span className="document-name">{document.filename}</span>
          </div>
        )}

        <div className="select-workspace-label">
          {t("DocumentList.SelectTargetWorkspace")}
        </div>

        {loading ? (
          <div className="loading-container">
            <ProgressSpinner style={{ width: "40px", height: "40px" }} />
          </div>
        ) : workspaces.length === 0 ? (
          <div className="no-workspaces">
            <i className="pi pi-info-circle" />
            <span>{t("DocumentList.NoOtherWorkspaces")}</span>
          </div>
        ) : (
          <div className="workspace-grid">
            {workspaces.map((workspace) => (
              <div
                key={workspace.id}
                className={`workspace-item ${selectedWorkspaceId === workspace.id ? "selected" : ""}`}
                onClick={() => handleWorkspaceSelect(workspace.id)}
              >
                <div
                  className="workspace-icon-container"
                  style={{ backgroundColor: workspace.color || "#6366f1" }}
                >
                  <i className={`pi ${getWorkspaceIconClass(workspace.icon)}`} />
                </div>
                <div className="workspace-details">
                  <span className="workspace-name">{workspace.name}</span>
                  <span className="workspace-doc-count">
                    {workspace.document_count || 0} {t("Workspace.Documents")}
                  </span>
                </div>
                {workspace.is_default && (
                  <span className="default-badge">{t("Workspace.Default")}</span>
                )}
                {selectedWorkspaceId === workspace.id && (
                  <i className="pi pi-check selected-check" />
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </Dialog>
  );
};

export default MoveDocumentDialog;
