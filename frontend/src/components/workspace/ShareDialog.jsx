import React, { useState, useEffect, useCallback } from "react";
import { Dialog } from "primereact/dialog";
import { Button } from "primereact/button";
import { InputText } from "primereact/inputtext";
import { InputTextarea } from "primereact/inputtextarea";
import { Chips } from "primereact/chips";
import { MultiSelect } from "primereact/multiselect";
import { Calendar } from "primereact/calendar";
import { DataTable } from "primereact/datatable";
import { Column } from "primereact/column";
import { Tag } from "primereact/tag";
import { confirmDialog } from "primereact/confirmdialog";
import { useTranslation } from "react-i18next";
import sharingService from "../../services/sharingService";
import { messageService } from "../../core/message/messageService";
import "./ShareDialog.scss";

export const ShareDialog = ({
  visible,
  onHide,
  document,
  onShareComplete
}) => {
  const { t } = useTranslation();
  const [activeTab, setActiveTab] = useState("share"); // share or manage
  const [loading, setLoading] = useState(false);
  const [shares, setShares] = useState([]);
  const [sharesLoading, setSharesLoading] = useState(false);

  // Share form state
  const [recipients, setRecipients] = useState([]);
  const [selectedPermissions, setSelectedPermissions] = useState(["read"]);
  const [message, setMessage] = useState("");
  const [expiresAt, setExpiresAt] = useState(null);

  const permissionOptions = sharingService.getPermissionOptions();

  // Load existing shares when managing
  const loadShares = useCallback(async () => {
    if (!document?.id) return;

    try {
      setSharesLoading(true);
      const data = await sharingService.getDocumentShares(document.id);
      setShares(data.shares || []);
    } catch (error) {
      console.error("Error loading shares:", error);
      // User might not have permission to view shares
      if (error.response?.status === 403) {
        messageService.warnToast(t("Share.NoPermissionToManage"));
      }
    } finally {
      setSharesLoading(false);
    }
  }, [document?.id, t]);

  useEffect(() => {
    if (visible && activeTab === "manage") {
      loadShares();
    }
  }, [visible, activeTab, loadShares]);

  // Handle share submission
  const handleShare = async () => {
    if (recipients.length === 0) {
      messageService.warnToast(t("Share.EnterRecipients"));
      return;
    }

    if (selectedPermissions.length === 0) {
      messageService.warnToast(t("Share.SelectPermissions"));
      return;
    }

    try {
      setLoading(true);

      const result = await sharingService.shareDocumentByUsername(
        document.id,
        recipients,
        selectedPermissions,
        message.trim() || null,
        expiresAt?.toISOString() || null
      );

      messageService.successToast(result.message || t("Share.Success"));

      // Reset form
      setRecipients([]);
      setMessage("");
      setExpiresAt(null);

      // Refresh shares list
      loadShares();

      if (onShareComplete) {
        onShareComplete();
      }
    } catch (error) {
      console.error("Error sharing document:", error);
      messageService.errorToast(error.response?.data?.detail || t("Share.Error"));
    } finally {
      setLoading(false);
    }
  };

  // Handle revoke access
  const handleRevoke = (share) => {
    confirmDialog({
      message: t("Share.RevokeConfirm", { name: share.user_name || share.user_email || "this user" }),
      header: t("Share.RevokeTitle"),
      icon: "pi pi-exclamation-triangle",
      acceptClassName: "p-button-danger",
      accept: async () => {
        try {
          await sharingService.revokeAccess(document.id, share.user_id);
          setShares(prev => prev.filter(s => s.id !== share.id));
          messageService.successToast(t("Share.RevokeSuccess"));
        } catch (error) {
          console.error("Error revoking access:", error);
          messageService.errorToast(error.response?.data?.detail || t("Share.RevokeError"));
        }
      }
    });
  };

  // Handle permission update
  const handleUpdatePermissions = async (share, newPermissions) => {
    try {
      await sharingService.updatePermissions(document.id, share.user_id, newPermissions);
      setShares(prev => prev.map(s =>
        s.id === share.id ? { ...s, permissions: newPermissions } : s
      ));
      messageService.successToast(t("Share.PermissionsUpdated"));
    } catch (error) {
      console.error("Error updating permissions:", error);
      messageService.errorToast(error.response?.data?.detail || t("Share.UpdateError"));
    }
  };

  // Permission tag template
  const permissionTagTemplate = (permission) => {
    const severityMap = {
      read: "info",
      update: "warning",
      delete: "danger",
      share: "success",
      full: "primary"
    };
    return (
      <Tag
        key={permission}
        value={permission}
        severity={severityMap[permission] || "info"}
        className="permission-tag"
      />
    );
  };

  // User column template
  const userTemplate = (rowData) => (
    <div className="share-user">
      <i className="pi pi-user" />
      <div className="user-info">
        <span className="user-name">{rowData.user_name || "Unknown"}</span>
        <span className="user-email">{rowData.user_email}</span>
      </div>
    </div>
  );

  // Permissions column template
  const permissionsTemplate = (rowData) => (
    <div className="permissions-list">
      {rowData.permissions?.map(permissionTagTemplate)}
    </div>
  );

  // Actions column template
  const actionsTemplate = (rowData) => (
    <div className="share-actions">
      <Button
        icon="pi pi-trash"
        className="p-button-text p-button-danger p-button-sm"
        onClick={() => handleRevoke(rowData)}
        tooltip={t("Share.Revoke")}
      />
    </div>
  );

  // Shared date template
  const sharedDateTemplate = (rowData) => {
    if (!rowData.shared_at) return "-";
    return new Date(rowData.shared_at).toLocaleDateString();
  };

  // Dialog footer
  const dialogFooter = (
    <div className="dialog-footer">
      <Button
        label={t("Common.Close")}
        icon="pi pi-times"
        onClick={onHide}
        className="p-button-text"
      />
      {activeTab === "share" && (
        <Button
          label={t("Share.ShareButton")}
          icon="pi pi-share-alt"
          onClick={handleShare}
          loading={loading}
          disabled={recipients.length === 0}
        />
      )}
    </div>
  );

  return (
    <Dialog
      visible={visible}
      onHide={onHide}
      header={
        <div className="share-dialog-header">
          <i className="pi pi-share-alt" />
          <span>{t("Share.Title")}: {document?.original_filename}</span>
        </div>
      }
      footer={dialogFooter}
      style={{ width: "600px" }}
      className="share-dialog"
      modal
    >
      {/* Tab navigation */}
      <div className="share-tabs">
        <Button
          label={t("Share.ShareTab")}
          icon="pi pi-plus"
          className={`tab-button ${activeTab === "share" ? "active" : ""}`}
          onClick={() => setActiveTab("share")}
        />
        <Button
          label={t("Share.ManageTab")}
          icon="pi pi-users"
          className={`tab-button ${activeTab === "manage" ? "active" : ""}`}
          onClick={() => setActiveTab("manage")}
        />
      </div>

      {/* Share tab content */}
      {activeTab === "share" && (
        <div className="share-form">
          <div className="field">
            <label>{t("Share.Recipients")}</label>
            <Chips
              value={recipients}
              onChange={(e) => setRecipients(e.value)}
              placeholder={t("Share.RecipientsPlaceholder")}
              separator=","
              className="recipients-chips"
            />
            <small className="field-help">{t("Share.RecipientsHelp")}</small>
          </div>

          <div className="field">
            <label>{t("Share.Permissions")}</label>
            <MultiSelect
              value={selectedPermissions}
              options={permissionOptions}
              onChange={(e) => setSelectedPermissions(e.value)}
              optionLabel="label"
              optionValue="value"
              placeholder={t("Share.SelectPermissions")}
              display="chip"
              className="permissions-select"
            />
          </div>

          <div className="field">
            <label>{t("Share.Message")}</label>
            <InputTextarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              rows={3}
              placeholder={t("Share.MessagePlaceholder")}
              className="message-input"
            />
          </div>

          <div className="field">
            <label>{t("Share.ExpiresAt")}</label>
            <Calendar
              value={expiresAt}
              onChange={(e) => setExpiresAt(e.value)}
              minDate={new Date()}
              showIcon
              showButtonBar
              placeholder={t("Share.ExpiresPlaceholder")}
              className="expires-calendar"
            />
            <small className="field-help">{t("Share.ExpiresHelp")}</small>
          </div>
        </div>
      )}

      {/* Manage tab content */}
      {activeTab === "manage" && (
        <div className="manage-shares">
          {sharesLoading ? (
            <div className="loading-indicator">
              <i className="pi pi-spin pi-spinner" />
              <span>{t("Common.Loading")}</span>
            </div>
          ) : shares.length === 0 ? (
            <div className="empty-message">
              <i className="pi pi-users" />
              <p>{t("Share.NoShares")}</p>
            </div>
          ) : (
            <DataTable
              value={shares}
              className="shares-table"
              size="small"
              scrollable
              scrollHeight="300px"
            >
              <Column
                header={t("Share.User")}
                body={userTemplate}
                style={{ minWidth: "180px" }}
              />
              <Column
                header={t("Share.Permissions")}
                body={permissionsTemplate}
                style={{ minWidth: "150px" }}
              />
              <Column
                header={t("Share.SharedDate")}
                body={sharedDateTemplate}
                style={{ minWidth: "100px" }}
              />
              <Column
                body={actionsTemplate}
                style={{ width: "60px" }}
              />
            </DataTable>
          )}
        </div>
      )}
    </Dialog>
  );
};

export default ShareDialog;
