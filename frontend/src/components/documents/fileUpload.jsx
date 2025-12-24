import React, { useRef, useState } from "react";
import { FileUpload } from "primereact/fileupload";
import { ProgressBar } from "primereact/progressbar";
import { useTranslation } from "react-i18next";
import documentService from "../../services/documentService";
import { messageService } from "../../core/message/messageService";
import { useWorkspace } from "../../contexts/WorkspaceContext";
import "./fileUpload.scss";

export const DocumentFileUpload = ({ onUploadSuccess }) => {
  const { t } = useTranslation();
  const fileUploadRef = useRef(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  // Get current workspace from context
  const { currentWorkspace, currentWorkspaceId } = useWorkspace();

  // Debug log workspace info
  console.log(`📁 DocumentFileUpload: Using workspace "${currentWorkspace?.name}" (ID: ${currentWorkspaceId})`);

  const handleUpload = async (e) => {
    // The uploadHandler receives an event object with files property
    const files = e.files;

    if (!files || files.length === 0) {
      messageService.infoToast(t("FileUpload.PleaseSelectFiles"));
      return;
    }

    await uploadFiles(files);
  };

  const uploadFiles = async (files) => {
    try {
      setUploading(true);
      setUploadProgress(0);

      const uploadedDocIds = []; // Track uploaded document IDs for WebSocket subscription

      for (let i = 0; i < files.length; i++) {
        const file = files[i];

        // Validate file type
        const validExtensions = [".pdf", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".doc", ".docx", ".xls", ".xlsx", ".txt", ".csv", ".tsv"];
        const fileExtension = "." + file.name.split(".").pop().toLowerCase();

        if (!validExtensions.includes(fileExtension)) {
          messageService.errorToast(
            t("FileUpload.InvalidFileType", { filename: file.name })
          );
          continue;
        }

        try {
          // Use workspace ID from context
          console.log(`📁 Uploading ${file.name} to workspace: ${currentWorkspaceId} (workspace: ${currentWorkspace?.name})`);
          console.log(`📁 currentWorkspace object:`, currentWorkspace);
          const response = await documentService.uploadDocument(file, currentWorkspaceId);

          if (response.status === "success") {
            messageService.successToast(
              t("FileUpload.UploadSuccess", { filename: file.name })
            );
            setUploadProgress(((i + 1) / files.length) * 100);

            // Track the uploaded document ID for WebSocket subscription
            if (response.document_id) {
              uploadedDocIds.push(response.document_id);
              console.log(`📤 Uploaded document ID: ${response.document_id} (${file.name})`);
            }
          }
        } catch (error) {
          messageService.errorToast(
            t("FileUpload.UploadFailed", { filename: file.name })
          );
          console.error("Upload error:", error);
        }
      }

      // Clear the file upload component
      if (fileUploadRef.current) {
        fileUploadRef.current.clear();
      }

      // Notify parent component with uploaded document IDs
      if (onUploadSuccess) {
        onUploadSuccess(uploadedDocIds);
      }
    } catch (error) {
      messageService.errorToast(t("FileUpload.ErrorUploading"));
      console.error("Error uploading files:", error);
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  return (
    <div className="file-upload-container">
      <div className="file-upload-header">
        <h2>{t("FileUpload.Title")}</h2>
        <p>{t("FileUpload.Description")}</p>
        {currentWorkspace && (
          <p className="workspace-info">
            <i className="pi pi-folder" style={{ color: currentWorkspace.color }} />
            <span>Uploading to: <strong>{currentWorkspace.name}</strong></span>
          </p>
        )}
      </div>

      <div className="file-upload-area">
        <FileUpload
          ref={fileUploadRef}
          name="files"
          multiple
          accept=".pdf,.png,.jpg,.jpeg,.gif,.bmp,.doc,.docx,.xls,.xlsx,.txt,.csv,.tsv"
          maxFileSize={52428800} // 50MB
          customUpload
          uploadHandler={handleUpload}
          chooseLabel={t("FileUpload.ChooseFiles")}
          uploadLabel={t("FileUpload.Upload")}
          cancelLabel={t("FileUpload.Cancel")}
        />
      </div>

      {uploading && (
        <div className="upload-progress">
          <ProgressBar value={uploadProgress} />
          <p>
            {t("FileUpload.Uploading")} {Math.round(uploadProgress)}%
          </p>
        </div>
      )}

      <div className="file-upload-info">
        <p>
          <strong>{t("FileUpload.SupportedFormats")}</strong>{" "}
          {t("FileUpload.SupportedFormatsList")}
        </p>
        <p>
          <strong>{t("FileUpload.MaxFileSize")}</strong>{" "}
          {t("FileUpload.MaxFileSizeValue")}
        </p>
      </div>
    </div>
  );
};
