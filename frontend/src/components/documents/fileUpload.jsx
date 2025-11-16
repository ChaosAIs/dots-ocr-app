import React, { useRef, useState } from "react";
import { FileUpload } from "primereact/fileupload";
import { ProgressBar } from "primereact/progressbar";
import { useTranslation } from "react-i18next";
import documentService from "../../services/documentService";
import { messageService } from "../../core/message/messageService";
import "./fileUpload.scss";

export const DocumentFileUpload = ({ onUploadSuccess }) => {
  const { t } = useTranslation();
  const fileUploadRef = useRef(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

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
          const response = await documentService.uploadDocument(file);
          if (response.status === "success") {
            messageService.successToast(
              t("FileUpload.UploadSuccess", { filename: file.name })
            );
            setUploadProgress(((i + 1) / files.length) * 100);
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

      // Notify parent component
      if (onUploadSuccess) {
        onUploadSuccess();
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

