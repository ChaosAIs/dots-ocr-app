import React, { useState, useEffect, useContext } from "react";
import { useTranslation } from "react-i18next";
import { messageService } from "../core/message/messageService";
import { Button } from "primereact/button";
import APP_CONFIG from "../core/config/appConfig";
import { ResultStatus } from "../core/enumertions/resultStatus";
import http from "../core/http/httpClient";
import { AuthContext } from "../core/auth/components/authProvider";
import { debugCurrentUserRoles, debugHasRole, debugHasPermission } from "../core/auth/debugAuth";

export const Sample = () => {
  const { t } = useTranslation();
  const [keyVaultMessage, setMessage] = useState("");
  const [translateMessage, setTranslateMessage] = useState("");
  const [notifications, setNotifications] = useState([]);
  const [invoices, setInvoices] = useState([]);

  const authService = useContext(AuthContext);

  useEffect(() => {
    if (authService && authService.isAuthenticated()) {
      http.get(APP_CONFIG.apiDomain + "/api/lookup/getnotifications").then((response) => {
        if (response.data && response.data.resultStatus === ResultStatus.Success) {
          setNotifications(response.data.item);
        }
      });

      http.get(APP_CONFIG.apiDomain + "/api/invoice/getinvoices").then((response) => {
        if (response.data && response.data.resultStatus === ResultStatus.Success) {
          setInvoices(response.data.item);
        }
      });
    }
  }, [authService]);

  /***
   *  Same as code in: constructor(props) { super(props); this.showMessages = this.showMessages.bind(this);}
   */
  const showMessages = () => {
    messageService.info(t("Pages.Sample.InformationTest"), false);
    messageService.warn(t("Pages.Sample.WarnTest"), false);
    messageService.error(t("Pages.Sample.ErrorTest"), false);
    messageService.success(t("Pages.Sample.SuccessTest"), false);
    messageService.emit();
  };

  const showMessage = (messageType) => {
    if (messageType === "success") {
      messageService.success(t("Pages.Sample.TestSuccessMessage"));
    } else if (messageType === "error") {
      messageService.error(t("Pages.Sample.TestErrorMessage"));
    }
  };

  const showToast = (messageType) => {
    if (messageType === "success") {
      messageService.successToast(t("Pages.Sample.TestSuccessToast"));
    } else if (messageType === "error") {
      messageService.errorToast(t("Pages.Sample.TestErrorToast"));
    }
  };

  const showKeyVaultMessage = () => {
    http.get(APP_CONFIG.apiDomain + "/api/lookup/getkeyvaultmessage").then((response) => {
      if (response.data && response.data.resultStatus === ResultStatus.Success) {
        setMessage(response.data.item);
      }
    });
  };

  const showTranslateMessage = () => {
    http.get(APP_CONFIG.apiDomain + "/api/lookup/gettranslatemessage").then((response) => {
      if (response.data && response.data.resultStatus === ResultStatus.Success) {
        setTranslateMessage(response.data.item);
      }
    });
  };

  const showConfirmDialog = (message) => {
    messageService.confirmDialog(message, (response) => {
      messageService.infoToast(t("Pages.Sample.Choiced", { response }));
    });
  };

  const showDeleteConfirmDialog = (message) => {
    messageService.confirmDeletionDialog(message, (response) => {
      messageService.infoToast(t("Pages.Sample.Choiced", { response }));
    });
  };

  const showDialog = (message, messageType) => {
    switch (messageType) {
      case "success":
        messageService.successDialog(message);
        break;
      case "error":
        messageService.errorDialog(message);
        break;
      case "warn":
        messageService.warnDialog(message);
        break;
      case "info":
        messageService.infoDialog(message);
        break;
      default:
        messageService.infoDialog(message);
        break;
    }
  };

  return (
    <>
      <h2>{t("Pages.Sample.Title")}</h2>
      {authService && typeof authService.isAuthenticated === 'function' && authService.isAuthenticated() && (
        <div>
          <h3>{t("Pages.Sample.TodaysNews")}</h3>
          <ul>
            {notifications.map((item) => {
              return <li key={item.id}>{item.message}</li>;
            })}
          </ul>
          <h3>{t("Pages.Sample.EncryptedInvoice")}</h3>
          <ul>
            {invoices.map((item) => {
              return (
                <li key={item.id}>
                  {item.invoiceNumber} - {item.vendorName}- {item.invoiceContent} - {item.amount}
                </li>
              );
            })}
          </ul>
        </div>
      )}
      <div className="flex flex-wrap">
        <div className="mb-2 mr-2">
          <Button className="btn btn-primary" onClick={showMessages}>
            {t("Pages.Sample.DemoMessageBox")}
          </Button>
        </div>
        <div className="mb-2 mr-2">
          <Button className="btn btn-primary" onClick={() => showMessage("success")}>
            {t("Pages.Sample.DemoSuccessMessage")}
          </Button>
        </div>
        <div className="mb-2 mr-2">
          <Button className="btn btn-primary" onClick={() => showMessage("error")}>
            {t("Pages.Sample.DemoErrorMessage")}
          </Button>
        </div>
      </div>
      <div className="flex flex-wrap">
        <div className="mb-2 mr-2">
          <Button className="btn btn-primary" onClick={() => showToast("error")}>
            {t("Pages.Sample.DemoErrorToast")}
          </Button>
        </div>
        <div className="mb-2 mr-2">
          <Button className="btn btn-primary" onClick={() => showToast("success")}>
            {t("Pages.Sample.DemoSuccessToast")}
          </Button>
        </div>
      </div>
      <div className="flex flex-wrap">
        <div className="mb-2 mr-2">
          <Button
            className="btn btn-primary"
            onClick={() =>
              showConfirmDialog(t("Pages.Sample.ConfirmMessage"))
            }
          >
            {t("Pages.Sample.DemoConfirmDialog")}
          </Button>
        </div>
        <div className="mb-2 mr-2">
          <Button
            className="btn btn-primary"
            onClick={() =>
              showDeleteConfirmDialog(t("Pages.Sample.DeleteConfirmMessage"))
            }
          >
            {t("Pages.Sample.DemoDeleteDialog")}
          </Button>
        </div>
        <div className="mb-2 mr-2">
          <Button className="btn btn-primary" onClick={() => showDialog(t("Pages.Sample.SuccessDialogMessage"), "success")}>
            {t("Pages.Sample.DemoSuccessDialog")}
          </Button>
        </div>
        <div className="mb-2 mr-2">
          <Button className="btn btn-primary" onClick={() => showDialog(t("Pages.Sample.ErrorDialogMessage"), "error")}>
            {t("Pages.Sample.DemoErrorDialog")}
          </Button>
        </div>
        <div className="mb-2 mr-2">
          <Button className="btn btn-primary" onClick={() => showDialog(t("Pages.Sample.WarnDialogMessage"), "warn")}>
            {t("Pages.Sample.DemoWarnDialog")}
          </Button>
        </div>
        <div className="mb-2 mr-2">
          <Button className="btn btn-primary" onClick={() => showDialog(t("Pages.Sample.InfoDialogMessage"), "info")}>
            {t("Pages.Sample.DemoInfoDialog")}
          </Button>
        </div>
      </div>
      <div className="flex flex-wrap">
        <div className="mb-2 mr-2">
          <Button className="btn btn-primary" onClick={() => showKeyVaultMessage()}>
            {t("Pages.Sample.DemoKeyVault")}
          </Button>
        </div>
        <div className="mb-2 mr-2">{keyVaultMessage}</div>
      </div>
      <div className="flex flex-wrap">
        <div className="mb-2 mr-2">
          <Button className="btn btn-primary" onClick={() => showTranslateMessage()}>
            {t("Pages.Sample.DemoTranslate")}
          </Button>
        </div>
        <div className="mb-2 mr-2">{translateMessage}</div>
      </div>

      <h4>{t("Pages.Sample.DebugAuth")}</h4>
      <div className="flex flex-wrap">
        <div className="mb-2 mr-2">
          <Button className="btn btn-primary" onClick={() => debugCurrentUserRoles()}>
            {t("Pages.Sample.DebugRoles")}
          </Button>
        </div>
        <div className="mb-2 mr-2">
          <Button className="btn btn-secondary" onClick={() => debugHasRole("2")}>
            {t("Pages.Sample.CheckAdminRole")}
          </Button>
        </div>
        <div className="mb-2 mr-2">
          <Button className="btn btn-secondary" onClick={() => debugHasRole("1")}>
            {t("Pages.Sample.CheckUserRole")}
          </Button>
        </div>
        <div className="mb-2 mr-2">
          <Button className="btn btn-secondary" onClick={() => debugHasPermission("1")}>
            {t("Pages.Sample.CheckLoginPermission")}
          </Button>
        </div>
      </div>
      <div className="mb-3">
        <small className="text-muted">
          {t("Pages.Sample.ConsoleTip")}
          <br />• <code>window.debugAuth.logRoles()</code>
          <br />• <code>window.debugAuth.hasRole(roleId)</code>
          <br />• <code>window.debugAuth.hasPermission(permissionId)</code>
        </small>
      </div>

      <h4>{t("Pages.Sample.CurrentEnvironment")}</h4>
      <span>{process.env.REACT_APP_ENV}</span>
      <h4>{t("Pages.Sample.CurrentConfig")}</h4>
      <span>{JSON.stringify(APP_CONFIG)}</span>
    </>
  );
};
