import React, { useState, useEffect, useContext } from "react";
import { messageService } from "../core/message/messageService";
import { Button } from "primereact/button";
import APP_CONFIG from "../core/config/appConfig";
import { ResultStatus } from "../core/enumertions/resultStatus";
import http from "../core/http/httpClient";
import { AuthContext } from "../core/auth/components/authProvider";
import { debugCurrentUserRoles, debugHasRole, debugHasPermission } from "../core/auth/debugAuth";

export const Sample = () => {
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
    messageService.info("Information Test", false);
    messageService.warn("Warn Test", false);
    messageService.error("Error Test", false);
    messageService.success("Success Test", false);
    messageService.emit();
  };

  const showMessage = (messageType) => {
    if (messageType === "success") {
      messageService.success("test success message.");
    } else if (messageType === "error") {
      messageService.error("test error message.");
    }
  };

  const showToast = (messageType) => {
    if (messageType === "success") {
      messageService.successToast("test success toast message.");
    } else if (messageType === "error") {
      messageService.errorToast("test error toast message.");
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
      messageService.infoToast("Choiced: " + response);
    });
  };

  const showDeleteConfirmDialog = (message) => {
    messageService.confirmDeletionDialog(message, (response) => {
      messageService.infoToast("Choiced: " + response);
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
      <h2>Sample Page</h2>
      {authService && typeof authService.isAuthenticated === 'function' && authService.isAuthenticated() && (
        <div>
          <h3>Today's News (Demo Remote Web API call)</h3>
          <ul>
            {notifications.map((item) => {
              return <li key={item.id}>{item.message}</li>;
            })}
          </ul>
          <h3>Encrypted Invoice Records Display Demo:</h3>
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
            Demo Message Box for multiple messages at one time
          </Button>
        </div>
        <div className="mb-2 mr-2">
          <Button className="btn btn-primary" onClick={() => showMessage("success")}>
            Demo Message Box for one success message
          </Button>
        </div>
        <div className="mb-2 mr-2">
          <Button className="btn btn-primary" onClick={() => showMessage("error")}>
            Demo Message Box for one error message
          </Button>
        </div>
      </div>
      <div className="flex flex-wrap">
        <div className="mb-2 mr-2">
          <Button className="btn btn-primary" onClick={() => showToast("error")}>
            Demo error Toast
          </Button>
        </div>
        <div className="mb-2 mr-2">
          <Button className="btn btn-primary" onClick={() => showToast("success")}>
            Demo Success Toast
          </Button>
        </div>
      </div>
      <div className="flex flex-wrap">
        <div className="mb-2 mr-2">
          <Button
            className="btn btn-primary"
            onClick={() =>
              showConfirmDialog("Are you sure you want to continue the process? please click 'Yes' to continue, or click 'No' to stop the process.")
            }
          >
            Demo Confirmation Dialog
          </Button>
        </div>
        <div className="mb-2 mr-2">
          <Button
            className="btn btn-primary"
            onClick={() =>
              showDeleteConfirmDialog("Are you sure you want to delete the item? please click 'Yes' to delete, or click 'No' to cancel the deletion.")
            }
          >
            Demo Confirm Deletion Dialog
          </Button>
        </div>
        <div className="mb-2 mr-2">
          <Button className="btn btn-primary" onClick={() => showDialog("Success Dialog Message", "success")}>
            Demo Success Message Dialog
          </Button>
        </div>
        <div className="mb-2 mr-2">
          <Button className="btn btn-primary" onClick={() => showDialog("Error Dialog Message", "error")}>
            Demo Error Message Dialog
          </Button>
        </div>
        <div className="mb-2 mr-2">
          <Button className="btn btn-primary" onClick={() => showDialog("Warn Dialog Message", "warn")}>
            Demo Warn Message Dialog
          </Button>
        </div>
        <div className="mb-2 mr-2">
          <Button className="btn btn-primary" onClick={() => showDialog("Info Dialog Message", "info")}>
            Demo Info Message Dialog
          </Button>
        </div>
      </div>
      <div className="flex flex-wrap">
        <div className="mb-2 mr-2">
          <Button className="btn btn-primary" onClick={() => showKeyVaultMessage()}>
            Demo Show Azure Key Vault Message
          </Button>
        </div>
        <div className="mb-2 mr-2">{keyVaultMessage}</div>
      </div>
      <div className="flex flex-wrap">
        <div className="mb-2 mr-2">
          <Button className="btn btn-primary" onClick={() => showTranslateMessage()}>
            Demo Show Server Side Message based on selected language
          </Button>
        </div>
        <div className="mb-2 mr-2">{translateMessage}</div>
      </div>

      <h4>🔧 Debug Authentication & Authorization</h4>
      <div className="flex flex-wrap">
        <div className="mb-2 mr-2">
          <Button className="btn btn-primary" onClick={() => debugCurrentUserRoles()}>
            🔍 Debug Current User Roles
          </Button>
        </div>
        <div className="mb-2 mr-2">
          <Button className="btn btn-secondary" onClick={() => debugHasRole("2")}>
            🎭 Check Administrator Role (2)
          </Button>
        </div>
        <div className="mb-2 mr-2">
          <Button className="btn btn-secondary" onClick={() => debugHasRole("1")}>
            🎭 Check User Role (1)
          </Button>
        </div>
        <div className="mb-2 mr-2">
          <Button className="btn btn-secondary" onClick={() => debugHasPermission("1")}>
            🔑 Check Login Permission (1)
          </Button>
        </div>
      </div>
      <div className="mb-3">
        <small className="text-muted">
          💡 Tip: You can also use these functions in the browser console:
          <br />• <code>window.debugAuth.logRoles()</code>
          <br />• <code>window.debugAuth.hasRole(roleId)</code>
          <br />• <code>window.debugAuth.hasPermission(permissionId)</code>
        </small>
      </div>

      <h4>Current Environment.</h4>
      <span>{process.env.REACT_APP_ENV}</span>
      <h4>Current config settings:</h4>
      <span>{JSON.stringify(APP_CONFIG)}</span>
    </>
  );
};
