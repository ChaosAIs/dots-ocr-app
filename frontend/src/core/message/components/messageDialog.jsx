import React, { Component } from "react";
import { messageService } from "../messageService";
import { Button } from "primereact/button";
import { Dialog } from "primereact/dialog";
import { confirmDialog } from "primereact/confirmdialog";
import "./messageDialog.scss";

class MessageDialog extends Component {
  constructor(props) {
    super(props);
    this.state = {
      visible: false,
      content: "",
      header: "",
      messageType: "info",
    };

    this.accept = this.accept.bind(this);
    this.reject = this.reject.bind(this);
  }

  /** It is reference of method passed from message service's caller,
   * which is component calling messageService to show confirm dialog.
   * */
  callback = (result) => {};

  /** It is "ok" button click in confirmation dialog. */
  accept() {
    this.callback(true);
  }

  /** It is "no" button click in confirmation dialog. */
  reject() {
    this.callback(false);
  }

  componentDidMount() {
    this.subscription = messageService.get().subscribe((message) => {
      if (message && message.modalType === "dialog") {
        this.callback = message.callback;
        // Note: Dialog message must be shown up immediately.
        this.showMessage(message);
      }
    });
  }

  componentWillUnmount() {
    // unsubscribe to ensure no memory leaks
    this.subscription.unsubscribe();
  }

  showMessage(message) {
    if (message.content !== "IsEmitNotify") {
      switch (message.messageType) {
        case "confirmation":
          // Display generic confirmation dialog with "Yes" and "No" buttons
          confirmDialog({
            className: "appDialog-info",
            message: <div id="appDialogContent">{this.renderContent("info", message.content)}</div>,
            header: "Confirmation",
            accept: this.accept,
            reject: this.reject,
          });
          break;
        case "deletionConfirmation":
          //
          // Display deletion confirmation dialog with "Yes" and "No" buttons. Style sheet a little bit difference with "confirmation" dialog.
          //
          confirmDialog({
            className: "appDialog-error",
            message: <div id="appDialogContent">{this.renderContent("info", message.content)}</div>,
            header: "Delete Confirmation",
            acceptClassName: "p-button-danger",
            accept: this.accept,
            reject: this.reject,
          });
          break;
        default:
          //
          // Display generic message dialog with "Ok" button only.
          //
          this.setState({
            visible: true,
            content: message.content,
            header: message.messageType,
            messageType: message.messageType,
          });
          break;
      }
    }
  }

  onHide(name) {
    this.setState({
      [`${name}`]: false,
    });
  }

  /** Work for generic message dialog footer rendering. Only show "Ok" button. */
  renderFooter(name) {
    return (
      <div>
        {/* <Button
          label="No"
          icon="pi pi-times"
          onClick={() => this.onHide(name)}
          className="p-button-text"
        /> */}
        <Button label="Ok" className="p-button-primary ok-button" onClick={() => this.onHide(name)} autoFocus />
      </div>
    );
  }

  renderHeader(title) {
    return <div className="appDialogHeader">{title}</div>;
  }

  renderContent(messageType, content) {
    switch (messageType) {
      case "success":
        return (
          <p>
            <i className="pi pi-check-circle" />
            &nbsp;&nbsp;{content}
          </p>
        );
      case "error":
        return (
          <p>
            <i className="pi pi-times-circle" />
            &nbsp;&nbsp;{content}
          </p>
        );
      case "warn":
        return (
          <p>
            <i className="pi pi-exclamation-circle" />
            &nbsp;&nbsp;{content}
          </p>
        );
      case "info":
        return (
          <p>
            <i className="pi pi-info-circle" />
            &nbsp;&nbsp;{content}
          </p>
        );
      default:
        return (
          <p>
            <i className="pi pi-info-circle" />
            &nbsp;&nbsp;{content}
          </p>
        );
    }
  }

  render() {
    return (
      <>
        <Dialog
          className={`appDialog-${this.state.messageType}`}
          header={this.renderHeader(this.state.header)}
          visible={this.state.visible}
          style={{ width: "50vw" }}
          footer={this.renderFooter("visible")}
          onHide={() => this.onHide("visible")}
        >
          <div id="appDialogContent">{this.renderContent(this.state.messageType, this.state.content)}</div>
        </Dialog>
      </>
    );
  }
}

export default MessageDialog;
