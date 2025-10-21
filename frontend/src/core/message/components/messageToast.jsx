import React, { Component } from "react";
import { messageService } from "../messageService";
import { Toast } from "primereact/toast";

class MessageToast extends Component {
  life = 3000;
  messages = [];

  componentDidMount() {
    this.subscription = messageService.get().subscribe((message) => {
      if (message && message.modalType === "toast") {
        if (message.content !== "IsEmitNotify") {
          //
          // Save the received message into temporary message collection.
          //
          this.messages.push(message);
          if (message.isEmit) {
            //
            // Show the messages in the snackbars message boxes.
            //
            this.showMessages();
          }
        } else {
          this.showMessages();
        }
      }
    });
  }

  showMessages() {
    if (this.messages.length > 0) {
      let primengMessages = [];
      this.messages.map((message) =>
        primengMessages.push({
          severity: message.messageType,
          life: this.life,
          detail: message.content,
        })
      );
      this.msgs.show(primengMessages);
      this.messages = [];
    }
  }

  componentWillUnmount() {
    // unsubscribe to ensure no memory leaks
    this.subscription.unsubscribe();
  }

  /**
   * Clean up the messages from collection, hide the messages in UI.
   */
  cleanupMessages = () => {
    this.message = [];
    if (this.msgs) this.msgs.clear();
  };

  render() {
    return (
      <React.Fragment>
        <Toast ref={(el) => (this.msgs = el)} />
      </React.Fragment>
    );
  }
}

export default MessageToast;
