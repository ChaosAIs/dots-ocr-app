import React, { Component } from "react";
import { messageService } from "../messageService";
import { Messages } from "primereact/messages";


class MessageBox extends Component {
  timeout = 5000;
  messages = [];

  messageBoxTimmer = {} ;

  componentDidMount() {
    this.subscription = messageService.get().subscribe((message) => {
      if (message && message.modalType === "snackbar") {
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
      this.messages.map((message) => (
        primengMessages.push({
          severity: message.messageType,
          sticky: true,
          detail: message.content,
        })
      ));
      this.msgs.show(primengMessages);
      this.messages = [];

      //
      // show the messages on messagebox for several seconds then close the messages boxes.
      //
      if (this.messageBoxTimmer){
          clearTimeout(this.messageBoxTimmer);
      }

      this.messageBoxTimmer = setTimeout(() => {
        this.cleanupMessages();
      }, this.timeout);
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
        <Messages ref={(el) => (this.msgs = el)}/>
      </React.Fragment>
    );
  }
}

export default MessageBox;
