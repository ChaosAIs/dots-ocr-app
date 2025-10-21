import { Subject } from "rxjs";

/** Work for subscribe message object crossing components
 *  No need to export this const, since it only works with messageService.
 */
const messageSubject = new Subject();

/** Temporary collection pool to keep all isEmit = false messages.
 *  Work for emit messages to the redirected page.
 *  Corporate with "delayEmit" method.
 * */
const messageSubjectPool = [];

/***
 * Reference: https://jasonwatmore.com/post/2019/02/13/react-rxjs-communicating-between-components-with-observable-subject#:~:text=React%20%2B%20RxJS%20App%20Component%20that,divs%20in%20the%20render%20method.
 * Note: This is a single tone service approach.
 * https://www.digitalocean.com/community/tutorials/js-js-singletons
 */
export const messageService = {
  /**
   * Subscribe "infor" messages to messageBox component only.
   * @info:  It is string content of the message.
   * @isEmit: default value = false, keep info messages in temp colleciton first without show up in UI.
   *          if isEmit = true, current message and other keeping messags will be show to UI immediately and temp collection in messageBox component will be clean up.
   *
   *
   *  Message json format:
   *
   * {
   *    content: '',
   *    messageType: 'info' or 'success' or 'warn' or 'error' or 'confirmation' or 'deletionConfirmation'.
   *    modalType: 'dialog' or 'snackbar' or 'toast',
   *    isEmit: true/false, (Define if message show up immediately or keep it in temp collection first without show)
   *    callback: (result) => {} // It is function reference passed from caller component through MessageService's subscribe.
   * }
   *
   * */
  info: (info = "", isEmit = true) => {
    sendMessage(info, isEmit, "snackbar", "info");
  },

  warn: (warn = "", isEmit = true) => {
    sendMessage(warn, isEmit, "snackbar", "warn");
  },

  error: (error = "", isEmit = true) => {
    sendMessage(error, isEmit, "snackbar", "error");
  },

  success: (success = "", isEmit = true) => {
    sendMessage(success, isEmit, "snackbar", "success");
  },

  infoToast: (info = "", isEmit = true) => {
    sendMessage(info, isEmit, "toast", "info");
  },

  warnToast: (warn = "", isEmit = true) => {
    sendMessage(warn, isEmit, "toast", "warn");
  },

  errorToast: (error = "", isEmit = true) => {
    sendMessage(error, isEmit, "toast", "error");
  },

  successToast: (success = "", isEmit = true) => {
    sendMessage(success, isEmit, "toast", "success");
  },

  /** Dialog to show information message only. */
  infoDialog: (info = "", title="info") => {
    sendMessage(info, true, "dialog", title);
  },

  /** Dialog to show warn message only. */
  warnDialog: (warn = "", title="warn") => {
    sendMessage(warn, true, "dialog", title);
  },

  /** Dialog to show error message only. */
  errorDialog: (error = "", title="error") => {
    sendMessage(error, true, "dialog", title);
  },

  /** Dialog to show success message only. */
  successDialog: (success = "", title="success") => {
    sendMessage(success, true, "dialog", title);
  },

  /** Confirm dialog. It has Yes and No two buttons */
  confirmDialog: (content = "", callback) => {
    sendMessageWithCallback(content, true, "dialog", "confirmation", callback);
  },

  /** Specialy for confirm deletion action process. It has Yes and No two buttons.  */
  confirmDeletionDialog: (content = "", callback) => {
    sendMessageWithCallback(
      content,
      true,
      "dialog",
      "deletionConfirmation",
      callback
    );
  },
  
  /**
   *  Try to listen and receive pop messages from subscribe.
   *  Message object json format:
   *
   *  {
   *    content: '',
   *    messageType: 'info' or 'success' or 'warn' or 'error' or 'confirmation' or 'deletionConfirmation'.
   *    modalType: 'dialog' or 'snackbar' or 'toast',
   *    isEmit: true/false,
   *    // callback function work for confirmation dialog (yes and no choice)
   *    callback?: (result) => {}
   *  }
   *
   */
  get: () => {
    return messageSubject.asObservable();
  },

  /** Emit holding messages from collection pool, and display them in UI immediately. */
  emit: () => {
    setTimeout(() => {
      if (messageSubjectPool && messageSubjectPool.length > 0) {
        while (messageSubjectPool.length > 0) {
          const m = messageSubjectPool.pop();
          m.isEmit = true;
          messageSubject.next(m);
        }
      }

      const snackBarNotify = {
        content: "IsEmitNotify",
        modalType: "snackbar",
        messageType: "info",
        isEmit: true,
      };

      const dialogNotify = {
        content: "IsEmitNotify",
        modalType: "dialog",
        messageType: "info",
        isEmit: true,
      };

      const toastNotify = {
        content: "IsEmitNotify",
        modalType: "toast",
        messageType: "info",
        isEmit: true,
      };

      messageSubject.next(snackBarNotify);
      messageSubject.next(dialogNotify);
      messageSubject.next(toastNotify);
    }, 200);
  },
};

function sendMessage(messageContent, isEmit, modelType, messageType) {
  if (messageContent && messageContent.length > 0) {
    const newMessage = {
      content: messageContent,
      modalType: modelType,
      messageType: messageType,
      isEmit: isEmit,
    };
    if (isEmit) {
      // Subscribe the message to components immediately.
      messageSubject.next(newMessage);
    } else {
      //
      // Keep the message in temporary collection pool.
      //
      messageSubjectPool.push(newMessage);
    }
  }
}

/**
 * Only work for confirmation dialog at this moment.
 * @param {*} messageContent
 * @param {*} isEmit
 * @param {*} modelType
 * @param {*} messageType
 * @param {*} callback callback function work for send back confirmation dialog response back to caller component.
 */
function sendMessageWithCallback(
  messageContent,
  isEmit,
  modelType,
  messageType,
  callback
) {
  if (messageContent && messageContent.length > 0) {
    const newMessage = {
      content: messageContent,
      modalType: modelType,
      messageType: messageType,
      isEmit: isEmit,
      callback: callback,
    };
    if (isEmit) {
      //
      // Subscribe the message to components immediately.
      //
      messageSubject.next(newMessage);
    } else {
      //
      // Keep the message in temporary collection pool.
      //
      messageSubjectPool.push(newMessage);
    }
  }
}
