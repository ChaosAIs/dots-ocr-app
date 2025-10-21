import APP_CONFIG from "../../core/config/appConfig";
import { Language } from "../../core/enumertions/language";
import CORE_ACTIONS from "../actions";

/**
 * language setup reducer. Keep the selected language code in local storage.
 */
const languageReducer = (state = Language.English, action) => {
  switch (action.type) {
    //
    // When selected langauge from the langauge drop down on the top menu, following action got triggered.
    //
    case CORE_ACTIONS.SET_LANGUAGE:
      //
      // Keep the selected language in the local storage.
      // Help httpClient to get current selected langauge and inject langauge into the http request header.
      // corpoate with httpClient.js
      //
      localStorage.setItem(
        `${APP_CONFIG.clientId}_CurrentLanguage`,
        action.payload
      );

      return action.payload; // it is new selected language code.
    default:
      return state;
  }
};

export default languageReducer;
