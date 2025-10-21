import { combineReducers } from "redux";
import languageReducer from "./languageReducer";

/** Combine all custom reducers as one.
 *  Note: We are able to have multiple reducers.
 */
const rootReducer = combineReducers({
   /** Work for multiple language support. keep the selected language code crossing components and pages. */
   language: languageReducer 
});

export default rootReducer;