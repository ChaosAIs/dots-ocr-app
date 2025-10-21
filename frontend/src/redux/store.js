import { createStore, applyMiddleware } from "redux";
import { thunk } from "redux-thunk";
import logger from "redux-logger";
import { persistStore, persistReducer} from "redux-persist";
import storage from "redux-persist/lib/storage";
import rootReducer from "./reducers/rootReducer";

/**
 *  Reference: https://edisondevadoss.medium.com/how-to-use-redux-persist-in-react-application-35943c1d8292
 * https://www.youtube.com/watch?v=9jULHSe41ls
 */
const persistConfig = {
  key: "language",
  storage: storage,
  //
  // It (whitelist) ensures which reducer want to save in persistence storage,
  // and rest of the reducers are not save in persistence storage.
  //
  whitelist: ["language"],
};
const pReducer = persistReducer(persistConfig, rootReducer);
//const middleware = applyMiddleware(thunk, logger);
const middleware = applyMiddleware(thunk, logger);

/**
 * Keeping the selected language in local storage through Redux.
 * Reference: https://edisondevadoss.medium.com/how-to-use-redux-persist-in-react-application-35943c1d8292
 */
const store = createStore(pReducer, middleware);

// /** Every time redux state store got updated, 
//  * notify system to save the current langauge code into localstorage of browser.
//  * */
// store.subscribe(()=>{
//   var state = store.getState();

//   //
//   // Keep the selected language in the local storage. 
//   // Help httpClient to get current selected langauge and inject langauge into the http request header.
//   // corpoate with httpClient.js
//   //
//   localStorage.setItem(`${APP_CONFIG.clientId}_CurrentLanguage`, state.language);

// });

/**
 * Persistor keeps the root store for Redux.
 */
const persistor = persistStore(store);

export { persistor, store };
