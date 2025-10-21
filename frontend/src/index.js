import React from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App";
import "./core/config/i18nConfig";
import { Suspense } from "react";
import { Provider } from "react-redux";
import { store } from "./redux/store";

// import reportWebVitals from './reportWebVitals';

const container = document.getElementById("root");
const root = createRoot(container);

root.render(
  //
  // Note: React.StrictMode only applies to Development mode. In production mode, it will not impact the production build.
  // Note: StrictMode affect axios to send two request (get,post) one time. It is about "prelight request"
  // TEMPORARILY DISABLED StrictMode to prevent duplicate API calls during development
  //
  // <React.StrictMode>
    <Suspense fallback={<span>Loading</span>}>
      {/*
       * Reference: https://stackoverflow.com/questions/36212860/subscribe-to-single-property-change-in-store-in-redux
       * Means the child components are able to access the redux store.
       */}
      <Provider store={store}>
        <App />
      </Provider>
    </Suspense>
  // </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
//reportWebVitals();
