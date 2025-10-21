/** Define current running application's environment.
 * value = "dev_cors" or "prod_nocors". 
 * Other settings get from server side api
 * */
const APP_ENV = process.env.REACT_APP_ENV;
const APP_CONFIG_API = process.env.REACT_APP_CONFIG_API;

console.log("APP_CONFIG_API", APP_CONFIG_API);

let app_config = null;

const getConfig = function () {
  if (app_config) return app_config;

  // Validate required environment variables
  if (!APP_ENV || !APP_CONFIG_API) {
    console.error("Missing required environment variables: REACT_APP_ENV or REACT_APP_CONFIG_API");
    return null;
  }

  // create a new XMLHttpRequest
  var xhr = new XMLHttpRequest();
  // get a callback when the server responds
  xhr.addEventListener("load", () => {
    //console.log("config", xhr.responseText);
    app_config = JSON.parse(xhr.responseText);
    // update the state of the component with the result here
    console.log(xhr.responseText);
  });

  var domain = window.location.protocol + "//" + window.location.host;

  //
  // Note:
  // As for APP_ENV = "dev_cors"
  // Hardcode url as "https://localhost:5001/api/Settings/GetIDSSettings",
  // since it only works for local machine development environment.
  // Send request to web api endpoint to get react app's config settings.
  // Note: Since React App is hosting in seperated domain (or different port number),
  // so, here, need to access web api end point's full url.
  // Get config settings from web api end point defined in "REACT_APP_CONFIG_API"
  // Setings stay in web api's appsettings.[environment].json file, section "Environment".
  //
  // As for APP_ENV = "prod_nocors"
  // Work for scenario of hosting react app under web api domain's sub path
  // Send request to web api endpoint to get react app's config settings.
  // Note: Since react app is hosting under web api domain sub path.
  // so, here, it can use relative url.
  //
  if (APP_ENV.indexOf("_cors") > 0) {
    // For any "**_cors" environments, need to get web api endpoint full path from environment file.
    // Currently, this setting only work for local development environment and corporate with environment file called ".env.development.local".
    xhr.open("GET", APP_CONFIG_API, false);
  } else {
    // For any "**_nocors" environments, call web api endpoint with current domain.
    // corporate with environment file called ".env.production.local".
    xhr.open("GET", domain + APP_CONFIG_API, false);
  }

  // send the request
  xhr.send();

  let stillWating = true;
  //we only wait for 30s to avoid chrome freeze
  setTimeout(() => {
    stillWating = false;
  }, 30000);

  while (!app_config && stillWating) {
    // Holding here.
  }

  return app_config;
};

const APP_CONFIG = getConfig();
/**
 *  Contains global config settings for application associated
 *  to current running environment.
 *
 *  As for debugging system:
 *  Modify setting in env.development.local before run: npm start
 *
 *  As for build system:
 *  Modify setting in env.production.local before run: npm build
 */

export default APP_CONFIG;
