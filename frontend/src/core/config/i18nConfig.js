import i18n from "i18next";
import { initReactI18next } from "react-i18next";

import Backend from "i18next-http-backend";
import LanguageDetector from "i18next-browser-languagedetector";
import APP_CONFIG from "./appConfig";

/**
 * Work for multiple languages support.
 *
 * Reference: https://www.cluemediator.com/implement-multi-languages-in-react
 */
// Log the basePath being used for i18n
console.log("🔧 i18n initialization - APP_CONFIG.basePath:", APP_CONFIG.basePath);

// Use PUBLIC_URL from environment or construct from basePath
// For development, use relative path; for production, use basePath
const publicUrl = process.env.PUBLIC_URL || APP_CONFIG.basePath || "";
const i18nLoadPath = `${publicUrl}/assets/i18n/{{ns}}/{{lng}}.json`;
console.log("🔧 i18n PUBLIC_URL:", publicUrl);
console.log("🔧 i18n loadPath:", i18nLoadPath);

i18n
  // load translation using http -> see /public/locales (i.e. https://github.com/i18next/react-i18next/tree/master/example/react/public/locales)
  // learn more: https://github.com/i18next/i18next-http-backend
  .use(Backend)
  // detect user language
  // learn more: https://github.com/i18next/i18next-browser-languageDetector
  .use(LanguageDetector)
  // pass the i18n instance to react-i18next.
  .use(initReactI18next)
  // init i18next
  // for all options read: https://www.i18next.com/overview/configuration-options
  .init({
    lng: "en",
    backend: {
      /* translation file path. Note: basePath default value = "/"  */
      loadPath: i18nLoadPath,
    },
    fallbackLng: "en",
    debug: false,
    /**
     * can have multiple namespace,
     * in case you want to divide a huge translation
     * into smaller pieces and load them on demand
     **/
    ns: ["translations"],
    defaultNS: "translations",
    // keySeparator: false, //Note: Diabled following line to support nested key in json files. (check json format in en.json or fr.json)
    interpolation: {
      escapeValue: false,
      formatSeparator: ",",
    },
    react: {
      useSuspense: false,
    },
  });

export default i18n;
