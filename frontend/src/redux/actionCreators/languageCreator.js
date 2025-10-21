import CORE_ACTIONS from "../actions";

/**
 * Action creator for setup current language.
 *
 * https://www.youtube.com/watch?v=9jULHSe41ls
 *
 * @param {*} selectedLangauge It is selected langauge code. value = "en" or "fr".
 * @returns
 */
export const setLanguage = (selectedLangauge) => {
  return (dispatch) => {
    dispatch({
      /** type is a generic name.
       *  It is defined which action routine
       *  will be applied to update the state. */
      type: CORE_ACTIONS.SET_LANGUAGE,
      /** payload is a generic name.
       * It is container to transfer date which is required to be applied to the state.
       * */
      payload: selectedLangauge,
    });
  };
};
