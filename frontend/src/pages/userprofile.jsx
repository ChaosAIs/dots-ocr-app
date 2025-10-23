import React, { useContext } from "react";
import { useTranslation } from "react-i18next";
import { AuthContext } from "../core/auth/components/authProvider";

export const UserProfile = () => {
  const { t } = useTranslation();
  const authService = useContext(AuthContext);
  const user = authService && typeof authService.getUser === 'function' ? authService.getUser() : null;
  return (
    <>
      <h2>{t("Pages.UserProfile.Title")}</h2>{t("Pages.UserProfile.CurrentUser")}
      {user ? user.userName : t("Pages.UserProfile.NotAuthenticated")}
    </>
  );
};
