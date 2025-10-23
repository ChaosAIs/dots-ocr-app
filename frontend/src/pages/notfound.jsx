import React from "react";
import { useTranslation } from "react-i18next";

export const NotFound = () => {
  const { t } = useTranslation();
  return <h2>{t("Pages.NotFound")}</h2>;
};
