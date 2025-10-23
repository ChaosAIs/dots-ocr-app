import React, { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import APP_CONFIG from "../core/config/appConfig";
import { ResultStatus } from "../core/enumertions/resultStatus";
import http from "../core/http/httpClient";

export const UserManagement = () => {
  const { t } = useTranslation();
  const [users, setUsers] = useState([]);
  useEffect(() => {
    //console.log("UserManagement called : " + APP_CONFIG.apiDoamin);
    http.get(APP_CONFIG.apiDomain + "/api/user/getusers").then((response) => {
      if (
        response.data &&
        response.data.resultStatus === ResultStatus.Success
      ) {
        setUsers(response.data.item);
      }
    });
  }, []);

  return (
    <>
      <h2>{t("Pages.UserManagement.Title")}</h2>
      <ul>
        {users.map((user) => {
          return <li key={user.id}>{user.userName}</li>;
        })}
      </ul>
    </>
  );
};
