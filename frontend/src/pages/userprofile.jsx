import React, { useContext } from "react";
import { AuthContext } from "../core/auth/components/authProvider";

export const UserProfile = () => {
  const authService = useContext(AuthContext);
  return (
    <>
      <h2>It is user profile page</h2>Current logon user name:
      {authService.getUser().userName}
    </>
  );
};
