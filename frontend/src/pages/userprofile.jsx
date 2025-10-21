import React, { useContext } from "react";
import { AuthContext } from "../core/auth/components/authProvider";

export const UserProfile = () => {
  const authService = useContext(AuthContext);
  const user = authService && typeof authService.getUser === 'function' ? authService.getUser() : null;
  return (
    <>
      <h2>It is user profile page</h2>Current logon user name:
      {user ? user.userName : "Not authenticated"}
    </>
  );
};
