import React, { useContext } from "react";
import { Navigate } from "react-router-dom";
import { AuthContext } from "../core/auth/components/authProvider";
import { messageService } from "../core/message/messageService";

/**
 * Route required authentication.
 *
 * Reference: https://github.com/Franpastoragusti/oidc-react-app
 *
 * TODO. Check roles and permissions here.
 * @param {*} param0
 * @returns
 */
export const PrivateRoute = ({ component: Component, roles, ...rest }) => {
  const authService = useContext(AuthContext);
  //
  // Demo get roles for the specified Route.
  // TODO. Developer is able to udpate the authorization check logic here.
  //
  const requiredRoles = roles;

  if (!!Component && authService.isAuthenticated()) {
    //
    // TODO. We can do further checking here with logon user's Roles, Permissions etc.
    // TODO. Further Authorization check here...
    //
    const user = authService.getUser();
    const userRoles = user ? user.roles : [];

    // console.log("PrivateRoute.UserRoles", JSON.stringify(userRoles));

    if (requiredRoles && requiredRoles.length > 0) {
      if (userRoles && userRoles.length > 0) {
        let result = requiredRoles.some((r1) => {
          if (
            userRoles.some((r2) => {
              return r2 === r1;
            })
          )
            return true;

          return false;
        });

        if (result) {
          return <Component {...rest} />;
        } else {
          messageService.warnToast(
            "You do not have permission to access.",
            false
          );

          // If not permission, by default go back to home page. Note: Developer can change this.
          return (
            <>
              <span>Unauthorized access</span>
              {messageService.emit()}
            </>
          );
        }
      } else {
        messageService.warnToast(
          "You do not have permission to access",
          false
        );

        // If no permission, by default go back to home page. Note: Developer can change this.
        return (
          <>
            <span>Unauthorized access</span>
            {messageService.emit()}
          </>
        );
      }
    } else {
      return <Component {...rest} />;
    }
  } else {
    //
    // User not authenticated - redirect to login page
    //
    console.warn("User not authenticated, redirecting to login");
    return <Navigate to="/login" replace />;
  }
};
