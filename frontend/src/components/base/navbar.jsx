import React, { useContext, useEffect } from "react";
import { Menubar } from "primereact/menubar";
import { Dropdown } from "primereact/dropdown";
import { Button } from "primereact/button";
import { AuthContext } from "../../core/auth/components/authProvider";
import APP_CONFIG from "../../core/config/appConfig";
import { loadingService } from "../../core/loading/loadingService";
import { useTranslation } from "react-i18next";
import { useSelector, useDispatch } from "react-redux";
import { bindActionCreators } from "redux";
import { actionCreators } from "../../redux/actionCreatorsExport";
import { Role } from "../../core/enumertions/role";

export const NavBar = () => {
  //
  // Try to get selected language code from the global redux store.
  //
  const state = useSelector((state) => state);
  const dispatch = useDispatch();
  //
  // Binding Redux custom action creators.
  // Work for update selected language code.
  //
  const { setLanguage } = bindActionCreators(actionCreators, dispatch);

  const authService = useContext(AuthContext);
  const user = authService.getUser();
  const isAuthorized = authService.isAuthenticated();
  const { t, i18n } = useTranslation();

  useEffect(() => {
    if (state.language && state.language.length > 0) {
      i18n.changeLanguage(state.language);
    }
  }, [i18n, state]);

  // Check if user has Administrator role
  const isSystemAdmin = isAuthorized && user?.roles?.includes(Role.Administrator);

  const items = [
    {
      label: t("Nav.Home"),
      icon: "pi pi-fw pi-home",
      url: `${APP_CONFIG.basePath}/home`,
      visible: true, // It is custom property. Work for show/hide menu items based on user authentication.
    },
    {
      label: "Admin",
      icon: "pi pi-fw pi-cog",
      visible: isSystemAdmin,
      items: [
        {
          label: "Partner Reference Data",
          icon: "pi pi-fw pi-users",
          url: `${APP_CONFIG.basePath}/admin/data-management`,
        }
      ],
    },
  ];

  const languages = [
    { label: t("Nav.English"), value: "en" },
    { label: t("Nav.French"), value: "fr" },
  ];

  const login = () => {
    // Show progress spinner.
    loadingService.httpRequestSent();
    authService.signin();
  };

  const logout = () => {
    // Show progress spinner.
    loadingService.httpRequestSent();
    authService.signout();
  };

  const languageDropdown = (
    <Dropdown
      value={state.language}
      options={languages}
      onChange={(e) => {
        i18n.changeLanguage(e.value);
        setLanguage(e.value);
      }}
      placeholder="Select a language"
    />
  );

  const start = (
    <div className="navbar-brand-container">
      <img
        alt="logo"
        src={`${APP_CONFIG.basePath}/logo.png`}
      />
      <span className="navbar-brand-text">       
        {t("Nav.DocumentConversion")}
      </span>
    </div>
  );
  const end = (
    <>
      {isAuthorized && <span className="p-2">{user.displayName}</span>}
      {languageDropdown} 
      {isAuthorized && (
        // it is logout button.
        <Button label="Logout" rounded className="p-button-gray p-button-rounded"
          title={t("Nav.Logout")}
          onClick={logout}
        ></Button>
      )}
      {!isAuthorized && (
        // It is login button.
        <Button label="Login" rounded className="p-button-gray"
          title={t("Nav.Login")}
          onClick={login}
        ></Button>
      )}
    </>
  );

  return (
    <>
      <div className="navbar-wrapper">
        <div className="navbar-container">
          <Menubar
            model={items.filter((i) => i.visible === true)}
            start={start}
            end={end}
          ></Menubar>
        </div>
      </div>
    </>
  );
};
