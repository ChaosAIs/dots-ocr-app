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
  const language = useSelector((state) => state.language);
  const dispatch = useDispatch();
  //
  // Binding Redux custom action creators.
  // Work for update selected language code.
  //
  const { setLanguage } = bindActionCreators(actionCreators, dispatch);

  const authService = useContext(AuthContext);
  const user = authService && typeof authService.getUser === 'function' ? authService.getUser() : null;
  const isAuthorized = authService && typeof authService.isAuthenticated === 'function' ? authService.isAuthenticated() : false;
  const { t, i18n } = useTranslation();

  useEffect(() => {
    if (language && language.length > 0) {
      i18n.changeLanguage(language);
    }
  }, [i18n, language]);

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
      label: t("Nav.Admin"),
      icon: "pi pi-fw pi-cog",
      visible: isSystemAdmin,
      items: [
        {
          label: t("Nav.DocumentManagement"),
          icon: "pi pi-fw pi-users",
          url: `${APP_CONFIG.basePath}/admin/data-management`,
        }
      ],
    },
  ];

  // Language options with hardcoded labels to avoid translation timing issues
  const languages = [
    { label: "English", value: "en" },
    { label: "Français", value: "fr" },
  ];

  const logout = () => {
    // Show progress spinner.
    loadingService.httpRequestSent();
    if (authService && typeof authService.signout === 'function') {
      authService.signout();
    }
  };

  const languageDropdown = (
    <Dropdown
      value={language}
      options={languages}
      onChange={(e) => {
        i18n.changeLanguage(e.value);
        setLanguage(e.value);
      }}
      placeholder={t("Nav.SelectLanguage")}
    />
  );

  const start = (
    <div className="navbar-brand-container">
      <span className="navbar-brand-text">       
        {t("Nav.DocumentConversion")}
      </span>
    </div>
  );
  const end = (
    <>
      {isAuthorized && <span className="p-2">{user?.displayName || "User"}</span>}
      {languageDropdown}
      {/* Always show logout button since we're using temporary auth */}
      {/* <Button label={t("Nav.Logout")} rounded className="p-button-gray p-button-rounded"
        title={t("Nav.Logout")}
        onClick={logout}
      ></Button> */}
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
