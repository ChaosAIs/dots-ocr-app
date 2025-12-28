import React, { useContext, useEffect, useRef } from "react";
import { Menubar } from "primereact/menubar";
import { Dropdown } from "primereact/dropdown";
import { Button } from "primereact/button";
import { Menu } from "primereact/menu";
import { AuthContext } from "../../core/auth/components/authProvider";
import APP_CONFIG from "../../core/config/appConfig";
import { loadingService } from "../../core/loading/loadingService";
import { useTranslation } from "react-i18next";
import { useSelector, useDispatch } from "react-redux";
import { bindActionCreators } from "redux";
import { actionCreators } from "../../redux/actionCreatorsExport";
import { Role } from "../../core/enumertions/role";
import { useNavigate } from "react-router-dom";
import { ThemePicker } from "./ThemePicker";

export const NavBar = () => {
  //
  // Try to get selected language code from the global redux store.
  //
  const language = useSelector((state) => state.language);
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const userMenuRef = useRef(null);
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
      label: t("Nav.Chat"),
      icon: "pi pi-fw pi-comments",
      url: `${APP_CONFIG.basePath}/chat`,
      visible: true, // Chat is available to all users
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

  const logout = async () => {
    // Show progress spinner.
    loadingService.httpRequestSent();
    try {
      if (authService && typeof authService.logout === 'function') {
        await authService.logout();
        // Redirect to login page after logout
        navigate('/login');
      }
      loadingService.httpResponseReceived();
    } catch (error) {
      console.error('Logout error:', error);
      loadingService.error();
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
    <span className="text-xl font-semibold text-primary white-space-nowrap">
      {t("Nav.DocumentConversion")}
    </span>
  );

  const userMenuItems = [
    {
      label: t("Nav.Profile"),
      icon: "pi pi-user",
      command: () => navigate('/userprofile')
    },
    {
      separator: true
    },
    {
      label: t("Nav.Logout"),
      icon: "pi pi-sign-out",
      command: logout
    }
  ];

  const end = (
    <div className="flex align-items-center gap-2">
      {languageDropdown}
      {isAuthorized && <ThemePicker />}
      {isAuthorized ? (
        <div className="flex align-items-center gap-2">
          <Button
            label={user?.full_name || user?.username || "User"}
            icon="pi pi-user"
            outlined
            onClick={(e) => userMenuRef.current.toggle(e)}
            aria-controls="user-menu"
            aria-haspopup
          />
          <Menu model={userMenuItems} popup ref={userMenuRef} id="user-menu" />
        </div>
      ) : (
        <Button
          label="Login"
          icon="pi pi-sign-in"
          outlined
          title="Login"
          onClick={() => navigate('/login')}
        />
      )}
    </div>
  );

  return (
    <Menubar
      model={items.filter((i) => i.visible === true)}
      start={start}
      end={end}
    />
  );
};
