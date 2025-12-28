import React, { useState, useEffect, useRef } from "react";
import { useTranslation } from "react-i18next";
import { InputText } from "primereact/inputtext";
import { InputTextarea } from "primereact/inputtextarea";
import { Password } from "primereact/password";
import { Button } from "primereact/button";
import { Divider } from "primereact/divider";
import { Toast } from "primereact/toast";
import { ProgressSpinner } from "primereact/progressspinner";
import { useAuth } from "../core/auth/components/authProvider";
import authService from "../services/authService";
import "./userprofile.scss";

export const UserProfile = () => {
    const { t } = useTranslation();
    const { user, refreshUser } = useAuth();
    const toast = useRef(null);

    // Active tab state
    const [activeTab, setActiveTab] = useState("profile");

    // Profile form state
    const [profileData, setProfileData] = useState({
        email: "",
        full_name: "",
        phone_number: "",
        address: "",
        city: "",
        state: "",
        country: "",
        postal_code: "",
        bio: ""
    });

    // Password form state
    const [passwordData, setPasswordData] = useState({
        old_password: "",
        new_password: "",
        confirm_password: ""
    });

    // Loading states
    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [changingPassword, setChangingPassword] = useState(false);

    // Load profile data on mount
    useEffect(() => {
        loadProfile();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const loadProfile = async () => {
        setLoading(true);
        try {
            const result = await authService.getProfile();
            if (result.success) {
                setProfileData({
                    email: result.profile.email || "",
                    full_name: result.profile.full_name || "",
                    phone_number: result.profile.phone_number || "",
                    address: result.profile.address || "",
                    city: result.profile.city || "",
                    state: result.profile.state || "",
                    country: result.profile.country || "",
                    postal_code: result.profile.postal_code || "",
                    bio: result.profile.bio || ""
                });
            } else {
                toast.current.show({
                    severity: "error",
                    summary: t("Pages.UserProfile.Error"),
                    detail: result.error || t("Pages.UserProfile.LoadError"),
                    life: 4000
                });
            }
        } catch (error) {
            toast.current.show({
                severity: "error",
                summary: t("Pages.UserProfile.Error"),
                detail: t("Pages.UserProfile.LoadError"),
                life: 4000
            });
        } finally {
            setLoading(false);
        }
    };

    const handleProfileChange = (e) => {
        const { name, value } = e.target;
        setProfileData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handlePasswordChange = (e) => {
        const { name, value } = e.target;
        setPasswordData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleProfileSubmit = async (e) => {
        e.preventDefault();

        // Validate email
        if (!profileData.email) {
            toast.current.show({
                severity: "warn",
                summary: t("Pages.UserProfile.ValidationError"),
                detail: t("Pages.UserProfile.EmailRequired"),
                life: 3000
            });
            return;
        }

        setSaving(true);
        try {
            const result = await authService.updateProfile(profileData);
            if (result.success) {
                toast.current.show({
                    severity: "success",
                    summary: t("Pages.UserProfile.Success"),
                    detail: t("Pages.UserProfile.ProfileUpdated"),
                    life: 3000
                });
                // Refresh user context if available
                if (refreshUser) {
                    refreshUser();
                }
            } else {
                toast.current.show({
                    severity: "error",
                    summary: t("Pages.UserProfile.Error"),
                    detail: result.error || t("Pages.UserProfile.UpdateError"),
                    life: 4000
                });
            }
        } catch (error) {
            toast.current.show({
                severity: "error",
                summary: t("Pages.UserProfile.Error"),
                detail: t("Pages.UserProfile.UpdateError"),
                life: 4000
            });
        } finally {
            setSaving(false);
        }
    };

    const handlePasswordSubmit = async (e) => {
        e.preventDefault();

        // Validate passwords
        if (!passwordData.old_password || !passwordData.new_password) {
            toast.current.show({
                severity: "warn",
                summary: t("Pages.UserProfile.ValidationError"),
                detail: t("Pages.UserProfile.PasswordRequired"),
                life: 3000
            });
            return;
        }

        if (passwordData.new_password !== passwordData.confirm_password) {
            toast.current.show({
                severity: "warn",
                summary: t("Pages.UserProfile.ValidationError"),
                detail: t("Pages.UserProfile.PasswordMismatch"),
                life: 3000
            });
            return;
        }

        if (passwordData.new_password.length < 8) {
            toast.current.show({
                severity: "warn",
                summary: t("Pages.UserProfile.ValidationError"),
                detail: t("Pages.UserProfile.PasswordTooShort"),
                life: 3000
            });
            return;
        }

        setChangingPassword(true);
        try {
            const result = await authService.changePassword(
                passwordData.old_password,
                passwordData.new_password
            );
            if (result.success) {
                toast.current.show({
                    severity: "success",
                    summary: t("Pages.UserProfile.Success"),
                    detail: t("Pages.UserProfile.PasswordChanged"),
                    life: 3000
                });
                // Clear password fields
                setPasswordData({
                    old_password: "",
                    new_password: "",
                    confirm_password: ""
                });
            } else {
                toast.current.show({
                    severity: "error",
                    summary: t("Pages.UserProfile.Error"),
                    detail: result.error || t("Pages.UserProfile.PasswordChangeError"),
                    life: 4000
                });
            }
        } catch (error) {
            toast.current.show({
                severity: "error",
                summary: t("Pages.UserProfile.Error"),
                detail: t("Pages.UserProfile.PasswordChangeError"),
                life: 4000
            });
        } finally {
            setChangingPassword(false);
        }
    };

    if (loading) {
        return (
            <div className="flex align-items-center justify-content-center" style={{ height: 'calc(100vh - 60px)' }}>
                <ProgressSpinner />
            </div>
        );
    }

    const renderProfileTab = () => (
        <form onSubmit={handleProfileSubmit}>
            {/* Account Info Section */}
            <div className="profile-section">
                <h4 className="section-title">
                    <i className="pi pi-id-card"></i>
                    {t("Pages.UserProfile.AccountInfo")}
                </h4>

                <div className="grid">
                    <div className="col-12 md:col-6">
                        <div className="field">
                            <label htmlFor="username">{t("Pages.UserProfile.Username")}</label>
                            <InputText
                                id="username"
                                value={user?.username || ""}
                                disabled
                                className="w-full disabled-input"
                            />
                            <small>{t("Pages.UserProfile.UsernameCannotChange")}</small>
                        </div>
                    </div>
                    <div className="col-12 md:col-6">
                        <div className="field">
                            <label htmlFor="email">
                                {t("Pages.UserProfile.Email")} <span className="required">*</span>
                            </label>
                            <InputText
                                id="email"
                                name="email"
                                type="email"
                                value={profileData.email}
                                onChange={handleProfileChange}
                                className="w-full"
                                required
                            />
                        </div>
                    </div>
                </div>
            </div>

            <Divider />

            {/* Personal Info Section */}
            <div className="profile-section">
                <h4 className="section-title">
                    <i className="pi pi-user"></i>
                    {t("Pages.UserProfile.PersonalInfo")}
                </h4>

                <div className="grid">
                    <div className="col-12 md:col-6">
                        <div className="field">
                            <label htmlFor="full_name">{t("Pages.UserProfile.FullName")}</label>
                            <InputText
                                id="full_name"
                                name="full_name"
                                value={profileData.full_name}
                                onChange={handleProfileChange}
                                className="w-full"
                            />
                        </div>
                    </div>
                    <div className="col-12 md:col-6">
                        <div className="field">
                            <label htmlFor="phone_number">{t("Pages.UserProfile.PhoneNumber")}</label>
                            <InputText
                                id="phone_number"
                                name="phone_number"
                                value={profileData.phone_number}
                                onChange={handleProfileChange}
                                className="w-full"
                            />
                        </div>
                    </div>
                </div>

                <div className="field">
                    <label htmlFor="bio">{t("Pages.UserProfile.Bio")}</label>
                    <InputTextarea
                        id="bio"
                        name="bio"
                        value={profileData.bio}
                        onChange={handleProfileChange}
                        rows={3}
                        className="w-full"
                        autoResize
                    />
                </div>
            </div>

            <Divider />

            {/* Address Section */}
            <div className="profile-section">
                <h4 className="section-title">
                    <i className="pi pi-map-marker"></i>
                    {t("Pages.UserProfile.AddressInfo")}
                </h4>

                <div className="field">
                    <label htmlFor="address">{t("Pages.UserProfile.Address")}</label>
                    <InputText
                        id="address"
                        name="address"
                        value={profileData.address}
                        onChange={handleProfileChange}
                        className="w-full"
                    />
                </div>

                <div className="grid">
                    <div className="col-12 md:col-6">
                        <div className="field">
                            <label htmlFor="city">{t("Pages.UserProfile.City")}</label>
                            <InputText
                                id="city"
                                name="city"
                                value={profileData.city}
                                onChange={handleProfileChange}
                                className="w-full"
                            />
                        </div>
                    </div>
                    <div className="col-12 md:col-6">
                        <div className="field">
                            <label htmlFor="state">{t("Pages.UserProfile.State")}</label>
                            <InputText
                                id="state"
                                name="state"
                                value={profileData.state}
                                onChange={handleProfileChange}
                                className="w-full"
                            />
                        </div>
                    </div>
                </div>

                <div className="grid">
                    <div className="col-12 md:col-6">
                        <div className="field">
                            <label htmlFor="country">{t("Pages.UserProfile.Country")}</label>
                            <InputText
                                id="country"
                                name="country"
                                value={profileData.country}
                                onChange={handleProfileChange}
                                className="w-full"
                            />
                        </div>
                    </div>
                    <div className="col-12 md:col-6">
                        <div className="field">
                            <label htmlFor="postal_code">{t("Pages.UserProfile.PostalCode")}</label>
                            <InputText
                                id="postal_code"
                                name="postal_code"
                                value={profileData.postal_code}
                                onChange={handleProfileChange}
                                className="w-full"
                            />
                        </div>
                    </div>
                </div>
            </div>

            <div className="flex justify-content-end mt-4">
                <Button
                    type="submit"
                    label={t("Pages.UserProfile.SaveChanges")}
                    icon="pi pi-save"
                    loading={saving}
                />
            </div>
        </form>
    );

    const renderSecurityTab = () => (
        <div>
            <div className="profile-section">
                <h4 className="section-title">
                    <i className="pi pi-key"></i>
                    {t("Pages.UserProfile.ChangePassword")}
                </h4>

                <form onSubmit={handlePasswordSubmit}>
                    <div className="grid">
                        <div className="col-12 md:col-6">
                            <div className="field">
                                <label htmlFor="old_password">{t("Pages.UserProfile.CurrentPassword")}</label>
                                <Password
                                    id="old_password"
                                    name="old_password"
                                    value={passwordData.old_password}
                                    onChange={handlePasswordChange}
                                    className="w-full"
                                    inputClassName="w-full"
                                    feedback={false}
                                    toggleMask
                                />
                            </div>
                        </div>
                    </div>

                    <div className="grid">
                        <div className="col-12 md:col-6">
                            <div className="field">
                                <label htmlFor="new_password">{t("Pages.UserProfile.NewPassword")}</label>
                                <Password
                                    id="new_password"
                                    name="new_password"
                                    value={passwordData.new_password}
                                    onChange={handlePasswordChange}
                                    className="w-full"
                                    inputClassName="w-full"
                                    toggleMask
                                />
                                <small>{t("Pages.UserProfile.PasswordRequirements")}</small>
                            </div>
                        </div>
                        <div className="col-12 md:col-6">
                            <div className="field">
                                <label htmlFor="confirm_password">{t("Pages.UserProfile.ConfirmPassword")}</label>
                                <Password
                                    id="confirm_password"
                                    name="confirm_password"
                                    value={passwordData.confirm_password}
                                    onChange={handlePasswordChange}
                                    className="w-full"
                                    inputClassName="w-full"
                                    feedback={false}
                                    toggleMask
                                />
                            </div>
                        </div>
                    </div>

                    <div className="flex justify-content-end mt-4">
                        <Button
                            type="submit"
                            label={t("Pages.UserProfile.UpdatePassword")}
                            icon="pi pi-lock"
                            severity="warning"
                            loading={changingPassword}
                        />
                    </div>
                </form>
            </div>
        </div>
    );

    return (
        <div className="user-profile-container">
            <Toast ref={toast} />

            <div className="user-profile-content">
                {/* Header */}
                <div className="profile-header">
                    <h2>{t("Pages.UserProfile.Title")}</h2>
                    <p>{t("Pages.UserProfile.Subtitle")}</p>
                </div>

                {/* Custom Tab Navigation */}
                <div className="profile-tabs">
                    <button
                        type="button"
                        className={`profile-tab ${activeTab === 'profile' ? 'active' : ''}`}
                        onClick={() => setActiveTab('profile')}
                    >
                        <i className="pi pi-user"></i>
                        <span>{t("Pages.UserProfile.ProfileTab")}</span>
                    </button>
                    <button
                        type="button"
                        className={`profile-tab ${activeTab === 'security' ? 'active' : ''}`}
                        onClick={() => setActiveTab('security')}
                    >
                        <i className="pi pi-lock"></i>
                        <span>{t("Pages.UserProfile.SecurityTab")}</span>
                    </button>
                </div>

                {/* Tab Content */}
                <div className="profile-tab-content">
                    {activeTab === 'profile' && renderProfileTab()}
                    {activeTab === 'security' && renderSecurityTab()}
                </div>
            </div>
        </div>
    );
};
