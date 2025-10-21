import APP_CONFIG from "./appConfig";

/**
 *  Identity Server 4 endpoints config.
 *  Reference: https://medium.com/@franciscopa91/how-to-implement-oidc-authentication-with-react-context-api-and-react-router-205e13f2d49
 */
export const IDENTITY_META_CONFIG = {
  issuer: APP_CONFIG.iamDomain,
  jwks_uri: APP_CONFIG.iamDomain + "/.well-known/openid-configuration/jwks",
  authorization_endpoint: APP_CONFIG.iamDomain + "/connect/authorize",
  token_endpoint: APP_CONFIG.iamDomain + "/connect/token",
  userinfo_endpoint: APP_CONFIG.iamDomain + "/connect/userinfo",
  end_session_endpoint: APP_CONFIG.iamDomain + "/connect/endsession",
  check_session_iframe: APP_CONFIG.iamDomain + "/connect/checksession",
  revocation_endpoint: APP_CONFIG.iamDomain + "/connect/revocation",
  introspection_endpoint: APP_CONFIG.iamDomain + "/connect/introspect",
};
