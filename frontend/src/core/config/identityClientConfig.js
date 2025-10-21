import APP_CONFIG from "./appConfig";

/**
 * Identity Server 4 client associated config.
 *
 */
export const IDENTITY_CLIENT_CONFIG = {
  authority: APP_CONFIG.iamDomain, //(string): The URL of the OIDC provider.
  client_id: APP_CONFIG.clientId, //(string): Your client application's identifier as registered with the OIDC provider.
  redirect_uri:
    APP_CONFIG.basePath.length > 0
      ? APP_CONFIG.appDomain + APP_CONFIG.basePath + "/auth-callback"
      : APP_CONFIG.appDomain + "/auth-callback", //The URI of your client application to receive a response from the OIDC provider.
  automaticSilentRenew: true, //(boolean, default: false): Flag to indicate if there should be an automatic attempt to renew the access token prior to its expiration.
  loadUserInfo: true, //(boolean, default: true): Flag to control if additional identity data is loaded from the user info endpoint in order to populate the user's profile.
  silent_redirect_uri:
    APP_CONFIG.basePath.length > 0
      ? APP_CONFIG.appDomain + APP_CONFIG.basePath + "/silent-auth-callback"
      : APP_CONFIG.appDomain + "/silent-auth-callback", //(string): The URL for the page containing the code handling the silent renew.

  post_logout_redirect_uri:
    APP_CONFIG.basePath.length > 0
      ? APP_CONFIG.appDomain + APP_CONFIG.basePath + "/"
      : APP_CONFIG.appDomain + "/", // (string): The OIDC post-logout redirect URI.
  //audience: "https://example.com", //is there a way to specific the audience when making the jwt
  response_type: "code", // PECK approach. //"id_token token", //(string, default: 'id_token'): The type of response desired from the OIDC provider.
  filterProtocolClaims: true, // PECK way.
  grant_type: "authorization_code", //"password",
  scope: APP_CONFIG.iamScope, //(string, default: 'openid'): The scope being requested from the OIDC provider.
  //
  // revoke (reference) access tokens at logout time
  //
  revokeAccessTokenOnSignout: true,
  // integer value of seconds. Default value = 300.
  // Work for resolve login infinity looping if client machine datetime is in correct.
  // Set clockSkew is 15 minutes now.
  clockSkew: 900,
  // Additional settings to help with state management
  includeIdTokenInSilentRenew: true,
  monitorSession: false, // Disable session monitoring to avoid conflicts
  checkSessionInterval: 10000, // Check session every 10 seconds
};
