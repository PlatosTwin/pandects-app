export type ZitadelLinkConfig = {
  authority: string;
  clientId: string;
  redirectUri: string;
  authorizationEndpoint: string;
  tokenEndpoint: string;
  scopes: string[];
  resource: string | null;
  audience: string | null;
};

function sanitizeOptionalEnv(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : null;
}

function defaultRedirectUri(): string | null {
  if (typeof window === "undefined") return null;
  return `${window.location.origin}/auth/zitadel/callback`;
}

export function resolveZitadelLinkConfig(): ZitadelLinkConfig | null {
  const authority = sanitizeOptionalEnv(import.meta.env.VITE_ZITADEL_AUTHORITY);
  const clientId = sanitizeOptionalEnv(import.meta.env.VITE_ZITADEL_CLIENT_ID);
  const redirectUri =
    sanitizeOptionalEnv(import.meta.env.VITE_ZITADEL_REDIRECT_URI) ?? defaultRedirectUri();

  if (!authority || !clientId || !redirectUri) return null;

  const trimmedAuthority = authority.replace(/\/+$/, "");
  const authorizationEndpoint =
    sanitizeOptionalEnv(import.meta.env.VITE_ZITADEL_AUTHORIZATION_ENDPOINT) ??
    `${trimmedAuthority}/oauth/v2/authorize`;
  const tokenEndpoint =
    sanitizeOptionalEnv(import.meta.env.VITE_ZITADEL_TOKEN_ENDPOINT) ??
    `${trimmedAuthority}/oauth/v2/token`;
  const scopeString =
    sanitizeOptionalEnv(import.meta.env.VITE_ZITADEL_SCOPES) ??
    "openid profile email offline_access sections:search agreements:search agreements:read";

  return {
    authority: trimmedAuthority,
    clientId,
    redirectUri,
    authorizationEndpoint,
    tokenEndpoint,
    scopes: scopeString.split(/\s+/).filter(Boolean),
    resource: sanitizeOptionalEnv(import.meta.env.VITE_ZITADEL_RESOURCE),
    audience: sanitizeOptionalEnv(import.meta.env.VITE_ZITADEL_AUDIENCE),
  };
}

export function isZitadelLinkConfigured(): boolean {
  return resolveZitadelLinkConfig() !== null;
}

export function buildZitadelAuthorizeUrl(
  config: ZitadelLinkConfig,
  input: {
    state: string;
    codeChallenge: string;
  },
): string {
  const query = new URLSearchParams({
    client_id: config.clientId,
    redirect_uri: config.redirectUri,
    response_type: "code",
    scope: config.scopes.join(" "),
    code_challenge: input.codeChallenge,
    code_challenge_method: "S256",
    state: input.state,
  });
  if (config.resource) query.set("resource", config.resource);
  if (config.audience) query.set("audience", config.audience);
  return `${config.authorizationEndpoint}?${query.toString()}`;
}
