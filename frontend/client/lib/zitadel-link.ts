const ZITADEL_LINK_STATE_KEY = "pandects.zitadel_link.state";
const ZITADEL_LINK_VERIFIER_KEY = "pandects.zitadel_link.verifier";
const ZITADEL_LINK_RETURN_TO_KEY = "pandects.zitadel_link.return_to";

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

function randomString(byteLength: number): string {
  const bytes = new Uint8Array(byteLength);
  window.crypto.getRandomValues(bytes);
  return base64UrlEncode(bytes);
}

function base64UrlEncode(input: Uint8Array): string {
  let binary = "";
  for (const value of input) {
    binary += String.fromCharCode(value);
  }
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

async function sha256(input: string): Promise<Uint8Array> {
  const encoded = new TextEncoder().encode(input);
  const digest = await window.crypto.subtle.digest("SHA-256", encoded);
  return new Uint8Array(digest);
}

function requiredSessionStorage(): Storage {
  if (typeof window === "undefined" || !window.sessionStorage) {
    throw new Error("ZITADEL linking requires a browser session.");
  }
  return window.sessionStorage;
}

function safeReturnToPath(value: string | null | undefined): string {
  if (!value) return "/account";
  const trimmed = value.trim();
  if (!trimmed.startsWith("/")) return "/account";
  if (trimmed.startsWith("//")) return "/account";
  return trimmed;
}

export async function startZitadelLinkFlow(params?: { returnTo?: string }): Promise<void> {
  const config = resolveZitadelLinkConfig();
  if (!config) {
    throw new Error("ZITADEL linking is not configured.");
  }
  if (typeof window === "undefined") {
    throw new Error("ZITADEL linking requires a browser environment.");
  }

  const storage = requiredSessionStorage();
  const state = randomString(24);
  const verifier = randomString(32);
  const codeChallenge = base64UrlEncode(await sha256(verifier));
  const returnTo = safeReturnToPath(params?.returnTo);

  storage.setItem(ZITADEL_LINK_STATE_KEY, state);
  storage.setItem(ZITADEL_LINK_VERIFIER_KEY, verifier);
  storage.setItem(ZITADEL_LINK_RETURN_TO_KEY, returnTo);

  window.location.assign(
    buildZitadelAuthorizeUrl(config, {
      state,
      codeChallenge,
    }),
  );
}

export async function finishZitadelLinkFlow(search: string): Promise<{
  accessToken: string;
  returnTo: string;
}> {
  const config = resolveZitadelLinkConfig();
  if (!config) {
    throw new Error("ZITADEL linking is not configured.");
  }

  const params = new URLSearchParams(search);
  const error = sanitizeOptionalEnv(params.get("error"));
  const errorDescription = sanitizeOptionalEnv(params.get("error_description"));
  if (error) {
    clearZitadelLinkState();
    throw new Error(errorDescription ?? error);
  }

  const code = sanitizeOptionalEnv(params.get("code"));
  const state = sanitizeOptionalEnv(params.get("state"));
  if (!code || !state) {
    clearZitadelLinkState();
    throw new Error("Missing ZITADEL authorization response.");
  }

  const storage = requiredSessionStorage();
  const expectedState = storage.getItem(ZITADEL_LINK_STATE_KEY);
  const verifier = storage.getItem(ZITADEL_LINK_VERIFIER_KEY);
  const returnTo = safeReturnToPath(storage.getItem(ZITADEL_LINK_RETURN_TO_KEY));
  if (!expectedState || !verifier || state !== expectedState) {
    clearZitadelLinkState();
    throw new Error("Invalid ZITADEL authorization state.");
  }

  const body = new URLSearchParams({
    grant_type: "authorization_code",
    client_id: config.clientId,
    code,
    code_verifier: verifier,
    redirect_uri: config.redirectUri,
  });
  if (config.resource) body.set("resource", config.resource);
  if (config.audience) body.set("audience", config.audience);

  let response: Response;
  try {
    response = await fetch(config.tokenEndpoint, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body,
    });
  } catch {
    clearZitadelLinkState();
    throw new Error("Could not reach ZITADEL to finish linking.");
  }

  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }

  if (!response.ok) {
    clearZitadelLinkState();
    if (payload && typeof payload === "object") {
      const description = sanitizeOptionalEnv(
        (payload as { error_description?: unknown }).error_description,
      );
      const tokenError = sanitizeOptionalEnv((payload as { error?: unknown }).error);
      throw new Error(description ?? tokenError ?? "ZITADEL token exchange failed.");
    }
    throw new Error("ZITADEL token exchange failed.");
  }

  const accessToken =
    payload && typeof payload === "object"
      ? sanitizeOptionalEnv((payload as { access_token?: unknown }).access_token)
      : null;
  if (!accessToken) {
    clearZitadelLinkState();
    throw new Error("ZITADEL token response did not include an access token.");
  }

  clearZitadelLinkState();
  return { accessToken, returnTo };
}

export function clearZitadelLinkState(): void {
  if (typeof window === "undefined" || !window.sessionStorage) return;
  window.sessionStorage.removeItem(ZITADEL_LINK_STATE_KEY);
  window.sessionStorage.removeItem(ZITADEL_LINK_VERIFIER_KEY);
  window.sessionStorage.removeItem(ZITADEL_LINK_RETURN_TO_KEY);
}
