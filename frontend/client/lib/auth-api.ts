import type { components } from "@shared/api.gen";
import { apiUrl } from "@/lib/api-config";
import { apiFetchJson } from "@/lib/api/client";
import type { UsagePeriod } from "@/lib/auth-types";

// Response types are generated from the backend spec (shared/api.gen.ts).
type Schemas = components["schemas"];
type AuthFlowResponse = Schemas["AuthFlowResponse"];
type AuthStatusResponse = Schemas["AuthStatusResponse"];

export type LegalAcceptancePayload = Omit<Schemas["AuthLegalAcceptance"], "docs"> & {
  docs: ["tos", "privacy", "license"];
};

/**
 * `AuthFlowResponse` narrowed per `status`: every website sign-in status
 * carries the user, and only `authenticated` can carry a session token.
 */
export type WebsiteAuthResult =
  | (Pick<AuthFlowResponse, "next_path" | "session_token"> & {
      status: "authenticated";
      user: Schemas["AuthUser"];
    })
  | (Pick<AuthFlowResponse, "next_path"> & {
      status: "legal_required" | "verification_required";
      user: Schemas["AuthUser"];
    });

/** `AuthFlowResponse` for the MCP-token flow of `/v1/auth/zitadel/complete`. */
export type McpTokenResult = Pick<
  AuthFlowResponse,
  "next_path" | "token_type" | "expires_in" | "scope"
> & {
  status: "mcp_token";
  access_token: string;
};

export async function fetchMe() {
  return apiFetchJson<Schemas["AuthMeResponse"]>(apiUrl("v1/auth/me"));
}

export async function logoutSession() {
  return apiFetchJson<AuthStatusResponse>(apiUrl("v1/auth/logout"), { method: "POST" });
}

export async function listApiKeys() {
  return apiFetchJson<Schemas["AuthApiKeysResponse"]>(apiUrl("v1/auth/api-keys"));
}

export async function createApiKey(name?: string) {
  return apiFetchJson<Schemas["AuthCreateApiKeyResponse"]>(apiUrl("v1/auth/api-keys"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export async function revokeApiKey(id: string) {
  return apiFetchJson<AuthStatusResponse>(apiUrl(`v1/auth/api-keys/${id}`), {
    method: "DELETE",
  });
}

export async function permanentlyDeleteApiKey(id: string) {
  return apiFetchJson<AuthStatusResponse>(apiUrl(`v1/auth/api-keys/${id}/permanent`), {
    method: "DELETE",
  });
}

export async function fetchUsage(params?: { period?: UsagePeriod; apiKeyId?: string }) {
  const query = new URLSearchParams();
  if (params?.period) query.set("period", params.period);
  if (params?.apiKeyId) query.set("api_key_id", params.apiKeyId);
  const suffix = query.toString();
  return apiFetchJson<Schemas["AuthUsageResponse"]>(
    apiUrl(`v1/auth/usage${suffix ? `?${suffix}` : ""}`),
  );
}

export async function grantOAuthConsent(payload: { client_id: string; scope: string }) {
  return apiFetchJson<Schemas["AuthOAuthConsentResponse"]>(
    apiUrl("v1/auth/oauth/consent"),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
}

export async function deleteAccount(payload: { confirm: string }) {
  return apiFetchJson<AuthStatusResponse>(apiUrl("v1/auth/account/delete"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function startZitadelWebsiteAuth(
  nextPath = "/account",
  provider: "google" | "email" = "email",
  prompt?: "login" | "create" | "select_account",
) {
  const query = new URLSearchParams({ next: nextPath, provider });
  if (prompt) query.set("prompt", prompt);
  return apiFetchJson<Schemas["AuthAuthorizeUrlResponse"]>(
    apiUrl(`v1/auth/zitadel/start?${query.toString()}`),
  );
}

export async function startZitadelGoogleWebsiteAuth(nextPath = "/account") {
  const query = new URLSearchParams({ next: nextPath });
  return apiFetchJson<Schemas["AuthAuthorizeUrlResponse"]>(
    apiUrl(`v1/auth/zitadel/google/start?${query.toString()}`),
  );
}

export async function startMcpAccessTokenAuth(nextPath = "/account") {
  const query = new URLSearchParams({ next: nextPath });
  return apiFetchJson<Schemas["AuthAuthorizeUrlResponse"]>(
    apiUrl(`v1/auth/mcp-token/start?${query.toString()}`),
  );
}

export async function loginWithPassword(payload: {
  email: string;
  password: string;
  next?: string;
}) {
  return apiFetchJson<WebsiteAuthResult>(apiUrl("v1/auth/login/password"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function signupWithPassword(payload: {
  email: string;
  password: string;
  first_name?: string;
  last_name?: string;
  next?: string;
  captcha_token?: string;
}) {
  return apiFetchJson<WebsiteAuthResult>(apiUrl("v1/auth/signup/password"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function requestPasswordReset(payload: {
  email: string;
  captcha_token?: string;
}) {
  return apiFetchJson<AuthStatusResponse>(apiUrl("v1/auth/password-reset/request"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

/** `AuthCaptchaSiteKeyResponse` narrowed on `enabled`: an enabled widget always has a site key. */
export type CaptchaSiteKeyResponse =
  | (Omit<Schemas["AuthCaptchaSiteKeyResponse"], "enabled" | "site_key"> & { enabled: false })
  | (Omit<Schemas["AuthCaptchaSiteKeyResponse"], "enabled" | "site_key"> & {
      enabled: true;
      site_key: string;
    });

export async function fetchCaptchaSiteKey() {
  return apiFetchJson<CaptchaSiteKeyResponse>(apiUrl("v1/auth/captcha/site-key"));
}

export async function confirmPasswordReset(payload: {
  user_id: string;
  code: string;
  password: string;
}) {
  return apiFetchJson<AuthStatusResponse>(apiUrl("v1/auth/password-reset/confirm"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function completeZitadelWebsiteAuth(
  payload:
    | { code: string; state: string }
    | { intent_id: string; intent_token: string; user_id?: string },
) {
  return apiFetchJson<WebsiteAuthResult | McpTokenResult>(apiUrl("v1/auth/zitadel/complete"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function finalizeZitadelWebsiteAuth(legal: LegalAcceptancePayload) {
  return apiFetchJson<WebsiteAuthResult>(apiUrl("v1/auth/zitadel/finalize"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ legal }),
  });
}

export async function completeEmailVerification(payload: {
  user_id: string;
  code: string;
  next?: string;
}) {
  return apiFetchJson<WebsiteAuthResult>(apiUrl("v1/auth/email/verify/confirm"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function resendEmailVerification(payload: { email: string }) {
  return apiFetchJson<Schemas["AuthVerificationResendResponse"]>(apiUrl("v1/auth/email/verify/resend"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function listExternalSubjects() {
  return apiFetchJson<Schemas["AuthExternalSubjectsResponse"]>(
    apiUrl("v1/auth/external-subjects"),
  );
}

export async function linkExternalSubject(payload: {
  provider?: string;
  access_token: string;
}) {
  return apiFetchJson<Schemas["AuthExternalLinkResponse"]>(apiUrl("v1/auth/external-subjects"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function startZitadelLink(nextPath = "/account") {
  const query = new URLSearchParams({ next: nextPath });
  return apiFetchJson<Schemas["AuthAuthorizeUrlResponse"]>(
    apiUrl(`v1/auth/external-subjects/zitadel/start?${query.toString()}`),
  );
}

export async function completeZitadelLink(payload: { code: string; state: string }) {
  return apiFetchJson<Schemas["AuthZitadelLinkCompleteResponse"]>(apiUrl("v1/auth/external-subjects/zitadel/complete"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function unlinkExternalSubject(id: number) {
  return apiFetchJson<AuthStatusResponse>(apiUrl(`v1/auth/external-subjects/${id}`), {
    method: "DELETE",
  });
}

export type FlagInaccurateSource = "search_result" | "agreement_view";

export async function flagAsInaccurate(payload: {
  source: FlagInaccurateSource;
  agreement_uuid: string;
  section_uuid?: string;
  message?: string;
  request_follow_up: boolean;
  issue_types: string[];
}) {
  return apiFetchJson<AuthStatusResponse>(apiUrl("v1/auth/flag-inaccurate"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
