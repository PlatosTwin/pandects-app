import { apiUrl } from "@/lib/api-config";
import { authFetchJson } from "@/lib/auth-fetch";
import type { ApiKeySummary, AuthUser, UsageByDay, UsagePeriod } from "@/lib/auth-types";

export type LegalAcceptancePayload = {
  checked_at_ms: number;
  docs: ["tos", "privacy", "license"];
};

export async function registerWithEmail(
  email: string,
  password: string,
  legal: LegalAcceptancePayload,
  captcha_token?: string,
) {
  return authFetchJson<{
    status: "verification_required";
    user: AuthUser;
    debug_token?: string;
  }>(apiUrl("v1/auth/register"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(
      captcha_token ? { email, password, legal, captcha_token } : { email, password, legal },
    ),
  });
}

export async function loginWithEmail(email: string, password: string) {
  return authFetchJson<{ user: AuthUser; session_token?: string }>(
    apiUrl("v1/auth/login"),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    },
  );
}

export async function resendVerificationEmail(email: string) {
  return authFetchJson<{ status: "sent" }>(apiUrl("v1/auth/email/resend"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });
}

export async function fetchMe() {
  return authFetchJson<{ user: AuthUser }>(apiUrl("v1/auth/me"));
}

export async function logoutSession() {
  return authFetchJson<{ status: "ok" }>(apiUrl("v1/auth/logout"), { method: "POST" });
}

export async function listApiKeys() {
  return authFetchJson<{ keys: ApiKeySummary[] }>(apiUrl("v1/auth/api-keys"));
}

export async function createApiKey(name?: string) {
  return authFetchJson<{
    api_key: { id: string; name: string | null; prefix: string; created_at: string };
    api_key_plaintext: string;
  }>(apiUrl("v1/auth/api-keys"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export async function revokeApiKey(id: string) {
  return authFetchJson<{ status: "revoked" }>(apiUrl(`v1/auth/api-keys/${id}`), {
    method: "DELETE",
  });
}

export async function fetchUsage(params?: { period?: UsagePeriod; apiKeyId?: string }) {
  const query = new URLSearchParams();
  if (params?.period) query.set("period", params.period);
  if (params?.apiKeyId) query.set("api_key_id", params.apiKeyId);
  const suffix = query.toString();
  return authFetchJson<{ by_day: UsageByDay[]; total: number }>(
    apiUrl(`v1/auth/usage${suffix ? `?${suffix}` : ""}`),
  );
}

export async function loginWithGoogleCredential(
  credential: string,
  legal?: LegalAcceptancePayload,
) {
  return authFetchJson<{ user: AuthUser; session_token?: string }>(
    apiUrl("v1/auth/google/credential"),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify(legal ? { credential, legal } : { credential }),
    },
  );
}

export async function deleteAccount(payload: { confirm: string }) {
  return authFetchJson<{ status: "deleted" }>(apiUrl("v1/auth/account/delete"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function requestPasswordReset(email: string) {
  return authFetchJson<{ status: "sent" }>(apiUrl("v1/auth/password/forgot"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });
}

export async function resetPassword(token: string, password: string) {
  return authFetchJson<{ status: "ok" }>(apiUrl("v1/auth/password/reset"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token, password }),
  });
}

export async function verifyEmail(token: string) {
  return authFetchJson<{ status: "ok" }>(apiUrl("v1/auth/email/verify"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
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
  return authFetchJson<{ status: "ok" }>(apiUrl("v1/auth/flag-inaccurate"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
