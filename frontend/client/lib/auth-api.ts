import { apiUrl } from "@/lib/api-config";
import { authFetchJson } from "@/lib/auth-fetch";
import type { ApiKeySummary, AuthUser, UsageByDay } from "@/lib/auth-types";

export type LegalAcceptancePayload = {
  checkedAtMs: number;
  docs: ["tos", "privacy", "license"];
};

export async function registerWithEmail(
  email: string,
  password: string,
  legal: LegalAcceptancePayload,
  captchaToken?: string,
) {
  return authFetchJson<{
    status: "verification_required";
    user: AuthUser;
    debugToken?: string;
  }>(apiUrl("api/auth/register"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(
      captchaToken ? { email, password, legal, captchaToken } : { email, password, legal },
    ),
  });
}

export async function loginWithEmail(email: string, password: string) {
  return authFetchJson<{ user: AuthUser; sessionToken?: string }>(
    apiUrl("api/auth/login"),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    },
  );
}

export async function resendVerificationEmail(email: string) {
  return authFetchJson<{ status: "sent" }>(apiUrl("api/auth/email/resend"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email }),
  });
}

export async function fetchMe() {
  return authFetchJson<{ user: AuthUser }>(apiUrl("api/auth/me"));
}

export async function logoutSession() {
  return authFetchJson<{ status: "ok" }>(apiUrl("api/auth/logout"), { method: "POST" });
}

export async function listApiKeys() {
  return authFetchJson<{ keys: ApiKeySummary[] }>(apiUrl("api/auth/api-keys"));
}

export async function createApiKey(name?: string) {
  return authFetchJson<{
    apiKey: { id: string; name: string | null; prefix: string; createdAt: string };
    apiKeyPlaintext: string;
  }>(apiUrl("api/auth/api-keys"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
}

export async function revokeApiKey(id: string) {
  return authFetchJson<{ status: "revoked" }>(apiUrl(`api/auth/api-keys/${id}`), {
    method: "DELETE",
  });
}

export async function fetchUsage() {
  return authFetchJson<{ byDay: UsageByDay[]; total: number }>(
    apiUrl("api/auth/usage"),
  );
}

export async function loginWithGoogleCredential(
  credential: string,
  legal?: LegalAcceptancePayload,
) {
  return authFetchJson<{ user: AuthUser; sessionToken?: string }>(
    apiUrl("api/auth/google/credential"),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(legal ? { credential, legal } : { credential }),
    },
  );
}

export async function deleteAccount(payload: { confirm: string }) {
  return authFetchJson<{ status: "deleted" }>(apiUrl("api/auth/account/delete"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
