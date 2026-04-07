import { apiUrl } from "@/lib/api-config";
import { authFetchJson } from "@/lib/auth-fetch";
import type {
  ApiKeySummary,
  AuthUser,
  ExternalSubjectLink,
  UsageByDay,
  UsagePeriod,
} from "@/lib/auth-types";

export type LegalAcceptancePayload = {
  checked_at_ms: number;
  docs: ["tos", "privacy", "license"];
};

export type WebsiteAuthResult =
  | {
      status: "authenticated";
      next_path: string;
      user: AuthUser;
      session_token?: string;
    }
  | {
      status: "legal_required";
      next_path: string;
      user: AuthUser;
    }
  | {
      status: "verification_required";
      next_path: string;
      user: AuthUser;
    };

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

export async function permanentlyDeleteApiKey(id: string) {
  return authFetchJson<{ status: "deleted" }>(apiUrl(`v1/auth/api-keys/${id}/permanent`), {
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

export async function deleteAccount(payload: { confirm: string }) {
  return authFetchJson<{ status: "deleted" }>(apiUrl("v1/auth/account/delete"), {
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
  return authFetchJson<{ authorize_url: string }>(
    apiUrl(`v1/auth/zitadel/start?${query.toString()}`),
  );
}

export async function startZitadelGoogleWebsiteAuth(nextPath = "/account") {
  const query = new URLSearchParams({ next: nextPath });
  return authFetchJson<{ authorize_url: string }>(
    apiUrl(`v1/auth/zitadel/google/start?${query.toString()}`),
  );
}

export async function loginWithPassword(payload: {
  email: string;
  password: string;
  next?: string;
}) {
  return authFetchJson<WebsiteAuthResult>(apiUrl("v1/auth/login/password"), {
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
}) {
  return authFetchJson<WebsiteAuthResult>(apiUrl("v1/auth/signup/password"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function requestPasswordReset(payload: { email: string }) {
  return authFetchJson<{ status: "requested" }>(apiUrl("v1/auth/password-reset/request"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function confirmPasswordReset(payload: {
  user_id: string;
  code: string;
  password: string;
}) {
  return authFetchJson<{ status: "updated" }>(apiUrl("v1/auth/password-reset/confirm"), {
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
  return authFetchJson<WebsiteAuthResult>(apiUrl("v1/auth/zitadel/complete"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function finalizeZitadelWebsiteAuth(legal: LegalAcceptancePayload) {
  return authFetchJson<WebsiteAuthResult>(apiUrl("v1/auth/zitadel/finalize"), {
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
  return authFetchJson<WebsiteAuthResult>(apiUrl("v1/auth/email/verify/confirm"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function resendEmailVerification(payload: { email: string }) {
  return authFetchJson<{
    status: "verification_required";
    user: AuthUser;
  }>(apiUrl("v1/auth/email/verify/resend"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function listExternalSubjects() {
  return authFetchJson<{ links: ExternalSubjectLink[] }>(
    apiUrl("v1/auth/external-subjects"),
  );
}

export async function linkExternalSubject(payload: {
  provider?: string;
  access_token: string;
}) {
  return authFetchJson<{
    status: "linked" | "already_linked";
    link: ExternalSubjectLink;
  }>(apiUrl("v1/auth/external-subjects"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function startZitadelLink(nextPath = "/account") {
  const query = new URLSearchParams({ next: nextPath });
  return authFetchJson<{ authorize_url: string }>(
    apiUrl(`v1/auth/external-subjects/zitadel/start?${query.toString()}`),
  );
}

export async function completeZitadelLink(payload: { code: string; state: string }) {
  return authFetchJson<{
    status: "linked" | "already_linked";
    link: ExternalSubjectLink;
    return_to: string;
  }>(apiUrl("v1/auth/external-subjects/zitadel/complete"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function unlinkExternalSubject(id: number) {
  return authFetchJson<{ status: "unlinked" }>(apiUrl(`v1/auth/external-subjects/${id}`), {
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
  return authFetchJson<{ status: "ok" }>(apiUrl("v1/auth/flag-inaccurate"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
