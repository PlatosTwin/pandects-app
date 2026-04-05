/** Authenticated fetch: adds session (cookie or Bearer) and CSRF for non-GET requests. */

import { authSessionTransport } from "@/lib/auth-transport";
import { getSessionToken } from "@/lib/auth-session";
import { apiUrl } from "@/lib/api-config";

export class AuthApiError extends Error {
  status: number;
  statusText: string;
  code: string | null;
  bodyText: string;

  constructor(params: {
    message: string;
    status: number;
    statusText: string;
    code: string | null;
    bodyText: string;
  }) {
    super(params.message);
    this.name = "AuthApiError";
    this.status = params.status;
    this.statusText = params.statusText;
    this.code = params.code;
    this.bodyText = params.bodyText;
  }
}

function ensureTerminalPunctuation(message: string): string {
  const trimmed = message.trim();
  if (!trimmed) return trimmed;
  return /[.!?]$/.test(trimmed) ? trimmed : `${trimmed}.`;
}

function getCookie(name: string): string | null {
  if (typeof document === "undefined") return null;
  const parts = document.cookie.split(";").map((p) => p.trim());
  for (const part of parts) {
    if (!part.startsWith(`${name}=`)) continue;
    return decodeURIComponent(part.slice(name.length + 1));
  }
  return null;
}

function isUnsafeMethod(method: string | undefined): boolean {
  const normalized = (method || "GET").toUpperCase();
  return !["GET", "HEAD", "OPTIONS"].includes(normalized);
}

let csrfInitPromise: Promise<void> | null = null;

async function ensureCsrfToken(): Promise<string | null> {
  const existing = getCookie("pdcts_csrf");
  if (existing) {
    csrfInitPromise = null;
    return existing;
  }
  if (!csrfInitPromise) {
    csrfInitPromise = fetch(apiUrl("v1/auth/csrf"), { credentials: "include" })
      .then(() => undefined)
      .catch(() => undefined);
  }

  await csrfInitPromise;
  const after = getCookie("pdcts_csrf");
  if (after) return after;

  // Fallback: some environments may block cookie access; the endpoint can return the token too.
  // Note: this does not make CSRF work without cookies; the server still requires the cookie.
  try {
    const res = await fetch(apiUrl("v1/auth/csrf"), { credentials: "include" });
    const data = (await res.json().catch(() => null)) as unknown;
    if (!data || typeof data !== "object") return null;
    const token = (data as { csrf_token?: unknown }).csrf_token;
    return typeof token === "string" && token.trim() ? token.trim() : null;
  } catch {
    return null;
  }
}

/** Fetch with session and CSRF; use for all API calls that require auth or cookies. */
export async function authFetch(
  input: RequestInfo | URL,
  init: RequestInit = {},
): Promise<Response> {
  const headers = new Headers(init.headers);

  const transport = authSessionTransport();
  if (transport === "bearer") {
    const token = getSessionToken();
    if (token && !headers.has("Authorization")) {
      headers.set("Authorization", `Bearer ${token}`);
    }
    return fetch(input, { ...init, headers });
  }

  if (isUnsafeMethod(init.method) && !headers.has("X-CSRF-Token")) {
    const token = getCookie("pdcts_csrf") ?? (await ensureCsrfToken());
    if (token) headers.set("X-CSRF-Token", token);
  }

  return fetch(input, { ...init, headers, credentials: "include" });
}

export async function authFetchJson<T>(
  input: RequestInfo | URL,
  init: RequestInit = {},
): Promise<T> {
  const res = await authFetch(input, init);
  if (!res.ok) {
    const contentType = res.headers.get("Content-Type") || "";
    const bodyText = await res.text().catch(() => "");
    let serverMessage: string | null = null;
    let serverCode: string | null = null;

    if (contentType.includes("application/json") && bodyText) {
      try {
        const body = JSON.parse(bodyText) as unknown;
        if (body && typeof body === "object") {
          const message = (body as { message?: unknown }).message;
          if (typeof message === "string" && message.trim()) {
            serverMessage = ensureTerminalPunctuation(message);
          }
          const errorCode = (body as { error?: unknown }).error;
          if (typeof errorCode === "string" && errorCode.trim()) {
            serverCode = errorCode.trim();
          }
        }
      } catch {
        serverMessage = null;
        serverCode = null;
      }
    }
    throw new AuthApiError({
      message:
        serverMessage ??
        ensureTerminalPunctuation(
          `HTTP ${res.status}: ${res.statusText}${bodyText ? ` — ${bodyText}` : ""}`,
        ),
      status: res.status,
      statusText: res.statusText,
      code: serverCode,
      bodyText,
    });
  }
  return (await res.json()) as T;
}
