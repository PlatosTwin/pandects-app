import { authSessionTransport } from "@/lib/auth-transport";
import { getSessionToken } from "@/lib/auth-session";
import { apiUrl } from "@/lib/api-config";

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

async function ensureCsrfCookie(): Promise<void> {
  const existing = getCookie("pdcts_csrf");
  if (existing) {
    csrfInitPromise = null;
    return;
  }
  if (!csrfInitPromise) {
    csrfInitPromise = fetch(apiUrl("api/auth/csrf"), { credentials: "include" })
      .then(() => undefined)
      .catch(() => undefined);
  }
  await csrfInitPromise;
}

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
    await ensureCsrfCookie();
    const csrf = getCookie("pdcts_csrf");
    if (csrf) headers.set("X-CSRF-Token", csrf);
  }

  return fetch(input, { ...init, headers, credentials: "include" });
}

export async function authFetchJson<T>(
  input: RequestInfo | URL,
  init: RequestInit = {},
): Promise<T> {
  const res = await authFetch(input, init);
  if (!res.ok) {
    const bodyText = await res.text().catch(() => "");
    throw new Error(
      `HTTP ${res.status}: ${res.statusText}${bodyText ? ` — ${bodyText}` : ""}`,
    );
  }
  return (await res.json()) as T;
}
