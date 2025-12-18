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
let csrfToken: string | null = null;

async function ensureCsrfToken(): Promise<void> {
  if (csrfToken) return;

  const existing = getCookie("pdcts_csrf");
  if (existing) {
    csrfToken = existing;
    csrfInitPromise = null;
    return;
  }

  if (!csrfInitPromise) {
    csrfInitPromise = fetch(apiUrl("api/auth/csrf"), { credentials: "include" })
      .then(async (res) => {
        if (!res.ok) return;
        const data = (await res.json().catch(() => null)) as unknown;
        if (!data || typeof data !== "object") return;
        const token = (data as { csrfToken?: unknown }).csrfToken;
        if (typeof token === "string" && token.trim()) csrfToken = token.trim();
      })
      .catch(() => undefined);
  }

  await csrfInitPromise;
  const after = getCookie("pdcts_csrf");
  if (after) csrfToken = after;
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
    await ensureCsrfToken();
    if (csrfToken) headers.set("X-CSRF-Token", csrfToken);
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
