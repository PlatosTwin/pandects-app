/** Authenticated fetch: adds session (cookie or Bearer) and CSRF for non-GET requests. */

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
  if (typeof window === "undefined") {
    // During prerender SSR the QueryClient is fresh per renderPage() call and
    // fetched data is discarded. We confirmed nothing reaches this path today
    // — useQuery callers are gated on IS_SERVER_RENDER. If something starts
    // reaching it, surface a hard error so the regression is obvious.
    throw new Error(
      `authFetch called during SSR (url=${String(input)}). Gate the caller on IS_SERVER_RENDER.`,
    );
  }
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
