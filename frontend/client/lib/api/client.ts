/**
 * The one API client for the Pandects backend.
 *
 * Every server call goes through `apiRequest` (or the `apiGet` / `apiPost` /
 * ... shorthands). It adds the session and CSRF token via `authFetch`, parses
 * JSON, and turns every failure into an `ApiError`:
 *
 * - non-2xx responses parse the server envelope `{ error, message }` into
 *   `code` and `message`, so the server's explanation reaches the UI;
 * - network failures (fetch rejects with a TypeError) become `status: 0`,
 *   `code: "network"`;
 * - aborts (`AbortError`) are re-thrown untouched so React Query can treat
 *   them as cancellations.
 *
 * SSR: `authFetch` throws during prerender. Query hooks must gate `enabled`
 * on `IS_SERVER_RENDER` (see `lib/query-client.ts`), so nothing reaches here
 * on the server.
 */

import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";

export const NETWORK_ERROR_CODE = "network";
export const INVALID_RESPONSE_CODE = "invalid_response";

export class ApiError extends Error {
  /** HTTP status; 0 when the request never got a response. */
  readonly status: number;
  /** Server error code from the `{ error }` envelope, or "network". */
  readonly code?: string;
  readonly statusText: string;
  /** Raw response body (empty for network failures). */
  readonly bodyText: string;
  /** The envelope's `message`, when the server sent one. */
  readonly serverMessage?: string;

  constructor(params: {
    message: string;
    status: number;
    code?: string;
    statusText?: string;
    bodyText?: string;
    serverMessage?: string;
  }) {
    super(params.message);
    this.name = "ApiError";
    this.status = params.status;
    this.code = params.code;
    this.statusText = params.statusText ?? "";
    this.bodyText = params.bodyText ?? "";
    this.serverMessage = params.serverMessage;
  }

  get isNetworkError(): boolean {
    return this.status === 0;
  }
}

export function isApiError(error: unknown): error is ApiError {
  return error instanceof ApiError;
}

/** True for a fetch the caller (or React Query) aborted. */
export function isAbortError(error: unknown): boolean {
  return (
    typeof error === "object" &&
    error !== null &&
    (error as { name?: unknown }).name === "AbortError"
  );
}

/** Best human-readable message for any thrown value. */
export function errorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error && error.message.trim()) return error.message;
  return fallback;
}

export type QueryParamValue =
  | string
  | number
  | boolean
  | null
  | undefined
  | ReadonlyArray<string | number>;

export type QueryParams = URLSearchParams | Record<string, QueryParamValue>;

function ensureTerminalPunctuation(message: string): string {
  const trimmed = message.trim();
  if (!trimmed) return trimmed;
  return /[.!?]$/.test(trimmed) ? trimmed : `${trimmed}.`;
}

function toSearchParams(params: QueryParams): URLSearchParams {
  if (params instanceof URLSearchParams) return params;
  const out = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === null || value === undefined) continue;
    if (Array.isArray(value)) {
      value.forEach((item) => out.append(key, String(item)));
    } else {
      out.set(key, String(value));
    }
  }
  return out;
}

/**
 * Absolute URL for an API path. `path` is relative to the API base
 * ("v1/sections"); an absolute http(s) URL is used as-is.
 */
export function buildApiUrl(path: string, params?: QueryParams): string {
  const base = /^https?:\/\//i.test(path) ? path : apiUrl(path);
  if (!params) return base;
  const query = toSearchParams(params).toString();
  if (!query) return base;
  return `${base}${base.includes("?") ? "&" : "?"}${query}`;
}

/** Parse a non-2xx response into an ApiError (envelope `{ error, message }`). */
export async function parseApiError(res: Response): Promise<ApiError> {
  const statusText = res.statusText ?? "";
  const contentType = res.headers?.get?.("Content-Type") ?? "";
  const bodyText =
    typeof res.text === "function" ? await res.text().catch(() => "") : "";

  let serverMessage: string | undefined;
  let serverCode: string | undefined;
  const looksLikeJson =
    contentType.includes("application/json") || /^\s*\{/.test(bodyText);
  if (looksLikeJson && bodyText) {
    try {
      const body = JSON.parse(bodyText) as unknown;
      if (body && typeof body === "object") {
        const message = (body as { message?: unknown }).message;
        if (typeof message === "string" && message.trim()) {
          serverMessage = ensureTerminalPunctuation(message);
        }
        const code = (body as { error?: unknown }).error;
        if (typeof code === "string" && code.trim()) {
          serverCode = code.trim();
        }
      }
    } catch {
      // Not JSON after all; fall back to the status line.
    }
  }

  // Short plain-text bodies are worth showing; HTML error pages are not.
  const plainBody =
    !looksLikeJson && bodyText && bodyText.length <= 200 && !/<[a-z!]/i.test(bodyText)
      ? bodyText.trim()
      : "";

  return new ApiError({
    message:
      serverMessage ??
      ensureTerminalPunctuation(
        `HTTP ${res.status}${statusText ? `: ${statusText}` : ""}${plainBody ? ` — ${plainBody}` : ""}`,
      ),
    status: res.status,
    code: serverCode,
    statusText,
    bodyText,
    serverMessage,
  });
}

/** Wrap a fetch rejection: network failures become ApiError(status 0). */
function wrapFetchFailure(error: unknown): unknown {
  if (isAbortError(error) || error instanceof ApiError) return error;
  if (error instanceof TypeError) {
    return new ApiError({
      message: "Unable to reach the server. Check your connection and try again.",
      status: 0,
      code: NETWORK_ERROR_CODE,
    });
  }
  return error;
}

export interface ApiRequestInit extends Omit<RequestInit, "body"> {
  params?: QueryParams;
  /** JSON-serialized and sent with `Content-Type: application/json`. */
  json?: unknown;
  body?: BodyInit | null;
}

/**
 * Send a request and parse the JSON response. Throws ApiError on any failure
 * except aborts. A 204 resolves to `undefined`.
 */
export async function apiRequest<T>(
  path: string,
  init: ApiRequestInit = {},
): Promise<T> {
  const { params, json, headers: rawHeaders, ...rest } = init;
  const headers = new Headers(rawHeaders);
  let body = rest.body;
  if (json !== undefined) {
    body = JSON.stringify(json);
    if (!headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  }

  let res: Response;
  try {
    res = await authFetch(buildApiUrl(path, params), { ...rest, headers, body });
  } catch (error) {
    throw wrapFetchFailure(error);
  }

  if (!res.ok) throw await parseApiError(res);
  if (res.status === 204) return undefined as T;
  try {
    return (await res.json()) as T;
  } catch (error) {
    if (isAbortError(error)) throw error;
    if (error instanceof TypeError) throw wrapFetchFailure(error);
    throw new ApiError({
      message: "The server sent a response that could not be read.",
      status: res.status,
      code: INVALID_RESPONSE_CODE,
      statusText: res.statusText,
    });
  }
}

type RequestOptions = Omit<ApiRequestInit, "method" | "params" | "json">;

export function apiGet<T>(
  path: string,
  params?: QueryParams,
  init: RequestOptions = {},
): Promise<T> {
  return apiRequest<T>(path, { ...init, method: "GET", params });
}

export function apiPost<T>(
  path: string,
  json?: unknown,
  init: RequestOptions = {},
): Promise<T> {
  return apiRequest<T>(path, { ...init, method: "POST", json });
}

export function apiPut<T>(
  path: string,
  json?: unknown,
  init: RequestOptions = {},
): Promise<T> {
  return apiRequest<T>(path, { ...init, method: "PUT", json });
}

export function apiPatch<T>(
  path: string,
  json?: unknown,
  init: RequestOptions = {},
): Promise<T> {
  return apiRequest<T>(path, { ...init, method: "PATCH", json });
}

export function apiDelete<T>(
  path: string,
  init: RequestOptions = {},
): Promise<T> {
  return apiRequest<T>(path, { ...init, method: "DELETE" });
}

/**
 * Drop-in for the old `authFetchJson(url, init)`: takes a full URL (usually
 * `apiUrl(...)`) and a raw RequestInit. Prefer `apiGet` / `apiPost` in new code.
 */
export function apiFetchJson<T>(
  url: string | URL,
  init: RequestInit = {},
): Promise<T> {
  return apiRequest<T>(String(url), init);
}
