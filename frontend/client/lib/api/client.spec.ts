import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const authFetch = vi.fn<(input: RequestInfo | URL, init?: RequestInit) => Promise<Response>>();
vi.mock("@/lib/auth-fetch", () => ({
  authFetch: (input: RequestInfo | URL, init?: RequestInit) => authFetch(input, init),
}));

const {
  ApiError,
  apiFetchJson,
  apiGet,
  apiPost,
  buildApiUrl,
} = await import("./client");

function jsonResponse(body: unknown, init: ResponseInit = {}): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

describe("api client", () => {
  beforeEach(() => {
    authFetch.mockReset();
  });
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("builds URLs from relative paths and params (arrays repeat, nullish dropped)", () => {
    const url = new URL(
      buildApiUrl("v1/sections", { year: ["2020", "2021"], page: 2, q: undefined, x: null }),
    );
    expect(url.pathname).toMatch(/\/v1\/sections$/);
    expect(url.searchParams.getAll("year")).toEqual(["2020", "2021"]);
    expect(url.searchParams.get("page")).toBe("2");
    expect(url.searchParams.has("q")).toBe(false);
    expect(url.searchParams.has("x")).toBe(false);
    expect(buildApiUrl("https://example.com/a?b=1", { c: "2" })).toBe(
      "https://example.com/a?b=1&c=2",
    );
  });

  it("returns parsed JSON on success", async () => {
    authFetch.mockResolvedValue(jsonResponse({ ok: 1 }));
    await expect(apiGet<{ ok: number }>("v1/thing")).resolves.toEqual({ ok: 1 });
  });

  it("returns undefined for 204", async () => {
    authFetch.mockResolvedValue(new Response(null, { status: 204 }));
    await expect(apiPost("v1/thing")).resolves.toBeUndefined();
  });

  it("serializes JSON bodies with a content type", async () => {
    authFetch.mockResolvedValue(jsonResponse({}));
    await apiPost("v1/thing", { a: 1 });
    const init = authFetch.mock.calls[0]?.[1];
    expect(init?.method).toBe("POST");
    expect(init?.body).toBe(JSON.stringify({ a: 1 }));
    expect(new Headers(init?.headers).get("Content-Type")).toBe("application/json");
  });

  it("parses the server envelope into ApiError with normalized punctuation", async () => {
    const body = JSON.stringify({ message: "Bad token", error: "bad_token" });
    authFetch.mockResolvedValue(
      new Response(body, {
        status: 401,
        statusText: "Unauthorized",
        headers: { "Content-Type": "application/json" },
      }),
    );

    const error = await apiFetchJson("https://example.com/api").catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({
      name: "ApiError",
      message: "Bad token.",
      status: 401,
      statusText: "Unauthorized",
      code: "bad_token",
      bodyText: body,
    });
  });

  it("falls back to the status line when the body is not an envelope", async () => {
    authFetch.mockResolvedValue(
      new Response("<html>bad gateway</html>", { status: 502, statusText: "Bad Gateway" }),
    );
    await expect(apiGet("v1/thing")).rejects.toMatchObject({
      status: 502,
      code: undefined,
      message: "HTTP 502: Bad Gateway.",
    });
  });

  it("maps network failures to status 0 / code network", async () => {
    authFetch.mockImplementation(async () => {
      throw new TypeError("Failed to fetch");
    });
    const error = await apiGet("v1/thing").catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 0, code: "network" });
  });

  it("re-throws aborts untouched", async () => {
    const abort = new DOMException("Aborted", "AbortError");
    authFetch.mockImplementation(async () => {
      throw abort;
    });
    const error = await apiGet("v1/thing").catch((e: unknown) => e);
    expect(error).toBe(abort);
  });
});
