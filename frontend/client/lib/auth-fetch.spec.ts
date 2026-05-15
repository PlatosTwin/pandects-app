import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

function installBrowserGlobals(cookie = "") {
  vi.stubGlobal("window", globalThis);
  vi.stubGlobal("document", { cookie });
}

async function importAuthFetchModule() {
  vi.resetModules();
  return await import("./auth-fetch");
}

describe("auth-fetch", () => {
  beforeEach(() => {
    vi.stubEnv("VITE_AUTH_SESSION_TRANSPORT", "cookie");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it("throws during SSR so prerender fetch regressions fail loudly", async () => {
    const { authFetch } = await importAuthFetchModule();

    await expect(authFetch("https://example.com/api")).rejects.toThrow(
      "authFetch called during SSR",
    );
  });

  it("adds the bearer token without overriding an existing authorization header", async () => {
    installBrowserGlobals();
    vi.stubEnv("VITE_AUTH_SESSION_TRANSPORT", "bearer");
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);

    const { authFetch } = await importAuthFetchModule();
    const { setSessionToken } = await import("./auth-session");
    setSessionToken("session-token");

    await authFetch("https://example.com/api");

    expect(fetchMock).toHaveBeenCalledWith(
      "https://example.com/api",
      expect.objectContaining({
        headers: expect.any(Headers),
      }),
    );
    const headers = fetchMock.mock.calls[0]?.[1]?.headers;
    expect(headers).toBeInstanceOf(Headers);
    expect((headers as Headers).get("Authorization")).toBe(
      "Bearer session-token",
    );

    fetchMock.mockClear();

    await authFetch("https://example.com/api", {
      headers: { Authorization: "Bearer existing-token" },
    });

    const overrideHeaders = fetchMock.mock.calls[0]?.[1]?.headers;
    expect((overrideHeaders as Headers).get("Authorization")).toBe(
      "Bearer existing-token",
    );
  });

  it("bootstraps CSRF from the endpoint response when the cookie is unavailable", async () => {
    installBrowserGlobals("");
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ csrf_token: "csrf-from-body" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ csrf_token: "csrf-from-body" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      )
      .mockResolvedValueOnce(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);

    const { authFetch } = await importAuthFetchModule();
    await authFetch("https://example.com/api", { method: "POST" });

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      expect.stringContaining("/v1/auth/csrf"),
      { credentials: "include" },
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      expect.stringContaining("/v1/auth/csrf"),
      { credentials: "include" },
    );
    const requestHeaders = fetchMock.mock.calls[2]?.[1]?.headers;
    expect((requestHeaders as Headers).get("X-CSRF-Token")).toBe(
      "csrf-from-body",
    );
    expect(fetchMock.mock.calls[2]?.[1]).toEqual(
      expect.objectContaining({
        credentials: "include",
        method: "POST",
      }),
    );
  });

  it("surfaces JSON API errors with normalized punctuation and code", async () => {
    installBrowserGlobals();
    vi.stubGlobal(
      "fetch",
      vi.fn<typeof fetch>().mockResolvedValue(
        new Response(JSON.stringify({ message: "Bad token", error: "bad_token" }), {
          status: 401,
          statusText: "Unauthorized",
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );

    const { AuthApiError, authFetchJson } = await importAuthFetchModule();

    await expect(authFetchJson("https://example.com/api")).rejects.toMatchObject({
      name: "AuthApiError",
      message: "Bad token.",
      status: 401,
      statusText: "Unauthorized",
      code: "bad_token",
      bodyText: JSON.stringify({ message: "Bad token", error: "bad_token" }),
    });
    await expect(authFetchJson("https://example.com/api")).rejects.toBeInstanceOf(
      AuthApiError,
    );
  });
});
