// @vitest-environment jsdom
/**
 * `clause_types` comes from the `/v1/taxonomy` cache, never from
 * `/v1/filter-options`, so every page shares one copy of the tree.
 */
import { createElement, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/auth-fetch", () => ({
  authFetch: vi.fn(),
}));
vi.mock("@/lib/analytics", () => ({
  trackEvent: vi.fn(),
}));

import { authFetch } from "@/lib/auth-fetch";
import { useFilterOptions } from "@/hooks/use-filter-options";

const mockAuthFetch = vi.mocked(authFetch);

const TREE = { Termination: { id: "t1", children: {} } };

function okResponse(body: unknown): Response {
  return { ok: true, status: 200, statusText: "OK", json: async () => body } as Response;
}

function requestedUrls(): string[] {
  return mockAuthFetch.mock.calls.map(([input]) => String(input));
}

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: { children: ReactNode }) =>
    createElement(QueryClientProvider, { client: queryClient }, children);
}

afterEach(() => {
  mockAuthFetch.mockReset();
});

describe("useFilterOptions", () => {
  it("serves clause_types from /v1/taxonomy without calling /v1/filter-options", async () => {
    mockAuthFetch.mockImplementation(async () => okResponse(TREE));

    const { result } = renderHook(() => useFilterOptions({ fields: ["clause_types"] }), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.clause_types).toEqual(TREE));
    const urls = requestedUrls();
    expect(urls.some((url) => url.includes("v1/taxonomy"))).toBe(true);
    expect(urls.some((url) => url.includes("v1/filter-options"))).toBe(false);
  });

  it("requests only the summary fields and returns industry labels", async () => {
    mockAuthFetch.mockImplementation(async (input) =>
      String(input).includes("v1/filter-options")
        ? okResponse({
            target_industries: ["311"],
            industry_labels: { "311": "Food Manufacturing" },
          })
        : okResponse(TREE),
    );

    const { result } = renderHook(
      () => useFilterOptions({ fields: ["target_industries", "clause_types"] }),
      { wrapper: createWrapper() },
    );

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.industry_labels).toEqual({ "311": "Food Manufacturing" });
    expect(result.current.clause_types).toEqual(TREE);
    const filterOptionsUrl = requestedUrls().find((url) => url.includes("v1/filter-options"));
    expect(filterOptionsUrl).toContain("fields=target_industries");
    expect(filterOptionsUrl).not.toContain("clause_types");
  });

  it("retries a failed filter-options load", async () => {
    let failOptions = true;
    mockAuthFetch.mockImplementation(async (input) => {
      if (!String(input).includes("v1/filter-options")) return okResponse(TREE);
      if (failOptions) {
        return {
          ok: false,
          status: 503,
          statusText: "Service Unavailable",
          headers: new Headers(),
          json: async () => ({}),
          text: async () => "",
        } as Response;
      }
      return okResponse({ target_counsels: ["Wachtell"] });
    });

    const { result } = renderHook(() => useFilterOptions(), { wrapper: createWrapper() });

    await waitFor(() => expect(result.current.error).not.toBeNull());
    failOptions = false;
    result.current.retry();
    await waitFor(() => expect(result.current.target_counsels).toEqual(["Wachtell"]));
    expect(result.current.error).toBeNull();
  });
});
