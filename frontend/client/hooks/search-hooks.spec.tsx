// @vitest-environment jsdom
/**
 * Behavioral tests for the three committed-search hooks. These pin down the
 * shared state machine (draft filters → committed snapshot → fetch → results /
 * selection / error modal) so the implementation can be refactored safely.
 */
import { createElement, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { SearchResponse, SearchResult } from "@shared/sections";
import type {
  TaxClauseSearchResponse,
  TaxClauseSearchResult,
} from "@shared/tax-clauses";
import type { TransactionSearchResponse } from "@shared/transactions";

vi.mock("@/lib/auth-fetch", () => ({
  authFetch: vi.fn(),
}));
vi.mock("@/lib/analytics", () => ({
  trackEvent: vi.fn(),
}));
vi.mock("@/lib/logger", () => ({
  logger: { error: vi.fn(), warn: vi.fn(), log: vi.fn(), debug: vi.fn() },
}));

import { authFetch } from "@/lib/auth-fetch";
import { useSections } from "@/hooks/use-sections";
import { useTaxClauses } from "@/hooks/use-tax-clauses";
import { useTransactionSearch } from "@/hooks/use-transaction-search";

const mockAuthFetch = vi.mocked(authFetch);

function sectionResult(overrides: Partial<SearchResult> = {}): SearchResult {
  return {
    id: "r1",
    year: "2021",
    target: "Target Co",
    acquirer: "Acquirer Co",
    article_title: "Article I",
    section_title: "Definitions",
    standard_id: [],
    xml: "<text>Body</text>",
    section_uuid: "s-uuid",
    agreement_uuid: "a-uuid",
    verified: true,
    ...overrides,
  };
}

function sectionsResponse(
  results: SearchResult[],
  overrides: Partial<SearchResponse> = {},
): SearchResponse {
  return {
    results,
    access: { tier: "anonymous" },
    total_count: results.length,
    total_count_is_approximate: false,
    page: 1,
    page_size: 10,
    total_pages: 1,
    has_next: false,
    has_prev: false,
    next_num: null,
    prev_num: null,
    ...overrides,
  };
}

function taxClauseResult(
  overrides: Partial<TaxClauseSearchResult> = {},
): TaxClauseSearchResult {
  return {
    id: "t1",
    clause_uuid: "c-uuid",
    agreement_uuid: "a-uuid",
    section_uuid: "s-uuid",
    clause_text: "Tax clause text",
    anchor_label: null,
    context_type: "covenant" as TaxClauseSearchResult["context_type"],
    source_method: null,
    tax_standard_ids: ["tx1"],
    year: "2020",
    target: "Target Co",
    acquirer: "Acquirer Co",
    verified: true,
    ...overrides,
  };
}

function taxClausesResponse(
  results: TaxClauseSearchResult[],
): TaxClauseSearchResponse {
  return {
    results,
    access: { tier: "anonymous" },
    total_count: results.length,
    total_count_is_approximate: false,
    page: 1,
    page_size: 10,
    total_pages: 1,
    has_next: false,
    has_prev: false,
    next_num: null,
    prev_num: null,
  } as TaxClauseSearchResponse;
}

function transactionsResponse(
  agreementUuids: string[],
): TransactionSearchResponse {
  return {
    results: agreementUuids.map((agreement_uuid) => ({
      agreement_uuid,
    })) as TransactionSearchResponse["results"],
    access: { tier: "anonymous" },
    total_count: agreementUuids.length,
    total_count_is_approximate: false,
    page: 1,
    page_size: 10,
    total_pages: 1,
    has_next: false,
    has_prev: false,
    next_num: null,
    prev_num: null,
  } as TransactionSearchResponse;
}

function okResponse(body: unknown): Response {
  return {
    ok: true,
    status: 200,
    statusText: "OK",
    json: async () => body,
  } as Response;
}

function errorResponse(status: number, statusText: string): Response {
  return {
    ok: false,
    status,
    statusText,
    json: async () => ({}),
  } as Response;
}

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      // The hooks set their own `retry` predicate; retryDelay: 0 keeps the
      // built-in single retry from stalling waitFor in the error tests.
      queries: {
        refetchOnWindowFocus: false,
        retry: false,
        retryDelay: 0,
        staleTime: 0,
      },
    },
  });
  return function Wrapper({ children }: { children: ReactNode }) {
    return createElement(
      QueryClientProvider,
      { client: queryClient },
      children,
    );
  };
}

function lastFetchUrl(): URL {
  const call = mockAuthFetch.mock.calls.at(-1);
  if (!call) throw new Error("authFetch was never called");
  return new URL(String(call[0]), "http://localhost");
}

beforeEach(() => {
  mockAuthFetch.mockReset();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("useSections", () => {
  it("does not fetch until a search is committed; commit fetches and exposes results", async () => {
    mockAuthFetch.mockResolvedValue(
      okResponse(sectionsResponse([sectionResult()])),
    );
    const { result } = renderHook(() => useSections(), {
      wrapper: createWrapper(),
    });

    expect(result.current.hasSearched).toBe(false);
    expect(mockAuthFetch).not.toHaveBeenCalled();

    act(() => {
      result.current.actions.updateFilter("year", ["2021"]);
    });
    // Editing the draft does not fetch.
    expect(mockAuthFetch).not.toHaveBeenCalled();

    await act(async () => {
      await result.current.actions.performSearch(true);
    });
    await waitFor(() =>
      expect(result.current.searchResults).toHaveLength(1),
    );

    expect(result.current.hasSearched).toBe(true);
    const url = lastFetchUrl();
    expect(url.pathname).toContain("v1/sections");
    expect(url.searchParams.getAll("year")).toEqual(["2021"]);
    expect(url.searchParams.get("page")).toBe("1");
    expect(result.current.total_count).toBe(1);
  });

  it("sorts results client-side without refetching", async () => {
    mockAuthFetch.mockResolvedValue(
      okResponse(
        sectionsResponse([
          sectionResult({ id: "a", year: "2019", target: "Beta" }),
          sectionResult({ id: "b", year: "2022", target: "Alpha" }),
        ]),
      ),
    );
    const { result } = renderHook(() => useSections(), {
      wrapper: createWrapper(),
    });

    await act(async () => {
      await result.current.actions.performSearch(true);
    });
    await waitFor(() =>
      expect(result.current.searchResults).toHaveLength(2),
    );
    // Default sort: year desc.
    expect(result.current.searchResults.map((r) => r.id)).toEqual(["b", "a"]);

    const fetchCount = mockAuthFetch.mock.calls.length;
    act(() => {
      result.current.actions.sortResults("target");
      result.current.actions.setSortDirection("asc");
    });
    expect(result.current.searchResults.map((r) => r.id)).toEqual(["b", "a"]);
    // Alpha (b) before Beta (a) ascending — already in place; flip to verify.
    act(() => {
      result.current.actions.toggleSortDirection();
    });
    expect(result.current.searchResults.map((r) => r.id)).toEqual(["a", "b"]);
    expect(mockAuthFetch.mock.calls.length).toBe(fetchCount);
  });

  it("tracks selection and clears it on a new committed search", async () => {
    mockAuthFetch.mockResolvedValue(
      okResponse(
        sectionsResponse([
          sectionResult({ id: "a" }),
          sectionResult({ id: "b" }),
        ]),
      ),
    );
    const { result } = renderHook(() => useSections(), {
      wrapper: createWrapper(),
    });

    await act(async () => {
      await result.current.actions.performSearch(true);
    });
    await waitFor(() =>
      expect(result.current.searchResults).toHaveLength(2),
    );

    act(() => {
      result.current.actions.toggleResultSelection("a");
    });
    expect(result.current.selectedResults).toEqual(new Set(["a"]));

    act(() => {
      result.current.actions.toggleSelectAll();
    });
    expect(result.current.selectedResults).toEqual(new Set(["a", "b"]));

    act(() => {
      result.current.actions.toggleSelectAll();
    });
    expect(result.current.selectedResults).toEqual(new Set());

    act(() => {
      result.current.actions.toggleResultSelection("b");
    });
    await act(async () => {
      await result.current.actions.performSearch(true);
    });
    await waitFor(() =>
      expect(result.current.selectedResults).toEqual(new Set()),
    );
  });

  it("surfaces non-404 errors via the modal and refetches on retry of the same query", async () => {
    // The hook retries a failed fetch once, so queue two 500s.
    mockAuthFetch
      .mockResolvedValueOnce(errorResponse(500, "Internal Server Error"))
      .mockResolvedValueOnce(errorResponse(500, "Internal Server Error"));
    const { result } = renderHook(() => useSections(), {
      wrapper: createWrapper(),
    });

    await act(async () => {
      await result.current.actions.performSearch(true);
    });
    await waitFor(() => expect(result.current.showErrorModal).toBe(true));
    expect(result.current.errorMessage).toMatch(/Network error/);
    expect(mockAuthFetch).toHaveBeenCalledTimes(2);

    // Re-running the identical search must hit the network again (nonce).
    mockAuthFetch.mockResolvedValueOnce(
      okResponse(sectionsResponse([sectionResult()])),
    );
    await act(async () => {
      await result.current.actions.performSearch(true);
    });
    await waitFor(() =>
      expect(result.current.searchResults).toHaveLength(1),
    );
    expect(result.current.showErrorModal).toBe(false);
    expect(mockAuthFetch).toHaveBeenCalledTimes(3);
  });

  it("keeps 404 responses silent (no error modal)", async () => {
    mockAuthFetch.mockResolvedValue(errorResponse(404, "Not Found"));
    const { result } = renderHook(() => useSections(), {
      wrapper: createWrapper(),
    });

    await act(async () => {
      await result.current.actions.performSearch(true);
    });
    await waitFor(() => expect(result.current.isSearching).toBe(false));
    expect(result.current.showErrorModal).toBe(false);
    expect(result.current.searchResults).toEqual([]);
  });

  it("paginates via goToPage and mirrors the fetched page into the draft", async () => {
    mockAuthFetch.mockResolvedValueOnce(
      okResponse(
        sectionsResponse([sectionResult()], {
          page: 1,
          total_pages: 3,
          has_next: true,
        }),
      ),
    );
    const { result } = renderHook(() => useSections(), {
      wrapper: createWrapper(),
    });
    await act(async () => {
      await result.current.actions.performSearch(true);
    });
    await waitFor(() =>
      expect(result.current.searchResults).toHaveLength(1),
    );

    mockAuthFetch.mockResolvedValueOnce(
      okResponse(
        sectionsResponse([sectionResult({ id: "p2" })], {
          page: 2,
          total_pages: 3,
          has_next: true,
          has_prev: true,
        }),
      ),
    );
    await act(async () => {
      await result.current.actions.goToPage(2);
    });
    await waitFor(() => expect(result.current.currentPage).toBe(2));
    expect(lastFetchUrl().searchParams.get("page")).toBe("2");
    expect(result.current.searchResults.map((r) => r.id)).toEqual(["p2"]);
  });

  it("clearFilters resets the draft and uncommits the search", async () => {
    mockAuthFetch.mockResolvedValue(
      okResponse(sectionsResponse([sectionResult()])),
    );
    const { result } = renderHook(() => useSections(), {
      wrapper: createWrapper(),
    });
    act(() => {
      result.current.actions.updateFilter("year", ["2021"]);
    });
    await act(async () => {
      await result.current.actions.performSearch(true);
    });
    await waitFor(() =>
      expect(result.current.searchResults).toHaveLength(1),
    );

    act(() => {
      result.current.actions.clearFilters();
    });
    expect(result.current.hasSearched).toBe(false);
    expect(result.current.filters.year).toEqual([]);
    expect(result.current.searchResults).toEqual([]);
  });
});

describe("useTaxClauses", () => {
  it("maps clauseType to tax_standard_id params and forwards include_rep_warranty", async () => {
    mockAuthFetch.mockResolvedValue(
      okResponse(taxClausesResponse([taxClauseResult()])),
    );
    const { result } = renderHook(() => useTaxClauses(), {
      wrapper: createWrapper(),
    });

    act(() => {
      result.current.actions.updateFilter("clauseType", ["tx1", "tx2"]);
      result.current.actions.setIncludeRepWarranty(true);
    });
    await act(async () => {
      await result.current.actions.performSearch(true);
    });
    await waitFor(() =>
      expect(result.current.searchResults).toHaveLength(1),
    );

    const url = lastFetchUrl();
    expect(url.pathname).toContain("v1/tax-clauses");
    expect(url.searchParams.getAll("tax_standard_id")).toEqual(["tx1", "tx2"]);
    expect(url.searchParams.get("standard_id")).toBeNull();
    expect(url.searchParams.get("include_rep_warranty")).toBe("true");
  });

  it("clears selection on a new committed search", async () => {
    mockAuthFetch.mockResolvedValue(
      okResponse(taxClausesResponse([taxClauseResult({ id: "t1" })])),
    );
    const { result } = renderHook(() => useTaxClauses(), {
      wrapper: createWrapper(),
    });
    await act(async () => {
      await result.current.actions.performSearch(true);
    });
    await waitFor(() =>
      expect(result.current.searchResults).toHaveLength(1),
    );

    act(() => {
      result.current.actions.toggleResultSelection("t1");
    });
    expect(result.current.selectedResults).toEqual(new Set(["t1"]));

    await act(async () => {
      await result.current.actions.performSearch(true);
    });
    await waitFor(() =>
      expect(result.current.selectedResults).toEqual(new Set()),
    );
  });
});

describe("useTransactionSearch", () => {
  it("commits caller-supplied filters, fetches deals, and supports selection + clear", async () => {
    mockAuthFetch.mockResolvedValue(
      okResponse(transactionsResponse(["a1", "a2"])),
    );
    const { result } = renderHook(() => useTransactionSearch(), {
      wrapper: createWrapper(),
    });

    expect(result.current.hasSearched).toBe(false);

    await act(async () => {
      await result.current.performSearch({
        filters: { year: ["2020"], page: 1, page_size: 10 },
        sortBy: "year",
        sortDirection: "desc",
      });
    });
    await waitFor(() => expect(result.current.results).toHaveLength(2));

    expect(result.current.hasSearched).toBe(true);
    const url = lastFetchUrl();
    expect(url.pathname).toContain("v1/search/agreements");
    expect(url.searchParams.get("sort_by")).toBe("year");

    act(() => {
      result.current.toggleSelectAll();
    });
    expect(result.current.selectedResults).toEqual(new Set(["a1", "a2"]));

    act(() => {
      result.current.clear();
    });
    expect(result.current.hasSearched).toBe(false);
    expect(result.current.selectedResults).toEqual(new Set());
  });

  it("opens the error modal when the fetch fails", async () => {
    mockAuthFetch.mockResolvedValue(errorResponse(500, "Internal Server Error"));
    const { result } = renderHook(() => useTransactionSearch(), {
      wrapper: createWrapper(),
    });
    await act(async () => {
      await result.current.performSearch({
        filters: { page: 1, page_size: 10 },
        sortBy: null,
        sortDirection: "desc",
      });
    });
    await waitFor(() => expect(result.current.showErrorModal).toBe(true));
  });
});
