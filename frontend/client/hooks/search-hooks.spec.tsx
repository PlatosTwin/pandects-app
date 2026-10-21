// @vitest-environment jsdom
/**
 * Behavioral tests for `useSearch` in each mode. These pin down the shared
 * state machine (draft filters → committed snapshot → fetch → results /
 * selection / error modal) and the per-mode params and telemetry.
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
import { trackEvent } from "@/lib/analytics";
import { useSearch } from "@/hooks/use-search";
import {
  DEALS_SEARCH_CONFIG,
  SECTIONS_SEARCH_CONFIG,
  TAX_CLAUSES_SEARCH_CONFIG,
} from "@/lib/search-configs";

const mockAuthFetch = vi.mocked(authFetch);
const mockTrackEvent = vi.mocked(trackEvent);

const useSections = () => useSearch(SECTIONS_SEARCH_CONFIG);
const useTaxClauses = () => useSearch(TAX_CLAUSES_SEARCH_CONFIG);
const useDeals = () => useSearch(DEALS_SEARCH_CONFIG);

function sectionResult(overrides: Partial<SearchResult> = {}): SearchResult {
  return {
    id: "r1",
    year: 2021,
    target: "Target Co",
    acquirer: "Acquirer Co",
    article_title: "Article I",
    section_title: "Definitions",
    standard_id: [],
    standard_id_labels: {},
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
    access: { tier: "anonymous", message: null },
    count_metadata: {},
    interpretation: {},
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
    tax_standard_id_labels: {},
    year: 2020,
    target: "Target Co",
    acquirer: "Acquirer Co",
    verified: true,
    transaction_price_total: null,
    transaction_consideration: null,
    deal_status: null,
    deal_type: null,
    target_counsel: null,
    acquirer_counsel: null,
    ...overrides,
  };
}

function taxClausesResponse(
  results: TaxClauseSearchResult[],
): TaxClauseSearchResponse {
  return {
    results,
    access: { tier: "anonymous", message: null },
    count_metadata: {},
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
    access: { tier: "anonymous", message: null },
    count_metadata: {},
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

function errorResponse(
  status: number,
  statusText: string,
  envelope?: { error: string; message: string },
): Response {
  const body = envelope ? JSON.stringify(envelope) : "";
  return {
    ok: false,
    status,
    statusText,
    headers: new Headers(envelope ? { "Content-Type": "application/json" } : {}),
    text: async () => body,
    json: async () => (envelope ?? {}),
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

function eventsNamed(name: string) {
  return mockTrackEvent.mock.calls.filter(([event]) => event === name);
}

beforeEach(() => {
  mockAuthFetch.mockReset();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("useSearch (sections)", () => {
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
      await result.current.actions.performSearch({ resetPage: true });
    });
    await waitFor(() => expect(result.current.results).toHaveLength(1));

    expect(result.current.hasSearched).toBe(true);
    const url = lastFetchUrl();
    expect(url.pathname).toContain("v1/sections");
    expect(url.searchParams.getAll("year")).toEqual(["2021"]);
    expect(url.searchParams.get("page")).toBe("1");
    expect(url.searchParams.getAll("metadata")).toEqual([
      "transaction_price_total",
      "deal_status",
      "deal_type",
      "purpose",
    ]);
    expect(url.searchParams.get("sort_by")).toBe("year");
    expect(url.searchParams.get("sort_direction")).toBe("desc");
    expect(result.current.totalCount).toBe(1);
  });

  it("keeps the server's order; sort changes apply on the next commit", async () => {
    mockAuthFetch.mockResolvedValue(
      okResponse(
        sectionsResponse([
          sectionResult({ id: "a", year: 2019, target: "Beta" }),
          sectionResult({ id: "b", year: 2022, target: "Alpha" }),
        ]),
      ),
    );
    const { result } = renderHook(() => useSections(), {
      wrapper: createWrapper(),
    });

    await act(async () => {
      await result.current.actions.performSearch({ resetPage: true });
    });
    await waitFor(() => expect(result.current.results).toHaveLength(2));
    // No client-side re-sort: rows render in the order the API returned.
    expect(result.current.results.map((r) => r.id)).toEqual(["a", "b"]);

    const fetchCount = mockAuthFetch.mock.calls.length;
    act(() => {
      result.current.actions.setSortBy("target");
      result.current.actions.setSortDirection("asc");
    });
    expect(result.current.sortBy).toBe("target");
    expect(result.current.sortDirection).toBe("asc");
    expect(mockAuthFetch.mock.calls.length).toBe(fetchCount);

    await act(async () => {
      await result.current.actions.performSearch();
    });
    await waitFor(() =>
      expect(mockAuthFetch.mock.calls.length).toBe(fetchCount + 1),
    );
    expect(lastFetchUrl().searchParams.get("sort_by")).toBe("target");
    expect(lastFetchUrl().searchParams.get("sort_direction")).toBe("asc");
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
      await result.current.actions.performSearch({ resetPage: true });
    });
    await waitFor(() => expect(result.current.results).toHaveLength(2));

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
      await result.current.actions.performSearch({ resetPage: true });
    });
    await waitFor(() =>
      expect(result.current.selectedResults).toEqual(new Set()),
    );
  });

  it("surfaces server errors via the modal and refetches on retry of the same query", async () => {
    // The hook retries a failed fetch once, so queue two 500s.
    mockAuthFetch
      .mockResolvedValueOnce(errorResponse(500, "Internal Server Error"))
      .mockResolvedValueOnce(errorResponse(500, "Internal Server Error"));
    const { result } = renderHook(() => useSections(), {
      wrapper: createWrapper(),
    });

    await act(async () => {
      await result.current.actions.performSearch({ resetPage: true });
    });
    await waitFor(() => expect(result.current.showErrorModal).toBe(true));
    expect(result.current.errorMessage).toMatch(/HTTP 500/);
    expect(result.current.errorMessage).not.toMatch(/Network error/);
    expect(result.current.error).toMatchObject({ status: 500 });
    expect(mockAuthFetch).toHaveBeenCalledTimes(2);

    // Re-running the identical search must hit the network again (nonce).
    mockAuthFetch.mockResolvedValueOnce(
      okResponse(sectionsResponse([sectionResult()])),
    );
    await act(async () => {
      await result.current.actions.performSearch({ resetPage: true });
    });
    await waitFor(() => expect(result.current.results).toHaveLength(1));
    expect(result.current.showErrorModal).toBe(false);
    expect(mockAuthFetch).toHaveBeenCalledTimes(3);
  });

  it("says 'network error' only when the server could not be reached", async () => {
    mockAuthFetch.mockRejectedValue(new TypeError("Failed to fetch"));
    const { result } = renderHook(() => useSections(), {
      wrapper: createWrapper(),
    });
    await act(async () => {
      await result.current.actions.performSearch({ resetPage: true });
    });
    await waitFor(() => expect(result.current.showErrorModal).toBe(true));
    expect(result.current.errorMessage).toMatch(/Network error/);
    expect(eventsNamed("api_error")).toContainEqual([
      "api_error",
      { endpoint: "v1/sections", kind: "network" },
    ]);
  });

  it.each([
    [403, "Forbidden", "Sign in to see more results.", /Sign in to see more results\./],
    [
      422,
      "Unprocessable Entity",
      "page_size must be at most 100",
      /could not run: page_size must be at most 100\./,
    ],
    [429, "Too Many Requests", "Rate limit exceeded", /Too many searches.*Rate limit exceeded\./],
  ])("shows status-specific text for HTTP %s", async (status, statusText, message, expected) => {
    mockAuthFetch.mockResolvedValue(
      errorResponse(status, statusText, { error: statusText, message }),
    );
    const { result } = renderHook(() => useSections(), {
      wrapper: createWrapper(),
    });
    await act(async () => {
      await result.current.actions.performSearch({ resetPage: true });
    });
    await waitFor(() => expect(result.current.showErrorModal).toBe(true));
    expect(result.current.errorMessage).toMatch(expected);
    expect(result.current.errorMessage).not.toMatch(/Network error/);
    expect(eventsNamed("api_error")).toContainEqual([
      "api_error",
      { endpoint: "v1/sections", status, status_text: statusText },
    ]);
  });

  it("keeps 404 responses silent (no error modal)", async () => {
    mockAuthFetch.mockResolvedValue(errorResponse(404, "Not Found"));
    const { result } = renderHook(() => useSections(), {
      wrapper: createWrapper(),
    });

    await act(async () => {
      await result.current.actions.performSearch({ resetPage: true });
    });
    await waitFor(() => expect(result.current.isSearching).toBe(false));
    expect(result.current.showErrorModal).toBe(false);
    expect(result.current.error).toBeNull();
    expect(result.current.results).toEqual([]);
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
      await result.current.actions.performSearch({ resetPage: true });
    });
    await waitFor(() => expect(result.current.results).toHaveLength(1));
    expect(result.current.totalPages).toBe(3);
    expect(result.current.hasNext).toBe(true);

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
    await waitFor(() =>
      expect(result.current.results.map((r) => r.id)).toEqual(["p2"]),
    );
    expect(result.current.currentPage).toBe(2);
    expect(result.current.hasPrev).toBe(true);
    expect(lastFetchUrl().searchParams.get("page")).toBe("2");
  });

  it("changePageSize resets to page 1 and reports the server-clamped page size", async () => {
    mockAuthFetch.mockResolvedValue(
      okResponse(sectionsResponse([sectionResult()], { page: 1, page_size: 10 })),
    );
    const { result } = renderHook(() => useSections(), {
      wrapper: createWrapper(),
    });
    await act(async () => {
      await result.current.actions.changePageSize(50);
    });
    await waitFor(() => expect(result.current.results).toHaveLength(1));
    expect(lastFetchUrl().searchParams.get("page_size")).toBe("50");
    expect(lastFetchUrl().searchParams.get("page")).toBe("1");
    // The API clamped 50 -> 10 for an anonymous caller.
    await waitFor(() => expect(result.current.pageSize).toBe(10));
    // The draft (and so the shareable URL) keeps the requested size.
    expect(result.current.filters.page_size).toBe(50);
  });

  it("fires performed + results_loaded telemetry once per Search commit", async () => {
    mockAuthFetch.mockResolvedValue(
      okResponse(sectionsResponse([sectionResult()], { total_pages: 4 })),
    );
    const { result } = renderHook(() => useSections(), {
      wrapper: createWrapper(),
    });
    act(() => {
      result.current.actions.updateFilter("year", ["2021", "2022"]);
    });
    await act(async () => {
      await result.current.actions.performSearch({ resetPage: true });
    });
    await waitFor(() => expect(eventsNamed("sections_results_loaded")).toHaveLength(1));
    expect(eventsNamed("sections_performed")).toEqual([
      [
        "sections_performed",
        {
          years_count: 2,
          targets_count: 0,
          acquirers_count: 0,
          clause_types_count: 0,
          page: 1,
          page_size: 25,
          sort_by: "year",
          sort_direction: "desc",
        },
      ],
    ]);
    expect(eventsNamed("sections_results_loaded")[0]?.[1]).toEqual({
      total_count: 1,
      total_pages: 4,
      page: 1,
      page_size: 25,
    });

    // Pagination is not a Search commit: no new performed/results events.
    await act(async () => {
      await result.current.actions.goToPage(2);
    });
    await waitFor(() => expect(lastFetchUrl().searchParams.get("page")).toBe("2"));
    expect(eventsNamed("sections_performed")).toHaveLength(1);
    expect(eventsNamed("sections_results_loaded")).toHaveLength(1);
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
      await result.current.actions.performSearch({ resetPage: true });
    });
    await waitFor(() => expect(result.current.results).toHaveLength(1));

    act(() => {
      result.current.actions.clearFilters();
    });
    expect(result.current.hasSearched).toBe(false);
    expect(result.current.filters.year).toEqual([]);
    expect(result.current.results).toEqual([]);
  });

  it("toggles array filters, trims text filters, and searches an override draft", async () => {
    mockAuthFetch.mockResolvedValue(
      okResponse(sectionsResponse([sectionResult()])),
    );
    const { result } = renderHook(() => useSections(), {
      wrapper: createWrapper(),
    });
    act(() => {
      result.current.actions.toggleFilterValue("deal_type", "merger");
      result.current.actions.toggleFilterValue("deal_type", "tender_offer");
      result.current.actions.toggleFilterValue("deal_type", "merger");
      result.current.actions.setTextFilterValue("agreement_uuid", "  a-1  ");
    });
    expect(result.current.filters.deal_type).toEqual(["tender_offer"]);
    expect(result.current.filters.agreement_uuid).toBe("a-1");

    await act(async () => {
      await result.current.actions.performSearch({
        filters: { year: ["1999"], page: 3, page_size: 25 },
        sortBy: null,
      });
    });
    await waitFor(() => expect(result.current.results).toHaveLength(1));
    const url = lastFetchUrl();
    expect(url.searchParams.getAll("year")).toEqual(["1999"]);
    expect(url.searchParams.get("deal_type")).toBeNull();
    expect(url.searchParams.get("page")).toBe("3");
    expect(url.searchParams.get("sort_by")).toBeNull();
  });
});

describe("useSearch (tax clauses)", () => {
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
      await result.current.actions.performSearch({ resetPage: true });
    });
    await waitFor(() => expect(result.current.results).toHaveLength(1));

    const url = lastFetchUrl();
    expect(url.pathname).toContain("v1/tax-clauses");
    expect(url.searchParams.getAll("tax_standard_id")).toEqual(["tx1", "tx2"]);
    expect(url.searchParams.get("standard_id")).toBeNull();
    expect(url.searchParams.get("include_rep_warranty")).toBe("true");
    expect(eventsNamed("tax_clauses_performed")[0]?.[1]).toMatchObject({
      tax_standard_ids_count: 2,
      include_rep_warranty: true,
    });

    act(() => {
      result.current.actions.setIncludeRepWarranty(false);
    });
    expect(result.current.filters.include_rep_warranty).toBeUndefined();
  });

  it("clears selection on a new committed search", async () => {
    mockAuthFetch.mockResolvedValue(
      okResponse(taxClausesResponse([taxClauseResult({ id: "t1" })])),
    );
    const { result } = renderHook(() => useTaxClauses(), {
      wrapper: createWrapper(),
    });
    await act(async () => {
      await result.current.actions.performSearch({ resetPage: true });
    });
    await waitFor(() => expect(result.current.results).toHaveLength(1));

    act(() => {
      result.current.actions.toggleResultSelection("t1");
    });
    expect(result.current.selectedResults).toEqual(new Set(["t1"]));

    await act(async () => {
      await result.current.actions.performSearch({ resetPage: true });
    });
    await waitFor(() =>
      expect(result.current.selectedResults).toEqual(new Set()),
    );
  });
});

describe("useSearch (deals)", () => {
  it("commits caller-supplied filters, fetches deals, and supports selection + clear", async () => {
    mockAuthFetch.mockResolvedValue(
      okResponse(transactionsResponse(["a1", "a2"])),
    );
    const { result } = renderHook(() => useDeals(), {
      wrapper: createWrapper(),
    });

    expect(result.current.hasSearched).toBe(false);

    await act(async () => {
      await result.current.actions.performSearch({
        filters: { year: ["2020"], page: 1, page_size: 10 },
        sortBy: "year",
        sortDirection: "desc",
        markAsSearched: true,
      });
    });
    await waitFor(() => expect(result.current.results).toHaveLength(2));

    expect(result.current.hasSearched).toBe(true);
    const url = lastFetchUrl();
    expect(url.pathname).toContain("v1/search/agreements");
    expect(url.searchParams.get("sort_by")).toBe("year");
    expect(url.searchParams.getAll("metadata")).toEqual([]);

    act(() => {
      result.current.actions.toggleSelectAll();
    });
    expect(result.current.selectedResults).toEqual(new Set(["a1", "a2"]));

    act(() => {
      result.current.actions.clearResults();
    });
    expect(result.current.hasSearched).toBe(false);
    expect(result.current.selectedResults).toEqual(new Set());
  });

  it("opens the error modal when the fetch fails (404 is not silent here)", async () => {
    mockAuthFetch.mockResolvedValue(errorResponse(404, "Not Found"));
    const { result } = renderHook(() => useDeals(), {
      wrapper: createWrapper(),
    });
    await act(async () => {
      await result.current.actions.performSearch({
        filters: { page: 1, page_size: 10 },
        sortBy: null,
        sortDirection: "desc",
      });
    });
    await waitFor(() => expect(result.current.showErrorModal).toBe(true));
    // Deals mode sends no api_error telemetry for search fetches.
    expect(eventsNamed("api_error")).toEqual([]);
  });
});
