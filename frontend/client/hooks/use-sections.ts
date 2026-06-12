import { useState, useCallback, useEffect, useMemo, useRef } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { SearchFilters, SearchResult, SearchResponse } from "@shared/sections";
import { apiUrl } from "@/lib/api-config";
import { buildSearchParams } from "@/lib/url-params";
import type { ClauseTypeTree } from "@/lib/clause-types";
import { trackEvent } from "@/lib/analytics";
import { authFetch } from "@/lib/auth-fetch";
import {
  DEFAULT_PAGE_SIZE,
  DEFAULT_PAGE,
  LARGE_PAGE_SIZE_FOR_CSV,
} from "@/lib/constants";
import { logger } from "@/lib/logger";
import { keys } from "@/lib/query-keys";
import {
  useCommittedSearchCore,
  type CommittedQuery,
  type SortDirection,
  type SortField,
} from "@/hooks/use-committed-search";

const EMPTY_FILTERS: SearchFilters = {
  year: [],
  target: [],
  acquirer: [],
  clauseType: [],
  standard_id: [],
  transaction_price_total: [],
  transaction_price_stock: [],
  transaction_price_cash: [],
  transaction_price_assets: [],
  transaction_consideration: [],
  target_type: [],
  acquirer_type: [],
  target_counsel: [],
  acquirer_counsel: [],
  target_industry: [],
  acquirer_industry: [],
  deal_status: [],
  attitude: [],
  deal_type: [],
  purpose: [],
  target_pe: [],
  acquirer_pe: [],
  page: DEFAULT_PAGE,
  page_size: DEFAULT_PAGE_SIZE,
};

const EMPTY_ACCESS: SearchResponse["access"] = { tier: "anonymous" };

function sortResultsArray(
  results: SearchResult[],
  sort_by: SortField | null,
  direction: SortDirection,
): SearchResult[] {
  if (!sort_by) return results;
  return [...results].sort((a, b) => {
    let comparison = 0;
    switch (sort_by) {
      case "year":
        comparison = parseInt(a.year, 10) - parseInt(b.year, 10);
        break;
      case "target":
        comparison = a.target.localeCompare(b.target);
        break;
      case "acquirer":
        comparison = a.acquirer.localeCompare(b.acquirer);
        break;
      default:
        return 0;
    }
    return direction === "desc" ? -comparison : comparison;
  });
}

function buildSectionsQueryString(
  committed: CommittedQuery<SearchFilters>,
): string {
  const params = buildSearchParams(committed.filters, committed.clauseTypesNested);
  if (committed.sortBy) {
    params.append("sort_by", committed.sortBy);
    params.append("sort_direction", committed.sortDirection);
  }
  return params.toString();
}

const SECTIONS_SEARCH_CONFIG = {
  buildQueryString: buildSectionsQueryString,
  fetchUrl: (queryString: string) => apiUrl(`v1/sections?${queryString}`),
  queryKey: (params: { q: string; n: number }) => keys.sections.search(params),
  getResults: (response: SearchResponse) => response.results,
  getResultId: (result: SearchResult) => result.id,
  silentNotFound: true,
  logLabel: "Search failed:",
  trackHttpError: (status: number, statusText: string) => {
    trackEvent("api_error", {
      endpoint: "api/sections",
      status,
      status_text: statusText,
    });
  },
  trackFailure: (error: unknown) => {
    trackEvent("api_error", {
      endpoint: "api/sections",
      kind:
        error instanceof TypeError && error.message.includes("fetch")
          ? "network"
          : "unknown",
    });
  },
};

/**
 * Declarative sections search.
 *
 * `filters` is the draft (what the sidebar is editing). Results come from a
 * `useQuery` driven by `committed` — the snapshot last submitted via
 * `performSearch`, `goToPage`, etc. Editing a sidebar filter does NOT refetch.
 *
 * Sort changes are applied both server-side (via the committed query) and
 * client-side via a `useMemo` over the fetched results. The client-side pass
 * keeps the visible order responsive when the user clicks a sort header
 * before the follow-up fetch lands.
 */
export function useSections() {
  const queryClient = useQueryClient();
  const [filters, setFilters] = useState<SearchFilters>({ ...EMPTY_FILTERS });
  const [currentSort, setCurrentSort] = useState<SortField | null>("year");
  const [sort_direction, setSortDirection] = useState<SortDirection>("desc");
  const fireResultsLoadedRef = useRef(false);

  const core = useCommittedSearchCore<SearchFilters, SearchResponse, SearchResult>(
    SECTIONS_SEARCH_CONFIG,
  );
  const { committed, responseData } = core;

  const searchResults = useMemo(
    () => sortResultsArray(core.results, currentSort, sort_direction),
    [core.results, currentSort, sort_direction],
  );
  const total_count = responseData?.total_count ?? 0;
  const totalCountIsApproximate = responseData?.total_count_is_approximate ?? false;
  const total_pages = responseData?.total_pages ?? 0;
  const has_next = responseData?.has_next ?? false;
  const has_prev = responseData?.has_prev ?? false;
  const access = responseData?.access ?? EMPTY_ACCESS;
  const hasSearched = committed !== null;

  // Fire results-loaded telemetry exactly once per Search/Enter commit.
  useEffect(() => {
    if (!responseData || !fireResultsLoadedRef.current || !committed) return;
    fireResultsLoadedRef.current = false;
    trackEvent("sections_results_loaded", {
      total_count: responseData.total_count,
      total_pages: responseData.total_pages,
      page: committed.filters.page,
      page_size: committed.filters.page_size,
    });
  }, [responseData, committed]);

  // Mirror committed page/page_size into the draft so pagination controls,
  // URL sync, and CSV use what we actually fetched.
  useEffect(() => {
    if (!responseData) return;
    setFilters((prev) =>
      prev.page === responseData.page && prev.page_size === responseData.page_size
        ? prev
        : { ...prev, page: responseData.page, page_size: responseData.page_size },
    );
  }, [responseData]);

  const updateFilter = useCallback(
    (field: keyof SearchFilters, value: string | string[] | number) => {
      if (field === "page" || field === "page_size") {
        setFilters((prev) => ({ ...prev, [field]: value as number }));
      } else {
        setFilters((prev) => ({ ...prev, [field]: value as string[] }));
      }
    },
    [],
  );

  const hydrateFilters = useCallback((next: Partial<SearchFilters>) => {
    setFilters((prev) => ({
      ...prev,
      ...next,
      page: next.page ?? prev.page,
      page_size: next.page_size ?? prev.page_size,
    }));
  }, []);

  const toggleFilterValue = useCallback(
    (field: keyof SearchFilters, value: string) => {
      if (field === "page" || field === "page_size") return;
      setFilters((prev) => {
        const currentValues = (prev[field] as string[]) || [];
        const newValues = currentValues.includes(value)
          ? currentValues.filter((v) => v !== value)
          : [...currentValues, value];
        return { ...prev, [field]: newValues };
      });
    },
    [],
  );

  const setTextFilterValue = useCallback(
    (field: keyof SearchFilters, value: string) => {
      if (field === "page" || field === "page_size") return;
      const trimmedValue = value.trim();
      setFilters((prev) => ({
        ...prev,
        [field]: trimmedValue || undefined,
      }));
    },
    [],
  );

  const setSortDirectionDirect = useCallback((direction: SortDirection) => {
    setSortDirection(direction);
  }, []);

  const performSearch = useCallback(
    async (
      resetPage = false,
      clauseTypesNested?: ClauseTypeTree,
      markAsSearched: boolean = resetPage,
      filtersOverride?: SearchFilters,
      overrideSortBy?: SortField | null,
      overrideSortDirection?: SortDirection,
    ) => {
      const baseFilters = filtersOverride ?? filters;
      const effective: SearchFilters = resetPage
        ? { ...baseFilters, page: 1 }
        : baseFilters;
      if (resetPage && !filtersOverride) {
        setFilters((prev) => ({ ...prev, page: 1 }));
      }
      const sortBy = overrideSortBy ?? currentSort;
      const sortDir = overrideSortDirection ?? sort_direction;

      if (markAsSearched) {
        fireResultsLoadedRef.current = true;
        trackEvent("sections_performed", {
          years_count: effective.year?.length ?? 0,
          targets_count: effective.target?.length ?? 0,
          acquirers_count: effective.acquirer?.length ?? 0,
          clause_types_count: effective.clauseType?.length ?? 0,
          standard_ids_count: effective.standard_id?.length ?? 0,
          page: effective.page,
          page_size: effective.page_size,
          sort_by: sortBy ?? "none",
          sort_direction: sortDir,
        });
      }

      core.commit({
        filters: effective,
        clauseTypesNested,
        sortBy,
        sortDirection: sortDir,
      });
    },
    [filters, currentSort, sort_direction, core.commit],
  );

  const downloadCSV = useCallback(
    async (clauseTypesNested?: ClauseTypeTree) => {
      let resultsToDownload: SearchResult[] = [];

      if (core.selectedResults.size > 0) {
        resultsToDownload = searchResults.filter((result) =>
          core.selectedResults.has(result.id),
        );
      } else {
        try {
          const searchFilters = {
            ...filters,
            page: undefined,
            page_size: undefined,
          };
          const params = buildSearchParams(searchFilters, clauseTypesNested, false);
          params.append("page_size", LARGE_PAGE_SIZE_FOR_CSV.toString());
          params.append("page", DEFAULT_PAGE.toString());
          const queryString = params.toString();
          const searchResponse = await queryClient.fetchQuery({
            queryKey: keys.sections.search({ q: queryString, csv: true }),
            queryFn: async () => {
              const res = await authFetch(apiUrl(`v1/sections?${queryString}`));
              if (!res.ok) {
                trackEvent("api_error", {
                  endpoint: "api/sections",
                  status: res.status,
                  status_text: res.statusText,
                });
                throw new Error(`HTTP ${res.status}: ${res.statusText}`);
              }
              return (await res.json()) as SearchResponse;
            },
            staleTime: 60 * 1000,
          });
          resultsToDownload = searchResponse.results;
        } catch (error) {
          logger.error("Failed to fetch all results for CSV:", error);
          trackEvent("api_error", {
            endpoint: "api/sections",
            kind:
              error instanceof TypeError && error.message.includes("fetch")
                ? "network"
                : "unknown",
          });
          resultsToDownload = searchResults;
        }
      }

      if (resultsToDownload.length === 0) return;

      trackEvent("sections_csv_download_click", {
        mode: core.selectedResults.size > 0 ? "selected" : "all",
        selected_count: core.selectedResults.size,
        downloaded_count: resultsToDownload.length,
        years_count: filters.year?.length ?? 0,
        targets_count: filters.target?.length ?? 0,
        acquirers_count: filters.acquirer?.length ?? 0,
        clause_types_count: filters.clauseType?.length ?? 0,
        standard_ids_count: filters.standard_id?.length ?? 0,
      });

      const headers = [
        "Year",
        "Target",
        "Acquirer",
        "Article Title",
        "Section Title",
        "Text",
        "Section UUID",
        "Agreement UUID",
      ];

      const csvContent = [
        headers.join(","),
        ...resultsToDownload.map((result) =>
          [
            result.year,
            `"${result.target}"`,
            `"${result.acquirer}"`,
            `"${result.article_title}"`,
            `"${result.section_title}"`,
            `"${(result.xml ?? "").replace(/"/g, '""')}"`,
            result.section_uuid,
            result.agreement_uuid,
          ].join(","),
        ),
      ].join("\n");

      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      const link = document.createElement("a");
      const objectUrl = URL.createObjectURL(blob);
      link.href = objectUrl;
      const selectedText = core.selectedResults.size > 0 ? "_selected" : "";
      link.download = `ma_clauses${selectedText}_${new Date().toISOString().split("T")[0]}.csv`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
    },
    [filters, searchResults, core.selectedResults, queryClient],
  );

  const clearFilters = useCallback(() => {
    setFilters({
      ...EMPTY_FILTERS,
      agreement_uuid: undefined,
      section_uuid: undefined,
    });
    core.uncommit();
  }, [core.uncommit]);

  const goToPage = useCallback(
    async (page: number, clauseTypesNested?: ClauseTypeTree) => {
      const nextFilters: SearchFilters = { ...filters, page };
      setFilters(nextFilters);
      await performSearch(false, clauseTypesNested, false, nextFilters);
    },
    [filters, performSearch],
  );

  const changePageSize = useCallback(
    async (page_size: number, clauseTypesNested?: ClauseTypeTree) => {
      const nextFilters: SearchFilters = { ...filters, page_size, page: 1 };
      setFilters(nextFilters);
      await performSearch(false, clauseTypesNested, false, nextFilters);
    },
    [filters, performSearch],
  );

  const sortResults = useCallback((sort_by: SortField) => {
    setCurrentSort(sort_by);
  }, []);

  const toggleSortDirection = useCallback(() => {
    setSortDirection((d) => (d === "asc" ? "desc" : "asc"));
  }, []);

  return {
    filters,
    isSearching: core.isSearching,
    searchResults,
    selectedResults: core.selectedResults,
    hasSearched,
    access,
    total_count,
    totalCountIsApproximate,
    total_pages,
    has_next,
    has_prev,
    currentSort,
    currentPage: filters.page || 1,
    page_size: filters.page_size || DEFAULT_PAGE_SIZE,
    showErrorModal: core.showErrorModal,
    errorMessage: core.errorMessage,
    sort_direction,
    actions: {
      updateFilter,
      hydrateFilters,
      toggleFilterValue,
      setTextFilterValue,
      performSearch,
      downloadCSV,
      clearFilters,
      goToPage,
      changePageSize,
      closeErrorModal: core.closeErrorModal,
      sortResults,
      toggleSortDirection,
      setSortDirection: setSortDirectionDirect,
      toggleResultSelection: core.toggleResultSelection,
      toggleSelectAll: core.toggleSelectAll,
      clearSelection: core.clearSelection,
    },
  };
}
