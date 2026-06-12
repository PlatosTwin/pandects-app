import { useCallback, useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { SearchFilters } from "@shared/sections";
import type {
  TaxClauseSearchResponse,
  TaxClauseSearchResult,
} from "@shared/tax-clauses";
import { apiUrl } from "@/lib/api-config";
import { buildSearchParams } from "@/lib/url-params";
import { authFetch } from "@/lib/auth-fetch";
import { trackEvent } from "@/lib/analytics";
import {
  DEFAULT_PAGE,
  DEFAULT_PAGE_SIZE,
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

export interface TaxClauseFilters extends SearchFilters {
  include_rep_warranty?: boolean;
}

const EMPTY_FILTERS: TaxClauseFilters = {
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
  include_rep_warranty: false,
  page: DEFAULT_PAGE,
  page_size: DEFAULT_PAGE_SIZE,
};

const EMPTY_ACCESS: TaxClauseSearchResponse["access"] = { tier: "anonymous" };

function buildTaxClauseParams(
  filters: TaxClauseFilters,
  includePagination = true,
): URLSearchParams {
  const params = buildSearchParams(filters, undefined, includePagination);
  params.delete("standard_id");
  if (filters.clauseType && filters.clauseType.length > 0) {
    filters.clauseType.forEach((standard_id) =>
      params.append("tax_standard_id", standard_id),
    );
  }
  if (filters.include_rep_warranty) {
    params.set("include_rep_warranty", "true");
  }
  return params;
}

function buildCommittedQueryString(
  committed: CommittedQuery<TaxClauseFilters>,
): string {
  const params = buildTaxClauseParams(committed.filters);
  if (committed.sortBy) {
    params.append("sort_by", committed.sortBy);
    params.append("sort_direction", committed.sortDirection);
  }
  return params.toString();
}

const TAX_CLAUSES_SEARCH_CONFIG = {
  buildQueryString: buildCommittedQueryString,
  fetchUrl: (queryString: string) => apiUrl(`v1/tax-clauses?${queryString}`),
  queryKey: (params: { q: string; n: number }) =>
    keys.taxClauses.search(params),
  getResults: (response: TaxClauseSearchResponse) => response.results,
  getResultId: (result: TaxClauseSearchResult) => result.id,
  silentNotFound: true,
  logLabel: "Tax clause search failed:",
  trackHttpError: (status: number, statusText: string) => {
    trackEvent("api_error", {
      endpoint: "api/tax-clauses",
      status,
      status_text: statusText,
    });
  },
  trackFailure: (error: unknown) => {
    trackEvent("api_error", {
      endpoint: "api/tax-clauses",
      kind:
        error instanceof TypeError && error.message.includes("fetch")
          ? "network"
          : "unknown",
    });
  },
};

/**
 * Declarative tax-clause search.
 *
 * `filters` is the draft (what the sidebar is editing). Results come from a
 * `useQuery` driven by `committed` — the snapshot last submitted via
 * `performSearch`, `goToPage`, etc. Editing a sidebar filter does NOT refetch;
 * only commit actions do.
 */
export function useTaxClauses() {
  const queryClient = useQueryClient();
  const [filters, setFilters] = useState<TaxClauseFilters>({ ...EMPTY_FILTERS });
  const [currentSort, setCurrentSort] = useState<SortField | null>("year");
  const [sort_direction, setSortDirection] = useState<SortDirection>("desc");
  const fireResultsLoadedRef = useRef(false);

  const core = useCommittedSearchCore<
    TaxClauseFilters,
    TaxClauseSearchResponse,
    TaxClauseSearchResult
  >(TAX_CLAUSES_SEARCH_CONFIG);
  const { committed, responseData } = core;

  const searchResults = core.results;
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
    trackEvent("tax_clauses_results_loaded", {
      total_count: responseData.total_count,
      total_pages: responseData.total_pages,
      page: committed.filters.page,
      page_size: committed.filters.page_size,
    });
  }, [responseData, committed]);

  // Mirror committed page/page_size back into draft so the UI (pagination
  // controls, URL sync, downloadCSV) reflects what we actually fetched.
  useEffect(() => {
    if (!responseData) return;
    setFilters((prev) =>
      prev.page === responseData.page && prev.page_size === responseData.page_size
        ? prev
        : { ...prev, page: responseData.page, page_size: responseData.page_size },
    );
  }, [responseData]);

  const updateFilter = useCallback(
    (field: keyof TaxClauseFilters, value: string | string[] | number | boolean) => {
      setFilters((prev) => ({ ...prev, [field]: value } as TaxClauseFilters));
    },
    [],
  );

  const hydrateFilters = useCallback((next: Partial<TaxClauseFilters>) => {
    setFilters((prev) => ({
      ...prev,
      ...next,
      page: next.page ?? prev.page,
      page_size: next.page_size ?? prev.page_size,
    }));
  }, []);

  const toggleFilterValue = useCallback(
    (field: keyof TaxClauseFilters, value: string) => {
      if (field === "page" || field === "page_size" || field === "include_rep_warranty")
        return;
      setFilters((prev) => {
        const current = (prev[field] as string[] | undefined) ?? [];
        const nextVals = current.includes(value)
          ? current.filter((v) => v !== value)
          : [...current, value];
        return { ...prev, [field]: nextVals };
      });
    },
    [],
  );

  const setIncludeRepWarranty = useCallback((value: boolean) => {
    setFilters((prev) => ({ ...prev, include_rep_warranty: value }));
  }, []);

  const performSearch = useCallback(
    async (
      resetPage = false,
      markAsSearched: boolean = resetPage,
      filtersOverride?: TaxClauseFilters,
      overrideSortBy?: SortField | null,
      overrideSortDirection?: SortDirection,
    ) => {
      const base = filtersOverride ?? filters;
      const effective: TaxClauseFilters = resetPage ? { ...base, page: 1 } : base;
      if (resetPage && !filtersOverride) {
        setFilters((prev) => ({ ...prev, page: 1 }));
      }
      const sortBy = overrideSortBy ?? currentSort;
      const sortDir = overrideSortDirection ?? sort_direction;

      if (markAsSearched) {
        fireResultsLoadedRef.current = true;
        trackEvent("tax_clauses_performed", {
          years_count: effective.year?.length ?? 0,
          targets_count: effective.target?.length ?? 0,
          acquirers_count: effective.acquirer?.length ?? 0,
          tax_standard_ids_count: effective.clauseType?.length ?? 0,
          include_rep_warranty: !!effective.include_rep_warranty,
          page: effective.page,
          page_size: effective.page_size,
          sort_by: sortBy ?? "none",
          sort_direction: sortDir,
        });
      }

      core.commit({
        filters: effective,
        sortBy,
        sortDirection: sortDir,
      });
    },
    [filters, currentSort, sort_direction, core.commit],
  );

  const downloadCSV = useCallback(async () => {
    let rows: TaxClauseSearchResult[] = [];
    if (core.selectedResults.size > 0) {
      rows = searchResults.filter((r) => core.selectedResults.has(r.id));
    } else {
      try {
        const params = buildTaxClauseParams(
          { ...filters, page: undefined, page_size: undefined },
          false,
        );
        params.append("page_size", LARGE_PAGE_SIZE_FOR_CSV.toString());
        params.append("page", DEFAULT_PAGE.toString());
        const queryString = params.toString();
        const body = await queryClient.fetchQuery({
          queryKey: keys.taxClauses.search({ q: queryString, csv: true }),
          queryFn: async () => {
            const res = await authFetch(apiUrl(`v1/tax-clauses?${queryString}`));
            if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
            return (await res.json()) as TaxClauseSearchResponse;
          },
          staleTime: 60 * 1000,
        });
        rows = body.results;
      } catch (error) {
        logger.error("Failed to fetch tax clauses for CSV:", error);
        rows = searchResults;
      }
    }

    if (rows.length === 0) return;

    trackEvent("tax_clauses_csv_download_click", {
      mode: core.selectedResults.size > 0 ? "selected" : "all",
      selected_count: core.selectedResults.size,
      downloaded_count: rows.length,
    });

    const headers = [
      "Year",
      "Target",
      "Acquirer",
      "Context",
      "Tax Clause Types",
      "Clause Text",
      "Clause UUID",
      "Section UUID",
      "Agreement UUID",
    ];
    const csv = [
      headers.join(","),
      ...rows.map((r) =>
        [
          r.year ?? "",
          `"${(r.target ?? "").replace(/"/g, '""')}"`,
          `"${(r.acquirer ?? "").replace(/"/g, '""')}"`,
          r.context_type,
          `"${r.tax_standard_ids.join("; ")}"`,
          `"${(r.clause_text ?? "").replace(/"/g, '""')}"`,
          r.clause_uuid,
          r.section_uuid,
          r.agreement_uuid,
        ].join(","),
      ),
    ].join("\n");

    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    const objectUrl = URL.createObjectURL(blob);
    link.href = objectUrl;
    const suffix = core.selectedResults.size > 0 ? "_selected" : "";
    link.download = `ma_tax_clauses${suffix}_${new Date().toISOString().split("T")[0]}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
  }, [filters, searchResults, core.selectedResults, queryClient]);

  const clearFilters = useCallback(() => {
    setFilters({ ...EMPTY_FILTERS });
    core.uncommit();
  }, [core.uncommit]);

  const goToPage = useCallback(
    async (page: number) => {
      const next: TaxClauseFilters = { ...filters, page };
      setFilters(next);
      await performSearch(false, false, next);
    },
    [filters, performSearch],
  );

  const changePageSize = useCallback(
    async (page_size: number) => {
      const next: TaxClauseFilters = { ...filters, page_size, page: 1 };
      setFilters(next);
      await performSearch(false, false, next);
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
    currentPage: filters.page || DEFAULT_PAGE,
    page_size: filters.page_size || DEFAULT_PAGE_SIZE,
    showErrorModal: core.showErrorModal,
    errorMessage: core.errorMessage,
    sort_direction,
    actions: {
      updateFilter,
      hydrateFilters,
      toggleFilterValue,
      setIncludeRepWarranty,
      performSearch,
      downloadCSV,
      clearFilters,
      goToPage,
      changePageSize,
      closeErrorModal: core.closeErrorModal,
      sortResults,
      toggleSortDirection,
      setSortDirection,
      toggleResultSelection: core.toggleResultSelection,
      toggleSelectAll: core.toggleSelectAll,
      clearSelection: core.clearSelection,
    },
  };
}
