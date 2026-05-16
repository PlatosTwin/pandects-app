import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
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
import { IS_SERVER_RENDER } from "@/lib/query-client";
import { keys } from "@/lib/query-keys";

export interface TaxClauseFilters extends SearchFilters {
  include_rep_warranty?: boolean;
}

type SortField = "year" | "target" | "acquirer";

interface CommittedQuery {
  filters: TaxClauseFilters;
  sortBy: SortField | null;
  sortDirection: "asc" | "desc";
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

function buildCommittedQueryString(committed: CommittedQuery): string {
  const params = buildTaxClauseParams(committed.filters);
  if (committed.sortBy) {
    params.append("sort_by", committed.sortBy);
    params.append("sort_direction", committed.sortDirection);
  }
  return params.toString();
}

class TaxClauseNotFoundError extends Error {
  constructor() {
    super("NOT_FOUND");
    this.name = "TaxClauseNotFoundError";
  }
}

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
  const [sort_direction, setSortDirection] = useState<"asc" | "desc">("desc");
  const [committed, setCommitted] = useState<CommittedQuery | null>(null);
  const [selectedResults, setSelectedResults] = useState<Set<string>>(new Set());
  const [showErrorModal, setShowErrorModal] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const fireResultsLoadedRef = useRef(false);

  const committedQueryString = useMemo(
    () => (committed ? buildCommittedQueryString(committed) : ""),
    [committed],
  );

  const query = useQuery<TaxClauseSearchResponse>({
    queryKey: keys.taxClauses.search({ q: committedQueryString }),
    enabled: !IS_SERVER_RENDER && committed !== null,
    staleTime: 60 * 1000,
    retry: (failureCount, error) =>
      !(error instanceof TaxClauseNotFoundError) && failureCount < 1,
    queryFn: async () => {
      const res = await authFetch(
        apiUrl(`v1/tax-clauses?${committedQueryString}`),
      );
      if (!res.ok) {
        if (res.status === 404) throw new TaxClauseNotFoundError();
        trackEvent("api_error", {
          endpoint: "api/tax-clauses",
          status: res.status,
          status_text: res.statusText,
        });
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }
      return (await res.json()) as TaxClauseSearchResponse;
    },
  });

  const responseData: TaxClauseSearchResponse | null = query.data ?? null;
  const searchResults = responseData?.results ?? [];
  const total_count = responseData?.total_count ?? 0;
  const totalCountIsApproximate = responseData?.total_count_is_approximate ?? false;
  const total_pages = responseData?.total_pages ?? 0;
  const has_next = responseData?.has_next ?? false;
  const has_prev = responseData?.has_prev ?? false;
  const access = responseData?.access ?? EMPTY_ACCESS;
  const hasSearched = committed !== null;
  const isSearching = query.isFetching;

  // Clear selection whenever a new committed query is issued.
  useEffect(() => {
    if (committed !== null) {
      setSelectedResults(new Set());
    }
  }, [committed]);

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

  // Surface real errors (non-404) via the error modal. 404 stays silent and
  // preserves prior results, matching the imperative hook's behavior.
  useEffect(() => {
    if (!query.error) return;
    if (query.error instanceof TaxClauseNotFoundError) return;
    logger.error("Tax clause search failed:", query.error);
    trackEvent("api_error", {
      endpoint: "api/tax-clauses",
      kind:
        query.error instanceof TypeError && query.error.message.includes("fetch")
          ? "network"
          : "unknown",
    });
    setErrorMessage(
      "Network error: unable to reach the back end database. Check your connection and try again.",
    );
    setShowErrorModal(true);
  }, [query.error]);

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
      overrideSortDirection?: "asc" | "desc",
    ) => {
      const base = filtersOverride ?? filters;
      const effective: TaxClauseFilters = resetPage ? { ...base, page: 1 } : base;
      if (resetPage && !filtersOverride) {
        setFilters((prev) => ({ ...prev, page: 1 }));
      }
      const sortBy = overrideSortBy ?? currentSort;
      const sortDir = overrideSortDirection ?? sort_direction;

      setShowErrorModal(false);

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

      setCommitted({ filters: effective, sortBy, sortDirection: sortDir });
    },
    [filters, currentSort, sort_direction],
  );

  const downloadCSV = useCallback(async () => {
    let rows: TaxClauseSearchResult[] = [];
    if (selectedResults.size > 0) {
      rows = searchResults.filter((r) => selectedResults.has(r.id));
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
      mode: selectedResults.size > 0 ? "selected" : "all",
      selected_count: selectedResults.size,
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
    const suffix = selectedResults.size > 0 ? "_selected" : "";
    link.download = `ma_tax_clauses${suffix}_${new Date().toISOString().split("T")[0]}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
  }, [filters, searchResults, selectedResults, queryClient]);

  const clearFilters = useCallback(() => {
    setFilters({ ...EMPTY_FILTERS });
    setSelectedResults(new Set());
    setCommitted(null);
  }, []);

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

  const closeErrorModal = useCallback(() => {
    setShowErrorModal(false);
    setErrorMessage("");
  }, []);

  const sortResults = useCallback((sort_by: SortField) => {
    setCurrentSort(sort_by);
  }, []);

  const toggleSortDirection = useCallback(() => {
    setSortDirection((d) => (d === "asc" ? "desc" : "asc"));
  }, []);

  const toggleResultSelection = useCallback((id: string) => {
    setSelectedResults((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const toggleSelectAll = useCallback(() => {
    const allSelected = searchResults.every((r) => selectedResults.has(r.id));
    setSelectedResults((prev) => {
      const next = new Set(prev);
      if (allSelected) {
        searchResults.forEach((r) => next.delete(r.id));
      } else {
        searchResults.forEach((r) => next.add(r.id));
      }
      return next;
    });
  }, [searchResults, selectedResults]);

  const clearSelection = useCallback(() => setSelectedResults(new Set()), []);

  return {
    filters,
    isSearching,
    searchResults,
    selectedResults,
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
    showErrorModal,
    errorMessage,
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
      closeErrorModal,
      sortResults,
      toggleSortDirection,
      setSortDirection,
      toggleResultSelection,
      toggleSelectAll,
      clearSelection,
    },
  };
}
