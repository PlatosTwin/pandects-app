import { useCallback, useEffect, useRef, useState } from "react";
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

function buildTaxClauseParams(
  filters: TaxClauseFilters,
  includePagination = true,
): URLSearchParams {
  const params = buildSearchParams(filters, undefined, includePagination);
  // `clauseType` in the shared filter shape stands in for taxonomy IDs; for tax
  // search, we send them under `tax_standard_id` and drop the sections key.
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

type SortField = "year" | "target" | "acquirer";

export function useTaxClauses() {
  const [filters, setFilters] = useState<TaxClauseFilters>({ ...EMPTY_FILTERS });
  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<TaxClauseSearchResult[]>([]);
  const [selectedResults, setSelectedResults] = useState<Set<string>>(new Set());
  const [hasSearched, setHasSearched] = useState(false);
  const [total_count, setTotalCount] = useState(0);
  const [totalCountIsApproximate, setTotalCountIsApproximate] = useState(false);
  const [total_pages, setTotalPages] = useState(0);
  const [has_next, setHasNext] = useState(false);
  const [has_prev, setHasPrev] = useState(false);
  const [next_num, setNextNum] = useState<number | null>(null);
  const [prev_num, setPrevNum] = useState<number | null>(null);
  const [showErrorModal, setShowErrorModal] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [sort_direction, setSortDirection] = useState<"asc" | "desc">("desc");
  const [currentSort, setCurrentSort] = useState<SortField | null>("year");
  const [access, setAccess] = useState<TaxClauseSearchResponse["access"]>({
    tier: "anonymous",
  });
  const searchResultsRef = useRef<TaxClauseSearchResult[]>([]);

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
      setIsSearching(true);
      setShowErrorModal(false);
      setSelectedResults(new Set());
      if (markAsSearched) setHasSearched(true);

      try {
        const base = filtersOverride ?? filters;
        const effective = resetPage ? { ...base, page: 1 } : base;
        if (resetPage && !filtersOverride) {
          setFilters((prev) => ({ ...prev, page: 1 }));
        }

        const params = buildTaxClauseParams(effective);
        const sortBy = overrideSortBy ?? currentSort;
        const sortDir = overrideSortDirection ?? sort_direction;
        if (sortBy) {
          params.append("sort_by", sortBy);
          params.append("sort_direction", sortDir);
        }

        if (markAsSearched) {
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

        const res = await authFetch(apiUrl(`v1/tax-clauses?${params.toString()}`));
        if (!res.ok) {
          if (res.status === 404) return;
          trackEvent("api_error", {
            endpoint: "api/tax-clauses",
            status: res.status,
            status_text: res.statusText,
          });
          throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }

        const body = (await res.json()) as TaxClauseSearchResponse;
        if (!filtersOverride) {
          setFilters((prev) => ({
            ...prev,
            page: body.page,
            page_size: body.page_size,
          }));
        }
        setSearchResults(body.results);
        searchResultsRef.current = body.results;
        setAccess(body.access);
        setTotalCount(body.total_count);
        setTotalCountIsApproximate(body.total_count_is_approximate);
        setTotalPages(body.total_pages);
        setHasNext(body.has_next);
        setHasPrev(body.has_prev);
        setNextNum(body.next_num);
        setPrevNum(body.prev_num);

        if (markAsSearched) {
          trackEvent("tax_clauses_results_loaded", {
            total_count: body.total_count,
            total_pages: body.total_pages,
            page: effective.page,
            page_size: effective.page_size,
          });
        }
      } catch (error) {
        logger.error("Tax clause search failed:", error);
        setSearchResults([]);
        searchResultsRef.current = [];
        setTotalCount(0);
        setTotalCountIsApproximate(false);
        setTotalPages(0);
        trackEvent("api_error", {
          endpoint: "api/tax-clauses",
          kind:
            error instanceof TypeError && error.message.includes("fetch")
              ? "network"
              : "unknown",
        });
        setErrorMessage(
          "Network error: unable to reach the back end database. Check your connection and try again.",
        );
        setShowErrorModal(true);
      } finally {
        setIsSearching(false);
      }
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
        const res = await authFetch(apiUrl(`v1/tax-clauses?${params.toString()}`));
        if (!res.ok) throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        const body = (await res.json()) as TaxClauseSearchResponse;
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
  }, [filters, searchResults, selectedResults]);

  const clearFilters = useCallback(() => {
    setFilters({ ...EMPTY_FILTERS });
    setSearchResults([]);
    searchResultsRef.current = [];
    setSelectedResults(new Set());
    setHasSearched(false);
    setTotalCount(0);
    setTotalCountIsApproximate(false);
    setTotalPages(0);
    setHasNext(false);
    setHasPrev(false);
    setNextNum(null);
    setPrevNum(null);
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

  useEffect(() => {
    searchResultsRef.current = searchResults;
  }, [searchResults]);

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
    next_num,
    prev_num,
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
