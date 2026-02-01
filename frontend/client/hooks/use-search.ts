import { useState, useCallback, useEffect, useRef } from "react";
import { SearchFilters, SearchResult, SearchResponse } from "@shared/search";
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

/** Search filters, results, pagination, and actions (performSearch, downloadCSV, etc.). */
export function useSearch() {
  const [filters, setFilters] = useState<SearchFilters>({
    year: [],
    target: [],
    acquirer: [],
    clauseType: [],
    standardId: [],
    transactionPriceTotal: [],
    transactionPriceStock: [],
    transactionPriceCash: [],
    transactionPriceAssets: [],
    transactionConsideration: [],
    targetType: [],
    acquirerType: [],
    targetIndustry: [],
    acquirerIndustry: [],
    dealStatus: [],
    attitude: [],
    dealType: [],
    purpose: [],
    targetPe: [],
    acquirerPe: [],
    page: DEFAULT_PAGE,
    pageSize: DEFAULT_PAGE_SIZE,
  });

  // Helper function to sort results
  const sortResultsArray = useCallback((
    results: SearchResult[],
    sortBy: "year" | "target" | "acquirer" | null,
    direction: "asc" | "desc",
  ): SearchResult[] => {
    if (!sortBy) return results;

    return [...results].sort((a, b) => {
      let comparison = 0;
      switch (sortBy) {
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
  }, []);

  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [selectedResults, setSelectedResults] = useState<Set<string>>(
    new Set(),
  );
  const [hasSearched, setHasSearched] = useState(false);
  const [totalCount, setTotalCount] = useState(0);
  const [totalPages, setTotalPages] = useState(0);
  const [showErrorModal, setShowErrorModal] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("desc");
  const [currentSort, setCurrentSort] = useState<
    "year" | "target" | "acquirer" | null
  >("year");
  const searchResultsRef = useRef<SearchResult[]>([]);
  const lastSortRef = useRef<{ sortBy: typeof currentSort; direction: "asc" | "desc" }>({
    sortBy: "year",
    direction: "desc",
  });
  // Pagination metadata from API
  const [hasNext, setHasNext] = useState(false);
  const [hasPrev, setHasPrev] = useState(false);
  const [nextNum, setNextNum] = useState<number | null>(null);
  const [prevNum, setPrevNum] = useState<number | null>(null);
  const [access, setAccess] = useState<SearchResponse["access"]>({
    tier: "anonymous",
  });

  const updateFilter = useCallback(
    (field: keyof SearchFilters, value: string | string[] | number) => {
      if (field === "page" || field === "pageSize") {
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
      pageSize: next.pageSize ?? prev.pageSize,
    }));
  }, []);

  const toggleFilterValue = useCallback(
    (field: keyof SearchFilters, value: string) => {
      if (field === "page" || field === "pageSize") return;

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
      if (field === "page" || field === "pageSize") return;

      const trimmedValue = value.trim();
      setFilters((prev) => ({
        ...prev,
        [field]: trimmedValue || undefined,
      }));
    },
    [],
  );

  const setSortDirectionDirect = useCallback((direction: "asc" | "desc") => {
    setSortDirection(direction);
  }, []);

  const performSearch = useCallback(
    async (
      resetPage = false,
      clauseTypesNested?: ClauseTypeTree,
      markAsSearched: boolean = resetPage,
      filtersOverride?: SearchFilters,
      overrideSortBy?: typeof currentSort,
      overrideSortDirection?: typeof sortDirection,
    ) => {
      setIsSearching(true);
      setShowErrorModal(false);
      setSelectedResults(new Set()); // Clear selected results when performing new search

      if (markAsSearched) {
        setHasSearched(true);
      }

      try {
        const baseFilters = filtersOverride ?? filters;
        const searchFilters = resetPage
          ? { ...baseFilters, page: 1 }
          : baseFilters;

        if (resetPage && !filtersOverride) {
          setFilters((prev) => ({ ...prev, page: 1 }));
        }

        if (markAsSearched) {
          trackEvent("search_performed", {
            years_count: searchFilters.year?.length ?? 0,
            targets_count: searchFilters.target?.length ?? 0,
            acquirers_count: searchFilters.acquirer?.length ?? 0,
            clause_types_count: searchFilters.clauseType?.length ?? 0,
            standard_ids_count: searchFilters.standardId?.length ?? 0,
            page: searchFilters.page,
            page_size: searchFilters.pageSize,
            sort_by: currentSort ?? "none",
            sort_direction: sortDirection,
          });
        }

        const params = buildSearchParams(searchFilters, clauseTypesNested);
        // Add sort parameters to the API request
        const effectiveSortBy = overrideSortBy ?? currentSort;
        const effectiveSortDirection = overrideSortDirection ?? sortDirection;
        if (effectiveSortBy) {
          params.append("sortBy", effectiveSortBy);
          params.append("sortDirection", effectiveSortDirection);
        }

        const queryString = params.toString();
        const res = await authFetch(apiUrl(`v1/search?${queryString}`));

        // Check if the response is ok (status 200-299)
        if (!res.ok) {
          if (res.status === 404) {
            return;
          }
          trackEvent("api_error", {
            endpoint: "api/search",
            status: res.status,
            status_text: res.statusText,
          });
          // Other HTTP errors
          throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }

        // Parse as SearchResponse with pagination metadata
        const searchResponse = (await res.json()) as SearchResponse;

        if (!filtersOverride) {
          setFilters((prev) => ({
            ...prev,
            page: searchResponse.page,
            pageSize: searchResponse.pageSize,
          }));
        }

        // Apply client-side sorting to the current page results
        const sortedResults = sortResultsArray(
          searchResponse.results,
          currentSort ?? "year",
          sortDirection,
        );

        setSearchResults(sortedResults);
        // Update the ref to track current results
        searchResultsRef.current = sortedResults;
        setAccess(searchResponse.access);
        setTotalCount(searchResponse.totalCount);
        setTotalPages(searchResponse.totalPages);
        setHasNext(searchResponse.hasNext);
        setHasPrev(searchResponse.hasPrev);
        setNextNum(searchResponse.nextNum);
        setPrevNum(searchResponse.prevNum);

        if (markAsSearched) {
          trackEvent("search_results_loaded", {
            total_count: searchResponse.totalCount,
            total_pages: searchResponse.totalPages,
            page: searchFilters.page,
            page_size: searchFilters.pageSize,
          });
        }
      } catch (error) {
        logger.error("Search failed:", error);
        setSearchResults([]);
        searchResultsRef.current = [];
        setTotalCount(0);
        setTotalPages(0);

        // Check if it's a network error
        if (error instanceof TypeError && error.message.includes("fetch")) {
          trackEvent("api_error", {
            endpoint: "api/search",
            kind: "network",
          });
          setErrorMessage(
            "Network error: unable to reach the back end database. Check your connection and try again.",
          );
          setShowErrorModal(true);
        } else {
          trackEvent("api_error", {
            endpoint: "api/search",
            kind: "unknown",
          });
          setErrorMessage(
            "Network error: unable to reach the back end database. Check your connection and try again.",
          );
          setShowErrorModal(true);
        }
      } finally {
        setIsSearching(false);
      }
    },
    [filters, currentSort, sortDirection, sortResultsArray],
  );

  // Helper function to check if any filters are applied
  const hasFiltersApplied = (searchFilters: SearchFilters) => {
    return !!(
      (searchFilters.year && searchFilters.year.length > 0) ||
      (searchFilters.target && searchFilters.target.length > 0) ||
      (searchFilters.acquirer && searchFilters.acquirer.length > 0) ||
      (searchFilters.clauseType && searchFilters.clauseType.length > 0) ||
      (searchFilters.transactionPriceTotal &&
        searchFilters.transactionPriceTotal.length > 0) ||
      (searchFilters.transactionPriceStock &&
        searchFilters.transactionPriceStock.length > 0) ||
      (searchFilters.transactionPriceCash &&
        searchFilters.transactionPriceCash.length > 0) ||
      (searchFilters.transactionPriceAssets &&
        searchFilters.transactionPriceAssets.length > 0) ||
      (searchFilters.transactionConsideration &&
        searchFilters.transactionConsideration.length > 0) ||
      (searchFilters.targetType && searchFilters.targetType.length > 0) ||
      (searchFilters.acquirerType && searchFilters.acquirerType.length > 0) ||
      (searchFilters.targetIndustry && searchFilters.targetIndustry.length > 0) ||
      (searchFilters.acquirerIndustry &&
        searchFilters.acquirerIndustry.length > 0) ||
      (searchFilters.dealStatus && searchFilters.dealStatus.length > 0) ||
      (searchFilters.attitude && searchFilters.attitude.length > 0) ||
      (searchFilters.dealType && searchFilters.dealType.length > 0) ||
      (searchFilters.purpose && searchFilters.purpose.length > 0) ||
      (searchFilters.targetPe && searchFilters.targetPe.length > 0) ||
      (searchFilters.acquirerPe && searchFilters.acquirerPe.length > 0)
    );
  };

  const downloadCSV = useCallback(
    async (clauseTypesNested?: ClauseTypeTree) => {
      let resultsToDownload: SearchResult[] = [];

      if (selectedResults.size > 0) {
        // If we have selected results, only download those from current page
        resultsToDownload = searchResults.filter((result) =>
          selectedResults.has(result.id),
        );
      } else {
        // If no selected results, we need to fetch all results for the current filters
        // This requires making a new API call without pagination to get all results
        try {
          const searchFilters = {
            ...filters,
            page: undefined,
            pageSize: undefined,
          };
          const params = buildSearchParams(
            searchFilters,
            clauseTypesNested,
            false,
          );

          // Set a very large page size to get all results
          params.append("pageSize", LARGE_PAGE_SIZE_FOR_CSV.toString());
          params.append("page", DEFAULT_PAGE.toString());

          const queryString = params.toString();
          const res = await fetch(apiUrl(`v1/search?${queryString}`));

          if (!res.ok) {
            trackEvent("api_error", {
              endpoint: "api/search",
              status: res.status,
              status_text: res.statusText,
            });
            throw new Error(`HTTP ${res.status}: ${res.statusText}`);
          }

          const searchResponse = (await res.json()) as SearchResponse;
          resultsToDownload = searchResponse.results;
        } catch (error) {
          logger.error("Failed to fetch all results for CSV:", error);
          trackEvent("api_error", {
            endpoint: "api/search",
            kind:
              error instanceof TypeError && error.message.includes("fetch")
                ? "network"
                : "unknown",
          });
          // Fallback to current page results
          resultsToDownload = searchResults;
        }
      }

      if (resultsToDownload.length === 0) return;

      trackEvent("search_csv_download_click", {
        mode: selectedResults.size > 0 ? "selected" : "all",
        selected_count: selectedResults.size,
        downloaded_count: resultsToDownload.length,
        years_count: filters.year?.length ?? 0,
        targets_count: filters.target?.length ?? 0,
        acquirers_count: filters.acquirer?.length ?? 0,
        clause_types_count: filters.clauseType?.length ?? 0,
        standard_ids_count: filters.standardId?.length ?? 0,
      });

      // Create CSV content
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
            `"${result.articleTitle}"`,
            `"${result.sectionTitle}"`,
            `"${(result.xml ?? "").replace(/"/g, '""')}"`,
            result.sectionUuid,
            result.agreementUuid,
          ].join(","),
        ),
      ].join("\n");

      // Create and download file
      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      const link = document.createElement("a");
      const objectUrl = URL.createObjectURL(blob);
      link.href = objectUrl;
      const selectedText = selectedResults.size > 0 ? "_selected" : "";
      link.download = `ma_clauses${selectedText}_${new Date().toISOString().split("T")[0]}.csv`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
    },
    [filters, searchResults, selectedResults],
  );

  const clearFilters = useCallback(() => {
    setFilters({
      year: [],
      target: [],
      acquirer: [],
      clauseType: [],
      standardId: [],
      transactionPriceTotal: [],
      transactionPriceStock: [],
      transactionPriceCash: [],
      transactionPriceAssets: [],
      transactionConsideration: [],
      targetType: [],
      acquirerType: [],
      targetIndustry: [],
      acquirerIndustry: [],
      dealStatus: [],
      attitude: [],
      dealType: [],
      purpose: [],
      targetPe: [],
      acquirerPe: [],
      agreementUuid: undefined,
      sectionUuid: undefined,
      page: DEFAULT_PAGE,
      pageSize: DEFAULT_PAGE_SIZE,
    });
    setSearchResults([]);
    searchResultsRef.current = [];
    setSelectedResults(new Set());
    setHasSearched(false);
    setTotalCount(0);
    setTotalPages(0);
    setHasNext(false);
    setHasPrev(false);
    setNextNum(null);
    setPrevNum(null);
  }, []);

  const goToPage = useCallback(
    async (page: number, clauseTypesNested?: ClauseTypeTree) => {
      const nextFilters: SearchFilters = { ...filters, page };
      setFilters(nextFilters);
      await performSearch(false, clauseTypesNested, false, nextFilters);
    },
    [filters, performSearch],
  );

  const changePageSize = useCallback(
    async (pageSize: number, clauseTypesNested?: ClauseTypeTree) => {
      const nextFilters: SearchFilters = { ...filters, pageSize, page: 1 };
      setFilters(nextFilters);
      await performSearch(false, clauseTypesNested, false, nextFilters);
    },
    [filters, performSearch],
  );

  const closeErrorModal = useCallback(() => {
    setShowErrorModal(false);
    setErrorMessage("");
  }, []);

  const sortResults = useCallback(
    (sortBy: "year" | "target" | "acquirer") => {
      setCurrentSort(sortBy);
      setSearchResults((prev) => {
        const sorted = sortResultsArray(prev, sortBy, sortDirection);
        searchResultsRef.current = sorted;
        return sorted;
      });
    },
    [sortDirection, sortResultsArray],
  );

  const toggleSortDirection = useCallback(() => {
    const newDirection = sortDirection === "asc" ? "desc" : "asc";
    setSortDirection(newDirection);
    // Pass the new direction directly so it's used immediately without waiting for state update
    performSearch(false, undefined, false, undefined, currentSort, newDirection);
  }, [sortDirection, currentSort, performSearch]);

  // Result selection handlers
  const toggleResultSelection = useCallback((resultId: string) => {
    setSelectedResults((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(resultId)) {
        newSet.delete(resultId);
      } else {
        newSet.add(resultId);
      }
      return newSet;
    });
  }, []);

  const toggleSelectAll = useCallback(() => {
    const allSelected = searchResults.every((result) =>
      selectedResults.has(result.id),
    );
    if (allSelected) {
      // Deselect all current page results
      setSelectedResults((prev) => {
        const newSet = new Set(prev);
        searchResults.forEach((result) => newSet.delete(result.id));
        return newSet;
      });
    } else {
      // Select all current page results
      setSelectedResults((prev) => {
        const newSet = new Set(prev);
        searchResults.forEach((result) => newSet.add(result.id));
        return newSet;
      });
    }
  }, [searchResults, selectedResults]);

  const clearSelection = useCallback(() => {
    setSelectedResults(new Set());
  }, []);

  // Keep ref in sync with state
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
    totalCount,
    totalPages,
    currentSort,
    currentPage: filters.page || 1,
    pageSize: filters.pageSize || DEFAULT_PAGE_SIZE,
    showErrorModal,
    errorMessage,
    sortDirection,
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
      closeErrorModal,
      sortResults,
      toggleSortDirection,
      setSortDirection: setSortDirectionDirect,
      toggleResultSelection,
      toggleSelectAll,
      clearSelection,
    },
  };
}
