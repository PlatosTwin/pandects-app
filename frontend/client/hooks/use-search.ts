import { useState, useCallback, useEffect } from "react";
import { SearchFilters, SearchResult, SearchResponse } from "@shared/search";
import { apiUrl } from "@/lib/api-config";
import { buildSearchParams, extractStandardIds } from "@/lib/url-params";
import {
  DEFAULT_PAGE_SIZE,
  DEFAULT_PAGE,
  LARGE_PAGE_SIZE_FOR_CSV,
} from "@/lib/constants";

export function useSearch() {
  const [filters, setFilters] = useState<SearchFilters>({
    year: [],
    target: [],
    acquirer: [],
    clauseType: [],
    standardId: [],
    transactionSize: [],
    transactionType: [],
    considerationType: [],
    targetType: [],
    page: DEFAULT_PAGE,
    pageSize: DEFAULT_PAGE_SIZE,
  });

  // Helper function to sort results
  const sortResultsArray = (
    results: SearchResult[],
    sortBy: "year" | "target" | "acquirer" | null,
    direction: "asc" | "desc",
  ): SearchResult[] => {
    if (!sortBy) return results;

    return [...results].sort((a, b) => {
      let comparison = 0;
      switch (sortBy) {
        case "year":
          comparison = parseInt(a.year) - parseInt(b.year);
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
  };

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
  const [showNoResultsModal, setShowNoResultsModal] = useState(false);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("desc");
  const [currentSort, setCurrentSort] = useState<
    "year" | "target" | "acquirer" | null
  >("year");
  // Pagination metadata from API
  const [hasNext, setHasNext] = useState(false);
  const [hasPrev, setHasPrev] = useState(false);
  const [nextNum, setNextNum] = useState<number | null>(null);
  const [prevNum, setPrevNum] = useState<number | null>(null);

  const updateFilter = useCallback(
    (field: keyof SearchFilters, value: string | string[]) => {
      if (field === "page" || field === "pageSize") {
        setFilters((prev) => ({ ...prev, [field]: value as number }));
      } else {
        setFilters((prev) => ({ ...prev, [field]: value as string[] }));
      }
    },
    [],
  );

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

  const performSearch = useCallback(
    async (resetPage = false, clauseTypesNested?: any) => {
      setIsSearching(true);
      setShowErrorModal(false);
      setShowNoResultsModal(false);
      setSelectedResults(new Set()); // Clear selected results when performing new search

      if (resetPage) {
        setHasSearched(true);
      }

      try {
        const searchFilters = resetPage ? { ...filters, page: 1 } : filters;
        if (resetPage) {
          setFilters((prev) => ({ ...prev, page: 1 }));
        }

        const params = buildSearchParams(searchFilters, clauseTypesNested);

        const queryString = params.toString();
        const res = await fetch(apiUrl(`api/search?${queryString}`));

        // Check if the response is ok (status 200-299)
        if (!res.ok) {
          if (res.status === 404) {
            return;
          }
          // Other HTTP errors
          throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }

        // Parse as SearchResponse with pagination metadata
        const searchResponse = (await res.json()) as SearchResponse;

        // Apply client-side sorting to the current page results
        const sortedResults = sortResultsArray(
          searchResponse.results,
          currentSort,
          sortDirection,
        );

        setSearchResults(sortedResults);
        setTotalCount(searchResponse.totalCount);
        setTotalPages(searchResponse.totalPages);
        setHasNext(searchResponse.hasNext);
        setHasPrev(searchResponse.hasPrev);
        setNextNum(searchResponse.nextNum);
        setPrevNum(searchResponse.prevNum);

        // Check if no results found with active filters
        if (
          searchResponse.totalCount === 0 &&
          hasFiltersApplied(searchFilters)
        ) {
          setShowNoResultsModal(true);
        }
      } catch (error) {
        console.error("Search failed:", error);
        setSearchResults([]);
        setTotalCount(0);
        setTotalPages(0);

        // Check if it's a network error
        if (error instanceof TypeError && error.message.includes("fetch")) {
          setErrorMessage(
            "Network error: unable to reach the back end database. Check your connection and try again.",
          );
          setShowErrorModal(true);
        } else {
          setErrorMessage(
            "Network error: unable to reach the back end database. Check your connection and try again.",
          );
          setShowErrorModal(true);
        }
      } finally {
        setIsSearching(false);
      }
    },
    [filters, currentSort, sortDirection],
  );

  // Helper function to check if any filters are applied
  const hasFiltersApplied = (searchFilters: SearchFilters) => {
    return !!(
      (searchFilters.year && searchFilters.year.length > 0) ||
      (searchFilters.target && searchFilters.target.length > 0) ||
      (searchFilters.acquirer && searchFilters.acquirer.length > 0) ||
      (searchFilters.clauseType && searchFilters.clauseType.length > 0) ||
      (searchFilters.transactionSize &&
        searchFilters.transactionSize.length > 0) ||
      (searchFilters.transactionType &&
        searchFilters.transactionType.length > 0) ||
      (searchFilters.considerationType &&
        searchFilters.considerationType.length > 0) ||
      (searchFilters.targetType && searchFilters.targetType.length > 0)
    );
  };

  const downloadCSV = useCallback(
    async (clauseTypesNested?: any) => {
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
          const params = new URLSearchParams();

          // Build the same filter parameters as in performSearch
          if (searchFilters.year && searchFilters.year.length > 0) {
            searchFilters.year.forEach((year) => params.append("year", year));
          }
          if (searchFilters.target && searchFilters.target.length > 0) {
            searchFilters.target.forEach((target) =>
              params.append("target", target),
            );
          }
          if (searchFilters.acquirer && searchFilters.acquirer.length > 0) {
            searchFilters.acquirer.forEach((acquirer) =>
              params.append("acquirer", acquirer),
            );
          }
          if (
            searchFilters.transactionSize &&
            searchFilters.transactionSize.length > 0
          ) {
            searchFilters.transactionSize.forEach((size) =>
              params.append("transactionSize", size),
            );
          }
          if (
            searchFilters.transactionType &&
            searchFilters.transactionType.length > 0
          ) {
            searchFilters.transactionType.forEach((type) =>
              params.append("transactionType", type),
            );
          }
          if (
            searchFilters.considerationType &&
            searchFilters.considerationType.length > 0
          ) {
            searchFilters.considerationType.forEach((type) =>
              params.append("considerationType", type),
            );
          }
          if (searchFilters.targetType && searchFilters.targetType.length > 0) {
            searchFilters.targetType.forEach((type) =>
              params.append("targetType", type),
            );
          }

          // Extract standard IDs from selected clause types and send them instead
          if (
            searchFilters.clauseType &&
            searchFilters.clauseType.length > 0 &&
            clauseTypesNested
          ) {
            const standardIds = extractStandardIds(
              searchFilters.clauseType,
              clauseTypesNested,
            );
            standardIds.forEach((standardId) =>
              params.append("standardId", standardId),
            );
          }

          // Set a very large page size to get all results
          params.append("pageSize", "10000");
          params.append("page", "1");

          const queryString = params.toString();
          const res = await fetch(apiUrl(`api/search?${queryString}`));

          if (!res.ok) {
            throw new Error(`HTTP ${res.status}: ${res.statusText}`);
          }

          const searchResponse = (await res.json()) as SearchResponse;
          resultsToDownload = searchResponse.results;
        } catch (error) {
          console.error("Failed to fetch all results for CSV:", error);
          // Fallback to current page results
          resultsToDownload = searchResults;
        }
      }

      if (resultsToDownload.length === 0) return;

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
            `"${result.xml.replace(/"/g, '""')}"`,
            result.sectionUuid,
            result.agreementUuid,
          ].join(","),
        ),
      ].join("\n");

      // Create and download file
      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      const selectedText = selectedResults.size > 0 ? "_selected" : "";
      link.download = `ma_clauses${selectedText}_${new Date().toISOString().split("T")[0]}.csv`;
      link.click();
      URL.revokeObjectURL(link.href);
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
      transactionSize: [],
      transactionType: [],
      considerationType: [],
      targetType: [],
      page: DEFAULT_PAGE,
      pageSize: DEFAULT_PAGE_SIZE,
    });
    setSearchResults([]);
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
    async (page: number, clauseTypesNested?: any) => {
      setFilters((prev) => ({ ...prev, page }));

      // Trigger a new search with the new page number
      await performSearch(false, clauseTypesNested);
    },
    [performSearch],
  );

  const changePageSize = useCallback(
    async (pageSize: number, clauseTypesNested?: any) => {
      setFilters((prev) => ({ ...prev, pageSize, page: 1 }));

      // Trigger a new search with the new page size and reset to page 1
      await performSearch(false, clauseTypesNested);
    },
    [performSearch],
  );

  const closeErrorModal = useCallback(() => {
    setShowErrorModal(false);
    setErrorMessage("");
  }, []);

  const closeNoResultsModal = useCallback(() => {
    setShowNoResultsModal(false);
  }, []);

  const sortResults = useCallback(
    (sortBy: "year" | "target" | "acquirer") => {
      setCurrentSort(sortBy);
      setSearchResults((prev) => sortResultsArray(prev, sortBy, sortDirection));
    },
    [sortDirection, sortResultsArray],
  );

  const toggleSortDirection = useCallback(() => {
    setSortDirection((prev) => (prev === "asc" ? "desc" : "asc"));
  }, []);

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

  // Auto-refresh results when sort direction changes
  useEffect(() => {
    if (currentSort && searchResults.length > 0) {
      setSearchResults((prev) =>
        sortResultsArray(prev, currentSort, sortDirection),
      );
    }
  }, [sortDirection, currentSort]);

  return {
    filters,
    isSearching,
    searchResults,
    selectedResults,
    hasSearched,
    totalCount,
    totalPages,
    currentPage: filters.page || 1,
    pageSize: filters.pageSize || 25,
    showErrorModal,
    errorMessage,
    showNoResultsModal,
    sortDirection,
    actions: {
      updateFilter,
      toggleFilterValue,
      performSearch,
      downloadCSV,
      clearFilters,
      goToPage,
      changePageSize,
      closeErrorModal,
      closeNoResultsModal,
      sortResults,
      toggleSortDirection,
      toggleResultSelection,
      toggleSelectAll,
    },
  };
}
