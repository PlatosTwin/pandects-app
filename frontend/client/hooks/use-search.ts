import { useState, useCallback } from "react";
import { SearchFilters, SearchResult, SearchResponse } from "@shared/search";

export function useSearch() {
  const [filters, setFilters] = useState<SearchFilters>({
    year: [],
    target: [],
    acquirer: [],
    clauseType: [],
    page: 1,
    pageSize: 25,
  });

  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [allResults, setAllResults] = useState<SearchResult[]>([]);
  const [hasSearched, setHasSearched] = useState(false);
  const [totalCount, setTotalCount] = useState(0);
  const [totalPages, setTotalPages] = useState(0);
  const [showErrorModal, setShowErrorModal] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [showNoResultsModal, setShowNoResultsModal] = useState(false);

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
    async (resetPage = false) => {
      setIsSearching(true);
      setShowErrorModal(false);
      setShowNoResultsModal(false);

      if (resetPage) {
        setHasSearched(true);
      }

      try {
        const searchFilters = resetPage ? { ...filters, page: 1 } : filters;
        if (resetPage) {
          setFilters((prev) => ({ ...prev, page: 1 }));
        }

        const params = new URLSearchParams();

        // Handle array filters - append each value separately
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
        if (searchFilters.clauseType && searchFilters.clauseType.length > 0) {
          searchFilters.clauseType.forEach((clauseType) =>
            params.append("clauseType", clauseType),
          );
        }

        // Handle pagination
        if (searchFilters.page)
          params.append("page", searchFilters.page.toString());
        if (searchFilters.pageSize)
          params.append("pageSize", searchFilters.pageSize.toString());

        const queryString = params.toString();
        const res = await fetch(
          `http://127.0.0.1:5000/api/search?${queryString}`,
        );

        // Check if the response is ok (status 200-299)
        if (!res.ok) {
          if (res.status === 404) {
            return;
          }
          // Other HTTP errors
          throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }

        // Try to parse as SearchResponse first, fallback to SearchResult[] for backward compatibility
        const responseData = await res.json();

        if (Array.isArray(responseData)) {
          // Backward compatibility: API returns SearchResult[]
          setAllResults(responseData);
          setTotalCount(responseData.length);
          setTotalPages(
            Math.ceil(responseData.length / searchFilters.pageSize!),
          );

          // Apply pagination on frontend
          const startIndex =
            (searchFilters.page! - 1) * searchFilters.pageSize!;
          const endIndex = startIndex + searchFilters.pageSize!;
          const paginatedResults = responseData.slice(startIndex, endIndex);
          setSearchResults(paginatedResults);

          // Check if no results found with active filters
          if (responseData.length === 0 && hasFiltersApplied(searchFilters)) {
            setShowNoResultsModal(true);
          }
        } else {
          // New format: API returns SearchResponse
          const searchResponse = responseData as SearchResponse;
          setAllResults(searchResponse.results);
          setSearchResults(searchResponse.results);
          setTotalCount(searchResponse.totalCount);
          setTotalPages(searchResponse.totalPages);

          // Check if no results found with active filters
          if (
            searchResponse.totalCount === 0 &&
            hasFiltersApplied(searchFilters)
          ) {
            setShowNoResultsModal(true);
          }
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
    [filters],
  );

  // Helper function to check if any filters are applied
  const hasFiltersApplied = (searchFilters: SearchFilters) => {
    return !!(
      (searchFilters.year && searchFilters.year.length > 0) ||
      (searchFilters.target && searchFilters.target.length > 0) ||
      (searchFilters.acquirer && searchFilters.acquirer.length > 0) ||
      (searchFilters.clauseType && searchFilters.clauseType.length > 0)
    );
  };

  const downloadCSV = useCallback(() => {
    if (searchResults.length === 0) return;

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
      ...searchResults.map((result) =>
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
    link.download = `ma_clauses_${new Date().toISOString().split("T")[0]}.csv`;
    link.click();
    URL.revokeObjectURL(link.href);
  }, [searchResults]);

  const clearFilters = useCallback(() => {
    setFilters({
      year: [],
      target: [],
      acquirer: [],
      clauseType: [],
      page: 1,
      pageSize: 25,
    });
    setSearchResults([]);
    setAllResults([]);
    setHasSearched(false);
    setTotalCount(0);
    setTotalPages(0);
  }, []);

  const goToPage = useCallback(
    (page: number) => {
      setFilters((prev) => ({ ...prev, page }));

      // Apply pagination on frontend if we have all results
      if (allResults.length > 0) {
        const startIndex = (page - 1) * (filters.pageSize || 25);
        const endIndex = startIndex + (filters.pageSize || 25);
        const paginatedResults = allResults.slice(startIndex, endIndex);
        setSearchResults(paginatedResults);
      }
    },
    [allResults, filters.pageSize],
  );

  const changePageSize = useCallback(
    (pageSize: number) => {
      setFilters((prev) => ({ ...prev, pageSize, page: 1 }));

      // Apply new page size on frontend if we have all results
      if (allResults.length > 0) {
        setTotalPages(Math.ceil(allResults.length / pageSize));
        const paginatedResults = allResults.slice(0, pageSize);
        setSearchResults(paginatedResults);
      }
    },
    [allResults],
  );

  const closeErrorModal = useCallback(() => {
    setShowErrorModal(false);
    setErrorMessage("");
  }, []);

  const closeNoResultsModal = useCallback(() => {
    setShowNoResultsModal(false);
  }, []);

  return {
    filters,
    isSearching,
    searchResults,
    hasSearched,
    totalCount,
    totalPages,
    currentPage: filters.page || 1,
    pageSize: filters.pageSize || 25,
    showErrorModal,
    errorMessage,
    showNoResultsModal,
    actions: {
      updateFilter,
      performSearch,
      downloadCSV,
      clearFilters,
      goToPage,
      changePageSize,
      closeErrorModal,
      closeNoResultsModal,
    },
  };
}
