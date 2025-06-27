import { useState, useCallback } from "react";
import { SearchFilters, SearchResult, SearchResponse } from "@shared/search";

export function useSearch() {
  const [filters, setFilters] = useState<SearchFilters>({
    year: "",
    target: "",
    acquirer: "",
    clauseType: "",
    page: 1,
    pageSize: 25,
  });

  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [hasSearched, setHasSearched] = useState(false);
  const [totalCount, setTotalCount] = useState(0);
  const [totalPages, setTotalPages] = useState(0);

  const updateFilter = useCallback(
    (field: keyof SearchFilters, value: string) => {
      setFilters((prev) => ({ ...prev, [field]: value }));
    },
    [],
  );

  const performSearch = useCallback(
    async (resetPage = false) => {
      setIsSearching(true);
      if (resetPage) {
        setHasSearched(true);
      }

      try {
        const searchFilters = resetPage ? { ...filters, page: 1 } : filters;
        if (resetPage) {
          setFilters((prev) => ({ ...prev, page: 1 }));
        }

        const params = new URLSearchParams();
        if (searchFilters.year) params.append("year", searchFilters.year);
        if (searchFilters.target) params.append("target", searchFilters.target);
        if (searchFilters.acquirer)
          params.append("acquirer", searchFilters.acquirer);
        if (searchFilters.clauseType)
          params.append("clauseType", searchFilters.clauseType);
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
          setSearchResults(responseData);
          setTotalCount(responseData.length);
          setTotalPages(1);
        } else {
          // New format: API returns SearchResponse
          const searchResponse = responseData as SearchResponse;
          setSearchResults(searchResponse.results);
          setTotalCount(searchResponse.totalCount);
          setTotalPages(searchResponse.totalPages);
        }
      } catch (error) {
        console.error("Search failed:", error);
        setSearchResults([]);
        setTotalCount(0);
        setTotalPages(0);
      } finally {
        setIsSearching(false);
      }
    },
    [filters],
  );

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
      year: "",
      target: "",
      acquirer: "",
      clauseType: "",
      page: 1,
      pageSize: 25,
    });
    setSearchResults([]);
    setHasSearched(false);
    setTotalCount(0);
    setTotalPages(0);
  }, []);

  const goToPage = useCallback((page: number) => {
    setFilters((prev) => ({ ...prev, page }));
  }, []);

  const changePageSize = useCallback((pageSize: number) => {
    setFilters((prev) => ({ ...prev, pageSize, page: 1 }));
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
    actions: {
      updateFilter,
      performSearch,
      downloadCSV,
      clearFilters,
      goToPage,
      changePageSize,
    },
  };
}
