import { useState, useCallback } from "react";
import { SearchFilters, SearchResult } from "@shared/search";

export function useSearch() {
  const [filters, setFilters] = useState<SearchFilters>({
    year: "",
    target: "",
    acquirer: "",
    clauseType: "",
  });

  const [isSearching, setIsSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [hasSearched, setHasSearched] = useState(false);

  const updateFilter = useCallback(
    (field: keyof SearchFilters, value: string) => {
      setFilters((prev) => ({ ...prev, [field]: value }));
    },
    [],
  );

  const performSearch = useCallback(async () => {
    setIsSearching(true);
    setHasSearched(true);

    try {
      const params = new URLSearchParams();
      if (filters.year) params.append('year', filters.year);
      if (filters.target) params.append('target', filters.target);
      if (filters.acquirer) params.append('acquirer', filters.acquirer);
      if (filters.clauseType) params.append('clauseType', filters.clauseType);

      const queryString = params.toString();
      const res = await fetch(`http://127.0.0.1:5000/api/search?${queryString}`);

      // Check if the response is ok (status 200-299)
      if (!res.ok) {
        if (res.status === 404) {
          // UUID not found in database
          // updateState({
          //   showErrorModal: true,
          //   errorMessage: "No page with that UUID exists in mna.llm_output.",
          // });
          return;
        }
        // Other HTTP errors
        throw new Error(`HTTP ${res.status}: ${res.statusText}`);
      }

      const responseData: SearchResult[] = await res.json();

      // Check if the response data is empty or null
      if (!responseData || Object.keys(responseData).length === 0) {
        // updateState({
        //   showErrorModal: true,
        //   errorMessage: "No page with that UUID exists in mna.llm_output.",
        // });
        return;
      }

      setSearchResults(responseData);

    } catch (error) {
      console.error("Search failed:", error);
      setSearchResults([]);
    } finally {
      setIsSearching(false);
    }
  }, [filters]);

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
    });
    setSearchResults([]);
    setHasSearched(false);
  }, []);

  return {
    filters,
    isSearching,
    searchResults,
    hasSearched,
    actions: {
      updateFilter,
      performSearch,
      downloadCSV,
      clearFilters,
    },
  };
}
