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
      // TODO: Replace with actual API call
      // const response = await fetch('/api/search', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify(filters)
      // });
      // const data = await response.json();

      // Simulate API call
      await new Promise((resolve) => setTimeout(resolve, 1500));

      // Mock search results - replace with actual API response
      const mockResults: SearchResult[] = [
        {
          id: "1",
          year: "2023",
          target: "TechCorp Inc.",
          acquirer: "MegaCorp LLC",
          articleTitle: "Article VII",
          sectionTitle: "Section 7.2",
          text: "In the event of a material adverse change affecting the target company, the acquirer may terminate this agreement with thirty (30) days written notice. Such termination shall not affect any obligations that have accrued prior to the effective date of termination.",
          sectionUuid: "uuid-section-001",
          agreementUuid: "uuid-agreement-001",
          announcementDate: "2023-03-15",
        },
        {
          id: "2",
          year: "2023",
          target: "DataSystems Corp",
          acquirer: "Global Holdings",
          articleTitle: "Article V",
          sectionTitle: "Section 5.4",
          text: "The target company represents and warrants that all material contracts to which it is a party are valid, binding, and enforceable in accordance with their terms, and no breach or default exists under any such contract.",
          sectionUuid: "uuid-section-002",
          agreementUuid: "uuid-agreement-002",
          announcementDate: "2023-06-22",
        },
        {
          id: "3",
          year: "2022",
          target: "CloudTech Solutions",
          acquirer: "Enterprise Systems",
          articleTitle: "Article IX",
          sectionTitle: "Section 9.1",
          text: "Each party shall indemnify and hold harmless the other party from and against any and all losses, damages, liabilities, costs, and expenses arising out of or resulting from any breach of the representations, warranties, or covenants contained herein.",
          sectionUuid: "uuid-section-003",
          agreementUuid: "uuid-agreement-003",
          announcementDate: "2022-11-08",
        },
      ];

      setSearchResults(mockResults);
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
          `"${result.text.replace(/"/g, '""')}"`,
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
