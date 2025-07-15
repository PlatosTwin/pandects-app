import { useState, useEffect } from "react";
import { FilterOptionsResponse } from "@shared/search";
import { apiUrl } from "@/lib/api-config";

interface UseFilterOptionsReturn {
  targets: string[];
  acquirers: string[];
  isLoading: boolean;
  error: string | null;
}

export function useFilterOptions(): UseFilterOptionsReturn {
  const [targets, setTargets] = useState<string[]>([]);
  const [acquirers, setAcquirers] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Check if we already have cached data in sessionStorage
    const cachedData = sessionStorage.getItem("filterOptions");
    if (cachedData) {
      try {
        const parsed: FilterOptionsResponse = JSON.parse(cachedData);
        setTargets(parsed.targets || []);
        setAcquirers(parsed.acquirers || []);
        setIsLoading(false);
        return;
      } catch (e) {
        // If parsing fails, continue to fetch from API
        sessionStorage.removeItem("filterOptions");
      }
    }

    // Fetch from API
    const fetchFilterOptions = async () => {
      try {
        const response = await fetch(apiUrl("api/filter-options"));

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data: FilterOptionsResponse = await response.json();

        // Update state
        setTargets(data.targets || []);
        setAcquirers(data.acquirers || []);

        // Cache in sessionStorage for future use
        sessionStorage.setItem("filterOptions", JSON.stringify(data));

        setError(null);
      } catch (err) {
        console.error("Failed to fetch filter options:", err);
        setError(
          err instanceof Error ? err.message : "Failed to fetch filter options",
        );

        // Fallback to empty arrays
        setTargets([]);
        setAcquirers([]);
      } finally {
        setIsLoading(false);
      }
    };

    fetchFilterOptions();
  }, []);

  return {
    targets,
    acquirers,
    isLoading,
    error,
  };
}
