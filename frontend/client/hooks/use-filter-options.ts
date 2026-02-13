import { useState, useEffect } from "react";
import { logger } from "@/lib/logger";
import { FilterOptionsResponse } from "@shared/search";
import { apiUrl } from "@/lib/api-config";
import { trackEvent } from "@/lib/analytics";
import { authFetch } from "@/lib/auth-fetch";

interface UseFilterOptionsReturn {
  targets: string[];
  acquirers: string[];
  target_industries: string[];
  acquirer_industries: string[];
  isLoading: boolean;
  error: string | null;
}

interface UseFilterOptionsOptions {
  enabled?: boolean;
  deferMs?: number;
}

export function useFilterOptions(
  options: UseFilterOptionsOptions = {},
): UseFilterOptionsReturn {
  const { enabled = true, deferMs = 0 } = options;
  const [targets, setTargets] = useState<string[]>([]);
  const [acquirers, setAcquirers] = useState<string[]>([]);
  const [target_industries, setTargetIndustries] = useState<string[]>([]);
  const [acquirer_industries, setAcquirerIndustries] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(enabled);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) {
      setIsLoading(false);
      return;
    }

    // Check if we already have cached data in sessionStorage
    const cachedData = sessionStorage.getItem("filterOptions");
    if (cachedData) {
      try {
        const parsed: FilterOptionsResponse = JSON.parse(cachedData);
        setTargets(parsed.targets || []);
        setAcquirers(parsed.acquirers || []);
        setTargetIndustries(parsed.target_industries || []);
        setAcquirerIndustries(parsed.acquirer_industries || []);
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
        const response = await authFetch(apiUrl("v1/filter-options"));

        if (!response.ok) {
          trackEvent("api_error", {
            endpoint: "api/filter-options",
            status: response.status,
            status_text: response.statusText,
          });
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data: FilterOptionsResponse = await response.json();

        // Update state
        setTargets(data.targets || []);
        setAcquirers(data.acquirers || []);
        setTargetIndustries(data.target_industries || []);
        setAcquirerIndustries(data.acquirer_industries || []);

        // Cache in sessionStorage for future use
        sessionStorage.setItem("filterOptions", JSON.stringify(data));

        setError(null);
      } catch (err) {
        logger.error("Failed to fetch filter options:", err);
        trackEvent("api_error", {
          endpoint: "api/filter-options",
          kind:
            err instanceof TypeError && err.message.includes("fetch")
              ? "network"
              : "unknown",
        });
        setError(
          err instanceof Error ? err.message : "Failed to fetch filter options",
        );

        // Fallback to empty arrays
        setTargets([]);
        setAcquirers([]);
        setTargetIndustries([]);
        setAcquirerIndustries([]);
      } finally {
        setIsLoading(false);
      }
    };

    if (deferMs > 0) {
      const timer = window.setTimeout(fetchFilterOptions, deferMs);
      return () => window.clearTimeout(timer);
    }

    fetchFilterOptions();
  }, [deferMs, enabled]);

  return {
    targets,
    acquirers,
    target_industries,
    acquirer_industries,
    isLoading,
    error,
  };
}
