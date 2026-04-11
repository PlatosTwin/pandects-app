import { useState, useEffect, useMemo } from "react";
import { logger } from "@/lib/logger";
import { FilterOptionsResponse } from "@shared/sections";
import { apiUrl } from "@/lib/api-config";
import { trackEvent } from "@/lib/analytics";
import { authFetch } from "@/lib/auth-fetch";

const FILTER_OPTIONS_CACHE_KEY = "filterOptions:v2";
const LEGACY_FILTER_OPTIONS_CACHE_KEY = "filterOptions";

interface UseFilterOptionsReturn {
  targets: string[];
  acquirers: string[];
  target_counsels: string[];
  acquirer_counsels: string[];
  target_industries: string[];
  acquirer_industries: string[];
  isLoading: boolean;
  error: string | null;
}

interface UseFilterOptionsOptions {
  enabled?: boolean;
  deferMs?: number;
  fields?: Array<keyof FilterOptionsResponse>;
}

function normalizeFields(fields?: Array<keyof FilterOptionsResponse>) {
  if (!fields || fields.length === 0) return null;
  return Array.from(new Set(fields)).sort();
}

function cacheKeyForFields(fields?: Array<keyof FilterOptionsResponse>) {
  const normalizedFields = normalizeFields(fields);
  if (!normalizedFields) return FILTER_OPTIONS_CACHE_KEY;
  return `${FILTER_OPTIONS_CACHE_KEY}:${normalizedFields.join(",")}`;
}

export function useFilterOptions(
  options: UseFilterOptionsOptions = {},
): UseFilterOptionsReturn {
  const { enabled = true, deferMs = 0, fields } = options;
  const fieldsKey = JSON.stringify(fields ?? []);
  const normalizedFields = useMemo(() => normalizeFields(fields), [fieldsKey]);
  const cacheKey = useMemo(() => cacheKeyForFields(fields), [fieldsKey]);
  const [targets, setTargets] = useState<string[]>([]);
  const [acquirers, setAcquirers] = useState<string[]>([]);
  const [target_counsels, setTargetCounsels] = useState<string[]>([]);
  const [acquirer_counsels, setAcquirerCounsels] = useState<string[]>([]);
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
    const cachedData = sessionStorage.getItem(cacheKey);
    if (cachedData) {
      try {
        const parsed: FilterOptionsResponse = JSON.parse(cachedData);
        setTargets(parsed.targets || []);
        setAcquirers(parsed.acquirers || []);
        setTargetCounsels(parsed.target_counsels || []);
        setAcquirerCounsels(parsed.acquirer_counsels || []);
        setTargetIndustries(parsed.target_industries || []);
        setAcquirerIndustries(parsed.acquirer_industries || []);
        setIsLoading(false);
        return;
      } catch (e) {
        // If parsing fails, continue to fetch from API
        sessionStorage.removeItem(cacheKey);
      }
    }
    sessionStorage.removeItem(LEGACY_FILTER_OPTIONS_CACHE_KEY);

    // Fetch from API
    const fetchFilterOptions = async () => {
      try {
        const params = new URLSearchParams();
        normalizedFields?.forEach((field) => {
          params.append("fields", field);
        });
        const endpoint = params.size
          ? `v1/filter-options?${params.toString()}`
          : "v1/filter-options";
        const response = await authFetch(apiUrl(endpoint));

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
        setTargetCounsels(data.target_counsels || []);
        setAcquirerCounsels(data.acquirer_counsels || []);
        setTargetIndustries(data.target_industries || []);
        setAcquirerIndustries(data.acquirer_industries || []);

        // Cache in sessionStorage for future use
        sessionStorage.setItem(cacheKey, JSON.stringify(data));

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
        setTargetCounsels([]);
        setAcquirerCounsels([]);
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
  }, [cacheKey, deferMs, enabled, normalizedFields]);

  return {
    targets,
    acquirers,
    target_counsels,
    acquirer_counsels,
    target_industries,
    acquirer_industries,
    isLoading,
    error,
  };
}
