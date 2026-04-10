import { useState, useEffect } from "react";
import { logger } from "@/lib/logger";
import { apiUrl } from "@/lib/api-config";
import { trackEvent } from "@/lib/analytics";
import { authFetch } from "@/lib/auth-fetch";
import type { ClauseTypeTree } from "@/lib/clause-types";

export type TaxonomyKind = "main" | "tax";

interface UseTaxonomyReturn {
  taxonomyTree: ClauseTypeTree | null;
  isLoading: boolean;
  error: string | null;
}

interface UseTaxonomyOptions {
  enabled?: boolean;
  deferMs?: number;
  fresh?: boolean;
  kind?: TaxonomyKind;
}

export function useTaxonomy(
  options: UseTaxonomyOptions = {},
): UseTaxonomyReturn {
  const { enabled = true, deferMs = 0, fresh = false, kind = "main" } = options;
  const [taxonomyTree, setTaxonomyTree] = useState<ClauseTypeTree | null>(null);
  const [isLoading, setIsLoading] = useState(enabled);
  const [error, setError] = useState<string | null>(null);
  const endpoint = kind === "tax" ? "v1/taxonomy/tax-clauses" : "v1/taxonomy";
  const cacheKey = kind === "tax" ? "taxClauseTaxonomyTree" : "taxonomyTree";
  const endpointLabel = kind === "tax" ? "api/taxonomy/tax-clauses" : "api/taxonomy";

  useEffect(() => {
    if (!enabled) {
      setIsLoading(false);
      return;
    }

    if (!fresh) {
      const cachedData = sessionStorage.getItem(cacheKey);
      if (cachedData) {
        try {
          const parsed: ClauseTypeTree = JSON.parse(cachedData);
          setTaxonomyTree(parsed);
          setIsLoading(false);
          return;
        } catch {
          sessionStorage.removeItem(cacheKey);
        }
      }
    }

    const fetchTaxonomy = async () => {
      try {
        const response = await authFetch(apiUrl(endpoint), {
          cache: fresh ? "no-store" : undefined,
        });

        if (!response.ok) {
          trackEvent("api_error", {
            endpoint: endpointLabel,
            status: response.status,
            status_text: response.statusText,
          });
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data: ClauseTypeTree = await response.json();
        setTaxonomyTree(data);
        sessionStorage.setItem(cacheKey, JSON.stringify(data));
        setError(null);
      } catch (err) {
        logger.error("Failed to fetch taxonomy:", err);
        trackEvent("api_error", {
          endpoint: endpointLabel,
          kind:
            err instanceof TypeError && err.message.includes("fetch")
              ? "network"
              : "unknown",
        });
        setError(err instanceof Error ? err.message : "Failed to fetch taxonomy");
        setTaxonomyTree(null);
      } finally {
        setIsLoading(false);
      }
    };

    if (deferMs > 0) {
      const timer = window.setTimeout(fetchTaxonomy, deferMs);
      return () => window.clearTimeout(timer);
    }

    fetchTaxonomy();
  }, [cacheKey, deferMs, enabled, endpoint, endpointLabel, fresh]);

  return {
    taxonomyTree,
    isLoading,
    error,
  };
}
