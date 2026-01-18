import { useState, useEffect } from "react";
import { apiUrl } from "@/lib/api-config";
import { trackEvent } from "@/lib/analytics";
import { authFetch } from "@/lib/auth-fetch";
import type { ClauseTypeTree } from "@/lib/clause-types";

interface UseTaxonomyReturn {
  taxonomyTree: ClauseTypeTree | null;
  isLoading: boolean;
  error: string | null;
}

interface UseTaxonomyOptions {
  enabled?: boolean;
  deferMs?: number;
  fresh?: boolean;
}

export function useTaxonomy(
  options: UseTaxonomyOptions = {},
): UseTaxonomyReturn {
  const { enabled = true, deferMs = 0, fresh = false } = options;
  const [taxonomyTree, setTaxonomyTree] = useState<ClauseTypeTree | null>(null);
  const [isLoading, setIsLoading] = useState(enabled);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) {
      setIsLoading(false);
      return;
    }

    if (!fresh) {
      const cachedData = sessionStorage.getItem("taxonomyTree");
      if (cachedData) {
        try {
          const parsed: ClauseTypeTree = JSON.parse(cachedData);
          setTaxonomyTree(parsed);
          setIsLoading(false);
          return;
        } catch {
          sessionStorage.removeItem("taxonomyTree");
        }
      }
    }

    const fetchTaxonomy = async () => {
      try {
        const response = await authFetch(apiUrl("v1/taxonomy"), {
          cache: fresh ? "no-store" : undefined,
        });

        if (!response.ok) {
          trackEvent("api_error", {
            endpoint: "api/taxonomy",
            status: response.status,
            status_text: response.statusText,
          });
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data: ClauseTypeTree = await response.json();
        setTaxonomyTree(data);
        sessionStorage.setItem("taxonomyTree", JSON.stringify(data));
        setError(null);
      } catch (err) {
        if (import.meta.env.DEV) {
          console.error("Failed to fetch taxonomy:", err);
        }
        trackEvent("api_error", {
          endpoint: "api/taxonomy",
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
  }, [deferMs, enabled, fresh]);

  return {
    taxonomyTree,
    isLoading,
    error,
  };
}
