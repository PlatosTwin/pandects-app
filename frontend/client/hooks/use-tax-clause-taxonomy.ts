import { useEffect, useState } from "react";
import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";
import type { ClauseTypeTree } from "@/lib/clause-types";
import { logger } from "@/lib/logger";

const CACHE_KEY = "taxClauseTaxonomy:v1";

interface UseTaxClauseTaxonomyOptions {
  enabled?: boolean;
}

export function useTaxClauseTaxonomy({ enabled = true }: UseTaxClauseTaxonomyOptions = {}) {
  const [taxonomy, setTaxonomy] = useState<ClauseTypeTree>({});
  const [isLoading, setIsLoading] = useState(enabled);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!enabled) {
      setIsLoading(false);
      return;
    }

    const cached = sessionStorage.getItem(CACHE_KEY);
    if (cached) {
      try {
        const parsed = JSON.parse(cached) as ClauseTypeTree;
        setTaxonomy(parsed);
        setIsLoading(false);
        return;
      } catch {
        sessionStorage.removeItem(CACHE_KEY);
      }
    }

    let cancelled = false;
    (async () => {
      try {
        const res = await authFetch(apiUrl("v1/taxonomy/tax-clauses"));
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = (await res.json()) as ClauseTypeTree;
        if (cancelled) return;
        setTaxonomy(data);
        sessionStorage.setItem(CACHE_KEY, JSON.stringify(data));
        setError(null);
      } catch (err) {
        if (cancelled) return;
        logger.error("Failed to load tax clause taxonomy:", err);
        setError(err instanceof Error ? err.message : "Failed to load tax taxonomy");
        setTaxonomy({});
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [enabled]);

  return { taxonomy, isLoading, error };
}
