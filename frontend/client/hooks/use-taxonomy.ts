import { useQuery } from "@tanstack/react-query";
import { trackEvent } from "@/lib/analytics";
import { apiGet, errorMessage, isApiError } from "@/lib/api/client";
import { IS_SERVER_RENDER } from "@/lib/query-client";
import { keys, type TaxonomyKind } from "@/lib/query-keys";
import type { ClauseTypeTree } from "@/lib/clause-types";

export type { TaxonomyKind };

interface UseTaxonomyReturn {
  taxonomyTree: ClauseTypeTree | null;
  isLoading: boolean;
  error: string | null;
}

interface UseTaxonomyOptions {
  enabled?: boolean;
  fresh?: boolean;
  kind?: TaxonomyKind;
}

const TAXONOMY_ENDPOINTS: Record<TaxonomyKind, string> = {
  main: "v1/taxonomy",
  tax: "v1/taxonomy/tax-clauses",
};

async function fetchTaxonomy(
  kind: TaxonomyKind,
  fresh: boolean,
  signal?: AbortSignal,
): Promise<ClauseTypeTree> {
  const endpoint = TAXONOMY_ENDPOINTS[kind];
  try {
    return await apiGet<ClauseTypeTree>(endpoint, undefined, {
      cache: fresh ? "no-store" : undefined,
      signal,
    });
  } catch (error) {
    if (isApiError(error) && !error.isNetworkError) {
      trackEvent("api_error", {
        endpoint,
        status: error.status,
        status_text: error.statusText,
      });
    }
    throw error;
  }
}

/** Clause taxonomy tree: `kind: "main"` (sections) or `"tax"` (tax clauses). */
export function useTaxonomy(
  options: UseTaxonomyOptions = {},
): UseTaxonomyReturn {
  const { enabled = true, fresh = false, kind = "main" } = options;

  const query = useQuery({
    queryKey: keys.taxonomy.byKind(kind),
    queryFn: ({ signal }) => fetchTaxonomy(kind, fresh, signal),
    enabled: enabled && !IS_SERVER_RENDER,
    staleTime: fresh ? 0 : 30 * 60 * 1000,
    gcTime: 60 * 60 * 1000,
  });

  return {
    taxonomyTree: query.data ?? null,
    isLoading: enabled && query.isLoading,
    error: query.error ? errorMessage(query.error, "Unable to load the taxonomy.") : null,
  };
}
