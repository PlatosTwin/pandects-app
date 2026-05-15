import { useQuery } from "@tanstack/react-query";
import { apiUrl } from "@/lib/api-config";
import { trackEvent } from "@/lib/analytics";
import { authFetch } from "@/lib/auth-fetch";
import { IS_SERVER_RENDER } from "@/lib/query-client";
import { keys } from "@/lib/query-keys";
import type { ClauseTypeTree } from "@/lib/clause-types";

export type TaxonomyKind = "main" | "tax";

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

async function fetchTaxonomy(
  kind: TaxonomyKind,
  fresh: boolean,
): Promise<ClauseTypeTree> {
  const endpoint = kind === "tax" ? "v1/taxonomy/tax-clauses" : "v1/taxonomy";
  const endpointLabel =
    kind === "tax" ? "api/taxonomy/tax-clauses" : "api/taxonomy";

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
  return (await response.json()) as ClauseTypeTree;
}

export function useTaxonomy(
  options: UseTaxonomyOptions = {},
): UseTaxonomyReturn {
  const { enabled = true, fresh = false, kind = "main" } = options;

  const query = useQuery({
    queryKey:
      kind === "tax" ? keys.taxClauseTaxonomy.all : keys.taxonomy.all,
    queryFn: () => fetchTaxonomy(kind, fresh),
    enabled: enabled && !IS_SERVER_RENDER,
    staleTime: fresh ? 0 : 30 * 60 * 1000,
    gcTime: 60 * 60 * 1000,
  });

  return {
    taxonomyTree: query.data ?? null,
    isLoading: enabled && query.isLoading,
    error: query.error ? (query.error as Error).message : null,
  };
}
