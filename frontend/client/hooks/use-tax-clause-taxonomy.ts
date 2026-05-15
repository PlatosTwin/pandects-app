import { useQuery } from "@tanstack/react-query";
import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";
import { IS_SERVER_RENDER } from "@/lib/query-client";
import { keys } from "@/lib/query-keys";
import type { ClauseTypeTree } from "@/lib/clause-types";

interface UseTaxClauseTaxonomyOptions {
  enabled?: boolean;
}

async function fetchTaxClauseTaxonomy(): Promise<ClauseTypeTree> {
  const res = await authFetch(apiUrl("v1/taxonomy/tax-clauses"));
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return (await res.json()) as ClauseTypeTree;
}

export function useTaxClauseTaxonomy({
  enabled = true,
}: UseTaxClauseTaxonomyOptions = {}) {
  const query = useQuery({
    queryKey: keys.taxClauseTaxonomy.all,
    queryFn: fetchTaxClauseTaxonomy,
    enabled: enabled && !IS_SERVER_RENDER,
    staleTime: 30 * 60 * 1000,
    gcTime: 60 * 60 * 1000,
  });

  return {
    taxonomy: query.data ?? {},
    isLoading: enabled && query.isLoading,
    error: query.error ? (query.error as Error).message : null,
  };
}
