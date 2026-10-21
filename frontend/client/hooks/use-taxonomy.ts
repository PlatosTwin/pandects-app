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
  /** The failure itself, for `ErrorState` (status-aware copy and retry). */
  rawError: Error | null;
  retry: () => void;
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
    rawError: query.error,
    retry: () => void query.refetch(),
  };
}

interface TaxonomyCountsResponse {
  counts: Record<string, number>;
}

const TAXONOMY_COUNT_ENDPOINTS: Record<TaxonomyKind, string> = {
  main: "v1/taxonomy/counts",
  tax: "v1/taxonomy/tax-clauses/counts",
};

interface UseTaxonomyCountsReturn {
  /** Sections (main) or tax clauses (tax) per node id; parents include descendants. */
  counts: Record<string, number> | null;
  isLoading: boolean;
  error: string | null;
}

/** Corpus count per taxonomy node, as search counts it. */
export function useTaxonomyCounts(
  options: { enabled?: boolean; kind?: TaxonomyKind } = {},
): UseTaxonomyCountsReturn {
  const { enabled = true, kind = "main" } = options;
  const query = useQuery({
    queryKey: keys.taxonomy.counts(kind),
    queryFn: ({ signal }) =>
      apiGet<TaxonomyCountsResponse>(TAXONOMY_COUNT_ENDPOINTS[kind], undefined, { signal }),
    enabled: enabled && !IS_SERVER_RENDER,
    staleTime: 30 * 60 * 1000,
    gcTime: 60 * 60 * 1000,
  });
  return {
    counts: query.data?.counts ?? null,
    isLoading: enabled && query.isLoading,
    error: query.error ? errorMessage(query.error, "Unable to load counts.") : null,
  };
}

/** One section (main) or tax clause (tax) tagged with a taxonomy node. */
export interface TaxonomyNodeExample {
  agreementUuid: string;
  sectionUuid: string;
  target: string;
  acquirer: string;
  year: string;
  /** Section title, or the clause's anchor label in the tax taxonomy. */
  heading: string | null;
  /** Section XML (main) or clause text (tax). */
  body: string;
  bodyFormat: "xml" | "text";
}

interface SectionExampleResponse {
  results: Array<{
    agreement_uuid: string;
    section_uuid: string;
    target: string;
    acquirer: string;
    year: string | number;
    section_title?: string | null;
    xml?: string | null;
  }>;
}

interface TaxClauseExampleResponse {
  results: Array<{
    agreement_uuid: string;
    section_uuid: string;
    target: string;
    acquirer: string;
    year: string | number;
    anchor_label?: string | null;
    clause_text: string;
  }>;
}

async function fetchTaxonomyNodeExample(
  kind: TaxonomyKind,
  standardId: string,
  signal?: AbortSignal,
): Promise<TaxonomyNodeExample | null> {
  if (kind === "tax") {
    const payload = await apiGet<TaxClauseExampleResponse>(
      "v1/tax-clauses",
      { tax_standard_id: standardId, page_size: 1 },
      { signal },
    );
    const clause = payload.results[0];
    if (!clause) return null;
    return {
      agreementUuid: clause.agreement_uuid,
      sectionUuid: clause.section_uuid,
      target: clause.target,
      acquirer: clause.acquirer,
      year: String(clause.year),
      heading: clause.anchor_label ?? null,
      body: clause.clause_text,
      bodyFormat: "text",
    };
  }
  const payload = await apiGet<SectionExampleResponse>(
    "v1/sections",
    { standard_id: standardId, page_size: 1 },
    { signal },
  );
  const section = payload.results[0];
  if (!section) return null;
  return {
    agreementUuid: section.agreement_uuid,
    sectionUuid: section.section_uuid,
    target: section.target,
    acquirer: section.acquirer,
    year: String(section.year),
    heading: section.section_title ?? null,
    body: section.xml ?? "",
    bodyFormat: "xml",
  };
}

/** The most recent section (or tax clause) tagged with `standardId`, or null. */
export function useTaxonomyNodeExample(
  kind: TaxonomyKind,
  standardId: string | null,
): { example: TaxonomyNodeExample | null; isLoading: boolean; error: unknown } {
  const query = useQuery({
    queryKey: keys.taxonomy.example(kind, standardId ?? ""),
    queryFn: ({ signal }) => fetchTaxonomyNodeExample(kind, standardId ?? "", signal),
    enabled: Boolean(standardId) && !IS_SERVER_RENDER,
    staleTime: 30 * 60 * 1000,
    gcTime: 60 * 60 * 1000,
  });
  return {
    example: query.data ?? null,
    isLoading: Boolean(standardId) && query.isLoading,
    error: query.error ?? null,
  };
}
