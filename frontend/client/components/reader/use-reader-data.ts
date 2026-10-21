import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";
import { logger } from "@/lib/logger";
import { originSearchParams, type AgreementOrigin } from "@/lib/agreement-links";
import type {
  AgreementSectionIndexItem,
  AgreementSectionIndexResponse,
} from "@shared/transactions";

/**
 * Section index (uuid, titles, taxonomy ids) for the contents type filter.
 *
 * DATA HOOK PLUG-IN POINT: this is the hand-rolled fetch the shared React
 * Query hook replaces. Swap the body for that hook and keep the return shape
 * ({ sectionIndex, isLoading, error }); nothing else in the reader changes.
 */
export function useAgreementSectionIndex(agreementUuid: string) {
  const [sectionIndex, setSectionIndex] = useState<AgreementSectionIndexItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    const fetchSectionIndex = async () => {
      try {
        setIsLoading(true);
        setError(null);
        const response = await authFetch(
          apiUrl(`v1/agreements/${agreementUuid}/sections`),
          { signal: controller.signal },
        );
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const payload = (await response.json()) as AgreementSectionIndexResponse;
        if (!cancelled) setSectionIndex(payload.results);
      } catch (fetchError) {
        if (!cancelled) {
          logger.error("Failed to fetch agreement sections index:", fetchError);
          setSectionIndex([]);
          setError(
            fetchError instanceof Error
              ? fetchError.message
              : "Unable to load agreement sections.",
          );
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    };

    void fetchSectionIndex();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [agreementUuid]);

  return { sectionIndex, isLoading, error };
}

/** Params the search page puts in its URL that don't belong in a scoped query. */
const NON_FILTER_PARAMS = ["mode", "sort_by", "sort_direction", "page", "page_size"];
/** A search only "matches sections" when it filters on sections. */
const SECTION_FILTER_PARAMS = ["standard_id", "section_uuid"];
const MATCH_PAGE_SIZE = 100;

export interface ReaderMatchesQuery {
  /** API path (without `v1/` prefix handling) and query string. */
  endpoint: "sections" | "tax-clauses";
  params: URLSearchParams;
}

/**
 * Builds the originating search, scoped to one agreement, or null when the
 * origin isn't a section-level search (deal-only filters, favorites, index).
 */
export function buildReaderMatchesQuery(
  origin: AgreementOrigin | null,
  agreementUuid: string,
): ReaderMatchesQuery | null {
  const source = originSearchParams(origin);
  if (!source) return null;
  const mode = source.get("mode");
  const endpoint = mode === "tax" ? "tax-clauses" : "sections";
  if (endpoint === "sections" && !SECTION_FILTER_PARAMS.some((key) => source.has(key))) {
    return null;
  }
  const params = new URLSearchParams();
  source.forEach((value, key) => {
    if (NON_FILTER_PARAMS.includes(key) || key === "agreement_uuid") return;
    // The search URL keeps tax taxonomy ids under `standard_id`; the tax
    // endpoint takes them as `tax_standard_id` (see use-tax-clauses).
    const apiKey = endpoint === "tax-clauses" && key === "standard_id" ? "tax_standard_id" : key;
    params.append(apiKey, value);
  });
  params.set("agreement_uuid", agreementUuid);
  params.set("page_size", String(MATCH_PAGE_SIZE));
  if (endpoint === "sections") params.set("include_dump", "false");
  return { endpoint, params };
}

interface MatchesResponse {
  results: Array<{ section_uuid?: string | null }>;
  total_count?: number;
}

export interface ReaderMatches {
  /** Matched section uuids, deduped, in API order (the reader re-sorts by document order). */
  sectionUuids: string[];
  /** Server-side total; can exceed `sectionUuids.length` when the page was capped. */
  totalCount: number;
}

/**
 * Every match of the originating search inside this agreement, by re-running
 * that search with `agreement_uuid` set. Disabled (data undefined) when the
 * origin isn't a section-level search.
 */
export function useReaderMatches(origin: AgreementOrigin | null, agreementUuid: string) {
  const query = buildReaderMatchesQuery(origin, agreementUuid);
  const queryString = query ? `${query.endpoint}?${query.params.toString()}` : null;
  return useQuery<ReaderMatches>({
    queryKey: ["reader-matches", queryString],
    enabled: queryString !== null,
    staleTime: 5 * 60 * 1000,
    queryFn: async ({ signal }) => {
      const response = await authFetch(apiUrl(`v1/${queryString}`), { signal });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const payload = (await response.json()) as MatchesResponse;
      const seen = new Set<string>();
      const sectionUuids: string[] = [];
      for (const result of payload.results ?? []) {
        const uuid = result.section_uuid;
        if (uuid && !seen.has(uuid)) {
          seen.add(uuid);
          sectionUuids.push(uuid);
        }
      }
      return {
        sectionUuids,
        totalCount: Math.max(payload.total_count ?? 0, sectionUuids.length),
      };
    },
  });
}
