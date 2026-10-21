import { useQuery } from "@tanstack/react-query";
import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";
import { originSearchParams, type AgreementOrigin } from "@/lib/agreement-links";
import { KEYWORD_SEARCH_SUPPORT } from "@/lib/search-configs";
import { KEYWORD_MATCH_URL_PARAM, KEYWORD_URL_PARAM } from "@/lib/url-params";
import { parseTextMatchMode } from "@shared/search";

/** Params the search page puts in its URL that don't belong in a scoped query. */
const NON_FILTER_PARAMS = ["mode", "sort_by", "sort_direction", "page", "page_size"];
/** A search only "matches sections" when it filters on sections (or a keyword). */
const SECTION_FILTER_PARAMS = ["standard_id", "section_uuid", KEYWORD_URL_PARAM];
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
  const keyword = source.get(KEYWORD_URL_PARAM)?.trim() ?? "";
  if (
    endpoint === "sections" &&
    !SECTION_FILTER_PARAMS.some((key) =>
      key === KEYWORD_URL_PARAM ? keyword !== "" : source.has(key),
    )
  ) {
    return null;
  }
  const params = new URLSearchParams();
  source.forEach((value, key) => {
    if (NON_FILTER_PARAMS.includes(key) || key === "agreement_uuid") return;
    // The search URL carries the keyword as `q`/`match`; the API takes
    // `text_query`/`text_match_mode`, and only in modes with keyword support.
    if (key === KEYWORD_URL_PARAM || key === KEYWORD_MATCH_URL_PARAM) return;
    // The search URL keeps tax taxonomy ids under `standard_id`; the tax
    // endpoint takes them as `tax_standard_id` (see use-tax-clauses).
    const apiKey = endpoint === "tax-clauses" && key === "standard_id" ? "tax_standard_id" : key;
    params.append(apiKey, value);
  });
  const keywordMode = endpoint === "tax-clauses" ? "tax" : "sections";
  if (keyword && KEYWORD_SEARCH_SUPPORT[keywordMode]) {
    params.set("text_query", keyword);
    const match = source.get(KEYWORD_MATCH_URL_PARAM);
    const matchMode = match ? parseTextMatchMode(match) : "phrase";
    if (matchMode !== "phrase") params.set("text_match_mode", matchMode);
  }
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
