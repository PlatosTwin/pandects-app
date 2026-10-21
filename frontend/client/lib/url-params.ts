import type { SearchFilters } from "@shared/sections";
import {
  parseTextMatchMode,
  type SearchMode,
  type SearchQueryFilters,
} from "@shared/search";
import { DEFAULT_PAGE, DEFAULT_PAGE_SIZE } from "@/lib/constants";

type ArrayFilterField = {
  [K in keyof SearchFilters]-?: NonNullable<SearchFilters[K]> extends string[]
    ? K
    : never;
}[keyof SearchFilters];
type TextFilterField = "agreement_uuid" | "section_uuid";

/**
 * Every filter that travels in a search URL / API query string, in the order
 * the params are written. The order is part of the URL format: shared deep
 * links and React Query keys depend on it, so append new fields at the end of
 * their group rather than reordering.
 *
 * `clauseType` is written as `standard_id` (section/deal taxonomy ids); tax
 * mode rewrites it to `tax_standard_id` (see `buildTaxClauseParams`).
 */
export const SEARCH_FILTER_FIELDS: ReadonlyArray<
  | { field: ArrayFilterField; param: string; kind: "array" }
  | { field: TextFilterField; param: string; kind: "text" }
> = [
  { field: "year", param: "year", kind: "array" },
  { field: "target", param: "target", kind: "array" },
  { field: "acquirer", param: "acquirer", kind: "array" },
  {
    field: "transaction_price_total",
    param: "transaction_price_total",
    kind: "array",
  },
  {
    field: "transaction_price_stock",
    param: "transaction_price_stock",
    kind: "array",
  },
  {
    field: "transaction_price_cash",
    param: "transaction_price_cash",
    kind: "array",
  },
  {
    field: "transaction_price_assets",
    param: "transaction_price_assets",
    kind: "array",
  },
  {
    field: "transaction_consideration",
    param: "transaction_consideration",
    kind: "array",
  },
  { field: "target_type", param: "target_type", kind: "array" },
  { field: "acquirer_type", param: "acquirer_type", kind: "array" },
  { field: "target_counsel", param: "target_counsel", kind: "array" },
  { field: "acquirer_counsel", param: "acquirer_counsel", kind: "array" },
  { field: "target_industry", param: "target_industry", kind: "array" },
  { field: "acquirer_industry", param: "acquirer_industry", kind: "array" },
  { field: "deal_status", param: "deal_status", kind: "array" },
  { field: "attitude", param: "attitude", kind: "array" },
  { field: "deal_type", param: "deal_type", kind: "array" },
  { field: "purpose", param: "purpose", kind: "array" },
  { field: "target_pe", param: "target_pe", kind: "array" },
  { field: "acquirer_pe", param: "acquirer_pe", kind: "array" },
  { field: "agreement_uuid", param: "agreement_uuid", kind: "text" },
  { field: "section_uuid", param: "section_uuid", kind: "text" },
  { field: "clauseType", param: "standard_id", kind: "array" },
];

const ARRAY_FILTER_FIELDS = SEARCH_FILTER_FIELDS.filter(
  (entry): entry is { field: ArrayFilterField; param: string; kind: "array" } =>
    entry.kind === "array",
);

/** A fresh, empty filter draft (every array filter empty, default paging). */
export function createEmptySearchFilters(): SearchFilters {
  const filters: SearchFilters = {};
  ARRAY_FILTER_FIELDS.forEach((entry) => {
    filters[entry.field] = [];
  });
  filters.page = DEFAULT_PAGE;
  filters.page_size = DEFAULT_PAGE_SIZE;
  return filters;
}

/**
 * Serialize search filters into query params (URL and API share this format).
 */
export function buildSearchParams(
  searchFilters: SearchFilters,
  includePagination = true,
): URLSearchParams {
  const params = new URLSearchParams();

  for (const entry of SEARCH_FILTER_FIELDS) {
    if (entry.kind === "array") {
      searchFilters[entry.field]?.forEach((value) =>
        params.append(entry.param, value),
      );
    } else {
      const value = searchFilters[entry.field]?.trim();
      if (value) params.append(entry.param, value);
    }
  }

  if (includePagination) {
    if (searchFilters.page)
      params.append("page", searchFilters.page.toString());
    if (searchFilters.page_size) {
      params.append("page_size", searchFilters.page_size.toString());
    }
  }

  return params;
}

/** Tax-clause API params: `clauseType` goes out as `tax_standard_id`. */
export function buildTaxClauseParams(
  filters: SearchQueryFilters,
  includePagination = true,
): URLSearchParams {
  const params = buildSearchParams(filters, includePagination);
  params.delete("standard_id");
  filters.clauseType?.forEach((id) => params.append("tax_standard_id", id));
  if (filters.include_rep_warranty) {
    params.set("include_rep_warranty", "true");
  }
  return params;
}

/**
 * Filters parsed from a shareable URL. `include_rep_warranty` is only
 * meaningful in tax mode.
 */
export type ParsedSearchFilters = SearchQueryFilters;

/** The UI once offered "1B - 5B", which the API never had; old links map to its two buckets. */
const LEGACY_PRICE_BUCKETS: Record<string, string[]> = {
  "1B - 5B": ["1B - 2B", "2B - 5B"],
};

function upgradeLegacyPriceBuckets(values: string[]): string[] {
  return Array.from(
    new Set(values.flatMap((value) => LEGACY_PRICE_BUCKETS[value] ?? [value])),
  );
}

export function parseSearchFilters(
  params: URLSearchParams,
): ParsedSearchFilters {
  const filters: ParsedSearchFilters = {};
  ARRAY_FILTER_FIELDS.forEach((entry) => {
    const values = params.getAll(entry.param);
    filters[entry.field] = entry.param.startsWith("transaction_price_")
      ? upgradeLegacyPriceBuckets(values)
      : values;
  });

  const agreementUuid = params.get("agreement_uuid");
  const sectionUuid = params.get("section_uuid");
  const page = params.get("page");
  const pageSize = params.get("page_size");

  if (agreementUuid) filters.agreement_uuid = agreementUuid;
  if (sectionUuid) filters.section_uuid = sectionUuid;
  if (page) filters.page = Number(page);
  if (pageSize) filters.page_size = Number(pageSize);
  if (params.get("include_rep_warranty") === "true") {
    filters.include_rep_warranty = true;
  }
  // Keyword search: `q` (the landing page links to /search?q=<text>) and
  // `match` (phrase is the default and is left out of the URL).
  const keyword = params.get("q")?.trim();
  if (keyword) {
    filters.text_query = keyword;
    const match = params.get("match");
    if (match) filters.text_match_mode = parseTextMatchMode(match);
  }

  return filters;
}

/** URL param for the keyword query. */
export const KEYWORD_URL_PARAM = "q";
/** URL param for the keyword match mode (omitted for the default, phrase). */
export const KEYWORD_MATCH_URL_PARAM = "match";

export function buildSearchStateParams({
  filters,
  mode,
  sortBy,
  sortDirection,
  taxIncludeRepWarranty,
}: {
  filters: SearchQueryFilters;
  mode: SearchMode;
  sortBy: "year" | "target" | "acquirer" | null;
  sortDirection: "asc" | "desc";
  taxIncludeRepWarranty?: boolean;
}): URLSearchParams {
  const params = buildSearchParams(filters, true);
  if (params.get("page") === String(DEFAULT_PAGE)) params.delete("page");
  if (params.get("page_size") === String(DEFAULT_PAGE_SIZE)) {
    params.delete("page_size");
  }
  if (mode !== "sections") params.set("mode", mode);
  if (mode === "tax" && taxIncludeRepWarranty) {
    params.set("include_rep_warranty", "true");
  }
  if (sortBy && sortBy !== "year") params.set("sort_by", sortBy);
  if (sortDirection !== "desc") params.set("sort_direction", sortDirection);
  // Appended last so URLs without a keyword keep their exact format.
  const keyword = filters.text_query?.trim();
  if (keyword) {
    params.set(KEYWORD_URL_PARAM, keyword);
    const match = filters.text_match_mode;
    if (match && match !== "phrase") params.set(KEYWORD_MATCH_URL_PARAM, match);
  }
  return params;
}
