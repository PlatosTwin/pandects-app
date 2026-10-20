/**
 * Per-mode configuration for `useSearch` (hooks/use-search.ts).
 *
 * Event names and payload fields are stable: analytics dashboards key on
 * them. `api_error` endpoint labels match the request paths ("v1/...").
 */
import type { SearchResult, SectionResultMetadata } from "@shared/sections";
import type { SearchMode } from "@shared/search";
import type { TaxClauseSearchResult } from "@shared/tax-clauses";
import type { TransactionSearchResult } from "@shared/transactions";
import { csvEscape } from "@/lib/csv-export";
import {
  formatCompactCurrencyValue,
  formatDateValue,
  formatEnumValue,
} from "@/lib/format-utils";
import { keys } from "@/lib/query-keys";
import { buildSearchParams, buildTaxClauseParams } from "@/lib/url-params";
import type { SearchEndpointConfig, SearchSnapshot } from "@/hooks/use-search";

/** Deal fields shown on each section result; the API returns them under `metadata`. */
export const SECTION_RESULT_METADATA_FIELDS = [
  "transaction_price_total",
  "deal_status",
  "deal_type",
  "purpose",
] as const satisfies readonly (keyof SectionResultMetadata)[];

function appendSort(
  params: URLSearchParams,
  { sortBy, sortDirection }: SearchSnapshot,
): URLSearchParams {
  if (sortBy) {
    params.set("sort_by", sortBy);
    params.set("sort_direction", sortDirection);
  }
  return params;
}

export function buildSectionsApiParams(snapshot: SearchSnapshot): URLSearchParams {
  const params = buildSearchParams(snapshot.filters);
  SECTION_RESULT_METADATA_FIELDS.forEach((field) => params.append("metadata", field));
  return appendSort(params, snapshot);
}

export function buildTaxClausesApiParams(snapshot: SearchSnapshot): URLSearchParams {
  return appendSort(buildTaxClauseParams(snapshot.filters), snapshot);
}

export function buildDealsApiParams(snapshot: SearchSnapshot): URLSearchParams {
  return appendSort(buildSearchParams(snapshot.filters), snapshot);
}

export const SECTIONS_SEARCH_CONFIG: SearchEndpointConfig<SearchResult> = {
  mode: "sections",
  endpoint: "v1/sections",
  queryKey: (params) => keys.sections.search(params),
  buildApiParams: buildSectionsApiParams,
  getResultId: (result) => result.id,
  silentNotFound: true,
  logLabel: "Search failed:",
  telemetry: {
    endpointLabel: "v1/sections",
    trackHttpErrors: true,
    trackQueryFailures: true,
    performed: {
      event: "sections_performed",
      payload: ({ filters, sortBy, sortDirection }) => ({
        years_count: filters.year?.length ?? 0,
        targets_count: filters.target?.length ?? 0,
        acquirers_count: filters.acquirer?.length ?? 0,
        clause_types_count: filters.clauseType?.length ?? 0,
        page: filters.page,
        page_size: filters.page_size,
        sort_by: sortBy ?? "none",
        sort_direction: sortDirection,
      }),
    },
    resultsLoadedEvent: "sections_results_loaded",
    csvDownload: {
      event: "sections_csv_download_click",
      payload: (filters) => ({
        years_count: filters.year?.length ?? 0,
        targets_count: filters.target?.length ?? 0,
        acquirers_count: filters.acquirer?.length ?? 0,
        clause_types_count: filters.clauseType?.length ?? 0,
      }),
    },
  },
  csv: {
    filenameStem: "ma_clauses",
    selectedFilenameStem: "ma_clauses_selected",
    failureLogLabel: "Failed to fetch all results for CSV:",
    headers: [
      "Year",
      "Target",
      "Acquirer",
      "Article Title",
      "Section Title",
      "Text",
      "Section UUID",
      "Agreement UUID",
    ],
    toRow: (result) =>
      [
        result.year,
        csvEscape(result.target),
        csvEscape(result.acquirer),
        csvEscape(result.article_title),
        csvEscape(result.section_title),
        csvEscape(result.xml),
        result.section_uuid,
        result.agreement_uuid,
      ].join(","),
  },
};

export const TAX_CLAUSES_SEARCH_CONFIG: SearchEndpointConfig<TaxClauseSearchResult> = {
  mode: "tax",
  endpoint: "v1/tax-clauses",
  queryKey: (params) => keys.taxClauses.search(params),
  buildApiParams: buildTaxClausesApiParams,
  getResultId: (result) => result.id,
  silentNotFound: true,
  logLabel: "Tax clause search failed:",
  telemetry: {
    endpointLabel: "v1/tax-clauses",
    trackHttpErrors: true,
    trackQueryFailures: true,
    performed: {
      event: "tax_clauses_performed",
      payload: ({ filters, sortBy, sortDirection }) => ({
        years_count: filters.year?.length ?? 0,
        targets_count: filters.target?.length ?? 0,
        acquirers_count: filters.acquirer?.length ?? 0,
        tax_standard_ids_count: filters.clauseType?.length ?? 0,
        include_rep_warranty: !!filters.include_rep_warranty,
        page: filters.page,
        page_size: filters.page_size,
        sort_by: sortBy ?? "none",
        sort_direction: sortDirection,
      }),
    },
    resultsLoadedEvent: "tax_clauses_results_loaded",
    csvDownload: { event: "tax_clauses_csv_download_click" },
  },
  csv: {
    filenameStem: "ma_tax_clauses",
    selectedFilenameStem: "ma_tax_clauses_selected",
    failureLogLabel: "Failed to fetch tax clauses for CSV:",
    headers: [
      "Year",
      "Target",
      "Acquirer",
      "Context",
      "Tax Clause Types",
      "Clause Text",
      "Clause UUID",
      "Section UUID",
      "Agreement UUID",
    ],
    toRow: (r) =>
      [
        r.year ?? "",
        csvEscape(r.target),
        csvEscape(r.acquirer),
        r.context_type,
        csvEscape(r.tax_standard_ids.join("; ")),
        csvEscape(r.clause_text),
        r.clause_uuid,
        r.section_uuid,
        r.agreement_uuid,
      ].join(","),
  },
};

/** Deals (transactions) mode. No performed/results/CSV-click events by design. */
export const DEALS_SEARCH_CONFIG: SearchEndpointConfig<TransactionSearchResult> = {
  mode: "transactions",
  endpoint: "v1/search/agreements",
  queryKey: (params) => keys.transactions.search(params),
  buildApiParams: buildDealsApiParams,
  getResultId: (result) => result.agreement_uuid,
  silentNotFound: false,
  logLabel: "Agreement search failed:",
  telemetry: {
    endpointLabel: "v1/search/agreements",
    trackHttpErrors: false,
    trackQueryFailures: false,
  },
  csv: {
    filenameStem: "ma_deals",
    selectedFilenameStem: "ma_deals_selected",
    failureLogLabel: "Failed to fetch all transactions for CSV:",
    headers: [
      "Year",
      "Target",
      "Acquirer",
      "Deal Type",
      "Deal Status",
      "Attitude",
      "Purpose",
      "Consideration",
      "Deal Value",
      "Filed",
      "Announced",
      "Closed",
      "Target Industry",
      "Acquirer Industry",
      "SEC Filing URL",
      "Agreement UUID",
    ],
    toRow: (r) =>
      [
        r.year ?? "",
        csvEscape(r.target),
        csvEscape(r.acquirer),
        csvEscape(formatEnumValue(r.deal_type)),
        csvEscape(formatEnumValue(r.deal_status)),
        csvEscape(formatEnumValue(r.attitude)),
        csvEscape(formatEnumValue(r.purpose)),
        csvEscape(formatEnumValue(r.transaction_consideration)),
        csvEscape(formatCompactCurrencyValue(r.transaction_price_total)),
        csvEscape(formatDateValue(r.filing_date)),
        csvEscape(formatDateValue(r.announce_date)),
        csvEscape(formatDateValue(r.close_date)),
        csvEscape(r.target_industry),
        csvEscape(r.acquirer_industry),
        csvEscape(r.url),
        r.agreement_uuid,
      ].join(","),
  },
};

export const SEARCH_CONFIGS = {
  sections: SECTIONS_SEARCH_CONFIG,
  tax: TAX_CLAUSES_SEARCH_CONFIG,
  transactions: DEALS_SEARCH_CONFIG,
} as const satisfies Record<SearchMode, SearchEndpointConfig<never>>;
