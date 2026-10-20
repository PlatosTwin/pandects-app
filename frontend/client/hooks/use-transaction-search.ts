import { useCallback, useState } from "react";
import { apiUrl } from "@/lib/api-config";
import { buildSearchParams } from "@/lib/url-params";
import { logger } from "@/lib/logger";
import {
  csvEscape,
  downloadCsvFile,
  fetchAllPagesForCsv,
  notifyCsvFailed,
  notifyCsvTruncated,
  type CsvExportRows,
} from "@/lib/csv-export";
import { keys } from "@/lib/query-keys";
import {
  formatCompactCurrencyValue,
  formatDateValue,
  formatEnumValue,
} from "@/lib/format-utils";
import type { ClauseTypeTree } from "@/lib/clause-types";
import type { SearchFilters } from "@shared/sections";
import type {
  TransactionSearchResponse,
  TransactionSearchResult,
} from "@shared/transactions";
import {
  useCommittedSearchCore,
  type CommittedQuery,
  type SortDirection,
  type SortField,
} from "@/hooks/use-committed-search";

const EMPTY_ACCESS: TransactionSearchResponse["access"] = { tier: "anonymous" };

function buildTransactionQueryString(
  committed: CommittedQuery<SearchFilters>,
): string {
  const params = buildSearchParams(committed.filters, committed.clauseTypesNested);
  if (committed.sortBy) {
    params.set("sort_by", committed.sortBy);
    params.set("sort_direction", committed.sortDirection);
  }
  return params.toString();
}

const TRANSACTIONS_SEARCH_CONFIG = {
  buildQueryString: buildTransactionQueryString,
  fetchUrl: (queryString: string) =>
    apiUrl(`v1/search/agreements?${queryString}`),
  queryKey: (params: { q: string; n: number }) =>
    keys.transactions.search(params),
  getResults: (response: TransactionSearchResponse) => response.results,
  getResultId: (result: TransactionSearchResult) => result.agreement_uuid,
  silentNotFound: false,
  logLabel: "Agreement search failed:",
};

/**
 * Declarative transaction (deal) search. Unlike `useTaxClauses` / `useSections`,
 * this hook does not own a filter draft — the parent passes filters in via
 * `performSearch(...)`. The hook only tracks the committed snapshot that drives
 * `useQuery`.
 */
export function useTransactionSearch() {
  const [hasSearched, setHasSearched] = useState(false);

  const core = useCommittedSearchCore<
    SearchFilters,
    TransactionSearchResponse,
    TransactionSearchResult
  >(TRANSACTIONS_SEARCH_CONFIG);
  const { committed, responseData } = core;

  const results = core.results;
  const totalCount = responseData?.total_count ?? 0;
  const totalCountIsApproximate = responseData?.total_count_is_approximate ?? false;
  const totalPages = responseData?.total_pages ?? 0;
  const hasNext = responseData?.has_next ?? false;
  const hasPrev = responseData?.has_prev ?? false;
  const access = responseData?.access ?? EMPTY_ACCESS;

  const performSearch = useCallback(
    async ({
      filters,
      clauseTypesNested,
      sortBy,
      sortDirection,
      markAsSearched = true,
    }: {
      filters: SearchFilters;
      clauseTypesNested?: ClauseTypeTree;
      sortBy: SortField | null;
      sortDirection: SortDirection;
      markAsSearched?: boolean;
    }) => {
      if (markAsSearched) setHasSearched(true);
      core.commit({
        filters,
        clauseTypesNested,
        sortBy,
        sortDirection,
      });
    },
    [core.commit],
  );

  const clear = useCallback(() => {
    core.uncommit();
    setHasSearched(false);
  }, [core.uncommit]);

  const downloadCSV = useCallback(async () => {
    const isSelection = core.selectedResults.size > 0;
    let rows: TransactionSearchResult[] = [];
    let exportedFrom: CsvExportRows<TransactionSearchResult> | null = null;

    if (isSelection) {
      rows = results.filter((r) => core.selectedResults.has(r.agreement_uuid));
    } else {
      // Export what the results show: the submitted query, not the sidebar draft.
      if (!committed) return;
      try {
        exportedFrom = await fetchAllPagesForCsv<TransactionSearchResult>(
          "v1/search/agreements",
          buildTransactionQueryString(committed),
        );
        rows = exportedFrom.rows;
      } catch (error) {
        logger.error("Failed to fetch all transactions for CSV:", error);
        notifyCsvFailed();
        return;
      }
    }

    if (rows.length === 0) return;

    downloadCsvFile(
      isSelection ? "ma_deals_selected" : "ma_deals",
      [
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
      rows.map((r) =>
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
      ),
    );

    if (exportedFrom?.truncated) {
      notifyCsvTruncated(
        rows.length,
        exportedFrom.totalCount,
        exportedFrom.totalCountIsApproximate,
      );
    }
  }, [committed, results, core.selectedResults]);

  return {
    isSearching: core.isSearching,
    results,
    hasSearched,
    totalCount,
    totalCountIsApproximate,
    totalPages,
    hasNext,
    hasPrev,
    access,
    showErrorModal: core.showErrorModal,
    errorMessage: core.errorMessage,
    selectedResults: core.selectedResults,
    performSearch,
    clear,
    closeErrorModal: core.closeErrorModal,
    toggleResultSelection: core.toggleResultSelection,
    toggleSelectAll: core.toggleSelectAll,
    clearSelection: core.clearSelection,
    downloadCSV,
  };
}
