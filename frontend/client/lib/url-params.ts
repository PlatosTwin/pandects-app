import { SearchFilters } from "@shared/search";
import type { ClauseTypeTree } from "@/lib/clause-types";

/**
 * Utility function to extract standard IDs from nested clause type structure
 */
export const extractStandardIds = (
  clauseTypeTexts: string[],
  clauseTypesNested: ClauseTypeTree,
): string[] => {
  const standardIds: string[] = [];

  const searchInNested = (obj: ClauseTypeTree): void => {
    for (const [key, value] of Object.entries(obj)) {
      if (typeof value === "string") {
        // This is a leaf node - check if the key matches any selected clause type
        if (clauseTypeTexts.includes(key)) {
          standardIds.push(value);
        }
      } else if (typeof value === "object" && value) {
        searchInNested(value);
      }
    }
  };

  searchInNested(clauseTypesNested);
  return standardIds;
};

/**
 * Build URL search parameters from search filters
 */
export const buildSearchParams = (
  searchFilters: SearchFilters,
  clauseTypesNested?: ClauseTypeTree,
  includePagination = true,
): URLSearchParams => {
  const params = new URLSearchParams();

  // Handle array filters - append each value separately
  if (searchFilters.year && searchFilters.year.length > 0) {
    searchFilters.year.forEach((year) => params.append("year", year));
  }

  if (searchFilters.target && searchFilters.target.length > 0) {
    searchFilters.target.forEach((target) => params.append("target", target));
  }

  if (searchFilters.acquirer && searchFilters.acquirer.length > 0) {
    searchFilters.acquirer.forEach((acquirer) =>
      params.append("acquirer", acquirer),
    );
  }

  if (
    searchFilters.transactionSize &&
    searchFilters.transactionSize.length > 0
  ) {
    searchFilters.transactionSize.forEach((size) =>
      params.append("transactionSize", size),
    );
  }

  if (
    searchFilters.transactionType &&
    searchFilters.transactionType.length > 0
  ) {
    searchFilters.transactionType.forEach((type) =>
      params.append("transactionType", type),
    );
  }

  if (
    searchFilters.considerationType &&
    searchFilters.considerationType.length > 0
  ) {
    searchFilters.considerationType.forEach((type) =>
      params.append("considerationType", type),
    );
  }

  if (searchFilters.targetType && searchFilters.targetType.length > 0) {
    searchFilters.targetType.forEach((type) =>
      params.append("targetType", type),
    );
  }

  // Extract standard IDs from selected clause types and send them instead
  if (
    searchFilters.clauseType &&
    searchFilters.clauseType.length > 0 &&
    clauseTypesNested
  ) {
    const standardIds = extractStandardIds(
      searchFilters.clauseType,
      clauseTypesNested,
    );
    standardIds.forEach((standardId) =>
      params.append("standardId", standardId),
    );
  }

  // Handle pagination if requested
  if (includePagination) {
    if (searchFilters.page)
      params.append("page", searchFilters.page.toString());
    if (searchFilters.pageSize)
      params.append("pageSize", searchFilters.pageSize.toString());
  }

  return params;
};
