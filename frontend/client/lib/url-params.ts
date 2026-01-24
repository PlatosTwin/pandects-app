import { SearchFilters } from "@shared/search";
import type { ClauseTypeTree } from "@/lib/clause-types";

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

  // Transaction price filters
  if (
    searchFilters.transactionPriceTotal &&
    searchFilters.transactionPriceTotal.length > 0
  ) {
    searchFilters.transactionPriceTotal.forEach((value) =>
      params.append("transactionPriceTotal", value),
    );
  }

  if (
    searchFilters.transactionPriceStock &&
    searchFilters.transactionPriceStock.length > 0
  ) {
    searchFilters.transactionPriceStock.forEach((value) =>
      params.append("transactionPriceStock", value),
    );
  }

  if (
    searchFilters.transactionPriceCash &&
    searchFilters.transactionPriceCash.length > 0
  ) {
    searchFilters.transactionPriceCash.forEach((value) =>
      params.append("transactionPriceCash", value),
    );
  }

  if (
    searchFilters.transactionPriceAssets &&
    searchFilters.transactionPriceAssets.length > 0
  ) {
    searchFilters.transactionPriceAssets.forEach((value) =>
      params.append("transactionPriceAssets", value),
    );
  }

  // New filters from DB definition
  if (
    searchFilters.transactionConsideration &&
    searchFilters.transactionConsideration.length > 0
  ) {
    searchFilters.transactionConsideration.forEach((value) =>
      params.append("transactionConsideration", value),
    );
  }

  if (searchFilters.targetType && searchFilters.targetType.length > 0) {
    searchFilters.targetType.forEach((type) =>
      params.append("targetType", type),
    );
  }

  if (searchFilters.acquirerType && searchFilters.acquirerType.length > 0) {
    searchFilters.acquirerType.forEach((type) =>
      params.append("acquirerType", type),
    );
  }

  if (searchFilters.targetIndustry && searchFilters.targetIndustry.length > 0) {
    searchFilters.targetIndustry.forEach((industry) =>
      params.append("targetIndustry", industry),
    );
  }

  if (
    searchFilters.acquirerIndustry &&
    searchFilters.acquirerIndustry.length > 0
  ) {
    searchFilters.acquirerIndustry.forEach((industry) =>
      params.append("acquirerIndustry", industry),
    );
  }

  if (searchFilters.dealStatus && searchFilters.dealStatus.length > 0) {
    searchFilters.dealStatus.forEach((status) =>
      params.append("dealStatus", status),
    );
  }

  if (searchFilters.attitude && searchFilters.attitude.length > 0) {
    searchFilters.attitude.forEach((attitude) =>
      params.append("attitude", attitude),
    );
  }

  if (searchFilters.dealType && searchFilters.dealType.length > 0) {
    searchFilters.dealType.forEach((type) => params.append("dealType", type));
  }

  if (searchFilters.purpose && searchFilters.purpose.length > 0) {
    searchFilters.purpose.forEach((purpose) =>
      params.append("purpose", purpose),
    );
  }

  if (searchFilters.targetPe && searchFilters.targetPe.length > 0) {
    searchFilters.targetPe.forEach((pe) => params.append("targetPe", pe));
  }

  if (searchFilters.acquirerPe && searchFilters.acquirerPe.length > 0) {
    searchFilters.acquirerPe.forEach((pe) => params.append("acquirerPe", pe));
  }

  void clauseTypesNested;
  // Send selected taxonomy IDs directly.
  if (searchFilters.clauseType && searchFilters.clauseType.length > 0) {
    searchFilters.clauseType.forEach((standardId) =>
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
