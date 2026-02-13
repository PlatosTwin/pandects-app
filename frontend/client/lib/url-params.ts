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
    searchFilters.transaction_price_total &&
    searchFilters.transaction_price_total.length > 0
  ) {
    searchFilters.transaction_price_total.forEach((value) =>
      params.append("transaction_price_total", value),
    );
  }

  if (
    searchFilters.transaction_price_stock &&
    searchFilters.transaction_price_stock.length > 0
  ) {
    searchFilters.transaction_price_stock.forEach((value) =>
      params.append("transaction_price_stock", value),
    );
  }

  if (
    searchFilters.transaction_price_cash &&
    searchFilters.transaction_price_cash.length > 0
  ) {
    searchFilters.transaction_price_cash.forEach((value) =>
      params.append("transaction_price_cash", value),
    );
  }

  if (
    searchFilters.transaction_price_assets &&
    searchFilters.transaction_price_assets.length > 0
  ) {
    searchFilters.transaction_price_assets.forEach((value) =>
      params.append("transaction_price_assets", value),
    );
  }

  // New filters from DB definition
  if (
    searchFilters.transaction_consideration &&
    searchFilters.transaction_consideration.length > 0
  ) {
    searchFilters.transaction_consideration.forEach((value) =>
      params.append("transaction_consideration", value),
    );
  }

  if (searchFilters.target_type && searchFilters.target_type.length > 0) {
    searchFilters.target_type.forEach((type) =>
      params.append("target_type", type),
    );
  }

  if (searchFilters.acquirer_type && searchFilters.acquirer_type.length > 0) {
    searchFilters.acquirer_type.forEach((type) =>
      params.append("acquirer_type", type),
    );
  }

  if (searchFilters.target_industry && searchFilters.target_industry.length > 0) {
    searchFilters.target_industry.forEach((industry) =>
      params.append("target_industry", industry),
    );
  }

  if (
    searchFilters.acquirer_industry &&
    searchFilters.acquirer_industry.length > 0
  ) {
    searchFilters.acquirer_industry.forEach((industry) =>
      params.append("acquirer_industry", industry),
    );
  }

  if (searchFilters.deal_status && searchFilters.deal_status.length > 0) {
    searchFilters.deal_status.forEach((status) =>
      params.append("deal_status", status),
    );
  }

  if (searchFilters.attitude && searchFilters.attitude.length > 0) {
    searchFilters.attitude.forEach((attitude) =>
      params.append("attitude", attitude),
    );
  }

  if (searchFilters.deal_type && searchFilters.deal_type.length > 0) {
    searchFilters.deal_type.forEach((type) => params.append("deal_type", type));
  }

  if (searchFilters.purpose && searchFilters.purpose.length > 0) {
    searchFilters.purpose.forEach((purpose) =>
      params.append("purpose", purpose),
    );
  }

  if (searchFilters.target_pe && searchFilters.target_pe.length > 0) {
    searchFilters.target_pe.forEach((pe) => params.append("target_pe", pe));
  }

  if (searchFilters.acquirer_pe && searchFilters.acquirer_pe.length > 0) {
    searchFilters.acquirer_pe.forEach((pe) => params.append("acquirer_pe", pe));
  }

  // Text filters (single values, not arrays)
  if (searchFilters.agreement_uuid && searchFilters.agreement_uuid.trim()) {
    params.append("agreement_uuid", searchFilters.agreement_uuid.trim());
  }

  if (searchFilters.section_uuid && searchFilters.section_uuid.trim()) {
    params.append("section_uuid", searchFilters.section_uuid.trim());
  }

  void clauseTypesNested;
  // Send selected taxonomy IDs directly.
  if (searchFilters.clauseType && searchFilters.clauseType.length > 0) {
    searchFilters.clauseType.forEach((standard_id) =>
      params.append("standard_id", standard_id),
    );
  }

  // Handle pagination if requested
  if (includePagination) {
    if (searchFilters.page)
      params.append("page", searchFilters.page.toString());
    if (searchFilters.page_size)
      params.append("page_size", searchFilters.page_size.toString());
  }

  return params;
};
