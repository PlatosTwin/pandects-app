/**
 * Test fixture: a filter draft with every field set, and the exact query
 * strings the pre-refactor builders produced for it. Used by the URL and API
 * param specs to prove the formats did not change (deep links depend on it).
 */
import type { SearchQueryFilters } from "@shared/search";

export const FULL_FILTERS: SearchQueryFilters = {
  year: ["2021", "2020"],
  target: ["Target A"],
  acquirer: ["Acq & Co"],
  clauseType: ["s1", "s2"],
  transaction_price_total: ["100M-1B"],
  transaction_price_stock: ["0-100M"],
  transaction_price_cash: ["1B+"],
  transaction_price_assets: ["0-100M"],
  transaction_consideration: ["cash"],
  target_type: ["public"],
  acquirer_type: ["private"],
  target_counsel: ["Firm A"],
  acquirer_counsel: ["Firm B"],
  target_industry: ["51"],
  acquirer_industry: ["52"],
  deal_status: ["complete"],
  attitude: ["friendly"],
  deal_type: ["merger"],
  purpose: ["strategic"],
  target_pe: ["true"],
  acquirer_pe: ["false"],
  agreement_uuid: " a-uuid ",
  section_uuid: "s-uuid",
  page: 3,
  page_size: 50,
};

const FILTER_PARAMS =
  "year=2021&year=2020&target=Target+A&acquirer=Acq+%26+Co&transaction_price_total=100M-1B" +
  "&transaction_price_stock=0-100M&transaction_price_cash=1B%2B&transaction_price_assets=0-100M" +
  "&transaction_consideration=cash&target_type=public&acquirer_type=private&target_counsel=Firm+A" +
  "&acquirer_counsel=Firm+B&target_industry=51&acquirer_industry=52&deal_status=complete" +
  "&attitude=friendly&deal_type=merger&purpose=strategic&target_pe=true&acquirer_pe=false" +
  "&agreement_uuid=a-uuid&section_uuid=s-uuid";

/** Golden strings captured from the previous implementation. */
export const GOLDEN = {
  /** sortBy "target", sortDirection "asc". */
  sectionsApi:
    `${FILTER_PARAMS}&standard_id=s1&standard_id=s2&page=3&page_size=50` +
    "&metadata=transaction_price_total&metadata=deal_status&metadata=deal_type&metadata=purpose" +
    "&sort_by=target&sort_direction=asc",
  /** sortBy "target", sortDirection "asc", include_rep_warranty true. */
  taxApi:
    `${FILTER_PARAMS}&page=3&page_size=50&tax_standard_id=s1&tax_standard_id=s2` +
    "&include_rep_warranty=true&sort_by=target&sort_direction=asc",
  /** sortBy "target", sortDirection "asc". */
  dealsApi:
    `${FILTER_PARAMS}&standard_id=s1&standard_id=s2&page=3&page_size=50` +
    "&sort_by=target&sort_direction=asc",
  /** mode sections, sortBy "target", sortDirection "asc". */
  urlSections:
    `${FILTER_PARAMS}&standard_id=s1&standard_id=s2&page=3&page_size=50` +
    "&sort_by=target&sort_direction=asc",
  /** mode tax, default sort, include_rep_warranty true. */
  urlTax:
    `${FILTER_PARAMS}&standard_id=s1&standard_id=s2&page=3&page_size=50` +
    "&mode=tax&include_rep_warranty=true",
  /** mode transactions, sortBy "acquirer", sortDirection "desc". */
  urlDeals:
    `${FILTER_PARAMS}&standard_id=s1&standard_id=s2&page=3&page_size=50` +
    "&mode=transactions&sort_by=acquirer",
};
