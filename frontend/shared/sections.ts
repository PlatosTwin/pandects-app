// M&A Search Types
import type { components } from "./api.gen";

type Schemas = components["schemas"];

export interface SearchFilters {
  year?: string[];
  target?: string[];
  acquirer?: string[];
  clauseType?: string[];
  standard_id?: string[];
  transaction_price_total?: string[];
  transaction_price_stock?: string[];
  transaction_price_cash?: string[];
  transaction_price_assets?: string[];
  transaction_consideration?: string[];
  target_type?: string[];
  acquirer_type?: string[];
  target_counsel?: string[];
  acquirer_counsel?: string[];
  target_industry?: string[];
  acquirer_industry?: string[];
  deal_status?: string[];
  attitude?: string[];
  deal_type?: string[];
  purpose?: string[];
  target_pe?: string[];
  acquirer_pe?: string[];
  agreement_uuid?: string;
  section_uuid?: string;
  page?: number;
  page_size?: number;
}

// Response types below are generated from the backend spec (shared/api.gen.ts).

/** One `GET /v1/sections` result. */
export type SearchResult = Schemas["SectionItem"];

/** Deal fields requested via `metadata=` (see SECTION_RESULT_METADATA_FIELDS). */
export type SectionResultMetadata = Schemas["SectionsResultMetadata"];

/** `GET /v1/sections`. */
export type SearchResponse = Schemas["SectionsResponse"];

/**
 * `GET /v1/filter-options`. Each list is present only when requested through
 * `fields` (default: all). `clause_types` is deprecated: read `GET /v1/taxonomy`
 * through `useTaxonomy`; `useFilterOptions` already serves `clause_types` from
 * that cache.
 */
export type FilterOptionsResponse = Schemas["FilterOptionsResponse"];
