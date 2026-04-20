// Tax clause precedent search types

export type TaxClauseContextType = "operative" | "rep_warranty";

export interface TaxClauseSearchResult {
  id: string;
  clause_uuid: string;
  agreement_uuid: string;
  section_uuid: string;
  clause_text: string;
  anchor_label: string | null;
  context_type: TaxClauseContextType;
  source_method: string | null;
  tax_standard_ids: string[];
  year: string;
  target: string;
  acquirer: string;
  verified: boolean;
  transaction_price_total?: string | null;
  transaction_consideration?: string | null;
  deal_status?: string | null;
  deal_type?: string | null;
  target_counsel?: string | null;
  acquirer_counsel?: string | null;
}

export interface TaxClauseSearchResponse {
  results: TaxClauseSearchResult[];
  access: {
    tier: "anonymous" | "user" | "api_key";
    message?: string | null;
  };
  total_count: number;
  total_count_is_approximate: boolean;
  page: number;
  page_size: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
  next_num: number | null;
  prev_num: number | null;
}
