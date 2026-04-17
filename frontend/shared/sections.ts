// M&A Search Types

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

export interface SearchResult {
  id: string;
  year: string;
  target: string;
  acquirer: string;
  article_title: string;
  section_title: string;
  standard_id: string[];
  xml: string;
  section_uuid: string;
  agreement_uuid: string;
  verified: boolean;
  transaction_price_total?: string | null;
  deal_status?: string | null;
  deal_type?: string | null;
  purpose?: string | null;
}

export interface SearchResponse {
  results: SearchResult[];
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

// Filter options response from API
export interface FilterOptionsResponse {
  targets: string[];
  acquirers: string[];
  target_counsels: string[];
  acquirer_counsels: string[];
  target_industries: string[];
  acquirer_industries: string[];
  clause_types?: Record<string, unknown>;
}

// Filter options (to be populated from API)
export interface FilterOptions {
  years: string[];
  targets: string[];
  acquirers: string[];
  clauseTypes: string[];
}

// CSV Export format
export interface CSVRow {
  year: string;
  target: string;
  acquirer: string;
  article_title: string;
  section_title: string;
  text: string;
  section_uuid: string;
  agreement_uuid: string;
}
