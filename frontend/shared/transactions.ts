export interface TransactionSearchMatchedSection {
  section_uuid: string;
  article_title: string | null;
  section_title: string | null;
  standard_id: string[];
  snippet: string | null;
}

export interface TransactionSearchResult {
  agreement_uuid: string;
  year: number | null;
  target: string | null;
  acquirer: string | null;
  filing_date: string | null;
  prob_filing: number | null;
  filing_company_name: string | null;
  filing_company_cik: string | null;
  form_type: string | null;
  exhibit_type: string | null;
  transaction_price_total: number | null;
  transaction_price_stock: number | null;
  transaction_price_cash: number | null;
  transaction_price_assets: number | null;
  transaction_consideration: string | null;
  target_type: string | null;
  acquirer_type: string | null;
  target_industry: string | null;
  acquirer_industry: string | null;
  announce_date: string | null;
  close_date: string | null;
  deal_status: string | null;
  attitude: string | null;
  deal_type: string | null;
  purpose: string | null;
  target_pe: boolean | null;
  acquirer_pe: boolean | null;
  url: string | null;
  match_count: number;
  matched_sections: TransactionSearchMatchedSection[];
}

export interface TransactionSearchResponse {
  results: TransactionSearchResult[];
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

export interface AgreementSectionIndexItem {
  section_uuid: string;
  article_title: string | null;
  section_title: string | null;
  article_order: number | null;
  section_order: number | null;
  standard_id: string[];
}

export interface AgreementSectionIndexResponse {
  agreement_uuid: string;
  results: AgreementSectionIndexItem[];
}
