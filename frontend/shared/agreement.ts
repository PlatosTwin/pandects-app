// Agreement Types

export interface Agreement {
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
  xml: string;
  url?: string;
  is_redacted?: boolean;
}

// Table of Contents types for XML navigation
export interface TOCItem {
  id: string;
  title: string;
  level: number;
  sectionUuid?: string;
  anchorId?: string;
  children?: TOCItem[];
}
