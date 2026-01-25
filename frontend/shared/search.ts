// M&A Search Types

export interface SearchFilters {
  year?: string[];
  target?: string[];
  acquirer?: string[];
  clauseType?: string[];
  standardId?: string[];
  // Transaction price filters (disabled for now)
  transactionPriceTotal?: string[];
  transactionPriceStock?: string[];
  transactionPriceCash?: string[];
  transactionPriceAssets?: string[];
  // New filters from DB definition
  transactionConsideration?: string[];
  targetType?: string[];
  acquirerType?: string[];
  targetIndustry?: string[];
  acquirerIndustry?: string[];
  dealStatus?: string[];
  attitude?: string[];
  dealType?: string[];
  purpose?: string[];
  targetPe?: string[];
  acquirerPe?: string[];
  agreementUuid?: string;
  sectionUuid?: string;
  page?: number;
  pageSize?: number;
}

export interface SearchResult {
  id: string;
  year: string;
  target: string;
  acquirer: string;
  articleTitle: string;
  sectionTitle: string;
  standardId: string[];
  xml: string;
  sectionUuid: string;
  agreementUuid: string;
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
  totalCount: number;
  page: number;
  pageSize: number;
  totalPages: number;
  hasNext: boolean;
  hasPrev: boolean;
  nextNum: number | null;
  prevNum: number | null;
}

// Filter options response from API
export interface FilterOptionsResponse {
  targets: string[];
  acquirers: string[];
  targetIndustries: string[];
  acquirerIndustries: string[];
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
  articleTitle: string;
  sectionTitle: string;
  text: string;
  sectionUuid: string;
  agreementUuid: string;
}
