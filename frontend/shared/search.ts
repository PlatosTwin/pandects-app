// M&A Search Types

export interface SearchFilters {
  year?: string[];
  target?: string[];
  acquirer?: string[];
  clauseType?: string[];
  standardId?: string[];
  transactionSize?: string[];
  transactionType?: string[];
  considerationType?: string[];
  targetType?: string[];
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
  xml: string;
  sectionUuid: string;
  agreementUuid: string;
}

export interface SearchResponse {
  results: SearchResult[];
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
