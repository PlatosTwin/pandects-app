// M&A Search Types

export interface SearchFilters {
  year?: string;
  target?: string;
  acquirer?: string;
  clauseType?: string;
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
