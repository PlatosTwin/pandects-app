// Response types for the dataset overview, leaderboard and trends endpoints.

// v1/agreements-index
export interface AgreementIndexRow {
  agreement_uuid: string;
  year: string | null;
  target: string | null;
  acquirer: string | null;
  consideration_type: string | null;
  total_consideration: number | null;
  verified: boolean;
}

export interface AgreementIndexResponse {
  results: AgreementIndexRow[];
  page: number;
  page_size: number;
  total_count: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
}

// v1/agreements-status-summary
export interface AgreementStatusYearRow {
  year: number;
  color: "green" | "yellow" | "red" | "gray";
  current_stage: string;
  count: number;
}

export interface MetadataFieldCoverageRow {
  field: string;
  label: string;
  ingested_eligible_agreements: number;
  ingested_covered_agreements: number;
  ingested_coverage_pct: number | null;
  processed_eligible_agreements: number;
  processed_covered_agreements: number;
  processed_coverage_pct: number | null;
  note: string | null;
}

export interface AgreementStatusSummaryResponse {
  years: AgreementStatusYearRow[];
  latest_filing_date: string | null;
  metadata_covered_agreements: number | null;
  metadata_coverage_pct: number | null;
  metadata_field_coverage: MetadataFieldCoverageRow[];
  taxonomy_covered_sections: number | null;
  taxonomy_coverage_pct: number | null;
}

// v1/agreements-deal-types-summary
export interface AgreementDealTypeYearRow {
  year: number;
  deal_type: string;
  count: number;
}

export interface AgreementDealTypeSummaryResponse {
  years: AgreementDealTypeYearRow[];
}

// v1/counsel-leaderboards
export interface CounselLeaderboardYear {
  year: number;
  deal_count: number;
  total_transaction_value: number;
}

export interface CounselLeaderboardEntry {
  counsel: string;
  deal_count: number;
  total_transaction_value: number;
  years: CounselLeaderboardYear[];
}

export interface CounselLeaderboardSide {
  annual: {
    year: number;
    top_by_count: CounselLeaderboardEntry[];
    top_by_value: CounselLeaderboardEntry[];
  }[];
  top_by_count: CounselLeaderboardEntry[];
  top_by_value: CounselLeaderboardEntry[];
}

export interface CounselLeaderboardResponse {
  buy_side: CounselLeaderboardSide;
  sell_side: CounselLeaderboardSide;
}

// v1/agreement-trends
export interface OwnershipMixRow {
  year: number;
  public_deal_count: number;
  private_deal_count: number;
  public_total_transaction_value: number;
  private_total_transaction_value: number;
}

export interface OwnershipDealSizeRow {
  year: number;
  public_deal_count: number;
  private_deal_count: number;
  public_p25_transaction_value: number | null;
  public_median_transaction_value: number | null;
  public_p75_transaction_value: number | null;
  private_p25_transaction_value: number | null;
  private_median_transaction_value: number | null;
  private_p75_transaction_value: number | null;
}

export interface BuyerTypeMatrixRow {
  target_bucket: string;
  buyer_bucket: string;
  deal_count: number;
  median_transaction_value: number | null;
}

export interface TargetIndustryByYearRow {
  year: number;
  industry: string;
  deal_count: number;
  total_transaction_value: number;
}

export interface IndustryPairingRow {
  target_industry: string;
  acquirer_industry: string;
  deal_count: number;
  total_transaction_value: number;
}

export interface AgreementTrendsResponse {
  ownership: {
    mix_by_year: OwnershipMixRow[];
    deal_size_by_year: OwnershipDealSizeRow[];
    buyer_type_matrix: BuyerTypeMatrixRow[];
  };
  industries: {
    target_industries_by_year: TargetIndustryByYearRow[];
    pairings: IndustryPairingRow[];
  };
}

// v1/dumps
export interface DumpInfo {
  manifest: string;
  sha256: string;
  sha256_url?: string;
  sql: string;
  timestamp: string;
  size_bytes?: number;
  warning?: string;
}
