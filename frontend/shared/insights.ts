// Response types for the dataset overview, leaderboard and trends endpoints,
// generated from the backend spec (shared/api.gen.ts).
import type { components } from "./api.gen";

type Schemas = components["schemas"];

// v1/agreements-summary
export type AgreementsSummaryResponse = Schemas["AgreementsSummaryResponse"];

// v1/agreements-index
export type AgreementIndexRow = Schemas["AgreementIndexRow"];
export type AgreementIndexResponse = Schemas["AgreementIndexResponse"];

// v1/agreements-status-summary
export type AgreementStatusYearRow = Schemas["AgreementStatusYearRow"];
export type MetadataFieldCoverageRow = Schemas["MetadataFieldCoverageRow"];
export type AgreementStatusSummaryResponse = Schemas["AgreementStatusSummaryResponse"];

// v1/agreements-deal-types-summary
export type AgreementDealTypeYearRow = Schemas["AgreementDealTypeYearRow"];
export type AgreementDealTypeSummaryResponse = Schemas["AgreementDealTypeSummaryResponse"];

// v1/counsel-leaderboards
export type CounselLeaderboardYear = Schemas["CounselLeaderboardYear"];
export type CounselLeaderboardEntry = Schemas["CounselLeaderboardEntry"];
export type CounselLeaderboardSide = Schemas["CounselLeaderboardSide"];
export type CounselLeaderboardResponse = Schemas["CounselLeaderboardResponse"];

// v1/agreement-trends
export type OwnershipMixRow = Schemas["OwnershipMixRow"];
export type OwnershipDealSizeRow = Schemas["OwnershipDealSizeRow"];
export type BuyerTypeMatrixRow = Schemas["BuyerTypeMatrixRow"];
export type TargetIndustryByYearRow = Schemas["TargetIndustryByYearRow"];
export type IndustryPairingRow = Schemas["IndustryPairingRow"];
export type AgreementTrendsResponse = Schemas["AgreementTrendsResponse"];

// v1/dumps (one entry of the array)
export type DumpInfo = Schemas["DumpEntry"];
