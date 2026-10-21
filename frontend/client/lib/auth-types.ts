// Response types are generated from the backend spec (shared/api.gen.ts).
import type { components } from "@shared/api.gen";

type Schemas = components["schemas"];

export type AuthUser = Schemas["AuthUser"];

export type ApiKeySummary = Schemas["AuthApiKeySummary"];

export type UsageByDay = Schemas["AuthUsageDay"];

/** A row of `GET /v1/auth/external-subjects`, or the `link` of a link response. */
export type ExternalSubjectLink = Schemas["AuthExternalSubject"] &
  Partial<Pick<Schemas["AuthExternalLink"], "provider">>;

export type UsagePeriod = "1w" | "1m" | "1y" | "all";
