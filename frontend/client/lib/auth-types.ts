export interface AuthUser {
  id: string;
  email: string;
  created_at?: string;
}

export interface ApiKeySummary {
  id: string;
  name: string | null;
  prefix: string;
  created_at: string;
  last_used_at: string | null;
  revoked_at: string | null;
}

export interface UsageByDay {
  day: string;
  count: number;
}

export interface ExternalSubjectLink {
  id: number;
  issuer: string;
  subject: string;
  created_at: string;
  provider?: string;
}

export type UsagePeriod = "1w" | "1m" | "1y" | "all";
