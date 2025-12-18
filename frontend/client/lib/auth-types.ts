export interface AuthUser {
  id: string;
  email: string;
  createdAt?: string;
}

export interface ApiKeySummary {
  id: string;
  name: string | null;
  prefix: string;
  createdAt: string;
  lastUsedAt: string | null;
  revokedAt: string | null;
}

export interface UsageByDay {
  day: string;
  count: number;
}

