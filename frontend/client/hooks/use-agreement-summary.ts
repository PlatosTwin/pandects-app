import { useQuery } from "@tanstack/react-query";
import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";
import { IS_SERVER_RENDER } from "@/lib/query-client";

export type AgreementSummary = {
  agreements: number;
  sections: number;
  pages: number;
  latest_filing_date: string | null;
};

async function fetchAgreementSummary(): Promise<AgreementSummary> {
  const res = await authFetch(apiUrl("v1/agreements-summary"));
  if (!res.ok) {
    throw new Error(`Summary request failed (${res.status})`);
  }
  return (await res.json()) as AgreementSummary;
}

export function useAgreementSummary() {
  const query = useQuery({
    queryKey: ["agreement-summary"] as const,
    queryFn: fetchAgreementSummary,
    enabled: !IS_SERVER_RENDER,
    staleTime: 5 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
  });

  return {
    summary: query.data ?? null,
    isLoading: query.isLoading,
    error: query.error
      ? query.error instanceof Error
        ? query.error.message
        : "Unable to load agreement summary."
      : null,
  };
}
