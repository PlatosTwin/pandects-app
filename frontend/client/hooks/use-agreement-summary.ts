import { useQuery } from "@tanstack/react-query";
import { apiGet, errorMessage } from "@/lib/api/client";
import { IS_SERVER_RENDER } from "@/lib/query-client";
import { keys } from "@/lib/query-keys";

export type AgreementSummary = {
  /** Every agreement ingested, including those pending processing or verification. */
  agreements: number;
  /** Agreements processed and searchable (the rows behind `v1/agreements-index`). */
  processed_agreements?: number | null;
  sections: number;
  pages: number;
  latest_filing_date: string | null;
};

function fetchAgreementSummary(signal?: AbortSignal): Promise<AgreementSummary> {
  return apiGet<AgreementSummary>("v1/agreements-summary", undefined, { signal });
}

export function useAgreementSummary() {
  const query = useQuery({
    queryKey: keys.agreements.summary,
    queryFn: ({ signal }) => fetchAgreementSummary(signal),
    enabled: !IS_SERVER_RENDER,
    staleTime: 5 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
  });

  return {
    summary: query.data ?? null,
    isLoading: query.isLoading,
    error: query.error
      ? errorMessage(query.error, "Unable to load agreement summary.")
      : null,
  };
}
