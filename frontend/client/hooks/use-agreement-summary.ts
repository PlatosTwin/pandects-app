import { useQuery } from "@tanstack/react-query";
import { apiGet, errorMessage } from "@/lib/api/client";
import { IS_SERVER_RENDER } from "@/lib/query-client";
import { keys } from "@/lib/query-keys";

import type { AgreementsSummaryResponse } from "@shared/insights";

/** `v1/agreements-summary` (generated from the backend spec). */
export type AgreementSummary = AgreementsSummaryResponse;

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
