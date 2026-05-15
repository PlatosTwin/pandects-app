import { useCallback, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Agreement } from "@shared/agreement";
import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";
import { logger } from "@/lib/logger";
import { keys } from "@/lib/query-keys";

async function fetchAgreementApi(
  agreementUuid: string,
  focusSectionUuid?: string,
): Promise<Agreement> {
  const params = new URLSearchParams();
  if (focusSectionUuid) {
    params.set("focus_section_uuid", focusSectionUuid);
    params.set("neighbor_sections", "1");
  }
  const suffix = params.toString() ? `?${params.toString()}` : "";
  const response = await authFetch(
    apiUrl(`v1/agreements/${agreementUuid}${suffix}`),
  );
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  return (await response.json()) as Agreement;
}

/**
 * Imperative agreement fetcher with cache-backed deduplication.
 *
 * Callers (AgreementModal, AgreementReader) drive fetches from useEffects keyed
 * on uuid/focus. We keep that imperative shape but route through React Query's
 * cache so repeat fetches of the same (uuid, focus) pair are deduplicated.
 */
export function useAgreement() {
  const queryClient = useQueryClient();
  const [agreement, setAgreement] = useState<Agreement | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAgreement = useCallback(
    async (agreement_uuid: string, focus_section_uuid?: string) => {
      setIsLoading(true);
      setError(null);
      try {
        const cacheKey = focus_section_uuid
          ? ([
              ...keys.agreement.detail(agreement_uuid),
              "focus",
              focus_section_uuid,
            ] as const)
          : keys.agreement.detail(agreement_uuid);

        const data = await queryClient.fetchQuery({
          queryKey: cacheKey,
          queryFn: () => fetchAgreementApi(agreement_uuid, focus_section_uuid),
          staleTime: 5 * 60 * 1000,
        });
        setAgreement(data);
      } catch (err) {
        logger.error("Failed to fetch agreement:", err);
        setError(err instanceof Error ? err.message : "Failed to load agreement");
      } finally {
        setIsLoading(false);
      }
    },
    [queryClient],
  );

  const clearAgreement = useCallback(() => {
    setAgreement(null);
    setError(null);
  }, []);

  return {
    agreement,
    isLoading,
    error,
    fetchAgreement,
    clearAgreement,
  };
}
