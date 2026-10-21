import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import type { Agreement } from "@shared/agreement";
import type {
  AgreementSectionIndexItem,
  AgreementSectionIndexResponse,
} from "@shared/transactions";
import { apiGet, errorMessage } from "@/lib/api/client";
import { logger } from "@/lib/logger";
import { IS_SERVER_RENDER } from "@/lib/query-client";
import { keys } from "@/lib/query-keys";

const AGREEMENT_STALE_TIME_MS = 5 * 60 * 1000;

export function fetchAgreement(
  agreementUuid: string,
  focusSectionUuid?: string | null,
  signal?: AbortSignal,
): Promise<Agreement> {
  return apiGet<Agreement>(
    `v1/agreements/${encodeURIComponent(agreementUuid)}`,
    focusSectionUuid
      ? { focus_section_uuid: focusSectionUuid, neighbor_sections: 1 }
      : undefined,
    { signal },
  );
}

/**
 * One agreement (metadata + XML). With `focusSectionUuid` the API also
 * returns the neighbors of that section; each (uuid, focus) pair is cached
 * separately.
 */
export function useAgreementQuery(
  agreementUuid: string | null | undefined,
  focusSectionUuid?: string | null,
) {
  const query = useQuery({
    queryKey: keys.agreement.detail(agreementUuid ?? "", focusSectionUuid),
    queryFn: ({ signal }) => fetchAgreement(agreementUuid!, focusSectionUuid, signal),
    enabled: !IS_SERVER_RENDER && Boolean(agreementUuid),
    staleTime: AGREEMENT_STALE_TIME_MS,
  });

  useEffect(() => {
    if (query.error) logger.error("Failed to fetch agreement:", query.error);
  }, [query.error]);

  return {
    agreement: query.data ?? null,
    // True until the first response for this (uuid, focus) arrives.
    isLoading: query.isPending,
    error: query.error ? errorMessage(query.error, "Failed to load agreement") : null,
    query,
  };
}

const EMPTY_SECTION_INDEX: AgreementSectionIndexItem[] = [];

/**
 * The agreement's section index (`v1/agreements/<uuid>/sections`), used for
 * the reader's table of contents and section-type filter.
 */
export function useAgreementSections(agreementUuid: string | null | undefined) {
  const query = useQuery({
    queryKey: keys.agreement.sections(agreementUuid ?? ""),
    queryFn: ({ signal }) =>
      apiGet<AgreementSectionIndexResponse>(
        `v1/agreements/${encodeURIComponent(agreementUuid!)}/sections`,
        undefined,
        { signal },
      ),
    enabled: !IS_SERVER_RENDER && Boolean(agreementUuid),
    staleTime: AGREEMENT_STALE_TIME_MS,
  });

  useEffect(() => {
    if (query.error) logger.error("Failed to fetch agreement sections index:", query.error);
  }, [query.error]);

  return {
    sections: query.data?.results ?? EMPTY_SECTION_INDEX,
    isLoading: query.isPending,
    error: query.error
      ? errorMessage(query.error, "Unable to load agreement sections.")
      : null,
    query,
  };
}
