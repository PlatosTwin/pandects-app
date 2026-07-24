import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";

import type { AgreementMetadata, SectionDetails } from "./types";

/** Server-side cap on agreement_uuids per request. */
const METADATA_BATCH_LIMIT = 500;

/**
 * Batch-fetches agreement metadata for the favorites page in one request per
 * 500 uuids (instead of one request per favorite). Uuids the server omits
 * (ineligible or unknown) and failed chunks resolve to null so callers can
 * render the "Details unavailable" state.
 */
export async function fetchAgreementMetadataBatch(
  agreementUuids: string[],
): Promise<Record<string, AgreementMetadata | null>> {
  const unique = Array.from(new Set(agreementUuids.filter(Boolean)));
  const out: Record<string, AgreementMetadata | null> = {};
  for (let i = 0; i < unique.length; i += METADATA_BATCH_LIMIT) {
    const chunk = unique.slice(i, i + METADATA_BATCH_LIMIT);
    let agreements: Record<string, AgreementMetadata> = {};
    try {
      const res = await authFetch(
        apiUrl("v1/me/favorites/agreement-metadata"),
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ agreement_uuids: chunk }),
        },
      );
      if (res.ok) {
        const body = (await res.json()) as {
          agreements: Record<string, AgreementMetadata>;
        };
        agreements = body.agreements;
      }
    } catch {
      // fall through: chunk resolves to nulls
    }
    for (const uuid of chunk) {
      out[uuid] = agreements[uuid] ?? null;
    }
  }
  return out;
}

/** Resolves to null when the fetch fails; callers treat null as "unavailable". */
export function fetchSectionDetails(
  sectionUuid: string,
): Promise<SectionDetails | null> {
  return authFetch(apiUrl(`v1/sections/${sectionUuid}`))
    .then((r) => (r.ok ? (r.json() as Promise<SectionDetails>) : null))
    .catch(() => null);
}
