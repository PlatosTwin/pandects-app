import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";
import type { Agreement } from "@shared/agreement";

import type { SectionDetails } from "./types";

export function fetchAgreementMetadata(
  agreementUuid: string,
): Promise<Agreement | null> {
  return authFetch(apiUrl(`v1/agreements/${agreementUuid}`))
    .then((r) => (r.ok ? (r.json() as Promise<Agreement>) : null))
    .catch(() => null);
}

export function fetchSectionDetails(
  sectionUuid: string,
): Promise<SectionDetails | null> {
  return authFetch(apiUrl(`v1/sections/${sectionUuid}`))
    .then((r) => (r.ok ? (r.json() as Promise<SectionDetails>) : null))
    .catch(() => null);
}
