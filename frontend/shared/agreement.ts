// Agreement Types
import type { components } from "./api.gen";

/** `GET /v1/agreements/{agreement_uuid}` (generated from the backend spec). */
export type Agreement = components["schemas"]["AgreementResponse"];

// Table of Contents types for XML navigation
export interface TOCItem {
  id: string;
  title: string;
  level: number;
  sectionUuid?: string;
  anchorId?: string;
  children?: TOCItem[];
}
