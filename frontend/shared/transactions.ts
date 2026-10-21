// Agreement ("transactions") search and section-index types (generated from the backend spec).
import type { components } from "./api.gen";

type Schemas = components["schemas"];

/** A matched-section preview on a `GET /v1/search/agreements` result. */
export type TransactionSearchMatchedSection = Schemas["AgreementSearchMatchedSection"];

/** One `GET /v1/search/agreements` result. */
export type TransactionSearchResult = Schemas["AgreementSearchResult"] & {
  /** Not returned by the API yet; the deals table shows the column when present. */
  target_counsel?: string | null;
};

/** `GET /v1/search/agreements`. */
export type TransactionSearchResponse = Omit<Schemas["AgreementSearchResponse"], "results"> & {
  results: TransactionSearchResult[];
};

export type AgreementSectionIndexItem = Schemas["AgreementSectionIndexItem"];

/** `GET /v1/agreements/{agreement_uuid}/sections`. */
export type AgreementSectionIndexResponse = Schemas["AgreementSectionIndexResponse"];
