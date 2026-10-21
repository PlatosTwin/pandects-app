// Tax clause precedent search types (generated from the backend spec).
import type { components } from "./api.gen";

type Schemas = components["schemas"];

/** One `GET /v1/tax-clauses` result. */
export type TaxClauseSearchResult = Schemas["TaxClauseItem"];

export type TaxClauseContextType = TaxClauseSearchResult["context_type"];

/** `GET /v1/tax-clauses`. */
export type TaxClauseSearchResponse = Schemas["TaxClausesResponse"];
