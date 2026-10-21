import type { TaxClauseSearchResult } from "@shared/tax-clauses";
import { safeOriginPath } from "@/lib/agreement-links";

/** Where the compare page's Back goes when it has no originating search. */
export const DEFAULT_COMPARE_ORIGIN = "/search?mode=tax";

/** `/compare/tax`, carrying the originating search so Back restores it. */
export function buildCompareHref(fromSearch: string | null | undefined): string {
  const from = compareOriginSearch(fromSearch);
  return from === DEFAULT_COMPARE_ORIGIN
    ? "/compare/tax"
    : `/compare/tax?${new URLSearchParams({ from }).toString()}`;
}

/** The `from` search path when it is a safe `/search` path, else the tax search. */
export function compareOriginSearch(from: string | null | undefined): string {
  const safe = safeOriginPath(from);
  return safe && /^\/search(?:[?#]|$)/.test(safe) ? safe : DEFAULT_COMPARE_ORIGIN;
}

const STORAGE_KEY = "pandects.taxCompare.v1";

export const TAX_COMPARE_MIN = 2;
export const TAX_COMPARE_MAX = 50;

export function stashCompareClauses(clauses: TaxClauseSearchResult[]): void {
  sessionStorage.setItem(STORAGE_KEY, JSON.stringify(clauses));
}

export function loadCompareClauses(): TaxClauseSearchResult[] {
  const raw = sessionStorage.getItem(STORAGE_KEY);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed as TaxClauseSearchResult[];
  } catch {
    sessionStorage.removeItem(STORAGE_KEY);
    return [];
  }
}

export function clearCompareClauses(): void {
  sessionStorage.removeItem(STORAGE_KEY);
}
