import type { TaxClauseSearchResult } from "@shared/tax-clauses";

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
