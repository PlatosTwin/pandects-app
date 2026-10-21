import type { ClauseTypeTree } from "@/lib/clause-types";
import { buildTaxonomyEntries } from "@/lib/taxonomy-search";
import { buildSearchStateParams, createEmptySearchFilters } from "@/lib/url-params";
import type { TaxonomyKind } from "@/lib/query-keys";

export type TaxonomyNodeLevel = 1 | 2 | 3;

export interface TaxonomyNodeInfo {
  id: string;
  label: string;
  level: TaxonomyNodeLevel;
  /** Ancestors, root first. */
  path: ReadonlyArray<{ id: string; label: string }>;
  /** Direct children, in tree order. */
  children: ReadonlyArray<{ id: string; label: string }>;
}

/** Every node of a taxonomy tree keyed by standard id. */
export function indexTaxonomyNodes(tree: ClauseTypeTree): Map<string, TaxonomyNodeInfo> {
  const index = new Map<string, TaxonomyNodeInfo>();
  for (const l1 of buildTaxonomyEntries(tree)) {
    index.set(l1.id, {
      id: l1.id,
      label: l1.label,
      level: 1,
      path: [],
      children: l1.children.map(({ id, label }) => ({ id, label })),
    });
    for (const l2 of l1.children) {
      index.set(l2.id, {
        id: l2.id,
        label: l2.label,
        level: 2,
        path: [{ id: l1.id, label: l1.label }],
        children: l2.children.map(({ id, label }) => ({ id, label })),
      });
      for (const l3 of l2.children) {
        index.set(l3.id, {
          id: l3.id,
          label: l3.label,
          level: 3,
          path: [
            { id: l1.id, label: l1.label },
            { id: l2.id, label: l2.label },
          ],
          children: [],
        });
      }
    }
  }
  return index;
}

/**
 * Search URL filtered to one taxonomy node. Search writes the clause-type
 * filter as `standard_id` in both modes; `mode=tax` selects tax clauses.
 */
export function taxonomySearchHref(kind: TaxonomyKind, standardId: string): string {
  const params = buildSearchStateParams({
    filters: { ...createEmptySearchFilters(), clauseType: [standardId] },
    mode: kind === "tax" ? "tax" : "sections",
    sortBy: null,
    sortDirection: "desc",
  });
  return `/search?${params.toString()}`;
}

/** Reader URL scrolled to one section. */
export function sectionReaderHref(agreementUuid: string, sectionUuid: string): string {
  return `/agreements/${encodeURIComponent(agreementUuid)}?focusSectionUuid=${encodeURIComponent(sectionUuid)}`;
}

/** Plain text of section XML (tags dropped, whitespace collapsed). */
export function plainTextFromXml(xml: string): string {
  let text: string;
  if (typeof DOMParser !== "undefined") {
    const parsed = new DOMParser().parseFromString(xml, "text/xml");
    text = parsed.getElementsByTagName("parsererror").length
      ? xml.replace(/<[^>]+>/g, " ")
      : (parsed.documentElement.textContent ?? "");
  } else {
    text = xml.replace(/<[^>]+>/g, " ");
  }
  return text.replace(/\s+/g, " ").trim();
}

/** First `maxChars` characters of `text`, cut at a word boundary. */
export function truncateAtWord(text: string, maxChars: number): string {
  if (text.length <= maxChars) return text;
  const cut = text.slice(0, maxChars);
  const lastSpace = cut.lastIndexOf(" ");
  return `${(lastSpace > maxChars * 0.6 ? cut.slice(0, lastSpace) : cut).trimEnd()}…`;
}

export const TAXONOMY_LEVEL_NAMES: Record<
  TaxonomyKind,
  Record<TaxonomyNodeLevel, string>
> = {
  main: { 1: "Category", 2: "Group", 3: "Section type" },
  tax: { 1: "Category", 2: "Group", 3: "Clause type" },
};

export function formatCount(count: number): string {
  return new Intl.NumberFormat("en-US").format(count);
}

export function countNoun(kind: TaxonomyKind, count: number): string {
  if (kind === "tax") return count === 1 ? "clause" : "clauses";
  return count === 1 ? "section" : "sections";
}
