import type { ClauseTypeNode, ClauseTypeTree } from "@/lib/clause-types";

type ClauseTypeValue = ClauseTypeTree[keyof ClauseTypeTree];

export type TaxonomyLevel3 = {
  label: string;
  id: string;
};

export type TaxonomyLevel2 = {
  label: string;
  id: string;
  children: TaxonomyLevel3[];
};

export type TaxonomyLevel1 = {
  label: string;
  id: string;
  children: TaxonomyLevel2[];
  l2Count: number;
  l3Count: number;
};

export type TaxonomySearchEntry = {
  id: string;
  l1: string;
  l2?: string;
  l3?: string;
  l1Id: string;
  l2Id?: string;
  l3Id?: string;
  l1Normalized: string;
  l2Normalized?: string;
  l3Normalized?: string;
  idNormalized: string;
};

export const normalizeForTaxonomySearch = (value: string) =>
  value.toLowerCase().replace(/[^a-z0-9]+/g, "");

const toNode = (value: ClauseTypeValue): ClauseTypeNode => value as ClauseTypeNode;

export const buildTaxonomyEntries = (tree: ClauseTypeTree): TaxonomyLevel1[] =>
  Object.entries(tree).map(([l1Label, l1Value]) => {
    const l1Node = toNode(l1Value);
    const l2Tree = l1Node.children ?? {};
    const l2Entries = Object.entries(l2Tree).map(([l2Label, l2Value]) => {
      const l2Node = toNode(l2Value);
      const l3Tree = l2Node.children ?? {};
      const l3Entries = Object.entries(l3Tree).map(([l3Label, l3Value]) => {
        const l3Node = toNode(l3Value);
        return { label: l3Label, id: l3Node.id };
      });
      return { label: l2Label, id: l2Node.id, children: l3Entries };
    });
    const l3Count = l2Entries.reduce(
      (sum, entry) => sum + entry.children.length,
      0,
    );
    return {
      label: l1Label,
      id: l1Node.id,
      children: l2Entries,
      l2Count: l2Entries.length,
      l3Count,
    };
  });

export const buildTaxonomySearchEntries = (
  taxonomyEntries: TaxonomyLevel1[],
): TaxonomySearchEntry[] => {
  const entries: TaxonomySearchEntry[] = [];
  taxonomyEntries.forEach((entry) => {
    entries.push({
      id: entry.id,
      l1: entry.label,
      l1Id: entry.id,
      l1Normalized: normalizeForTaxonomySearch(entry.label),
      idNormalized: normalizeForTaxonomySearch(entry.id),
    });
    entry.children.forEach((child) => {
      entries.push({
        id: child.id,
        l1: entry.label,
        l2: child.label,
        l1Id: entry.id,
        l2Id: child.id,
        l1Normalized: normalizeForTaxonomySearch(entry.label),
        l2Normalized: normalizeForTaxonomySearch(child.label),
        idNormalized: normalizeForTaxonomySearch(child.id),
      });
      child.children.forEach((leaf) => {
        entries.push({
          id: leaf.id,
          l1: entry.label,
          l2: child.label,
          l3: leaf.label,
          l1Id: entry.id,
          l2Id: child.id,
          l3Id: leaf.id,
          l1Normalized: normalizeForTaxonomySearch(entry.label),
          l2Normalized: normalizeForTaxonomySearch(child.label),
          l3Normalized: normalizeForTaxonomySearch(leaf.label),
          idNormalized: normalizeForTaxonomySearch(leaf.id),
        });
      });
    });
  });
  return entries;
};

export const filterTaxonomySearchEntries = (
  entries: TaxonomySearchEntry[],
  query: string,
): TaxonomySearchEntry[] => {
  const normalizedQuery = normalizeForTaxonomySearch(query);
  if (!normalizedQuery) {
    return [];
  }

  return entries.filter(
    (entry) =>
      entry.idNormalized.includes(normalizedQuery) ||
      entry.l1Normalized.includes(normalizedQuery) ||
      entry.l2Normalized?.includes(normalizedQuery) ||
      entry.l3Normalized?.includes(normalizedQuery),
  );
};
