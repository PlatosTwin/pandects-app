/**
 * Pure helpers for the Search page: facet groups and counts, section result
 * grouping, plain-text section bodies, taxonomy lookups, agreement links and
 * the recent-searches list. No React here, so each piece is unit-tested.
 */
import type { SearchResult } from "@shared/sections";
import type { SearchMode, SearchQueryFilters } from "@shared/search";
import type { ClauseTypeNode, ClauseTypeTree } from "@/lib/clause-types";
import { AVAILABLE_YEARS, TRANSACTION_PRICE_OPTIONS } from "@/lib/constants";
import { KEYWORD_SEARCH_SUPPORT } from "@/lib/search-configs";
import { decodeXmlEntities } from "@/lib/text-utils";

// ---------------------------------------------------------------------------
// Facet groups
// ---------------------------------------------------------------------------

export type FacetField = Exclude<keyof SearchQueryFilters, "page" | "page_size" | "standard_id">;

export type FacetGroupId = "tax" | "deal" | "parties" | "counsel" | "advanced";

export const PRICE_FIELDS = [
  "transaction_price_total",
  "transaction_price_cash",
  "transaction_price_stock",
  "transaction_price_assets",
] as const;
export type PriceField = (typeof PRICE_FIELDS)[number];

/** Fields each rail group owns. The clause type and keyword live in the query bar. */
export const FACET_GROUP_FIELDS: Record<FacetGroupId, readonly FacetField[]> = {
  tax: ["include_rep_warranty"],
  deal: [
    "year",
    ...PRICE_FIELDS,
    "deal_status",
    "deal_type",
    "transaction_consideration",
    "attitude",
    "purpose",
  ],
  parties: [
    "target",
    "acquirer",
    "target_industry",
    "acquirer_industry",
    "target_type",
    "acquirer_type",
    "target_pe",
    "acquirer_pe",
  ],
  counsel: ["target_counsel", "acquirer_counsel"],
  advanced: ["agreement_uuid", "section_uuid"],
};

/** Fields whose whole selection counts as one active filter (a range). */
const RANGE_FIELDS = new Set<FacetField>(["year", ...PRICE_FIELDS]);

/** Active-filter count for one field: ranges count once, lists count per value. */
export function countFieldActive(filters: SearchQueryFilters, field: FacetField): number {
  const value = filters[field];
  if (Array.isArray(value)) {
    if (value.length === 0) return 0;
    return RANGE_FIELDS.has(field) ? 1 : value.length;
  }
  if (typeof value === "string") return value.trim() ? 1 : 0;
  if (typeof value === "boolean") return value ? 1 : 0;
  return 0;
}

export function countActiveFilters(
  filters: SearchQueryFilters,
  fields: readonly FacetField[],
): number {
  return fields.reduce((sum, field) => sum + countFieldActive(filters, field), 0);
}

/** The groups shown in a mode (the tax-only group appears in Tax mode). */
export function facetGroupsForMode(mode: SearchMode): FacetGroupId[] {
  return mode === "tax"
    ? ["tax", "deal", "parties", "counsel", "advanced"]
    : ["deal", "parties", "counsel", "advanced"];
}

/** Count of every rail filter active in this mode (for the mobile "Filters (n)"). */
export function countRailFilters(filters: SearchQueryFilters, mode: SearchMode): number {
  return facetGroupsForMode(mode).reduce(
    (sum, group) => sum + countActiveFilters(filters, FACET_GROUP_FIELDS[group]),
    0,
  );
}

/** A patch that clears the given fields (arrays to [], text and flags removed). */
export function resetFieldsPatch(fields: readonly FacetField[]): Partial<SearchQueryFilters> {
  const patch: Record<string, unknown> = {};
  for (const field of fields) {
    if (field === "agreement_uuid" || field === "section_uuid" || field === "text_query") {
      patch[field] = undefined;
    } else if (field === "include_rep_warranty") {
      patch[field] = undefined;
    } else if (field === "text_match_mode") {
      patch[field] = undefined;
    } else {
      patch[field] = [];
    }
  }
  return patch as Partial<SearchQueryFilters>;
}

/**
 * The keyword this mode actually searches, or "". Modes without keyword
 * support keep it in the draft (and URL) so switching back restores it, but
 * it is not an active filter there.
 */
export function activeKeyword(filters: SearchQueryFilters, mode: SearchMode): string {
  return KEYWORD_SEARCH_SUPPORT[mode] ? (filters.text_query?.trim() ?? "") : "";
}

/** True when anything narrows the search (any facet, clause type, keyword or UUID). */
export function hasAnySearchFilter(filters: SearchQueryFilters, mode: SearchMode): boolean {
  if ((filters.clauseType?.length ?? 0) > 0) return true;
  if (activeKeyword(filters, mode)) return true;
  return facetGroupsForMode(mode).some(
    (group) => countActiveFilters(filters, FACET_GROUP_FIELDS[group]) > 0,
  );
}

/**
 * Filters to keep when the mode changes. Section and tax clause types come
 * from different taxonomies, so the clause-type selection is dropped when
 * moving into or out of Tax mode.
 */
export function filtersForModeSwitch(
  filters: SearchQueryFilters,
  from: SearchMode,
  to: SearchMode,
): SearchQueryFilters {
  if ((from === "tax") === (to === "tax")) return filters;
  return { ...filters, clauseType: [] };
}

// ---------------------------------------------------------------------------
// Ranges (years and deal-value buckets)
// ---------------------------------------------------------------------------

/** Newest first, like AVAILABLE_YEARS. */
export function yearsBetween(from: string, to: string): string[] {
  const lo = Math.min(Number(from), Number(to));
  const hi = Math.max(Number(from), Number(to));
  return AVAILABLE_YEARS.filter((year) => Number(year) >= lo && Number(year) <= hi);
}

export interface RangeSummary {
  from: string | null;
  to: string | null;
  /** False when the selection has gaps (e.g. an old link listing 2019 and 2021). */
  contiguous: boolean;
}

export function summarizeYearRange(years: readonly string[]): RangeSummary {
  if (years.length === 0) return { from: null, to: null, contiguous: true };
  const numbers = years.map(Number).filter(Number.isFinite).sort((a, b) => a - b);
  const from = String(numbers[0]);
  const to = String(numbers[numbers.length - 1]);
  const contiguous = new Set(numbers).size === Number(to) - Number(from) + 1;
  return { from, to, contiguous };
}

/** Buckets from `from` to `to` inclusive, in TRANSACTION_PRICE_OPTIONS order. */
export function bucketsBetween(from: string, to: string): string[] {
  const a = TRANSACTION_PRICE_OPTIONS.indexOf(from);
  const b = TRANSACTION_PRICE_OPTIONS.indexOf(to);
  if (a < 0 || b < 0) return [];
  return TRANSACTION_PRICE_OPTIONS.slice(Math.min(a, b), Math.max(a, b) + 1);
}

export function summarizeBucketRange(buckets: readonly string[]): RangeSummary {
  const indexes = buckets
    .map((bucket) => TRANSACTION_PRICE_OPTIONS.indexOf(bucket))
    .filter((index) => index >= 0)
    .sort((a, b) => a - b);
  if (indexes.length === 0) return { from: null, to: null, contiguous: true };
  const lo = indexes[0];
  const hi = indexes[indexes.length - 1];
  return {
    from: TRANSACTION_PRICE_OPTIONS[lo],
    to: TRANSACTION_PRICE_OPTIONS[hi],
    contiguous: new Set(indexes).size === hi - lo + 1,
  };
}

/** "$100M – $1B", "$20B+", "$0 – $100M". */
export function formatBucketRange(from: string, to: string): string {
  const lower = from.split(" - ")[0];
  const upper = to.includes("+") ? null : to.split(" - ")[1];
  const money = (value: string) => (value === "0" ? "$0" : `$${value}`);
  return upper ? `${money(lower)} – ${money(upper)}` : `${money(lower.replace("+", ""))}+`;
}

// ---------------------------------------------------------------------------
// Section text and taxonomy paths
// ---------------------------------------------------------------------------

/** Section XML to plain paragraphs: drops page markers, keeps text/definition blocks. */
export function sectionXmlToPlainText(xml: string | null | undefined): string {
  if (!xml) return "";
  const withoutPages = xml
    .replace(/<pageUUID>[\s\S]*?<\/pageUUID>/g, " ")
    .replace(/<page>[\s\S]*?<\/page>/g, " ");
  return withoutPages
    .split(/<\/(?:text|definition)>/i)
    .map((chunk) =>
      decodeXmlEntities(chunk.replace(/<[^>]+>/g, " "))
        .replace(/\s+/g, " ")
        .trim(),
    )
    .filter(Boolean)
    .join("\n");
}

const PAGE_MARKER_RE =
  /(?:\b\d{1,4}\s+)?\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/gi;

/** Drops page-break markers (page number + page UUID) that leak into API snippets. */
export function cleanSnippet(snippet: string): string {
  return snippet.replace(PAGE_MARKER_RE, " ").replace(/\s+/g, " ").trim();
}

export function isRedactedSectionXml(xml: string | null | undefined): boolean {
  return Boolean(xml && /redacted="true"/.test(xml));
}

const PLACEHOLDER_TEXT_RE =
  /^\[?\s*(?:intentionally\s+(?:omitted|deleted|left\s+blank)|reserved|omitted|deleted)\s*\.?\s*\]?\.?$/i;

/** No stored text, or only an "[Intentionally Omitted]" / "[Reserved]" placeholder. */
export function isEmptySectionText(text: string): boolean {
  const trimmed = text.trim();
  return trimmed.length === 0 || PLACEHOLDER_TEXT_RE.test(trimmed);
}

/** Drop repeated adjacent segments: "Other / Unclassified" x3 shows once. */
export function collapseTaxonomyPath(path: readonly string[] | undefined): string[] {
  const out: string[] = [];
  for (const raw of path ?? []) {
    const part = raw.trim();
    if (!part) continue;
    if (out.length > 0 && out[out.length - 1].toLowerCase() === part.toLowerCase()) continue;
    out.push(part);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Section result grouping
// ---------------------------------------------------------------------------

export interface SectionResultItem {
  result: SearchResult;
  /** 1-based position across pages (telemetry and accessible names). */
  position: number;
  text: string;
  redacted: boolean;
  empty: boolean;
}

export interface SectionResultGroup {
  key: string;
  agreementUuid: string;
  /** First section with text; null when every section in the run is empty. */
  lead: SectionResultItem | null;
  /** Further sections with text from the same agreement. */
  more: SectionResultItem[];
  /** Sections with no stored text (hidden behind a toggle). */
  empty: SectionResultItem[];
}

/**
 * Groups consecutive hits from the same agreement into one card, and pulls
 * sections with no stored text out of the main flow.
 */
export function groupSectionResults(
  results: readonly SearchResult[],
  startPosition = 1,
): SectionResultGroup[] {
  const groups: SectionResultGroup[] = [];
  results.forEach((result, index) => {
    const redacted = isRedactedSectionXml(result.xml);
    const text = redacted ? "" : sectionXmlToPlainText(result.xml);
    const item: SectionResultItem = {
      result,
      position: startPosition + index,
      text,
      redacted,
      empty: !redacted && isEmptySectionText(text),
    };
    const last = groups[groups.length - 1];
    const group =
      last && last.agreementUuid === result.agreement_uuid
        ? last
        : (() => {
            const created: SectionResultGroup = {
              key: result.id,
              agreementUuid: result.agreement_uuid,
              lead: null,
              more: [],
              empty: [],
            };
            groups.push(created);
            return created;
          })();
    if (item.empty) group.empty.push(item);
    else if (!group.lead) group.lead = item;
    else group.more.push(item);
  });
  return groups;
}

// ---------------------------------------------------------------------------
// Taxonomy lookups
// ---------------------------------------------------------------------------

type TreeValue = ClauseTypeTree[string];

function isNode(value: TreeValue): value is ClauseTypeNode {
  return typeof value === "object" && value !== null && Object.prototype.hasOwnProperty.call(value, "id");
}

/**
 * Ids of taxonomy entries whose label matches `pattern`, leaves first. Used
 * to find starter-query clause types by label instead of hardcoding ids.
 */
export function findClauseTypeIds(tree: ClauseTypeTree, pattern: RegExp): string[] {
  const leaves: string[] = [];
  const branches: string[] = [];
  const visit = (node: ClauseTypeTree) => {
    for (const [label, value] of Object.entries(node)) {
      if (typeof value === "string") {
        if (pattern.test(label)) leaves.push(value);
        continue;
      }
      if (isNode(value)) {
        const hasChildren = Boolean(value.children && Object.keys(value.children).length > 0);
        if (pattern.test(label)) (hasChildren ? branches : leaves).push(value.id);
        if (value.children) visit(value.children);
        continue;
      }
      visit(value);
    }
  };
  visit(tree);
  return [...leaves, ...branches];
}

// ---------------------------------------------------------------------------
// Recent searches (per-browser convenience; storage may be unavailable)
// ---------------------------------------------------------------------------

export interface RecentSearch {
  /** Query string of the search URL (no leading "?"). */
  search: string;
  label: string;
  mode: SearchMode;
}

export const RECENT_SEARCHES_KEY = "pandects.recentSearches.v1";
export const MAX_RECENT_SEARCHES = 5;

function isRecentSearch(value: unknown): value is RecentSearch {
  if (!value || typeof value !== "object") return false;
  const entry = value as Record<string, unknown>;
  return (
    typeof entry.search === "string" &&
    typeof entry.label === "string" &&
    (entry.mode === "sections" || entry.mode === "transactions" || entry.mode === "tax")
  );
}

export function loadRecentSearches(): RecentSearch[] {
  try {
    const raw = window.localStorage.getItem(RECENT_SEARCHES_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter(isRecentSearch).slice(0, MAX_RECENT_SEARCHES) : [];
  } catch {
    return [];
  }
}

/**
 * Puts `entry` first (deduplicated by URL) and returns the stored list. With
 * `replaceNewest`, the entry replaces the current first one: a quick series of
 * facet tweaks is one search being refined, not five searches.
 */
export function saveRecentSearch(
  entry: RecentSearch,
  { replaceNewest = false }: { replaceNewest?: boolean } = {},
): RecentSearch[] {
  const existing = loadRecentSearches();
  const rest = (replaceNewest ? existing.slice(1) : existing).filter((item) => item.search !== entry.search);
  const next = [entry, ...rest].slice(0, MAX_RECENT_SEARCHES);
  try {
    window.localStorage.setItem(RECENT_SEARCHES_KEY, JSON.stringify(next));
  } catch {
    // Storage unavailable (private mode, blocked): the list just isn't kept.
  }
  return next;
}

export const MODE_NOUNS: Record<SearchMode, { one: string; many: string; title: string }> = {
  sections: { one: "section", many: "sections", title: "Sections" },
  transactions: { one: "deal", many: "deals", title: "Deals" },
  tax: { one: "tax clause", many: "tax clauses", title: "Tax" },
};

/** Short label for a search: `Sections · "reverse termination fee" · Termination Fee +1 · 2 filters`. */
export function describeSearch(
  mode: SearchMode,
  filters: SearchQueryFilters,
  clauseTypeLabelById: Record<string, string>,
): string {
  const parts: string[] = [MODE_NOUNS[mode].title];
  const keyword = activeKeyword(filters, mode);
  if (keyword) parts.push(`“${keyword}”`);
  const clauseTypes = filters.clauseType ?? [];
  if (clauseTypes.length > 0) {
    const first = clauseTypeLabelById[clauseTypes[0]] ?? "Clause type";
    parts.push(clauseTypes.length > 1 ? `${first} +${clauseTypes.length - 1}` : first);
  }
  const railCount = countRailFilters(filters, mode);
  if (railCount > 0) parts.push(railCount === 1 ? "1 filter" : `${railCount} filters`);
  if (parts.length === 1) parts.push(mode === "transactions" ? "All deals" : "No filters");
  return parts.join(" · ");
}
