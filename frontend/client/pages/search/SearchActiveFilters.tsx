import { X } from "lucide-react";
import { formatIndustry } from "@/lib/naics";
import { formatFilterOption } from "@/lib/text-utils";
import type { SearchMode, SearchQueryFilters } from "@shared/search";
import {
  PRICE_FIELDS,
  activeKeyword,
  formatBucketRange,
  summarizeBucketRange,
  summarizeYearRange,
  type FacetField,
} from "./search-ui";
import { TEXT_MATCH_MODE_LABELS } from "./keyword-query";

const LIST_FIELDS: ReadonlyArray<readonly [FacetField, string]> = [
  ["target", "Target"],
  ["acquirer", "Acquirer"],
  ["deal_status", "Status"],
  ["deal_type", "Deal type"],
  ["transaction_consideration", "Consideration"],
  ["attitude", "Attitude"],
  ["purpose", "Purpose"],
  ["target_type", "Target type"],
  ["acquirer_type", "Acquirer type"],
  ["target_industry", "Target industry"],
  ["acquirer_industry", "Acquirer industry"],
  ["target_pe", "Target PE"],
  ["acquirer_pe", "Acquirer PE"],
  ["target_counsel", "Target counsel"],
  ["acquirer_counsel", "Acquirer counsel"],
];

const ENUM_FIELDS = new Set<FacetField>([
  "deal_status",
  "deal_type",
  "transaction_consideration",
  "attitude",
  "purpose",
  "target_type",
  "acquirer_type",
]);

const PRICE_LABELS: Record<(typeof PRICE_FIELDS)[number], string> = {
  transaction_price_total: "Value",
  transaction_price_cash: "Cash",
  transaction_price_stock: "Stock",
  transaction_price_assets: "Assets",
};

export interface ActiveChip {
  key: string;
  label: string;
  value: string;
  /** Patch that removes this chip's filter. */
  remove: Partial<SearchQueryFilters>;
}

/** Every active filter as a removable chip, in a stable order. */
export function useActiveChips(
  filters: SearchQueryFilters,
  mode: SearchMode,
  clauseTypeLabelById: Record<string, string>,
  /** NAICS label per industry code (`industry_labels` from filter options). */
  industryLabels: Record<string, string> = {},
): ActiveChip[] {
  const chips: ActiveChip[] = [];
  const keyword = activeKeyword(filters, mode);
  if (keyword) {
    const match = filters.text_match_mode && filters.text_match_mode !== "phrase"
      ? ` (${TEXT_MATCH_MODE_LABELS[filters.text_match_mode].toLowerCase()})`
      : "";
    chips.push({
      key: "q",
      label: "Keyword",
      value: `“${keyword}”${match}`,
      remove: { text_query: undefined, text_match_mode: undefined },
    });
  }
  for (const id of filters.clauseType ?? []) {
    chips.push({
      key: `clause:${id}`,
      label: mode === "tax" ? "Tax clause type" : "Clause type",
      value: clauseTypeLabelById[id] ?? "Unknown clause type",
      remove: { clauseType: (filters.clauseType ?? []).filter((value) => value !== id) },
    });
  }
  const years = filters.year ?? [];
  if (years.length > 0) {
    const range = summarizeYearRange(years);
    chips.push({
      key: "year",
      label: "Year",
      value: !range.contiguous
        ? years.slice().sort().join(", ")
        : range.from === range.to
          ? (range.from ?? "")
          : `${range.from}–${range.to}`,
      remove: { year: [] },
    });
  }
  for (const field of PRICE_FIELDS) {
    const buckets = filters[field] ?? [];
    if (buckets.length === 0) continue;
    const range = summarizeBucketRange(buckets);
    chips.push({
      key: field,
      label: PRICE_LABELS[field],
      value:
        range.from && range.to && range.contiguous
          ? formatBucketRange(range.from, range.to)
          : `${buckets.length} bands`,
      remove: { [field]: [] },
    });
  }
  for (const [field, label] of LIST_FIELDS) {
    const values = (filters[field] as string[] | undefined) ?? [];
    for (const value of values) {
      const display =
        field === "target_industry" || field === "acquirer_industry"
          ? (formatIndustry(value, industryLabels[value]) ?? value)
          : field === "target_pe" || field === "acquirer_pe"
            ? value === "true"
              ? "Yes"
              : "No"
            : ENUM_FIELDS.has(field)
              ? formatFilterOption(value)
              : value;
      chips.push({
        key: `${field}:${value}`,
        label,
        value: display,
        remove: { [field]: values.filter((item) => item !== value) },
      });
    }
  }
  if (filters.agreement_uuid) {
    chips.push({ key: "agreement_uuid", label: "Agreement", value: filters.agreement_uuid, remove: { agreement_uuid: undefined } });
  }
  if (filters.section_uuid) {
    chips.push({ key: "section_uuid", label: "Section", value: filters.section_uuid, remove: { section_uuid: undefined } });
  }
  if (mode === "tax" && filters.include_rep_warranty) {
    chips.push({
      key: "include_rep_warranty",
      label: "Includes",
      value: "Reps & warranties",
      remove: { include_rep_warranty: undefined },
    });
  }
  return chips;
}

export function SearchActiveFilters({
  chips,
  onRemove,
  onClearAll,
}: {
  chips: ActiveChip[];
  onRemove: (patch: Partial<SearchQueryFilters>) => void;
  onClearAll: () => void;
}) {
  if (chips.length === 0) return null;
  return (
    <ul aria-label="Active filters" className="flex min-w-0 flex-wrap items-center gap-1.5">
      {chips.map((chip) => (
        <li key={chip.key} className="min-w-0 max-w-full">
          <span className="inline-flex max-w-full items-center gap-1 rounded-md border border-border bg-background py-0.5 pl-2 pr-0.5 text-xs">
            <span className="shrink-0 text-muted-foreground">{chip.label}:</span>
            <span className="min-w-0 truncate text-foreground" title={chip.value}>
              {chip.value}
            </span>
            <button
              type="button"
              onClick={() => onRemove(chip.remove)}
              className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-sm text-muted-foreground hover:bg-accent/60 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              aria-label={`Remove ${chip.label} filter: ${chip.value}`}
            >
              <X className="h-3 w-3" aria-hidden="true" />
            </button>
          </span>
        </li>
      ))}
      <li>
        <button
          type="button"
          onClick={onClearAll}
          className="rounded-md px-2 py-1 text-xs font-medium text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          Clear all
        </button>
      </li>
    </ul>
  );
}
