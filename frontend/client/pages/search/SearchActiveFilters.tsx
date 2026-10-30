import { useEffect, useRef } from "react";
import { X } from "lucide-react";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
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
  /**
   * Multi-value filters (clause types, parties, enums): chips sharing a group
   * render as one chip with a "+N" list, and `removeGroup` clears them all.
   */
  group?: { key: string; removeGroup: Partial<SearchQueryFilters> };
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
      group: { key: "clauseType", removeGroup: { clauseType: [] } },
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
        group: { key: field, removeGroup: { [field]: [] } },
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

type ChipRun = { key: string; chips: ActiveChip[] };

/** Consecutive chips of one group collapse into a run; others stand alone. */
export function groupChips(chips: ActiveChip[]): ChipRun[] {
  const runs: ChipRun[] = [];
  for (const chip of chips) {
    const last = runs[runs.length - 1];
    if (chip.group && last && last.chips[0].group?.key === chip.group.key) {
      last.chips.push(chip);
    } else {
      runs.push({ key: chip.group?.key ?? chip.key, chips: [chip] });
    }
  }
  return runs;
}

const CHIP_CLASS =
  "inline-flex max-w-full items-center gap-1 rounded-md border border-border bg-background py-0.5 pl-2 pr-0.5 text-xs shadow-raised";
const CHIP_REMOVE_CLASS =
  "inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-sm text-muted-foreground transition-colors hover:bg-accent/60 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring";

function ChipRemoveButton({
  label,
  onClick,
  ...data
}: { label: string; onClick: () => void } & Record<`data-${string}`, string | boolean>) {
  return (
    <button type="button" onClick={onClick} className={CHIP_REMOVE_CLASS} aria-label={label} title={label} {...data}>
      <X className="h-3 w-3" aria-hidden="true" />
    </button>
  );
}

/** Where focus goes once a removal re-renders the chips. */
type FocusTarget = { popoverKey?: string; index: number; runIndex: number };
type RemoveChip = (patch: Partial<SearchQueryFilters>, focus: FocusTarget) => void;

function GroupedChip({
  chips,
  runKey,
  runIndex,
  onRemove,
}: {
  chips: ActiveChip[];
  runKey: string;
  runIndex: number;
  onRemove: RemoveChip;
}) {
  const [first] = chips;
  const more = chips.length - 1;
  return (
    <span className={CHIP_CLASS}>
      <span className="shrink-0 text-muted-foreground">{first.label}:</span>
      <span className="min-w-0 truncate text-foreground" title={first.value}>
        {first.value}
      </span>
      <Popover>
        <PopoverTrigger asChild>
          <button
            type="button"
            className="shrink-0 rounded-sm px-1 font-medium text-primary transition-colors hover:bg-accent/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            aria-label={`Show all ${chips.length} ${first.label} filters`}
          >
            +{more}
          </button>
        </PopoverTrigger>
        <PopoverContent
          align="start"
          className="w-80 max-w-[calc(100vw-2rem)] p-2"
          data-chip-popover={runKey}
        >
          <p className="px-2 pb-1 pt-0.5 text-xs font-medium text-muted-foreground">
            {first.label} ({chips.length})
          </p>
          <ul className="max-h-72 overflow-y-auto p-1">
            {chips.map((chip, index) => (
              <li
                key={chip.key}
                className="flex items-center justify-between gap-2 rounded-sm py-0.5 pl-2 text-xs hover:bg-accent/40"
              >
                <span className="min-w-0 text-foreground">{chip.value}</span>
                <ChipRemoveButton
                  label={`Remove ${chip.label} filter: ${chip.value}`}
                  onClick={() => onRemove(chip.remove, { popoverKey: runKey, index, runIndex })}
                  data-chip-remove
                />
              </li>
            ))}
          </ul>
        </PopoverContent>
      </Popover>
      <ChipRemoveButton
        label={`Remove all ${chips.length} ${first.label} filters`}
        onClick={() => onRemove(first.group?.removeGroup ?? first.remove, { index: 0, runIndex })}
        data-chip-remove-run
      />
    </span>
  );
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
  const listRef = useRef<HTMLUListElement>(null);
  const pendingFocus = useRef<FocusTarget | null>(null);
  const remove: RemoveChip = (patch, focus) => {
    pendingFocus.current = focus;
    onRemove(patch);
  };

  // A removed chip takes its focused button with it. Move focus to the next
  // value in the open list, else to the chip now in that place, else to
  // Clear all, else to the results, so keyboard users can keep removing.
  useEffect(() => {
    const target = pendingFocus.current;
    if (!target) return;
    pendingFocus.current = null;
    const frame = requestAnimationFrame(() => {
      const pick = (nodes: NodeListOf<HTMLElement> | undefined, index: number) =>
        nodes && nodes.length > 0 ? nodes[Math.min(index, nodes.length - 1)] : null;
      const next =
        (target.popoverKey
          ? pick(
              document.querySelectorAll<HTMLElement>(
                `[data-chip-popover="${target.popoverKey}"] [data-chip-remove]`,
              ),
              target.index,
            )
          : null) ??
        pick(listRef.current?.querySelectorAll<HTMLElement>("[data-chip-remove-run]"), target.runIndex) ??
        listRef.current?.querySelector<HTMLElement>("[data-chip-clear]") ??
        document.getElementById("search-results");
      next?.focus();
    });
    return () => cancelAnimationFrame(frame);
  }, [chips]);

  if (chips.length === 0) return null;
  return (
    <ul ref={listRef} aria-label="Active filters" className="flex min-w-0 flex-wrap items-center gap-1.5">
      {groupChips(chips).map(({ key, chips: run }, runIndex) => (
        <li key={key} className="min-w-0 max-w-full">
          {run.length > 1 ? (
            <GroupedChip chips={run} runKey={key} runIndex={runIndex} onRemove={remove} />
          ) : (
            <span className={CHIP_CLASS}>
              <span className="shrink-0 text-muted-foreground">{run[0].label}:</span>
              <span className="min-w-0 truncate text-foreground" title={run[0].value}>
                {run[0].value}
              </span>
              <ChipRemoveButton
                label={`Remove ${run[0].label} filter: ${run[0].value}`}
                onClick={() => remove(run[0].remove, { index: 0, runIndex })}
                data-chip-remove-run
              />
            </span>
          )}
        </li>
      ))}
      <li>
        <button
          type="button"
          onClick={() => {
            pendingFocus.current = { index: 0, runIndex: 0 };
            onClearAll();
          }}
          data-chip-clear
          className="rounded-md px-2 py-1 text-xs font-medium text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          Clear all
        </button>
      </li>
    </ul>
  );
}
