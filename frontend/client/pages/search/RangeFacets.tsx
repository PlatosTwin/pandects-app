import { useId, useState } from "react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { AVAILABLE_YEARS, TRANSACTION_PRICE_OPTIONS } from "@/lib/constants";
import { cn } from "@/lib/utils";
import type { SearchQueryFilters } from "@shared/search";
import {
  PRICE_FIELDS,
  summarizeBucketRange,
  summarizeYearRange,
  yearsBetween,
  type PriceField,
} from "./search-ui";

const ANY = "any";

function RangeSelect({
  label,
  value,
  options,
  onChange,
  anyLabel,
}: {
  label: string;
  value: string;
  options: ReadonlyArray<{ value: string; label: string }>;
  onChange: (value: string) => void;
  anyLabel: string;
}) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className="h-9 min-w-0 flex-1 text-sm" aria-label={label}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent className="max-h-72">
        <SelectItem value={ANY}>{anyLabel}</SelectItem>
        {options.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

const YEARS_ASC = [...AVAILABLE_YEARS].reverse();
const YEAR_OPTIONS = YEARS_ASC.map((year) => ({ value: year, label: year }));

/** Year as a from/to range; writes the `year` list the API takes. */
export function YearRangeFilter({
  years,
  onChange,
}: {
  years: string[];
  onChange: (years: string[]) => void;
}) {
  const labelId = useId();
  const range = summarizeYearRange(years);
  const from = range.from ?? ANY;
  const to = range.to ?? ANY;

  const apply = (nextFrom: string, nextTo: string) => {
    if (nextFrom === ANY && nextTo === ANY) {
      onChange([]);
      return;
    }
    onChange(
      yearsBetween(
        nextFrom === ANY ? YEARS_ASC[0] : nextFrom,
        nextTo === ANY ? YEARS_ASC[YEARS_ASC.length - 1] : nextTo,
      ),
    );
  };

  return (
    <div role="group" aria-labelledby={labelId}>
      <div id={labelId} className="mb-1.5 text-sm font-medium text-foreground">
        Year
      </div>
      <div className="flex items-center gap-2">
        <RangeSelect
          label="Year from"
          value={from}
          options={YEAR_OPTIONS}
          anyLabel="Any"
          onChange={(value) => apply(value, to)}
        />
        <span className="text-sm text-muted-foreground" aria-hidden="true">
          to
        </span>
        <RangeSelect
          label="Year to"
          value={to}
          options={YEAR_OPTIONS}
          anyLabel="Any"
          onChange={(value) => apply(from, value)}
        />
      </div>
      {!range.contiguous ? (
        <p className="mt-1.5 text-xs text-muted-foreground">
          This link selects {years.length} separate years. Picking a range replaces them.
        </p>
      ) : null}
    </div>
  );
}

const COMPONENT_LABELS: Record<PriceField, string> = {
  transaction_price_total: "Total",
  transaction_price_cash: "Cash",
  transaction_price_stock: "Stock",
  transaction_price_assets: "Assets",
};

const money = (value: string) => (value === "0" ? "$0" : `$${value}`);
const LOWER_OPTIONS = TRANSACTION_PRICE_OPTIONS.map((bucket) => ({
  value: bucket,
  label: money(bucket.split(" - ")[0].replace("+", "")),
}));
const UPPER_OPTIONS = TRANSACTION_PRICE_OPTIONS.filter((bucket) => bucket.includes(" - ")).map(
  (bucket) => ({ value: bucket, label: money(bucket.split(" - ")[1]) }),
);

/**
 * Deal value as a min/max range per component (total, cash, stock, assets).
 * The API filters on value buckets, so the range maps to the buckets it spans.
 */
export function ValueRangeFilter({
  filters,
  onChange,
}: {
  filters: SearchQueryFilters;
  onChange: (field: PriceField, buckets: string[]) => void;
}) {
  const labelId = useId();
  const firstActive = PRICE_FIELDS.find((field) => (filters[field]?.length ?? 0) > 0);
  const [component, setComponent] = useState<PriceField>(firstActive ?? "transaction_price_total");
  const buckets = filters[component] ?? [];
  const range = summarizeBucketRange(buckets);
  const from = range.from ?? ANY;
  const to = range.to ?? ANY;

  const apply = (nextFrom: string, nextTo: string) => {
    if (nextFrom === ANY && nextTo === ANY) {
      onChange(component, []);
      return;
    }
    const a = nextFrom === ANY ? 0 : TRANSACTION_PRICE_OPTIONS.indexOf(nextFrom);
    const b =
      nextTo === ANY ? TRANSACTION_PRICE_OPTIONS.length - 1 : TRANSACTION_PRICE_OPTIONS.indexOf(nextTo);
    onChange(component, TRANSACTION_PRICE_OPTIONS.slice(Math.min(a, b), Math.max(a, b) + 1));
  };

  return (
    <div role="group" aria-labelledby={labelId}>
      <div id={labelId} className="mb-1.5 text-sm font-medium text-foreground">
        Deal value
      </div>
      <SegmentedControl
        size="sm"
        aria-label="Deal value component"
        value={component}
        onValueChange={(value) => setComponent(value as PriceField)}
        className="mb-2 w-full"
        options={PRICE_FIELDS.map((field) => ({
          value: field,
          label: (
            <span className="inline-flex items-center gap-1">
              {COMPONENT_LABELS[field]}
              <span
                className={cn(
                  "h-1.5 w-1.5 rounded-full bg-primary",
                  (filters[field]?.length ?? 0) === 0 && "invisible",
                )}
                aria-hidden="true"
              />
              {(filters[field]?.length ?? 0) > 0 ? <span className="sr-only">(active)</span> : null}
            </span>
          ),
        }))}
      />
      <div className="flex items-center gap-2">
        <RangeSelect
          label={`${COMPONENT_LABELS[component]} value from`}
          value={from}
          options={LOWER_OPTIONS}
          anyLabel="Min"
          onChange={(value) => apply(value, to)}
        />
        <span className="text-sm text-muted-foreground" aria-hidden="true">
          to
        </span>
        <RangeSelect
          label={`${COMPONENT_LABELS[component]} value to`}
          value={to === TRANSACTION_PRICE_OPTIONS[TRANSACTION_PRICE_OPTIONS.length - 1] ? ANY : to}
          options={UPPER_OPTIONS}
          anyLabel="Max"
          onChange={(value) => apply(from, value)}
        />
      </div>
      {!range.contiguous ? (
        <p className="mt-1.5 text-xs text-muted-foreground">
          This link selects separate value bands. Picking a range replaces them.
        </p>
      ) : null}
    </div>
  );
}
