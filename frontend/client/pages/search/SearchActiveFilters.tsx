import { X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { formatFilterOption } from "@/lib/text-utils";
import type { SearchFilters } from "@shared/sections";
import type { SearchMode } from "@shared/search";

const HARDCODED_ENUM_FIELDS = new Set([
  "transaction_price_total",
  "transaction_price_stock",
  "transaction_price_cash",
  "transaction_price_assets",
  "transaction_consideration",
  "target_type",
  "acquirer_type",
  "deal_status",
  "attitude",
  "deal_type",
  "purpose",
  "target_pe",
  "acquirer_pe",
]);

interface SearchActiveFiltersProps {
  filters: SearchFilters;
  searchMode: SearchMode;
  clauseTypeLabelById: Record<string, string>;
  onToggleFilterValue: (field: string, value: string) => void;
  onTextFilterChange: (field: string, value: string) => void;
  onClearAll: () => void;
}

export function SearchActiveFilters({
  filters,
  searchMode,
  clauseTypeLabelById,
  onToggleFilterValue,
  onTextFilterChange,
  onClearAll,
}: SearchActiveFiltersProps) {
  const hasAny =
    filters.year.length > 0 ||
    filters.target.length > 0 ||
    filters.acquirer.length > 0 ||
    filters.clauseType.length > 0 ||
    filters.transaction_price_total.length > 0 ||
    filters.transaction_price_stock.length > 0 ||
    filters.transaction_price_cash.length > 0 ||
    filters.transaction_price_assets.length > 0 ||
    filters.transaction_consideration.length > 0 ||
    filters.target_type.length > 0 ||
    filters.acquirer_type.length > 0 ||
    filters.target_counsel.length > 0 ||
    filters.acquirer_counsel.length > 0 ||
    filters.target_industry.length > 0 ||
    filters.acquirer_industry.length > 0 ||
    filters.deal_status.length > 0 ||
    filters.attitude.length > 0 ||
    filters.deal_type.length > 0 ||
    filters.purpose.length > 0 ||
    filters.target_pe.length > 0 ||
    filters.acquirer_pe.length > 0 ||
    Boolean(filters.agreement_uuid) ||
    Boolean(filters.section_uuid);

  if (!hasAny) return null;

  const fieldList = [
    ["year", "Year", filters.year],
    ["target", "Target", filters.target],
    ["acquirer", "Acquirer", filters.acquirer],
    ["clauseType", searchMode === "tax" ? "Tax clause type" : "Section type", filters.clauseType],
    ["transaction_price_total", "Price (total)", filters.transaction_price_total],
    ["transaction_price_stock", "Price (stock)", filters.transaction_price_stock],
    ["transaction_price_cash", "Price (cash)", filters.transaction_price_cash],
    ["transaction_price_assets", "Price (assets)", filters.transaction_price_assets],
    ["transaction_consideration", "Consideration", filters.transaction_consideration],
    ["target_type", "Target type", filters.target_type],
    ["acquirer_type", "Acquirer type", filters.acquirer_type],
    ["target_counsel", "Target counsel", filters.target_counsel],
    ["acquirer_counsel", "Acquirer counsel", filters.acquirer_counsel],
    ["target_industry", "Target industry", filters.target_industry],
    ["acquirer_industry", "Acquirer industry", filters.acquirer_industry],
    ["deal_status", "Status", filters.deal_status],
    ["attitude", "Attitude", filters.attitude],
    ["deal_type", "Deal type", filters.deal_type],
    ["purpose", "Purpose", filters.purpose],
    ["target_pe", "Target PE", filters.target_pe],
    ["acquirer_pe", "Acquirer PE", filters.acquirer_pe],
  ] as const;

  return (
    <>
      <div className="hidden h-5 w-0.5 shrink-0 rounded-full bg-border/80 sm:block" aria-hidden="true" />
      {fieldList.flatMap(([field, label, values]) =>
        values.map((value) => {
          const displayValue =
            field === "clauseType"
              ? clauseTypeLabelById[value] ?? value
              : HARDCODED_ENUM_FIELDS.has(field)
                ? formatFilterOption(value)
                : value;
          return (
            <Badge
              key={`${field}:${value}`}
              variant="outline"
              className="flex max-w-full items-center gap-1 rounded-md bg-background px-2 py-1"
            >
              <span className="text-muted-foreground">{label}:</span>
              <span className="min-w-0 truncate">{displayValue}</span>
              <button
                type="button"
                onClick={() => onToggleFilterValue(field, value)}
                className="ml-1 inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-sm text-muted-foreground hover:bg-accent/60 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:h-5 sm:w-5 sm:min-h-0 sm:min-w-0"
                aria-label={`Remove ${label} filter: ${displayValue}`}
              >
                <X className="h-3 w-3" aria-hidden="true" />
              </button>
            </Badge>
          );
        }),
      )}
      {filters.agreement_uuid && (
        <Badge variant="outline" className="flex max-w-full items-center gap-1 rounded-md bg-background px-2 py-1">
          <span className="text-muted-foreground">Agreement UUID:</span>
          <span className="min-w-0 truncate">{filters.agreement_uuid}</span>
          <button
            type="button"
            onClick={() => onTextFilterChange("agreement_uuid", "")}
            className="ml-1 inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-sm text-muted-foreground hover:bg-accent/60 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring sm:h-5 sm:w-5 sm:min-h-0 sm:min-w-0"
            aria-label={`Remove Agreement UUID filter: ${filters.agreement_uuid}`}
          >
            <X className="h-3 w-3" aria-hidden="true" />
          </button>
        </Badge>
      )}
      {filters.section_uuid && (
        <Badge variant="outline" className="flex max-w-full items-center gap-1 rounded-md bg-background px-2 py-1">
          <span className="text-muted-foreground">Section UUID:</span>
          <span className="min-w-0 truncate">{filters.section_uuid}</span>
          <button
            type="button"
            onClick={() => onTextFilterChange("section_uuid", "")}
            className="ml-1 inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-sm text-muted-foreground hover:bg-accent/60 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring sm:h-5 sm:w-5 sm:min-h-0 sm:min-w-0"
            aria-label={`Remove Section UUID filter: ${filters.section_uuid}`}
          >
            <X className="h-3 w-3" aria-hidden="true" />
          </button>
        </Badge>
      )}
      <Button
        variant="ghost"
        size="sm"
        onClick={onClearAll}
        className="h-7 px-2 text-muted-foreground hover:text-foreground"
      >
        Clear all
      </Button>
    </>
  );
}
