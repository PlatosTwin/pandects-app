import { ArrowDown, ArrowUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

export type ResultsSortField = "year" | "target" | "acquirer";
export type ResultsDensity = "comfy" | "compact";

export interface ResultsToolbarSelection {
  allSelected: boolean;
  someSelected: boolean;
  onToggleSelectAll: () => void;
  /** Accessible name for the select-all checkbox. */
  selectAllLabel: string;
  /** Text rendered next to the checkbox, e.g. "3 of 25 selected". */
  countLabel: string;
}

export interface ResultsToolbarProps {
  /**
   * "labeled" (sections and deals): desktop-only selection, labeled
   * "Density:" / "Sort by:" controls, live-region selection count.
   * "inline" (tax clauses): selection visible on mobile, single-row
   * "Sort: X" select with trailing density toggle.
   */
  variant?: "labeled" | "inline";
  /** Omit to render no selection controls (e.g. selection disabled). */
  selection?: ResultsToolbarSelection;
  sortBy: ResultsSortField;
  sortDirection: "asc" | "desc";
  onSortResults: (field: ResultsSortField) => void;
  onToggleSortDirection: () => void;
  /** Accessible name for the sort-field select. */
  sortSelectLabel: string;
  density?: ResultsDensity;
  onDensityChange?: (density: ResultsDensity) => void;
}

/**
 * Shared toolbar above search result lists: select-all controls, density
 * toggle, and sort controls. The two variants preserve the exact DOM/ARIA
 * shape each result list shipped with before extraction.
 */
export function ResultsToolbar({
  variant = "labeled",
  selection,
  sortBy,
  sortDirection,
  onSortResults,
  onToggleSortDirection,
  sortSelectLabel,
  density = "comfy",
  onDensityChange,
}: ResultsToolbarProps) {
  const checkedState = (sel: ResultsToolbarSelection) =>
    sel.allSelected ? true : sel.someSelected ? "indeterminate" : false;

  if (variant === "inline") {
    return (
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        {selection ? (
          <div className="flex items-center gap-2">
            <Checkbox
              checked={checkedState(selection)}
              onCheckedChange={() => selection.onToggleSelectAll()}
              className="h-5 w-5 data-[state=checked]:bg-primary data-[state=checked]:border-primary sm:h-4 sm:w-4"
              aria-label={selection.selectAllLabel}
            />
            <span className="text-sm text-muted-foreground">{selection.countLabel}</span>
          </div>
        ) : null}
        <div className="flex w-full flex-wrap items-center gap-2 sm:w-auto">
          <Select value={sortBy} onValueChange={(v) => onSortResults(v as ResultsSortField)}>
            <SelectTrigger
              className="h-11 flex-1 sm:h-8 sm:w-[140px] sm:flex-none"
              aria-label={sortSelectLabel}
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="year">Sort: Year</SelectItem>
              <SelectItem value="target">Sort: Target</SelectItem>
              <SelectItem value="acquirer">Sort: Acquirer</SelectItem>
            </SelectContent>
          </Select>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={onToggleSortDirection}
            className="h-11 w-11 p-0 sm:h-8 sm:w-auto sm:px-3"
            aria-label={`Change sort direction. Current direction: ${sortDirection === "asc" ? "ascending" : "descending"}`}
          >
            {sortDirection === "asc" ? (
              <ArrowUp className="h-4 w-4" aria-hidden="true" />
            ) : (
              <ArrowDown className="h-4 w-4" aria-hidden="true" />
            )}
          </Button>
          {onDensityChange && (
            <ToggleGroup
              type="single"
              value={density}
              onValueChange={(v) => {
                if (v === "comfy" || v === "compact") onDensityChange(v);
              }}
              className="hidden sm:flex"
            >
              <ToggleGroupItem
                value="comfy"
                aria-label="Comfortable density"
                className="h-8 px-2 text-xs"
              >
                Comfy
              </ToggleGroupItem>
              <ToggleGroupItem
                value="compact"
                aria-label="Compact density"
                className="h-8 px-2 text-xs"
              >
                Compact
              </ToggleGroupItem>
            </ToggleGroup>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
      {selection ? (
        // Bulk selection is intentionally desktop-only in this variant; mobile
        // result cards stay focused on opening individual agreements.
        <div className="hidden items-center gap-2 sm:flex">
          <Checkbox
            checked={checkedState(selection)}
            onCheckedChange={() => selection.onToggleSelectAll()}
            className="data-[state=checked]:bg-primary data-[state=checked]:border-primary"
            aria-label={selection.selectAllLabel}
          />
          <span className="text-sm text-muted-foreground" aria-live="polite">
            {selection.countLabel}
          </span>
        </div>
      ) : null}

      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-end">
        <div className="flex w-full flex-col gap-2 sm:w-auto sm:flex-row sm:items-center sm:gap-3">
          {/* Density */}
          <div className="hidden items-center gap-2 sm:flex">
            <span className="hidden text-sm text-muted-foreground sm:inline">Density:</span>
            <ToggleGroup
              type="single"
              aria-label="Results density"
              value={density}
              onValueChange={(value) => {
                if (value === "comfy" || value === "compact") {
                  onDensityChange?.(value);
                }
              }}
              variant="outline"
              size="xs"
              className="justify-start"
            >
              <ToggleGroupItem
                value="compact"
                aria-label="Compact density"
                className="text-muted-foreground data-[state=on]:text-foreground"
              >
                Compact
              </ToggleGroupItem>
              <ToggleGroupItem
                value="comfy"
                aria-label="Comfy density"
                className="text-muted-foreground data-[state=on]:text-foreground"
              >
                Comfy
              </ToggleGroupItem>
            </ToggleGroup>
          </div>

          {/* Sort */}
          <div className="flex items-center gap-2">
            <span className="hidden text-sm text-muted-foreground sm:inline">Sort by:</span>
            <Select
              value={sortBy}
              onValueChange={(value) => onSortResults(value as ResultsSortField)}
            >
              <SelectTrigger className="h-11 w-full sm:h-9 sm:w-[160px]" aria-label={sortSelectLabel}>
                <SelectValue placeholder="Sort by" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="year">Year</SelectItem>
                <SelectItem value="target">Target</SelectItem>
                <SelectItem value="acquirer">Acquirer</SelectItem>
              </SelectContent>
            </Select>
            <Button
              variant="ghost"
              size="sm"
              onClick={onToggleSortDirection}
              className="h-10 w-10 p-1 hover:bg-muted/40 sm:h-8 sm:w-8"
              title={`Sort ${sortDirection === "asc" ? "descending" : "ascending"}`}
              aria-label={`Sort ${sortDirection === "asc" ? "descending" : "ascending"}`}
            >
              {sortDirection === "asc" ? (
                <ArrowUp className="w-4 h-4" aria-hidden="true" />
              ) : (
                <ArrowDown className="w-4 h-4" aria-hidden="true" />
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
