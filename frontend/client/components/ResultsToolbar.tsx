import { ArrowDown, ArrowUp, Rows2, Rows4 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";

export type ResultsSortField = "year" | "target" | "acquirer";
export type ResultsDensity = "comfy" | "compact";

export const SORT_FIELD_LABELS: Record<ResultsSortField, string> = {
  year: "Year",
  target: "Target",
  acquirer: "Acquirer",
};

export interface ResultsToolbarProps {
  sortBy: ResultsSortField;
  sortDirection: "asc" | "desc";
  onSortResults: (field: ResultsSortField) => void;
  onToggleSortDirection: () => void;
  /** Accessible name for the sort-field select. */
  sortSelectLabel: string;
  density?: ResultsDensity;
  /** Omit to hide the density toggle. */
  onDensityChange?: (density: ResultsDensity) => void;
  disabled?: boolean;
  className?: string;
}

/** Sort field + direction and the density toggle, for the results header. */
export function ResultsToolbar({
  sortBy,
  sortDirection,
  onSortResults,
  onToggleSortDirection,
  sortSelectLabel,
  density = "comfy",
  onDensityChange,
  disabled = false,
  className,
}: ResultsToolbarProps) {
  return (
    <div className={cn("flex items-center gap-1", className)}>
      <Select
        value={sortBy}
        onValueChange={(value) => onSortResults(value as ResultsSortField)}
        disabled={disabled}
      >
        <SelectTrigger className="h-9 w-[8.5rem] text-sm sm:h-8" aria-label={sortSelectLabel}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {(Object.keys(SORT_FIELD_LABELS) as ResultsSortField[]).map((field) => (
            <SelectItem key={field} value={field}>
              Sort: {SORT_FIELD_LABELS[field]}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        onClick={onToggleSortDirection}
        disabled={disabled}
        className="h-9 w-9 p-0 sm:h-8 sm:w-8"
        title={`Sort ${sortDirection === "asc" ? "descending" : "ascending"}`}
        aria-label={`Sort ${sortDirection === "asc" ? "descending" : "ascending"}`}
      >
        {sortDirection === "asc" ? (
          <ArrowUp className="h-4 w-4" aria-hidden="true" />
        ) : (
          <ArrowDown className="h-4 w-4" aria-hidden="true" />
        )}
      </Button>
      {onDensityChange ? (
        <ToggleGroup
          type="single"
          aria-label="Results density"
          value={density}
          onValueChange={(value) => {
            if (value === "comfy" || value === "compact") onDensityChange(value);
          }}
          variant="outline"
          size="xs"
          className="hidden sm:flex"
        >
          <ToggleGroupItem value="comfy" aria-label="Comfy density" title="Comfy density">
            <Rows2 className="h-3.5 w-3.5" aria-hidden="true" />
          </ToggleGroupItem>
          <ToggleGroupItem value="compact" aria-label="Compact density" title="Compact density">
            <Rows4 className="h-3.5 w-3.5" aria-hidden="true" />
          </ToggleGroupItem>
        </ToggleGroup>
      ) : null}
    </div>
  );
}
