import { Link } from "react-router-dom";
import { ArrowDown, ArrowUp, ExternalLink, Tag } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { FlagAsInaccurateButton } from "@/components/FlagAsInaccurateButton";
import { cn } from "@/lib/utils";
import type { TaxClauseSearchResult } from "@shared/tax-clauses";

interface TaxClauseResultsListProps {
  results: TaxClauseSearchResult[];
  getAgreementHref: (result: TaxClauseSearchResult) => string;
  clauseTypeLabelById: Record<string, string>;
  selectedResults: Set<string>;
  onToggleResultSelection: (id: string) => void;
  onToggleSelectAll: () => void;
  sortBy: "year" | "target" | "acquirer";
  sortDirection: "asc" | "desc";
  onSortResults: (field: "year" | "target" | "acquirer") => void;
  onToggleSortDirection: () => void;
  density?: "comfy" | "compact";
  onDensityChange?: (density: "comfy" | "compact") => void;
  className?: string;
}

export function TaxClauseResultsList({
  results,
  getAgreementHref,
  clauseTypeLabelById,
  selectedResults,
  onToggleResultSelection,
  onToggleSelectAll,
  sortBy,
  sortDirection,
  onSortResults,
  onToggleSortDirection,
  density = "comfy",
  onDensityChange,
  className,
}: TaxClauseResultsListProps) {
  const allSelected =
    results.length > 0 && results.every((r) => selectedResults.has(r.id));
  const someSelected =
    !allSelected && results.some((r) => selectedResults.has(r.id));

  const isCompact = density === "compact";

  return (
    <div className={cn("space-y-4", className)}>
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2">
          <Checkbox
            checked={allSelected ? true : someSelected ? "indeterminate" : false}
            onCheckedChange={() => onToggleSelectAll()}
            aria-label="Select all tax clauses on this page"
          />
          <span className="text-sm text-muted-foreground">
            {selectedResults.size > 0
              ? `${selectedResults.size} selected`
              : "Select all on page"}
          </span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Select value={sortBy} onValueChange={(v) => onSortResults(v as "year" | "target" | "acquirer")}>
            <SelectTrigger className="h-8 w-[140px]">
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
            aria-label={`Sort direction: ${sortDirection === "asc" ? "ascending" : "descending"}`}
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
              <ToggleGroupItem value="comfy" aria-label="Comfortable density" className="h-8 px-2 text-xs">
                Comfy
              </ToggleGroupItem>
              <ToggleGroupItem value="compact" aria-label="Compact density" className="h-8 px-2 text-xs">
                Compact
              </ToggleGroupItem>
            </ToggleGroup>
          )}
        </div>
      </div>

      <ul className="space-y-3" role="list">
        {results.map((result) => {
          const isSelected = selectedResults.has(result.id);
          const target = result.target?.trim() || "Unknown target";
          const acquirer = result.acquirer?.trim() || "Unknown acquirer";
          const year = result.year ?? "—";
          return (
            <li
              key={result.id}
              className={cn(
                "rounded-xl border border-border bg-card shadow-sm transition-colors",
                isSelected && "ring-2 ring-primary/30",
                isCompact ? "p-3" : "p-4",
              )}
            >
              <div className="flex items-start gap-3">
                <Checkbox
                  checked={isSelected}
                  onCheckedChange={() => onToggleResultSelection(result.id)}
                  className="mt-1"
                  aria-label={`Select tax clause ${result.clause_uuid}`}
                />
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                    <span className="font-medium text-foreground">{year}</span>
                    <span aria-hidden="true">·</span>
                    <span className="truncate">
                      <span className="text-muted-foreground">Target:</span>{" "}
                      <span className="font-medium text-foreground">{target}</span>
                    </span>
                    <span aria-hidden="true">·</span>
                    <span className="truncate">
                      <span className="text-muted-foreground">Acquirer:</span>{" "}
                      <span className="font-medium text-foreground">{acquirer}</span>
                    </span>
                    {result.context_type === "rep_warranty" && (
                      <Badge variant="outline" className="border-amber-300/70 bg-amber-50 text-amber-900 dark:bg-amber-900/20 dark:text-amber-200">
                        Reps & warranties
                      </Badge>
                    )}
                  </div>

                  {result.tax_standard_ids.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {result.tax_standard_ids.map((sid) => (
                        <Badge key={sid} variant="secondary" className="gap-1 text-xs">
                          <Tag className="h-3 w-3" aria-hidden="true" />
                          {clauseTypeLabelById[sid] ?? sid}
                        </Badge>
                      ))}
                    </div>
                  )}

                  <p
                    className={cn(
                      "mt-3 whitespace-pre-wrap text-sm text-foreground",
                      isCompact && "line-clamp-4",
                    )}
                  >
                    {result.clause_text}
                  </p>

                  <div className="mt-3 flex flex-wrap items-center gap-2">
                    <Button asChild size="sm" variant="outline" className="gap-1.5">
                      <Link to={getAgreementHref(result)}>
                        <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
                        Open agreement
                      </Link>
                    </Button>
                    <FlagAsInaccurateButton
                      source="search_result"
                      agreement_uuid={result.agreement_uuid}
                      className="shrink-0"
                    />
                  </div>
                </div>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
