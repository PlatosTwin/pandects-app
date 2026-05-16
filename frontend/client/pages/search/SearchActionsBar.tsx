import { Search as SearchIcon, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { TAX_COMPARE_MAX, TAX_COMPARE_MIN } from "@/lib/tax-compare-handoff";
import type { SearchFilters } from "@shared/sections";
import type { SearchMode } from "@shared/search";
import { SearchActiveFilters } from "./SearchActiveFilters";

interface SearchActionsBarProps {
  searchMode: SearchMode;
  isSearching: boolean;
  selectedSize: number;
  resultsLength: number;
  onSearch: () => void;
  onDownloadCSV: () => void;
  onClearFilters: () => void;
  taxIncludeRepWarranty: boolean;
  onTaxIncludeRepWarrantyChange: (value: boolean) => void;
  taxSelectedCount: number;
  onTaxCompare: () => void;
  filters: SearchFilters;
  clauseTypeLabelById: Record<string, string>;
  onToggleFilterValue: (field: string, value: string) => void;
  onTextFilterChange: (field: string, value: string) => void;
}

export function SearchActionsBar({
  searchMode,
  isSearching,
  selectedSize,
  resultsLength,
  onSearch,
  onDownloadCSV,
  onClearFilters,
  taxIncludeRepWarranty,
  onTaxIncludeRepWarrantyChange,
  taxSelectedCount,
  onTaxCompare,
  filters,
  clauseTypeLabelById,
  onToggleFilterValue,
  onTextFilterChange,
}: SearchActionsBarProps) {
  const downloadDisabled = resultsLength === 0 && selectedSize === 0;

  const compareDisabled =
    taxSelectedCount < TAX_COMPARE_MIN || taxSelectedCount > TAX_COMPARE_MAX;
  const compareLabel =
    taxSelectedCount === 0
      ? `Compare (select ${TAX_COMPARE_MIN}–${TAX_COMPARE_MAX})`
      : taxSelectedCount < TAX_COMPARE_MIN
        ? `Compare (select ${TAX_COMPARE_MIN - taxSelectedCount} more)`
        : taxSelectedCount > TAX_COMPARE_MAX
          ? `Compare (max ${TAX_COMPARE_MAX})`
          : `Compare (${taxSelectedCount})`;

  return (
    <div className="border-b border-border bg-muted/20 px-4 py-2.5 backdrop-blur supports-[backdrop-filter]:bg-muted/20 sm:px-8">
      <div className="flex flex-wrap items-center gap-2" role="toolbar" aria-label="Search actions">
        <Button
          onClick={onSearch}
          disabled={isSearching}
          className="h-11 flex-1 gap-2 sm:h-9 sm:flex-none sm:w-auto"
          variant="default"
          size="sm"
          aria-describedby="search-results-status"
        >
          <SearchIcon
            className={cn("h-4 w-4", isSearching && "animate-spin-custom")}
            aria-hidden="true"
          />
          <span>{isSearching ? "Searching..." : "Search"}</span>
        </Button>

        <Tooltip>
          <TooltipTrigger asChild>
            <span className="hidden sm:inline-block">
              <Button
                onClick={onDownloadCSV}
                disabled={downloadDisabled}
                variant="outline"
                size="sm"
                className="h-11 w-full gap-2 text-muted-foreground hover:text-foreground sm:h-9 sm:w-auto"
                aria-label={
                  downloadDisabled
                    ? "Download CSV (disabled: no results to download. Run a search first.)"
                    : "Download CSV"
                }
              >
                <Download className="h-4 w-4" aria-hidden="true" />
                <span className="sm:inline">
                  Download CSV
                  {selectedSize > 0 && ` (${selectedSize})`}
                </span>
              </Button>
            </span>
          </TooltipTrigger>
          {downloadDisabled && (
            <TooltipContent>
              <p>No results to download. Run a search first.</p>
            </TooltipContent>
          )}
        </Tooltip>

        <Button
          onClick={onClearFilters}
          variant="outline"
          size="sm"
          className="h-11 text-muted-foreground hover:text-foreground sm:h-9"
        >
          Reset filters
        </Button>

        {searchMode === "tax" && (
          <label className="flex min-h-11 items-center gap-2 rounded-md border border-border bg-background px-2 py-1 text-xs text-muted-foreground sm:min-h-0">
            <input
              type="checkbox"
              checked={taxIncludeRepWarranty}
              onChange={(e) => onTaxIncludeRepWarrantyChange(e.target.checked)}
              className="h-3.5 w-3.5"
            />
            Include reps &amp; warranties clauses
          </label>
        )}

        {searchMode === "tax" && (
          <Tooltip>
            <TooltipTrigger asChild>
              <span>
                <Button
                  onClick={onTaxCompare}
                  disabled={compareDisabled}
                  variant="outline"
                  size="sm"
                  className="h-11 sm:h-9"
                >
                  {compareLabel}
                </Button>
              </span>
            </TooltipTrigger>
            {compareDisabled && (
              <TooltipContent>
                <p>
                  Select {TAX_COMPARE_MIN}–{TAX_COMPARE_MAX} tax clauses to compare.
                </p>
              </TooltipContent>
            )}
          </Tooltip>
        )}

        <SearchActiveFilters
          filters={filters}
          searchMode={searchMode}
          clauseTypeLabelById={clauseTypeLabelById}
          onToggleFilterValue={onToggleFilterValue}
          onTextFilterChange={onTextFilterChange}
          onClearAll={onClearFilters}
        />
      </div>
    </div>
  );
}
