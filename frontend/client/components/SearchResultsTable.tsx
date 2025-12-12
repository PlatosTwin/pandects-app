import { useState } from "react";
import { ExternalLink, ArrowUp, ArrowDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { Checkbox } from "@/components/ui/checkbox";
import { Button } from "@/components/ui/button";
import { XMLRenderer } from "@/components/XMLRenderer";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface SearchResult {
  id: string;
  year: string;
  target: string;
  acquirer: string;
  articleTitle: string;
  sectionTitle: string;
  xml: string;
  sectionUuid: string;
  agreementUuid: string;
}

interface SearchResultsTableProps {
  searchResults: SearchResult[];
  selectedResults: Set<string>;
  sortBy?: "year" | "target" | "acquirer";
  sortDirection: "asc" | "desc";
  onToggleResultSelection: (resultId: string) => void;
  onToggleSelectAll: () => void;
  onOpenAgreement: (result: SearchResult) => void;
  onSortResults: (sortBy: "year" | "target" | "acquirer") => void;
  onToggleSortDirection: () => void;
  density?: "comfy" | "compact";
  onDensityChange?: (density: "comfy" | "compact") => void;
  currentPage?: number;
  pageSize?: number;
  className?: string;
}

// Utility function to truncate text and determine if tooltip is needed
const truncateText = (text: string, maxLength: number = 75) => {
  if (text.length <= maxLength) {
    return { truncated: text, needsTooltip: false };
  }
  return {
    truncated: text.substring(0, maxLength) + "...",
    needsTooltip: true,
  };
};

export function SearchResultsTable({
  searchResults,
  selectedResults,
  sortBy = "year",
  sortDirection,
  onToggleResultSelection,
  onToggleSelectAll,
  onOpenAgreement,
  onSortResults,
  onToggleSortDirection,
  density = "comfy",
  onDensityChange,
  currentPage = 1,
  pageSize = 25,
  className,
}: SearchResultsTableProps) {
  const allSelected =
    searchResults.length > 0 &&
    searchResults.every((result) => selectedResults.has(result.id));
  const someSelected =
    searchResults.some((result) => selectedResults.has(result.id)) &&
    !allSelected;

  return (
    <div className={cn("space-y-4", className)}>
      {/* Header with Select All and Sort Controls */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Checkbox
              checked={allSelected ? true : someSelected ? "indeterminate" : false}
              onCheckedChange={() => onToggleSelectAll()}
              className="data-[state=checked]:bg-primary data-[state=checked]:border-primary"
            />
            <span className="text-sm text-muted-foreground">
              {selectedResults.size > 0
                ? `${selectedResults.size} of ${searchResults.length} selected`
                : "Select all"}
            </span>
          </div>
        </div>

        {/* Sort Controls */}
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-end">
          <div className="flex items-center justify-between gap-2 sm:justify-end">
            <span className="text-sm text-muted-foreground sm:hidden">
              Sort & density
            </span>
          </div>

          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-3">
            {/* Density */}
            <div className="flex items-center gap-2">
              <span className="hidden text-sm text-muted-foreground sm:inline">
                Density:
              </span>
              <ToggleGroup
                type="single"
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
              <span className="hidden text-sm text-muted-foreground sm:inline">
                Sort by:
              </span>
            <select
              className="h-9 w-full rounded border border-input bg-background px-3 py-1 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent sm:w-auto"
              onChange={(e) =>
                onSortResults(e.target.value as "year" | "target" | "acquirer")
              }
              value={sortBy}
            >
              <option value="year">Year</option>
              <option value="target">Target</option>
              <option value="acquirer">Acquirer</option>
            </select>
            <Button
              variant="ghost"
              size="sm"
              onClick={onToggleSortDirection}
              className="p-1 h-8 w-8"
              title={`Sort ${sortDirection === "asc" ? "descending" : "ascending"}`}
            >
              {sortDirection === "asc" ? (
                <ArrowUp className="w-4 h-4" />
              ) : (
                <ArrowDown className="w-4 h-4" />
              )}
            </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Results Grid */}
      <TooltipProvider>
        <div className={cn("grid", density === "compact" ? "gap-2" : "gap-4")}>
          {searchResults.map((result, index) => {
            const targetText = truncateText(result.target, 75);
            const acquirerText = truncateText(result.acquirer, 75);
            const isSelected = selectedResults.has(result.id);
            const resultNumber = (currentPage - 1) * pageSize + index + 1;

            return (
              <div
                key={result.id}
                className={cn(
                  "rounded-lg border bg-card shadow-sm overflow-hidden transition-colors",
                  isSelected
                    ? "border-primary/40 bg-primary/5"
                    : "border-border",
                )}
              >
                {/* Header with metadata and checkbox */}
                <div
                  className={cn(
                    density === "compact" ? "px-3 py-2" : "px-4 py-3",
                    "border-b",
                    isSelected
                      ? "bg-primary/10 border-primary/20"
                      : "bg-muted/40 border-border",
                  )}
                >
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                    <div className="flex items-start gap-3">
                      <Checkbox
                        checked={isSelected}
                        onCheckedChange={() =>
                          onToggleResultSelection(result.id)
                        }
                        className="data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                      />
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-bold text-muted-foreground">
                            {resultNumber}
                          </span>
                          <span className="inline-flex items-center rounded-full bg-background px-2 py-0.5 text-xs font-medium text-muted-foreground ring-1 ring-border">
                            {result.year}
                          </span>
                        </div>

                        <div
                          className={cn(
                            "grid text-sm text-foreground",
                            density === "compact" ? "mt-1 gap-0.5" : "mt-2 gap-1",
                          )}
                        >
                          <div className="min-w-0 break-words">
                            <span className="font-bold">T:</span>{" "}
                            {targetText.needsTooltip ? (
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <span className="cursor-help break-words">
                                    {targetText.truncated}
                                  </span>
                                </TooltipTrigger>
                                <TooltipContent>
                                  <p>{result.target}</p>
                                </TooltipContent>
                              </Tooltip>
                            ) : (
                              <span className="break-words">
                                {targetText.truncated}
                              </span>
                            )}
                          </div>
                          <div className="min-w-0 break-words">
                            <span className="font-bold">A:</span>{" "}
                            {acquirerText.needsTooltip ? (
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <span className="cursor-help break-words">
                                    {acquirerText.truncated}
                                  </span>
                                </TooltipTrigger>
                                <TooltipContent>
                                  <p>{result.acquirer}</p>
                                </TooltipContent>
                              </Tooltip>
                            ) : (
                              <span className="break-words">
                                {acquirerText.truncated}
                              </span>
                            )}
                          </div>
                          <div
                            className="text-xs text-muted-foreground break-words"
                            title={`${result.articleTitle} >> ${result.sectionTitle}`}
                          >
                            {result.articleTitle} &gt;&gt; {result.sectionTitle}
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Open Agreement Button */}
                    <div className="sm:ml-4">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onOpenAgreement(result)}
                        className="flex items-center gap-2 text-primary hover:bg-primary/10"
                      >
                        <ExternalLink className="w-4 h-4" />
                        <span className="hidden sm:inline">Open Agreement</span>
                        <span className="sm:hidden">Open</span>
                      </Button>
                    </div>
                  </div>
                </div>

                {/* Clause text */}
                <div className={cn(density === "compact" ? "p-3" : "p-4")}>
                  <div
                    className={cn(
                      "overflow-y-auto text-sm text-foreground leading-relaxed",
                      density === "compact" ? "h-28" : "h-36",
                    )}
                    style={{
                      scrollbarWidth: "thin",
                      scrollbarColor:
                        "hsl(var(--border)) hsl(var(--background))",
                    }}
                  >
                    <XMLRenderer xmlContent={result.xml} mode="search" />
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </TooltipProvider>
    </div>
  );
}
