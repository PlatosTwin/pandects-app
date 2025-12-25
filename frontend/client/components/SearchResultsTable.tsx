import { ExternalLink, ArrowUp, ArrowDown, BadgeCheck } from "lucide-react";
import { cn } from "@/lib/utils";
import { Checkbox } from "@/components/ui/checkbox";
import { Button } from "@/components/ui/button";
import { XMLRenderer } from "@/components/XMLRenderer";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { SearchResult } from "@shared/search";

import { useCallback, useEffect, useRef, useState } from "react";

function TruncatedText({
  text,
  className,
}: {
  text: string;
  className?: string;
}) {
  const containerRef = useRef<HTMLSpanElement | null>(null);
  const measureRef = useRef<HTMLSpanElement | null>(null);
  const [displayText, setDisplayText] = useState(text);

  const measureText = useCallback((value: string) => {
    const el = measureRef.current;
    if (!el) return 0;
    el.textContent = value;
    return el.scrollWidth;
  }, []);

  const recompute = useCallback(() => {
    const container = containerRef.current;
    if (!container) return;

    const availableWidth = container.clientWidth;
    if (!availableWidth) {
      setDisplayText(text);
      return;
    }

    if (measureText(text) <= availableWidth) {
      setDisplayText(text);
      return;
    }

    const suffix = "...";
    let low = 0;
    let high = text.length;
    while (low < high) {
      const mid = Math.ceil((low + high) / 2);
      const candidate = text.slice(0, mid).trimEnd();
      if (measureText(candidate + suffix) <= availableWidth) {
        low = mid;
      } else {
        high = mid - 1;
      }
    }

    let truncated = text.slice(0, low).trimEnd();
    truncated = truncated.replace(/[•·,:;|-]+$/u, "").trimEnd();
    setDisplayText(truncated ? truncated + suffix : suffix);
  }, [measureText, text]);

  useEffect(() => {
    recompute();
  }, [recompute]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const ro = new ResizeObserver(recompute);
    ro.observe(container);
    return () => ro.disconnect();
  }, [recompute]);

  return (
    <span
      ref={containerRef}
      aria-label={displayText !== text ? text : undefined}
      title={displayText !== text ? text : undefined}
      className={cn(
        "relative block max-w-full min-w-0 overflow-hidden whitespace-nowrap",
        className,
      )}
    >
      <span
        ref={measureRef}
        aria-hidden
        className="pointer-events-none absolute left-0 top-0 whitespace-nowrap opacity-0"
      />
      {displayText}
    </span>
  );
}

interface SearchResultsTableProps {
  searchResults: SearchResult[];
  selectedResults: Set<string>;
  clauseTypePathByStandardId: Record<string, readonly string[]>;
  sortBy?: "year" | "target" | "acquirer";
  sortDirection: "asc" | "desc";
  onToggleResultSelection: (resultId: string) => void;
  onToggleSelectAll: () => void;
  onOpenAgreement: (result: SearchResult, position: number) => void;
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
  clauseTypePathByStandardId,
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
              aria-label="Select all results"
            />
            <span className="text-sm text-muted-foreground" aria-live="polite">
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
              <label
                htmlFor="search-sort-by"
                className="hidden text-sm text-muted-foreground sm:inline"
              >
                Sort by:
              </label>
              <select
                id="search-sort-by"
                className="h-9 w-full rounded border border-input bg-background px-3 py-1 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:border-transparent sm:w-auto"
                onChange={(e) =>
                  onSortResults(e.target.value as "year" | "target" | "acquirer")
                }
                value={sortBy}
                aria-label="Sort by"
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

      {/* Results Grid */}
	      <TooltipProvider>
        <div
          role="list"
          className={cn("grid", density === "compact" ? "gap-2" : "gap-4")}
        >
	          {searchResults.map((result, index) => {
	            const targetText = truncateText(result.target, 75);
	            const acquirerText = truncateText(result.acquirer, 75);
	            const isSelected = selectedResults.has(result.id);
	            const resultNumber = (currentPage - 1) * pageSize + index + 1;
	            const standardId =
	              typeof result.standardId === "string"
	                ? result.standardId.trim()
	                : null;
	            const clauseTypePath = standardId
	              ? clauseTypePathByStandardId[standardId]
	              : undefined;
	            const clauseTypeLabel = clauseTypePath
	              ?.map((part) => part.trim())
	              .join("\u2022 ");
	            const showDevFallbackPill =
	              import.meta.env.DEV && (!clauseTypePath || !clauseTypeLabel);
	
		            return (
	              <div
                  role="listitem"
		                key={result.id}
                className={cn(
	                  "relative overflow-hidden rounded-lg border bg-card shadow-sm transition-colors",
	                  isSelected
	                    ? "border-primary/40 bg-primary/5"
	                    : "border-border hover:bg-muted/10",
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
                        aria-label={`Select result ${resultNumber}`}
                      />
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-xs font-semibold text-muted-foreground">
                            #{resultNumber}
                          </span>
                          <span className="inline-flex items-center rounded-full bg-background px-2 py-0.5 text-xs font-medium text-foreground ring-1 ring-border">
                            {result.year}
                          </span>
                          {result.verified ? (
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <button
                                  type="button"
                                  aria-label="Verified agreement"
                                  className="inline-flex items-center gap-1 rounded-full bg-emerald-500/10 px-2 py-0.5 text-xs font-medium text-emerald-700 ring-1 ring-emerald-500/20 transition-colors hover:bg-emerald-500/15 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background dark:text-emerald-300"
                                >
                                  <BadgeCheck className="h-3.5 w-3.5" aria-hidden="true" />
                                  <span className="hidden sm:inline">Verified</span>
                                </button>
                              </TooltipTrigger>
                              <TooltipContent side="bottom">
                                <p>This agreement has been verified by hand.</p>
                              </TooltipContent>
                            </Tooltip>
                          ) : null}
                          {clauseTypeLabel ? (
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <span className="hidden sm:inline-flex max-w-[18rem] min-w-0 cursor-help items-center rounded-full bg-background px-2 py-0.5 text-xs font-medium text-muted-foreground ring-1 ring-border">
                                  <TruncatedText text={clauseTypeLabel} />
                                </span>
                              </TooltipTrigger>
                              <TooltipContent className="max-w-sm">
                                {clauseTypePath ? (
                                  <div className="space-y-1">
                                    {clauseTypePath.map((part, partIndex) => (
                                      <p key={`${partIndex}-${part}`}>{part}</p>
                                    ))}
                                  </div>
                                ) : null}
                              </TooltipContent>
                            </Tooltip>
                          ) : showDevFallbackPill ? (
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <span className="hidden sm:inline-flex max-w-[18rem] min-w-0 cursor-help items-center rounded-full bg-background px-2 py-0.5 text-xs font-medium text-muted-foreground ring-1 ring-border">
                                  <TruncatedText text="Clause type unavailable" />
                                </span>
                              </TooltipTrigger>
                              <TooltipContent className="max-w-sm">
                                {standardId ? (
                                  <>
                                    <p>standardId: {standardId}</p>
                                    <p>
                                      Not found in the clause-type mapping used
                                      by the frontend.
                                    </p>
                                  </>
                                ) : (
                                  <>
                                    <p>Missing standardId in search results.</p>
                                    <p>
                                      Restart the Flask API or deploy the latest
                                      backend so `/api/search` returns
                                      `standardId`.
                                    </p>
                                  </>
                                )}
                              </TooltipContent>
                            </Tooltip>
                          ) : null}
                        </div>

                        <div
                          className={cn(
                            "grid text-sm text-foreground",
                            density === "compact" ? "mt-1 gap-1" : "mt-2 gap-1.5",
                          )}
                        >
                          <div className="min-w-0 break-words">
                            <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                              Target
                            </span>{" "}
                            {targetText.needsTooltip ? (
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <span className="cursor-help break-words font-medium">
                                    {targetText.truncated}
                                  </span>
                                </TooltipTrigger>
                                <TooltipContent>
                                  <p>{result.target}</p>
                                </TooltipContent>
                              </Tooltip>
                            ) : (
                              <span className="break-words font-medium">
                                {targetText.truncated}
                              </span>
                            )}
                          </div>
                          <div className="min-w-0 break-words">
                            <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                              Acquirer
                            </span>{" "}
                            {acquirerText.needsTooltip ? (
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <span className="cursor-help break-words font-medium">
                                    {acquirerText.truncated}
                                  </span>
                                </TooltipTrigger>
                                <TooltipContent>
                                  <p>{result.acquirer}</p>
                                </TooltipContent>
                              </Tooltip>
                            ) : (
                              <span className="break-words font-medium">
                                {acquirerText.truncated}
                              </span>
                            )}
                          </div>
                          <span
                            className="inline-flex w-full min-w-0 max-w-full items-center rounded-full bg-background px-2 py-0.5 text-xs text-muted-foreground ring-1 ring-border sm:w-auto"
                            title={`${result.articleTitle} >> ${result.sectionTitle}`}
                          >
                            <TruncatedText
                              text={`${result.articleTitle} \u2192 ${result.sectionTitle}`}
                              className="max-w-full sm:max-w-[22rem]"
                            />
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Open Agreement Button */}
                    <div className="sm:ml-4">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onOpenAgreement(result, resultNumber)}
                        className="flex items-center gap-2 text-primary hover:bg-primary/10"
                      >
                        <ExternalLink className="w-4 h-4" aria-hidden="true" />
                        <span className="hidden sm:inline">Open Agreement</span>
                        <span className="sm:hidden">Open</span>
                      </Button>
                    </div>
                  </div>
                </div>

                {/* Clause text */}
                <div className={cn(density === "compact" ? "p-3" : "p-4")}>
                  {result.xml ? (
                    <div
                      className={cn(
                        "rounded-md border border-border bg-muted/20",
                        density === "compact" ? "h-28 p-2" : "h-36 p-3",
                      )}
                    >
                      <div
                        className="h-full overflow-y-auto text-sm leading-relaxed text-foreground"
                        style={{
                          scrollbarWidth: "thin",
                          scrollbarColor:
                            "hsl(var(--border)) hsl(var(--background))",
                        }}
                      >
                        <XMLRenderer xmlContent={result.xml} mode="search" />
                      </div>
                    </div>
                  ) : (
                    <div className={cn(density === "compact" ? "h-28" : "h-36")}>
                      <Alert className="h-full">
                        <AlertTitle>Sign in to view clause text</AlertTitle>
                        <AlertDescription>
                          Search and filters work in limited mode, but the text is
                          hidden until you create an account.
                        </AlertDescription>
                      </Alert>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </TooltipProvider>
    </div>
  );
}
