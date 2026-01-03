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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { SearchResult } from "@shared/search";

import { useEffect, useRef, useState } from "react";

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
  const [expandedResults, setExpandedResults] = useState<Set<string>>(
    () => new Set(),
  );
  const [expandableResults, setExpandableResults] = useState<Set<string>>(
    () => new Set(),
  );
  const snippetRefs = useRef(new Map<string, HTMLDivElement | null>());
  const allSelected =
    searchResults.length > 0 &&
    searchResults.every((result) => selectedResults.has(result.id));
  const someSelected =
    searchResults.some((result) => selectedResults.has(result.id)) &&
    !allSelected;

  const toggleExpandedResult = (resultId: string) => {
    setExpandedResults((prev) => {
      const next = new Set(prev);
      if (next.has(resultId)) {
        next.delete(resultId);
      } else {
        next.add(resultId);
      }
      return next;
    });
  };

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const isMobile = window.matchMedia("(max-width: 639px)").matches;
    if (!isMobile) {
      setExpandableResults(new Set());
      return;
    }

    const recalculateExpandableResults = () => {
      const currentIds = new Set(searchResults.map((result) => result.id));
      setExpandableResults((prev) => {
        const next = new Set<string>();
        currentIds.forEach((id) => {
          if (prev.has(id)) {
            next.add(id);
          }
        });

        searchResults.forEach((result) => {
          const el = snippetRefs.current.get(result.id);
          if (el && el.scrollHeight > el.clientHeight + 1) {
            next.add(result.id);
          }
        });

        return next;
      });
    };

    const rafId = window.requestAnimationFrame(recalculateExpandableResults);
    const resizeObserver = new ResizeObserver(() => {
      window.requestAnimationFrame(recalculateExpandableResults);
    });

    snippetRefs.current.forEach((node) => {
      if (node) {
        resizeObserver.observe(node);
      }
    });

    const handleResize = () => {
      window.requestAnimationFrame(recalculateExpandableResults);
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.cancelAnimationFrame(rafId);
      resizeObserver.disconnect();
      window.removeEventListener("resize", handleResize);
    };
  }, [searchResults, density]);

  return (
    <div className={cn("space-y-4", className)}>
      {/* Header with Select All and Sort Controls */}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="hidden items-center gap-3 sm:flex">
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
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-3">
            {/* Density */}
            <div className="hidden items-center gap-2 sm:flex">
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
              <label className="hidden text-sm text-muted-foreground sm:inline">
                Sort by:
              </label>
              <Select
                value={sortBy}
                onValueChange={(value) =>
                  onSortResults(value as "year" | "target" | "acquirer")
                }
              >
                <SelectTrigger className="h-9 w-full sm:w-[160px]">
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
                className="h-10 w-10 p-1 sm:h-8 sm:w-8"
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
          className={cn(
            "grid gap-4",
            density === "compact" ? "sm:gap-2" : "sm:gap-4",
          )}
        >
          {searchResults.map((result, index) => {
            const targetText = truncateText(result.target, 75);
            const acquirerText = truncateText(result.acquirer, 75);
            const isSelected = selectedResults.has(result.id);
            const isExpanded = expandedResults.has(result.id);
            const canExpand = expandableResults.has(result.id) || isExpanded;
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
            const clauseTypeText = clauseTypeLabel
              ? truncateText(clauseTypeLabel, 75)
              : null;
            const sectionSummary = `${result.articleTitle} \u2192 ${result.sectionTitle}`;
            const sectionSummaryText = truncateText(sectionSummary, 120);
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
                    "border-b px-4 py-3",
                    density === "compact" ? "sm:px-3 sm:py-2" : "sm:px-4 sm:py-3",
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
                        className="h-6 w-6 data-[state=checked]:bg-primary data-[state=checked]:border-primary sm:h-4 sm:w-4"
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
                                <span
                                  tabIndex={0}
                                  className="hidden sm:inline-flex max-w-[18rem] min-w-0 cursor-help items-center rounded-full bg-background px-2 py-0.5 text-xs font-medium text-muted-foreground ring-1 ring-border focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                                >
                                  <span title={clauseTypeLabel}>
                                    {clauseTypeText?.truncated}
                                  </span>
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
                                <span
                                  tabIndex={0}
                                  className="hidden sm:inline-flex max-w-[18rem] min-w-0 cursor-help items-center rounded-full bg-background px-2 py-0.5 text-xs font-medium text-muted-foreground ring-1 ring-border focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                                >
                                  <span>Clause type unavailable</span>
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
                            "mt-2 flex flex-col gap-1.5 text-sm text-foreground sm:grid",
                            density === "compact" ? "sm:mt-1 sm:gap-1" : "sm:mt-2 sm:gap-1.5",
                          )}
                        >
                          <div className="min-w-0 break-words">
                            <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                              Target
                            </span>{" "}
                            {targetText.needsTooltip ? (
                              <Tooltip>
                                <TooltipTrigger asChild>
                                  <span
                                    tabIndex={0}
                                    className="cursor-help break-words font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                                  >
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
                                  <span
                                    tabIndex={0}
                                    className="cursor-help break-words font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                                  >
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
                            <span
                              className="block max-w-full min-w-0 truncate sm:max-w-[22rem]"
                              title={sectionSummary}
                            >
                              {sectionSummaryText.truncated}
                            </span>
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Open Agreement Button */}
                    <div className="sm:ml-4">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => onOpenAgreement(result, resultNumber)}
                        className="flex h-11 items-center gap-2 border-primary/40 px-4 text-primary hover:bg-primary/10 sm:h-9 sm:px-3"
                      >
                        <ExternalLink className="w-4 h-4" aria-hidden="true" />
                        <span className="hidden sm:inline">Open Agreement</span>
                        <span className="sm:hidden">Open</span>
                      </Button>
                    </div>
                  </div>
                </div>

                {/* Clause text */}
                <div className={cn("p-4", density === "compact" ? "sm:p-3" : "sm:p-4")}>
                  {result.xml ? (
                    <>
                      <div
                        className={cn(
                          "rounded-md border border-border bg-muted/20 p-3",
                          density === "compact"
                            ? "sm:h-28 sm:p-2"
                            : "sm:h-36 sm:p-3",
                        )}
                      >
                        <div
                          ref={(node) => {
                            if (node) {
                              snippetRefs.current.set(result.id, node);
                            } else {
                              snippetRefs.current.delete(result.id);
                            }
                          }}
                          className={cn(
                            "text-sm leading-relaxed text-foreground",
                            "overflow-hidden sm:h-full sm:overflow-y-auto",
                            isExpanded ? "line-clamp-none" : "line-clamp-3",
                            "sm:line-clamp-none",
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
                      {canExpand ? (
                        <div className="mt-2 sm:hidden">
                          <Button
                            type="button"
                            variant="link"
                            size="sm"
                            onClick={() => toggleExpandedResult(result.id)}
                            className="h-8 px-0"
                          >
                            {isExpanded ? "Show less" : "Show more"}
                          </Button>
                        </div>
                      ) : null}
                    </>
                  ) : (
                    <div className={cn(density === "compact" ? "sm:h-28" : "sm:h-36")}>
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
