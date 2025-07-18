import { useState } from "react";
import { ExternalLink, ArrowUp, ArrowDown } from "lucide-react";
import { cn } from "@/lib/utils";
import { Checkbox } from "@/components/ui/checkbox";
import { Button } from "@/components/ui/button";
import { XMLRenderer } from "@/components/XMLRenderer";
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
  sortDirection: "asc" | "desc";
  onToggleResultSelection: (resultId: string) => void;
  onToggleSelectAll: () => void;
  onOpenAgreement: (result: SearchResult) => void;
  onSortResults: (sortBy: "year" | "target" | "acquirer") => void;
  onToggleSortDirection: () => void;
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
  sortDirection,
  onToggleResultSelection,
  onToggleSelectAll,
  onOpenAgreement,
  onSortResults,
  onToggleSortDirection,
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
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Checkbox
              checked={allSelected}
              ref={(checkbox) => {
                if (checkbox) {
                  (checkbox as any).indeterminate = someSelected;
                }
              }}
              onCheckedChange={onToggleSelectAll}
              className="data-[state=checked]:bg-material-blue data-[state=checked]:border-material-blue"
            />
            <span className="text-sm text-material-text-secondary">
              {selectedResults.size > 0
                ? `${selectedResults.size} of ${searchResults.length} selected`
                : "Select all"}
            </span>
          </div>
        </div>

        {/* Sort Controls */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-material-text-secondary">Sort by:</span>
          <div className="flex items-center gap-2">
            <select
              className="text-sm border border-gray-300 rounded px-3 py-1 bg-white text-material-text-primary focus:outline-none focus:ring-2 focus:ring-material-blue focus:border-transparent"
              onChange={(e) =>
                onSortResults(e.target.value as "year" | "target" | "acquirer")
              }
              defaultValue="year"
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

      {/* Results Grid */}
      <TooltipProvider>
        <div className="grid gap-4">
          {searchResults.map((result, index) => {
            const targetText = truncateText(result.target, 75);
            const acquirerText = truncateText(result.acquirer, 75);
            const isSelected = selectedResults.has(result.id);
            const resultNumber = (currentPage - 1) * pageSize + index + 1;

            return (
              <div
                key={result.id}
                className={cn(
                  "bg-white rounded-lg border shadow-sm overflow-hidden transition-colors",
                  isSelected
                    ? "border-material-blue bg-material-blue-light"
                    : "border-gray-200",
                )}
              >
                {/* Header with metadata and checkbox */}
                <div
                  className={cn(
                    "px-4 py-2 border-b",
                    isSelected
                      ? "bg-blue-50 border-blue-200"
                      : "bg-gray-50 border-gray-200",
                  )}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <Checkbox
                        checked={isSelected}
                        onCheckedChange={() =>
                          onToggleResultSelection(result.id)
                        }
                        className="data-[state=checked]:bg-material-blue data-[state=checked]:border-material-blue"
                      />
                      <span className="text-sm font-bold text-material-text-secondary min-w-[2rem]">
                        {resultNumber}.
                      </span>
                      <div className="flex items-center gap-4 text-sm flex-1 min-w-0">
                        <span className="text-material-text-primary font-medium flex-shrink-0">
                          {result.year}
                        </span>
                        <span className="text-material-text-primary flex-shrink-0">
                          <span className="font-bold">T:</span>{" "}
                          {targetText.needsTooltip ? (
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <span className="cursor-help">
                                  {targetText.truncated}
                                </span>
                              </TooltipTrigger>
                              <TooltipContent>
                                <p>{result.target}</p>
                              </TooltipContent>
                            </Tooltip>
                          ) : (
                            targetText.truncated
                          )}
                        </span>
                        <span className="text-material-text-primary flex-shrink-0">
                          <span className="font-bold">A:</span>{" "}
                          {acquirerText.needsTooltip ? (
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <span className="cursor-help">
                                  {acquirerText.truncated}
                                </span>
                              </TooltipTrigger>
                              <TooltipContent>
                                <p>{result.acquirer}</p>
                              </TooltipContent>
                            </Tooltip>
                          ) : (
                            acquirerText.truncated
                          )}
                        </span>
                        <span
                          className="text-material-text-secondary"
                          title={`${result.articleTitle} >> ${result.sectionTitle}`}
                        >
                          {result.articleTitle} &gt;&gt; {result.sectionTitle}
                        </span>
                      </div>
                    </div>

                    {/* Open Agreement Button */}
                    <div className="ml-4">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => onOpenAgreement(result)}
                        className="flex items-center gap-2 text-material-blue hover:bg-material-blue-light"
                      >
                        <ExternalLink className="w-4 h-4" />
                        <span className="hidden sm:inline">Open Agreement</span>
                      </Button>
                    </div>
                  </div>
                </div>

                {/* Clause text */}
                <div className="p-4">
                  <div
                    className="h-36 overflow-y-auto text-sm text-material-text-primary leading-relaxed"
                    style={{
                      scrollbarWidth: "thin",
                      scrollbarColor: "#e5e7eb #f9fafb",
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
