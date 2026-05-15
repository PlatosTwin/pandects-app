import { Suspense, type ReactNode } from "react";
import { Building2, FileText, Layers, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { SearchMode } from "@shared/search";
import {
  SearchPagination,
  SearchPaginationFallback,
} from "./lazy";

interface SearchResultsPanelProps {
  searchMode: SearchMode;
  hasHydrated: boolean;
  isSearching: boolean;
  hasSearched: boolean;
  totalCount: number;
  totalCountIsApproximate: boolean;
  totalPages: number;
  hasNext: boolean;
  hasPrev: boolean;
  currentPage: number;
  pageSize: number;
  accessTier: string;
  selectedSize: number;
  onGoToPage: (page: number) => void;
  onPageSizeChange: (size: number) => void;
  onDownloadCSV: () => void;
  onClearSelection: () => void;
  resultsList: ReactNode;
}

export function SearchResultsPanel({
  searchMode,
  hasHydrated,
  isSearching,
  hasSearched,
  totalCount,
  totalCountIsApproximate,
  totalPages,
  hasNext,
  hasPrev,
  currentPage,
  pageSize,
  accessTier,
  selectedSize,
  onGoToPage,
  onPageSizeChange,
  onDownloadCSV,
  onClearSelection,
  resultsList,
}: SearchResultsPanelProps) {
  const isLimited = accessTier === "anonymous";

  const renderPagination = () =>
    hasHydrated ? (
      <Suspense fallback={<SearchPaginationFallback />}>
        <SearchPagination
          currentPage={currentPage}
          totalPages={totalPages}
          pageSize={pageSize}
          totalCount={totalCount}
          totalCountIsApproximate={totalCountIsApproximate}
          hasNext={hasNext}
          hasPrev={hasPrev}
          onPageChange={onGoToPage}
          onPageSizeChange={onPageSizeChange}
          isLoading={isSearching}
          isLimited={isLimited}
        />
      </Suspense>
    ) : (
      <SearchPaginationFallback />
    );

  return (
    <main className="flex-1 overflow-auto" aria-labelledby="search-page-title">
      <div className="px-4 py-4 sm:px-8 sm:py-5">
        <div id="search-results-status" className="sr-only" role="status" aria-live="polite">
          {isSearching
            ? "Searching."
            : hasSearched
              ? `${totalCount} ${searchMode === "sections" ? "sections" : searchMode === "tax" ? "tax clauses" : "deals"} found.`
              : "No search has been run."}
        </div>

        {!hasSearched && (
          <div className="mx-auto max-w-3xl space-y-4">
            <div className="rounded-2xl border border-border bg-card p-6 shadow-sm">
              <div className="flex items-start gap-4">
                <div className="mt-0.5 rounded-lg bg-primary/10 p-2 text-primary">
                  <Sparkles className="h-5 w-5" aria-hidden="true" />
                </div>
                <div className="min-w-0">
                  <h2 className="text-base font-semibold text-foreground">
                    {searchMode === "sections"
                      ? "Find specific sections across the corpus"
                      : searchMode === "tax"
                        ? "Find tax clause precedents across the corpus"
                        : "Find deals matching your criteria"}
                  </h2>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {searchMode === "sections"
                      ? "Pick filters to narrow the corpus, then search to load matched sections and jump straight into the relevant agreement passage."
                      : searchMode === "tax"
                        ? "Filter by tax clause type and deal metadata to surface precedent drafting language. Reps & warranties clauses are excluded by default."
                        : "Pick filters to narrow the corpus, then search to load matching deals with the sections that triggered each result."}
                  </p>
                </div>
              </div>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-xl border border-border bg-card/60 p-4">
                <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                  {searchMode === "sections" ? (
                    <FileText className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
                  ) : (
                    <Building2 className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
                  )}
                  Try filtering by
                </div>
                <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
                  <li>• Clause type (e.g. material adverse effect)</li>
                  <li>• Year, deal status, or attitude</li>
                  <li>• Target or acquirer counsel</li>
                </ul>
              </div>
              <div className="rounded-xl border border-border bg-card/60 p-4">
                <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                  <Layers className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
                  What you'll see
                </div>
                <p className="mt-2 text-sm text-muted-foreground">
                  {searchMode === "sections"
                    ? "Section text with the matched clause type, plus quick links to the full agreement."
                    : "Each unique deal grouped together, with the matched sections that brought it into the result set."}
                </p>
              </div>
            </div>
          </div>
        )}

        {hasSearched && (
          <div className="space-y-4">
            {totalCount === 0 ? (
              <div
                className="mx-auto max-w-3xl text-center py-12 text-muted-foreground"
                role="status"
                aria-live="polite"
              >
                <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" aria-hidden="true" />
                <p className="text-foreground font-medium">
                  {searchMode === "sections"
                    ? "No sections found."
                    : searchMode === "tax"
                      ? "No tax clauses found."
                      : "No deals found."}
                </p>
                <p className="text-sm mt-2">Try adjusting your filters and search again.</p>
              </div>
            ) : (
              <>
                {renderPagination()}
                {resultsList}

                {selectedSize > 0 && (
                  <div className="rounded-xl border border-border bg-muted/20 p-4 backdrop-blur supports-[backdrop-filter]:bg-muted/20">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                      <div className="text-sm text-muted-foreground">{selectedSize} selected</div>
                      <div className="flex flex-wrap gap-2">
                        <Button variant="outline" size="sm" onClick={onDownloadCSV}>
                          Download selected
                        </Button>
                        <Button variant="outline" size="sm" onClick={onClearSelection}>
                          Clear selection
                        </Button>
                      </div>
                    </div>
                  </div>
                )}

                {renderPagination()}
              </>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
