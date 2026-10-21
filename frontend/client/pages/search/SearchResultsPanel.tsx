import type { ReactNode } from "react";
import { Download, MoreHorizontal, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { ErrorState, EmptyState } from "@/components/StateMessage";
import { formatNumber } from "@/lib/format-utils";
import { TAX_COMPARE_MAX, TAX_COMPARE_MIN } from "@/lib/tax-compare-handoff";
import { cn } from "@/lib/utils";
import type { SearchMode } from "@shared/search";
import { MODE_NOUNS } from "./search-ui";

export interface SearchResultsPanelProps {
  mode: SearchMode;
  hasSearched: boolean;
  isSearching: boolean;
  /** The search has been running for a while (shows "Still searching…"). */
  isSlow: boolean;
  /** A commit is in flight and no rows from any earlier commit are on screen. */
  isFirstLoad: boolean;
  error: unknown;
  errorTitle?: string;
  errorBody?: string;
  onRetry?: () => void;
  signInHref: string;
  totalCount: number;
  totalCountIsApproximate: boolean;
  resultsLength: number;
  /** Mobile "Filters (n)" button (opens the sheet). */
  mobileFilters: ReactNode;
  /** Sort + density controls. */
  sortControls: ReactNode;
  /** Active filter chips with the one Clear all. */
  chips: ReactNode;
  /** Page-level select-all (sections and tax; the deals table has its own). */
  selectPage?: { checked: boolean | "indeterminate"; onToggle: () => void; label: string };
  onDownloadAll: () => void;
  selectedSize: number;
  onDownloadSelected: () => void;
  onClearSelection: () => void;
  /** Tax mode: compare the selected clauses. */
  onCompare?: () => void;
  emptyState: ReactNode;
  results: ReactNode;
  pagination: ReactNode;
  skeleton: ReactNode;
}

function countText(mode: SearchMode, total: number, approximate: boolean): string {
  const noun = total === 1 ? MODE_NOUNS[mode].one : MODE_NOUNS[mode].many;
  return `${approximate ? "about " : ""}${formatNumber(total)} ${noun}`;
}

function SelectionBar({
  mode,
  selectedSize,
  onDownloadSelected,
  onClearSelection,
  onCompare,
}: Pick<SearchResultsPanelProps, "mode" | "selectedSize" | "onDownloadSelected" | "onClearSelection" | "onCompare">) {
  const compareDisabled = selectedSize < TAX_COMPARE_MIN || selectedSize > TAX_COMPARE_MAX;
  const compareHint =
    selectedSize < TAX_COMPARE_MIN
      ? `Select at least ${TAX_COMPARE_MIN} clauses to compare.`
      : `Compare up to ${TAX_COMPARE_MAX} clauses at once.`;
  return (
    <div
      role="region"
      aria-label="Selected results"
      className="pointer-events-none sticky bottom-4 z-30 mt-4 flex justify-center px-2 pb-[env(safe-area-inset-bottom)]"
    >
      <div className="pointer-events-auto flex max-w-full flex-wrap items-center gap-2 rounded-xl border border-border bg-popover/95 px-3 py-2 shadow-lg supports-[backdrop-filter]:bg-popover/80 supports-[backdrop-filter]:backdrop-blur">
        <span className="px-1 text-sm font-medium text-foreground" aria-live="polite">
          {selectedSize} selected
        </span>
        <Button size="sm" variant="outline" className="h-9 gap-1.5" onClick={onDownloadSelected}>
          <Download className="h-4 w-4" aria-hidden="true" />
          Download CSV
        </Button>
        {mode === "tax" && onCompare ? (
          <Tooltip>
            <TooltipTrigger asChild>
              <span>
                <Button size="sm" className="h-9" onClick={onCompare} disabled={compareDisabled}>
                  Compare ({selectedSize})
                </Button>
              </span>
            </TooltipTrigger>
            {compareDisabled ? <TooltipContent>{compareHint}</TooltipContent> : null}
          </Tooltip>
        ) : null}
        <Button
          size="sm"
          variant="ghost"
          className="h-9 w-9 p-0"
          onClick={onClearSelection}
          aria-label="Clear selection"
          title="Clear selection"
        >
          <X className="h-4 w-4" aria-hidden="true" />
        </Button>
      </div>
    </div>
  );
}

/**
 * Results column: a sticky header (count, chips, sort, overflow menu), the
 * results or a state message, bottom pagination, and a floating selection
 * bar once something is selected.
 */
export function SearchResultsPanel(props: SearchResultsPanelProps) {
  const {
    mode,
    hasSearched,
    isSearching,
    isSlow,
    isFirstLoad,
    error,
    totalCount,
    totalCountIsApproximate,
    resultsLength,
    selectedSize,
  } = props;
  const showRows = hasSearched && !error && !isFirstLoad && resultsLength > 0;
  const status = error
    ? "The search failed."
    : isSearching
      ? isSlow
        ? "Still searching…"
        : "Searching…"
      : hasSearched
        ? `${countText(mode, totalCount, totalCountIsApproximate)} found.`
        : "";

  return (
    <section
      id="search-results"
      tabIndex={-1}
      aria-labelledby="search-results-heading"
      className="flex min-w-0 flex-1 flex-col outline-none"
    >
      <h2 id="search-results-heading" className="sr-only">
        Results
      </h2>
      <div className="z-10 border-b border-border bg-background/95 px-4 py-2 supports-[backdrop-filter]:bg-background/80 supports-[backdrop-filter]:backdrop-blur sm:px-6 lg:sticky lg:top-[calc(4rem+var(--search-bar-h,0px))] lg:px-6">
        <div className="flex min-h-10 flex-wrap items-center gap-x-3 gap-y-2">
          {props.mobileFilters}
          {props.selectPage && showRows ? (
            <Checkbox
              checked={props.selectPage.checked}
              onCheckedChange={props.selectPage.onToggle}
              aria-label={props.selectPage.label}
              className="hidden sm:inline-flex"
            />
          ) : null}
          <p
            id="search-results-status"
            role="status"
            aria-live="polite"
            className="flex min-w-0 flex-1 items-center gap-2 text-xs font-medium leading-tight text-foreground sm:flex-none sm:text-sm"
          >
            {isSearching ? (
              <span
                className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent"
                aria-hidden="true"
              />
            ) : null}
            <span>
              {isSearching && (isFirstLoad || isSlow)
                ? status
                : hasSearched && !error
                  ? countText(mode, totalCount, totalCountIsApproximate)
                  : status}
            </span>
          </p>
          <div className="hidden min-w-0 flex-1 lg:block">{props.chips}</div>
          <div className={cn("ml-auto flex shrink-0 items-center gap-1", !hasSearched && "hidden")}>
            {props.sortControls}
            <DropdownMenu>
              <DropdownMenuTrigger asChild className="hidden sm:inline-flex">
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-9 w-9 p-0 sm:h-8 sm:w-8"
                  aria-label="More result actions"
                  title="More"
                >
                  <MoreHorizontal className="h-4 w-4" aria-hidden="true" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem disabled={!showRows} onSelect={props.onDownloadAll}>
                  <Download className="mr-2 h-4 w-4" aria-hidden="true" />
                  Download CSV (all results)
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
        {/* Below lg the chips get their own row; the sheet hides what's applied. */}
        <div className="mt-2 empty:hidden lg:hidden">{props.chips}</div>
      </div>

      <div className="flex-1 px-4 py-4 sm:px-6">
        {!hasSearched ? (
          props.emptyState
        ) : error ? (
          <ErrorState
            error={error as { status?: number; code?: string; message?: string }}
            title={props.errorTitle}
            body={props.errorBody}
            onRetry={props.onRetry}
            signInHref={props.signInHref}
          />
        ) : isFirstLoad ? (
          <div aria-busy="true">
            {isSlow ? (
              <p className="mb-3 text-sm text-muted-foreground">
                Still searching… Broad keyword searches can take up to 20 seconds.
              </p>
            ) : null}
            {props.skeleton}
          </div>
        ) : resultsLength === 0 ? (
          <EmptyState
            title={`No ${MODE_NOUNS[mode].many} match these filters`}
            body="Remove a filter or try a different keyword."
          />
        ) : (
          <>
            <div
              aria-busy={isSearching}
              className={cn("transition-opacity", isSearching && "pointer-events-none opacity-60")}
            >
              {props.results}
            </div>
            <div className="mt-6">{props.pagination}</div>
          </>
        )}
        {selectedSize > 0 && showRows ? (
          <SelectionBar
            mode={mode}
            selectedSize={selectedSize}
            onDownloadSelected={props.onDownloadSelected}
            onClearSelection={props.onClearSelection}
            onCompare={props.onCompare}
          />
        ) : null}
      </div>
    </section>
  );
}
