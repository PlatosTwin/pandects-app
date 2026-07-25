import { Link } from "react-router-dom";
import {
  ArrowUpRight,
  BadgeCheck,
  Copy,
  Check,
  FileText,
  Layers,
  Link2,
} from "lucide-react";
import { cn } from "@/lib/utils";
import {
  BREAKPOINT_SM,
  DEFAULT_TRUNCATION_LENGTH,
} from "@/lib/constants";
import { truncateText } from "@/lib/text-utils";
import { Checkbox } from "@/components/ui/checkbox";
import { Button } from "@/components/ui/button";
import { Badge, badgeVariants } from "@/components/ui/badge";
import { XMLRenderer } from "@/components/XMLRenderer";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ResultsToolbar } from "@/components/ResultsToolbar";
import { TooltipProvider } from "@/components/ui/tooltip";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { AdaptiveTooltip } from "@/components/ui/adaptive-tooltip";
import { useToast } from "@/hooks/use-toast";
import { FlagAsInaccurateButton } from "@/components/FlagAsInaccurateButton";
import { StarButton } from "@/components/StarButton";
import type { SearchResult } from "@shared/sections";

import { memo, useCallback, useEffect, useRef, useState } from "react";

const tailUuid = (uuid: string, n = 6) => uuid.slice(-n);

const formatCurrency = (value: string | null | undefined) => {
  if (!value) return null;
  const num = parseFloat(value);
  if (isNaN(num)) return value;
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(num);
};

interface SearchResultsTableProps {
  searchResults: SearchResult[];
  selectedResults: Set<string>;
  clauseTypePathByStandardId: Record<string, readonly string[]>;
  sort_by?: "year" | "target" | "acquirer";
  sort_direction: "asc" | "desc";
  onToggleResultSelection: (resultId: string) => void;
  onToggleSelectAll: () => void;
  onOpenAgreement: (result: SearchResult, position: number) => void;
  getAgreementHref: (result: SearchResult) => string;
  onSortResults: (sort_by: "year" | "target" | "acquirer") => void;
  onToggleSortDirection: () => void;
  density?: "comfy" | "compact";
  onDensityChange?: (density: "comfy" | "compact") => void;
  currentPage?: number;
  page_size?: number;
  className?: string;
}

export function SearchResultsTable({
  searchResults,
  selectedResults,
  clauseTypePathByStandardId,
  sort_by = "year",
  sort_direction,
  onToggleResultSelection,
  onToggleSelectAll,
  onOpenAgreement,
  getAgreementHref,
  onSortResults,
  onToggleSortDirection,
  density = "comfy",
  onDensityChange,
  currentPage = 1,
  page_size = 25,
  className,
}: SearchResultsTableProps) {
  const { toast } = useToast();
  const [expandedResults, setExpandedResults] = useState<Set<string>>(
    () => new Set(),
  );
  const [expandableResults, setExpandableResults] = useState<Set<string>>(
    () => new Set(),
  );
  const snippetRefs = useRef(new Map<string, HTMLDivElement | null>());
  const copySuccessTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );
  const [copiedForResult, setCopiedForResult] = useState<{
    resultId: string;
    kind: "section" | "agreement" | "link";
  } | null>(null);
  const pendingCopyRef = useRef<{
    text: string;
    label: string;
    resultId: string;
    kind: "section" | "agreement" | "link";
  } | null>(null);

  const buildSectionLinkUrl = useCallback(
    (result: SearchResult) => {
      if (typeof window === "undefined") return "";
      return new URL(getAgreementHref(result), window.location.origin).toString();
    },
    [getAgreementHref],
  );

  const registerSnippetRef = useCallback(
    (resultId: string, node: HTMLDivElement | null) => {
      if (node) {
        snippetRefs.current.set(resultId, node);
      } else {
        snippetRefs.current.delete(resultId);
      }
    },
    [],
  );

  const performCopy = useCallback(
    async (
      text: string,
      label: string,
      resultId: string,
      kind: "section" | "agreement" | "link",
    ) => {
      try {
        await navigator.clipboard.writeText(text);
        // Store pending copy to process after dropdown closes
        pendingCopyRef.current = { text, label, resultId, kind };
      } catch {
        // For errors, we can show immediately since dropdown might not be open
        setTimeout(() => {
          toast({ title: "Copy failed", variant: "destructive" });
        }, 0);
      }
    },
    [],
  );

  const processPendingCopy = useCallback(() => {
    if (pendingCopyRef.current) {
      const { label, resultId, kind } = pendingCopyRef.current;
      pendingCopyRef.current = null;
      // Use double RAF + setTimeout to ensure dropdown animation completes
      // and avoid forced synchronous layout (reflow)
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          // Defer toast and state updates to avoid reflow
          setTimeout(() => {
            toast({ title: `${label} copied` });
            if (copySuccessTimeoutRef.current) {
              clearTimeout(copySuccessTimeoutRef.current);
            }
            setCopiedForResult({ resultId, kind });
            copySuccessTimeoutRef.current = setTimeout(() => {
              setCopiedForResult(null);
              copySuccessTimeoutRef.current = null;
            }, 1000);
          }, 0);
        });
      });
    }
  }, [toast]);

  useEffect(
    () => () => {
      if (copySuccessTimeoutRef.current) {
        clearTimeout(copySuccessTimeoutRef.current);
      }
    },
    [],
  );
  const allSelected =
    searchResults.length > 0 &&
    searchResults.every((result) => selectedResults.has(result.id));
  const someSelected =
    searchResults.some((result) => selectedResults.has(result.id)) &&
    !allSelected;

  const toggleExpandedResult = useCallback((resultId: string) => {
    setExpandedResults((prev) => {
      const next = new Set(prev);
      if (next.has(resultId)) {
        next.delete(resultId);
      } else {
        next.add(resultId);
      }
      return next;
    });
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const isMobile = window.innerWidth < BREAKPOINT_SM;
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
      <ResultsToolbar
        selection={{
          allSelected,
          someSelected,
          onToggleSelectAll,
          selectAllLabel: "Select all results",
          countLabel:
            selectedResults.size > 0
              ? `${selectedResults.size} of ${searchResults.length} selected`
              : "Select all",
        }}
        sortBy={sort_by}
        sortDirection={sort_direction}
        onSortResults={onSortResults}
        onToggleSortDirection={onToggleSortDirection}
        sortSelectLabel="Sort section results by"
        density={density}
        onDensityChange={onDensityChange}
      />

      {/* Results Grid */}
      <TooltipProvider>
        <div
          role="list"
          className={cn(
            "grid gap-4",
            density === "compact" ? "sm:gap-2" : "sm:gap-4",
          )}
        >
          {searchResults.map((result, index) => (
            <SearchResultRow
              key={result.id}
              result={result}
              resultNumber={(currentPage - 1) * page_size + index + 1}
              isSelected={selectedResults.has(result.id)}
              isExpanded={expandedResults.has(result.id)}
              isExpandable={expandableResults.has(result.id)}
              isCopied={copiedForResult?.resultId === result.id}
              clauseTypePathByStandardId={clauseTypePathByStandardId}
              density={density}
              getAgreementHref={getAgreementHref}
              onOpenAgreement={onOpenAgreement}
              onToggleResultSelection={onToggleResultSelection}
              onToggleExpanded={toggleExpandedResult}
              performCopy={performCopy}
              processPendingCopy={processPendingCopy}
              buildSectionLinkUrl={buildSectionLinkUrl}
              registerSnippetRef={registerSnippetRef}
            />
          ))}
        </div>
      </TooltipProvider>
    </div>
  );
}

interface SearchResultRowProps {
  result: SearchResult;
  resultNumber: number;
  isSelected: boolean;
  isExpanded: boolean;
  isExpandable: boolean;
  isCopied: boolean;
  clauseTypePathByStandardId: Record<string, readonly string[]>;
  density: "comfy" | "compact";
  getAgreementHref: (result: SearchResult) => string;
  onOpenAgreement: (result: SearchResult, position: number) => void;
  onToggleResultSelection: (resultId: string) => void;
  onToggleExpanded: (resultId: string) => void;
  performCopy: (
    text: string,
    label: string,
    resultId: string,
    kind: "section" | "agreement" | "link",
  ) => void;
  processPendingCopy: () => void;
  buildSectionLinkUrl: (result: SearchResult) => string;
  registerSnippetRef: (resultId: string, node: HTMLDivElement | null) => void;
}

// Memoized so table-level state changes (e.g. the transient copy feedback)
// don't re-render — and re-run the XMLRenderer parse for — every other row.
const SearchResultRow = memo(function SearchResultRow({
  result,
  resultNumber,
  isSelected,
  isExpanded,
  isExpandable,
  isCopied,
  clauseTypePathByStandardId,
  density,
  getAgreementHref,
  onOpenAgreement,
  onToggleResultSelection,
  onToggleExpanded,
  performCopy,
  processPendingCopy,
  buildSectionLinkUrl,
  registerSnippetRef,
}: SearchResultRowProps) {
  const targetText = truncateText(
    result.target,
    DEFAULT_TRUNCATION_LENGTH,
  );
  const acquirerText = truncateText(
    result.acquirer,
    DEFAULT_TRUNCATION_LENGTH,
  );
  const canExpand = isExpandable || isExpanded;
  const standardIds = result.standard_id
    .map((value) => value.trim())
    .filter(Boolean);
  const matchedStandardId = standardIds.find(
    (value) => clauseTypePathByStandardId[value],
  );
  const standard_id = matchedStandardId ?? standardIds[0] ?? null;
  const clauseTypePath = standard_id
    ? clauseTypePathByStandardId[standard_id]
    : undefined;
  const clauseTypeLabel = clauseTypePath
    ?.map((part) => part.trim())
    .join(" \u2022 ");
  const clauseTypeText = clauseTypeLabel
    ? truncateText(clauseTypeLabel, DEFAULT_TRUNCATION_LENGTH)
    : null;
  const showDevFallbackPill =
    import.meta.env.DEV && (!clauseTypePath || !clauseTypeLabel);

  const hasNewFields =
    result.transaction_price_total ||
    result.deal_status ||
    result.deal_type ||
    result.purpose;

  return (
    <div
      role="listitem"
      className={cn(
        "group relative overflow-hidden rounded-lg border border-l-4 bg-card shadow-sm transition-all hover:shadow-md",
        isSelected
          ? "border-primary/40 border-l-primary bg-primary/5 shadow-md"
          : "border-border border-l-primary/35 hover:border-l-primary/60",
      )}
    >
      {/* Header with metadata and checkbox */}
      <div
        className={cn(
          "border-b px-4 py-3 sm:px-5 sm:py-4",
          density === "compact" && "sm:py-3",
          isSelected
            ? "bg-primary/10 border-primary/20"
            : "bg-muted/20 border-border",
        )}
      >
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div className="flex min-w-0 flex-1 items-start gap-3">
            {/* Per-row selection follows the desktop-only bulk action
                model for search results. */}
            <div className="hidden shrink-0 items-center justify-center pt-0.5 sm:flex">
              <Checkbox
                checked={isSelected}
                onCheckedChange={() =>
                  onToggleResultSelection(result.id)
                }
                className="data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                aria-label={`Select result ${resultNumber}`}
              />
            </div>
            <div className="min-w-0 flex-1">
              {/* Top row: Number, Year, Verified, Clause Type */}
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs font-semibold tabular-nums text-muted-foreground">
                  #{resultNumber}
                </span>
                <Badge
                  variant="outline"
                  className="px-2 py-0.5 font-medium"
                >
                  {result.year}
                </Badge>
                {result.verified ? (
                  <AdaptiveTooltip
                    trigger={
                      <button
                        type="button"
                        aria-label="Verified agreement"
                        className={cn(
                          badgeVariants({ variant: "success" }),
                          "hidden gap-1 hover:bg-emerald-500/15 sm:inline-flex",
                        )}
                      >
                        <BadgeCheck
                          className="h-3.5 w-3.5"
                          aria-hidden="true"
                        />
                        <span className="hidden sm:inline">
                          Verified
                        </span>
                      </button>
                    }
                    content={
                      <p>This agreement has been verified by hand.</p>
                    }
                    tooltipProps={{ side: "bottom" }}
                    popoverProps={{
                      side: "bottom",
                      className: "w-auto max-w-sm p-2 text-sm",
                    }}
                  />
                ) : null}
                {clauseTypeLabel ? (
                  <AdaptiveTooltip
                    trigger={
                      <button
                        type="button"
                        aria-label={`Clause type: ${clauseTypeLabel}`}
                        className={cn(
                          badgeVariants({ variant: "metadata" }),
                          "hidden max-w-[28rem] min-w-0 cursor-help sm:inline-flex lg:max-w-[36rem]",
                        )}
                      >
                        <span title={clauseTypeLabel}>
                          {clauseTypeText?.truncated}
                        </span>
                      </button>
                    }
                    content={
                      clauseTypePath ? (
                        <div className="space-y-1">
                          {clauseTypePath.map((part, partIndex) => (
                            <p key={`${partIndex}-${part}`}>{part}</p>
                          ))}
                        </div>
                      ) : null
                    }
                    tooltipProps={{ className: "max-w-sm" }}
                    popoverProps={{
                      className: "w-auto max-w-sm p-2 text-sm",
                    }}
                  />
                ) : showDevFallbackPill ? (
                  <AdaptiveTooltip
                    trigger={
                      <button
                        type="button"
                        aria-label="Clause type unavailable"
                        className={cn(
                          badgeVariants({ variant: "metadata" }),
                          "hidden max-w-[28rem] min-w-0 cursor-help sm:inline-flex lg:max-w-[36rem]",
                        )}
                      >
                        <span>Clause type unavailable</span>
                      </button>
                    }
                    content={
                      standard_id ? (
                        <>
                          <p>standard_id: {standard_id}</p>
                          <p>
                            Not found in the clause-type mapping used
                            by the frontend.
                          </p>
                        </>
                      ) : (
                        <>
                          <p>
                            Missing standard_id in search results.
                          </p>
                          <p>
                            Restart the Flask API or deploy the latest
                            backend so `/v1/sections` returns
                            `standard_id`.
                          </p>
                        </>
                      )
                    }
                    tooltipProps={{ className: "max-w-sm" }}
                    popoverProps={{
                      className: "w-auto max-w-sm p-2 text-sm",
                    }}
                  />
                ) : null}
              </div>

              {/* Main info: Target and Acquirer */}
              <div className="mt-2 min-w-0">
                <h2
                  className="min-w-0 break-words text-lg font-bold leading-snug text-foreground"
                  title={result.target}
                >
                  <span className="mr-1.5 align-middle text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                    Target
                  </span>
                  {targetText.needsTooltip ? (
                    <AdaptiveTooltip
                      trigger={
                        <button
                          type="button"
                          aria-label={`Target: ${result.target}`}
                          className="cursor-help break-words bg-transparent p-0 text-left font-bold focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                        >
                          {targetText.truncated}
                        </button>
                      }
                      content={<p>{result.target}</p>}
                      popoverProps={{
                        className: "w-auto max-w-sm p-2 text-sm",
                      }}
                    />
                  ) : (
                    <span className="break-words">
                      {targetText.truncated}
                    </span>
                  )}
                </h2>
                <div
                  className="mt-0.5 min-w-0 break-words text-base text-muted-foreground"
                  title={result.acquirer}
                >
                  <span className="mr-1.5 align-middle text-[11px] font-semibold uppercase tracking-wide">
                    Acquirer
                  </span>
                  {acquirerText.needsTooltip ? (
                    <AdaptiveTooltip
                      trigger={
                        <button
                          type="button"
                          aria-label={`Acquirer: ${result.acquirer}`}
                          className="cursor-help break-words bg-transparent p-0 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                        >
                          {acquirerText.truncated}
                        </button>
                      }
                      content={<p>{result.acquirer}</p>}
                      popoverProps={{
                        className: "w-auto max-w-sm p-2 text-sm",
                      }}
                    />
                  ) : (
                    <span className="break-words">
                      {acquirerText.truncated}
                    </span>
                  )}
                </div>
              </div>

              {/* Article and Section pills - separate */}
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <Badge
                  variant="muted"
                  className="hidden gap-1 sm:inline-flex"
                  title={result.article_title}
                >
                  <Layers
                    className="h-3 w-3 shrink-0"
                    aria-hidden="true"
                  />
                  <span className="max-w-[12rem] truncate sm:max-w-[20rem]">
                    {
                      truncateText(
                        result.article_title,
                        DEFAULT_TRUNCATION_LENGTH,
                      ).truncated
                    }
                  </span>
                </Badge>
                <Badge
                  variant="muted"
                  className="gap-1"
                  title={result.section_title}
                >
                  <FileText
                    className="h-3 w-3 shrink-0"
                    aria-hidden="true"
                  />
                  <span className="max-w-[12rem] truncate sm:max-w-[20rem]">
                    {
                      truncateText(
                        result.section_title,
                        DEFAULT_TRUNCATION_LENGTH,
                      ).truncated
                    }
                  </span>
                </Badge>
              </div>

              {/* New fields row - hidden on mobile */}
              {hasNewFields && (
                <div className="mt-2 hidden flex-wrap items-center gap-2 sm:flex">
                  {result.transaction_price_total && (
                    <Badge
                      variant="outline"
                      className="px-2 py-0.5 text-xs font-medium"
                    >
                      {formatCurrency(result.transaction_price_total)}
                    </Badge>
                  )}
                  {result.deal_status && (
                    <Badge
                      variant="outline"
                      className="px-2 py-0.5 text-xs"
                    >
                      {result.deal_status}
                    </Badge>
                  )}
                  {result.deal_type && (
                    <Badge
                      variant="outline"
                      className="px-2 py-0.5 text-xs"
                    >
                      {result.deal_type}
                    </Badge>
                  )}
                  {result.purpose && (
                    <Badge
                      variant="outline"
                      className="px-2 py-0.5 text-xs"
                      title={result.purpose}
                    >
                      <span className="max-w-[12rem] truncate">
                        {
                          truncateText(
                            result.purpose,
                            DEFAULT_TRUNCATION_LENGTH,
                          ).truncated
                        }
                      </span>
                    </Badge>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Actions: Flag + Open Agreement */}
          <div className="flex w-full flex-wrap items-center gap-2 sm:ml-4 sm:w-auto sm:justify-end">
            <StarButton
              itemType="section"
              itemUuid={result.section_uuid}
              context={{
                agreement_uuid: result.agreement_uuid,
                target: result.target,
                acquirer: result.acquirer,
                article_title: result.article_title,
                section_title: result.section_title,
                standard_id: result.standard_id,
                xml: result.xml,
                position: resultNumber,
              }}
              className="shrink-0"
            />
            <FlagAsInaccurateButton
              source="search_result"
              agreement_uuid={result.agreement_uuid}
              section_uuid={result.section_uuid}
              className="shrink-0"
            />
            <Button
              size="sm"
              asChild
              className="flex h-11 min-w-0 flex-1 items-center justify-center gap-1.5 px-4 shadow-sm sm:h-9 sm:w-auto sm:flex-none sm:px-3"
            >
              <Link
                to={getAgreementHref(result)}
                onClick={() => onOpenAgreement(result, resultNumber)}
                aria-label={`Open agreement for result ${resultNumber}: ${result.target} acquired by ${result.acquirer}`}
              >
                Open agreement
                <ArrowUpRight className="h-4 w-4" aria-hidden="true" />
              </Link>
            </Button>
          </div>
        </div>
      </div>

      {/* Content area */}
      <div
        className={cn(
          "p-4",
          density === "compact" ? "sm:p-3" : "sm:p-4",
        )}
      >
        {/* Clause text */}
        {result.xml ? (
          <>
            <div
              className={cn(
                "relative rounded-md border border-border bg-muted/20 p-3",
                density === "compact"
                  ? "sm:h-28 sm:p-2"
                  : "sm:h-36 sm:p-3",
              )}
            >
              {/* Copy button - positioned in top-right corner of inner box */}
              <div className="absolute right-2 top-2 z-10">
                <DropdownMenu
                  onOpenChange={(open) => {
                    if (!open) {
                      // Process pending copy after dropdown closes
                      processPendingCopy();
                    }
                  }}
                >
                  <DropdownMenuTrigger asChild>
                    <button
                      type="button"
                      title="Copy…"
                      className={cn(
                        "flex min-h-[44px] min-w-[44px] shrink-0 items-center justify-center rounded-md border border-border bg-background shadow-sm text-muted-foreground transition-colors hover:bg-muted/40 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background data-[state=open]:bg-muted/40 data-[state=open]:text-foreground sm:h-7 sm:w-7",
                        isCopied &&
                          "bg-primary/10 text-primary border-primary/20",
                      )}
                      aria-label="Copy"
                      aria-haspopup="menu"
                    >
                      {isCopied ? (
                        <Check
                          className="h-3.5 w-3.5 text-primary"
                          aria-hidden="true"
                        />
                      ) : (
                        <Copy
                          className="h-3.5 w-3.5"
                          aria-hidden="true"
                        />
                      )}
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent
                    align="end"
                    side="bottom"
                    sideOffset={4}
                    className="min-w-[14rem]"
                  >
                    <DropdownMenuItem
                      onClick={() =>
                        performCopy(
                          result.section_uuid,
                          "Section UUID",
                          result.id,
                          "section",
                        )
                      }
                      title="Copy section UUID"
                      className="flex cursor-pointer items-center gap-2 py-2"
                    >
                      <span
                        className="flex h-5 w-5 shrink-0 items-center justify-center rounded bg-muted text-xs font-medium text-muted-foreground"
                        aria-hidden="true"
                      >
                        §
                      </span>
                      <span className="min-w-0 flex-1 truncate">
                        Section UUID
                      </span>
                      <span
                        className="ml-2 shrink-0 font-mono text-xs text-muted-foreground tabular-nums"
                        title={result.section_uuid}
                      >
                        …{tailUuid(result.section_uuid)}
                      </span>
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onClick={() =>
                        performCopy(
                          result.agreement_uuid,
                          "Agreement UUID",
                          result.id,
                          "agreement",
                        )
                      }
                      title="Copy agreement UUID"
                      className="flex cursor-pointer items-center gap-2 py-2"
                    >
                      <span className="flex h-5 w-5 shrink-0 items-center justify-center">
                        <FileText
                          className="h-4 w-4 text-muted-foreground"
                          aria-hidden="true"
                        />
                      </span>
                      <span className="min-w-0 flex-1 truncate">
                        Agreement UUID
                      </span>
                      <span
                        className="ml-2 shrink-0 font-mono text-xs text-muted-foreground tabular-nums"
                        title={result.agreement_uuid}
                      >
                        …{tailUuid(result.agreement_uuid)}
                      </span>
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onClick={() => {
                        // Extract text content from XML by stripping tags
                        const textContent = result.xml
                          ? result.xml
                              .replace(/<[^>]+>/g, "")
                              .replace(/\s+/g, " ")
                              .trim()
                          : "";
                        performCopy(
                          textContent,
                          "Section text",
                          result.id,
                          "section",
                        );
                      }}
                      title="Copy section text"
                      className="flex cursor-pointer items-center gap-2 py-2"
                    >
                      <span className="flex h-5 w-5 shrink-0 items-center justify-center">
                        <FileText
                          className="h-4 w-4 text-muted-foreground"
                          aria-hidden="true"
                        />
                      </span>
                      <span className="min-w-0 flex-1 truncate">
                        Section text
                      </span>
                    </DropdownMenuItem>
                    <DropdownMenuItem
                      onClick={() =>
                        performCopy(
                          buildSectionLinkUrl(result),
                          "Agreement link",
                          result.id,
                          "link",
                        )
                      }
                      title="Copy link to this section"
                      className="flex cursor-pointer items-center gap-2 py-2"
                    >
                      <span className="flex h-5 w-5 shrink-0 items-center justify-center">
                        <Link2
                          className="h-4 w-4 text-muted-foreground"
                          aria-hidden="true"
                        />
                      </span>
                      <span className="min-w-0 flex-1 truncate">
                        Agreement link
                      </span>
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
              <div
                ref={(node) => registerSnippetRef(result.id, node)}
                className={cn(
                  "text-sm leading-relaxed text-foreground pr-10",
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
                  onClick={() => onToggleExpanded(result.id)}
                  className="h-8 px-0"
                >
                  {isExpanded ? "Show less" : "Show more"}
                </Button>
              </div>
            ) : null}
          </>
        ) : (
          <div
            className={cn(
              density === "compact" ? "sm:h-28" : "sm:h-36",
            )}
          >
            <Alert className="h-full">
              <AlertTitle>Section text not available</AlertTitle>
              <AlertDescription>
                This section has no stored text.
              </AlertDescription>
            </Alert>
          </div>
        )}
      </div>
    </div>
  );
});
