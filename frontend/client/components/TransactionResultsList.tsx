import { useState } from "react";
import { Link } from "react-router-dom";
import {
  ArrowDown,
  ArrowUp,
  ArrowUpRight,
  Building2,
  Calendar,
  ChevronDown,
  CircleDollarSign,
  ExternalLink,
  FileText,
  Gavel,
  Layers,
  ShieldCheck,
  Tag,
} from "lucide-react";
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
import {
  formatCompactCurrencyValue,
  formatDateValue,
  formatEnumValue,
} from "@/lib/format-utils";
import { cn } from "@/lib/utils";
import type { TransactionSearchResult } from "@shared/transactions";

interface TransactionResultsListProps {
  results: TransactionSearchResult[];
  getAgreementHref: (
    result: TransactionSearchResult,
    focusSectionUuid?: string | null,
  ) => string;
  showClauseContext?: boolean;
  clauseTypeLabelById?: Record<string, string>;
  currentPage?: number;
  pageSize?: number;
  className?: string;
  selectedResults?: Set<string>;
  onToggleResultSelection?: (agreementUuid: string) => void;
  onToggleSelectAll?: () => void;
  sortBy?: "year" | "target" | "acquirer";
  sortDirection?: "asc" | "desc";
  onSortResults?: (field: "year" | "target" | "acquirer") => void;
  onToggleSortDirection?: () => void;
  density?: "comfy" | "compact";
  onDensityChange?: (density: "comfy" | "compact") => void;
}

const INITIAL_MATCHED_VISIBLE = 3;

function formatParties(result: TransactionSearchResult) {
  const target = result.target?.trim() || "Unknown target";
  const acquirer = result.acquirer?.trim() || "Unknown acquirer";
  return { target, acquirer };
}

function MetaPill({
  icon: Icon,
  label,
  value,
  className,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "inline-flex min-w-0 items-center gap-1.5 rounded-full border border-border bg-background/60 px-2.5 py-1 text-xs",
        className,
      )}
    >
      <Icon className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
      <span className="text-muted-foreground">{label}</span>
      <span className="truncate font-medium text-foreground">{value}</span>
    </div>
  );
}

export function TransactionResultsList({
  results,
  getAgreementHref,
  showClauseContext = true,
  clauseTypeLabelById,
  currentPage = 1,
  pageSize = 25,
  className,
  selectedResults,
  onToggleResultSelection,
  onToggleSelectAll,
  sortBy = "year",
  sortDirection = "desc",
  onSortResults,
  onToggleSortDirection,
  density = "comfy",
  onDensityChange,
}: TransactionResultsListProps) {
  const [expandedMatches, setExpandedMatches] = useState<Set<string>>(
    () => new Set(),
  );

  const selectionEnabled = Boolean(onToggleResultSelection);
  const allSelected =
    selectionEnabled &&
    results.length > 0 &&
    results.every((r) => selectedResults?.has(r.agreement_uuid));
  const someSelected =
    selectionEnabled &&
    !allSelected &&
    results.some((r) => selectedResults?.has(r.agreement_uuid));

  const toggleExpandedMatches = (agreementUuid: string) => {
    setExpandedMatches((prev) => {
      const next = new Set(prev);
      if (next.has(agreementUuid)) next.delete(agreementUuid);
      else next.add(agreementUuid);
      return next;
    });
  };

  return (
    <div className={cn("space-y-4", className)}>
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        {selectionEnabled && results.length > 0 ? (
          <div className="hidden items-center gap-2 sm:flex">
            <Checkbox
              checked={allSelected ? true : someSelected ? "indeterminate" : false}
              onCheckedChange={() => onToggleSelectAll?.()}
              className="data-[state=checked]:bg-primary data-[state=checked]:border-primary"
              aria-label="Select all deals"
            />
            <span className="text-sm text-muted-foreground" aria-live="polite">
              {(selectedResults?.size ?? 0) > 0
                ? `${selectedResults?.size} of ${results.length} selected`
                : "Select all"}
            </span>
          </div>
        ) : null}

        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-end">
          <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-3">
            <div className="hidden items-center gap-2 sm:flex">
              <span className="hidden text-sm text-muted-foreground sm:inline">Density:</span>
              <ToggleGroup
                type="single"
                aria-label="Results density"
                value={density}
                onValueChange={(value) => {
                  if (value === "comfy" || value === "compact") onDensityChange?.(value);
                }}
                variant="outline"
                size="xs"
                className="justify-start"
              >
                <ToggleGroupItem value="compact" aria-label="Compact density" className="text-muted-foreground data-[state=on]:text-foreground">
                  Compact
                </ToggleGroupItem>
                <ToggleGroupItem value="comfy" aria-label="Comfy density" className="text-muted-foreground data-[state=on]:text-foreground">
                  Comfy
                </ToggleGroupItem>
              </ToggleGroup>
            </div>

            <div className="flex items-center gap-2">
              <label className="hidden text-sm text-muted-foreground sm:inline">Sort by:</label>
              <Select
                value={sortBy}
                onValueChange={(value) => onSortResults?.(value as "year" | "target" | "acquirer")}
              >
                <SelectTrigger className="h-9 w-full sm:w-[160px]" aria-label="Sort results by">
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
                className="h-10 w-10 p-1 hover:bg-muted/40 sm:h-8 sm:w-8"
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
      <ol
        role="list"
        aria-label="Deal search results"
        className={cn("space-y-4", density === "compact" && "sm:space-y-2")}
      >
      {results.map((result, index) => {
        const rank = (currentPage - 1) * pageSize + index + 1;
        const href = getAgreementHref(
          result,
          showClauseContext ? result.matched_sections[0]?.section_uuid ?? null : null,
        );
        const { target, acquirer } = formatParties(result);
        const titleLabel = `${target} — acquired by ${acquirer}`;
        const dealValue = formatCompactCurrencyValue(result.transaction_price_total);
        const isMatchesExpanded = expandedMatches.has(result.agreement_uuid);
        const matchedSections = isMatchesExpanded
          ? result.matched_sections
          : result.matched_sections.slice(0, INITIAL_MATCHED_VISIBLE);
        const hiddenMatchCount = Math.max(
          0,
          result.matched_sections.length - INITIAL_MATCHED_VISIBLE,
        );

        const isSelected =
          selectionEnabled && !!selectedResults?.has(result.agreement_uuid);
        return (
          <li
            key={result.agreement_uuid}
            className={cn(
              "group relative overflow-hidden rounded-xl border bg-card shadow-sm transition-all hover:border-border hover:shadow-md",
              isSelected ? "border-primary/40" : "border-border",
            )}
          >
            {/* Header */}
            <div
              className={cn(
                "border-b px-4 py-3 sm:px-5 sm:py-4",
                isSelected
                  ? "bg-primary/10 border-primary/20"
                  : "bg-muted/20 border-border",
              )}
            >
              <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                {selectionEnabled ? (
                  <div className="hidden shrink-0 items-center justify-center pt-0.5 sm:flex">
                    <Checkbox
                      checked={isSelected}
                      onCheckedChange={() =>
                        onToggleResultSelection?.(result.agreement_uuid)
                      }
                      className="data-[state=checked]:bg-primary data-[state=checked]:border-primary"
                      aria-label={`Select deal ${rank}`}
                    />
                  </div>
                ) : null}
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-xs font-semibold tabular-nums text-muted-foreground">
                      #{rank}
                    </span>
                    {result.year ? (
                      <Badge variant="outline" className="px-2 py-0.5 font-medium">
                        {result.year}
                      </Badge>
                    ) : null}
                    {result.deal_status ? (
                      <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
                        <ShieldCheck className="h-3 w-3" aria-hidden="true" />
                        {formatEnumValue(result.deal_status)}
                      </span>
                    ) : null}
                    {result.deal_type ? (
                      <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
                        <Tag className="h-3 w-3" aria-hidden="true" />
                        {formatEnumValue(result.deal_type)}
                      </span>
                    ) : null}
                  </div>
                  <h3
                    className="mt-2 text-base font-semibold leading-snug text-foreground sm:text-lg"
                    title={titleLabel}
                  >
                    <span className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                      Target
                    </span>{" "}
                    {target}
                  </h3>
                  <div
                    className="mt-0.5 text-sm text-muted-foreground"
                    title={`Acquirer: ${acquirer}`}
                  >
                    <span className="text-[11px] font-semibold uppercase tracking-wide">
                      Acquirer
                    </span>{" "}
                    {acquirer}
                  </div>
                  {result.transaction_consideration ? (
                    <div className="mt-1 text-xs text-muted-foreground">
                      Consideration: {formatEnumValue(result.transaction_consideration)}
                    </div>
                  ) : null}
                </div>
                <div className="flex flex-row-reverse items-center justify-between gap-3 sm:flex-col sm:items-end sm:justify-start">
                  {result.transaction_price_total !== null ? (
                    <div className="text-right">
                      <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                        Deal value
                      </div>
                      <div
                        className="text-lg font-semibold tabular-nums text-foreground sm:text-xl"
                        title={`$${result.transaction_price_total.toLocaleString()}`}
                      >
                        {dealValue}
                      </div>
                    </div>
                  ) : (
                    <div className="text-right text-xs text-muted-foreground">
                      Deal value unavailable
                    </div>
                  )}
                  <div className="flex items-center gap-1">
                    <FlagAsInaccurateButton
                      source="search_result"
                      agreement_uuid={result.agreement_uuid}
                      className="shrink-0"
                    />
                    <Button asChild size="sm" className="gap-1.5 shadow-sm">
                      <Link
                        to={href}
                        aria-label={`Open agreement ${titleLabel}`}
                      >
                        Open agreement
                        <ArrowUpRight className="h-4 w-4" aria-hidden="true" />
                      </Link>
                    </Button>
                  </div>
                </div>
              </div>
            </div>

            {/* Body */}
            <div className="space-y-4 px-4 py-4 sm:px-5">
              {/* Quick facts */}
              <div className="flex flex-wrap gap-2">
                {result.filing_date ? (
                  <MetaPill
                    icon={Calendar}
                    label="Filed"
                    value={formatDateValue(result.filing_date)}
                  />
                ) : null}
                {result.announce_date ? (
                  <MetaPill
                    icon={Calendar}
                    label="Announced"
                    value={formatDateValue(result.announce_date)}
                  />
                ) : null}
                {result.target_industry ? (
                  <MetaPill
                    icon={Building2}
                    label="Target industry"
                    value={result.target_industry}
                  />
                ) : null}
                {result.acquirer_industry ? (
                  <MetaPill
                    icon={Building2}
                    label="Acquirer industry"
                    value={result.acquirer_industry}
                  />
                ) : null}
                {result.purpose ? (
                  <MetaPill
                    icon={CircleDollarSign}
                    label="Purpose"
                    value={formatEnumValue(result.purpose)}
                  />
                ) : null}
                {result.attitude ? (
                  <MetaPill
                    icon={Gavel}
                    label="Attitude"
                    value={formatEnumValue(result.attitude)}
                  />
                ) : null}
                {result.url ? (
                  <a
                    href={result.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 rounded-full border border-border bg-background/60 px-2.5 py-1 text-xs text-muted-foreground transition-colors hover:border-primary/40 hover:bg-primary/5 hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                  >
                    <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
                    SEC filing
                  </a>
                ) : null}
              </div>

              {/* Matched sections */}
              {showClauseContext && result.matched_sections.length > 0 ? (
                <div className="rounded-lg border border-border bg-muted/20">
                  <div className="flex items-center justify-between gap-3 border-b border-border/50 px-3 py-2 sm:px-4">
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      <Layers className="h-3.5 w-3.5" aria-hidden="true" />
                      Why this deal matched
                    </div>
                    <Badge variant="secondary" className="font-medium">
                      {result.match_count}{" "}
                      {result.match_count === 1 ? "section" : "sections"}
                    </Badge>
                  </div>
                  <ul className="divide-y divide-border/50">
                    {matchedSections.map((section) => {
                      const sectionTitle =
                        section.section_title?.trim() ||
                        section.article_title?.trim() ||
                        "Matched section";
                      const subtitle =
                        section.section_title && section.article_title
                          ? section.article_title
                          : null;
                      return (
                        <li key={section.section_uuid}>
                          <Link
                            to={getAgreementHref(result, section.section_uuid)}
                            className="flex items-start gap-3 px-3 py-3 transition-colors hover:bg-background/70 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring sm:px-4"
                          >
                            <FileText
                              className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground"
                              aria-hidden="true"
                            />
                            <div className="min-w-0 flex-1">
                              <div className="flex flex-wrap items-center gap-2">
                                <span className="font-medium text-foreground">
                                  {sectionTitle}
                                </span>
                                {section.standard_id.slice(0, 2).map((id) => {
                                  const label =
                                    clauseTypeLabelById?.[id] ?? id;
                                  return (
                                    <Badge
                                      key={id}
                                      variant="outline"
                                      className="bg-background/60 px-1.5 py-0 text-[10px] font-medium uppercase tracking-wide text-muted-foreground"
                                      title={label}
                                    >
                                      <span className="max-w-[14rem] truncate">
                                        {label}
                                      </span>
                                    </Badge>
                                  );
                                })}
                              </div>
                              {subtitle ? (
                                <div className="mt-0.5 text-xs text-muted-foreground">
                                  {subtitle}
                                </div>
                              ) : null}
                              {section.snippet ? (
                                <p className="mt-1.5 line-clamp-2 text-sm leading-relaxed text-muted-foreground">
                                  {section.snippet}
                                </p>
                              ) : null}
                            </div>
                            <ArrowUpRight
                              className="mt-1 h-4 w-4 shrink-0 text-muted-foreground/60 transition-colors group-hover:text-foreground"
                              aria-hidden="true"
                            />
                          </Link>
                        </li>
                      );
                    })}
                  </ul>
                  {hiddenMatchCount > 0 ? (
                    <div className="border-t border-border/50 px-3 py-2 sm:px-4">
                      <button
                        type="button"
                        onClick={() => toggleExpandedMatches(result.agreement_uuid)}
                        className="inline-flex items-center gap-1 text-xs font-medium text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                        aria-expanded={isMatchesExpanded}
                      >
                        {isMatchesExpanded ? (
                          <>
                            Show fewer
                            <ChevronDown className="h-3.5 w-3.5 rotate-180" aria-hidden="true" />
                          </>
                        ) : (
                          <>
                            Show {hiddenMatchCount} more
                            {hiddenMatchCount === 1 ? " section" : " sections"}
                            <ChevronDown className="h-3.5 w-3.5" aria-hidden="true" />
                          </>
                        )}
                      </button>
                    </div>
                  ) : null}
                </div>
              ) : null}
            </div>
          </li>
        );
      })}
      </ol>
    </div>
  );
}
