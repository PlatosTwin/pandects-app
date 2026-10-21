import { Fragment, useState, type MouseEvent, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { ArrowDown, ArrowUp, ArrowUpDown, ChevronRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { ExternalLink } from "@/components/ui/external-link";
import { FlagAsInaccurateButton } from "@/components/FlagAsInaccurateButton";
import { StarButton } from "@/components/StarButton";
import type { ResultsDensity, ResultsSortField } from "@/components/ResultsToolbar";
import { useNaics } from "@/hooks/use-naics";
import { formatCompactCurrencyValue, formatDateValue, formatEnumValue } from "@/lib/format-utils";
import { formatNaicsIndustry } from "@/lib/naics";
import { cn } from "@/lib/utils";
import type { TransactionSearchResult } from "@shared/transactions";
import { Highlight, literalTermsPattern } from "@/pages/search/Highlight";
import { cleanSnippet } from "@/pages/search/search-ui";

export interface TransactionResultsListProps {
  results: TransactionSearchResult[];
  getAgreementHref: (result: TransactionSearchResult, focusSectionUuid?: string | null) => string;
  /** A clause type, section UUID or keyword is active, so matched sections mean something. */
  showMatches: boolean;
  clauseTypeLabelById: Record<string, string>;
  selectedResults: Set<string>;
  onToggleResultSelection: (agreementUuid: string) => void;
  onToggleSelectAll: () => void;
  sortBy: ResultsSortField;
  sortDirection: "asc" | "desc";
  /** Header click: a new column sorts (keeping direction); the active column flips. */
  onSort: (field: ResultsSortField) => void;
  density?: ResultsDensity;
  /** Fallback highlight when a result carries no `matched_terms`. */
  highlightPattern?: RegExp | null;
}

const DASH = "—";

function SortHeader({
  field,
  label,
  sortBy,
  sortDirection,
  onSort,
  className,
}: {
  field: ResultsSortField;
  label: string;
  sortBy: ResultsSortField;
  sortDirection: "asc" | "desc";
  onSort: (field: ResultsSortField) => void;
  className?: string;
}) {
  const active = sortBy === field;
  const Icon = !active ? ArrowUpDown : sortDirection === "asc" ? ArrowUp : ArrowDown;
  return (
    <th
      scope="col"
      aria-sort={active ? (sortDirection === "asc" ? "ascending" : "descending") : "none"}
      className={cn("px-2 py-2 text-left font-medium", className)}
    >
      <button
        type="button"
        onClick={() => onSort(field)}
        className={cn(
          "-ml-1 inline-flex items-center gap-1 rounded px-1 py-0.5 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
          active && "text-foreground",
        )}
      >
        {label}
        <Icon className={cn("h-3.5 w-3.5", !active && "opacity-50")} aria-hidden="true" />
      </button>
    </th>
  );
}

function Fact({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="truncate text-sm text-foreground">{children}</dd>
    </div>
  );
}

const yesNo = (value: boolean | number | null | undefined) =>
  value === null || value === undefined ? DASH : value ? "Yes" : "No";

function DealDetails({
  result,
  props,
}: {
  result: TransactionSearchResult;
  props: TransactionResultsListProps;
}) {
  const { labelByCode } = useNaics();
  const pattern = (terms?: string[]) => literalTermsPattern(terms) ?? props.highlightPattern ?? null;
  return (
    <div className="grid gap-5 px-3 py-4 md:grid-cols-[minmax(0,2fr)_minmax(0,3fr)] md:px-4">
      <dl className="grid grid-cols-2 gap-x-4 gap-y-3 self-start lg:grid-cols-3">
        <Fact label="Filed">{formatDateValue(result.filing_date)}</Fact>
        <Fact label="Announced">{formatDateValue(result.announce_date)}</Fact>
        <Fact label="Closed">{formatDateValue(result.close_date)}</Fact>
        <Fact label="Deal type">{formatEnumValue(result.deal_type)}</Fact>
        <Fact label="Consideration">{formatEnumValue(result.transaction_consideration)}</Fact>
        <Fact label="Status">{formatEnumValue(result.deal_status)}</Fact>
        <Fact label="Attitude">{formatEnumValue(result.attitude)}</Fact>
        <Fact label="Purpose">{formatEnumValue(result.purpose)}</Fact>
        <Fact label="Target type">{formatEnumValue(result.target_type)}</Fact>
        <Fact label="Acquirer type">{formatEnumValue(result.acquirer_type)}</Fact>
        <Fact label="Target PE-backed">{yesNo(result.target_pe)}</Fact>
        <Fact label="Acquirer PE-backed">{yesNo(result.acquirer_pe)}</Fact>
        <Fact label="Cash">{formatCompactCurrencyValue(result.transaction_price_cash)}</Fact>
        <Fact label="Stock">{formatCompactCurrencyValue(result.transaction_price_stock)}</Fact>
        <div className="col-span-full">
          <Fact label="Target industry">{formatNaicsIndustry(labelByCode, result.target_industry) ?? DASH}</Fact>
        </div>
        <div className="col-span-full">
          <Fact label="Acquirer industry">
            {formatNaicsIndustry(labelByCode, result.acquirer_industry) ?? DASH}
          </Fact>
        </div>
        {result.url ? (
          <div className="col-span-full">
            <ExternalLink href={result.url} className="text-sm font-medium text-primary hover:underline">
              SEC filing
            </ExternalLink>
          </div>
        ) : null}
      </dl>
      <div className="min-w-0">
        <h4 className="text-sm font-semibold text-foreground">
          {props.showMatches ? "Why this matched" : "Agreement"}
        </h4>
        {props.showMatches && result.matched_sections.length > 0 ? (
          <>
            <ul className="mt-2 space-y-2">
              {result.matched_sections.map((section) => {
                const title = section.section_title?.trim() || section.article_title?.trim() || "Matched section";
                const types = section.standard_id
                  .map((id) => props.clauseTypeLabelById[id])
                  .filter(Boolean)
                  .slice(0, 2);
                return (
                  <li key={section.section_uuid} className="rounded-md border border-border bg-background p-3">
                    <Link
                      to={props.getAgreementHref(result, section.section_uuid)}
                      className="text-sm font-medium text-foreground hover:text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    >
                      {title}
                    </Link>
                    {types.length > 0 ? (
                      <div className="mt-0.5 text-xs text-muted-foreground">{types.join(" · ")}</div>
                    ) : null}
                    {section.snippet ? (
                      <p className="mt-1.5 text-sm leading-relaxed text-foreground/90">
                        <Highlight text={cleanSnippet(section.snippet)} pattern={pattern(section.matched_terms)} />
                      </p>
                    ) : null}
                  </li>
                );
              })}
            </ul>
            {result.match_count > result.matched_sections.length ? (
              <p className="mt-2 text-xs text-muted-foreground">
                Showing {result.matched_sections.length} of {result.match_count} matching sections.
              </p>
            ) : null}
          </>
        ) : (
          <p className="mt-2 text-sm text-muted-foreground">
            {props.showMatches
              ? "No matched section preview for this deal."
              : "Pick a clause type or keyword to see which sections match."}
          </p>
        )}
        <Button asChild size="sm" className="mt-3">
          <Link to={props.getAgreementHref(result, props.showMatches ? result.matched_sections[0]?.section_uuid : null)}>
            Open agreement
          </Link>
        </Button>
      </div>
    </div>
  );
}

const INTERACTIVE = "a,button,input,label,[role=checkbox],[role=menuitem],[data-row-ignore]";

/**
 * Deals as a dense, sortable table. A row (or its chevron) expands in place
 * to show the deal facts and, with a clause type or keyword, the sections
 * that matched. Also serves as the "browse all deals" view with no filters.
 */
export function TransactionResultsList(props: TransactionResultsListProps) {
  const {
    results,
    selectedResults,
    onToggleResultSelection,
    onToggleSelectAll,
    sortBy,
    sortDirection,
    onSort,
    density = "comfy",
    showMatches,
  } = props;
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
  const showCounsel = results.some((result) => result.target_counsel !== undefined);
  const allSelected = results.length > 0 && results.every((r) => selectedResults.has(r.agreement_uuid));
  const someSelected = !allSelected && results.some((r) => selectedResults.has(r.agreement_uuid));
  const cellY = density === "compact" ? "py-1.5" : "py-2.5";
  const columnCount = 8 + (showCounsel ? 1 : 0) + (showMatches ? 1 : 0) + 1;

  const toggle = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const onRowClick = (event: MouseEvent<HTMLTableRowElement>, id: string) => {
    if ((event.target as HTMLElement).closest(INTERACTIVE)) return;
    if (window.getSelection()?.toString()) return;
    toggle(id);
  };

  return (
    <div className="overflow-hidden rounded-lg border border-border bg-card">
      <table className="w-full table-fixed border-collapse text-sm">
        <caption className="sr-only">
          Deals. Column headers for year, target and acquirer sort the table; the chevron in each row shows details.
        </caption>
        <thead className="border-b border-border bg-muted/40 text-xs text-muted-foreground">
          <tr>
            <th scope="col" className="w-9 py-2 pl-3 pr-1">
              <Checkbox
                checked={allSelected ? true : someSelected ? "indeterminate" : false}
                onCheckedChange={onToggleSelectAll}
                aria-label="Select all deals on this page"
              />
            </th>
            <th scope="col" className="w-8 px-0">
              <span className="sr-only">Details</span>
            </th>
            <SortHeader field="year" label="Year" {...{ sortBy, sortDirection, onSort }} className="w-[4.25rem]" />
            <SortHeader field="target" label="Target" {...{ sortBy, sortDirection, onSort }} />
            <SortHeader
              field="acquirer"
              label="Acquirer"
              {...{ sortBy, sortDirection, onSort }}
              className="hidden md:table-cell"
            />
            <th scope="col" className="w-20 px-2 py-2 text-right font-medium sm:w-24">
              Value
            </th>
            <th scope="col" className="hidden w-28 px-2 py-2 text-left font-medium xl:table-cell">
              Consideration
            </th>
            <th scope="col" className="hidden w-28 px-2 py-2 text-left font-medium lg:table-cell">
              Status
            </th>
            {showCounsel ? (
              <th scope="col" className="hidden px-2 py-2 text-left font-medium xl:table-cell">
                Target counsel
              </th>
            ) : null}
            {showMatches ? (
              <th scope="col" className="hidden w-20 pl-2 pr-4 py-2 text-right font-medium sm:table-cell">
                Matches
              </th>
            ) : null}
            <th scope="col" className="hidden w-20 px-2 py-2 text-right font-medium md:table-cell xl:w-[8.5rem]">
              <span className="sr-only">Actions</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {results.map((result) => {
            const id = result.agreement_uuid;
            const isOpen = expanded.has(id);
            const isSelected = selectedResults.has(id);
            const target = result.target?.trim() || "Unknown target";
            const acquirer = result.acquirer?.trim() || "Unknown acquirer";
            const detailsId = `deal-details-${id}`;
            return (
              <Fragment key={id}>
                {/* Mouse convenience only: the chevron button is the keyboard path. */}
                <tr
                  onClick={(event) => onRowClick(event, id)}
                  className={cn(
                    "cursor-pointer border-b border-border align-top last:border-b-0 hover:bg-accent/40",
                    isSelected && "bg-primary/5",
                    isOpen && "bg-muted/30",
                  )}
                >
                  <td className={cn("pl-3 pr-1", cellY)}>
                    <Checkbox
                      checked={isSelected}
                      onCheckedChange={() => onToggleResultSelection(id)}
                      aria-label={`Select ${target} acquired by ${acquirer}`}
                      className="mt-0.5"
                    />
                  </td>
                  <td className={cn("px-0", cellY)}>
                    <button
                      type="button"
                      onClick={() => toggle(id)}
                      aria-expanded={isOpen}
                      aria-controls={detailsId}
                      aria-label={`${isOpen ? "Hide" : "Show"} details for ${target}`}
                      className="-mt-0.5 inline-flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    >
                      <ChevronRight className={cn("h-4 w-4 transition-transform", isOpen && "rotate-90")} aria-hidden="true" />
                    </button>
                  </td>
                  <td className={cn("px-2 tabular-nums text-muted-foreground", cellY)}>{result.year ?? DASH}</td>
                  <td className={cn("min-w-0 px-2", cellY)}>
                    <div className="truncate font-medium text-foreground" title={target}>
                      {target}
                    </div>
                    <div className="truncate text-xs text-muted-foreground md:hidden" title={acquirer}>
                      <span aria-hidden="true">← </span>
                      <span className="sr-only">acquired by </span>
                      {acquirer}
                    </div>
                  </td>
                  <td className={cn("hidden min-w-0 px-2 md:table-cell", cellY)}>
                    <div className="truncate text-foreground" title={acquirer}>
                      {acquirer}
                    </div>
                  </td>
                  <td className={cn("px-2 text-right tabular-nums text-foreground", cellY)}>
                    {result.transaction_price_total != null
                      ? formatCompactCurrencyValue(result.transaction_price_total)
                      : DASH}
                  </td>
                  <td className={cn("hidden px-2 text-muted-foreground xl:table-cell", cellY)}>
                    {formatEnumValue(result.transaction_consideration)}
                  </td>
                  <td className={cn("hidden px-2 lg:table-cell", cellY)}>
                    {result.deal_status ? <Badge variant="muted">{formatEnumValue(result.deal_status)}</Badge> : DASH}
                  </td>
                  {showCounsel ? (
                    <td className={cn("hidden min-w-0 px-2 text-muted-foreground xl:table-cell", cellY)}>
                      <div className="truncate" title={result.target_counsel ?? undefined}>
                        {result.target_counsel || DASH}
                      </div>
                    </td>
                  ) : null}
                  {showMatches ? (
                    <td className={cn("hidden pl-2 pr-4 text-right tabular-nums sm:table-cell", cellY)}>
                      {result.match_count > 0 ? result.match_count : DASH}
                    </td>
                  ) : null}
                  <td className={cn("hidden px-2 md:table-cell", cellY)}>
                    <div className="-my-1 flex items-center justify-end gap-0.5" data-row-ignore>
                      <span className="hidden items-center gap-0.5 xl:flex">
                        <StarButton
                          itemType="agreement"
                          itemUuid={id}
                          context={{ target: result.target, acquirer: result.acquirer, filing_date: result.filing_date }}
                          className="shrink-0"
                        />
                        <FlagAsInaccurateButton source="search_result" agreement_uuid={id} className="shrink-0" />
                      </span>
                      <Button asChild size="sm" variant="outline" className="ml-1 h-8 px-3">
                        <Link
                          to={props.getAgreementHref(result, showMatches ? result.matched_sections[0]?.section_uuid : null)}
                          aria-label={`Open agreement: ${target} acquired by ${acquirer}`}
                        >
                          Open
                        </Link>
                      </Button>
                    </div>
                  </td>
                </tr>
                {isOpen ? (
                  <tr id={detailsId} className="border-b border-border bg-muted/20 last:border-b-0">
                    <td colSpan={columnCount} className="p-0">
                      <div className="flex items-center gap-1 border-b border-border/60 px-3 py-2 xl:hidden">
                        <StarButton
                          itemType="agreement"
                          itemUuid={id}
                          context={{ target: result.target, acquirer: result.acquirer, filing_date: result.filing_date }}
                          className="shrink-0"
                        />
                        <FlagAsInaccurateButton source="search_result" agreement_uuid={id} className="shrink-0" />
                      </div>
                      <DealDetails result={result} props={props} />
                    </td>
                  </tr>
                ) : null}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
