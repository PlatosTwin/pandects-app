import { Link } from "react-router-dom";
import { Badge } from "@/components/ui/badge";
import { FlagAsInaccurateButton } from "@/components/FlagAsInaccurateButton";
import { StarButton } from "@/components/StarButton";
import type { ResultsDensity } from "@/components/ResultsToolbar";
import { formatCompactCurrencyValue, formatEnumValue } from "@/lib/format-utils";
import { cn } from "@/lib/utils";
import type { TaxClauseSearchResult } from "@shared/tax-clauses";
import { ClampedText, MetaLine, Parties, VerifiedBadge } from "@/pages/search/ResultBits";
import { ResultRow, RowActions } from "@/pages/search/ResultRow";

export interface TaxClauseResultsListProps {
  results: TaxClauseSearchResult[];
  getAgreementHref: (result: TaxClauseSearchResult) => string;
  /** Fallback labels (the tax taxonomy tree) for ids the result doesn't label. */
  clauseTypeLabelById?: Record<string, string>;
  selectedResults: Set<string>;
  onToggleResultSelection: (id: string) => void;
  density?: ResultsDensity;
  isAnonymous: boolean;
  signInHref: string;
  /** Clause types searched for; a matching type is named first. */
  preferredTypeIds?: readonly string[];
}

/** Clause-type labels for a result, the searched-for types first. Never raw ids. */
function clauseLabels(
  result: TaxClauseSearchResult,
  labels: Record<string, string>,
  preferred: readonly string[],
): string[] {
  const ids = [...result.tax_standard_ids].sort(
    (a, b) => Number(!preferred.includes(a)) - Number(!preferred.includes(b)),
  );
  const own = result.tax_standard_id_labels ?? {};
  return [...new Set(ids.map((id) => own[id] ?? labels[id]).filter(Boolean))];
}

/** Tax clause results: clause type first, then the clause text, in the shared row shell. */
export function TaxClauseResultsList({
  results,
  getAgreementHref,
  clauseTypeLabelById = {},
  selectedResults,
  onToggleResultSelection,
  density = "comfy",
  isAnonymous,
  signInHref,
  preferredTypeIds = [],
}: TaxClauseResultsListProps) {
  return (
    <ul aria-label="Tax clause results" className={density === "compact" ? "space-y-2" : "space-y-3"}>
      {results.map((result) => {
        const href = getAgreementHref(result);
        const [primary, ...otherLabels] = clauseLabels(result, clauseTypeLabelById, preferredTypeIds);
        const title = primary ?? "Unclassified tax clause";
        const price = result.transaction_price_total != null ? Number(result.transaction_price_total) : null;
        const text = result.clause_text?.trim() ?? "";
        return (
          <li key={result.id}>
            <ResultRow
              density={density}
              selected={selectedResults.has(result.id)}
              onToggleSelected={() => onToggleResultSelection(result.id)}
              selectLabel={`Select ${title} clause from ${result.target || "unknown target"}`}
              meta={
                <MetaLine
                  items={[
                    result.year ? <span className="tabular-nums">{result.year}</span> : null,
                    <Parties target={result.target} acquirer={result.acquirer} />,
                    price != null && Number.isFinite(price) ? (
                      <span className="tabular-nums">{formatCompactCurrencyValue(price)}</span>
                    ) : null,
                    result.deal_status ? <Badge variant="muted">{formatEnumValue(result.deal_status)}</Badge> : null,
                    result.context_type === "rep_warranty" ? (
                      <Badge variant="metadata">Reps &amp; warranties</Badge>
                    ) : null,
                    result.verified ? <VerifiedBadge /> : null,
                  ]}
                />
              }
              title={
                <h3
                  className={cn("text-base font-semibold leading-snug", density === "compact" && "text-sm")}
                >
                  <Link
                    to={href}
                    className="text-foreground hover:text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    {title}
                  </Link>
                </h3>
              }
              subtitle={
                result.anchor_label || otherLabels.length > 0 ? (
                  <span title={otherLabels.length > 0 ? `Also: ${otherLabels.join("; ")}` : undefined}>
                    {[
                      result.anchor_label ? `Clause ${result.anchor_label}` : null,
                      otherLabels.length > 0
                        ? `+${otherLabels.length} more ${otherLabels.length === 1 ? "type" : "types"}`
                        : null,
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                  </span>
                ) : null
              }
              actions={
                <RowActions
                  star={
                    <StarButton
                      itemType="tax_clause"
                      itemUuid={result.clause_uuid}
                      context={{
                        agreement_uuid: result.agreement_uuid,
                        target: result.target,
                        acquirer: result.acquirer,
                      }}
                      className="shrink-0"
                    />
                  }
                  flag={
                    <FlagAsInaccurateButton
                      source="search_result"
                      agreement_uuid={result.agreement_uuid}
                      section_uuid={result.section_uuid}
                      className="shrink-0"
                    />
                  }
                  openHref={href}
                  openLabel={`Open the agreement for this ${title} clause (${result.target || "unknown target"})`}
                />
              }
            >
              {text ? (
                <ClampedText text={text} lines={density === "compact" ? 3 : 6} moreLabel="Show full clause" />
              ) : isAnonymous ? (
                <p className="text-sm text-muted-foreground">
                  <Link to={signInHref} className="font-medium text-primary hover:underline">
                    Sign in
                  </Link>{" "}
                  to read this clause.
                </p>
              ) : (
                <p className="text-sm text-muted-foreground">No stored text.</p>
              )}
            </ResultRow>
          </li>
        );
      })}
    </ul>
  );
}
