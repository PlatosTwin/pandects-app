import { useEffect, useMemo, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { ArrowLeft, ExternalLink, Filter, X } from "lucide-react";
import { PageShell } from "@/components/PageShell";
import { EmptyState } from "@/components/StateMessage";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { buildAgreementHref } from "@/lib/agreement-links";
import { cn } from "@/lib/utils";
import type { TaxClauseSearchResult } from "@shared/tax-clauses";
import {
  clearCompareClauses,
  compareOriginSearch,
  loadCompareClauses,
  stashCompareClauses,
  TAX_COMPARE_MAX,
  TAX_COMPARE_MIN,
} from "@/lib/tax-compare-handoff";
import { extractNgrams, highlightPhrase } from "@/lib/ngram-analysis";

export default function TaxClauseCompare() {
  const location = useLocation();
  // The originating search travels as `from`, so Back restores its filters.
  const backHref = compareOriginSearch(new URLSearchParams(location.search).get("from"));
  const [clauses, setClauses] = useState<TaxClauseSearchResult[]>([]);
  const [selectedPhrase, setSelectedPhrase] = useState<string | null>(null);
  const [typeFilter, setTypeFilter] = useState<string>("all");
  const [minN, setMinN] = useState<number>(3);
  const [maxN, setMaxN] = useState<number>(6);

  useEffect(() => {
    setClauses(loadCompareClauses());
  }, []);

  const allStandardIds = useMemo(() => {
    const set = new Set<string>();
    clauses.forEach((c) => c.tax_standard_ids.forEach((s) => set.add(s)));
    return Array.from(set).sort();
  }, [clauses]);

  const filteredClauses = useMemo(() => {
    if (typeFilter === "all") return clauses;
    return clauses.filter((c) => c.tax_standard_ids.includes(typeFilter));
  }, [clauses, typeFilter]);

  const phrases = useMemo(() => {
    if (filteredClauses.length < TAX_COMPARE_MIN) return [];
    return extractNgrams(
      filteredClauses.map((c) => c.clause_text ?? ""),
      { minN, maxN, minClauseCount: 2, topK: 60 },
    );
  }, [filteredClauses, minN, maxN]);

  const removeClause = (id: string) => {
    const next = clauses.filter((c) => c.id !== id);
    setClauses(next);
    stashCompareClauses(next);
  };

  const readerHref = (r: TaxClauseSearchResult) =>
    buildAgreementHref({
      agreementUuid: r.agreement_uuid,
      sectionUuid: r.section_uuid,
      from: location,
    });

  const backLink = (
    <Button asChild variant="outline" size="sm">
      <Link to={backHref}>
        <ArrowLeft className="mr-1.5 h-4 w-4" aria-hidden="true" />
        Back to tax search
      </Link>
    </Button>
  );

  if (clauses.length === 0) {
    return (
      <PageShell layout="data" title="Tax clause comparison">
        <EmptyState
          title="Nothing to compare yet"
          body={`Select ${TAX_COMPARE_MIN}–${TAX_COMPARE_MAX} tax clauses from the search page, then click Compare to surface common drafting language.`}
          action={backLink}
        />
      </PageShell>
    );
  }

  const phraseLengthControls = (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      <span>Phrase length</span>
      <Select
        value={String(minN)}
        onValueChange={(v) => {
          const next = Number(v);
          setMinN(next);
          setMaxN((prev) => Math.max(prev, next));
        }}
      >
        <SelectTrigger className="h-8 w-16" aria-label="Shortest phrase length">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {[2, 3, 4, 5].map((n) => (
            <SelectItem key={n} value={String(n)}>
              {n}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <span>–</span>
      <Select
        value={String(maxN)}
        onValueChange={(v) => {
          const next = Number(v);
          setMaxN(next);
          setMinN((prev) => Math.min(prev, next));
        }}
      >
        <SelectTrigger className="h-8 w-16" aria-label="Longest phrase length">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {[4, 5, 6, 7, 8, 10].map((n) => (
            <SelectItem key={n} value={String(n)}>
              {n}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );

  return (
    <PageShell
      layout="data"
      title="Tax clause comparison"
      actions={
        <div className="flex flex-wrap items-center gap-3">
          {phraseLengthControls}
          {backLink}
        </div>
      }
    >
      <div className="grid gap-6 lg:grid-cols-[18rem_minmax(0,1fr)]">
        {/* Selected clauses + sub-filter */}
        <aside className="self-start rounded-xl border border-border bg-card lg:sticky lg:top-24">
          <div className="border-b border-border px-4 py-3">
            <h2 className="text-sm font-semibold text-foreground">
              Selected clauses ({clauses.length})
            </h2>
            <div className="mt-3 flex items-center gap-2">
              <Filter className="h-3.5 w-3.5 text-muted-foreground" aria-hidden="true" />
              <Select value={typeFilter} onValueChange={setTypeFilter}>
                <SelectTrigger className="h-8 text-xs" aria-label="Filter by tax type">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All tax types</SelectItem>
                  {allStandardIds.map((sid) => (
                    <SelectItem key={sid} value={sid}>
                      {sid}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <ul className="max-h-80 overflow-y-auto p-2 lg:max-h-[calc(100vh-16rem)]" role="list">
            {clauses.map((c) => (
              <li
                key={c.id}
                className="group rounded-md p-2 text-xs hover:bg-muted/50"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="font-medium text-foreground">
                      {c.year ?? "—"} · {c.target?.trim() || "Unknown"}
                    </div>
                    <div className="text-muted-foreground truncate">
                      {c.acquirer?.trim() || "Unknown acquirer"}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => removeClause(c.id)}
                    className="rounded-sm opacity-0 transition-opacity focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 group-hover:opacity-100"
                    aria-label="Remove from compare"
                  >
                    <X className="h-3.5 w-3.5 text-muted-foreground" />
                  </button>
                </div>
              </li>
            ))}
          </ul>
          <div className="border-t border-border p-3">
            <Button
              variant="ghost"
              size="sm"
              className="w-full text-xs text-muted-foreground"
              onClick={() => {
                clearCompareClauses();
                setClauses([]);
              }}
            >
              Clear all
            </Button>
          </div>
        </aside>

        {/* Compare pane */}
        <div className="grid min-w-0 gap-6">
          {/* Top: common phrases */}
          <section>
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-sm font-semibold text-foreground">
                Common drafting phrases
              </h2>
              <span className="text-xs text-muted-foreground">
                {phrases.length} phrases · {filteredClauses.length} clauses
              </span>
            </div>
            {phrases.length === 0 ? (
              <div className="rounded-lg border border-dashed border-border bg-muted/20 p-6 text-sm text-muted-foreground">
                No shared phrases found at current settings. Try widening the
                phrase length range or clearing the tax type filter.
              </div>
            ) : (
              <div className="flex flex-wrap gap-2">
                {phrases.map((p) => {
                  const isSelected = selectedPhrase === p.phrase;
                  return (
                    <button
                      key={p.phrase}
                      type="button"
                      onClick={() =>
                        setSelectedPhrase(isSelected ? null : p.phrase)
                      }
                      className={cn(
                        "inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs transition-colors",
                        isSelected
                          ? "border-primary bg-primary/10 text-primary"
                          : "border-border bg-background text-foreground hover:border-primary/40 hover:bg-muted/40",
                      )}
                    >
                      <span className="font-medium">{p.phrase}</span>
                      <Badge variant="outline" className="h-4 text-[10px]">
                        {p.clauseCount}/{filteredClauses.length}
                      </Badge>
                    </button>
                  );
                })}
              </div>
            )}
          </section>

          {/* Bottom: stacked clause reader */}
          <section>
            <h2 className="mb-3 text-sm font-semibold text-foreground">
              Clauses{selectedPhrase && <span className="ml-2 text-muted-foreground font-normal">— highlighting "{selectedPhrase}"</span>}
            </h2>
            <ul className="space-y-3" role="list">
              {filteredClauses.map((c) => {
                const parts = highlightPhrase(c.clause_text ?? "", selectedPhrase ?? "");
                return (
                  <li
                    key={c.id}
                    className="rounded-xl border border-border bg-card p-4 shadow-sm"
                  >
                    <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                      <span className="font-medium text-foreground">
                        {c.year ?? "—"}
                      </span>
                      <span aria-hidden="true">·</span>
                      <span>
                        <span className="text-muted-foreground">Target:</span>{" "}
                        <span className="font-medium text-foreground">
                          {c.target?.trim() || "Unknown target"}
                        </span>
                      </span>
                      <span aria-hidden="true">·</span>
                      <span>
                        <span className="text-muted-foreground">Acquirer:</span>{" "}
                        <span className="font-medium text-foreground">
                          {c.acquirer?.trim() || "Unknown acquirer"}
                        </span>
                      </span>
                      {c.context_type === "rep_warranty" && (
                        <Badge variant="warning">Reps & warranties</Badge>
                      )}
                    </div>
                    {c.tax_standard_ids.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {c.tax_standard_ids.map((sid) => (
                          <Badge key={sid} variant="secondary" className="text-xs">
                            {sid}
                          </Badge>
                        ))}
                      </div>
                    )}
                    <p className="mt-3 whitespace-pre-wrap text-sm text-foreground">
                      {parts.map((part, idx) =>
                        part.match ? (
                          <mark
                            key={idx}
                            className="rounded bg-primary/20 px-0.5 text-foreground"
                          >
                            {part.text}
                          </mark>
                        ) : (
                          <span key={idx}>{part.text}</span>
                        ),
                      )}
                    </p>
                    <div className="mt-3">
                      <Button asChild size="sm" variant="outline" className="gap-1.5">
                        <Link to={readerHref(c)}>
                          <ExternalLink className="h-3.5 w-3.5" aria-hidden="true" />
                          Open agreement
                        </Link>
                      </Button>
                    </div>
                  </li>
                );
              })}
            </ul>
          </section>
        </div>
      </div>
    </PageShell>
  );
}
