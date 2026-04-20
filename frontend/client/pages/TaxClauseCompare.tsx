import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, ExternalLink, Filter, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import type { TaxClauseSearchResult } from "@shared/tax-clauses";
import {
  clearCompareClauses,
  loadCompareClauses,
  stashCompareClauses,
  TAX_COMPARE_MAX,
  TAX_COMPARE_MIN,
} from "@/lib/tax-compare-handoff";
import { extractNgrams, highlightPhrase } from "@/lib/ngram-analysis";

export default function TaxClauseCompare() {
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

  const buildAgreementHref = (r: TaxClauseSearchResult) => {
    const params = new URLSearchParams();
    params.set("from", "/compare/tax");
    if (r.section_uuid) params.set("focusSectionUuid", r.section_uuid);
    return `/agreements/${r.agreement_uuid}?${params.toString()}`;
  };

  if (clauses.length === 0) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-12 sm:px-8">
        <div className="rounded-2xl border border-border/60 bg-card p-8 text-center shadow-sm">
          <h1 className="text-xl font-semibold text-foreground">
            Nothing to compare yet
          </h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Select {TAX_COMPARE_MIN}–{TAX_COMPARE_MAX} tax clauses from the search
            page, then click Compare to surface common drafting language.
          </p>
          <div className="mt-5">
            <Button asChild variant="default" size="sm">
              <Link to="/search?mode=tax">
                <ArrowLeft className="mr-1.5 h-4 w-4" aria-hidden="true" />
                Back to tax search
              </Link>
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-full w-full">
      {/* Left rail: selected clauses + sub-filter */}
      <aside className="hidden w-80 shrink-0 border-r border-border bg-card lg:block">
        <div className="sticky top-0 flex h-full flex-col">
          <div className="border-b border-border px-4 py-3">
            <div className="flex items-center justify-between gap-2">
              <h2 className="text-sm font-semibold text-foreground">
                Selected clauses ({clauses.length})
              </h2>
              <Button asChild size="sm" variant="ghost" className="h-7 px-2 text-xs">
                <Link to="/search?mode=tax">
                  <ArrowLeft className="mr-1 h-3.5 w-3.5" aria-hidden="true" />
                  Search
                </Link>
              </Button>
            </div>
            <div className="mt-3 flex items-center gap-2">
              <Filter className="h-3.5 w-3.5 text-muted-foreground" aria-hidden="true" />
              <Select value={typeFilter} onValueChange={setTypeFilter}>
                <SelectTrigger className="h-8 text-xs">
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
          <ul className="flex-1 overflow-y-auto p-2" role="list">
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
                    className="opacity-0 transition-opacity group-hover:opacity-100"
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
        </div>
      </aside>

      {/* Main compare pane */}
      <div className="flex flex-1 flex-col min-w-0">
        <div className="border-b border-border px-4 py-3 sm:px-8">
          <div className="flex items-center justify-between gap-3">
            <h1 className="text-xl font-semibold tracking-tight">Tax clause comparison</h1>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <span>Phrase length</span>
              <Select
                value={String(minN)}
                onValueChange={(v) => setMinN(Number(v))}
              >
                <SelectTrigger className="h-8 w-16">
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
                onValueChange={(v) => setMaxN(Number(v))}
              >
                <SelectTrigger className="h-8 w-16">
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
          </div>
        </div>

        <div className="flex-1 overflow-auto">
          <div className="grid gap-6 px-4 py-5 sm:px-8">
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
                <div className="rounded-lg border border-dashed border-border/80 bg-muted/20 p-6 text-sm text-muted-foreground">
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
                      className="rounded-xl border border-border/60 bg-card p-4 shadow-sm"
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
                          <Badge
                            variant="outline"
                            className="border-amber-300/70 bg-amber-50 text-amber-900 dark:bg-amber-900/20 dark:text-amber-200"
                          >
                            Reps & warranties
                          </Badge>
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
                          <Link to={buildAgreementHref(c)}>
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
      </div>
    </div>
  );
}
