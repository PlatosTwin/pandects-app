import { Link } from "react-router-dom";
import { History, Search as SearchIcon } from "lucide-react";
import type { ClauseTypeTree } from "@/lib/clause-types";
import { buildSearchStateParams, createEmptySearchFilters } from "@/lib/url-params";
import type { SearchMode, SearchQueryFilters } from "@shared/search";
import { findClauseTypeIds, MODE_NOUNS, type RecentSearch } from "./search-ui";

export interface StarterQuery {
  key: string;
  label: string;
  hint: string;
  href: string;
}

function hrefFor(mode: SearchMode, patch: Partial<SearchQueryFilters>): string {
  const params = buildSearchStateParams({
    filters: { ...createEmptySearchFilters(), ...patch },
    mode,
    sortBy: "year",
    sortDirection: "desc",
  });
  return `/search?${params.toString()}`;
}

interface StarterSpec {
  key: string;
  label: string;
  hint: string;
  mode: SearchMode;
  /** Clause type found by label in the loaded taxonomy (never a hardcoded id). */
  clauseLabel?: RegExp;
  patch?: Partial<SearchQueryFilters>;
}

const SECTION_STARTERS: StarterSpec[] = [
  {
    key: "mae-definition",
    label: "MAE definition",
    hint: "Keyword “Material Adverse Effect means”",
    mode: "sections",
    patch: { text_query: "Material Adverse Effect means" },
  },
  {
    key: "mae-condition",
    label: "MAE closing condition",
    hint: "Clause type",
    mode: "sections",
    clauseLabel: /material adverse effect/i,
  },
  {
    key: "termination-fee",
    label: "Termination fee",
    hint: "Clause type",
    mode: "sections",
    clauseLabel: /termination fee/i,
  },
  {
    key: "reverse-termination-fee",
    label: "Reverse termination fee",
    hint: "Keyword “reverse termination fee”",
    mode: "sections",
    patch: { text_query: "reverse termination fee" },
  },
  {
    key: "specific-performance",
    label: "Specific performance",
    hint: "Clause type",
    mode: "sections",
    clauseLabel: /specific performance/i,
  },
  {
    key: "deals-2025",
    label: "Deals from 2025",
    hint: "All deals with 2025 agreements",
    mode: "transactions",
    patch: { year: ["2025"] },
  },
];

const TAX_STARTERS: StarterSpec[] = [
  { key: "tax-338", label: "Section 338 elections", hint: "Tax clause type", mode: "tax", clauseLabel: /section 338/i },
  {
    key: "tax-ppa",
    label: "Purchase price allocation",
    hint: "Tax clause type",
    mode: "tax",
    clauseLabel: /^purchase price allocation$/i,
  },
  { key: "tax-indemnity", label: "Tax indemnity", hint: "Tax clause type", mode: "tax", clauseLabel: /tax indemnity obligation/i },
  { key: "tax-straddle", label: "Straddle periods", hint: "Tax clause type", mode: "tax", clauseLabel: /straddle period/i },
  { key: "tax-transfer", label: "Transfer taxes", hint: "Tax clause type", mode: "tax", clauseLabel: /transfer tax/i },
];

/** Starter queries for the mode; ones whose clause type isn't in the taxonomy are left out. */
export function buildStarterQueries(
  mode: SearchMode,
  sectionTree: ClauseTypeTree,
  taxTree: ClauseTypeTree,
): StarterQuery[] {
  const specs = mode === "tax" ? TAX_STARTERS : SECTION_STARTERS;
  const tree = mode === "tax" ? taxTree : sectionTree;
  return specs.flatMap((spec) => {
    let patch = spec.patch ?? {};
    if (spec.clauseLabel) {
      const [id] = findClauseTypeIds(tree, spec.clauseLabel);
      if (!id) return [];
      patch = { ...patch, clauseType: [id] };
    }
    return [{ key: spec.key, label: spec.label, hint: spec.hint, href: hrefFor(spec.mode, patch) }];
  });
}

export function SearchEmptyState({
  mode,
  starters,
  recent,
  isLoadingStarters,
}: {
  mode: SearchMode;
  starters: StarterQuery[];
  recent: RecentSearch[];
  isLoadingStarters: boolean;
}) {
  return (
    <div className="space-y-8 py-2">
      <div>
        <h2 className="text-lg font-semibold text-foreground">
          {mode === "tax" ? "Find tax clause precedents" : "Find clauses across agreements"}
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Pick a {mode === "tax" ? "tax clause type" : "clause type"}
          {mode === "tax" ? "" : ", type a keyword,"} or set a filter. Results update as you go.
        </p>
      </div>

      <section aria-labelledby="starter-queries-heading">
        <h3 id="starter-queries-heading" className="text-sm font-semibold text-foreground">
          Try a search
        </h3>
        {isLoadingStarters && starters.length === 0 ? (
          <p className="mt-2 text-sm text-muted-foreground">Loading clause types…</p>
        ) : (
          <ul className="mt-2 grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
            {starters.map((starter) => (
              <li key={starter.key}>
                <Link
                  to={starter.href}
                  className="flex h-full items-start gap-3 rounded-lg border border-border bg-card px-3 py-2.5 transition-colors hover:border-primary/40 hover:bg-accent/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  <SearchIcon className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" aria-hidden="true" />
                  <span className="min-w-0">
                    <span className="block text-sm font-medium text-foreground">{starter.label}</span>
                    <span className="block text-xs text-muted-foreground">{starter.hint}</span>
                  </span>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>

      {recent.length > 0 ? (
        <section aria-labelledby="recent-searches-heading">
          <h3 id="recent-searches-heading" className="text-sm font-semibold text-foreground">
            Recent searches
          </h3>
          <ul className="mt-2 divide-y divide-border rounded-lg border border-border bg-card">
            {recent.map((entry) => (
              <li key={entry.search}>
                <Link
                  to={`/search?${entry.search}`}
                  className="flex items-center gap-3 px-3 py-2.5 text-sm text-foreground hover:bg-accent/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring"
                >
                  <History className="h-4 w-4 shrink-0 text-muted-foreground" aria-hidden="true" />
                  <span className="min-w-0 truncate">{entry.label}</span>
                  <span className="sr-only">({MODE_NOUNS[entry.mode].title})</span>
                </Link>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  );
}
