import { forwardRef, useId, useState, type FormEvent } from "react";
import { Search as SearchIcon, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ReferenceLink } from "@/components/ui/reference-link";
import { SegmentedControl } from "@/components/ui/segmented-control";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { NestedCheckboxFilter } from "@/components/NestedCheckboxFilter";
import type { ClauseTypeTree } from "@/lib/clause-types";
import { cn } from "@/lib/utils";
import type { SearchMode, TextMatchMode } from "@shared/search";
import { ClauseTypeInfo } from "./ClauseTypeInfo";
import { TEXT_MATCH_MODE_LABELS, validateKeywordQuery, MAX_KEYWORD_CHARS } from "./keyword-query";
import { MODE_NOUNS } from "./search-ui";

const MODE_ORDER: readonly SearchMode[] = ["sections", "transactions", "tax"];

export interface SearchQueryBarProps {
  mode: SearchMode;
  onModeChange: (mode: SearchMode) => void;
  clauseTypes: ClauseTypeTree;
  clauseTypeLabelById: Record<string, string>;
  selectedClauseTypes: string[];
  onToggleClauseType: (id: string) => void;
  onClearClauseTypes: () => void;
  isLoadingTaxonomy: boolean;
  /** The applied keyword (from the URL / last submit). */
  keyword: string;
  matchMode: TextMatchMode;
  keywordSupported: boolean;
  /** Apply the keyword and match mode and run the search. */
  onSubmit: (keyword: string, matchMode: TextMatchMode) => void;
  isSearching: boolean;
}

/**
 * The search query bar: mode switch, the clause-type picker (the primary
 * control), and a keyword field. Facet changes apply on their own; the
 * keyword applies on Enter / Search.
 */
export const SearchQueryBar = forwardRef<HTMLDivElement, SearchQueryBarProps>(function SearchQueryBar(
  {
    mode,
    onModeChange,
    clauseTypes,
    clauseTypeLabelById,
    selectedClauseTypes,
    onToggleClauseType,
    onClearClauseTypes,
    isLoadingTaxonomy,
    keyword,
    matchMode,
    keywordSupported,
    onSubmit,
    isSearching,
  },
  ref,
) {
  const inputId = useId();
  const errorId = useId();
  const [draft, setDraft] = useState(keyword);
  const [draftMatch, setDraftMatch] = useState<TextMatchMode>(matchMode);
  const [error, setError] = useState<string | null>(null);
  const [synced, setSynced] = useState({ keyword, matchMode });
  // Pick up keyword changes from outside (URL, starter query, Clear all).
  if (synced.keyword !== keyword || synced.matchMode !== matchMode) {
    setSynced({ keyword, matchMode });
    setDraft(keyword);
    setDraftMatch(matchMode);
    setError(null);
  }

  const isTax = mode === "tax";
  const clauseLabel = isTax ? "Tax clause type" : "Clause type";
  const clauseNoun = isTax ? "tax clause types" : "clause types";
  // The info panel describes the first selected clause type.
  const infoId = selectedClauseTypes[0] ?? null;

  const submit = (nextMatch: TextMatchMode = draftMatch) => {
    const trimmed = draft.trim();
    if (keywordSupported && trimmed) {
      const problem = validateKeywordQuery(trimmed, nextMatch);
      setError(problem);
      if (problem) return;
    } else {
      setError(null);
    }
    onSubmit(keywordSupported ? trimmed : keyword, nextMatch);
  };

  const onFormSubmit = (event: FormEvent) => {
    event.preventDefault();
    submit();
  };

  const keywordInput = (
    // Its own row on phones (above the match select), so a long phrase stays
    // readable. Without the select (tax mode) it shares the row with Search.
    <div className={cn("relative min-w-0 flex-1", keywordSupported && "basis-full sm:basis-0")}>
      <SearchIcon
        className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
        aria-hidden="true"
      />
      <input
        id={inputId}
        type="search"
        value={draft}
        maxLength={MAX_KEYWORD_CHARS}
        onChange={(event) => {
          setDraft(event.target.value);
          if (error) setError(null);
        }}
        disabled={!keywordSupported}
        placeholder={
          keywordSupported
            ? "Keyword or phrase"
            : "Keyword search isn't available for tax clauses"
        }
        aria-label="Keyword"
        aria-invalid={error ? true : undefined}
        aria-describedby={error ? errorId : undefined}
        className={cn(
          "h-10 w-full rounded-md border border-input bg-background pl-9 pr-8 text-base text-foreground placeholder:text-muted-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-60 sm:text-sm [&::-webkit-search-cancel-button]:hidden",
          error && "border-destructive",
        )}
      />
      {draft && keywordSupported ? (
        <button
          type="button"
          onClick={() => {
            setDraft("");
            setError(null);
            if (keyword) onSubmit("", draftMatch);
          }}
          className="absolute right-1.5 top-1/2 inline-flex h-7 w-7 -translate-y-1/2 items-center justify-center rounded text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-label="Clear keyword"
        >
          <X className="h-4 w-4" aria-hidden="true" />
        </button>
      ) : null}
    </div>
  );

  return (
    <div
      ref={ref}
      className="z-20 border-b border-border bg-background/95 py-3 supports-[backdrop-filter]:bg-background/80 supports-[backdrop-filter]:backdrop-blur lg:sticky lg:top-16"
    >
      <div className="app-frame">
        <h1 className="sr-only" id="search-page-title">
          Search {MODE_NOUNS[mode].many}
        </h1>
        <form
          role="search"
          aria-labelledby="search-page-title"
          onSubmit={onFormSubmit}
          className="flex flex-col gap-2 md:flex-row md:flex-wrap md:items-center lg:flex-nowrap"
        >
          <SegmentedControl
            aria-label="Search mode"
            value={mode}
            onValueChange={onModeChange}
            className="w-full shrink-0 md:w-auto [&>*]:flex-1 md:[&>*]:flex-none"
            options={MODE_ORDER.map((value) => ({ value, label: MODE_NOUNS[value].title }))}
          />
          <div className="flex min-w-0 items-center gap-1 md:w-64 lg:w-56 xl:w-72">
            <NestedCheckboxFilter
              label={clauseLabel}
              placeholder={`All ${clauseNoun}`}
              noun={clauseNoun}
              data={clauseTypes}
              selectedValues={selectedClauseTypes}
              onToggle={onToggleClauseType}
              onClear={onClearClauseTypes}
              labelById={clauseTypeLabelById}
              isLoading={isLoadingTaxonomy}
              className="min-w-0 flex-1"
              triggerClassName={selectedClauseTypes.length > 0 ? "border-primary/50 bg-primary/5" : undefined}
            />
            {infoId ? (
              <ClauseTypeInfo
                key={`${mode}:${infoId}`}
                kind={isTax ? "tax" : "main"}
                standardId={infoId}
                label={clauseTypeLabelById[infoId] ?? clauseLabel}
              />
            ) : (
              <Tooltip>
                <TooltipTrigger asChild>
                  <ReferenceLink
                    to="/taxonomy"
                    hideIcon
                    className="tooltip-help-trigger shrink-0"
                    aria-label="About the clause taxonomy (opens the Taxonomy page in a new tab)"
                  >
                    ?
                  </ReferenceLink>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="max-w-[220px] text-xs">
                  How clause types are defined. Opens the Taxonomy page in a new tab.
                </TooltipContent>
              </Tooltip>
            )}
          </div>
          <div className="flex min-w-0 flex-1 flex-wrap items-center gap-2 sm:flex-nowrap md:basis-full lg:basis-auto">
            {keywordInput}
            {keywordSupported ? (
              <Select
                value={draftMatch}
                onValueChange={(value) => {
                  const next = value as TextMatchMode;
                  setDraftMatch(next);
                  if (keyword || draft.trim()) submit(next);
                }}
              >
                <SelectTrigger className="h-10 flex-1 text-sm sm:w-[8.5rem] sm:flex-none sm:shrink-0" aria-label="Keyword match">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {(Object.keys(TEXT_MATCH_MODE_LABELS) as TextMatchMode[]).map((value) => (
                    <SelectItem key={value} value={value}>
                      {TEXT_MATCH_MODE_LABELS[value]}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            ) : null}
            <Button type="submit" className="h-10 w-10 shrink-0 gap-2 px-0 sm:w-auto sm:px-4" aria-label="Search" aria-describedby="search-results-status">
              {isSearching ? (
                <span
                  className="h-4 w-4 animate-spin rounded-full border-2 border-primary-foreground border-t-transparent"
                  aria-hidden="true"
                />
              ) : (
                <SearchIcon className="h-4 w-4 sm:hidden" aria-hidden="true" />
              )}
              <span className="hidden sm:inline">Search</span>
            </Button>
          </div>
        </form>
        {error ? (
          <p id={errorId} role="alert" className="mt-1.5 text-sm text-destructive">
            {error}
          </p>
        ) : null}
      </div>
    </div>
  );
});
