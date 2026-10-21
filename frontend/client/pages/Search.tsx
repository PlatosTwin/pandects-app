import {
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactElement,
} from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { SlidersHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "@/components/ui/sheet";
import { ResultsToolbar, type ResultsDensity, type ResultsSortField } from "@/components/ResultsToolbar";
import { useHydrated } from "@/components/landing/use-hydrated";
import { useAuth } from "@/hooks/use-auth";
import { useFilterOptions, useFilterOptionSearch } from "@/hooks/use-filter-options";
import { describeSearchError, useSearch, type SortDirection, type SortField } from "@/hooks/use-search";
import { useTaxonomy } from "@/hooks/use-taxonomy";
import { trackEvent } from "@/lib/analytics";
import { isApiError } from "@/lib/api/client";
import { buildAccountPathWithNext } from "@/lib/auth-next";
import { buildAgreementHref } from "@/lib/agreement-links";
import { indexClauseTypeLabels, indexClauseTypePaths } from "@/lib/clause-type-index";
import type { ClauseTypeTree } from "@/lib/clause-types";
import { DEFAULT_PAGE_SIZE } from "@/lib/constants";
import {
  DEALS_SEARCH_CONFIG,
  KEYWORD_SEARCH_SUPPORT,
  MAX_KEYWORD_SEARCH_PAGE,
  SEARCH_CONFIGS,
  SECTIONS_SEARCH_CONFIG,
  TAX_CLAUSES_SEARCH_CONFIG,
} from "@/lib/search-configs";
import { stashCompareClauses } from "@/lib/tax-compare-handoff";
import {
  buildSearchStateParams,
  createEmptySearchFilters,
  parseSearchFilters,
} from "@/lib/url-params";
import { parseSearchMode, type SearchMode, type SearchQueryFilters, type TextMatchMode } from "@shared/search";
import type { SearchFilters, SearchResult } from "@shared/sections";
import type { TaxClauseSearchResult } from "@shared/tax-clauses";
import type { TransactionSearchResult } from "@shared/transactions";

import { keywordHighlightPattern } from "./search/Highlight";
import { validateKeywordQuery } from "./search/keyword-query";
import {
  SearchPagination,
  SearchPaginationFallback,
  SearchResultsTable,
  SearchResultsTableFallback,
  SearchSidebar,
  SearchSidebarFallback,
  TaxClauseResultsList,
  TransactionResultsFallback,
  TransactionResultsList,
} from "./search/lazy";
import { SearchActiveFilters, useActiveChips } from "./search/SearchActiveFilters";
import { buildStarterQueries, SearchEmptyState } from "./search/SearchEmptyState";
import { SearchQueryBar } from "./search/SearchQueryBar";
import { SearchResultsPanel } from "./search/SearchResultsPanel";
import {
  countRailFilters,
  describeSearch,
  filtersForModeSwitch,
  hasAnySearchFilter,
  loadRecentSearches,
  saveRecentSearch,
  type RecentSearch,
} from "./search/search-ui";

/** Facet edits apply on their own after this pause (see the Search page notes). */
const LIVE_APPLY_DELAY_MS = 350;
/** After this long a search shows "Still searching…". */
const SLOW_SEARCH_MS = 3000;
/** Commits closer together than this replace the newest recent search instead of adding one. */
const RECENT_REFINE_WINDOW_MS = 20_000;

/** CSV-export telemetry per mode (stable event names). */
const EXPORT_EVENTS: Record<SearchMode, string> = {
  sections: "sections_export_click",
  tax: "tax_clauses_export_click",
  transactions: "transactions_export_click",
};

const EMPTY_TREE: ClauseTypeTree = {};

/** Every draft field, cleared (hydrate merges, so absent keys must be listed). */
function blankFilters(): SearchQueryFilters {
  return {
    ...createEmptySearchFilters(),
    agreement_uuid: undefined,
    section_uuid: undefined,
    text_query: undefined,
    text_match_mode: undefined,
    include_rep_warranty: undefined,
  };
}

/** What a mode actually sends, minus paging and sort: equal keys mean "already applied". */
function applyKeyFor(mode: SearchMode, filters: SearchQueryFilters): string {
  return SEARCH_CONFIGS[mode]
    .buildApiParams({ filters: { ...filters, page: 1, page_size: 0 }, sortBy: null, sortDirection: "desc" })
    .toString();
}

function parseSort(params: URLSearchParams): { sortBy: SortField; sortDirection: SortDirection } {
  const sortBy = params.get("sort_by");
  const direction = params.get("sort_direction");
  return {
    sortBy: sortBy === "target" || sortBy === "acquirer" ? sortBy : "year",
    sortDirection: direction === "asc" ? "asc" : "desc",
  };
}

function readDensity(): ResultsDensity {
  try {
    return window.localStorage.getItem("pandects.resultsDensity") === "compact" ? "compact" : "comfy";
  } catch {
    return "comfy";
  }
}

/**
 * Search page.
 *
 * - The URL is the source of truth for a shared search: an outside URL change
 *   (first load, a starter or recent-search link, back/forward) re-hydrates
 *   the draft; draft changes are written back with `replace`.
 * - Facet changes apply live after a short pause; the keyword applies on
 *   Enter / Search. Results stay on screen (dimmed) while the next page loads.
 * - Sections and Tax with nothing selected show starter queries; Deals with
 *   no filters browses every deal.
 */
export default function Search() {
  const hydrated = useHydrated();
  const { status: authStatus } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [mode, setMode] = useState<SearchMode>(() => parseSearchMode(searchParams.get("mode")));

  // The sections search owns the draft (filters + sort) shared by every mode;
  // the other modes search that draft via performSearch({ filters, ... }).
  const sectionSearch = useSearch(SECTIONS_SEARCH_CONFIG);
  const dealSearch = useSearch(DEALS_SEARCH_CONFIG);
  const taxSearch = useSearch(TAX_CLAUSES_SEARCH_CONFIG);
  const searches = useMemo(
    () => ({ sections: sectionSearch, transactions: dealSearch, tax: taxSearch }) as const,
    [sectionSearch, dealSearch, taxSearch],
  );
  // Effects and callbacks read the latest hook state through this ref.
  const searchesRef = useRef(searches);
  useEffect(() => {
    searchesRef.current = searches;
  });
  const active = searches[mode];
  const { filters, sortBy, sortDirection } = sectionSearch;
  const { hydrateFilters, setSortBy, setSortDirection, toggleFilterValue } = sectionSearch.actions;
  const currentSort: SortField = sortBy ?? "year";

  const [density, setDensity] = useState<ResultsDensity>(readDensity);
  const [recentSearches, setRecentSearches] = useState<RecentSearch[]>(loadRecentSearches);
  const [mobileFiltersOpen, setMobileFiltersOpen] = useState(false);
  const [barHeight, setBarHeight] = useState(0);
  const barRef = useRef<HTMLDivElement | null>(null);
  const lastRecordAtRef = useRef(0);

  useEffect(() => {
    window.scrollTo({ top: 0, left: 0 });
  }, []);

  // Sticky offsets below the query bar follow its real height.
  useEffect(() => {
    const node = barRef.current;
    if (!node || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => setBarHeight(node.offsetHeight));
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const currentUrl = `${location.pathname}${location.search}`;
  const signInHref = useMemo(
    () => buildAccountPathWithNext(`${location.pathname}${location.search}${location.hash}`),
    [location.hash, location.pathname, location.search],
  );

  // --- Filter data -----------------------------------------------------------
  const filterOptions = useFilterOptions({
    enabled: hydrated,
    fields: ["target_counsels", "acquirer_counsels", "target_industries", "acquirer_industries", "clause_types"],
  });
  const loadFilterOptionSearch = useFilterOptionSearch();
  const loadTargetOptions = useCallback((q: string) => loadFilterOptionSearch("target", q), [loadFilterOptionSearch]);
  const loadAcquirerOptions = useCallback(
    (q: string) => loadFilterOptionSearch("acquirer", q),
    [loadFilterOptionSearch],
  );
  const taxTaxonomy = useTaxonomy({ kind: "tax", enabled: hydrated && mode === "tax" });
  const sectionTree = filterOptions.clause_types;
  const taxTree = taxTaxonomy.taxonomyTree ?? EMPTY_TREE;
  const clauseTree = mode === "tax" ? taxTree : sectionTree;
  const isLoadingTaxonomy = mode === "tax" ? taxTaxonomy.isLoading : filterOptions.isLoading;
  const clauseTypeLabelById = useMemo(() => indexClauseTypeLabels(clauseTree), [clauseTree]);
  const clauseTypePathById = useMemo(() => indexClauseTypePaths(sectionTree), [sectionTree]);

  // --- Committing searches -----------------------------------------------------
  const appliedKeyRef = useRef<Record<SearchMode, string | null>>({
    sections: null,
    transactions: null,
    tax: null,
  });

  const runSearch = useCallback(
    (
      targetMode: SearchMode,
      nextFilters: SearchQueryFilters,
      options: { markAsSearched?: boolean; sortBy?: SortField; sortDirection?: SortDirection } = {},
    ) => {
      appliedKeyRef.current[targetMode] = applyKeyFor(targetMode, nextFilters);
      const nextSortBy = options.sortBy ?? currentSort;
      const nextDirection = options.sortDirection ?? sortDirection;
      void searchesRef.current[targetMode].actions.performSearch({
        filters: nextFilters,
        markAsSearched: options.markAsSearched ?? false,
        sortBy: nextSortBy,
        sortDirection: nextDirection,
      });
      if (options.markAsSearched && hasAnySearchFilter(nextFilters, targetMode)) {
        const params = buildSearchStateParams({
          filters: { ...nextFilters, page: 1 },
          mode: targetMode,
          sortBy: nextSortBy,
          sortDirection: nextDirection,
        });
        const now = Date.now();
        const refine = now - lastRecordAtRef.current < RECENT_REFINE_WINDOW_MS;
        lastRecordAtRef.current = now;
        const entry: RecentSearch = {
          search: params.toString(),
          label: describeSearch(targetMode, nextFilters, clauseTypeLabelById),
          mode: targetMode,
        };
        setRecentSearches(saveRecentSearch(entry, { replaceNewest: refine }));
      }
    },
    [currentSort, sortDirection, clauseTypeLabelById],
  );

  // --- URL <-> state -----------------------------------------------------------
  const lastUrlRef = useRef<string | null>(null);
  const skipUrlWriteRef = useRef(false);

  // Hydrate from the URL when it changes from outside (not from our own write).
  useEffect(() => {
    const current = searchParams.toString();
    if (current === lastUrlRef.current) return;
    lastUrlRef.current = current;
    skipUrlWriteRef.current = true;

    const nextMode = parseSearchMode(searchParams.get("mode"));
    const parsed = parseSearchFilters(searchParams);
    const nextFilters: SearchQueryFilters = {
      ...blankFilters(),
      ...parsed,
      page: parsed.page ?? 1,
      page_size: parsed.page_size ?? DEFAULT_PAGE_SIZE,
    };
    const sort = parseSort(searchParams);
    hydrateFilters(nextFilters);
    setSortBy(sort.sortBy);
    setSortDirection(sort.sortDirection);
    setMode(nextMode);
    Object.values(searchesRef.current).forEach((search) => search.actions.clearSelection());

    const shouldSearch =
      nextMode === "transactions" || searchParams.has("page") || hasAnySearchFilter(nextFilters, nextMode);
    if (shouldSearch) {
      runSearch(nextMode, nextFilters, { markAsSearched: true, ...sort });
    } else {
      appliedKeyRef.current[nextMode] = applyKeyFor(nextMode, nextFilters);
      searchesRef.current[nextMode].actions.clearResults();
    }
    // runSearch is read at URL-change time only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams, hydrateFilters, setSortBy, setSortDirection]);

  // Write the draft back to the URL.
  useEffect(() => {
    if (skipUrlWriteRef.current) {
      // The hydration pass still sees the previous draft; skip it once.
      skipUrlWriteRef.current = false;
      return;
    }
    const next = buildSearchStateParams({
      filters,
      mode,
      sortBy: currentSort,
      sortDirection,
      taxIncludeRepWarranty: filters.include_rep_warranty,
    }).toString();
    if (next !== searchParams.toString()) {
      lastUrlRef.current = next;
      setSearchParams(next, { replace: true });
    }
  }, [filters, mode, currentSort, sortDirection, searchParams, setSearchParams]);

  // Live apply: once the draft differs from what this mode last searched,
  // search it (or, with nothing selected in Sections/Tax, show the starters).
  const applyKey = useMemo(() => applyKeyFor(mode, filters), [mode, filters]);
  useEffect(() => {
    if (appliedKeyRef.current[mode] === applyKey) return;
    const timer = window.setTimeout(() => {
      if (mode !== "transactions" && !hasAnySearchFilter(filters, mode)) {
        appliedKeyRef.current[mode] = applyKey;
        searchesRef.current[mode].actions.clearResults();
        return;
      }
      runSearch(mode, { ...filters, page: 1 }, { markAsSearched: true });
    }, LIVE_APPLY_DELAY_MS);
    return () => window.clearTimeout(timer);
  }, [applyKey, mode, filters, runSearch]);

  // --- Slow-search notice --------------------------------------------------------
  const commitKey = `${mode}:${active.committed?.nonce ?? 0}`;
  const [slowCommitKey, setSlowCommitKey] = useState<string | null>(null);
  useEffect(() => {
    if (!active.isSearching) return;
    const timer = window.setTimeout(() => setSlowCommitKey(commitKey), SLOW_SEARCH_MS);
    return () => window.clearTimeout(timer);
  }, [active.isSearching, commitKey]);
  const isSlow = active.isSearching && slowCommitKey === commitKey;

  // --- Handlers ----------------------------------------------------------------
  const patchFilters = useCallback(
    (patch: Partial<SearchQueryFilters>) => {
      Object.entries(patch).forEach(([field, value]) => {
        trackEvent("sections_filter_change", {
          filter_field: field,
          filter_value: (Array.isArray(value) ? value.join(",") : String(value ?? "")).substring(0, 50),
          current_filters: Object.keys(filters).length,
        });
      });
      hydrateFilters({ ...patch, page: 1 });
    },
    [filters, hydrateFilters],
  );

  const onToggleFilterValue = useCallback(
    (field: string, value: string) => {
      trackEvent("sections_filter_change", {
        filter_field: field,
        filter_value: value.substring(0, 50),
        current_filters: Object.keys(filters).length,
      });
      toggleFilterValue(field as keyof SearchFilters, value);
      hydrateFilters({ page: 1 });
    },
    [filters, toggleFilterValue, hydrateFilters],
  );

  const clearAll = () => {
    trackEvent("sections_filters_cleared", { filter_count: Object.values(filters).flat().length });
    hydrateFilters({ ...blankFilters(), page_size: filters.page_size });
  };

  const onModeChange = (nextMode: SearchMode) => {
    if (nextMode === mode) return;
    const next = filtersForModeSwitch(filters, mode, nextMode);
    if (next !== filters) hydrateFilters({ clauseType: next.clauseType, page: 1 });
    else hydrateFilters({ page: 1 });
    Object.values(searches).forEach((search) => search.actions.clearSelection());
    setMode(nextMode);
  };

  const onKeywordSubmit = (keyword: string, matchMode: TextMatchMode) => {
    const patch: Partial<SearchQueryFilters> = {
      text_query: keyword || undefined,
      text_match_mode: keyword && matchMode !== "phrase" ? matchMode : undefined,
      page: 1,
    };
    hydrateFilters(patch);
    runSearch(mode, { ...filters, ...patch }, { markAsSearched: true });
  };

  const goToPage = (page: number) => {
    hydrateFilters({ page });
    runSearch(mode, { ...filters, page });
    document.getElementById("search-results-status")?.scrollIntoView({ block: "nearest" });
  };

  const changePageSize = (pageSize: number) => {
    hydrateFilters({ page: 1, page_size: pageSize });
    runSearch(mode, { ...filters, page: 1, page_size: pageSize });
  };

  const sortResults = (field: ResultsSortField, direction: SortDirection = sortDirection) => {
    setSortBy(field);
    setSortDirection(direction);
    hydrateFilters({ page: 1 });
    runSearch(mode, { ...filters, page: 1 }, { sortBy: field, sortDirection: direction });
  };
  const toggleSortDirection = () => sortResults(currentSort, sortDirection === "asc" ? "desc" : "asc");
  const onTableSort = (field: ResultsSortField) =>
    field === currentSort
      ? toggleSortDirection()
      : sortResults(field, field === "year" ? "desc" : "asc");

  const onDensityChange = (next: ResultsDensity) => {
    setDensity(next);
    try {
      window.localStorage.setItem("pandects.resultsDensity", next);
    } catch {
      // Storage unavailable: the choice lasts for this visit only.
    }
  };

  const downloadCsv = (scope: "all" | "selected") => {
    trackEvent(EXPORT_EVENTS[mode], {
      export_format: "csv",
      result_count: scope === "selected" ? active.selectedResults.size : active.totalCount,
      is_filtered: Object.values(filters).flat().length > 0,
    });
    void active.actions.downloadCSV(scope);
  };

  const onTaxCompare = () => {
    const clauses = taxSearch.results.filter((r) => taxSearch.selectedResults.has(r.id));
    stashCompareClauses(clauses);
    trackEvent("tax_clauses_compare_click", { selected_count: clauses.length });
    navigate("/compare/tax");
  };

  // --- Derived view data -------------------------------------------------------
  const committedFilters = active.committed?.filters;
  const keywordApplied = Boolean(KEYWORD_SEARCH_SUPPORT[mode] && committedFilters?.text_query?.trim());
  const highlightPattern = useMemo(
    () =>
      keywordApplied
        ? keywordHighlightPattern(committedFilters?.text_query, committedFilters?.text_match_mode)
        : null,
    [keywordApplied, committedFilters?.text_query, committedFilters?.text_match_mode],
  );
  const chips = useActiveChips(filters, mode, clauseTypeLabelById, filterOptions.industry_labels);
  const railCount = countRailFilters(filters, mode);
  const isAnonymous = active.hasSearched ? active.access.tier === "anonymous" : authStatus === "anonymous";
  const starters = useMemo(() => buildStarterQueries(mode, sectionTree, taxTree), [mode, sectionTree, taxTree]);
  const startPosition = (active.currentPage - 1) * active.pageSize + 1;
  const hrefFor = (agreementUuid: string, sectionUuid?: string | null) =>
    buildAgreementHref({ agreementUuid, sectionUuid, from: currentUrl });

  const onOpenSection = (result: SearchResult, position: number) => {
    trackEvent("sections_result_click", { position, year: result.year, verified: result.verified });
  };

  const error = active.error;
  const errorStatus = isApiError(error) ? error.status : undefined;
  const isClientError = errorStatus !== undefined && errorStatus >= 400 && errorStatus < 500 && errorStatus !== 429;
  // The API rejects a bad keyword with a generic 422; explain it the way the query bar would.
  const keywordProblem =
    error && keywordApplied && committedFilters?.text_query
      ? validateKeywordQuery(committedFilters.text_query, committedFilters.text_match_mode ?? "phrase")
      : null;

  const pageSelection = (() => {
    if (mode === "transactions" || active.results.length === 0) return undefined;
    const ids = (active.results as Array<{ id: string }>).map((r) => r.id);
    const selectedCount = ids.filter((id) => active.selectedResults.has(id)).length;
    return {
      checked: selectedCount === 0 ? false : selectedCount === ids.length ? true : ("indeterminate" as const),
      onToggle: active.actions.toggleSelectAll,
      label: `Select all ${mode === "tax" ? "tax clauses" : "sections"} on this page`,
    };
  })();

  const resultsByMode: Record<SearchMode, { skeleton: ReactElement; render: () => ReactElement }> = {
    sections: {
      skeleton: <SearchResultsTableFallback />,
      render: () => (
        <SearchResultsTable
          searchResults={sectionSearch.results}
          startPosition={startPosition}
          selectedResults={sectionSearch.selectedResults}
          onToggleResultSelection={sectionSearch.actions.toggleResultSelection}
          clauseTypePathByStandardId={clauseTypePathById}
          getAgreementHref={(r) => hrefFor(r.agreement_uuid, r.section_uuid)}
          onOpenAgreement={onOpenSection}
          density={density}
          highlightPattern={highlightPattern}
          signInHref={signInHref}
          preferredTypeIds={committedFilters?.clauseType ?? []}
        />
      ),
    },
    tax: {
      skeleton: <SearchResultsTableFallback />,
      render: () => (
        <TaxClauseResultsList
          results={taxSearch.results}
          getAgreementHref={(r: TaxClauseSearchResult) => hrefFor(r.agreement_uuid, r.section_uuid)}
          clauseTypeLabelById={clauseTypeLabelById}
          selectedResults={taxSearch.selectedResults}
          onToggleResultSelection={taxSearch.actions.toggleResultSelection}
          density={density}
          isAnonymous={isAnonymous}
          signInHref={signInHref}
          preferredTypeIds={committedFilters?.clauseType ?? []}
        />
      ),
    },
    transactions: {
      skeleton: <TransactionResultsFallback />,
      render: () => (
        <TransactionResultsList
          results={dealSearch.results}
          getAgreementHref={(r: TransactionSearchResult, sectionUuid?: string | null) =>
            hrefFor(r.agreement_uuid, sectionUuid)
          }
          showMatches={Boolean(
            (committedFilters?.clauseType?.length ?? 0) > 0 ||
              committedFilters?.section_uuid?.trim() ||
              keywordApplied,
          )}
          clauseTypeLabelById={clauseTypeLabelById}
          selectedResults={dealSearch.selectedResults}
          onToggleResultSelection={dealSearch.actions.toggleResultSelection}
          onToggleSelectAll={dealSearch.actions.toggleSelectAll}
          sortBy={currentSort}
          sortDirection={sortDirection}
          onSort={onTableSort}
          density={density}
          highlightPattern={highlightPattern}
        />
      ),
    },
  };
  const view = resultsByMode[mode];

  const sidebarProps = {
    mode,
    filters,
    target_counsels: filterOptions.target_counsels,
    acquirer_counsels: filterOptions.acquirer_counsels,
    target_industries: filterOptions.target_industries,
    acquirer_industries: filterOptions.acquirer_industries,
    industry_labels: filterOptions.industry_labels,
    isLoadingFilterOptions: filterOptions.isLoading,
    onToggleFilterValue,
    onPatchFilters: patchFilters,
    onTextFilterChange: (field: "agreement_uuid" | "section_uuid", value: string) =>
      patchFilters({ [field]: value || undefined }),
    loadTargetOptions,
    loadAcquirerOptions,
    notice: filterOptions.error ? (
      <p role="alert" className="py-2 text-sm text-destructive dark:text-red-400">
        Some filter options couldn't load. {filterOptions.error}
      </p>
    ) : null,
  };

  const mobileFilters = (
    <Button
      type="button"
      variant="outline"
      size="sm"
      className="h-9 gap-2 lg:hidden"
      onClick={() => setMobileFiltersOpen(true)}
      aria-haspopup="dialog"
    >
      <SlidersHorizontal className="h-4 w-4" aria-hidden="true" />
      Filters{railCount > 0 ? ` (${railCount})` : ""}
    </Button>
  );

  const keywordCapped = keywordApplied ? MAX_KEYWORD_SEARCH_PAGE : undefined;
  const pagination = hydrated ? (
    <Suspense fallback={<SearchPaginationFallback />}>
      <SearchPagination
        currentPage={active.currentPage}
        totalPages={active.totalPages}
        pageSize={active.pageSize}
        totalCount={active.totalCount}
        totalCountIsApproximate={active.totalCountIsApproximate}
        hasNext={active.hasNext}
        hasPrev={active.hasPrev}
        onPageChange={goToPage}
        onPageSizeChange={changePageSize}
        isLoading={active.isSearching}
        isLimited={active.access.tier === "anonymous"}
        lockNavigation={mode !== "transactions"}
        signInHref={signInHref}
        maxPage={keywordCapped}
      />
    </Suspense>
  ) : (
    <SearchPaginationFallback />
  );

  const style = { "--search-bar-h": `${barHeight}px` } as CSSProperties;

  return (
    // overflow-x-clip, not -hidden: "hidden" makes this a scroll container,
    // which stops the sticky query bar, filter rail and results header from sticking.
    <div className="w-full overflow-x-clip" style={style}>
      <SearchQueryBar
        ref={barRef}
        mode={mode}
        onModeChange={onModeChange}
        clauseTypes={clauseTree}
        clauseTypeLabelById={clauseTypeLabelById}
        selectedClauseTypes={filters.clauseType ?? []}
        onToggleClauseType={(id) => onToggleFilterValue("clauseType", id)}
        onClearClauseTypes={() => patchFilters({ clauseType: [] })}
        isLoadingTaxonomy={isLoadingTaxonomy}
        keyword={filters.text_query ?? ""}
        matchMode={filters.text_match_mode ?? "phrase"}
        keywordSupported={KEYWORD_SEARCH_SUPPORT[mode]}
        onSubmit={onKeywordSubmit}
        isSearching={active.isSearching}
      />

      <div className="flex min-h-[calc(100vh-4rem)]">
        {hydrated ? (
          <Suspense fallback={<SearchSidebarFallback />}>
            <SearchSidebar {...sidebarProps} />
          </Suspense>
        ) : (
          <SearchSidebarFallback />
        )}

        <SearchResultsPanel
          mode={mode}
          hasSearched={active.hasSearched}
          isSearching={active.isSearching}
          isSlow={isSlow}
          isFirstLoad={active.isSearching && active.results.length === 0}
          error={error}
          errorTitle={isClientError ? "This search couldn't run" : undefined}
          errorBody={error ? (keywordProblem ?? describeSearchError(error)) : undefined}
          onRetry={isClientError ? undefined : () => runSearch(mode, filters)}
          signInHref={signInHref}
          totalCount={active.totalCount}
          totalCountIsApproximate={active.totalCountIsApproximate}
          resultsLength={active.results.length}
          mobileFilters={mobileFilters}
          sortControls={
            <ResultsToolbar
              sortBy={currentSort}
              sortDirection={sortDirection}
              onSortResults={(field) => sortResults(field)}
              onToggleSortDirection={toggleSortDirection}
              sortSelectLabel={`Sort ${mode === "transactions" ? "deals" : mode === "tax" ? "tax clauses" : "sections"} by`}
              density={density}
              onDensityChange={onDensityChange}
            />
          }
          chips={<SearchActiveFilters chips={chips} onRemove={patchFilters} onClearAll={clearAll} />}
          selectPage={pageSelection}
          onDownloadAll={() => downloadCsv("all")}
          selectedSize={active.selectedResults.size}
          onDownloadSelected={() => downloadCsv("selected")}
          onClearSelection={active.actions.clearSelection}
          onCompare={mode === "tax" ? onTaxCompare : undefined}
          emptyState={
            <SearchEmptyState
              mode={mode}
              starters={starters}
              recent={recentSearches}
              isLoadingStarters={isLoadingTaxonomy}
            />
          }
          skeleton={view.skeleton}
          results={hydrated ? <Suspense fallback={view.skeleton}>{view.render()}</Suspense> : view.skeleton}
          pagination={pagination}
        />
      </div>

      <Sheet open={mobileFiltersOpen} onOpenChange={setMobileFiltersOpen}>
        <SheetContent side="bottom" className="flex max-h-[85dvh] flex-col gap-0 rounded-t-xl p-0">
          <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
            <SheetTitle className="text-base">Filters{railCount > 0 ? ` (${railCount})` : ""}</SheetTitle>
            <SheetDescription className="sr-only">Filters apply as you change them.</SheetDescription>
            {chips.length > 0 ? (
              <button
                type="button"
                onClick={clearAll}
                className="mr-8 text-sm font-medium text-primary hover:underline"
              >
                Clear all
              </button>
            ) : null}
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain">
            {hydrated && mobileFiltersOpen ? (
              <Suspense fallback={<SearchSidebarFallback variant="sheet" />}>
                <SearchSidebar {...sidebarProps} variant="sheet" />
              </Suspense>
            ) : null}
          </div>
          <div className="border-t border-border p-3 pb-[calc(0.75rem+env(safe-area-inset-bottom))]">
            <Button type="button" className="h-11 w-full" onClick={() => setMobileFiltersOpen(false)}>
              {active.hasSearched && !active.isSearching
                ? `Show ${active.totalCountIsApproximate ? "about " : ""}${active.totalCount.toLocaleString()} results`
                : "Show results"}
            </Button>
          </div>
        </SheetContent>
      </Sheet>
    </div>
  );
}
