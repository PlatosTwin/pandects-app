import {
  Suspense,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactElement,
} from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { AVAILABLE_YEARS, BREAKPOINT_LG } from "@/lib/constants";
import type { SearchFilters } from "@shared/sections";
import { useSearch, type SortDirection, type SortField } from "@/hooks/use-search";
import { useTaxonomy } from "@/hooks/use-taxonomy";
import { useFilterOptions, useFilterOptionSearch } from "@/hooks/use-filter-options";
import { Alert, AlertDescription } from "@/components/ui/alert";
import ErrorModal from "@/components/ErrorModal";
import { useAuth } from "@/hooks/use-auth";
import type { ClauseTypeTree } from "@/lib/clause-types";
import { indexClauseTypeLabels, indexClauseTypePaths } from "@/lib/clause-type-index";
import { scheduleWhenBrowserIdle, trackEvent } from "@/lib/analytics";
import { buildAccountPathWithNext } from "@/lib/auth-next";
import {
  buildSearchStateParams,
  parseSearchFilters,
  type ParsedSearchFilters,
} from "@/lib/url-params";
import {
  DEALS_SEARCH_CONFIG,
  SECTIONS_SEARCH_CONFIG,
  TAX_CLAUSES_SEARCH_CONFIG,
} from "@/lib/search-configs";
import { stashCompareClauses } from "@/lib/tax-compare-handoff";
import { parseSearchMode, type SearchMode } from "@shared/search";
import type { TransactionSearchResult } from "@shared/transactions";
import type { TaxClauseSearchResult } from "@shared/tax-clauses";

import {
  SearchResultsTable,
  SearchResultsTableFallback,
  SearchSidebar,
  SearchSidebarFallback,
  TaxClauseResultsList,
  TransactionResultsFallback,
  TransactionResultsList,
} from "./search/lazy";
import { SearchHeader } from "./search/SearchHeader";
import { SearchActionsBar } from "./search/SearchActionsBar";
import { SearchResultsPanel } from "./search/SearchResultsPanel";

/** Mode-specific labels and telemetry the unified search hook doesn't own. */
const MODE_UI: Record<SearchMode, { exportEvent: string; sidebarClauseTypeLabel: string }> = {
  sections: { exportEvent: "sections_export_click", sidebarClauseTypeLabel: "Section Type" },
  tax: { exportEvent: "tax_clauses_export_click", sidebarClauseTypeLabel: "Tax clause type" },
  transactions: {
    exportEvent: "transactions_export_click",
    sidebarClauseTypeLabel: "Section Type",
  },
};

const EMPTY_CLAUSE_TYPE_TREE: ClauseTypeTree = {};

export default function Search() {
  const { status: authStatus } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [hasHydrated, setHasHydrated] = useState(false);
  const [searchMode, setSearchMode] = useState<SearchMode>(() =>
    parseSearchMode(searchParams.get("mode")),
  );
  const modeButtonRefs = useRef<Record<SearchMode, HTMLButtonElement | null>>({
    sections: null,
    transactions: null,
    tax: null,
  });
  const isHydratingFromUrlRef = useRef(true);
  // The sections search owns the draft (filters + sort) shared by all modes;
  // the other modes search that draft via performSearch({ filters, ... }).
  const sectionSearch = useSearch(SECTIONS_SEARCH_CONFIG);
  const transactionSearch = useSearch(DEALS_SEARCH_CONFIG);
  const taxSearch = useSearch(TAX_CLAUSES_SEARCH_CONFIG);
  const searches = {
    sections: sectionSearch,
    transactions: transactionSearch,
    tax: taxSearch,
  } as const;
  const {
    filters,
    results: searchResults,
    selectedResults,
    hasSearched: hasSearchedSections,
    sortBy: currentSort,
    sortDirection: sort_direction,
    currentPage,
    pageSize: page_size,
    actions,
  } = sectionSearch;

  const {
    toggleFilterValue,
    setTextFilterValue,
    clearFilters: clearSectionFilters,
    setSortBy: sortResults,
    toggleSortDirection,
    toggleResultSelection,
    toggleSelectAll,
    clearSelection,
    hydrateFilters,
    updateFilter,
    setSortDirection,
    setIncludeRepWarranty,
  } = actions;
  const loadSearchFilterOptions = useFilterOptionSearch();
  const signInPath = useMemo(
    () => buildAccountPathWithNext(`${location.pathname}${location.search}${location.hash}`),
    [location.hash, location.pathname, location.search],
  );

  useEffect(() => {
    setHasHydrated(true);
    if (typeof window !== "undefined") {
      window.scrollTo({ top: 0, left: 0 });
    }
  }, []);

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isMobileFiltersOpen, setIsMobileFiltersOpen] = useState(false);
  const [shouldRenderDesktopSidebar, setShouldRenderDesktopSidebar] = useState(false);
  const [isDesktopLayout, setIsDesktopLayout] = useState(() => {
    if (typeof window === "undefined") return true;
    return window.innerWidth >= BREAKPOINT_LG;
  });
  const [resultsDensity, setResultsDensity] = useState<"comfy" | "compact">(() => {
    try {
      const stored = localStorage.getItem("pandects.resultsDensity");
      return stored === "compact" ? "compact" : "comfy";
    } catch {
      return "comfy";
    }
  });

  const updateResultsDensity = (density: "comfy" | "compact") => {
    setResultsDensity(density);
    try {
      localStorage.setItem("pandects.resultsDensity", density);
    } catch {
      // ignore
    }
  };

  // Navigation itself happens via the row's <Link>; this only records the click.
  const openAgreement = (result: (typeof searchResults)[number], position: number) => {
    trackEvent("sections_result_click", {
      position,
      year: result.year,
      verified: result.verified,
    });
  };

  const handleModeChange = (value: SearchMode) => {
    const nextMode: SearchMode = parseSearchMode(value);
    if (nextMode === searchMode) return;
    setSearchMode(nextMode);
    clearSelection();
    transactionSearch.actions.clearSelection();
    taxSearch.actions.clearSelection();
  };

  useEffect(() => {
    let wasDesktop: boolean | null = null;
    const handleResize = () => {
      const isDesktop = window.innerWidth >= BREAKPOINT_LG;
      setIsDesktopLayout(isDesktop);
      // Only reset the collapse state when the desktop/mobile breakpoint
      // actually crosses; same-side resizes keep the user's choice.
      if (wasDesktop !== isDesktop) {
        wasDesktop = isDesktop;
        setSidebarCollapsed(!isDesktop);
      }
    };

    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    if (!isDesktopLayout) {
      setShouldRenderDesktopSidebar(false);
      return;
    }

    if (hasSearchedSections || transactionSearch.hasSearched || taxSearch.hasSearched) {
      setShouldRenderDesktopSidebar(true);
      return;
    }

    const cancelDeferredSidebar = scheduleWhenBrowserIdle(() => {
      setShouldRenderDesktopSidebar(true);
    }, 1800);

    return cancelDeferredSidebar;
  }, [hasSearchedSections, isDesktopLayout, taxSearch.hasSearched, transactionSearch.hasSearched]);

  const shouldLoadFilterData =
    (isDesktopLayout && shouldRenderDesktopSidebar) ||
    isMobileFiltersOpen ||
    hasSearchedSections ||
    transactionSearch.hasSearched ||
    taxSearch.hasSearched;

  const {
    target_counsels,
    acquirer_counsels,
    target_industries,
    acquirer_industries,
    clause_types,
    isLoading: isLoadingFilterOptions,
    error: filterOptionsError,
  } = useFilterOptions({
    enabled: shouldLoadFilterData,
    fields: [
      "target_counsels",
      "acquirer_counsels",
      "target_industries",
      "acquirer_industries",
      "clause_types",
    ],
  });

  const years = AVAILABLE_YEARS;
  const { taxonomyTree: taxClauseTaxonomy, isLoading: isLoadingTaxTaxonomy } = useTaxonomy({
    kind: "tax",
    enabled: searchMode === "tax" && shouldLoadFilterData,
  });
  const taxClauseTypesNested: ClauseTypeTree = taxClauseTaxonomy ?? EMPTY_CLAUSE_TYPE_TREE;
  const sectionClauseTypesNested: ClauseTypeTree = clause_types;
  const clauseTypesNested: ClauseTypeTree =
    searchMode === "tax" ? taxClauseTypesNested : sectionClauseTypesNested;

  const clauseTypePathByStandardId = useMemo(
    () => indexClauseTypePaths(clauseTypesNested),
    [clauseTypesNested],
  );
  const clauseTypeLabelById = useMemo(
    () => indexClauseTypeLabels(clauseTypesNested),
    [clauseTypesNested],
  );

  // Error modals stay cross-mode ORs: an error surfaced by any mode's hook
  // must remain visible even after the user switches modes.
  const activeShowErrorModal =
    sectionSearch.showErrorModal || transactionSearch.showErrorModal || taxSearch.showErrorModal;
  const activeErrorMessage =
    sectionSearch.errorMessage || transactionSearch.errorMessage || taxSearch.errorMessage;

  // Dispatches to the search for `nextMode`, which may differ from the
  // current mode while hydrating from the URL.
  const runActiveSearch = async (
    nextMode: SearchMode,
    nextFilters: ParsedSearchFilters,
    markAsSearched: boolean = true,
    // Callers that just changed sort state must pass the next values
    // explicitly: setState hasn't landed yet, so the closure below still
    // holds the previous sort.
    overrideSortBy?: SortField,
    overrideSortDirection?: SortDirection,
  ) => {
    await searches[nextMode].actions.performSearch({
      filters: nextFilters,
      markAsSearched,
      sortBy: overrideSortBy ?? currentSort,
      sortDirection: overrideSortDirection ?? sort_direction,
    });
  };

  const buildAgreementHref = (agreementUuid: string, focusSectionUuid?: string | null) => {
    const params = new URLSearchParams();
    params.set("from", `${location.pathname}${location.search}`);
    if (focusSectionUuid) {
      params.set("focusSectionUuid", focusSectionUuid);
    }
    return `/agreements/${agreementUuid}?${params.toString()}`;
  };

  const getSectionAgreementHref = (result: (typeof searchResults)[number]) =>
    buildAgreementHref(result.agreement_uuid, result.section_uuid);

  const getTransactionAgreementHref = (
    result: TransactionSearchResult,
    focusSectionUuid?: string | null,
  ) =>
    buildAgreementHref(
      result.agreement_uuid,
      focusSectionUuid ?? result.matched_sections[0]?.section_uuid ?? null,
    );

  const dealClauseContextActive =
    filters.clauseType.length > 0 || Boolean(filters.section_uuid?.trim());

  useEffect(() => {
    if (!isHydratingFromUrlRef.current) return;

    const nextMode = parseSearchMode(searchParams.get("mode"));
    const nextFilters = parseSearchFilters(searchParams);
    const nextSortBy = searchParams.get("sort_by");
    const nextSortDirection = searchParams.get("sort_direction");

    hydrateFilters(nextFilters);
    setSearchMode(nextMode);
    const hydratedSortBy =
      nextSortBy === "year" || nextSortBy === "target" || nextSortBy === "acquirer"
        ? nextSortBy
        : undefined;
    if (hydratedSortBy) {
      sortResults(hydratedSortBy);
    }
    const hydratedSortDirection =
      nextSortDirection === "asc" || nextSortDirection === "desc"
        ? nextSortDirection
        : undefined;
    if (hydratedSortDirection) {
      setSortDirection(hydratedSortDirection);
    }

    const shouldSearch =
      searchParams.toString().length > 0 &&
      (nextFilters.page !== undefined ||
        nextFilters.agreement_uuid !== undefined ||
        nextFilters.section_uuid !== undefined ||
        Object.values(nextFilters).some((value) =>
          Array.isArray(value) ? value.length > 0 : false,
        ));

    isHydratingFromUrlRef.current = false;
    if (shouldSearch) {
      void runActiveSearch(nextMode, nextFilters, true, hydratedSortBy, hydratedSortDirection);
    }
  }, [hydrateFilters, searchParams, setSortDirection, sortResults]);

  useEffect(() => {
    if (isHydratingFromUrlRef.current) return;
    const nextParams = buildSearchStateParams({
      filters,
      mode: searchMode,
      sortBy: currentSort,
      sortDirection: sort_direction,
      taxIncludeRepWarranty: filters.include_rep_warranty,
    });
    if (nextParams.toString() !== searchParams.toString()) {
      setSearchParams(nextParams, { replace: true });
    }
  }, [currentSort, filters, searchMode, searchParams, setSearchParams, sort_direction]);

  const trackingActions = {
    toggleFilterValue: (field: string, value: string) => {
      trackEvent("sections_filter_change", {
        filter_field: field,
        filter_value: value.substring(0, 50),
        current_filters: Object.keys(filters).length,
      });
      toggleFilterValue(field as keyof SearchFilters, value);
    },
    setTextFilterValue: (field: string, value: string) => {
      trackEvent("sections_filter_change", {
        filter_field: field,
        filter_value: value.substring(0, 50),
        current_filters: Object.keys(filters).length,
      });
      setTextFilterValue(field as keyof SearchFilters, value);
    },
    performSearch: async () => {
      const nextFilters: SearchFilters = { ...filters, page: 1 };
      updateFilter("page", 1);
      await runActiveSearch(searchMode, nextFilters);
    },
    clearFilters: () => {
      trackEvent("sections_filters_cleared", {
        filter_count: Object.values(filters).flat().length,
      });
      clearSectionFilters();
      transactionSearch.actions.clearResults();
      taxSearch.actions.clearFilters();
    },
    goToPage: async (page: number) => {
      const nextFilters: SearchFilters = { ...filters, page };
      updateFilter("page", page);
      await runActiveSearch(searchMode, nextFilters, false);
    },
    changePageSize: async (size: number) => {
      const nextFilters: SearchFilters = { ...filters, page_size: size, page: 1 };
      updateFilter("page_size", size);
      updateFilter("page", 1);
      await runActiveSearch(searchMode, nextFilters, false);
    },
    sortResults: async (field: string) => {
      const nextField = field as SortField;
      sortResults(nextField);
      const nextFilters: SearchFilters = { ...filters };
      await runActiveSearch(searchMode, nextFilters, false, nextField, sort_direction);
    },
    toggleSortDirection: async () => {
      const nextDirection: SortDirection = sort_direction === "asc" ? "desc" : "asc";
      toggleSortDirection();
      const nextFilters: SearchFilters = { ...filters };
      await runActiveSearch(
        searchMode,
        nextFilters,
        false,
        currentSort ?? undefined,
        nextDirection,
      );
    },
  };

  // Per-mode results rendering; every other per-mode value comes from the
  // active search's unified shape below.
  const resultsByMode: Record<
    SearchMode,
    { resultsFallback: ReactElement; renderResults: () => ReactElement }
  > = {
    sections: {
      resultsFallback: <SearchResultsTableFallback />,
      renderResults: () => (
        <SearchResultsTable
          searchResults={searchResults}
          selectedResults={selectedResults}
          clauseTypePathByStandardId={clauseTypePathByStandardId}
          sort_by={currentSort ?? "year"}
          sort_direction={sort_direction}
          onToggleResultSelection={toggleResultSelection}
          onToggleSelectAll={toggleSelectAll}
          onOpenAgreement={openAgreement}
          getAgreementHref={getSectionAgreementHref}
          onSortResults={(field) => void trackingActions.sortResults(field)}
          onToggleSortDirection={() => void trackingActions.toggleSortDirection()}
          density={resultsDensity}
          onDensityChange={updateResultsDensity}
          currentPage={currentPage}
          page_size={page_size}
        />
      ),
    },
    tax: {
      resultsFallback: <SearchResultsTableFallback />,
      renderResults: () => (
        <TaxClauseResultsList
          results={taxSearch.results}
          getAgreementHref={(r: TaxClauseSearchResult) =>
            buildAgreementHref(r.agreement_uuid, r.section_uuid)
          }
          clauseTypeLabelById={clauseTypeLabelById}
          selectedResults={taxSearch.selectedResults}
          onToggleResultSelection={taxSearch.actions.toggleResultSelection}
          onToggleSelectAll={taxSearch.actions.toggleSelectAll}
          sortBy={currentSort ?? "year"}
          sortDirection={sort_direction}
          onSortResults={(field) => void trackingActions.sortResults(field)}
          onToggleSortDirection={() => void trackingActions.toggleSortDirection()}
          density={resultsDensity}
          onDensityChange={updateResultsDensity}
        />
      ),
    },
    transactions: {
      resultsFallback: <TransactionResultsFallback />,
      renderResults: () => (
        <TransactionResultsList
          results={transactionSearch.results}
          getAgreementHref={getTransactionAgreementHref}
          showClauseContext={dealClauseContextActive}
          clauseTypeLabelById={clauseTypeLabelById}
          currentPage={currentPage}
          pageSize={page_size}
          selectedResults={transactionSearch.selectedResults}
          onToggleResultSelection={transactionSearch.actions.toggleResultSelection}
          onToggleSelectAll={transactionSearch.actions.toggleSelectAll}
          sortBy={currentSort ?? "year"}
          sortDirection={sort_direction}
          onSortResults={(field) => void trackingActions.sortResults(field)}
          onToggleSortDirection={() => void trackingActions.toggleSortDirection()}
          density={resultsDensity}
          onDensityChange={updateResultsDensity}
        />
      ),
    },
  };
  const activeSearch = searches[searchMode];
  const adapter = {
    isSearching: activeSearch.isSearching,
    hasSearched: activeSearch.hasSearched,
    totalCount: activeSearch.totalCount,
    totalCountIsApproximate: activeSearch.totalCountIsApproximate,
    totalPages: activeSearch.totalPages,
    hasNext: activeSearch.hasNext,
    hasPrev: activeSearch.hasPrev,
    accessTier: activeSearch.access?.tier ?? "anonymous",
    selectedSize: activeSearch.selectedResults.size,
    resultsLength: activeSearch.results.length,
    sidebarClauseTypeLabel: MODE_UI[searchMode].sidebarClauseTypeLabel,
    isLoadingSidebarTaxonomy:
      searchMode === "tax" ? isLoadingTaxTaxonomy : isLoadingFilterOptions,
    clearSelection: activeSearch.actions.clearSelection,
    downloadCSV: () => {
      trackEvent(MODE_UI[searchMode].exportEvent, {
        export_format: "csv",
        result_count:
          activeSearch.selectedResults.size > 0
            ? activeSearch.selectedResults.size
            : activeSearch.totalCount,
        is_filtered: Object.values(filters).flat().length > 0,
      });
      void activeSearch.actions.downloadCSV();
    },
    ...resultsByMode[searchMode],
  };

  // Latest search action for the global Enter handler, so the listener below
  // can be attached once instead of re-subscribing every render.
  const enterSearchRef = useRef<() => void>(() => {});
  useEffect(() => {
    enterSearchRef.current = () => {
      if (adapter.isSearching) return;
      void runActiveSearch(searchMode, filters);
    };
  });

  // Allow Enter to trigger search when focus isn't inside an input/control.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Enter") return;

      const activeElement = document.activeElement as HTMLElement | null;
      const activeTag = activeElement?.tagName;
      const isEditable =
        activeElement?.isContentEditable ||
        activeTag === "INPUT" ||
        activeTag === "TEXTAREA" ||
        activeTag === "SELECT" ||
        activeTag === "BUTTON" ||
        activeTag === "A";

      // Radix stamps open overlay content and triggers (dialogs, sheets,
      // popovers, dropdowns, selects) with data-state="open"; while any
      // overlay is open, Enter belongs to it.
      const hasOpenOverlay = document.querySelector('[data-state="open"]') !== null;

      if (!isEditable && !hasOpenOverlay) {
        enterSearchRef.current();
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const sidebarCommonProps = {
    filters,
    years,
    target_counsels,
    acquirer_counsels,
    target_industries,
    acquirer_industries,
    clauseTypesNested,
    clauseTypeLabelById,
    clauseTypeSectionLabel: adapter.sidebarClauseTypeLabel,
    isLoadingFilterOptions,
    isLoadingTaxonomy: adapter.isLoadingSidebarTaxonomy,
    onToggleFilterValue: toggleFilterValue,
    onTextFilterChange: trackingActions.setTextFilterValue,
    onClearFilters: trackingActions.clearFilters,
    loadTargetOptions: (query: string) => loadSearchFilterOptions("target", query),
    loadAcquirerOptions: (query: string) => loadSearchFilterOptions("acquirer", query),
  };

  const mobileSidebar = hasHydrated ? (
    <Suspense fallback={<SearchSidebarFallback variant="sheet" />}>
      <SearchSidebar variant="sheet" {...sidebarCommonProps} />
    </Suspense>
  ) : (
    <SearchSidebarFallback variant="sheet" />
  );

  const onTaxCompare = () => {
    const clauses = taxSearch.results.filter((r) =>
      taxSearch.selectedResults.has(r.id),
    );
    stashCompareClauses(clauses);
    trackEvent("tax_clauses_compare_click", {
      selected_count: clauses.length,
    });
    navigate("/compare/tax");
  };

  const resultsList = hasHydrated ? (
    <Suspense fallback={adapter.resultsFallback}>{adapter.renderResults()}</Suspense>
  ) : (
    adapter.resultsFallback
  );

  return (
    // overflow-x-clip, not -hidden: "hidden" makes this a scroll container,
    // which stops the filter sidebar from sticking to the viewport.
    <div className="w-full overflow-x-clip">
      <div className="flex min-h-full">
        {isDesktopLayout ? (
          shouldRenderDesktopSidebar && hasHydrated ? (
            <Suspense fallback={<SearchSidebarFallback />}>
              <SearchSidebar
                {...sidebarCommonProps}
                onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
                isCollapsed={sidebarCollapsed}
              />
            </Suspense>
          ) : (
            <SearchSidebarFallback />
          )
        ) : null}

        <div className="flex min-w-0 flex-1 flex-col">
          <SearchHeader
            searchMode={searchMode}
            onModeChange={handleModeChange}
            modeButtonRefs={modeButtonRefs}
            isMobileFiltersOpen={isMobileFiltersOpen}
            onMobileFiltersOpenChange={setIsMobileFiltersOpen}
            authStatus={authStatus}
            signInPath={signInPath}
            mobileSidebar={mobileSidebar}
          />

          {filterOptionsError && (
            <div className="mx-4 mt-3 sm:mx-8">
              <Alert variant="destructive" role="alert">
                <div className="text-sm font-medium leading-none tracking-tight">
                  Filter options error
                </div>
                <AlertDescription>{filterOptionsError}</AlertDescription>
              </Alert>
            </div>
          )}

          <SearchActionsBar
            searchMode={searchMode}
            isSearching={adapter.isSearching}
            selectedSize={adapter.selectedSize}
            resultsLength={adapter.resultsLength}
            onSearch={() => void trackingActions.performSearch()}
            onDownloadCSV={adapter.downloadCSV}
            onClearFilters={trackingActions.clearFilters}
            taxIncludeRepWarranty={!!filters.include_rep_warranty}
            onTaxIncludeRepWarrantyChange={setIncludeRepWarranty}
            taxSelectedCount={taxSearch.selectedResults.size}
            onTaxCompare={onTaxCompare}
            filters={filters}
            clauseTypeLabelById={clauseTypeLabelById}
            onToggleFilterValue={trackingActions.toggleFilterValue}
            onTextFilterChange={trackingActions.setTextFilterValue}
          />

          <SearchResultsPanel
            searchMode={searchMode}
            hasHydrated={hasHydrated}
            isSearching={adapter.isSearching}
            hasSearched={adapter.hasSearched}
            totalCount={adapter.totalCount}
            totalCountIsApproximate={adapter.totalCountIsApproximate}
            totalPages={adapter.totalPages}
            hasNext={adapter.hasNext}
            hasPrev={adapter.hasPrev}
            currentPage={currentPage}
            pageSize={page_size}
            accessTier={adapter.accessTier}
            selectedSize={adapter.selectedSize}
            onGoToPage={(page) => void trackingActions.goToPage(page)}
            onPageSizeChange={(size) => void trackingActions.changePageSize(size)}
            onDownloadCSV={adapter.downloadCSV}
            onClearSelection={adapter.clearSelection}
            resultsList={resultsList}
          />
        </div>
      </div>

      {activeShowErrorModal && (
        <ErrorModal
          isOpen={activeShowErrorModal}
          onClose={() => {
            actions.closeErrorModal();
            transactionSearch.actions.closeErrorModal();
            taxSearch.actions.closeErrorModal();
          }}
          message={activeErrorMessage}
        />
      )}
    </div>
  );
}
