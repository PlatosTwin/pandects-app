import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { AVAILABLE_YEARS, BREAKPOINT_LG } from "@/lib/constants";
import type { SearchFilters } from "@shared/sections";
import { useSections } from "@/hooks/use-sections";
import { useTransactionSearch } from "@/hooks/use-transaction-search";
import { useTaxClauses } from "@/hooks/use-tax-clauses";
import { useTaxClauseTaxonomy } from "@/hooks/use-tax-clause-taxonomy";
import { useFilterOptions } from "@/hooks/use-filter-options";
import { Alert, AlertDescription } from "@/components/ui/alert";
import ErrorModal from "@/components/ErrorModal";
import { useAuth } from "@/hooks/use-auth";
import type { ClauseTypeTree } from "@/lib/clause-types";
import { indexClauseTypeLabels, indexClauseTypePaths } from "@/lib/clause-type-index";
import { scheduleWhenBrowserIdle, trackEvent } from "@/lib/analytics";
import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";
import { buildAccountPathWithNext } from "@/lib/auth-next";
import { buildSearchStateParams, parseSearchFilters } from "@/lib/url-params";
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
  const {
    filters,
    isSearching: isSearchingSections,
    searchResults,
    selectedResults,
    hasSearched: hasSearchedSections,
    total_count: totalCountSections,
    totalCountIsApproximate: totalCountIsApproximateSections,
    total_pages: totalPagesSections,
    has_next: hasNextSections,
    has_prev: hasPrevSections,
    currentSort,
    currentPage,
    page_size,
    showErrorModal: showSectionsErrorModal,
    errorMessage: sectionsErrorMessage,
    sort_direction,
    access: sectionAccess,
    actions,
  } = useSections();
  const transactionSearch = useTransactionSearch();
  const taxSearch = useTaxClauses();

  const {
    toggleFilterValue,
    setTextFilterValue,
    performSearch: performSectionSearch,
    downloadCSV,
    clearFilters: clearSectionFilters,
    sortResults,
    toggleSortDirection,
    toggleResultSelection,
    toggleSelectAll,
    clearSelection,
    hydrateFilters,
    updateFilter,
    setSortDirection,
  } = actions;
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

  const openAgreement = (result: (typeof searchResults)[number], position: number) => {
    trackEvent("sections_result_click", {
      position,
      year: result.year,
      verified: result.verified,
    });
    navigate(getSectionAgreementHref(result));
  };

  const handleModeChange = (value: SearchMode) => {
    const nextMode: SearchMode = parseSearchMode(value);
    if (nextMode === searchMode) return;
    setSearchMode(nextMode);
    clearSelection();
    transactionSearch.clearSelection();
    taxSearch.actions.clearSelection();
  };

  useEffect(() => {
    const handleResize = () => {
      const isDesktop = window.innerWidth >= BREAKPOINT_LG;
      setIsDesktopLayout(isDesktop);
      setSidebarCollapsed(!isDesktop);
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
    targets,
    acquirers,
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
  const { taxonomy: taxClauseTypesNested, isLoading: isLoadingTaxTaxonomy } = useTaxClauseTaxonomy({
    enabled: searchMode === "tax" && shouldLoadFilterData,
  });
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

  const activeIsSearching =
    searchMode === "sections"
      ? isSearchingSections
      : searchMode === "tax"
        ? taxSearch.isSearching
        : transactionSearch.isSearching;
  const activeHasSearched =
    searchMode === "sections"
      ? hasSearchedSections
      : searchMode === "tax"
        ? taxSearch.hasSearched
        : transactionSearch.hasSearched;
  const activeTotalCount =
    searchMode === "sections"
      ? totalCountSections
      : searchMode === "tax"
        ? taxSearch.total_count
        : transactionSearch.totalCount;
  const activeTotalCountIsApproximate =
    searchMode === "sections"
      ? totalCountIsApproximateSections
      : searchMode === "tax"
        ? taxSearch.totalCountIsApproximate
        : transactionSearch.totalCountIsApproximate;
  const activeTotalPages =
    searchMode === "sections"
      ? totalPagesSections
      : searchMode === "tax"
        ? taxSearch.total_pages
        : transactionSearch.totalPages;
  const activeHasNext =
    searchMode === "sections"
      ? hasNextSections
      : searchMode === "tax"
        ? taxSearch.has_next
        : transactionSearch.hasNext;
  const activeHasPrev =
    searchMode === "sections"
      ? hasPrevSections
      : searchMode === "tax"
        ? taxSearch.has_prev
        : transactionSearch.hasPrev;
  const activeAccess =
    searchMode === "sections"
      ? sectionAccess
      : searchMode === "tax"
        ? taxSearch.access
        : transactionSearch.access;
  const activeSelectedSize =
    searchMode === "sections"
      ? selectedResults.size
      : searchMode === "tax"
        ? taxSearch.selectedResults.size
        : transactionSearch.selectedResults.size;
  const activeResultsLength =
    searchMode === "sections"
      ? searchResults.length
      : searchMode === "tax"
        ? taxSearch.searchResults.length
        : transactionSearch.results.length;
  const activeShowErrorModal =
    showSectionsErrorModal || transactionSearch.showErrorModal || taxSearch.showErrorModal;
  const activeErrorMessage =
    sectionsErrorMessage || transactionSearch.errorMessage || taxSearch.errorMessage;

  const runActiveSearch = async (
    nextMode: SearchMode,
    nextFilters: SearchFilters,
    markAsSearched: boolean = true,
  ) => {
    if (nextMode === "sections") {
      await performSectionSearch(false, sectionClauseTypesNested, markAsSearched, nextFilters);
      return;
    }
    if (nextMode === "tax") {
      await taxSearch.actions.performSearch(
        false,
        markAsSearched,
        {
          ...nextFilters,
          include_rep_warranty: taxSearch.filters.include_rep_warranty,
        },
        currentSort,
        sort_direction,
      );
      return;
    }
    await transactionSearch.performSearch({
      filters: nextFilters,
      clauseTypesNested: sectionClauseTypesNested,
      sortBy: currentSort,
      sortDirection: sort_direction,
      markAsSearched,
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
    if (nextSortBy === "year" || nextSortBy === "target" || nextSortBy === "acquirer") {
      sortResults(nextSortBy);
    }
    if (nextSortDirection === "asc" || nextSortDirection === "desc") {
      setSortDirection(nextSortDirection);
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
      void runActiveSearch(nextMode, nextFilters);
    }
  }, [hydrateFilters, searchParams, setSortDirection, sortResults]);

  useEffect(() => {
    if (isHydratingFromUrlRef.current) return;
    const nextParams = buildSearchStateParams({
      filters,
      mode: searchMode,
      sortBy: currentSort,
      sortDirection: sort_direction,
      clauseTypesNested,
    });
    if (nextParams.toString() !== searchParams.toString()) {
      setSearchParams(nextParams, { replace: true });
    }
  }, [
    clauseTypesNested,
    currentSort,
    filters,
    searchMode,
    searchParams,
    setSearchParams,
    sort_direction,
  ]);

  const loadSearchFilterOptions = async (field: "target" | "acquirer", query: string) => {
    const params = new URLSearchParams();
    if (query.trim()) {
      params.set("query", query.trim());
    }
    params.set("limit", "100");
    const response = await authFetch(apiUrl(`v1/filter-options/${field}?${params.toString()}`));
    if (!response.ok) {
      throw new Error(`Unable to load ${field} options.`);
    }
    const payload = (await response.json()) as { options?: unknown };
    return Array.isArray(payload.options)
      ? payload.options.filter((value): value is string => typeof value === "string")
      : [];
  };

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
    downloadCSV: () => {
      if (searchMode === "sections") {
        trackEvent("sections_export_click", {
          export_format: "csv",
          result_count: selectedResults.size > 0 ? selectedResults.size : totalCountSections,
          is_filtered: Object.values(filters).flat().length > 0,
        });
        downloadCSV();
      } else if (searchMode === "tax") {
        trackEvent("tax_clauses_export_click", {
          export_format: "csv",
          result_count:
            taxSearch.selectedResults.size > 0
              ? taxSearch.selectedResults.size
              : taxSearch.total_count,
          is_filtered: Object.values(filters).flat().length > 0,
        });
        void taxSearch.actions.downloadCSV();
      } else {
        trackEvent("transactions_export_click", {
          export_format: "csv",
          result_count:
            transactionSearch.selectedResults.size > 0
              ? transactionSearch.selectedResults.size
              : transactionSearch.totalCount,
          is_filtered: Object.values(filters).flat().length > 0,
        });
        void transactionSearch.downloadCSV(sectionClauseTypesNested, filters);
      }
    },
    clearFilters: () => {
      trackEvent("sections_filters_cleared", {
        filter_count: Object.values(filters).flat().length,
      });
      clearSectionFilters();
      transactionSearch.clear();
      transactionSearch.clearSelection();
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
      const nextField = field as "year" | "target" | "acquirer";
      sortResults(nextField);
      const nextFilters: SearchFilters = { ...filters };
      await runActiveSearch(searchMode, nextFilters, false);
    },
    toggleSortDirection: async () => {
      toggleSortDirection();
      const nextFilters: SearchFilters = { ...filters };
      await runActiveSearch(searchMode, nextFilters, false);
    },
  };

  // Allow Enter to trigger search when focus isn't inside an input/control.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Enter" || activeIsSearching) return;

      const activeElement = document.activeElement as HTMLElement | null;
      const activeTag = activeElement?.tagName;
      const isEditable =
        activeElement?.isContentEditable ||
        activeTag === "INPUT" ||
        activeTag === "TEXTAREA" ||
        activeTag === "SELECT" ||
        activeTag === "BUTTON";

      const hasOpenDropdown =
        document.querySelector(".absolute.top-full") ||
        document.querySelector('[role="dialog"]');

      const isInsideDropdown =
        activeElement?.closest('[role="combobox"]') ||
        activeElement?.closest(".absolute") ||
        activeElement?.closest('[role="dialog"]');

      if (!isEditable && !isInsideDropdown && !hasOpenDropdown) {
        void runActiveSearch(searchMode, filters);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeIsSearching, filters, runActiveSearch, searchMode]);

  const sidebarCommonProps = {
    filters,
    years,
    targets,
    acquirers,
    target_counsels,
    acquirer_counsels,
    target_industries,
    acquirer_industries,
    clauseTypesNested,
    clauseTypeLabelById,
    clauseTypeSectionLabel: searchMode === "tax" ? "Tax clause type" : "Section Type",
    isLoadingFilterOptions,
    isLoadingTaxonomy:
      searchMode === "tax" ? isLoadingTaxTaxonomy : isLoadingFilterOptions,
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
    const clauses = taxSearch.searchResults.filter((r) =>
      taxSearch.selectedResults.has(r.id),
    );
    stashCompareClauses(clauses);
    trackEvent("tax_clauses_compare_click", {
      selected_count: clauses.length,
    });
    navigate("/compare/tax");
  };

  const onClearActiveSelection =
    searchMode === "sections"
      ? clearSelection
      : searchMode === "tax"
        ? taxSearch.actions.clearSelection
        : transactionSearch.clearSelection;

  const resultsList = hasHydrated ? (
    searchMode === "tax" ? (
      <Suspense fallback={<SearchResultsTableFallback />}>
        <TaxClauseResultsList
          results={taxSearch.searchResults}
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
      </Suspense>
    ) : searchMode === "sections" ? (
      <Suspense fallback={<SearchResultsTableFallback />}>
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
      </Suspense>
    ) : (
      <Suspense fallback={<TransactionResultsFallback />}>
        <TransactionResultsList
          results={transactionSearch.results}
          getAgreementHref={getTransactionAgreementHref}
          showClauseContext={dealClauseContextActive}
          clauseTypeLabelById={clauseTypeLabelById}
          currentPage={currentPage}
          pageSize={page_size}
          selectedResults={transactionSearch.selectedResults}
          onToggleResultSelection={transactionSearch.toggleResultSelection}
          onToggleSelectAll={transactionSearch.toggleSelectAll}
          sortBy={currentSort ?? "year"}
          sortDirection={sort_direction}
          onSortResults={(field) => void trackingActions.sortResults(field)}
          onToggleSortDirection={() => void trackingActions.toggleSortDirection()}
          density={resultsDensity}
          onDensityChange={updateResultsDensity}
        />
      </Suspense>
    )
  ) : searchMode === "transactions" ? (
    <TransactionResultsFallback />
  ) : (
    <SearchResultsTableFallback />
  );

  return (
    <div className="w-full overflow-x-hidden">
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
            isSearching={activeIsSearching}
            selectedSize={activeSelectedSize}
            resultsLength={activeResultsLength}
            onSearch={() => void trackingActions.performSearch()}
            onDownloadCSV={trackingActions.downloadCSV}
            onClearFilters={trackingActions.clearFilters}
            taxIncludeRepWarranty={!!taxSearch.filters.include_rep_warranty}
            onTaxIncludeRepWarrantyChange={taxSearch.actions.setIncludeRepWarranty}
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
            isSearching={activeIsSearching}
            hasSearched={activeHasSearched}
            totalCount={activeTotalCount}
            totalCountIsApproximate={activeTotalCountIsApproximate}
            totalPages={activeTotalPages}
            hasNext={activeHasNext}
            hasPrev={activeHasPrev}
            currentPage={currentPage}
            pageSize={page_size}
            accessTier={activeAccess?.tier ?? "anonymous"}
            selectedSize={activeSelectedSize}
            onGoToPage={(page) => void trackingActions.goToPage(page)}
            onPageSizeChange={(size) => void trackingActions.changePageSize(size)}
            onDownloadCSV={trackingActions.downloadCSV}
            onClearSelection={onClearActiveSelection}
            resultsList={resultsList}
          />
        </div>
      </div>

      {activeShowErrorModal && (
        <ErrorModal
          isOpen={activeShowErrorModal}
          onClose={() => {
            actions.closeErrorModal();
            transactionSearch.closeErrorModal();
            taxSearch.actions.closeErrorModal();
          }}
          message={activeErrorMessage}
        />
      )}
    </div>
  );
}
