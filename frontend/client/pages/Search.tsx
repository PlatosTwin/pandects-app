import React, { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation, useNavigate, useSearchParams } from "react-router-dom";
import { AVAILABLE_YEARS, BREAKPOINT_LG } from "@/lib/constants";
import { formatFilterOption } from "@/lib/text-utils";
import { cn } from "@/lib/utils";
import type { SearchFilters } from "@shared/sections";
import {
  Search as SearchIcon,
  Download,
  Layers,
  FileText,
  SlidersHorizontal,
  Sparkles,
  X,
  Building2,
} from "lucide-react";
import { useSections } from "@/hooks/use-sections";
import { useTransactionSearch } from "@/hooks/use-transaction-search";
import { useTaxClauses } from "@/hooks/use-tax-clauses";
import { useTaxClauseTaxonomy } from "@/hooks/use-tax-clause-taxonomy";
import { useFilterOptions } from "@/hooks/use-filter-options";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import ErrorModal from "@/components/ErrorModal";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useAuth } from "@/hooks/use-auth";
import type { ClauseTypeTree } from "@/lib/clause-types";
import { indexClauseTypeLabels, indexClauseTypePaths } from "@/lib/clause-type-index";
import { scheduleWhenBrowserIdle, trackEvent } from "@/lib/analytics";
import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";
import { buildAccountPathWithNext } from "@/lib/auth-next";
import { buildSearchStateParams, parseSearchFilters } from "@/lib/url-params";
import {
  stashCompareClauses,
  TAX_COMPARE_MAX,
  TAX_COMPARE_MIN,
} from "@/lib/tax-compare-handoff";
import { parseSearchMode, type SearchMode } from "@shared/search";
import type { TransactionSearchResult } from "@shared/transactions";
import type { TaxClauseSearchResult } from "@shared/tax-clauses";

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

export default function Search() {
  const { status: authStatus } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [hasHydrated, setHasHydrated] = useState(false);
  const [searchMode, setSearchMode] = useState<SearchMode>(() =>
    parseSearchMode(searchParams.get("mode")),
  );
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
  const [resultsDensity, setResultsDensity] = useState<"comfy" | "compact">(
    () => {
      try {
        const stored = localStorage.getItem("pandects.resultsDensity");
        return stored === "compact" ? "compact" : "comfy";
      } catch {
        return "comfy";
      }
    }
  );

  const updateResultsDensity = (density: "comfy" | "compact") => {
    setResultsDensity(density);
    try {
      localStorage.setItem("pandects.resultsDensity", density);
    } catch {
      // ignore
    }
  };

  const openAgreement = (
    result: (typeof searchResults)[number],
    position: number,
  ) => {
    trackEvent("sections_result_click", {
      position,
      year: result.year,
      verified: result.verified,
    });
    navigate(getSectionAgreementHref(result));
  };

  const handleModeChange = async (value: string) => {
    const nextMode: SearchMode = parseSearchMode(value);
    if (nextMode === searchMode) return;
    setSearchMode(nextMode);
    clearSelection();
    transactionSearch.clearSelection();
    taxSearch.actions.clearSelection();
  };

  // Auto-collapse sidebar on tablet and mobile
  useEffect(() => {
    const handleResize = () => {
      const isDesktop = window.innerWidth >= BREAKPOINT_LG;
      setIsDesktopLayout(isDesktop);
      setSidebarCollapsed(!isDesktop);
    };

    // Set initial state
    handleResize();

    // Listen for window resize
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
  const {
    taxonomy: taxClauseTypesNested,
    isLoading: isLoadingTaxTaxonomy,
  } = useTaxClauseTaxonomy({ enabled: searchMode === "tax" && shouldLoadFilterData });
  const sectionClauseTypesNested: ClauseTypeTree = clause_types;
  const clauseTypesNested: ClauseTypeTree =
    searchMode === "tax" ? taxClauseTypesNested : sectionClauseTypesNested;

  const clauseTypePathByStandardId = useMemo(
    () => indexClauseTypePaths(clauseTypesNested),
    [clauseTypesNested]
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
      await performSectionSearch(
        false,
        sectionClauseTypesNested,
        markAsSearched,
        nextFilters,
      );
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

  const buildAgreementHref = (
    agreementUuid: string,
    focusSectionUuid?: string | null,
  ) => {
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
      (
        nextFilters.page !== undefined ||
        nextFilters.agreement_uuid !== undefined ||
        nextFilters.section_uuid !== undefined ||
        Object.values(nextFilters).some((value) =>
          Array.isArray(value) ? value.length > 0 : false,
        )
      );

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

  const loadSearchFilterOptions = async (
    field: "target" | "acquirer",
    query: string,
  ) => {
    const params = new URLSearchParams();
    if (query.trim()) {
      params.set("query", query.trim());
    }
    params.set("limit", "100");
    const response = await authFetch(
      apiUrl(`v1/filter-options/${field}?${params.toString()}`),
    );
    if (!response.ok) {
      throw new Error(`Unable to load ${field} options.`);
    }
    const payload = (await response.json()) as { options?: unknown };
    return Array.isArray(payload.options)
      ? payload.options.filter((value): value is string => typeof value === "string")
      : [];
  };

  // Wrap actions with tracking
  const trackingActions = {
    toggleFilterValue: (field: string, value: string) => {
      trackEvent("sections_filter_change", {
        filter_field: field,
        filter_value: value.substring(0, 50), // truncate long values
        current_filters: Object.keys(filters).length,
      });
      toggleFilterValue(field as keyof SearchFilters, value);
    },
    setTextFilterValue: (field: string, value: string) => {
      trackEvent("sections_filter_change", {
        filter_field: field,
        filter_value: value.substring(0, 50), // truncate long values
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
  }, [
    activeIsSearching,
    filters,
    runActiveSearch,
    searchMode,
  ]);

  return (
    <div className="w-full overflow-x-hidden">
      <div className="flex min-h-full">
        {isDesktopLayout ? (
          shouldRenderDesktopSidebar && hasHydrated ? (
            <Suspense fallback={<SearchSidebarFallback />}>
              <SearchSidebar
                filters={filters}
                years={years}
                targets={targets}
                acquirers={acquirers}
                target_counsels={target_counsels}
                acquirer_counsels={acquirer_counsels}
                target_industries={target_industries}
                acquirer_industries={acquirer_industries}
                clauseTypesNested={clauseTypesNested}
                clauseTypeLabelById={clauseTypeLabelById}
                clauseTypeSectionLabel={searchMode === "tax" ? "Tax clause type" : "Section Type"}
                isLoadingFilterOptions={isLoadingFilterOptions}
                isLoadingTaxonomy={searchMode === "tax" ? isLoadingTaxTaxonomy : isLoadingFilterOptions}
                onToggleFilterValue={toggleFilterValue}
                onTextFilterChange={trackingActions.setTextFilterValue}
                onClearFilters={trackingActions.clearFilters}
                loadTargetOptions={(query) => loadSearchFilterOptions("target", query)}
                loadAcquirerOptions={(query) => loadSearchFilterOptions("acquirer", query)}
                onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
                isCollapsed={sidebarCollapsed}
              />
            </Suspense>
          ) : (
            <SearchSidebarFallback />
          )
        ) : null}

        <div className="flex min-w-0 flex-1 flex-col">
          {/* Row 1: title + tabs + mobile filters */}
          <div className="border-b border-border px-4 py-3 sm:px-8">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
              <div className="flex min-w-0 flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center">
                <h1
                  id="search-page-title"
                  className="shrink-0 text-xl font-semibold tracking-tight text-foreground"
                >
                  M&A Search
                </h1>
                <div
                  role="radiogroup"
                  aria-label="Search mode"
                  className="grid min-h-10 w-full grid-cols-3 items-center rounded-lg border border-border bg-muted/40 p-1 sm:w-auto sm:rounded-full"
                  onKeyDown={(e) => {
                    if (
                      ![
                        "ArrowLeft",
                        "ArrowRight",
                        "ArrowUp",
                        "ArrowDown",
                        "Home",
                        "End",
                      ].includes(e.key)
                    ) {
                      return;
                    }
                    e.preventDefault();
                    const order: SearchMode[] = ["sections", "transactions", "tax"];
                    const idx = order.indexOf(searchMode);
                    const nextIdx =
                      e.key === "Home"
                        ? 0
                        : e.key === "End"
                          ? order.length - 1
                          : (idx +
                              (e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 1) +
                              order.length) %
                            order.length;
                    void handleModeChange(order[nextIdx]);
                  }}
                >
                  {(["sections", "transactions", "tax"] as const).map((mode) => (
                    <button
                      key={mode}
                      type="button"
                      role="radio"
                      aria-checked={searchMode === mode}
                      tabIndex={searchMode === mode ? 0 : -1}
                      onClick={() => void handleModeChange(mode)}
                      className={cn(
                        "min-h-8 rounded-md px-3 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:rounded-full",
                        searchMode === mode
                          ? "bg-primary/10 text-primary shadow-sm"
                          : "text-muted-foreground hover:text-foreground",
                      )}
                    >
                      {mode === "sections" ? "Sections" : mode === "transactions" ? "Deals" : "Tax"}
                    </button>
                  ))}
                </div>
              </div>

              <div className="shrink-0 lg:hidden">
                <Sheet
                  open={isMobileFiltersOpen}
                  onOpenChange={setIsMobileFiltersOpen}
                >
                  <SheetTrigger asChild>
                    <Button variant="outline" size="sm" className="h-11 w-full gap-2 sm:w-auto">
                      <SlidersHorizontal className="h-4 w-4" aria-hidden="true" />
                      Filters
                    </Button>
                  </SheetTrigger>
                  <SheetContent
                    side="left"
                    className="w-[min(340px,100vw)] max-w-full p-0"
                  >
                    <SheetTitle className="sr-only">Search filters</SheetTitle>
                    <SheetDescription className="sr-only">
                      {searchMode === "sections"
                        ? "Filter agreement section results."
                        : searchMode === "tax"
                          ? "Filter tax clause results."
                          : "Filter deal results."}
                    </SheetDescription>
                    {hasHydrated ? (
                      <Suspense fallback={<SearchSidebarFallback variant="sheet" />}>
                        <SearchSidebar
                          variant="sheet"
                          filters={filters}
                          years={years}
                          targets={targets}
                          acquirers={acquirers}
                          target_counsels={target_counsels}
                          acquirer_counsels={acquirer_counsels}
                          target_industries={target_industries}
                          acquirer_industries={acquirer_industries}
                          clauseTypesNested={clauseTypesNested}
                          clauseTypeLabelById={clauseTypeLabelById}
                          clauseTypeSectionLabel={searchMode === "tax" ? "Tax clause type" : "Section Type"}
                          isLoadingFilterOptions={isLoadingFilterOptions}
                          isLoadingTaxonomy={searchMode === "tax" ? isLoadingTaxTaxonomy : isLoadingFilterOptions}
                          onToggleFilterValue={toggleFilterValue}
                          onTextFilterChange={trackingActions.setTextFilterValue}
                          onClearFilters={trackingActions.clearFilters}
                          loadTargetOptions={(query) => loadSearchFilterOptions("target", query)}
                          loadAcquirerOptions={(query) => loadSearchFilterOptions("acquirer", query)}
                        />
                      </Suspense>
                    ) : (
                      <SearchSidebarFallback variant="sheet" />
                    )}
                  </SheetContent>
                </Sheet>
              </div>
            </div>

            {authStatus === "anonymous" && (
              <div className="mt-3">
                <Alert className="py-3">
                  <Sparkles className="h-4 w-4" aria-hidden="true" />
                  <div className="text-sm font-medium leading-none tracking-tight">
                    Limited mode
                  </div>
                  <AlertDescription>
                    <div className="grid gap-2">
                      <p>
                        Sign in to view section text, open full agreements, unlock
                        higher page sizes, and use the MCP server.
                      </p>
                      <div>
                        <Button asChild size="sm" variant="outline">
                          <Link to={signInPath}>Sign in to unlock access</Link>
                        </Button>
                      </div>
                    </div>
                  </AlertDescription>
                </Alert>
              </div>
            )}
          </div>

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

          {/* Row 2: actions + active filter chips */}
          <div className="border-b border-border bg-muted/20 px-4 py-2.5 backdrop-blur supports-[backdrop-filter]:bg-muted/20 sm:px-8">
            <div className="flex flex-wrap items-center gap-2" role="toolbar" aria-label="Search actions">
              <Button
                onClick={() => void trackingActions.performSearch()}
                disabled={activeIsSearching}
                className="h-11 flex-1 gap-2 sm:h-9 sm:flex-none sm:w-auto"
                variant="default"
                size="sm"
                aria-describedby="search-results-status"
              >
                <SearchIcon
                  className={cn("h-4 w-4", activeIsSearching && "animate-spin-custom")}
                  aria-hidden="true"
                />
                <span>{activeIsSearching ? "Searching..." : "Search"}</span>
              </Button>

              {(() => {
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
                const downloadDisabled =
                  activeResultsLength === 0 && activeSelectedSize === 0;
                return (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <span className="hidden sm:inline-block">
                        <Button
                          onClick={() => trackingActions.downloadCSV()}
                          disabled={downloadDisabled}
                          variant="outline"
                          size="sm"
                          className="h-11 w-full gap-2 text-muted-foreground hover:text-foreground sm:h-9 sm:w-auto"
                          aria-label={
                            downloadDisabled
                              ? "Download CSV (disabled: no results to download. Run a search first.)"
                              : "Download CSV"
                          }
                        >
                          <Download className="h-4 w-4" aria-hidden="true" />
                          <span className="sm:inline">
                            Download CSV
                            {activeSelectedSize > 0 && ` (${activeSelectedSize})`}
                          </span>
                        </Button>
                      </span>
                    </TooltipTrigger>
                    {downloadDisabled && (
                      <TooltipContent>
                        <p>No results to download. Run a search first.</p>
                      </TooltipContent>
                    )}
                  </Tooltip>
                );
              })()}

              <Button
                onClick={trackingActions.clearFilters}
                variant="outline"
                size="sm"
                className="h-11 text-muted-foreground hover:text-foreground sm:h-9"
              >
                Reset filters
              </Button>

              {searchMode === "tax" && (
                <label className="flex min-h-11 items-center gap-2 rounded-md border border-border bg-background px-2 py-1 text-xs text-muted-foreground sm:min-h-0">
                  <input
                    type="checkbox"
                    checked={!!taxSearch.filters.include_rep_warranty}
                    onChange={(e) =>
                      taxSearch.actions.setIncludeRepWarranty(e.target.checked)
                    }
                    className="h-3.5 w-3.5"
                  />
                  Include reps &amp; warranties clauses
                </label>
              )}

              {searchMode === "tax" && (() => {
                const selected = taxSearch.selectedResults.size;
                const disabled = selected < TAX_COMPARE_MIN || selected > TAX_COMPARE_MAX;
                const label =
                  selected === 0
                    ? `Compare (select ${TAX_COMPARE_MIN}–${TAX_COMPARE_MAX})`
                    : selected < TAX_COMPARE_MIN
                      ? `Compare (select ${TAX_COMPARE_MIN - selected} more)`
                      : selected > TAX_COMPARE_MAX
                        ? `Compare (max ${TAX_COMPARE_MAX})`
                        : `Compare (${selected})`;
                return (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <span>
                        <Button
                          onClick={() => {
                            const clauses = taxSearch.searchResults.filter((r) =>
                              taxSearch.selectedResults.has(r.id),
                            );
                            stashCompareClauses(clauses);
                            trackEvent("tax_clauses_compare_click", {
                              selected_count: clauses.length,
                            });
                            navigate("/compare/tax");
                          }}
                          disabled={disabled}
                          variant="outline"
                          size="sm"
                          className="h-11 sm:h-9"
                        >
                          {label}
                        </Button>
                      </span>
                    </TooltipTrigger>
                    {disabled && (
                      <TooltipContent>
                        <p>
                          Select {TAX_COMPARE_MIN}–{TAX_COMPARE_MAX} tax clauses to compare.
                        </p>
                      </TooltipContent>
                    )}
                  </Tooltip>
                );
              })()}

              {/* Active filter chips inline */}
              {(filters.year.length > 0 ||
                filters.target.length > 0 ||
                filters.acquirer.length > 0 ||
                filters.clauseType.length > 0 ||
                filters.transaction_price_total.length > 0 ||
                filters.transaction_price_stock.length > 0 ||
                filters.transaction_price_cash.length > 0 ||
                filters.transaction_price_assets.length > 0 ||
                filters.transaction_consideration.length > 0 ||
                filters.target_type.length > 0 ||
                filters.acquirer_type.length > 0 ||
                filters.target_counsel.length > 0 ||
                filters.acquirer_counsel.length > 0 ||
                filters.target_industry.length > 0 ||
                filters.acquirer_industry.length > 0 ||
                filters.deal_status.length > 0 ||
                filters.attitude.length > 0 ||
                filters.deal_type.length > 0 ||
                filters.purpose.length > 0 ||
                filters.target_pe.length > 0 ||
                filters.acquirer_pe.length > 0 ||
                filters.agreement_uuid ||
                filters.section_uuid) && (
                <>
                  <div className="hidden h-5 w-0.5 shrink-0 rounded-full bg-border/80 sm:block" aria-hidden="true" />
                  {(
                    [
                      ["year", "Year", filters.year],
                      ["target", "Target", filters.target],
                      ["acquirer", "Acquirer", filters.acquirer],
                      ["clauseType", searchMode === "tax" ? "Tax clause type" : "Section type", filters.clauseType],
                      ["transaction_price_total", "Price (total)", filters.transaction_price_total],
                      ["transaction_price_stock", "Price (stock)", filters.transaction_price_stock],
                      ["transaction_price_cash", "Price (cash)", filters.transaction_price_cash],
                      ["transaction_price_assets", "Price (assets)", filters.transaction_price_assets],
                      ["transaction_consideration", "Consideration", filters.transaction_consideration],
                      ["target_type", "Target type", filters.target_type],
                      ["acquirer_type", "Acquirer type", filters.acquirer_type],
                      ["target_counsel", "Target counsel", filters.target_counsel],
                      ["acquirer_counsel", "Acquirer counsel", filters.acquirer_counsel],
                      ["target_industry", "Target industry", filters.target_industry],
                      ["acquirer_industry", "Acquirer industry", filters.acquirer_industry],
                      ["deal_status", "Status", filters.deal_status],
                      ["attitude", "Attitude", filters.attitude],
                      ["deal_type", "Deal type", filters.deal_type],
                      ["purpose", "Purpose", filters.purpose],
                      ["target_pe", "Target PE", filters.target_pe],
                      ["acquirer_pe", "Acquirer PE", filters.acquirer_pe],
                    ] as const
                  ).flatMap(([field, label, values]) =>
                    values.map((value) => {
                      const isHardcodedEnum = [
                        "transaction_price_total",
                        "transaction_price_stock",
                        "transaction_price_cash",
                        "transaction_price_assets",
                        "transaction_consideration",
                        "target_type",
                        "acquirer_type",
                        "deal_status",
                        "attitude",
                        "deal_type",
                        "purpose",
                        "target_pe",
                        "acquirer_pe",
                      ].includes(field);
                      const displayValue =
                        field === "clauseType"
                          ? clauseTypeLabelById[value] ?? value
                          : isHardcodedEnum
                            ? formatFilterOption(value)
                            : value;
                      return (
                        <Badge
                          key={`${field}:${value}`}
                          variant="outline"
                          className="flex max-w-full items-center gap-1 rounded-md bg-background px-2 py-1"
                        >
                          <span className="text-muted-foreground">{label}:</span>
                          <span className="min-w-0 truncate">{displayValue}</span>
                          <button
                            type="button"
                            onClick={() => trackingActions.toggleFilterValue(field, value)}
                            className="ml-1 inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-sm text-muted-foreground hover:bg-accent/60 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:h-5 sm:w-5 sm:min-h-0 sm:min-w-0"
                            aria-label={`Remove ${label} filter: ${displayValue}`}
                          >
                            <X className="h-3 w-3" aria-hidden="true" />
                          </button>
                        </Badge>
                      );
                    })
                  )}
                  {filters.agreement_uuid && (
                    <Badge variant="outline" className="flex max-w-full items-center gap-1 rounded-md bg-background px-2 py-1">
                      <span className="text-muted-foreground">Agreement UUID:</span>
                      <span className="min-w-0 truncate">{filters.agreement_uuid}</span>
                      <button
                        type="button"
                        onClick={() => trackingActions.setTextFilterValue("agreement_uuid", "")}
                        className="ml-1 inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-sm text-muted-foreground hover:bg-accent/60 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring sm:h-5 sm:w-5 sm:min-h-0 sm:min-w-0"
                        aria-label={`Remove Agreement UUID filter: ${filters.agreement_uuid}`}
                      >
                        <X className="h-3 w-3" aria-hidden="true" />
                      </button>
                    </Badge>
                  )}
                  {filters.section_uuid && (
                    <Badge variant="outline" className="flex max-w-full items-center gap-1 rounded-md bg-background px-2 py-1">
                      <span className="text-muted-foreground">Section UUID:</span>
                      <span className="min-w-0 truncate">{filters.section_uuid}</span>
                      <button
                        type="button"
                        onClick={() => trackingActions.setTextFilterValue("section_uuid", "")}
                        className="ml-1 inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-sm text-muted-foreground hover:bg-accent/60 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring sm:h-5 sm:w-5 sm:min-h-0 sm:min-w-0"
                        aria-label={`Remove Section UUID filter: ${filters.section_uuid}`}
                      >
                        <X className="h-3 w-3" aria-hidden="true" />
                      </button>
                    </Badge>
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={trackingActions.clearFilters}
                    className="h-7 px-2 text-muted-foreground hover:text-foreground"
                  >
                    Clear all
                  </Button>
                </>
              )}
            </div>
          </div>

          <main className="flex-1 overflow-auto" aria-labelledby="search-page-title">
            <div className="px-4 py-4 sm:px-8 sm:py-5">
              <div id="search-results-status" className="sr-only" role="status" aria-live="polite">
                {activeIsSearching
                  ? "Searching."
                  : activeHasSearched
                    ? `${activeTotalCount} ${searchMode === "sections" ? "sections" : searchMode === "tax" ? "tax clauses" : "deals"} found.`
                    : "No search has been run."}
              </div>
              {!activeHasSearched && (
                <div className="mx-auto max-w-3xl space-y-4">
                  <div className="rounded-2xl border border-border bg-card p-6 shadow-sm">
                    <div className="flex items-start gap-4">
                      <div className="mt-0.5 rounded-lg bg-primary/10 p-2 text-primary">
                        <Sparkles className="h-5 w-5" aria-hidden="true" />
                      </div>
                      <div className="min-w-0">
                        <h2 className="text-base font-semibold text-foreground">
                          {searchMode === "sections"
                            ? "Find specific sections across the corpus"
                            : searchMode === "tax"
                              ? "Find tax clause precedents across the corpus"
                              : "Find deals matching your criteria"}
                        </h2>
                        <p className="mt-1 text-sm text-muted-foreground">
                          {searchMode === "sections"
                            ? "Pick filters to narrow the corpus, then search to load matched sections and jump straight into the relevant agreement passage."
                            : searchMode === "tax"
                              ? "Filter by tax clause type and deal metadata to surface precedent drafting language. Reps & warranties clauses are excluded by default."
                              : "Pick filters to narrow the corpus, then search to load matching deals with the sections that triggered each result."}
                        </p>
                      </div>
                    </div>
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="rounded-xl border border-border bg-card/60 p-4">
                      <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                        {searchMode === "sections" ? (
                          <FileText className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
                        ) : (
                          <Building2 className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
                        )}
                        Try filtering by
                      </div>
                      <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
                        <li>• Clause type (e.g. material adverse effect)</li>
                        <li>• Year, deal status, or attitude</li>
                        <li>• Target or acquirer counsel</li>
                      </ul>
                    </div>
                    <div className="rounded-xl border border-border bg-card/60 p-4">
                      <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                        <Layers className="h-4 w-4 text-muted-foreground" aria-hidden="true" />
                        {searchMode === "sections" ? "What you'll see" : "What you'll see"}
                      </div>
                      <p className="mt-2 text-sm text-muted-foreground">
                        {searchMode === "sections"
                          ? "Section text with the matched clause type, plus quick links to the full agreement."
                          : "Each unique deal grouped together, with the matched sections that brought it into the result set."}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {activeHasSearched && (
                <div className="space-y-4">
                  {activeTotalCount === 0 ? (
                    <div
                      className="mx-auto max-w-3xl text-center py-12 text-muted-foreground"
                      role="status"
                      aria-live="polite"
                    >
                      <FileText
                        className="h-12 w-12 mx-auto mb-4 opacity-50"
                        aria-hidden="true"
                      />
                      <p className="text-foreground font-medium">
                        {searchMode === "sections"
                          ? "No sections found."
                          : searchMode === "tax"
                            ? "No tax clauses found."
                            : "No deals found."}
                      </p>
                      <p className="text-sm mt-2">
                        Try adjusting your filters and search again.
                      </p>
                    </div>
                  ) : (
                    <>
                      {hasHydrated ? (
                        <>
                          <Suspense fallback={<SearchPaginationFallback />}>
                            <SearchPagination
                              currentPage={currentPage}
                              totalPages={activeTotalPages}
                              pageSize={page_size}
                              totalCount={activeTotalCount}
                              totalCountIsApproximate={activeTotalCountIsApproximate}
                              hasNext={activeHasNext}
                              hasPrev={activeHasPrev}
                              onPageChange={(page) =>
                                void trackingActions.goToPage(page)
                              }
                              onPageSizeChange={(nextPageSize) =>
                                void trackingActions.changePageSize(nextPageSize)
                              }
                              isLoading={activeIsSearching}
                              isLimited={(activeAccess?.tier ?? "anonymous") === "anonymous"}
                            />
                          </Suspense>

                          {searchMode === "tax" ? (
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
                          )}
                        </>
                      ) : (
                        <>
                          <SearchPaginationFallback />
                          {searchMode === "transactions" ? (
                            <TransactionResultsFallback />
                          ) : (
                            <SearchResultsTableFallback />
                          )}
                        </>
                      )}

                      {(() => {
                        const activeSelectedSize =
                          searchMode === "sections"
                            ? selectedResults.size
                            : searchMode === "tax"
                              ? taxSearch.selectedResults.size
                              : transactionSearch.selectedResults.size;
                        if (activeSelectedSize === 0) return null;
                        const onClear =
                          searchMode === "sections"
                            ? clearSelection
                            : searchMode === "tax"
                              ? taxSearch.actions.clearSelection
                              : transactionSearch.clearSelection;
                        return (
                          <div className="rounded-xl border border-border bg-muted/20 p-4 backdrop-blur supports-[backdrop-filter]:bg-muted/20">
                            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                              <div className="text-sm text-muted-foreground">
                                {activeSelectedSize} selected
                              </div>
                              <div className="flex flex-wrap gap-2">
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => trackingActions.downloadCSV()}
                                >
                                  Download selected
                                </Button>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={onClear}
                                >
                                  Clear selection
                                </Button>
                              </div>
                            </div>
                          </div>
                        );
                      })()}

                      {hasHydrated ? (
                        <Suspense fallback={<SearchPaginationFallback />}>
                          <SearchPagination
                            currentPage={currentPage}
                            totalPages={activeTotalPages}
                            pageSize={page_size}
                            totalCount={activeTotalCount}
                            totalCountIsApproximate={activeTotalCountIsApproximate}
                            hasNext={activeHasNext}
                            hasPrev={activeHasPrev}
                            onPageChange={(page) =>
                              void trackingActions.goToPage(page)
                            }
                            onPageSizeChange={(nextPageSize) =>
                              void trackingActions.changePageSize(nextPageSize)
                            }
                            isLoading={activeIsSearching}
                            isLimited={(activeAccess?.tier ?? "anonymous") === "anonymous"}
                          />
                        </Suspense>
                      ) : (
                        <SearchPaginationFallback />
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          </main>
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
