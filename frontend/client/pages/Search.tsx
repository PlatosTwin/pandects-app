import React, { Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
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
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { buildSearchStateParams, parseSearchFilters } from "@/lib/url-params";
import type { SearchMode } from "@shared/search";
import type { TransactionSearchResult } from "@shared/transactions";

const SearchPagination = lazy(() =>
  import("@/components/SearchPagination").then((mod) => ({
    default: mod.SearchPagination,
  })),
);
const SearchResultsTable = lazy(() =>
  import("@/components/SearchResultsTable").then((mod) => ({
    default: mod.SearchResultsTable,
  })),
);
const SearchSidebar = lazy(() =>
  import("@/components/SearchSidebar").then((mod) => ({
    default: mod.SearchSidebar,
  })),
);
const TransactionResultsList = lazy(() =>
  import("@/components/TransactionResultsList").then((mod) => ({
    default: mod.TransactionResultsList,
  })),
);

function SearchSidebarFallback({
  variant = "sidebar",
}: {
  variant?: "sidebar" | "sheet";
}) {
  const content = (
    <div className="space-y-5 p-4">
      <Skeleton className="h-5 w-28" />
      <Skeleton className="h-10 w-full" />
      <Skeleton className="h-24 w-full" />
      <Skeleton className="h-24 w-full" />
      <Skeleton className="h-24 w-full" />
    </div>
  );

  if (variant === "sheet") {
    return <div className="h-full overflow-y-auto">{content}</div>;
  }

  return (
    <div className="hidden h-screen w-80 border-r border-b border-border bg-card lg:block">
      {content}
    </div>
  );
}

function SearchPaginationFallback() {
  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
      <Skeleton className="h-9 w-56" />
      <Skeleton className="h-9 w-48" />
    </div>
  );
}

function SearchResultsTableFallback() {
  return (
    <div className="space-y-4">
      {Array.from({ length: 4 }).map((_, index) => (
        <div
          key={index}
          className="rounded-lg border border-border/60 bg-card p-4 shadow-sm"
        >
          <Skeleton className="h-5 w-48" />
          <Skeleton className="mt-3 h-4 w-full" />
          <Skeleton className="mt-2 h-4 w-5/6" />
          <Skeleton className="mt-4 h-20 w-full" />
        </div>
      ))}
    </div>
  );
}

function TransactionResultsFallback() {
  return (
    <div className="space-y-4">
      {Array.from({ length: 3 }).map((_, index) => (
        <div
          key={index}
          className="rounded-lg border border-border/60 bg-card p-5 shadow-sm"
        >
          <Skeleton className="h-6 w-72" />
          <Skeleton className="mt-3 h-4 w-full" />
          <Skeleton className="mt-2 h-4 w-5/6" />
          <Skeleton className="mt-5 h-24 w-full" />
        </div>
      ))}
    </div>
  );
}

export default function Search() {
  const { status: authStatus } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const [hasHydrated, setHasHydrated] = useState(false);
  const [searchMode, setSearchMode] = useState<SearchMode>(() => {
    const mode = searchParams.get("mode");
    return mode === "transactions" ? "transactions" : "sections";
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
    const nextMode: SearchMode =
      value === "transactions" ? "transactions" : "sections";
    if (nextMode === searchMode) return;
    setSearchMode(nextMode);
    clearSelection();
    transactionSearch.clearSelection();
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

    if (hasSearchedSections || transactionSearch.hasSearched) {
      setShouldRenderDesktopSidebar(true);
      return;
    }

    const cancelDeferredSidebar = scheduleWhenBrowserIdle(() => {
      setShouldRenderDesktopSidebar(true);
    }, 1800);

    return cancelDeferredSidebar;
  }, [hasSearchedSections, isDesktopLayout, transactionSearch.hasSearched]);

  const shouldLoadFilterData =
    (isDesktopLayout && shouldRenderDesktopSidebar) ||
    isMobileFiltersOpen ||
    hasSearchedSections ||
    transactionSearch.hasSearched;

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
  const clauseTypesNested: ClauseTypeTree = clause_types;

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
      : transactionSearch.isSearching;
  const activeHasSearched =
    searchMode === "sections"
      ? hasSearchedSections
      : transactionSearch.hasSearched;
  const activeTotalCount =
    searchMode === "sections"
      ? totalCountSections
      : transactionSearch.totalCount;
  const activeTotalCountIsApproximate =
    searchMode === "sections"
      ? totalCountIsApproximateSections
      : transactionSearch.totalCountIsApproximate;
  const activeTotalPages =
    searchMode === "sections"
      ? totalPagesSections
      : transactionSearch.totalPages;
  const activeHasNext =
    searchMode === "sections"
      ? hasNextSections
      : transactionSearch.hasNext;
  const activeHasPrev =
    searchMode === "sections"
      ? hasPrevSections
      : transactionSearch.hasPrev;
  const activeAccess =
    searchMode === "sections"
      ? sectionAccess
      : transactionSearch.access;
  const activeShowErrorModal =
    showSectionsErrorModal || transactionSearch.showErrorModal;
  const activeErrorMessage =
    sectionsErrorMessage || transactionSearch.errorMessage;

  const runActiveSearch = async (
    nextMode: SearchMode,
    nextFilters: SearchFilters,
    markAsSearched: boolean = true,
  ) => {
    if (nextMode === "sections") {
      await performSectionSearch(
        false,
        clauseTypesNested,
        markAsSearched,
        nextFilters,
      );
      return;
    }
    await transactionSearch.performSearch({
      filters: nextFilters,
      clauseTypesNested,
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

    const nextMode = searchParams.get("mode") === "transactions"
      ? "transactions"
      : "sections";
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
      } else {
        trackEvent("transactions_export_click", {
          export_format: "csv",
          result_count:
            transactionSearch.selectedResults.size > 0
              ? transactionSearch.selectedResults.size
              : transactionSearch.totalCount,
          is_filtered: Object.values(filters).flat().length > 0,
        });
        void transactionSearch.downloadCSV(clauseTypesNested, filters);
      }
    },
    clearFilters: () => {
      trackEvent("sections_filters_cleared", {
        filter_count: Object.values(filters).flat().length,
      });
      clearSectionFilters();
      transactionSearch.clear();
      transactionSearch.clearSelection();
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
    <div className="w-full">
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
                isLoadingFilterOptions={isLoadingFilterOptions}
                isLoadingTaxonomy={isLoadingFilterOptions}
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

        <div className="flex flex-col flex-1 min-w-0">
          <div className="border-b border-border px-4 py-4 sm:px-8 sm:py-6">
            <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
              <div className="min-w-0 space-y-4">
                <div>
                  <h1 className="text-2xl font-semibold tracking-tight text-foreground sm:text-3xl">
                    M&A Search
                  </h1>
                  <p
                    className="mt-1 text-sm text-muted-foreground"
                    aria-live="polite"
                  >
                    {searchMode === "sections"
                      ? "Search section-level matches across the corpus."
                      : "Search deals and review which matched sections brought each agreement into the result set."}
                  </p>
                </div>
                <Tabs value={searchMode} onValueChange={handleModeChange}>
                  <TabsList
                    className="inline-flex h-auto items-stretch gap-1 rounded-lg border border-border bg-muted/40 p-1"
                    aria-label="Choose what to search"
                  >
                    <TabsTrigger
                      value="sections"
                      className="group flex h-auto min-w-[150px] items-center gap-2.5 rounded-md px-3 py-2 text-left data-[state=active]:bg-background data-[state=active]:shadow-sm sm:min-w-[200px]"
                    >
                      <span
                        className={cn(
                          "flex h-8 w-8 shrink-0 items-center justify-center rounded-md transition-colors",
                          searchMode === "sections"
                            ? "bg-primary/10 text-primary"
                            : "bg-muted text-muted-foreground group-hover:bg-muted/80",
                        )}
                        aria-hidden="true"
                      >
                        <FileText className="h-4 w-4" />
                      </span>
                      <span className="flex min-w-0 flex-col leading-tight">
                        <span className="text-sm font-semibold">Sections</span>
                        <span className="hidden text-xs font-normal text-muted-foreground sm:block">
                          Section-level matches
                        </span>
                      </span>
                    </TabsTrigger>
                    <TabsTrigger
                      value="transactions"
                      className="group flex h-auto min-w-[150px] items-center gap-2.5 rounded-md px-3 py-2 text-left data-[state=active]:bg-background data-[state=active]:shadow-sm sm:min-w-[200px]"
                    >
                      <span
                        className={cn(
                          "flex h-8 w-8 shrink-0 items-center justify-center rounded-md transition-colors",
                          searchMode === "transactions"
                            ? "bg-primary/10 text-primary"
                            : "bg-muted text-muted-foreground group-hover:bg-muted/80",
                        )}
                        aria-hidden="true"
                      >
                        <Building2 className="h-4 w-4" />
                      </span>
                      <span className="flex min-w-0 flex-col leading-tight">
                        <span className="text-sm font-semibold">Deals</span>
                        <span className="hidden text-xs font-normal text-muted-foreground sm:block">
                          Whole transactions
                        </span>
                      </span>
                    </TabsTrigger>
                  </TabsList>
                </Tabs>
              </div>

              <div className="lg:hidden">
                <Sheet
                  open={isMobileFiltersOpen}
                  onOpenChange={setIsMobileFiltersOpen}
                >
                  <SheetTrigger asChild>
                    <Button variant="outline" size="sm" className="gap-2">
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
                      Filter agreement section results.
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
                          isLoadingFilterOptions={isLoadingFilterOptions}
                          isLoadingTaxonomy={isLoadingFilterOptions}
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
              <div className="mt-4">
                <Alert className="py-3 sm:py-4">
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
            <div className="mx-4 mt-4 sm:mx-8">
              <Alert variant="destructive" role="alert">
                <div className="text-sm font-medium leading-none tracking-tight">
                  Filter options error
                </div>
                <AlertDescription>{filterOptionsError}</AlertDescription>
              </Alert>
            </div>
          )}

          <div className="border-b border-border bg-muted/20 px-4 py-4 backdrop-blur supports-[backdrop-filter]:bg-muted/20 sm:px-8">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-center">
                <Button
                  onClick={() => void trackingActions.performSearch()}
                  disabled={activeIsSearching}
                  className="w-full gap-2 sm:w-auto"
                  variant="default"
                >
                  <SearchIcon
                    className={cn(
                      "h-4 w-4",
                      activeIsSearching && "animate-spin-custom"
                    )}
                    aria-hidden="true"
                  />
                  <span>{activeIsSearching ? "Searching..." : "Search"}</span>
                </Button>

                <div className="flex items-center gap-3">
                  {(() => {
                    const activeSelectedSize =
                      searchMode === "sections"
                        ? selectedResults.size
                        : transactionSearch.selectedResults.size;
                    const activeResultsLength =
                      searchMode === "sections"
                        ? searchResults.length
                        : transactionSearch.results.length;
                    const downloadDisabled =
                      activeResultsLength === 0 && activeSelectedSize === 0;
                    return (
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <span className="inline-block">
                            <Button
                              onClick={() => trackingActions.downloadCSV()}
                              disabled={downloadDisabled}
                              variant="outline"
                              size="sm"
                              className="gap-2 border-0 bg-transparent px-0 text-muted-foreground hover:text-foreground sm:border sm:bg-background sm:px-3 sm:text-foreground"
                              aria-label={
                                downloadDisabled
                                  ? "Download CSV (disabled: no results to download. Run a search first.)"
                                  : "Download CSV"
                              }
                            >
                              <Download className="h-4 w-4" aria-hidden="true" />
                              <span>
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
                    className="px-3 text-muted-foreground hover:text-foreground"
                  >
                    Reset filters
                  </Button>
                </div>
              </div>
            </div>
          </div>

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
            <div className="border-b border-border px-4 py-3 sm:px-8">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs font-medium text-muted-foreground">
                  Active filters
                </span>

                {(
                  [
                    ["year", "Year", filters.year],
                    ["target", "Target", filters.target],
                    ["acquirer", "Acquirer", filters.acquirer],
                    ["clauseType", "Section type", filters.clauseType],
                    ["transaction_price_total", "Transaction price (total)", filters.transaction_price_total],
                    ["transaction_price_stock", "Transaction price (stock)", filters.transaction_price_stock],
                    ["transaction_price_cash", "Transaction price (cash)", filters.transaction_price_cash],
                    ["transaction_price_assets", "Transaction price (assets)", filters.transaction_price_assets],
                    ["transaction_consideration", "Transaction consideration", filters.transaction_consideration],
                    ["target_type", "Target type", filters.target_type],
                    ["acquirer_type", "Acquirer type", filters.acquirer_type],
                    ["target_counsel", "Target counsel", filters.target_counsel],
                    ["acquirer_counsel", "Acquirer counsel", filters.acquirer_counsel],
                    ["target_industry", "Target industry", filters.target_industry],
                    ["acquirer_industry", "Acquirer industry", filters.acquirer_industry],
                    ["deal_status", "Deal status", filters.deal_status],
                    ["attitude", "Attitude", filters.attitude],
                    ["deal_type", "Deal type", filters.deal_type],
                    ["purpose", "Purpose", filters.purpose],
                    ["target_pe", "Target PE", filters.target_pe],
                    ["acquirer_pe", "Acquirer PE", filters.acquirer_pe],
                  ] as const
                ).flatMap(([field, label, values]) =>
                  values.map((value) => {
                    // Only format hardcoded enum values, not database values
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
                        className="flex items-center gap-1 rounded-md bg-background px-2 py-1"
                      >
                        <span className="text-muted-foreground">{label}:</span>
                        <span className="truncate">{displayValue}</span>
                        <button
                          type="button"
                          onClick={() => trackingActions.toggleFilterValue(field, value)}
                          className="ml-1 inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-sm text-muted-foreground hover:bg-accent/60 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:h-6 sm:w-6"
                          aria-label={`Remove ${label} filter: ${displayValue}`}
                        >
                          <X className="h-3 w-3" aria-hidden="true" />
                        </button>
                      </Badge>
                    );
                  })
                )}

                {/* Text filters */}
                {filters.agreement_uuid && (
                  <Badge
                    key="agreement_uuid"
                    variant="outline"
                    className="flex items-center gap-1 rounded-md bg-background px-2 py-1"
                  >
                    <span className="text-muted-foreground">Agreement UUID:</span>
                    <span className="truncate">{filters.agreement_uuid}</span>
                    <button
                      type="button"
                      onClick={() => trackingActions.setTextFilterValue("agreement_uuid", "")}
                      className="ml-1 inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-sm text-muted-foreground hover:bg-accent/60 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:h-6 sm:w-6"
                      aria-label={`Remove Agreement UUID filter: ${filters.agreement_uuid}`}
                    >
                      <X className="h-3 w-3" aria-hidden="true" />
                    </button>
                  </Badge>
                )}

                {filters.section_uuid && (
                  <Badge
                    key="section_uuid"
                    variant="outline"
                    className="flex items-center gap-1 rounded-md bg-background px-2 py-1"
                  >
                    <span className="text-muted-foreground">Section UUID:</span>
                    <span className="truncate">{filters.section_uuid}</span>
                    <button
                      type="button"
                      onClick={() => trackingActions.setTextFilterValue("section_uuid", "")}
                      className="ml-1 inline-flex min-h-[44px] min-w-[44px] items-center justify-center rounded-sm text-muted-foreground hover:bg-accent/60 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:h-6 sm:w-6"
                      aria-label={`Remove Section UUID filter: ${filters.section_uuid}`}
                    >
                      <X className="h-3 w-3" aria-hidden="true" />
                    </button>
                  </Badge>
                )}

                <div className="ml-auto">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={trackingActions.clearFilters}
                    className="h-8 px-2 text-muted-foreground hover:text-foreground"
                  >
                    Clear all filters
                  </Button>
                </div>
              </div>
            </div>
          )}

          <div className="flex-1 overflow-auto">
            <div className="px-4 py-6 sm:px-8 sm:py-8">
              {!activeHasSearched && (
                <div className="mx-auto max-w-3xl space-y-4">
                  <div className="rounded-2xl border border-border/60 bg-card p-6 shadow-sm">
                    <div className="flex items-start gap-4">
                      <div className="mt-0.5 rounded-lg bg-primary/10 p-2 text-primary">
                        <Sparkles className="h-5 w-5" aria-hidden="true" />
                      </div>
                      <div className="min-w-0">
                        <h2 className="text-base font-semibold text-foreground">
                          {searchMode === "sections"
                            ? "Find specific sections across the corpus"
                            : "Find deals matching your criteria"}
                        </h2>
                        <p className="mt-1 text-sm text-muted-foreground">
                          {searchMode === "sections"
                            ? "Pick filters to narrow the corpus, then search to load matched sections and jump straight into the relevant agreement passage."
                            : "Pick filters to narrow the corpus, then search to load matching deals with the sections that triggered each result."}
                        </p>
                      </div>
                    </div>
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="rounded-xl border border-border/60 bg-card/60 p-4">
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
                    <div className="rounded-xl border border-border/60 bg-card/60 p-4">
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
                <div className="space-y-6">
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

                          {searchMode === "sections" ? (
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
                          {searchMode === "sections" ? (
                            <SearchResultsTableFallback />
                          ) : (
                            <TransactionResultsFallback />
                          )}
                        </>
                      )}

                      {(() => {
                        const activeSelectedSize =
                          searchMode === "sections"
                            ? selectedResults.size
                            : transactionSearch.selectedResults.size;
                        if (activeSelectedSize === 0) return null;
                        const onClear =
                          searchMode === "sections"
                            ? clearSelection
                            : transactionSearch.clearSelection;
                        return (
                          <div className="rounded-xl border border-border/60 bg-muted/20 p-4 backdrop-blur supports-[backdrop-filter]:bg-muted/20">
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
          </div>
        </div>
      </div>

      {activeShowErrorModal && (
        <ErrorModal
          isOpen={activeShowErrorModal}
          onClose={() => {
            actions.closeErrorModal();
            transactionSearch.closeErrorModal();
          }}
          message={activeErrorMessage}
        />
      )}
    </div>
  );
}
