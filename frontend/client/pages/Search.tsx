import React, { Suspense, lazy, useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { AVAILABLE_YEARS, BREAKPOINT_LG } from "@/lib/constants";
import { formatFilterOption } from "@/lib/text-utils";
import { cn } from "@/lib/utils";
import type { SearchFilters } from "@shared/search";
import {
  Search as SearchIcon,
  Download,
  FileText,
  SlidersHorizontal,
  Sparkles,
  X,
} from "lucide-react";
import { useSearch } from "@/hooks/use-search";
import { useFilterOptions } from "@/hooks/use-filter-options";
import { useTaxonomy } from "@/hooks/use-taxonomy";
import { SearchPagination } from "@/components/SearchPagination";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { useAuth } from "@/hooks/use-auth";
import type { ClauseTypeTree } from "@/lib/clause-types";
import { indexClauseTypeLabels, indexClauseTypePaths } from "@/lib/clause-type-index";
import { trackEvent } from "@/lib/analytics";
import { apiUrl } from "@/lib/api-config";

const SearchSidebar = lazy(() =>
  import("@/components/SearchSidebar").then((mod) => ({
    default: mod.SearchSidebar,
  })),
);
const SearchResultsTable = lazy(() =>
  import("@/components/SearchResultsTable").then((mod) => ({
    default: mod.SearchResultsTable,
  })),
);
const AgreementModal = lazy(() =>
  import("@/components/AgreementModal").then((mod) => ({
    default: mod.AgreementModal,
  })),
);
const ErrorModal = lazy(() => import("@/components/ErrorModal"));

export default function Search() {
  const { status: authStatus } = useAuth();
  const {
    filters,
    isSearching,
    searchResults,
    selectedResults,
    hasSearched,
    total_count,
    total_pages,
    currentSort,
    currentPage,
    page_size,
    showErrorModal,
    errorMessage,
    sort_direction,
    access,
    actions,
  } = useSearch();

  const {
    toggleFilterValue,
    setTextFilterValue,
    performSearch,
    downloadCSV,
    clearFilters,
    goToPage,
    changePageSize,
    closeErrorModal,
    sortResults,
    toggleSortDirection,
    toggleResultSelection,
    toggleSelectAll,
    clearSelection,
  } = actions;

  // Wrap actions with tracking
  const trackingActions = {
    toggleFilterValue: (field: string, value: string) => {
      trackEvent("search_filter_change", {
        filter_field: field,
        filter_value: value.substring(0, 50), // truncate long values
        current_filters: Object.keys(filters).length,
      });
      toggleFilterValue(field as keyof SearchFilters, value);
    },
    setTextFilterValue: (field: string, value: string) => {
      trackEvent("search_filter_change", {
        filter_field: field,
        filter_value: value.substring(0, 50), // truncate long values
        current_filters: Object.keys(filters).length,
      });
      setTextFilterValue(field as keyof SearchFilters, value);
    },
    performSearch: (force?: boolean) => {
      if (force || !hasSearched) {
        trackEvent("search_performed", {
          filter_count: Object.values(filters).flat().length,
          has_results: searchResults.length > 0,
          result_count: total_count,
        });
      }
      performSearch(force, clauseTypesNested);
    },
    downloadCSV: () => {
      trackEvent("search_export_click", {
        export_format: "csv",
        result_count: selectedResults.size > 0 ? selectedResults.size : total_count,
        is_filtered: Object.values(filters).flat().length > 0,
      });
      downloadCSV();
    },
    clearFilters: () => {
      trackEvent("search_filters_cleared", {
        filter_count: Object.values(filters).flat().length,
      });
      clearFilters();
    },
    goToPage: (page: number) => {
      trackEvent("search_pagination", {
        page_number: page,
        total_pages: total_pages,
        direction: page > currentPage ? "next" : "previous",
      });
      goToPage(page, clauseTypesNested);
    },
    changePageSize: (size: number) => {
      trackEvent("search_page_size_change", {
        old_page_size: page_size,
        new_page_size: size,
      });
      changePageSize(size, clauseTypesNested);
    },
    sortResults: (field: string) => {
      trackEvent("search_sort_change", {
        sort_field: field,
        sort_direction: currentSort === field ? "reversed" : "initial",
      });
      sortResults(field as "year" | "target" | "acquirer");
    },
    toggleSortDirection: () => {
      trackEvent("search_sort_direction_toggle", {
        sort_field: currentSort,
        new_direction: sort_direction === "asc" ? "desc" : "asc",
      });
      toggleSortDirection();
    },
  };

  // Get dynamic filter options
  const {
    targets,
    acquirers,
    target_industries,
    acquirer_industries,
    isLoading: isLoadingFilterOptions,
    error: filterOptionsError,
  } = useFilterOptions({ deferMs: 1200 });

  const [searchParams, setSearchParams] = useSearchParams();

  // Agreement modal state
  const [selectedAgreement, setSelectedAgreement] = useState<{
    agreement_uuid: string;
    section_uuid: string;
    metadata: {
      year: string;
      target: string;
      acquirer: string;
    };
  } | null>(null);

  const openAgreement = (result: (typeof searchResults)[0], position: number) => {
    trackEvent("search_result_click", {
      position,
      year: result.year,
      verified: result.verified,
    });
    setSelectedAgreement({
      agreement_uuid: result.agreement_uuid,
      section_uuid: result.section_uuid,
      metadata: {
        year: result.year,
        target: result.target,
        acquirer: result.acquirer,
      },
    });
  };

  const closeAgreement = () => {
    setSelectedAgreement(null);
    if (
      searchParams.has("agreement_uuid") ||
      searchParams.has("section_uuid")
    ) {
      const next = new URLSearchParams(searchParams);
      next.delete("agreement_uuid");
      next.delete("section_uuid");
      setSearchParams(next, { replace: true });
    }
  };

  const agreementUuidFromUrl = searchParams.get("agreement_uuid");
  const sectionUuidFromUrl = searchParams.get("section_uuid");

  useEffect(() => {
    if (
      !agreementUuidFromUrl ||
      !sectionUuidFromUrl ||
      selectedAgreement != null
    )
      return;
    setSelectedAgreement({
      agreement_uuid: agreementUuidFromUrl,
      section_uuid: sectionUuidFromUrl,
      metadata: { year: "", target: "", acquirer: "" },
    });
  }, [agreementUuidFromUrl, sectionUuidFromUrl, selectedAgreement]);

  // Static years data (not dynamic for now)
  const years = AVAILABLE_YEARS;

  const { taxonomyTree, isLoading: isLoadingTaxonomy } = useTaxonomy({
    deferMs: 1200,
  });
  const clauseTypesNested: ClauseTypeTree = taxonomyTree ?? {};

  const clauseTypePathByStandardId = useMemo(
    () => indexClauseTypePaths(clauseTypesNested),
    [clauseTypesNested]
  );
  const clauseTypeLabelById = useMemo(
    () => indexClauseTypeLabels(clauseTypesNested),
    [clauseTypesNested],
  );

  // Allow Enter to trigger search when focus isn't inside an input/control.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Enter" || isSearching) return;

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
        trackingActions.performSearch(true);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [clauseTypesNested, isSearching, performSearch]);

  useEffect(() => {
    const warmup = () => {
      void fetch(apiUrl("v1/dumps")).catch(() => undefined);
    };
    const schedule = window.requestIdleCallback
      ? window.requestIdleCallback(warmup, { timeout: 2500 })
      : window.setTimeout(warmup, 1800);
    return () => {
      if (window.cancelIdleCallback) {
        window.cancelIdleCallback(schedule as number);
      } else {
        window.clearTimeout(schedule as number);
      }
    };
  }, []);

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isMobileFiltersOpen, setIsMobileFiltersOpen] = useState(false);
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

  // Auto-collapse sidebar on tablet and mobile
  useEffect(() => {
    const handleResize = () => {
      const isTabletOrMobile = window.innerWidth < BREAKPOINT_LG;
      setSidebarCollapsed(isTabletOrMobile);
    };

    // Set initial state
    handleResize();

    // Listen for window resize
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return (
    <div className="w-full">
      <div className="flex min-h-full">
        <div className="hidden lg:block">
          <Suspense
            fallback={
              <div className="h-screen w-16 border-r border-border bg-card lg:w-80" />
            }
          >
            <SearchSidebar
              filters={filters}
              years={years}
              targets={targets}
              acquirers={acquirers}
              target_industries={target_industries}
              acquirer_industries={acquirer_industries}
              clauseTypesNested={clauseTypesNested}
              clauseTypeLabelById={clauseTypeLabelById}
              isLoadingFilterOptions={isLoadingFilterOptions}
              isLoadingTaxonomy={isLoadingTaxonomy}
              onToggleFilterValue={toggleFilterValue}
              onTextFilterChange={trackingActions.setTextFilterValue}
              onClearFilters={clearFilters}
              onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
              isCollapsed={sidebarCollapsed}
            />
          </Suspense>
        </div>

        <div className="flex flex-col flex-1 min-w-0">
          <div className="border-b border-border px-4 py-4 sm:px-8 sm:py-6">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <FileText className="h-6 w-6 text-muted-foreground" aria-hidden="true" />
                <h1 className="text-2xl font-semibold text-foreground sm:text-3xl">
                  M&A Clause Search
                </h1>
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
                      Filter agreement clause results.
                    </SheetDescription>
                    <Suspense
                      fallback={
                        <div className="p-4 text-sm text-muted-foreground">
                          Loading filters...
                        </div>
                      }
                    >
                      <SearchSidebar
                        variant="sheet"
                        filters={filters}
                        years={years}
                        targets={targets}
                        acquirers={acquirers}
                        target_industries={target_industries}
                        acquirer_industries={acquirer_industries}
                        clauseTypesNested={clauseTypesNested}
                        clauseTypeLabelById={clauseTypeLabelById}
                        isLoadingFilterOptions={isLoadingFilterOptions}
                        isLoadingTaxonomy={isLoadingTaxonomy}
                        onToggleFilterValue={toggleFilterValue}
                        onTextFilterChange={trackingActions.setTextFilterValue}
                        onClearFilters={clearFilters}
                      />
                    </Suspense>
                  </SheetContent>
                </Sheet>
              </div>
            </div>

            {authStatus === "anonymous" && (
              <div className="mt-4">
                <Alert className="py-3 sm:py-4">
                  <Sparkles className="h-4 w-4" aria-hidden="true" />
                  <AlertTitle>Limited mode</AlertTitle>
                  <AlertDescription>
                    Sign in to view clause text, open full agreements, and unlock
                    higher page sizes.
                  </AlertDescription>
                </Alert>
              </div>
            )}
          </div>

          {filterOptionsError && (
            <div className="mx-4 mt-4 sm:mx-8">
              <Alert variant="destructive" role="alert">
                <AlertTitle>Filter options error</AlertTitle>
                <AlertDescription>{filterOptionsError}</AlertDescription>
              </Alert>
            </div>
          )}

          <div className="border-b border-border bg-muted/20 px-4 py-4 backdrop-blur supports-[backdrop-filter]:bg-muted/20 sm:px-8">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-center">
                <Button
                  onClick={() => trackingActions.performSearch(true)}
                  disabled={isSearching}
                  className="w-full gap-2 sm:w-auto"
                  variant="default"
                >
                  <SearchIcon
                    className={cn(
                      "h-4 w-4",
                      isSearching && "animate-spin-custom"
                    )}
                    aria-hidden="true"
                  />
                  <span>{isSearching ? "Searching..." : "Search"}</span>
                </Button>

                <div className="flex items-center gap-3">
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <span className="inline-block">
                        <Button
                          onClick={() => trackingActions.downloadCSV()}
                          disabled={
                            searchResults.length === 0 && selectedResults.size === 0
                          }
                          variant="outline"
                          size="sm"
                          className="gap-2 border-0 bg-transparent px-0 text-muted-foreground hover:text-foreground sm:border sm:bg-background sm:px-3 sm:text-foreground"
                          aria-label={
                            searchResults.length === 0 && selectedResults.size === 0
                              ? "Download CSV (disabled: no results to download. Run a search first.)"
                              : "Download CSV"
                          }
                        >
                          <Download className="h-4 w-4" aria-hidden="true" />
                          <span>
                            Download CSV
                            {selectedResults.size > 0 && ` (${selectedResults.size})`}
                          </span>
                        </Button>
                      </span>
                    </TooltipTrigger>
                    {searchResults.length === 0 && selectedResults.size === 0 && (
                      <TooltipContent>
                        <p>No results to download. Run a search first.</p>
                      </TooltipContent>
                    )}
                  </Tooltip>

                  <Button
                    onClick={clearFilters}
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
                    ["clauseType", "Clause type", filters.clauseType],
                    ["transaction_price_total", "Transaction price (total)", filters.transaction_price_total],
                    ["transaction_price_stock", "Transaction price (stock)", filters.transaction_price_stock],
                    ["transaction_price_cash", "Transaction price (cash)", filters.transaction_price_cash],
                    ["transaction_price_assets", "Transaction price (assets)", filters.transaction_price_assets],
                    ["transaction_consideration", "Transaction consideration", filters.transaction_consideration],
                    ["target_type", "Target type", filters.target_type],
                    ["acquirer_type", "Acquirer type", filters.acquirer_type],
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
                    onClick={clearFilters}
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
              {!hasSearched && (
                <div className="mx-auto max-w-3xl">
                  <div className="rounded-2xl border border-border/60 bg-card p-6 shadow-sm">
                    <div className="flex items-start gap-4">
                      <div className="mt-0.5 rounded-lg bg-primary/10 p-2 text-primary">
                        <Sparkles className="h-5 w-5" aria-hidden="true" />
                      </div>
                      <div className="min-w-0">
                        <h2 className="text-base font-semibold text-foreground">
                          Start with filters, then search
                        </h2>
                        <p className="mt-1 text-sm text-muted-foreground">
                          Pick a year, target/acquirer, and clause types to
                          narrow the corpus. Then run a search to load results
                          (and open full agreements from any result).
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {hasSearched && (
                <div className="space-y-6">
                  {total_count === 0 ? (
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
                        No clauses found.
                      </p>
                      <p className="text-sm mt-2">
                        Try adjusting your filters and search again.
                      </p>
                    </div>
                  ) : (
                    <>
                      <SearchPagination
                        currentPage={currentPage}
                        totalPages={total_pages}
                        pageSize={page_size}
                        totalCount={total_count}
                        onPageChange={(page) =>
                          trackingActions.goToPage(page)
                        }
                        onPageSizeChange={(nextPageSize) =>
                          trackingActions.changePageSize(nextPageSize)
                        }
                        isLoading={isSearching}
                        isLimited={(access?.tier ?? "anonymous") === "anonymous"}
                      />

                      <Suspense
                        fallback={
                          <div
                            className="rounded-lg border border-border/60 bg-card p-6 text-sm text-muted-foreground"
                            role="status"
                            aria-live="polite"
                          >
                            Loading results...
                          </div>
                        }
                      >
                        <SearchResultsTable
                          searchResults={searchResults}
                          selectedResults={selectedResults}
                          clauseTypePathByStandardId={clauseTypePathByStandardId}
                          sort_by={currentSort ?? "year"}
                          sort_direction={sort_direction}
                          onToggleResultSelection={toggleResultSelection}
                          onToggleSelectAll={toggleSelectAll}
                          onOpenAgreement={openAgreement}
                          onSortResults={sortResults}
                          onToggleSortDirection={toggleSortDirection}
                          density={resultsDensity}
                          onDensityChange={updateResultsDensity}
                          currentPage={currentPage}
                          page_size={page_size}
                        />
                      </Suspense>

                      {selectedResults.size > 0 && (
                        <div className="rounded-xl border border-border/60 bg-muted/20 p-4 backdrop-blur supports-[backdrop-filter]:bg-muted/20">
                          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                            <div className="text-sm text-muted-foreground">
                              {selectedResults.size} selected
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
                                onClick={clearSelection}
                              >
                                Clear selection
                              </Button>
                            </div>
                          </div>
                        </div>
                      )}

                      <SearchPagination
                        currentPage={currentPage}
                        totalPages={total_pages}
                        pageSize={page_size}
                        totalCount={total_count}
                        onPageChange={(page) =>
                          trackingActions.goToPage(page)
                        }
                        onPageSizeChange={(nextPageSize) =>
                          trackingActions.changePageSize(nextPageSize)
                        }
                        isLoading={isSearching}
                        isLimited={(access?.tier ?? "anonymous") === "anonymous"}
                      />
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {showErrorModal && (
        <Suspense fallback={null}>
          <ErrorModal
            isOpen={showErrorModal}
            onClose={closeErrorModal}
            message={errorMessage}
          />
        </Suspense>
      )}

      {selectedAgreement && (
        <Suspense fallback={null}>
          <AgreementModal
            isOpen={!!selectedAgreement}
            onClose={closeAgreement}
            agreement_uuid={selectedAgreement.agreement_uuid}
            targetSectionUuid={selectedAgreement.section_uuid}
            agreementMetadata={selectedAgreement.metadata}
          />
        </Suspense>
      )}
    </div>
  );
}
