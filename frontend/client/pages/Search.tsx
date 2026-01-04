import React, { useEffect, useMemo, useState } from "react";
import { AVAILABLE_YEARS } from "@/lib/constants";
import { cn } from "@/lib/utils";
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
import { SearchPagination } from "@/components/SearchPagination";
import ErrorModal from "@/components/ErrorModal";
import { AgreementModal } from "@/components/AgreementModal";
import { SearchSidebar } from "@/components/SearchSidebar";
import { SearchResultsTable } from "@/components/SearchResultsTable";
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
import { useAuth } from "@/hooks/use-auth";
import type { ClauseTypeTree } from "@/lib/clause-types";
import { indexClauseTypePaths } from "@/lib/clause-type-index";
import { trackEvent } from "@/lib/analytics";

export default function Search() {
  const { status: authStatus } = useAuth();
  const {
    filters,
    isSearching,
    searchResults,
    selectedResults,
    hasSearched,
    totalCount,
    totalPages,
    currentSort,
    currentPage,
    pageSize,
    showErrorModal,
    errorMessage,
    sortDirection,
    access,
    actions,
  } = useSearch();

  const {
    toggleFilterValue,
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

  // Get dynamic filter options
  const {
    targets,
    acquirers,
    isLoading: isLoadingFilterOptions,
    error: filterOptionsError,
  } = useFilterOptions();

  // Agreement modal state
  const [selectedAgreement, setSelectedAgreement] = useState<{
    agreementUuid: string;
    sectionUuid: string;
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
      agreementUuid: result.agreementUuid,
      sectionUuid: result.sectionUuid,
      metadata: {
        year: result.year,
        target: result.target,
        acquirer: result.acquirer,
      },
    });
  };

  const closeAgreement = () => {
    setSelectedAgreement(null);
  };

  // Static years data (not dynamic for now)
  const years = AVAILABLE_YEARS;

  const clauseTypesNested = useMemo(
    (): ClauseTypeTree => ({
      "Conditions, Termination & Closing": {
        "Closing Conditions": {
          "Conditions to Closing – Mutual or Each Party": "0d3d3806dc32e929",
          "Conditions to Closing – Party Specific": "27f3e7771de2bfc4",
        },
        "Closing Mechanics": {
          "Accountant Consents & Comfort Letters": "ba6016031fc0754d",
          "Appraisal / Dissenters' Rights": "452d682a4fa9b1dc",
          "Asset Purchase Mechanics": "18bc2f6f399f1fe5",
          "Basic Merger Mechanics": "7bc3272aa3b778e7",
          "Closing Deliverables": "56be5748315e41d0",
          "Effective Time; Closing Filings": "e66e3d365e3b3156",
          "Effects of the Merger": "d7f5778a35a8dac6",
          "Exchange & Payment Mechanics (Share Surrender)": "3f0f0f021e26686a",
          "Offer; Tender Procedures": "655641a8ddf0a5e3",
          "Shareholder & Stockholder Meetings": "468ae85912916eca",
          "Stock Purchase Mechanics": "bb235fc028d4b0ff",
          "Top-Up Options & Short-Form Merger Provisions": "0495fdd6552c9122",
        },
        "Termination Rights & Fees": {
          "Termination Effects": "787011023387144e",
          "Termination Fees & Break-Up Fees": "5d80283576bffb71",
          "Termination Rights & Triggers": "b02f7059318d484d",
        },
      },
      "Consideration & Economics": {
        "Assumption of Liabilities": {
          "Assumed Liabilities": "4c2aa4862b6dc6ef",
        },
        "Purchase Price & Adjustments": {
          "Contingent Value Rights (CVR) & Earn-Out Agreements":
            "753cece4c610bcc3",
          Distributions: "17a1631998eb3cef",
          "Fairness Opinions & Financial Advisors": "d7ae412fdcc9d956",
          "Purchase Price and Post-Closing Adjustments": "2b4119d25d6309ee",
        },
        "Stock & Equity Consideration": {
          "Hook Stock & Treasury Shares": "84bfd0173448dd4d",
          "Stock Issuance & Reservation": "2014181a8401598b",
        },
      },
      Covenants: {
        "Disclosure Schedule Interpretation": {
          "Disclosure Schedule Interpretation & Cross-References":
            "12778ed5811cb2c7",
        },
        "Interim Covenants": {
          "Access to Information / Inspection Rights": "878cd2cffec068a3",
          "Disclosure Schedule Updates & Notice of Certain Events":
            "f1b8333aee3d87e6",
          "Interim Operating Covenants & Forbearances": "a152d97ffdd63fd7",
          "Merger Sub Covenants": "d5bb9a0288084408",
          "Multiple Seller Coordination & Relationship Provisions":
            "5052442de8422063",
          "No-Shop / Non-Solicitation Covenants": "ed56961095cffa4c",
          "Pre-Closing Reorganization & Structure Steps": "d00a3fc743aab08f",
          "Reasonable Best Efforts; Cooperation": "d709883fed7af17c",
          "Regulatory / Antitrust Covenants": "2002043c983c4b4a",
          "Tax Covenants": "ae15b97a33b37b1a",
        },
        "Post-Closing Covenants": {
          "Covenant Not to Compete / Non-Compete": "1b0b9e1dffa2e8c4",
          "Employee Matters & Benefits": "6ee3ede3e3b91063",
          "Indemnification, Insurance & D&O": "aa3618c9c4c2ad46",
          "Post-Closing Access to Records": "4e23ea5a68012fc7",
          "Post-Closing Cooperation & Transition": "c42b21b52aac38a4",
          "Survival & Statute of Limitations": "6b6fcd8bcbcb8dd0",
          "Transaction Announcements & Public Relations": "7bf8b47d83b5b24f",
        },
      },
      "Definitions & Miscellaneous": {
        "Definitions & Defined Terms": {
          "Definition - Affiliate": "1e1a8e5e4b6b15e3",
          "Definition - Business Day": "ba57f2a3e2d1b2b6",
          "Definition - Change in Recommendation": "e8f19e7e5b3a9a4b",
          "Definition - Confidentiality Agreement": "7f3b5e2c1a8d7e6f",
          "Definition - Contract": "c9d2f7a8b3e6d9c2",
          "Definition - Disclosure Letter / Disclosure Schedule":
            "3e8c9f4b7e6a2d1c",
          "Definition - Financing": "9a6e7d3c5b4a8f2e",
          "Definition - GAAP": "6f4a8e2d7c9b5e3a",
          "Definition - Intellectual Property": "4b2e8a6f3d7c5e9a",
          "Definition - Knowledge": "2c5f9a3e7b6d4c8a",
          "Definition - Laws": "9e3a6f2c8b5d7e4a",
          "Definition - Liens": "7a4e2f8c6b3d9a5e",
          "Definition - Material Adverse Effect / Change": "5c8a3f7e2b6d4a9c",
          "Definition - Material Contract": "8f2a6e4c7b9d3a5e",
          "Definition - Permitted Liens": "1a9e5c8f2b4d7a6c",
          "Definition - Regulatory Approval": "6d4a9c2e8f5b7c3a",
          "Definition - Representatives": "3f7a5e8c2b6d9a4c",
          "Definition - SEC Documents / Filings": "4c9a2f6e8b3d5a7c",
          "Definition - Securities Act": "8a5c2f9e4b7d6a3c",
          "Definition - Subsidiary": "2e6a4c9f8b5d3a7e",
          "Definition - Superior Proposal": "7c3a8e5f2b4d9a6c",
          "Definition - Takeover Proposal": "5f9a3c6e8b2d7a4c",
          "Definition - Taxes": "9c6a2e4f8b7d5a3c",
        },
        "Miscellaneous Provisions": {
          "Amendment & Waiver": "a4c8e2f6b9d7a5c3",
          "Assignment & Successors": "6e9a3c4f8b2d5a7c",
          "Counterparts & Electronic Signatures": "3c7a9e2f4b8d6a5c",
          "Entire Agreement": "8f4a6c2e9b5d7a3c",
          "Expenses & Costs": "2a9c6e4f8b7d3a5c",
          "Further Assurances": "7e3a5c9f2b4d8a6c",
          "Governing Law": "4c8a6e2f9b5d7a3c",
          Headings: "9f2a5c8e4b7d6a3c",
          "Jurisdiction & Venue": "6a4c9e2f8b3d5a7c",
          "No Third-Party Beneficiaries": "3e7a4c8f2b9d6a5c",
          Notices: "8c5a2f9e4b6d7a3c",
          "Schedules & Exhibits": "5f8a3c6e2b4d9a7c",
          Severability: "2c9a6e4f8b7d5a3c",
          "Specific Performance": "7a4c8e2f9b5d6a3c",
        },
      },
      "Parties & Structure": {
        "Corporate Structure & Entities": {
          "Merger Sub Formation & Corporate Structure": "9e6a2c4f8b5d7a3c",
          "Parties to the Agreement": "4a8c6e2f9b3d5a7c",
        },
        "Shareholder Matters": {
          "Board Recommendation & Fiduciary Duties": "6c9a4e2f8b7d5a3c",
          "Shareholder Approval & Voting": "3f7a5c8e2b4d9a6c",
          "Voting Agreements & Support Agreements": "8a2c6e4f9b5d7a3c",
        },
      },
      "Representations & Warranties": {
        "Business & Operations": {
          "Affiliate Transactions": "5c8a3f6e2b9d4a7c",
          "Brokers & Financial Advisors": "2e9a4c6f8b3d5a7c",
          "Business Operations & Conduct": "7f3a6c8e2b5d9a4c",
          "Contracts & Agreements": "4a8c2e6f9b7d5a3c",
          "Customer & Supplier Relationships": "9c6a4e2f8b3d7a5c",
          "Environmental Matters": "6e3a9c4f8b2d5a7c",
          Insurance: "3f8a5c6e2b9d4a7c",
          "Intellectual Property": "8c2a6e4f9b7d5a3c",
          "Labor & Employment": "5a9c3e6f8b4d7a2c",
          "Litigation & Legal Proceedings": "2f7a4c8e6b9d5a3c",
          "No Material Adverse Change": "9a5c2e6f8b4d7a3c",
          "Organization & Authority": "6c8a3e4f2b9d5a7c",
          "Permits & Compliance": "4e7a9c2f8b6d5a3c",
          "Properties & Real Estate": "7a3c5e8f2b4d9a6c",
          "Related Party Transactions": "1c9a6e4f8b2d5a7c",
          "Taxes & Tax Returns": "8f5a2c6e9b4d7a3c",
        },
        "Financial & Accounting": {
          "Absence of Undisclosed Liabilities": "3a8c6e2f9b5d7a4c",
          "Accounts Receivable": "6e4a9c2f8b7d5a3c",
          "Books & Records": "9c7a3e6f2b4d8a5c",
          "Financial Statements": "4f8a5c2e6b9d7a3c",
          "Internal Controls": "7e2a6c9f4b8d5a3c",
        },
        "Legal & Regulatory": {
          "Anti-Corruption & FCPA": "2c9a5e6f8b4d7a3c",
          Capitalization: "8a4c6e2f9b7d5a3c",
          "Compliance with Laws": "5f7a3c8e2b9d6a4c",
          "Government Contracts": "9c2a6e4f8b5d7a3c",
          "Regulatory Matters": "6a8c3e4f2b9d5a7c",
          "SEC Documents & Exchange Act Compliance": "3e5a9c6f8b2d7a4c",
          "Securities Law Compliance": "7f2a4c8e6b9d5a3c",
        },
      },
    }),
    []
  );

  const clauseTypePathByStandardId = useMemo(
    () => indexClauseTypePaths(clauseTypesNested),
    [clauseTypesNested]
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
        performSearch(true, clauseTypesNested);
      }
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [clauseTypesNested, isSearching, performSearch]);

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
      const isTabletOrMobile = window.innerWidth < 1024; // lg breakpoint
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
          <SearchSidebar
            filters={filters}
            years={years}
            targets={targets}
            acquirers={acquirers}
            clauseTypesNested={clauseTypesNested}
            isLoadingFilterOptions={isLoadingFilterOptions}
            onToggleFilterValue={toggleFilterValue}
            onClearFilters={clearFilters}
            onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
            isCollapsed={sidebarCollapsed}
          />
        </div>

        <div className="flex flex-col flex-1 min-w-0">
          <div className="border-b border-border px-4 py-4 sm:px-8 sm:py-6">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-3">
                <FileText className="h-6 w-6 text-muted-foreground" aria-hidden="true" />
                <h1 className="text-lg font-semibold text-foreground sm:text-xl sm:font-normal">
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
                    <SearchSidebar
                      variant="sheet"
                      filters={filters}
                      years={years}
                      targets={targets}
                      acquirers={acquirers}
                      clauseTypesNested={clauseTypesNested}
                      isLoadingFilterOptions={isLoadingFilterOptions}
                      onToggleFilterValue={toggleFilterValue}
                      onClearFilters={clearFilters}
                    />
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
            <div
              className="mx-4 mt-4 rounded-lg border border-destructive/30 bg-destructive/5 p-4 text-sm sm:mx-8"
              role="alert"
            >
              <p className="font-medium text-foreground">
                Filter options error
              </p>
              <p className="mt-1 text-muted-foreground">{filterOptionsError}</p>
            </div>
          )}

          <div className="border-b border-border bg-background/60 px-4 py-4 backdrop-blur sm:px-8">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex flex-col gap-2 sm:flex-row sm:flex-wrap sm:items-center">
                <Button
                  onClick={() => performSearch(true, clauseTypesNested)}
                  disabled={isSearching}
                  className="w-full gap-2 sm:w-auto"
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
                  <Button
                    onClick={() => downloadCSV(clauseTypesNested)}
                    disabled={
                      searchResults.length === 0 && selectedResults.size === 0
                    }
                    variant="outline"
                    size="sm"
                    className="gap-2 border-0 bg-transparent px-0 text-muted-foreground hover:text-foreground sm:border sm:bg-background sm:px-3 sm:text-foreground"
                  >
                    <Download className="h-4 w-4" aria-hidden="true" />
                    <span>
                      Download CSV
                      {selectedResults.size > 0 && ` (${selectedResults.size})`}
                    </span>
                  </Button>

                  <Button
                    onClick={clearFilters}
                    variant="ghost"
                    size="sm"
                    className="px-0 text-muted-foreground hover:text-foreground sm:px-3"
                  >
                    Reset filters
                  </Button>
                </div>
              </div>

              <Button
                onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
                variant="outline"
                size="sm"
                className="hidden lg:inline-flex"
              >
                {sidebarCollapsed ? "Show filters" : "Hide filters"}
              </Button>
            </div>
          </div>

          {(filters.year.length > 0 ||
            filters.target.length > 0 ||
            filters.acquirer.length > 0 ||
            filters.clauseType.length > 0) && (
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
                  ] as const
                ).flatMap(([field, label, values]) =>
                  values.map((value) => (
                    <Badge
                      key={`${field}:${value}`}
                      variant="outline"
                      className="flex items-center gap-1 rounded-md bg-background px-2 py-1"
                    >
                      <span className="text-muted-foreground">{label}:</span>
                      <span className="truncate">{value}</span>
                      <button
                        type="button"
                        onClick={() => toggleFilterValue(field, value)}
                        className="ml-1 inline-flex h-6 w-6 items-center justify-center rounded-sm text-muted-foreground hover:bg-accent hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                        aria-label={`Remove ${label} filter: ${value}`}
                      >
                        <X className="h-3 w-3" aria-hidden="true" />
                      </button>
                    </Badge>
                  ))
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
                  <div className="rounded-2xl border border-border bg-card p-6 shadow-sm">
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
                  {totalCount === 0 ? (
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
                        totalPages={totalPages}
                        pageSize={pageSize}
                        totalCount={totalCount}
                        onPageChange={(page) =>
                          goToPage(page, clauseTypesNested)
                        }
                        onPageSizeChange={(nextPageSize) =>
                          changePageSize(nextPageSize, clauseTypesNested)
                        }
                        isLoading={isSearching}
                        isLimited={(access?.tier ?? "anonymous") === "anonymous"}
                      />

                      <SearchResultsTable
                        searchResults={searchResults}
                        selectedResults={selectedResults}
                        clauseTypePathByStandardId={clauseTypePathByStandardId}
                        sortBy={currentSort ?? "year"}
                        sortDirection={sortDirection}
                        onToggleResultSelection={toggleResultSelection}
                        onToggleSelectAll={toggleSelectAll}
                        onOpenAgreement={openAgreement}
                        onSortResults={sortResults}
                        onToggleSortDirection={toggleSortDirection}
                        density={resultsDensity}
                        onDensityChange={updateResultsDensity}
                        currentPage={currentPage}
                        pageSize={pageSize}
                      />

                      {selectedResults.size > 0 && (
                        <div className="rounded-xl border border-border bg-background/70 p-4 backdrop-blur">
                          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                            <div className="text-sm text-muted-foreground">
                              {selectedResults.size} selected
                            </div>
                            <div className="flex flex-wrap gap-2">
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => downloadCSV(clauseTypesNested)}
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
                        totalPages={totalPages}
                        pageSize={pageSize}
                        totalCount={totalCount}
                        onPageChange={(page) =>
                          goToPage(page, clauseTypesNested)
                        }
                        onPageSizeChange={(nextPageSize) =>
                          changePageSize(nextPageSize, clauseTypesNested)
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

      <ErrorModal
        isOpen={showErrorModal}
        onClose={closeErrorModal}
        message={errorMessage}
      />

      {selectedAgreement && (
        <AgreementModal
          isOpen={!!selectedAgreement}
          onClose={closeAgreement}
          agreementUuid={selectedAgreement.agreementUuid}
          targetSectionUuid={selectedAgreement.sectionUuid}
          agreementMetadata={selectedAgreement.metadata}
        />
      )}
    </div>
  );
}
