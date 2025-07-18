import React, { useState } from "react";
import { AVAILABLE_YEARS } from "@/lib/constants";
import { cn } from "@/lib/utils";
import {
  Search as SearchIcon,
  Download,
  FileText,
  PanelLeftOpen,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { useSearch } from "@/hooks/use-search";
import { useFilterOptions } from "@/hooks/use-filter-options";
import { SearchPagination } from "@/components/SearchPagination";
import ErrorModal from "@/components/ErrorModal";
import InfoModal from "@/components/InfoModal";
import { AgreementModal } from "@/components/AgreementModal";
import { SearchSidebar } from "@/components/SearchSidebar";
import { SearchResultsTable } from "@/components/SearchResultsTable";
import { Button } from "@/components/ui/button";
import Navigation from "@/components/Navigation";

export default function Search() {
  const {
    filters,
    isSearching,
    searchResults,
    selectedResults,
    hasSearched,
    totalCount,
    totalPages,
    currentPage,
    pageSize,
    showErrorModal,
    errorMessage,
    showNoResultsModal,
    sortDirection,
    actions,
  } = useSearch();

  // Handle Enter key for search
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !isSearching) {
      // Check if any filter elements are focused or dropdowns are open
      const activeElement = document.activeElement as HTMLElement;

      // Check if any dropdowns are currently open by looking for expanded dropdown containers
      const hasOpenDropdown =
        document.querySelector(".absolute.top-full") || // CheckboxFilter dropdowns
        document.querySelector('[role="dialog"]'); // Modal dialogs

      // Don't trigger search if:
      // - An input is focused
      // - A button with dropdown functionality is focused
      // - Any element inside a dropdown/modal is focused
      // - Any dropdown is currently open
      const isInputFocused = activeElement?.tagName === "INPUT";
      const isButtonFocused = activeElement?.tagName === "BUTTON";
      const isInsideDropdown =
        activeElement?.closest('[role="combobox"]') ||
        activeElement?.closest(".absolute") || // Dropdown containers
        activeElement?.closest('[role="dialog"]'); // Modal containers

      if (
        !isInputFocused &&
        !isButtonFocused &&
        !isInsideDropdown &&
        !hasOpenDropdown
      ) {
        actions.performSearch(true, clauseTypesNested);
      }
    }
  };

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

  const openAgreement = (result: (typeof searchResults)[0]) => {
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

  const clauseTypesNested = {
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
  };

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <div className="min-h-screen bg-cream flex flex-col">
      <Navigation />
      <div
        className="w-full font-roboto flex-1 relative"
        onKeyDown={handleKeyDown}
        tabIndex={-1}
      >
        <div className="flex min-h-full">
          {/* Collapsible Sidebar with Filters */}
          <SearchSidebar
            filters={filters}
            years={years}
            targets={targets}
            acquirers={acquirers}
            clauseTypesNested={clauseTypesNested}
            isLoadingFilterOptions={isLoadingFilterOptions}
            onToggleFilterValue={actions.toggleFilterValue}
            onClearFilters={actions.clearFilters}
            onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
            isCollapsed={sidebarCollapsed}
          />

          {/* Main Content Area */}
          <div className="flex flex-col flex-1 min-w-0">
            {/* Header */}
            <div className="flex items-center gap-3 border-b border-material-divider px-8 py-6">
              <FileText className="w-6 h-6 text-material-text-secondary" />
              <h1 className="text-xl font-normal text-material-text-primary">
                M&A Clause Search
              </h1>
            </div>

            {/* Filter Options Error */}
            {filterOptionsError && (
              <div className="mx-6 mt-6 bg-red-50 border border-red-200 rounded-md p-4">
                <p className="text-red-800 text-sm">
                  <strong>Filter Options Error:</strong> {filterOptionsError}
                </p>
                <p className="text-red-600 text-xs mt-1">
                  Some filter options may not be available. Please refresh the
                  page to try again.
                </p>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex items-center gap-4 px-8 py-6 border-b border-material-divider">
              <Button
                onClick={() => actions.performSearch(true, clauseTypesNested)}
                disabled={isSearching}
                className={cn(
                  "flex items-center justify-center gap-2 px-6 py-3 bg-material-blue text-white text-[15px] font-medium leading-[26px] tracking-[0.46px] uppercase",
                  "shadow-[0px_1px_5px_0px_rgba(0,0,0,0.12),0px_2px_2px_0px_rgba(0,0,0,0.14),0px_3px_1px_-2px_rgba(0,0,0,0.20)]",
                  "hover:shadow-[0px_2px_8px_0px_rgba(0,0,0,0.15),0px_3px_4px_0px_rgba(0,0,0,0.18),0px_4px_2px_-2px_rgba(0,0,0,0.25)]",
                )}
              >
                <SearchIcon
                  className={cn(
                    "w-5 h-5",
                    isSearching && "animate-spin-custom",
                  )}
                />
                <span>{isSearching ? "Searching..." : "Search"}</span>
              </Button>

              <Button
                onClick={() => actions.downloadCSV(clauseTypesNested)}
                disabled={
                  searchResults.length === 0 && selectedResults.size === 0
                }
                variant="outline"
                className={cn(
                  "flex items-center justify-center gap-2 px-6 py-3 border-material-blue text-material-blue text-[15px] font-medium leading-[26px] tracking-[0.46px] uppercase",
                  "hover:bg-material-blue-light",
                )}
              >
                <Download className="w-5 h-5" />
                <span>
                  Download CSV{" "}
                  {selectedResults.size > 0 && `(${selectedResults.size})`}
                </span>
              </Button>
            </div>

            {/* Main Content - Scrollable */}
            <div className="flex-1 overflow-auto">
              <div className="px-8 py-8">
                {/* Search Results */}
                {hasSearched && (
                  <div className="flex flex-col gap-6">
                    <div className="flex items-center justify-between">
                      <h2 className="text-lg font-medium text-material-text-primary">
                        Search Results
                      </h2>
                    </div>

                    {totalCount === 0 ? (
                      <div className="text-center py-12 text-material-text-secondary">
                        <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
                        <p>No clauses found matching your search criteria.</p>
                        <p className="text-sm mt-2">
                          Try adjusting your filters and search again.
                        </p>
                      </div>
                    ) : (
                      <>
                        {/* Top pagination controls */}
                        <SearchPagination
                          currentPage={currentPage}
                          totalPages={totalPages}
                          pageSize={pageSize}
                          totalCount={totalCount}
                          onPageChange={(page) =>
                            actions.goToPage(page, clauseTypesNested)
                          }
                          onPageSizeChange={(pageSize) =>
                            actions.changePageSize(pageSize, clauseTypesNested)
                          }
                          isLoading={isSearching}
                        />

                        {/* Search Results Table with Checkboxes */}
                        <SearchResultsTable
                          searchResults={searchResults}
                          selectedResults={selectedResults}
                          sortDirection={sortDirection}
                          onToggleResultSelection={
                            actions.toggleResultSelection
                          }
                          onToggleSelectAll={actions.toggleSelectAll}
                          onOpenAgreement={openAgreement}
                          onSortResults={actions.sortResults}
                          onToggleSortDirection={actions.toggleSortDirection}
                        />

                        {/* Bottom pagination controls */}
                        <SearchPagination
                          currentPage={currentPage}
                          totalPages={totalPages}
                          pageSize={pageSize}
                          totalCount={totalCount}
                          onPageChange={(page) =>
                            actions.goToPage(page, clauseTypesNested)
                          }
                          onPageSizeChange={(pageSize) =>
                            actions.changePageSize(pageSize, clauseTypesNested)
                          }
                          isLoading={isSearching}
                        />
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Backend Connection Error Modal */}
        <ErrorModal
          isOpen={showErrorModal}
          onClose={actions.closeErrorModal}
          message={errorMessage}
        />

        {/* No Results Info Modal */}
        <InfoModal
          isOpen={showNoResultsModal}
          onClose={actions.closeNoResultsModal}
          title="No Results Found"
          message="No results to display given the selected filters."
        />

        {/* Agreement Modal */}
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
    </div>
  );
}
