import { useState } from "react";
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

      console.log("Global search keydown:", {
        isInputFocused,
        isButtonFocused,
        isInsideDropdown,
        hasOpenDropdown,
        activeElement: activeElement?.tagName,
        className: activeElement?.className,
      });

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
  const years = [
    "2020",
    "2019",
    "2018",
    "2017",
    "2016",
    "2015",
    "2014",
    "2013",
    "2012",
    "2011",
    "2010",
    "2009",
    "2008",
    "2007",
    "2006",
    "2005",
    "2004",
    "2003",
    "2002",
    "2001",
    "2000",
  ];

  const clauseTypesNested = {
    "Conditions, Termination & Closing": {
      "Closing Conditions": {
        "Conditions to Closing \u2013 Mutual or Each Party": "0d3d3806dc32e929",
        "Conditions to Closing \u2013 Party Specific": "27f3e7771de2bfc4",
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
        "Restrictive Covenants (Non-Competition / Non-Solicitation)":
          "dc4f5d9952565c02",
        "Transfer Restrictions & Lock-Up": "3cbbb51dc2926305",
      },
      "Post-Closing Covenants": {
        "Further Assurances & Post-Closing Cooperation": "27ef2efa0f3398f1",
        "Integration Planning & Transitional Matters": "94f771fc4289b31a",
        "Miscellaneous Covenants": "821c9acdcc7ad1a7",
        "Transition Services": "b5fddac82e36c386",
        "Use of Name & Trademark Covenants": "5a7fae57519678c3",
      },
    },
    "Employment & Benefits": {
      "Employee Matters": {
        "Equity Awards, Stock Options & Company Benefit Equity":
          "1b02124b75bc10c4",
        "Labor, Employment & Employee Benefit Plans": "c773fe515c18494a",
      },
    },
    "Financing & Funding": {
      "Debt & Instruments": {
        "Debt Instruments and Financing Arrangements": "5941cf1b1a2f1063",
        "Derivatives & Hedging Arrangements": "c92c796e9c6d19f0",
      },
      "Financing Commitments & Ability": {
        "Financing Ability / Funds Availability": "0f9291f14f678472",
        "Financing Source Provisions & Waivers": "3e1b582e2f9cc125",
        "Parent/Holdco Guarantees": "6d0c8ff46e9aafe0",
      },
      "Financing Covenants": {
        "Financing Cooperation & Support Covenants": "cb4ab225a7e5e348",
      },
    },
    "General Provisions": {
      "Boilerplate & Construction": {
        "Alternative Transaction Structure / Change of Method":
          "7dbd17612e9dabe5",
        "Amendments; Extensions; Waivers": "c20cd24a0d84ba56",
        "Arbitration & Dispute Resolution": "f383ecddb78f002f",
        "Assignment & Successors": "305bb88f1e98b4d4",
        Confidentiality: "6df8bc88edff7e3d",
        "Counterparts & Electronic Signatures": "d44006cdbbb8ad0c",
        "Entire Agreement": "10cd17a4560a7a7d",
        "Governing Law; Jurisdiction; Jury Trial": "b79e1996d2f7bcac",
        "Language & Translation": "6b403871d6bd4b42",
        "No Third Party Beneficiaries": "f4591e658883f659",
        "Notices & Communications": "40bba1e9ee7e2325",
        "Relationship of the Parties; Independent Contractor Status":
          "89a01278af3a9a40",
        "Remedies; Specific Performance": "5b8b1666661c6c43",
        "Service of Process": "433ef1632fe94c19",
        "Severability and Boilerplate Provisions": "2560701d69ecbaff",
      },
      Definitions: {
        "Definitions & Interpretation": "beee4c5deefdc2d0",
      },
      "Expenses & Advisors": {
        "Brokers & Finders Fees": "da24cbb98240a799",
        "Fees and Expenses": "a7227a5a1e16bdbb",
      },
    },
    "Governance & Shareholder Matters": {
      "Board & Committees": {
        "Board Approval & Recommendation": "77d8c1b6c3edaa15",
        "Conflicts Committee Matters": "098f9d5c048af2ae",
      },
      "Post-Merger Governance": {
        "Directors, Officers & Governance of Surviving Entity":
          "e38fc136fafbd07b",
      },
      "Shareholder Agreements & Voting": {
        "Securityholder Representative Provisions": "8766fbc645223b7d",
        "Voting, Support & Shareholder Agreements": "dccc050c7b770819",
      },
      "Takeover & Anti-Takeover": {
        "Antitakeover / Takeover Statutes": "9dacfb2c929bbd9a",
        "UK Takeover Panel Matters": "fc39027ced6e9b5e",
      },
    },
    "Indemnification & Liability": {
      "Indemnification Mechanics": {
        "Indemnifiable Losses Definition": "c76c574a8f437a05",
        "Indemnification Escrow & Holdback Provisions": "726b4f89306e679a",
        "Indemnification Limitations (Caps & Baskets)": "a73e0398ad8b0104",
        "Indemnification Procedures": "9613f111a9bf9db6",
      },
      Indemnities: {
        "Indemnification by Purchaser / Buyer": "b8ad9d98fefc6844",
        "Indemnification by Seller / Target": "3f4a09babc73019c",
      },
      "Survival & Recourse": {
        "D&O Indemnification & Insurance": "10866d4d64fc32c8",
        "Indemnification \u2013 Exclusive Remedy / Sole Remedy":
          "0fefebbe62102092",
        "Non-Recourse & Limited Liability": "8718c336dc859307",
        "Seller Releases & Waivers": "d7ce59cfd33e134d",
        "Survival / Non-Survival of Reps & Warranties": "949a439a8c96ba76",
      },
    },
    "Regulatory & Compliance": {
      "Antitrust & Competition": {
        "Antitrust & Competition Filings": "3dd07ef88fe2d06c",
      },
      "Government Approvals": {
        "Governmental Approvals & Consents": "5736fd3809ff8d95",
      },
      "Insider Trading & Section 16": {
        "Section 16 & Insider Trading Compliance (incl. Rule 16b-3)":
          "2ef0d8d1ce1c1a8f",
      },
      "Securities Filings & Disclosure": {
        "Proxy Statement, Prospectus & Registration Matters":
          "0fd42cc616dfd56b",
        "Public Disclosure, Press Releases & SEC Communications":
          "2d33954063e3ae0e",
        "Securities Documents & Capital Markets Filings": "7fcc5649a3985190",
      },
      "Stock Exchange": {
        "Stock Exchange Listing": "52748f62ccff7015",
      },
    },
    "Representations & Warranties": {
      "Assets & Liabilities": {
        "Accounts Receivable & Accounts Payable": "bf3939f8db1cdb5c",
        "Assets; Inventory & Goodwill": "4c49669ab3e2a089",
        "Casualty Loss and Condemnation": "ee49e752cab4d692",
        "Deposits & Deposit Liabilities (Banking)": "d429b02c43f3f808",
        "Excluded Assets": "8f15d7260c5376bf",
        "Loan Portfolio & Asset Quality": "9f853034983a9a6a",
        "Real Property & Tangible Assets": "93314630278e3f4d",
        "Undisclosed Liabilities": "3589284e71431996",
      },
      "Books & Records": {
        "Books & Records": "842de2d70fbf1443",
      },
      "Capital Structure": {
        Capitalization: "2091c8e1a4d26971",
        "Stock Ownership Representations": "06b871f2e90569f1",
      },
      "Changes & Updates": {
        "Absence of Certain Changes / Events": "ede429a629cd3b41",
      },
      "Contracts & Relationships": {
        "Customers & Suppliers; Business Relationships": "7233d89a27c42583",
        "Material Contracts & Commitments": "330d438a248a3d42",
        "Related Party Transactions": "31ca3ad3458e621c",
      },
      "Corporate Organization & Authority": {
        "Authority & Non-Contravention": "ae87408e3761beb1",
        "Charter & Bylaws of Surviving Entity": "770417d31342dfdd",
        "General / Introductory Representations & Warranties":
          "ab254c22aefab482",
        "Merger Sub Representations": "431556a5a84f90cf",
        "Organizational Matters (Existence, Qualification, Subsidiaries)":
          "206b2da0fb295d63",
        "Solvency Representations": "914f006e880c9b38",
      },
      "Disclosure & Accuracy": {
        "Disclaimer of Other Representations": "a41fcf985e05fab1",
        "Disclosure; Accuracy of Information": "921ae6080f3d21a9",
      },
      "Financial Information & Controls": {
        "Financial Projections & Forecasts": "60bc1d30739af1fd",
        "Financial Statements & Accounting": "8516ebe5cd9bd0da",
        "Interim Financial Information": "d202d93d2dd353f1",
        "Internal Controls & Sarbanes-Oxley Compliance": "4f1df8f12a4b5a3d",
      },
      "Insurance & Warranty": {
        "Insurance Matters": "b6a4646b170e7a10",
        "Product Warranty & Liability": "10ac69129ed22eb0",
      },
      "Intellectual Property": {
        "Intellectual Property": "2ba0f1c7aea9472c",
      },
      "Litigation & Disputes": {
        "Litigation & Legal Proceedings": "662bf7d543cdddcf",
        "Shareholder & Securityholder Litigation": "db6a86b940dd8b4f",
      },
      "Operational Compliance & Permits": {
        "Anti-Corruption & FCPA Compliance": "9fe92383900ee52e",
        "Compliance with Laws and Regulatory Matters": "d0a4735e95f26ac9",
        "Data Privacy & Cybersecurity": "f39f22b4fcc6f60d",
        "Environmental Matters": "8edf005cfab27fe0",
        "Permits & Franchises": "e8e20e5fda2cbbda",
      },
    },
    "Tax Matters": {
      "Tax Covenants": {
        "Other Tax Covenants / Tax Treatment of Payments": "e22eb654144286c2",
      },
      "Tax Disputes": {
        "Tax Proceedings; Disputes and Audit Contests": "56e9ab8aac48d8f8",
      },
      "Tax Representations": {
        "Tax Representations": "64756455cb6a8407",
      },
      "Transfer Taxes": {
        "Transfer Taxes": "7fd16bf902d13048",
      },
    },
  };

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <div
      className="w-full font-roboto min-h-screen relative"
      onKeyDown={handleKeyDown}
      tabIndex={-1}
    >
      {/* Toggle Button - Positioned independently */}
      <button
        onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
        className="absolute bg-white border border-gray-200 rounded-full p-1 h-6 w-6 shadow-md hover:shadow-lg flex items-center justify-center transition-all duration-300 ease-in-out"
        style={{
          top: "140px",
          left: sidebarCollapsed ? "36px" : "308px",
          zIndex: 10000,
        }}
      >
        {sidebarCollapsed ? (
          <ChevronRight className="h-3 w-3" />
        ) : (
          <ChevronLeft className="h-3 w-3" />
        )}
      </button>

      <div className="flex min-h-screen">
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
          isCollapsed={sidebarCollapsed}
        />

        {/* Main Content Area */}
        <div className="flex flex-col flex-1 min-w-0">
          {/* Header */}
          <div className="flex items-center gap-3 border-b border-gray-200 p-6">
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
          <div className="flex items-center gap-4 p-6 border-b border-gray-200">
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
                className={cn("w-5 h-5", isSearching && "animate-spin-custom")}
              />
              <span>{isSearching ? "Searching..." : "Search"}</span>
            </Button>

            <Button
              onClick={actions.downloadCSV}
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
            <div className="p-6">
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
                        onPageChange={actions.goToPage}
                        onPageSizeChange={actions.changePageSize}
                        isLoading={isSearching}
                      />

                      {/* Search Results Table with Checkboxes */}
                      <SearchResultsTable
                        searchResults={searchResults}
                        selectedResults={selectedResults}
                        sortDirection={sortDirection}
                        onToggleResultSelection={actions.toggleResultSelection}
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
                        onPageChange={actions.goToPage}
                        onPageSizeChange={actions.changePageSize}
                        isLoading={isSearching}
                      />
                    </>
                  )}
                </div>
              )}
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
