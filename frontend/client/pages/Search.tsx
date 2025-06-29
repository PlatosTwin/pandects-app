import { useState } from "react";
import { cn } from "@/lib/utils";
import {
  Search as SearchIcon,
  Download,
  FileText,
  ExternalLink,
} from "lucide-react";
import { useSearch } from "@/hooks/use-search";
import { SearchPagination } from "@/components/SearchPagination";
import ErrorModal from "@/components/ErrorModal";
import InfoModal from "@/components/InfoModal";
import { XMLRenderer } from "@/components/XMLRenderer";
import { AgreementModal } from "@/components/AgreementModal";
import { CheckboxFilter } from "@/components/CheckboxFilter";

export default function Search() {
  const {
    filters,
    isSearching,
    searchResults,
    hasSearched,
    totalCount,
    totalPages,
    currentPage,
    pageSize,
    showErrorModal,
    errorMessage,
    showNoResultsModal,
    actions,
  } = useSearch();

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

  // Placeholder data for dropdowns
  const years = ["2020", "2019", "2018", "2017", "2016", "2015", "2014", "2013", "2012", "2011", "2010", "2009", "2008", "2007", "2006", "2005", "2004", "2003", "2002", "2001", "2000"];
  const targets = [
    "Apple Inc.",
    "Microsoft Corp.",
    "Google LLC",
    "Amazon.com Inc.",
    "Meta Platforms Inc.",
  ];
  const acquirers = [
    "Berkshire Hathaway",
    "JPMorgan Chase",
    "Bank of America",
    "Wells Fargo",
    "Goldman Sachs",
  ];
  const clauseTypes = [
    'Conditions to Closing',
    'Covenants and Additional Agreements',
    'Definitions',
    'Dispute Resolution',
    'Indemnification',
    'Miscellaneous and General Provisions',
    'Organizational Documents and Governance',
    'Representations and Warranties',
    'Tax Matters',
    'Termination',
    'Transaction Mechanics'
  ];

  return (
    <div className="w-full font-roboto flex flex-col">
      <div className="flex flex-col gap-8 p-12">
        {/* Header */}
        <div className="flex items-center gap-3">
          <FileText className="w-6 h-6 text-material-text-secondary" />
          <h1 className="text-xl font-normal text-material-text-primary">
            M&A Clause Search
          </h1>
        </div>

        {/* Search Filters */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <CheckboxFilter
            label="Year"
            options={years}
            selectedValues={filters.year || []}
            onToggle={(value) => actions.toggleFilterValue("year", value)}
          />

          <CheckboxFilter
            label="Target"
            options={targets}
            selectedValues={filters.target || []}
            onToggle={(value) => actions.toggleFilterValue("target", value)}
          />

          <CheckboxFilter
            label="Acquirer"
            options={acquirers}
            selectedValues={filters.acquirer || []}
            onToggle={(value) => actions.toggleFilterValue("acquirer", value)}
          />

          <CheckboxFilter
            label="Clause Type"
            options={clauseTypes}
            selectedValues={filters.clauseType || []}
            onToggle={(value) => actions.toggleFilterValue("clauseType", value)}
          />
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-4">
          <button
            onClick={() => actions.performSearch(true)}
            disabled={isSearching}
            className={cn(
              "flex items-center justify-center gap-2 px-6 py-3 rounded-md bg-material-blue text-white text-[15px] font-medium leading-[26px] tracking-[0.46px] uppercase transition-all duration-200",
              "shadow-[0px_1px_5px_0px_rgba(0,0,0,0.12),0px_2px_2px_0px_rgba(0,0,0,0.14),0px_3px_1px_-2px_rgba(0,0,0,0.20)]",
              "hover:shadow-[0px_2px_8px_0px_rgba(0,0,0,0.15),0px_3px_4px_0px_rgba(0,0,0,0.18),0px_4px_2px_-2px_rgba(0,0,0,0.25)]",
              "disabled:opacity-50 disabled:cursor-not-allowed",
            )}
          >
            <SearchIcon
              className={cn("w-5 h-5", isSearching && "animate-spin-custom")}
            />
            <span>{isSearching ? "Searching..." : "Search"}</span>
          </button>

          <button
            onClick={actions.downloadCSV}
            disabled={searchResults.length === 0}
            className={cn(
              "flex items-center justify-center gap-2 px-6 py-3 rounded-md border border-material-blue text-material-blue text-[15px] font-medium leading-[26px] tracking-[0.46px] uppercase transition-all duration-200",
              "hover:bg-material-blue-light",
              "disabled:opacity-50 disabled:cursor-not-allowed disabled:border-gray-400 disabled:text-gray-400",
            )}
          >
            <Download className="w-5 h-5" />
            <span>Download CSV</span>
          </button>
        </div>

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

                {/* Results grid */}
                <div className="grid gap-4">
                  {searchResults.map((result) => (
                    <div
                      key={result.id}
                      className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden"
                    >
                      {/* Header with metadata */}
                      <div className="bg-gray-50 px-4 py-3 border-b border-gray-200">
                        <div className="flex items-center justify-between">
                          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm flex-1">
                            <div>
                              <span className="font-medium text-material-text-secondary">
                                Year:
                              </span>
                              <div className="text-material-text-primary">
                                {result.year}
                              </div>
                            </div>
                            <div>
                              <span className="font-medium text-material-text-secondary">
                                Target:
                              </span>
                              <div className="text-material-text-primary">
                                {result.target}
                              </div>
                            </div>
                            <div>
                              <span className="font-medium text-material-text-secondary">
                                Acquirer:
                              </span>
                              <div className="text-material-text-primary">
                                {result.acquirer}
                              </div>
                            </div>
                            <div>
                              <span className="font-medium text-material-text-secondary">
                                Article:
                              </span>
                              <div className="text-material-text-primary">
                                {result.articleTitle}
                              </div>
                            </div>
                            <div>
                              <span className="font-medium text-material-text-secondary">
                                Section:
                              </span>
                              <div className="text-material-text-primary">
                                {result.sectionTitle}
                              </div>
                            </div>
                          </div>

                          {/* Open Agreement Button */}
                          <div className="ml-4">
                            <button
                              onClick={() => openAgreement(result)}
                              className="flex items-center gap-2 px-3 py-2 text-sm text-material-blue hover:bg-material-blue-light rounded transition-colors"
                              title="Open source agreement"
                            >
                              <ExternalLink className="w-4 h-4" />
                              <span className="hidden sm:inline">
                                Open Agreement
                              </span>
                            </button>
                          </div>
                        </div>
                      </div>

                      {/* Clause text */}
                      <div className="p-4">
                        <div
                          className="h-32 overflow-y-auto text-sm text-material-text-primary leading-relaxed"
                          style={{
                            scrollbarWidth: "thin",
                            scrollbarColor: "#e5e7eb #f9fafb",
                          }}
                        >
                          <XMLRenderer xmlContent={result.xml} mode="search" />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

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
  );
}
