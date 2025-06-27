import { cn } from "@/lib/utils";
import { Search as SearchIcon, Download, FileText } from "lucide-react";
import { useSearch } from "@/hooks/use-search";

export default function Search() {
  const { filters, isSearching, searchResults, hasSearched, actions } =
    useSearch();

  // Placeholder data for dropdowns
  const years = ["2024", "2023", "2022", "2021", "2020", "2019"];
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
    "Termination",
    "Material Adverse Change",
    "Representations",
    "Warranties",
    "Indemnification",
  ];

  return (
    <div className="w-full font-roboto flex flex-col">
      <div className="flex flex-col gap-8 p-12">
        {/* Header */}
        <div className="flex items-center gap-4">
          <FileText className="w-8 h-8 text-material-blue" />
          <h1 className="text-2xl font-medium text-material-text-primary">
            M&A Clause Search
          </h1>
        </div>

        {/* Search Filters */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {/* Announcement Year */}
          <div className="flex flex-col gap-2">
            <label className="text-xs font-normal text-material-text-secondary tracking-[0.15px]">
              Announcement Year
            </label>
            <div className="relative">
              <select
                value={filters.announcementYear}
                onChange={(e) =>
                  actions.updateFilter("announcementYear", e.target.value)
                }
                className="w-full text-base font-normal text-material-text-primary bg-transparent border-none border-b border-[rgba(0,0,0,0.42)] py-2 focus:outline-none focus:border-material-blue appearance-none pr-8"
              >
                <option value="">All Years</option>
                {years.map((year) => (
                  <option key={year} value={year}>
                    {year}
                  </option>
                ))}
              </select>
              <div className="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none">
                <svg
                  className="w-4 h-4 text-material-text-secondary"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </div>
              <div className="absolute bottom-0 left-0 right-0 h-px bg-[rgba(0,0,0,0.42)]" />
            </div>
          </div>

          {/* Target */}
          <div className="flex flex-col gap-2">
            <label className="text-xs font-normal text-material-text-secondary tracking-[0.15px]">
              Target
            </label>
            <div className="relative">
              <select
                value={filters.target}
                onChange={(e) => actions.updateFilter("target", e.target.value)}
                className="w-full text-base font-normal text-material-text-primary bg-transparent border-none border-b border-[rgba(0,0,0,0.42)] py-2 focus:outline-none focus:border-material-blue appearance-none pr-8"
              >
                <option value="">All Targets</option>
                {targets.map((target) => (
                  <option key={target} value={target}>
                    {target}
                  </option>
                ))}
              </select>
              <div className="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none">
                <svg
                  className="w-4 h-4 text-material-text-secondary"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </div>
              <div className="absolute bottom-0 left-0 right-0 h-px bg-[rgba(0,0,0,0.42)]" />
            </div>
          </div>

          {/* Acquirer */}
          <div className="flex flex-col gap-2">
            <label className="text-xs font-normal text-material-text-secondary tracking-[0.15px]">
              Acquirer
            </label>
            <div className="relative">
              <select
                value={filters.acquirer}
                onChange={(e) =>
                  actions.updateFilter("acquirer", e.target.value)
                }
                className="w-full text-base font-normal text-material-text-primary bg-transparent border-none border-b border-[rgba(0,0,0,0.42)] py-2 focus:outline-none focus:border-material-blue appearance-none pr-8"
              >
                <option value="">All Acquirers</option>
                {acquirers.map((acquirer) => (
                  <option key={acquirer} value={acquirer}>
                    {acquirer}
                  </option>
                ))}
              </select>
              <div className="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none">
                <svg
                  className="w-4 h-4 text-material-text-secondary"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </div>
              <div className="absolute bottom-0 left-0 right-0 h-px bg-[rgba(0,0,0,0.42)]" />
            </div>
          </div>

          {/* Clause Type */}
          <div className="flex flex-col gap-2">
            <label className="text-xs font-normal text-material-text-secondary tracking-[0.15px]">
              Clause Type
            </label>
            <div className="relative">
              <select
                value={filters.clauseType}
                onChange={(e) =>
                  actions.updateFilter("clauseType", e.target.value)
                }
                className="w-full text-base font-normal text-material-text-primary bg-transparent border-none border-b border-[rgba(0,0,0,0.42)] py-2 focus:outline-none focus:border-material-blue appearance-none pr-8"
              >
                <option value="">All Types</option>
                {clauseTypes.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
              <div className="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none">
                <svg
                  className="w-4 h-4 text-material-text-secondary"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M19 9l-7 7-7-7"
                  />
                </svg>
              </div>
              <div className="absolute bottom-0 left-0 right-0 h-px bg-[rgba(0,0,0,0.42)]" />
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-4">
          <button
            onClick={actions.performSearch}
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
          <div className="flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-medium text-material-text-primary">
                Search Results
              </h2>
              <span className="text-sm text-material-text-secondary">
                {searchResults.length}{" "}
                {searchResults.length === 1 ? "result" : "results"} found
              </span>
            </div>

            {searchResults.length === 0 ? (
              <div className="text-center py-12 text-material-text-secondary">
                <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
                <p>No clauses found matching your search criteria.</p>
                <p className="text-sm mt-2">
                  Try adjusting your filters and search again.
                </p>
              </div>
            ) : (
              <div className="grid gap-4">
                {searchResults.map((result) => (
                  <div
                    key={result.id}
                    className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden"
                  >
                    {/* Header with metadata */}
                    <div className="bg-gray-50 px-4 py-3 border-b border-gray-200">
                      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
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
                            {result.article}
                          </div>
                        </div>
                        <div>
                          <span className="font-medium text-material-text-secondary">
                            Section:
                          </span>
                          <div className="text-material-text-primary">
                            {result.section}
                          </div>
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
                        {result.text}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
