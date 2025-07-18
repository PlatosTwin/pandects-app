import React, { useState } from "react";
import { RotateCcw, ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { CheckboxFilter } from "@/components/CheckboxFilter";
import { NestedCheckboxFilter } from "@/components/NestedCheckboxFilter";
import { Button } from "@/components/ui/button";

interface SearchSidebarProps {
  filters: {
    year?: string[];
    target?: string[];
    acquirer?: string[];
    clauseType?: string[];
    transactionSize?: string[];
    transactionType?: string[];
    considerationType?: string[];
    targetType?: string[];
  };
  years: string[];
  targets: string[];
  acquirers: string[];
  clauseTypesNested: any;
  isLoadingFilterOptions: boolean;
  onToggleFilterValue: (field: string, value: string) => void;
  onClearFilters: () => void;
  onToggleCollapse: () => void;
  isCollapsed: boolean;
  className?: string;
}

export function SearchSidebar({
  filters,
  years,
  targets,
  acquirers,
  clauseTypesNested,
  isLoadingFilterOptions,
  onToggleFilterValue,
  onClearFilters,
  onToggleCollapse,
  isCollapsed,
  className,
}: SearchSidebarProps) {
  const [showContent, setShowContent] = useState(false);

  // Control content visibility with proper timing
  React.useEffect(() => {
    if (isCollapsed) {
      // Hide content immediately when collapsing
      setShowContent(false);
    } else {
      // Show content after expansion animation completes
      const timer = setTimeout(() => setShowContent(true), 320);
      return () => clearTimeout(timer);
    }
  }, [isCollapsed]);

  // Initialize content visibility on mount
  React.useEffect(() => {
    if (!isCollapsed) {
      setShowContent(true);
    }
  }, []);

  return (
    <div
      className={cn(
        "flex-shrink-0 bg-white border-r border-gray-200 transition-all duration-300 ease-in-out sticky top-0 h-screen",
        isCollapsed ? "w-16" : "w-80",
        className,
      )}
    >
      {/* Sidebar Content */}
      {showContent && (
        <div className="h-full flex flex-col">
          {/* Header */}
          <div className="border-b border-gray-200 p-4">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-lg font-medium text-gray-900">
                Search Filters
              </h2>
              <button
                onClick={onToggleCollapse}
                className="text-gray-400 hover:text-gray-600 transition-colors p-1"
                title="Collapse sidebar"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
            </div>
            <Button
              onClick={onClearFilters}
              variant="outline"
              size="sm"
              className="w-full justify-center gap-2"
            >
              <RotateCcw className="w-4 h-4" />
              Clear Filters
            </Button>
          </div>

          {/* Scrollable Content */}
          <div className="flex-1 overflow-y-auto p-4">
            <div className="space-y-8">
              {/* Year Filter */}
              <div>
                <CheckboxFilter
                  label="Year"
                  options={years}
                  selectedValues={filters.year || []}
                  onToggle={(value) => onToggleFilterValue("year", value)}
                  tabIndex={1}
                />
              </div>

              {/* Target Filter */}
              <div className="relative">
                <CheckboxFilter
                  label="Target"
                  options={targets}
                  selectedValues={filters.target || []}
                  onToggle={(value) => onToggleFilterValue("target", value)}
                  tabIndex={2}
                />
                {isLoadingFilterOptions && (
                  <div className="absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center rounded">
                    <div className="w-4 h-4 animate-spin border-2 border-material-blue border-t-transparent rounded-full" />
                  </div>
                )}
              </div>

              {/* Acquirer Filter */}
              <div className="relative">
                <CheckboxFilter
                  label="Acquirer"
                  options={acquirers}
                  selectedValues={filters.acquirer || []}
                  onToggle={(value) => onToggleFilterValue("acquirer", value)}
                  tabIndex={3}
                />
                {isLoadingFilterOptions && (
                  <div className="absolute inset-0 bg-white bg-opacity-75 flex items-center justify-center rounded">
                    <div className="w-4 h-4 animate-spin border-2 border-material-blue border-t-transparent rounded-full" />
                  </div>
                )}
              </div>

              {/* Clause Type Filter */}
              <div>
                <NestedCheckboxFilter
                  label="Clause Type"
                  data={clauseTypesNested}
                  selectedValues={filters.clauseType || []}
                  onToggle={(value) => onToggleFilterValue("clauseType", value)}
                  useModal={true}
                  tabIndex={4}
                />
              </div>

              {/* Transaction Size Filter */}
              <div>
                <CheckboxFilter
                  label="Transaction Size"
                  options={[
                    "100M - 250M",
                    "250M - 500M",
                    "500M - 750M",
                    "750M - 1B",
                    "1B - 5B",
                    "5B - 10B",
                    "10B - 20B",
                    "20B+",
                  ]}
                  selectedValues={filters.transactionSize || []}
                  onToggle={(value) =>
                    onToggleFilterValue("transactionSize", value)
                  }
                  tabIndex={5}
                  hideSearch={true}
                  disabled={true}
                />
              </div>

              {/* Transaction Type Filter */}
              <div>
                <CheckboxFilter
                  label="Transaction Type"
                  options={["Strategic", "Financial"]}
                  selectedValues={filters.transactionType || []}
                  onToggle={(value) =>
                    onToggleFilterValue("transactionType", value)
                  }
                  tabIndex={6}
                  hideSearch={true}
                  disabled={true}
                />
              </div>

              {/* Consideration Type Filter */}
              <div>
                <CheckboxFilter
                  label="Consideration Type"
                  options={["All stock", "All cash", "Mixed"]}
                  selectedValues={filters.considerationType || []}
                  onToggle={(value) =>
                    onToggleFilterValue("considerationType", value)
                  }
                  tabIndex={7}
                  hideSearch={true}
                  disabled={true}
                />
              </div>

              {/* Target Type Filter */}
              <div>
                <CheckboxFilter
                  label="Target Type"
                  options={["Public", "Private"]}
                  selectedValues={filters.targetType || []}
                  onToggle={(value) => onToggleFilterValue("targetType", value)}
                  tabIndex={8}
                  hideSearch={true}
                  disabled={true}
                />
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Collapsed State Content */}
      {isCollapsed && (
        <div className="h-full pt-4 flex justify-center">
          <button
            onClick={onToggleCollapse}
            className="text-gray-400 hover:text-gray-600 transition-colors p-1"
            title="Expand sidebar"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
}
