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
  };
  years: string[];
  targets: string[];
  acquirers: string[];
  clauseTypesNested: any;
  isLoadingFilterOptions: boolean;
  onToggleFilterValue: (field: string, value: string) => void;
  onClearFilters: () => void;
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
  isCollapsed,
  className,
}: SearchSidebarProps) {
  const [isTransitioning, setIsTransitioning] = useState(false);

  // Track transitions
  React.useEffect(() => {
    setIsTransitioning(true);
    const timer = setTimeout(() => setIsTransitioning(false), 300);
    return () => clearTimeout(timer);
  }, [isCollapsed]);

  return (
    <div
      className={cn(
        "flex-shrink-0 bg-white border-r border-gray-200 transition-all duration-300 ease-in-out relative overflow-hidden",
        isCollapsed ? "w-12" : "w-80",
        className,
      )}
    >
      {/* Sidebar Content */}
      <div
        className={cn(
          "h-full flex flex-col",
          isCollapsed || isTransitioning
            ? "opacity-0 pointer-events-none"
            : "opacity-100",
          "transition-opacity duration-150",
        )}
      >
        {/* Header */}
        <div className="border-b border-gray-200 p-4">
          <h2 className="text-lg font-medium text-gray-900">Search Filters</h2>
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
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-gray-200 p-4">
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
      </div>

      {/* Collapsed State Icon */}
      {isCollapsed && (
        <div className="flex flex-col items-center justify-center h-full">
          <div className="writing-mode-vertical-rl text-sm text-gray-500 transform rotate-180">
            Filters
          </div>
        </div>
      )}
    </div>
  );
}
