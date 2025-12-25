import React, { useState } from "react";
import {
  TRANSACTION_SIZE_OPTIONS,
  TRANSACTION_TYPE_OPTIONS,
  CONSIDERATION_TYPE_OPTIONS,
  TARGET_TYPE_OPTIONS,
  SIDEBAR_ANIMATION_DELAY,
} from "@/lib/constants";
import { RotateCcw, ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { CheckboxFilter } from "@/components/CheckboxFilter";
import { NestedCheckboxFilter } from "@/components/NestedCheckboxFilter";
import { Button } from "@/components/ui/button";
import type { ClauseTypeTree } from "@/lib/clause-types";

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
  clauseTypesNested: ClauseTypeTree;
  isLoadingFilterOptions: boolean;
  onToggleFilterValue: (field: string, value: string) => void;
  onClearFilters: () => void;
  onToggleCollapse?: () => void;
  isCollapsed?: boolean;
  variant?: "sidebar" | "sheet";
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
  isCollapsed = false,
  variant = "sidebar",
  className,
}: SearchSidebarProps) {
  const [showContent, setShowContent] = useState(false);
  const toggleCollapse = onToggleCollapse ?? (() => {});

  // Control content visibility with proper timing
  React.useEffect(() => {
    if (variant === "sheet") {
      setShowContent(true);
      return;
    }
    if (isCollapsed) {
      // Hide content immediately when collapsing
      setShowContent(false);
    } else {
      // Show content after expansion animation completes
      const timer = setTimeout(
        () => setShowContent(true),
        SIDEBAR_ANIMATION_DELAY,
      );
      return () => clearTimeout(timer);
    }
  }, [isCollapsed, variant]);

  // Initialize content visibility on mount
  React.useEffect(() => {
    if (variant === "sheet" || !isCollapsed) {
      setShowContent(true);
    }
  }, [isCollapsed, variant]);

  const filtersContent = (
    <div className="space-y-8">
      {/* Year Filter */}
      <div>
        <CheckboxFilter
          label="Year"
          options={years}
          selectedValues={filters.year || []}
          onToggle={(value) => onToggleFilterValue("year", value)}
        />
      </div>

      {/* Target Filter */}
      <div className="relative">
        <CheckboxFilter
          label="Target"
          options={targets}
          selectedValues={filters.target || []}
          onToggle={(value) => onToggleFilterValue("target", value)}
        />
        {isLoadingFilterOptions && (
          <div
            className="absolute inset-0 flex items-center justify-center rounded bg-background/70"
            role="status"
            aria-live="polite"
          >
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
            <span className="sr-only">Loading filter options</span>
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
        />
        {isLoadingFilterOptions && (
          <div
            className="absolute inset-0 flex items-center justify-center rounded bg-background/70"
            role="status"
            aria-live="polite"
          >
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
            <span className="sr-only">Loading filter options</span>
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
        />
      </div>

      {/* Transaction Size Filter */}
      <div>
        <CheckboxFilter
          label="Transaction Size"
          options={TRANSACTION_SIZE_OPTIONS}
          selectedValues={filters.transactionSize || []}
          onToggle={(value) => onToggleFilterValue("transactionSize", value)}
          hideSearch={true}
          disabled={true}
        />
      </div>

      {/* Transaction Type Filter */}
      <div>
        <CheckboxFilter
          label="Transaction Type"
          options={TRANSACTION_TYPE_OPTIONS}
          selectedValues={filters.transactionType || []}
          onToggle={(value) => onToggleFilterValue("transactionType", value)}
          hideSearch={true}
          disabled={true}
        />
      </div>

      {/* Consideration Type Filter */}
      <div>
        <CheckboxFilter
          label="Consideration Type"
          options={CONSIDERATION_TYPE_OPTIONS}
          selectedValues={filters.considerationType || []}
          onToggle={(value) => onToggleFilterValue("considerationType", value)}
          hideSearch={true}
          disabled={true}
        />
      </div>

      {/* Target Type Filter */}
      <div>
        <CheckboxFilter
          label="Target Type"
          options={TARGET_TYPE_OPTIONS}
          selectedValues={filters.targetType || []}
          onToggle={(value) => onToggleFilterValue("targetType", value)}
          hideSearch={true}
          disabled={true}
        />
      </div>
    </div>
  );

  if (variant === "sheet") {
    return (
      <div className={cn("flex h-full flex-col", className)}>
        <div className="border-b border-border p-4">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-base font-semibold text-foreground">Filters</h2>
          </div>
          <Button
            onClick={onClearFilters}
            variant="outline"
            size="sm"
            className="w-full justify-center gap-2"
          >
            <RotateCcw className="h-4 w-4" aria-hidden="true" />
            Clear Filters
          </Button>
        </div>

        <div className="flex-1 overflow-y-auto p-4">{filtersContent}</div>
      </div>
    );
  }

  return (
    <>
      {/* Backdrop for tablet overlay */}
      {!isCollapsed && (
        <button
          type="button"
          className="fixed inset-0 border-0 bg-black bg-opacity-50 p-0 z-40 lg:hidden"
          onClick={toggleCollapse}
          aria-label="Close filters"
        />
      )}

      <div
        className={cn(
          "bg-card border-r border-border transition-all duration-300 ease-in-out h-screen",
          // Desktop: normal sidebar behavior
          "lg:flex-shrink-0 lg:sticky lg:top-0 lg:relative",
          isCollapsed ? "lg:w-16" : "lg:w-80",
          // Tablet/Mobile: overlay when expanded, collapsed when not
          isCollapsed
            ? "w-16 flex-shrink-0 sticky top-0"
            : "w-80 fixed top-0 left-0 z-50 lg:static lg:z-auto",
          className,
        )}
      >
        {/* Sidebar Content */}
        {showContent && (
          <div className="h-full flex flex-col">
            {/* Header */}
            <div className="border-b border-border p-4">
              <div className="flex items-center justify-between mb-3">
                <h2 className="text-lg font-medium text-foreground">
                  Search Filters
                </h2>
                <button
                  type="button"
                  onClick={toggleCollapse}
                  className="rounded-md p-1 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                  title="Collapse sidebar"
                  aria-label="Collapse sidebar"
                >
                  <ChevronLeft className="w-4 h-4" aria-hidden="true" />
                </button>
              </div>
              <Button
                onClick={onClearFilters}
                variant="outline"
                size="sm"
                className="w-full justify-center gap-2"
              >
                <RotateCcw className="w-4 h-4" aria-hidden="true" />
                Clear Filters
              </Button>
            </div>

            {/* Scrollable Content */}
            <div className="flex-1 overflow-y-auto p-4">{filtersContent}</div>
          </div>
        )}

        {/* Collapsed State Content */}
        {isCollapsed && (
          <div className="h-full pt-4 flex justify-center">
            <button
              type="button"
              onClick={toggleCollapse}
              className="rounded-md p-1 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
              title="Expand sidebar"
              aria-label="Expand sidebar"
            >
              <ChevronRight className="w-4 h-4" aria-hidden="true" />
            </button>
          </div>
        )}
      </div>
    </>
  );
}
