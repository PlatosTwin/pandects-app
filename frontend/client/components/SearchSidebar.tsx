import React, { useState } from "react";
import {
  TRANSACTION_CONSIDERATION_OPTIONS,
  TRANSACTION_PRICE_OPTIONS,
  TARGET_TYPE_OPTIONS,
  ACQUIRER_TYPE_OPTIONS,
  DEAL_STATUS_OPTIONS,
  ATTITUDE_OPTIONS,
  DEAL_TYPE_OPTIONS,
  PURPOSE_OPTIONS,
  PE_OPTIONS,
  SIDEBAR_ANIMATION_DELAY,
} from "@/lib/constants";
import { RotateCcw, ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { CheckboxFilter } from "@/components/CheckboxFilter";
import { NestedCheckboxFilter } from "@/components/NestedCheckboxFilter";
import { Button } from "@/components/ui/button";
import { AdaptiveTooltip } from "@/components/ui/adaptive-tooltip";
import { LoadingSpinner } from "@/components/ui/loading-spinner";
import type { ClauseTypeTree } from "@/lib/clause-types";

interface SearchSidebarProps {
  filters: {
    year?: string[];
    target?: string[];
    acquirer?: string[];
    clauseType?: string[];
    transactionPriceTotal?: string[];
    transactionPriceStock?: string[];
    transactionPriceCash?: string[];
    transactionPriceAssets?: string[];
    transactionConsideration?: string[];
    targetType?: string[];
    acquirerType?: string[];
    targetIndustry?: string[];
    acquirerIndustry?: string[];
    dealStatus?: string[];
    attitude?: string[];
    dealType?: string[];
    purpose?: string[];
    targetPe?: string[];
    acquirerPe?: string[];
  };
  years: string[];
  targets: string[];
  acquirers: string[];
  targetIndustries: string[];
  acquirerIndustries: string[];
  clauseTypesNested: ClauseTypeTree;
  clauseTypeLabelById: Record<string, string>;
  isLoadingFilterOptions: boolean;
  isLoadingTaxonomy: boolean;
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
  targetIndustries,
  acquirerIndustries,
  clauseTypesNested,
  clauseTypeLabelById,
  isLoadingFilterOptions,
  isLoadingTaxonomy,
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
            <LoadingSpinner size="sm" aria-label="Loading filter options" />
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
            <LoadingSpinner size="sm" aria-label="Loading filter options" />
          </div>
        )}
      </div>

      {/* Clause Type Filter */}
      <div className="relative">
        <NestedCheckboxFilter
          label="Clause Type"
          labelAddon={
            <AdaptiveTooltip
              trigger={
                <button
                  type="button"
                  className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-border/60 text-[10px] font-semibold text-muted-foreground transition-colors hover:border-emerald-500/40 hover:text-foreground cursor-help focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                  aria-label="Learn more about the taxonomy"
                >
                  ?
                </button>
              }
              content={
                <>
                  Learn more about the taxonomy on the{" "}
                  <a
                    href="/taxonomy"
                    className="font-medium text-primary underline underline-offset-2"
                  >
                    Taxonomy page
                  </a>
                  .
                </>
              }
              tooltipProps={{
                side: "top",
                className: "max-w-[220px] text-xs",
              }}
              delayDuration={0}
              popoverProps={{
                side: "top",
                className: "w-auto max-w-[220px] p-2 text-xs",
              }}
            />
          }
          data={clauseTypesNested}
          selectedValues={filters.clauseType || []}
          onToggle={(value) => onToggleFilterValue("clauseType", value)}
          labelById={clauseTypeLabelById}
          useModal={true}
        />
        {isLoadingTaxonomy && (
          <div
            className="absolute inset-0 flex items-center justify-center rounded bg-background/70"
            role="status"
            aria-live="polite"
          >
            <LoadingSpinner size="sm" aria-label="Loading taxonomy" />
          </div>
        )}
      </div>

      {/* Transaction Price Filters - Nested and Disabled */}
      <div className="space-y-4">
        <div className="text-xs font-normal text-muted-foreground tracking-[0.15px]">
          Transaction Price
        </div>
        <div className="ml-4 space-y-3">
          <CheckboxFilter
            label="Total"
            options={TRANSACTION_PRICE_OPTIONS}
            selectedValues={filters.transactionPriceTotal || []}
            onToggle={(value) => onToggleFilterValue("transactionPriceTotal", value)}
            hideSearch={true}
            disabled={true}
          />
          <CheckboxFilter
            label="Stock"
            options={TRANSACTION_PRICE_OPTIONS}
            selectedValues={filters.transactionPriceStock || []}
            onToggle={(value) => onToggleFilterValue("transactionPriceStock", value)}
            hideSearch={true}
            disabled={true}
          />
          <CheckboxFilter
            label="Cash"
            options={TRANSACTION_PRICE_OPTIONS}
            selectedValues={filters.transactionPriceCash || []}
            onToggle={(value) => onToggleFilterValue("transactionPriceCash", value)}
            hideSearch={true}
            disabled={true}
          />
          <CheckboxFilter
            label="Assets"
            options={TRANSACTION_PRICE_OPTIONS}
            selectedValues={filters.transactionPriceAssets || []}
            onToggle={(value) => onToggleFilterValue("transactionPriceAssets", value)}
            hideSearch={true}
            disabled={true}
          />
        </div>
      </div>

      {/* Transaction Consideration Filter */}
      <div>
        <CheckboxFilter
          label="Transaction Consideration"
          options={TRANSACTION_CONSIDERATION_OPTIONS}
          selectedValues={filters.transactionConsideration || []}
          onToggle={(value) => onToggleFilterValue("transactionConsideration", value)}
          hideSearch={true}
          disabled={true}
          formatValues={true}
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
          formatValues={true}
        />
      </div>

      {/* Acquirer Type Filter */}
      <div>
        <CheckboxFilter
          label="Acquirer Type"
          options={ACQUIRER_TYPE_OPTIONS}
          selectedValues={filters.acquirerType || []}
          onToggle={(value) => onToggleFilterValue("acquirerType", value)}
          hideSearch={true}
          disabled={true}
          formatValues={true}
        />
      </div>

      {/* Target Industry Filter */}
      <div>
        <CheckboxFilter
          label="Target Industry"
          options={targetIndustries}
          selectedValues={filters.targetIndustry || []}
          onToggle={(value) => onToggleFilterValue("targetIndustry", value)}
          hideSearch={false}
          disabled={true}
        />
      </div>

      {/* Acquirer Industry Filter */}
      <div>
        <CheckboxFilter
          label="Acquirer Industry"
          options={acquirerIndustries}
          selectedValues={filters.acquirerIndustry || []}
          onToggle={(value) => onToggleFilterValue("acquirerIndustry", value)}
          hideSearch={false}
          disabled={true}
        />
      </div>

      {/* Deal Status Filter */}
      <div>
        <CheckboxFilter
          label="Deal Status"
          options={DEAL_STATUS_OPTIONS}
          selectedValues={filters.dealStatus || []}
          onToggle={(value) => onToggleFilterValue("dealStatus", value)}
          hideSearch={true}
          disabled={true}
          formatValues={true}
        />
      </div>

      {/* Attitude Filter */}
      <div>
        <CheckboxFilter
          label="Attitude"
          options={ATTITUDE_OPTIONS}
          selectedValues={filters.attitude || []}
          onToggle={(value) => onToggleFilterValue("attitude", value)}
          hideSearch={true}
          disabled={true}
          formatValues={true}
        />
      </div>

      {/* Deal Type Filter */}
      <div>
        <CheckboxFilter
          label="Deal Type"
          options={DEAL_TYPE_OPTIONS}
          selectedValues={filters.dealType || []}
          onToggle={(value) => onToggleFilterValue("dealType", value)}
          hideSearch={true}
          disabled={true}
          formatValues={true}
        />
      </div>

      {/* Purpose Filter */}
      <div>
        <CheckboxFilter
          label="Purpose"
          options={PURPOSE_OPTIONS}
          selectedValues={filters.purpose || []}
          onToggle={(value) => onToggleFilterValue("purpose", value)}
          hideSearch={true}
          disabled={true}
          formatValues={true}
        />
      </div>

      {/* Target PE Filter */}
      <div>
        <CheckboxFilter
          label="Target PE"
          options={PE_OPTIONS}
          selectedValues={filters.targetPe || []}
          onToggle={(value) => onToggleFilterValue("targetPe", value)}
          hideSearch={true}
          disabled={true}
          formatValues={true}
        />
      </div>

      {/* Acquirer PE Filter */}
      <div>
        <CheckboxFilter
          label="Acquirer PE"
          options={PE_OPTIONS}
          selectedValues={filters.acquirerPe || []}
          onToggle={(value) => onToggleFilterValue("acquirerPe", value)}
          hideSearch={true}
          disabled={true}
          formatValues={true}
        />
      </div>
    </div>
  );

  if (variant === "sheet") {
    return (
      <div className={cn("flex h-full flex-col", className)}>
        <div className="border-b border-border bg-muted/20 p-4">
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
          "bg-card border-r border-b border-border transition-all duration-300 ease-in-out h-screen",
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
            <div className="border-b border-border bg-muted/20 p-4">
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
