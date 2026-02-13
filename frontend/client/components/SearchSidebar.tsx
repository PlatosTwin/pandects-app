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
import { TextFilter } from "@/components/TextFilter";
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
    transaction_price_total?: string[];
    transaction_price_stock?: string[];
    transaction_price_cash?: string[];
    transaction_price_assets?: string[];
    transaction_consideration?: string[];
    target_type?: string[];
    acquirer_type?: string[];
    target_industry?: string[];
    acquirer_industry?: string[];
    deal_status?: string[];
    attitude?: string[];
    deal_type?: string[];
    purpose?: string[];
    target_pe?: string[];
    acquirer_pe?: string[];
    agreement_uuid?: string;
    section_uuid?: string;
  };
  years: string[];
  targets: string[];
  acquirers: string[];
  target_industries: string[];
  acquirer_industries: string[];
  clauseTypesNested: ClauseTypeTree;
  clauseTypeLabelById: Record<string, string>;
  isLoadingFilterOptions: boolean;
  isLoadingTaxonomy: boolean;
  onToggleFilterValue: (field: string, value: string) => void;
  onTextFilterChange: (field: string, value: string) => void;
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
  target_industries,
  acquirer_industries,
  clauseTypesNested,
  clauseTypeLabelById,
  isLoadingFilterOptions,
  isLoadingTaxonomy,
  onToggleFilterValue,
  onTextFilterChange,
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
                  className="tooltip-help-trigger"
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
            selectedValues={filters.transaction_price_total || []}
            onToggle={(value) => onToggleFilterValue("transaction_price_total", value)}
            hideSearch={true}
            disabled={true}
          />
          <CheckboxFilter
            label="Stock"
            options={TRANSACTION_PRICE_OPTIONS}
            selectedValues={filters.transaction_price_stock || []}
            onToggle={(value) => onToggleFilterValue("transaction_price_stock", value)}
            hideSearch={true}
            disabled={true}
          />
          <CheckboxFilter
            label="Cash"
            options={TRANSACTION_PRICE_OPTIONS}
            selectedValues={filters.transaction_price_cash || []}
            onToggle={(value) => onToggleFilterValue("transaction_price_cash", value)}
            hideSearch={true}
            disabled={true}
          />
          <CheckboxFilter
            label="Assets"
            options={TRANSACTION_PRICE_OPTIONS}
            selectedValues={filters.transaction_price_assets || []}
            onToggle={(value) => onToggleFilterValue("transaction_price_assets", value)}
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
          selectedValues={filters.transaction_consideration || []}
          onToggle={(value) => onToggleFilterValue("transaction_consideration", value)}
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
          selectedValues={filters.target_type || []}
          onToggle={(value) => onToggleFilterValue("target_type", value)}
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
          selectedValues={filters.acquirer_type || []}
          onToggle={(value) => onToggleFilterValue("acquirer_type", value)}
          hideSearch={true}
          disabled={true}
          formatValues={true}
        />
      </div>

      {/* Target Industry Filter */}
      <div>
        <CheckboxFilter
          label="Target Industry"
          options={target_industries}
          selectedValues={filters.target_industry || []}
          onToggle={(value) => onToggleFilterValue("target_industry", value)}
          hideSearch={false}
          disabled={true}
        />
      </div>

      {/* Acquirer Industry Filter */}
      <div>
        <CheckboxFilter
          label="Acquirer Industry"
          options={acquirer_industries}
          selectedValues={filters.acquirer_industry || []}
          onToggle={(value) => onToggleFilterValue("acquirer_industry", value)}
          hideSearch={false}
          disabled={true}
        />
      </div>

      {/* Deal Status Filter */}
      <div>
        <CheckboxFilter
          label="Deal Status"
          options={DEAL_STATUS_OPTIONS}
          selectedValues={filters.deal_status || []}
          onToggle={(value) => onToggleFilterValue("deal_status", value)}
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
          selectedValues={filters.deal_type || []}
          onToggle={(value) => onToggleFilterValue("deal_type", value)}
          hideSearch={true}
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
          selectedValues={filters.target_pe || []}
          onToggle={(value) => onToggleFilterValue("target_pe", value)}
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
          selectedValues={filters.acquirer_pe || []}
          onToggle={(value) => onToggleFilterValue("acquirer_pe", value)}
          hideSearch={true}
          disabled={true}
          formatValues={true}
        />
      </div>

      {/* Agreement UUID Filter */}
      <div>
        <TextFilter
          label="Agreement UUID"
          value={filters.agreement_uuid || ""}
          onChange={(value) => onTextFilterChange("agreement_uuid", value)}
          placeholder="Enter agreement UUID"
        />
      </div>

      {/* Section UUID Filter */}
      <div>
        <TextFilter
          label="Section UUID"
          value={filters.section_uuid || ""}
          onChange={(value) => onTextFilterChange("section_uuid", value)}
          placeholder="Enter section UUID"
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
