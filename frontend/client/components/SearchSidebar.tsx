import { useCallback, useMemo, useState, type ReactNode } from "react";
import { ChevronDown } from "lucide-react";
import {
  ACQUIRER_TYPE_OPTIONS,
  ATTITUDE_OPTIONS,
  DEAL_STATUS_OPTIONS,
  DEAL_TYPE_OPTIONS,
  PE_OPTIONS,
  PURPOSE_OPTIONS,
  TARGET_TYPE_OPTIONS,
  TRANSACTION_CONSIDERATION_OPTIONS,
} from "@/lib/constants";
import { cn } from "@/lib/utils";
import { CheckboxFilter, CheckboxList } from "@/components/CheckboxFilter";
import { TextFilter } from "@/components/TextFilter";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { formatIndustry } from "@/lib/naics";
import type { SearchMode, SearchQueryFilters } from "@shared/search";
import { ValueRangeFilter, YearRangeFilter } from "@/pages/search/RangeFacets";
import {
  FACET_GROUP_FIELDS,
  countActiveFilters,
  facetGroupsForMode,
  resetFieldsPatch,
  type FacetGroupId,
  type PriceField,
} from "@/pages/search/search-ui";

export interface SearchSidebarProps {
  mode: SearchMode;
  filters: SearchQueryFilters;
  target_counsels: string[];
  acquirer_counsels: string[];
  target_industries: string[];
  acquirer_industries: string[];
  /** NAICS label per industry code (`industry_labels` from filter options). */
  industry_labels?: Record<string, string>;
  isLoadingFilterOptions: boolean;
  /** Toggle one value of an array filter. */
  onToggleFilterValue: (field: string, value: string) => void;
  /** Replace several fields at once (ranges, group resets). */
  onPatchFilters: (patch: Partial<SearchQueryFilters>) => void;
  /** Set (or clear, with "") a UUID filter. */
  onTextFilterChange: (field: "agreement_uuid" | "section_uuid", value: string) => void;
  loadTargetOptions?: (query: string) => Promise<string[]>;
  loadAcquirerOptions?: (query: string) => Promise<string[]>;
  /** "rail": sticky desktop column with its own scroll; "sheet": inside the mobile sheet. */
  variant?: "rail" | "sheet";
  /** Shown above the groups (e.g. a filter-options load error). */
  notice?: ReactNode;
  className?: string;
}

const EMPTY_LABELS: Record<string, string> = {};

const GROUP_TITLES: Record<FacetGroupId, string> = {
  tax: "Tax clauses",
  deal: "Deal",
  parties: "Parties",
  counsel: "Counsel",
  advanced: "Advanced",
};

const PE_LABELS: Record<string, string> = { true: "PE-backed", false: "Not PE-backed" };

function FacetGroup({
  id,
  title,
  activeCount,
  open,
  onOpenChange,
  onReset,
  children,
}: {
  id: FacetGroupId;
  title: string;
  activeCount: number;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onReset: () => void;
  children: ReactNode;
}) {
  return (
    <Collapsible open={open} onOpenChange={onOpenChange} className="border-b border-border last:border-b-0">
      <div className="flex items-center gap-2 py-1">
        <CollapsibleTrigger
          className="flex min-h-10 flex-1 items-center gap-2 rounded-md px-1 text-left text-sm font-semibold text-foreground hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-controls={`facet-group-${id}`}
        >
          <ChevronDown
            className={cn("h-4 w-4 shrink-0 text-muted-foreground transition-transform", !open && "-rotate-90")}
            aria-hidden="true"
          />
          <span>{title}</span>{" "}
          {activeCount > 0 ? (
            <span className="rounded-full bg-primary/10 px-1.5 text-xs font-medium tabular-nums text-primary">
              {activeCount}
              {" "}<span className="sr-only">active</span>
            </span>
          ) : null}
        </CollapsibleTrigger>
        {activeCount > 0 ? (
          <button
            type="button"
            onClick={onReset}
            className="rounded-md px-2 py-1 text-xs font-medium text-muted-foreground hover:bg-accent/60 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            aria-label={`Reset ${title} filters`}
          >
            Reset
          </button>
        ) : null}
      </div>
      <CollapsibleContent id={`facet-group-${id}`} className="space-y-5 px-1 pb-5 pt-1">
        {children}
      </CollapsibleContent>
    </Collapsible>
  );
}

/**
 * Faceted filters, grouped (Deal · Parties · Counsel · Advanced; Tax mode
 * adds a Tax clauses group). Each group header shows how many of its
 * filters are active and resets just that group.
 */
export function SearchSidebar({
  mode,
  filters,
  target_counsels,
  acquirer_counsels,
  target_industries,
  acquirer_industries,
  industry_labels = EMPTY_LABELS,
  isLoadingFilterOptions,
  onToggleFilterValue,
  onPatchFilters,
  onTextFilterChange,
  loadTargetOptions,
  loadAcquirerOptions,
  variant = "rail",
  notice,
  className,
}: SearchSidebarProps) {
  const groups = facetGroupsForMode(mode);
  const counts = Object.fromEntries(
    groups.map((group) => [group, countActiveFilters(filters, FACET_GROUP_FIELDS[group])]),
  ) as Record<FacetGroupId, number>;
  // Deal (and the Tax group) start open; other groups start open only when
  // they already hold an active filter (e.g. from a shared link).
  const [openGroups, setOpenGroups] = useState<Record<FacetGroupId, boolean>>(() => ({
    tax: true,
    deal: true,
    parties: counts.parties > 0,
    counsel: counts.counsel > 0,
    advanced: counts.advanced > 0,
  }));

  // Industry values stay NAICS codes on the wire; the picker shows
  // "Label (code)" and maps toggles back to the code.
  const industryDisplay = useCallback(
    (code: string) => formatIndustry(code, industry_labels[code]) ?? code,
    [industry_labels],
  );
  const industryCodeByDisplay = useMemo(() => {
    const map = new Map<string, string>();
    for (const code of [
      ...target_industries,
      ...acquirer_industries,
      ...(filters.target_industry ?? []),
      ...(filters.acquirer_industry ?? []),
    ]) {
      map.set(industryDisplay(code), code);
    }
    return map;
  }, [target_industries, acquirer_industries, filters.target_industry, filters.acquirer_industry, industryDisplay]);
  const toggleIndustry = (field: string, display: string) =>
    onToggleFilterValue(field, industryCodeByDisplay.get(display) ?? display);

  const toggle = (field: string) => (value: string) => onToggleFilterValue(field, value);

  const groupContent: Record<FacetGroupId, ReactNode> = {
    tax: (
      <label className="flex cursor-pointer items-start gap-2 text-sm text-foreground">
        <input
          type="checkbox"
          checked={Boolean(filters.include_rep_warranty)}
          onChange={(event) =>
            onPatchFilters({ include_rep_warranty: event.target.checked ? true : undefined })
          }
          className="mt-0.5 h-4 w-4 rounded border-input accent-primary"
        />
        <span>
          Include reps &amp; warranties clauses
          <span className="block text-xs text-muted-foreground">Excluded by default.</span>
        </span>
      </label>
    ),
    deal: (
      <>
        <YearRangeFilter years={filters.year ?? []} onChange={(year) => onPatchFilters({ year })} />
        <ValueRangeFilter
          filters={filters}
          onChange={(field: PriceField, buckets) => onPatchFilters({ [field]: buckets })}
        />
        <CheckboxList
          label="Status"
          options={DEAL_STATUS_OPTIONS}
          selectedValues={filters.deal_status ?? []}
          onToggle={toggle("deal_status")}
        />
        <CheckboxList
          label="Deal type"
          options={DEAL_TYPE_OPTIONS}
          selectedValues={filters.deal_type ?? []}
          onToggle={toggle("deal_type")}
        />
        <CheckboxList
          label="Consideration"
          options={TRANSACTION_CONSIDERATION_OPTIONS}
          selectedValues={filters.transaction_consideration ?? []}
          onToggle={toggle("transaction_consideration")}
        />
        <CheckboxList
          label="Attitude"
          options={ATTITUDE_OPTIONS}
          selectedValues={filters.attitude ?? []}
          onToggle={toggle("attitude")}
        />
        <CheckboxList
          label="Purpose"
          options={PURPOSE_OPTIONS}
          selectedValues={filters.purpose ?? []}
          onToggle={toggle("purpose")}
        />
      </>
    ),
    parties: (
      <>
        <CheckboxFilter
          label="Target"
          options={[]}
          selectedValues={filters.target ?? []}
          onToggle={toggle("target")}
          asyncSearch={loadTargetOptions ? { loadOptions: loadTargetOptions } : undefined}
        />
        <CheckboxFilter
          label="Acquirer"
          options={[]}
          selectedValues={filters.acquirer ?? []}
          onToggle={toggle("acquirer")}
          asyncSearch={loadAcquirerOptions ? { loadOptions: loadAcquirerOptions } : undefined}
        />
        <CheckboxFilter
          label="Target Industry"
          options={target_industries.map(industryDisplay)}
          selectedValues={(filters.target_industry ?? []).map(industryDisplay)}
          onToggle={(value) => toggleIndustry("target_industry", value)}
          disabled={isLoadingFilterOptions && target_industries.length === 0}
        />
        <CheckboxFilter
          label="Acquirer Industry"
          options={acquirer_industries.map(industryDisplay)}
          selectedValues={(filters.acquirer_industry ?? []).map(industryDisplay)}
          onToggle={(value) => toggleIndustry("acquirer_industry", value)}
          disabled={isLoadingFilterOptions && acquirer_industries.length === 0}
        />
        <CheckboxList
          label="Target type"
          options={TARGET_TYPE_OPTIONS}
          selectedValues={filters.target_type ?? []}
          onToggle={toggle("target_type")}
        />
        <CheckboxList
          label="Acquirer type"
          options={ACQUIRER_TYPE_OPTIONS}
          selectedValues={filters.acquirer_type ?? []}
          onToggle={toggle("acquirer_type")}
        />
        <CheckboxList
          label="Target private equity"
          options={PE_OPTIONS}
          selectedValues={filters.target_pe ?? []}
          onToggle={toggle("target_pe")}
          formatOption={(value) => PE_LABELS[value] ?? value}
        />
        <CheckboxList
          label="Acquirer private equity"
          options={PE_OPTIONS}
          selectedValues={filters.acquirer_pe ?? []}
          onToggle={toggle("acquirer_pe")}
          formatOption={(value) => PE_LABELS[value] ?? value}
        />
      </>
    ),
    counsel: (
      <>
        <CheckboxFilter
          label="Target Counsel"
          options={target_counsels}
          selectedValues={filters.target_counsel ?? []}
          onToggle={toggle("target_counsel")}
          disabled={isLoadingFilterOptions && target_counsels.length === 0}
        />
        <CheckboxFilter
          label="Acquirer Counsel"
          options={acquirer_counsels}
          selectedValues={filters.acquirer_counsel ?? []}
          onToggle={toggle("acquirer_counsel")}
          disabled={isLoadingFilterOptions && acquirer_counsels.length === 0}
        />
      </>
    ),
    advanced: (
      <>
        <TextFilter
          key={`agreement:${filters.agreement_uuid ?? ""}`}
          label="Agreement UUID"
          value={filters.agreement_uuid ?? ""}
          onCommit={(value) => onTextFilterChange("agreement_uuid", value)}
          placeholder="e.g. 4e1e8fda-2492-…"
        />
        <TextFilter
          key={`section:${filters.section_uuid ?? ""}`}
          label="Section UUID"
          value={filters.section_uuid ?? ""}
          onCommit={(value) => onTextFilterChange("section_uuid", value)}
          placeholder="e.g. ffcce93b-1257-…"
        />
      </>
    ),
  };

  const body = (
    <div
      role="region"
      aria-label="Search filters"
      aria-busy={isLoadingFilterOptions}
      // From xl (a 288px rail) pl-7 plus the groups' px-1 puts the content on
      // the app-frame gutter, in line with the query bar and the logo; the
      // 256px lg rail needs the width for the deal-value toggle.
      className="px-4 xl:pl-7"
    >
      {notice}
      {groups.map((group) => (
        <FacetGroup
          key={group}
          id={group}
          title={GROUP_TITLES[group]}
          activeCount={counts[group]}
          open={openGroups[group]}
          onOpenChange={(open) => setOpenGroups((prev) => ({ ...prev, [group]: open }))}
          onReset={() => onPatchFilters(resetFieldsPatch(FACET_GROUP_FIELDS[group]))}
        >
          {groupContent[group]}
        </FacetGroup>
      ))}
    </div>
  );

  if (variant === "sheet") {
    return <div className={cn("pb-[calc(1rem+env(safe-area-inset-bottom))]", className)}>{body}</div>;
  }

  return (
    // Sticks under the site header plus the sticky query bar and scrolls on
    // its own. The page wrapper uses overflow-x-clip (not -hidden), which
    // would make it a scroll container and break `sticky`.
    <aside
      aria-label="Filters"
      className={cn(
        "sticky top-[calc(4rem+var(--search-bar-h,0px))] hidden h-[calc(100vh-4rem-var(--search-bar-h,0px))] w-64 shrink-0 self-start overflow-y-auto overscroll-contain border-r border-border bg-card pb-8 pt-2 lg:block xl:w-72 2xl:w-80",
        className,
      )}
    >
      {body}
    </aside>
  );
}
