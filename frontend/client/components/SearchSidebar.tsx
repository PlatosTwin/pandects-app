import { RotateCcw } from "lucide-react";
import { cn } from "@/lib/utils";
import { CheckboxFilter } from "@/components/CheckboxFilter";
import { NestedCheckboxFilter } from "@/components/NestedCheckboxFilter";
import { Button } from "@/components/ui/button";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";

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
  className,
}: SearchSidebarProps) {
  return (
    <SidebarProvider defaultOpen={true}>
      <Sidebar className={cn("w-80", className)} collapsible="offcanvas">
        <SidebarHeader className="border-b border-sidebar-border">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-medium text-sidebar-foreground">
              Search Filters
            </h2>
            <SidebarTrigger />
          </div>
        </SidebarHeader>

        <SidebarContent className="px-4 py-6">
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
        </SidebarContent>

        <SidebarFooter className="border-t border-sidebar-border p-4">
          <Button
            onClick={onClearFilters}
            variant="outline"
            size="sm"
            className="w-full justify-center gap-2"
          >
            <RotateCcw className="w-4 h-4" />
            Clear Filters
          </Button>
        </SidebarFooter>
      </Sidebar>
    </SidebarProvider>
  );
}
