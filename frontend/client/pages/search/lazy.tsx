import { lazy } from "react";
import { Skeleton } from "@/components/ui/skeleton";

/**
 * Lazy-loaded search result components and their skeleton fallbacks, so the
 * Search route shell stays small. A new mode adds its component and fallback here.
 */

export const SearchPagination = lazy(() =>
  import("@/components/SearchPagination").then((mod) => ({ default: mod.SearchPagination })),
);
export const SearchResultsTable = lazy(() =>
  import("@/components/SearchResultsTable").then((mod) => ({ default: mod.SearchResultsTable })),
);
export const SearchSidebar = lazy(() =>
  import("@/components/SearchSidebar").then((mod) => ({ default: mod.SearchSidebar })),
);
export const TransactionResultsList = lazy(() =>
  import("@/components/TransactionResultsList").then((mod) => ({ default: mod.TransactionResultsList })),
);
export const TaxClauseResultsList = lazy(() =>
  import("@/components/TaxClauseResultsList").then((mod) => ({ default: mod.TaxClauseResultsList })),
);

export function SearchSidebarFallback({ variant = "rail" }: { variant?: "rail" | "sheet" }) {
  const content = (
    <div className="space-y-5 p-4" aria-hidden="true">
      <Skeleton className="h-5 w-24" />
      <Skeleton className="h-9 w-full" />
      <Skeleton className="h-9 w-full" />
      <Skeleton className="h-20 w-full" />
      <Skeleton className="h-5 w-24" />
      <Skeleton className="h-5 w-24" />
    </div>
  );
  if (variant === "sheet") return content;
  return (
    <div className="sticky top-[calc(4rem+var(--search-bar-h,0px))] hidden h-[calc(100vh-4rem-var(--search-bar-h,0px))] w-64 shrink-0 self-start border-r border-border bg-card lg:block xl:w-72 2xl:w-80">
      {content}
    </div>
  );
}

export function SearchPaginationFallback() {
  return (
    <div className="flex items-center justify-between gap-3" aria-hidden="true">
      <Skeleton className="h-8 w-40" />
      <Skeleton className="h-8 w-48" />
    </div>
  );
}

export function SearchResultsTableFallback() {
  return (
    <div className="space-y-3" aria-hidden="true">
      {Array.from({ length: 4 }).map((_, index) => (
        <div key={index} className="rounded-lg border border-border bg-card p-4">
          <Skeleton className="h-4 w-64" />
          <Skeleton className="mt-3 h-5 w-80 max-w-full" />
          <Skeleton className="mt-3 h-4 w-full" />
          <Skeleton className="mt-2 h-4 w-11/12" />
          <Skeleton className="mt-2 h-4 w-4/5" />
        </div>
      ))}
    </div>
  );
}

export function TransactionResultsFallback() {
  return (
    <div className="overflow-hidden rounded-lg border border-border bg-card" aria-hidden="true">
      <Skeleton className="h-9 w-full rounded-none" />
      {Array.from({ length: 8 }).map((_, index) => (
        <div key={index} className="flex gap-4 border-t border-border px-4 py-3">
          <Skeleton className="h-4 w-10" />
          <Skeleton className="h-4 flex-1" />
          <Skeleton className="hidden h-4 flex-1 md:block" />
          <Skeleton className="h-4 w-14" />
        </div>
      ))}
    </div>
  );
}
