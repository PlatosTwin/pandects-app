import { lazy } from "react";
import { Skeleton } from "@/components/ui/skeleton";

/**
 * Lazy-loaded search result components and their skeleton fallbacks.
 *
 * Kept in a separate module so the main Search.tsx route shell stays focused
 * on orchestration. Adding a new search mode means adding both the lazy
 * component import and its fallback here.
 */

export const SearchPagination = lazy(() =>
  import("@/components/SearchPagination").then((mod) => ({
    default: mod.SearchPagination,
  })),
);
export const SearchResultsTable = lazy(() =>
  import("@/components/SearchResultsTable").then((mod) => ({
    default: mod.SearchResultsTable,
  })),
);
export const SearchSidebar = lazy(() =>
  import("@/components/SearchSidebar").then((mod) => ({
    default: mod.SearchSidebar,
  })),
);
export const TransactionResultsList = lazy(() =>
  import("@/components/TransactionResultsList").then((mod) => ({
    default: mod.TransactionResultsList,
  })),
);
export const TaxClauseResultsList = lazy(() =>
  import("@/components/TaxClauseResultsList").then((mod) => ({
    default: mod.TaxClauseResultsList,
  })),
);

export function SearchSidebarFallback({
  variant = "sidebar",
}: {
  variant?: "sidebar" | "sheet";
}) {
  const content = (
    <div className="space-y-5 p-4">
      <Skeleton className="h-5 w-28" />
      <Skeleton className="h-10 w-full" />
      <Skeleton className="h-24 w-full" />
      <Skeleton className="h-24 w-full" />
      <Skeleton className="h-24 w-full" />
    </div>
  );

  if (variant === "sheet") {
    return <div className="h-full overflow-y-auto">{content}</div>;
  }

  return (
    <div className="hidden h-screen w-80 border-r border-b border-border bg-card lg:block">
      {content}
    </div>
  );
}

export function SearchPaginationFallback() {
  return (
    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
      <Skeleton className="h-9 w-56" />
      <Skeleton className="h-9 w-48" />
    </div>
  );
}

export function SearchResultsTableFallback() {
  return (
    <div className="space-y-4">
      {Array.from({ length: 4 }).map((_, index) => (
        <div
          key={index}
          className="rounded-lg border border-border bg-card p-4 shadow-sm"
        >
          <Skeleton className="h-5 w-48" />
          <Skeleton className="mt-3 h-4 w-full" />
          <Skeleton className="mt-2 h-4 w-5/6" />
          <Skeleton className="mt-4 h-20 w-full" />
        </div>
      ))}
    </div>
  );
}

export function TransactionResultsFallback() {
  return (
    <div className="space-y-4">
      {Array.from({ length: 3 }).map((_, index) => (
        <div
          key={index}
          className="rounded-lg border border-border bg-card p-5 shadow-sm"
        >
          <Skeleton className="h-6 w-72" />
          <Skeleton className="mt-3 h-4 w-full" />
          <Skeleton className="mt-2 h-4 w-5/6" />
          <Skeleton className="mt-5 h-24 w-full" />
        </div>
      ))}
    </div>
  );
}
