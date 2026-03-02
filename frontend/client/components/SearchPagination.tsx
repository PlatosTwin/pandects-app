import { cn } from "@/lib/utils";
import { formatNumber } from "@/lib/format-utils";
import { ChevronLeft, ChevronRight } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";

interface SearchPaginationProps {
  currentPage: number;
  totalPages: number;
  pageSize: number;
  totalCount: number;
  totalCountIsApproximate?: boolean;
  hasNext: boolean;
  hasPrev: boolean;
  onPageChange: (page: number) => void;
  onPageSizeChange: (pageSize: number) => void;
  isLoading?: boolean;
  isLimited?: boolean;
}

export function SearchPagination({
  currentPage,
  totalPages,
  pageSize,
  totalCount,
  totalCountIsApproximate = false,
  hasNext,
  hasPrev,
  onPageChange,
  onPageSizeChange,
  isLoading = false,
  isLimited = false,
}: SearchPaginationProps) {
  const pageSizeOptions = [10, 25, 50, 100];
  const navigationDisabled = isLoading || isLimited;

  const getVisiblePages = () => {
    const delta = 2; // Number of pages to show on each side of current page
    const range = [];
    const rangeWithDots = [];

    for (
      let i = Math.max(2, currentPage - delta);
      i <= Math.min(totalPages - 1, currentPage + delta);
      i++
    ) {
      range.push(i);
    }

    if (currentPage - delta > 2) {
      rangeWithDots.push(1, "...");
    } else {
      rangeWithDots.push(1);
    }

    rangeWithDots.push(...range);

    if (currentPage + delta < totalPages - 1) {
      rangeWithDots.push("...", totalPages);
    } else if (totalPages > 1) {
      rangeWithDots.push(totalPages);
    }

    return rangeWithDots;
  };

  const visiblePages = getVisiblePages();
  const startResult = (currentPage - 1) * pageSize + 1;
  const endResult = totalCountIsApproximate
    ? (currentPage - 1) * pageSize + (hasNext ? pageSize : Math.min(pageSize, totalCount))
    : Math.min(currentPage * pageSize, totalCount);

  if (totalCount === 0) {
    return null;
  }

  return (
    <nav
      className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4"
      aria-label="Search results pagination"
    >
      {/* Results per page selector */}
      <div className="hidden items-center gap-2 text-sm text-muted-foreground sm:flex">
        <span id="results-per-page-label">Results per page:</span>
        <Select
          value={pageSize.toString()}
          onValueChange={(value) => onPageSizeChange(parseInt(value, 10))}
          disabled={navigationDisabled}
        >
          <SelectTrigger
            className="h-8 w-20 rounded-none border-none border-b border-input bg-transparent focus:border-ring"
            aria-label="Results per page"
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {pageSizeOptions.map((size) => (
              <SelectItem key={size} value={size.toString()}>
                {size}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Results info */}
      <div className="text-sm text-muted-foreground" aria-live="polite">
        Showing {formatNumber(startResult)} to {formatNumber(endResult)} of{" "}
        {totalCountIsApproximate ? "approx. " : ""}
        {formatNumber(totalCount)} results
      </div>

      {/* Jump to page input (desktop only) */}
      <div className="hidden items-center gap-2 text-sm sm:flex">
        {!totalCountIsApproximate && (
          <>
            <label htmlFor="jump-to-page" className="text-muted-foreground">
              Go to:
            </label>
            <input
              id="jump-to-page"
              type="number"
              min={1}
              max={totalPages}
              defaultValue={currentPage}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  const page = parseInt((e.target as HTMLInputElement).value, 10);
                  if (page >= 1 && page <= totalPages && !navigationDisabled) {
                    onPageChange(page);
                    (e.target as HTMLInputElement).value = "";
                  }
                }
              }}
              className="h-8 w-16 rounded-md border border-input bg-background px-2 text-sm text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
              disabled={navigationDisabled}
              aria-label="Jump to page number"
            />
          </>
        )}
        {totalCountIsApproximate && (
          <span className="text-muted-foreground">Page {formatNumber(currentPage)}</span>
        )}
      </div>

      {/* Pagination controls */}
      <div className="flex w-full items-center gap-2 sm:w-auto sm:gap-1">
        {/* Mobile buttons */}
        <div className="flex w-full gap-2 sm:hidden">
          <Button
            onClick={() => onPageChange(currentPage - 1)}
            disabled={!hasPrev || navigationDisabled}
            variant="outline"
            size="sm"
            className="h-11 flex-1"
            aria-label="Previous page"
          >
            <ChevronLeft className="w-4 h-4" aria-hidden="true" />
            Previous
          </Button>
          <Button
            onClick={() => onPageChange(currentPage + 1)}
            disabled={!hasNext || navigationDisabled}
            variant="outline"
            size="sm"
            className="h-11 flex-1"
            aria-label="Next page"
          >
            Next
            <ChevronRight className="w-4 h-4" aria-hidden="true" />
          </Button>
        </div>

        {/* Desktop buttons */}
        <div className="hidden items-center gap-1 sm:flex">
          <Button
            onClick={() => onPageChange(currentPage - 1)}
            disabled={!hasPrev || navigationDisabled}
            variant="ghost"
            size="sm"
            className="h-8 gap-1 px-3 text-muted-foreground hover:text-foreground"
            aria-label="Previous page"
          >
            <ChevronLeft className="w-4 h-4" aria-hidden="true" />
            <span className="hidden sm:inline">Previous</span>
          </Button>

          {/* Page numbers */}
          <div className="hidden items-center gap-1 sm:flex">
            {!totalCountIsApproximate && (
              <>
                {visiblePages.map((page, index) => {
                  if (page === "...") {
                    return (
                      <span
                        key={`ellipsis-${index}`}
                        className="px-2 py-1 text-sm text-muted-foreground"
                      >
                        ...
                      </span>
                    );
                  }

                  const pageNumber = page as number;
                  const isActive = pageNumber === currentPage;

                  return (
                    <Button
                      key={pageNumber}
                      onClick={() => onPageChange(pageNumber)}
                      disabled={navigationDisabled}
                      variant={isActive ? "default" : "ghost"}
                      size="icon"
                      className={cn(
                        "h-8 w-8",
                        !isActive && "text-muted-foreground hover:text-foreground",
                      )}
                      aria-label={`Page ${pageNumber}`}
                      aria-current={isActive ? "page" : undefined}
                    >
                      {pageNumber}
                    </Button>
                  );
                })}
              </>
            )}
          </div>

          <Button
            onClick={() => onPageChange(currentPage + 1)}
            disabled={!hasNext || navigationDisabled}
            variant="ghost"
            size="sm"
            className="h-8 gap-1 px-3 text-muted-foreground hover:text-foreground"
            aria-label="Next page"
          >
            <span className="hidden sm:inline">Next</span>
            <ChevronRight className="w-4 h-4" aria-hidden="true" />
          </Button>
        </div>
      </div>
    </nav>
  );
}
