import { cn } from "@/lib/utils";
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
  onPageChange: (page: number) => void;
  onPageSizeChange: (pageSize: number) => void;
  isLoading?: boolean;
}

export function SearchPagination({
  currentPage,
  totalPages,
  pageSize,
  totalCount,
  onPageChange,
  onPageSizeChange,
  isLoading = false,
}: SearchPaginationProps) {
  const formatNumber = (value: number) => new Intl.NumberFormat().format(value);
  const pageSizeOptions = [10, 25, 50, 100];

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
  const endResult = Math.min(currentPage * pageSize, totalCount);

  if (totalCount === 0) {
    return null;
  }

  return (
    <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
      {/* Results per page selector */}
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <span>Results per page:</span>
        <Select
          value={pageSize.toString()}
          onValueChange={(value) => onPageSizeChange(parseInt(value, 10))}
          disabled={isLoading}
        >
          <SelectTrigger className="h-8 w-20 rounded-none border-none border-b border-input bg-transparent focus:border-ring">
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
      <div className="text-sm text-muted-foreground">
        Showing {formatNumber(startResult)} to {formatNumber(endResult)} of{" "}
        {formatNumber(totalCount)} results
      </div>

      {/* Pagination controls */}
      <div className="flex items-center gap-1">
        {/* Previous button */}
        <Button
          onClick={() => onPageChange(currentPage - 1)}
          disabled={currentPage === 1 || isLoading}
          variant="ghost"
          size="sm"
          className="h-8 gap-1 px-3 text-muted-foreground hover:text-foreground"
        >
          <ChevronLeft className="w-4 h-4" />
          <span className="hidden sm:inline">Previous</span>
        </Button>

        {/* Page numbers */}
        <div className="flex items-center gap-1">
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
                disabled={isLoading}
                variant={isActive ? "default" : "ghost"}
                size="icon"
                className={cn(
                  "h-8 w-8",
                  !isActive && "text-muted-foreground hover:text-foreground",
                )}
              >
                {pageNumber}
              </Button>
            );
          })}
        </div>

        {/* Next button */}
        <Button
          onClick={() => onPageChange(currentPage + 1)}
          disabled={currentPage === totalPages || isLoading}
          variant="ghost"
          size="sm"
          className="h-8 gap-1 px-3 text-muted-foreground hover:text-foreground"
        >
          <span className="hidden sm:inline">Next</span>
          <ChevronRight className="w-4 h-4" />
        </Button>
      </div>
    </div>
  );
}
