import type { MutableRefObject, ReactNode } from "react";
import { Link } from "react-router-dom";
import { SlidersHorizontal, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { cn } from "@/lib/utils";
import { parseSearchMode, type SearchMode } from "@shared/search";

interface SearchHeaderProps {
  searchMode: SearchMode;
  onModeChange: (mode: SearchMode) => void;
  modeButtonRefs: MutableRefObject<Record<SearchMode, HTMLButtonElement | null>>;
  isMobileFiltersOpen: boolean;
  onMobileFiltersOpenChange: (open: boolean) => void;
  authStatus: string;
  signInPath: string;
  mobileSidebar: ReactNode;
}

const MODE_ORDER: readonly SearchMode[] = ["sections", "transactions", "tax"];

export function SearchHeader({
  searchMode,
  onModeChange,
  modeButtonRefs,
  isMobileFiltersOpen,
  onMobileFiltersOpenChange,
  authStatus,
  signInPath,
  mobileSidebar,
}: SearchHeaderProps) {
  return (
    <div className="border-b border-border px-4 py-3 sm:px-8">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex min-w-0 flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center">
          <h1
            id="search-page-title"
            className="shrink-0 text-xl font-semibold tracking-tight text-foreground"
          >
            M&A Search
          </h1>
          <div
            role="radiogroup"
            aria-label="Search mode"
            className="grid min-h-10 w-full grid-cols-3 items-center rounded-lg border border-border bg-muted/40 p-1 sm:w-auto sm:rounded-full"
            onKeyDown={(e) => {
              if (
                !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(e.key)
              ) {
                return;
              }
              e.preventDefault();
              const idx = MODE_ORDER.indexOf(searchMode);
              const nextIdx =
                e.key === "Home"
                  ? 0
                  : e.key === "End"
                    ? MODE_ORDER.length - 1
                    : (idx +
                        (e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 1) +
                        MODE_ORDER.length) %
                      MODE_ORDER.length;
              const nextMode = MODE_ORDER[nextIdx];
              onModeChange(nextMode);
              requestAnimationFrame(() => {
                modeButtonRefs.current[nextMode]?.focus();
              });
            }}
          >
            {MODE_ORDER.map((mode) => (
              <button
                key={mode}
                ref={(node) => {
                  modeButtonRefs.current[mode] = node;
                }}
                type="button"
                role="radio"
                aria-checked={searchMode === mode}
                tabIndex={searchMode === mode ? 0 : -1}
                onClick={() => onModeChange(parseSearchMode(mode))}
                className={cn(
                  "min-h-8 rounded-md px-3 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:rounded-full",
                  searchMode === mode
                    ? "bg-primary/10 text-primary shadow-sm"
                    : "text-muted-foreground hover:text-foreground",
                )}
              >
                {mode === "sections" ? "Sections" : mode === "transactions" ? "Deals" : "Tax"}
              </button>
            ))}
          </div>
        </div>

        <div className="shrink-0 lg:hidden">
          <Sheet open={isMobileFiltersOpen} onOpenChange={onMobileFiltersOpenChange}>
            <SheetTrigger asChild>
              <Button variant="outline" size="sm" className="h-11 w-full gap-2 sm:w-auto">
                <SlidersHorizontal className="h-4 w-4" aria-hidden="true" />
                Filters
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="w-[min(340px,100vw)] max-w-full p-0">
              <SheetTitle className="sr-only">Search filters</SheetTitle>
              <SheetDescription className="sr-only">
                {searchMode === "sections"
                  ? "Filter agreement section results."
                  : searchMode === "tax"
                    ? "Filter tax clause results."
                    : "Filter deal results."}
              </SheetDescription>
              {mobileSidebar}
            </SheetContent>
          </Sheet>
        </div>
      </div>

      {authStatus === "anonymous" && (
        <div className="mt-3">
          <Alert className="py-3">
            <Sparkles className="h-4 w-4" aria-hidden="true" />
            <div className="text-sm font-medium leading-none tracking-tight">Limited mode</div>
            <AlertDescription>
              <div className="grid gap-2">
                <p>
                  Sign in to view section text, open full agreements, unlock higher page sizes, and
                  use the MCP server.
                </p>
                <div>
                  <Button asChild size="sm" variant="outline">
                    <Link to={signInPath}>Sign in to unlock access</Link>
                  </Button>
                </div>
              </div>
            </AlertDescription>
          </Alert>
        </div>
      )}
    </div>
  );
}
