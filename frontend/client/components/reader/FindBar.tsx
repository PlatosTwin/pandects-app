import { forwardRef } from "react";
import { ChevronDown, ChevronUp, Search, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { FindInDocumentState } from "@/components/reader/use-find-in-document";

interface FindBarProps {
  query: string;
  onQueryChange: (value: string) => void;
  onClose: () => void;
  find: FindInDocumentState;
  className?: string;
}

/** "3 of 17" while settled; nothing stale while the query is still applying. */
export function findStatusText(query: string, find: FindInDocumentState): string {
  if (!query.trim()) return "";
  if (find.isPending) return "";
  if (find.count === 0) return "No results";
  return `${find.currentIndex + 1} of ${find.count}`;
}

/**
 * Find-in-agreement bar: input · "3 of 17" · previous · next · close.
 * Enter steps forward, Shift+Enter back, Escape closes.
 */
export const FindBar = forwardRef<HTMLInputElement, FindBarProps>(function FindBar(
  { query, onQueryChange, onClose, find, className },
  inputRef,
) {
  const status = findStatusText(query, find);
  const canStep = !find.isPending && find.count > 0;

  return (
    <div
      role="search"
      aria-label="Find in agreement"
      className={cn(
        "flex h-10 min-w-0 items-center gap-0.5 rounded-lg border border-input bg-background pl-2.5 pr-0.5 focus-within:ring-2 focus-within:ring-ring",
        className,
      )}
    >
      <Search className="h-4 w-4 shrink-0 text-muted-foreground" aria-hidden="true" />
      <input
        ref={inputRef}
        type="search"
        value={query}
        onChange={(event) => onQueryChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            if (event.shiftKey) find.previous();
            else find.next();
          } else if (event.key === "Escape") {
            event.preventDefault();
            onClose();
          }
        }}
        placeholder="Find in agreement"
        aria-label="Find in agreement"
        aria-describedby="reader-find-status"
        autoComplete="off"
        spellCheck={false}
        className="h-full min-w-0 flex-1 bg-transparent px-2 text-sm text-foreground outline-none placeholder:text-muted-foreground [&::-webkit-search-cancel-button]:hidden"
      />
      <span
        id="reader-find-status"
        className="shrink-0 whitespace-nowrap px-1 text-xs tabular-nums text-muted-foreground"
        aria-live="polite"
      >
        {status}
        {query.trim() && find.isPending ? <span className="sr-only">Searching</span> : null}
      </span>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className="h-8 w-8 shrink-0 text-muted-foreground hover:text-foreground"
        onClick={find.previous}
        disabled={!canStep}
        aria-label="Previous match"
        title="Previous match (Shift+Enter)"
      >
        <ChevronUp aria-hidden="true" />
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className="h-8 w-8 shrink-0 text-muted-foreground hover:text-foreground"
        onClick={find.next}
        disabled={!canStep}
        aria-label="Next match"
        title="Next match (Enter)"
      >
        <ChevronDown aria-hidden="true" />
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className="h-8 w-8 shrink-0 text-muted-foreground hover:text-foreground"
        onClick={onClose}
        aria-label="Close find"
        title="Close (Esc)"
      >
        <X aria-hidden="true" />
      </Button>
    </div>
  );
});
