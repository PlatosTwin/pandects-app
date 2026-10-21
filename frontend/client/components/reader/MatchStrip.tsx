import { Link } from "react-router-dom";
import { ChevronLeft, ChevronRight, Loader2, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface MatchStripProps {
  /** Short label of the current match, e.g. "§7.08". */
  label: string;
  /** 0-based position of the current match in `total`, or -1 when unknown. */
  index: number;
  total: number;
  /** More matches exist than were loaded (the search page size was capped). */
  hasMore?: boolean;
  onPrevious?: () => void;
  onNext?: () => void;
  onDismiss: () => void;
  /** Return to the originating results (single-match fallback). */
  backHref?: string | null;
  /** A jump to the match is in progress (long agreements take a moment). */
  jumping?: boolean;
  className?: string;
}

/**
 * "Matched §7.08 · 2 of 5 ‹ ›" — steps through every match of the
 * originating search in this agreement. Dismissible.
 */
export function MatchStrip({
  label,
  index,
  total,
  hasMore = false,
  onPrevious,
  onNext,
  onDismiss,
  backHref,
  jumping = false,
  className,
}: MatchStripProps) {
  const canStep = total > 1 && Boolean(onPrevious && onNext);
  const position =
    total > 1 && index >= 0 ? `${index + 1} of ${total}${hasMore ? "+" : ""}` : null;

  return (
    <div
      role="region"
      aria-label="Search matches in this agreement"
      className={cn(
        "flex h-10 items-center gap-1 rounded-lg border border-primary/20 bg-card pl-3 pr-1 text-sm shadow-raised",
        className,
      )}
    >
      <span className="flex min-w-0 flex-1 items-center gap-1.5" aria-live="polite">
        {jumping ? (
          <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-primary" aria-hidden="true" />
        ) : (
          <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-primary" aria-hidden="true" />
        )}
        <span className="shrink-0 font-medium text-primary">
          {jumping ? "Jumping to" : "Matched"}
        </span>
        <span className="min-w-0 truncate font-medium text-foreground" title={label}>
          {label}
          {jumping ? "…" : ""}
        </span>
        {position ? (
          <span className="shrink-0 tabular-nums text-muted-foreground">· {position}</span>
        ) : null}
      </span>
      {canStep ? (
        <>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-8 w-8 shrink-0 text-muted-foreground hover:text-foreground"
            onClick={onPrevious}
            aria-label="Previous matched section"
          >
            <ChevronLeft aria-hidden="true" />
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="h-8 w-8 shrink-0 text-muted-foreground hover:text-foreground"
            onClick={onNext}
            aria-label="Next matched section"
          >
            <ChevronRight aria-hidden="true" />
          </Button>
        </>
      ) : backHref ? (
        <Button
          asChild
          variant="ghost"
          size="sm"
          className="h-8 shrink-0 px-2 text-muted-foreground hover:text-foreground"
        >
          <Link to={backHref}>Back to matches</Link>
        </Button>
      ) : null}
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className="h-8 w-8 shrink-0 text-muted-foreground hover:text-foreground"
        onClick={onDismiss}
        aria-label="Dismiss match navigation"
      >
        <X aria-hidden="true" />
      </Button>
    </div>
  );
}
