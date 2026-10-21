import { Link } from "react-router-dom";
import { ArrowLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { AgreementOrigin } from "@/lib/agreement-links";

/**
 * Leading part of the reader header. With an origin: "← Back to Search"
 * (label from `from`). Without one: a "Search ›" breadcrumb, so the reader
 * never pretends the visitor came from results they never saw.
 */
export function ReaderBackLink({
  origin,
  compact,
}: {
  origin: AgreementOrigin | null;
  /** Icon-only on narrow screens. */
  compact: boolean;
}) {
  if (origin) {
    return (
      <Button
        asChild
        variant="ghost"
        size="sm"
        className="-ml-2 h-10 shrink-0 gap-1.5 px-2 text-muted-foreground hover:text-foreground"
      >
        <Link to={origin.href} aria-label={compact ? origin.label : undefined}>
          <ArrowLeft className="h-4 w-4" aria-hidden="true" />
          {compact ? null : <span>{origin.label}</span>}
        </Link>
      </Button>
    );
  }

  return (
    <nav aria-label="Breadcrumb" className="shrink-0">
      <ol className="flex items-center">
        <li className="flex items-center">
          {compact ? (
            <Button
              asChild
              variant="ghost"
              size="icon"
              className="-ml-2 h-10 w-10 text-muted-foreground hover:text-foreground"
            >
              <Link to="/search" aria-label="Search">
                <ArrowLeft className="h-4 w-4" aria-hidden="true" />
              </Link>
            </Button>
          ) : (
            <>
              <Link
                to="/search"
                className="rounded-md px-1 text-sm text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                Search
              </Link>
              <ChevronRight
                className="mx-0.5 h-3.5 w-3.5 text-muted-foreground/70"
                aria-hidden="true"
              />
            </>
          )}
        </li>
      </ol>
    </nav>
  );
}
