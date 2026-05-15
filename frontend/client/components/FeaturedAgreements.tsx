import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { useRecentAgreements } from "@/hooks/use-recent-agreements";
import { Skeleton } from "@/components/ui/skeleton";
import { trackEvent } from "@/lib/analytics";
import { cn } from "@/lib/utils";

interface FeaturedAgreementsProps {
  limit?: number;
  className?: string;
}

const FILING_DATE_FORMATTER = new Intl.DateTimeFormat("en-US", {
  year: "numeric",
  month: "short",
});

function formatFilingDate(value: string | null): string | null {
  if (!value) return null;
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return null;
  return FILING_DATE_FORMATTER.format(dt);
}

/**
 * Compact rail of the most-recently-filed agreements, each clickable through
 * to its detail page. Rendered on the landing page so first-time visitors see
 * real, current data — not just a marketing card.
 *
 * SSR-safe: returns an empty placeholder during prerender (useRecentAgreements
 * is gated on IS_SERVER_RENDER) and a skeleton while loading on hydration. If
 * the fetch fails we render nothing rather than an error UI; the landing page
 * still works without this section.
 */
export function FeaturedAgreements({
  limit = 4,
  className,
}: FeaturedAgreementsProps) {
  // Defer rendering until after hydration. Prerender returns null (the query
  // is gated on IS_SERVER_RENDER), so anything we render on the very first
  // client paint would diverge from the SSR DOM and trigger a hydration
  // mismatch warning. One-shot effect flips us into the live state.
  const [hydrated, setHydrated] = useState(false);
  useEffect(() => setHydrated(true), []);

  const { results, isLoading, error } = useRecentAgreements(limit);

  if (!hydrated) return null;
  if (error || (!isLoading && results.length === 0)) {
    return null;
  }

  return (
    <section
      aria-labelledby="featured-agreements-heading"
      className={cn("w-full", className)}
    >
      <div className="mb-3 flex items-baseline justify-between gap-3">
        <h2
          id="featured-agreements-heading"
          className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground"
        >
          Latest agreements
        </h2>
        <Link
          to="/agreement-index"
          className="text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
          onClick={() =>
            trackEvent("landing_featured_agreements_view_all", {
              to_path: "/agreement-index",
            })
          }
        >
          View all →
        </Link>
      </div>

      <ul
        className="grid gap-2 sm:grid-cols-2"
        aria-busy={isLoading || undefined}
      >
        {isLoading
          ? Array.from({ length: limit }).map((_, index) => (
              <li key={`skeleton-${index}`}>
                <FeaturedAgreementSkeleton />
              </li>
            ))
          : results.map((result, position) => (
              <li key={result.agreement_uuid}>
                <FeaturedAgreementCard result={result} position={position} />
              </li>
            ))}
      </ul>
    </section>
  );
}

function FeaturedAgreementSkeleton() {
  return (
    <div className="rounded-lg border border-border bg-background/60 p-3">
      <Skeleton className="h-4 w-3/4" />
      <Skeleton className="mt-2 h-3 w-2/3" />
      <Skeleton className="mt-3 h-3 w-24" />
    </div>
  );
}

function FeaturedAgreementCard({
  result,
  position,
}: {
  result: NonNullable<ReturnType<typeof useRecentAgreements>["results"]>[number];
  position: number;
}) {
  const filed = formatFilingDate(result.filing_date);
  const target = result.target?.trim() || "Unknown target";
  const acquirer = result.acquirer?.trim() || "Unknown acquirer";

  return (
    <Link
      to={`/agreements/${result.agreement_uuid}`}
      onClick={() =>
        trackEvent("landing_featured_agreement_click", {
          position,
          year: result.year ?? undefined,
          agreement_uuid: result.agreement_uuid,
        })
      }
      className="group block h-full rounded-lg border border-border bg-background/70 p-3 text-left transition-colors hover:border-border hover:bg-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium text-foreground">
            {target}
          </p>
          <p className="truncate text-xs text-muted-foreground">
            acquired by {acquirer}
          </p>
        </div>
        <ArrowRight
          className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground/60 transition-colors group-hover:text-foreground"
          aria-hidden="true"
        />
      </div>
      {filed ? (
        <p className="mt-2 text-[11px] font-medium uppercase tracking-[0.08em] text-muted-foreground/80">
          Filed {filed}
        </p>
      ) : null}
    </Link>
  );
}
