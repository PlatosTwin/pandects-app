import { Link } from "react-router-dom";

import { buildAgreementHref } from "@/lib/agreement-links";
import { useHydrated } from "@/components/landing/use-hydrated";
import { ArrowRight } from "lucide-react";
import { useRecentAgreements } from "@/hooks/use-recent-agreements";
import { Skeleton } from "@/components/ui/skeleton";
import { eyebrowClassName } from "@/components/ui/eyebrow";
import { trackEvent } from "@/lib/analytics";
import { cn } from "@/lib/utils";

interface FeaturedAgreementsProps {
  limit?: number;
  className?: string;
}

type RecentAgreement = ReturnType<
  typeof useRecentAgreements
>["results"][number];

const ALL_DEALS_PATH = "/search?mode=transactions";

const ROW = "flex min-h-[3.75rem] items-center gap-3 px-4 py-2.5";

/**
 * The most recently filed agreements as a short list of rows (target,
 * acquirer, year), each opening the agreement. Shown on the landing page so
 * first-time visitors see real, current data.
 *
 * SSR-safe: the query is disabled during prerender, so the server and the
 * first client paint both render the skeleton rows; after hydration the rows
 * fill in. If the fetch fails (for example while the API is locked) the
 * section renders nothing; the landing page works without it.
 */
export function FeaturedAgreements({
  limit = 3,
  className,
}: FeaturedAgreementsProps) {
  const hydrated = useHydrated();

  const { results, isLoading, error } = useRecentAgreements(limit);
  const loading = !hydrated || isLoading;

  if (!loading && (error || results.length === 0)) {
    return null;
  }

  return (
    <section
      aria-labelledby="featured-agreements-heading"
      className={cn("min-w-0", className)}
    >
      <div className="mb-2 flex h-6 items-center justify-between gap-3">
        <h2 id="featured-agreements-heading" className={eyebrowClassName}>
          Latest agreements
        </h2>
        <Link
          to={ALL_DEALS_PATH}
          className="inline-flex min-h-11 items-center rounded-md text-sm font-medium text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:min-h-0"
          onClick={() =>
            trackEvent("landing_featured_agreements_view_all", {
              to_path: ALL_DEALS_PATH,
            })
          }
        >
          Browse all deals
        </Link>
      </div>

      <ul
        className="divide-y divide-border overflow-hidden rounded-xl border border-border bg-card"
        aria-busy={loading || undefined}
      >
        {loading
          ? Array.from({ length: limit }).map((_, index) => (
              <li key={`skeleton-${index}`} aria-hidden="true" className={ROW}>
                <div className="min-w-0 flex-1">
                  <Skeleton className="h-4 w-3/5" />
                  <Skeleton className="mt-2 h-3 w-2/5" />
                </div>
                <Skeleton className="h-3 w-10" />
              </li>
            ))
          : results.map((result, position) => (
              <li key={result.agreement_uuid}>
                <FeaturedAgreementRow result={result} position={position} />
              </li>
            ))}
      </ul>
    </section>
  );
}

function FeaturedAgreementRow({
  result,
  position,
}: {
  result: RecentAgreement;
  position: number;
}) {
  const target = result.target?.trim() || "Unknown target";
  const acquirer = result.acquirer?.trim() || "Unknown acquirer";

  return (
    <Link
      to={buildAgreementHref({ agreementUuid: result.agreement_uuid, from: "/" })}
      onClick={() =>
        trackEvent("landing_featured_agreement_click", {
          position,
          year: result.year ?? undefined,
          agreement_uuid: result.agreement_uuid,
        })
      }
      className={cn(
        ROW,
        "group transition-colors hover:bg-accent/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring",
      )}
    >
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-foreground">{target}</p>
        <p className="truncate text-xs text-muted-foreground">
          acquired by {acquirer}
        </p>
      </div>
      {result.year ? (
        <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
          {result.year}
        </span>
      ) : null}
      <ArrowRight
        className="h-4 w-4 shrink-0 text-muted-foreground transition-colors group-hover:text-foreground"
        aria-hidden="true"
      />
    </Link>
  );
}
