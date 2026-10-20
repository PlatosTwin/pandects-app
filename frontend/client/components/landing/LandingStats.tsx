import type { ReactNode } from "react";
import { useHydrated } from "@/components/landing/use-hydrated";
import { Skeleton } from "@/components/ui/skeleton";
import { useAgreementSummary } from "@/hooks/use-agreement-summary";
import { formatNumber } from "@/lib/format-utils";
import { cn } from "@/lib/utils";

const LATEST_FILING_FORMATTER = new Intl.DateTimeFormat("en-US", {
  year: "numeric",
  month: "short",
});

function formatLatestFiling(value: string | null): string | null {
  if (!value) return null;
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return null;
  return LATEST_FILING_FORMATTER.format(dt);
}

const PILL =
  "inline-flex h-6 items-center whitespace-nowrap rounded-full border border-border bg-card px-2.5 text-xs text-muted-foreground";

function StatPill({ children }: { children: ReactNode }) {
  return <li className={PILL}>{children}</li>;
}

/**
 * Corpus facts under the landing search box. The row has a fixed minimum
 * height (two pill lines below md, one at md+), so it never moves the content
 * below it: the live pills show a skeleton until the summary loads and are
 * simply left out when the summary fails (for example while the API is
 * locked). The static facts always render first, so the row is never empty
 * and nothing in it moves when the live pills drop out.
 */
export function LandingStats({ className }: { className?: string }) {
  // The summary query is disabled during prerender, so render the skeleton
  // on the server and on the first client paint to keep hydration in sync.
  const hydrated = useHydrated();

  const { summary, isLoading } = useAgreementSummary();
  const loading = !hydrated || isLoading;
  const latestFiling = summary
    ? formatLatestFiling(summary.latest_filing_date)
    : null;

  return (
    <ul
      aria-label="Dataset facts"
      aria-busy={loading || undefined}
      className={cn(
        "flex min-h-[3.5rem] flex-wrap content-start items-center gap-2 md:min-h-6",
        className,
      )}
    >
      <StatPill>Sourced from EDGAR</StatPill>
      <StatPill>Updated monthly</StatPill>
      {loading ? (
        <>
          <li aria-hidden="true">
            <Skeleton className="h-6 w-44 rounded-full" />
          </li>
          <li aria-hidden="true">
            <Skeleton className="h-6 w-36 rounded-full" />
          </li>
        </>
      ) : summary ? (
        <>
          <StatPill>
            <span className="mr-1 font-medium text-foreground">
              {formatNumber(summary.agreements)}
            </span>
            agreements indexed
          </StatPill>
          {latestFiling ? (
            <StatPill>Latest filing {latestFiling}</StatPill>
          ) : null}
        </>
      ) : null}
    </ul>
  );
}
