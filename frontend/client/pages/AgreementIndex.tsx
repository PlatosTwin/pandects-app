import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";

import { AgreementIndexOverview } from "@/components/AgreementIndexOverview";
import { PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { Eyebrow } from "@/components/ui/eyebrow";
import { Skeleton } from "@/components/ui/skeleton";
import { useAgreementSummary } from "@/hooks/use-agreement-summary";
import { formatDateValue } from "@/lib/format-utils";

const BROWSE_DEALS_HREF = "/search?mode=transactions";

const formatCount = (value: number | null | undefined) =>
  typeof value === "number" ? value.toLocaleString("en-US") : "—";

function StatCard({
  label,
  value,
  description,
  loading,
}: {
  label: string;
  value: ReactNode;
  description: string;
  loading: boolean;
}) {
  return (
    <div className="rounded-xl border border-border bg-card p-5">
      <Eyebrow>{label}</Eyebrow>
      <div className="mt-2 flex min-h-8 items-center text-2xl font-semibold tabular-nums text-foreground">
        {loading ? (
          <>
            <Skeleton className="h-7 w-28" />
            <span className="sr-only">Loading</span>
          </>
        ) : (
          value
        )}
      </div>
      <p className="mt-1 text-xs text-muted-foreground">{description}</p>
    </div>
  );
}

export default function AgreementIndex() {
  const { summary, isLoading, error } = useAgreementSummary();
  // Every stat card counts processed agreements: the ones users can browse and search.
  const processed = error ? null : (summary?.processed_agreements ?? null);
  const ingested = error ? null : (summary?.agreements ?? null);
  const sections = error ? null : (summary?.sections ?? null);
  const latestFilingDate = error ? null : (summary?.latest_filing_date ?? null);

  return (
    <PageShell
      layout="data"
      title="Agreement Index"
      description={
        <span className="prose-copy">Explore our deal dataset.</span>
      }
      actions={
        <Button asChild>
          <Link to={BROWSE_DEALS_HREF}>
            Browse all deals
            <ArrowRight aria-hidden="true" />
          </Link>
        </Button>
      }
    >
      <section aria-label="Dataset totals" aria-busy={isLoading} className="mb-8">
        <div className="grid gap-4 md:grid-cols-3">
          <StatCard
            label="Agreements"
            value={formatCount(processed)}
            description="Processed and searchable"
            loading={isLoading}
          />
          <StatCard
            label="Sections"
            value={formatCount(sections)}
            description="Indexed across processed agreements"
            loading={isLoading}
          />
          <StatCard
            label="Latest filing"
            value={latestFilingDate ? formatDateValue(latestFilingDate) : "—"}
            description="Most recent filing ingested"
            loading={isLoading}
          />
        </div>
        <p className="mt-3 min-h-5 text-sm text-muted-foreground" aria-live="polite">
          {error
            ? "Dataset totals are unavailable right now."
            : processed !== null && ingested !== null
              ? `${formatCount(processed)} agreements processed and searchable of ${formatCount(ingested)} ingested; the rest are pending processing or verification.`
              : null}
        </p>
      </section>

      <AgreementIndexOverview />
    </PageShell>
  );
}
