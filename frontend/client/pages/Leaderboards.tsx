import { useId, useMemo, useState } from "react";

import { CounselLeaderboardChart } from "@/components/CounselLeaderboardChart";
import { PageShell } from "@/components/PageShell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { useIsMobile } from "@/hooks/use-mobile";
import { useCounselLeaderboards } from "@/hooks/use-insights";
import { CHART_SERIES_PALETTE } from "@/lib/chart-palette";
import { formatCompactCurrencyValue } from "@/lib/format-utils";
import type { CounselLeaderboardSide } from "@shared/insights";

type LeaderboardMetric = "deal_count" | "total_transaction_value";
type LeaderboardView = "table" | "chart";

type CounselSectionProps = {
  description: string;
  side: CounselLeaderboardSide;
  title: string;
};

const LEADERBOARD_COLORS = CHART_SERIES_PALETTE;

function formatDealCount(value: number) {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0,
  }).format(value);
}

function formatTransactionValue(value: number) {
  return formatCompactCurrencyValue(value);
}

function LeaderboardSection({ description, side, title }: CounselSectionProps) {
  const isMobile = useIsMobile();
  const [metric, setMetric] = useState<LeaderboardMetric>("deal_count");
  const [view, setView] = useState<LeaderboardView>("table");
  const [tableYear, setTableYear] = useState<string>("all");
  const descriptionId = useId();
  const tableId = useId();

  // Offer the five most recent years that actually appear in the response,
  // rather than the last five calendar years (which can have no data yet).
  const yearOptions = useMemo(
    () => [
      ...side.annual
        .map((entry) => entry.year)
        .sort((a, b) => b - a)
        .slice(0, 5)
        .map((year) => ({
          value: String(year),
          label: String(year),
        })),
      { value: "all", label: "All time" },
    ],
    [side.annual],
  );
  const selectedAnnual = useMemo(
    () => side.annual.find((entry) => String(entry.year) === tableYear) ?? null,
    [side.annual, tableYear],
  );
  const tableRows =
    tableYear === "all"
      ? metric === "deal_count"
        ? side.top_by_count
        : side.top_by_value
      : metric === "deal_count"
        ? selectedAnnual?.top_by_count ?? []
        : selectedAnnual?.top_by_value ?? [];
  const chartRows = metric === "deal_count" ? side.top_by_count : side.top_by_value;
  const chartSeries = useMemo(
    () =>
      chartRows.map((row, index) => ({
        key: `firm_${index}`,
        label: row.counsel,
        color: LEADERBOARD_COLORS[index % LEADERBOARD_COLORS.length],
      })),
    [chartRows],
  );
  const chartData = useMemo(() => {
    const years = new Set<number>();
    chartRows.forEach((row) => {
      row.years.forEach((yearRow) => {
        years.add(yearRow.year);
      });
    });

    return Array.from(years)
      .sort((a, b) => a - b)
      .map((year) => {
        const nextRow: { year: number } & Record<string, number> = { year };
        chartRows.forEach((row, index) => {
          const yearRow = row.years.find((item) => item.year === year);
          nextRow[`firm_${index}`] = yearRow ? yearRow[metric] : 0;
        });
        return nextRow;
      });
  }, [chartRows, metric]);
  const activeView: LeaderboardView = isMobile ? "table" : view;

  return (
    <section aria-labelledby={descriptionId} className="space-y-4">
      <Card variant="subtle">
        <CardHeader className="gap-4 md:flex-row md:items-end md:justify-between">
          <div className="space-y-2">
            <h2 id={descriptionId} className="text-xl font-semibold leading-none tracking-tight">
              {title}
            </h2>
            <CardDescription className="max-w-3xl prose-copy text-sm sm:text-base">
              {description}
            </CardDescription>
          </div>
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
            <div className="flex flex-col gap-1">
              <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Rank by
              </span>
              <ToggleGroup
                type="single"
                value={metric}
                onValueChange={(value) => {
                  if (value === "deal_count" || value === "total_transaction_value") {
                    setMetric(value);
                  }
                }}
                variant="outline"
                aria-label={`${title} ranking metric`}
                className="justify-start"
              >
                <ToggleGroupItem value="deal_count" aria-label="Rank by deal count">
                  Deal count
                </ToggleGroupItem>
                <ToggleGroupItem
                  value="total_transaction_value"
                  aria-label="Rank by transaction value"
                >
                  Value
                </ToggleGroupItem>
              </ToggleGroup>
            </div>
            {!isMobile ? (
              <div className="flex flex-col gap-1">
                <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  View
                </span>
                <ToggleGroup
                  type="single"
                  value={view}
                  onValueChange={(value) => {
                    if (value === "table" || value === "chart") {
                      setView(value);
                    }
                  }}
                  variant="outline"
                  aria-label={`${title} view`}
                  className="justify-start"
                >
                  <ToggleGroupItem value="table" aria-label="Show table">
                    Table
                  </ToggleGroupItem>
                  <ToggleGroupItem value="chart" aria-label="Show chart">
                    Chart
                  </ToggleGroupItem>
                </ToggleGroup>
              </div>
            ) : null}
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {activeView === "table" ? (
            <div className="flex flex-col gap-2">
              <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Table window
              </span>
              <RadioGroup
                value={tableYear}
                onValueChange={setTableYear}
                aria-label={`${title} table year`}
                className="flex flex-wrap gap-3"
              >
                {yearOptions.map((option) => (
                  <label
                    key={option.value}
                    className="flex items-center gap-2 rounded-full border border-border bg-background/80 px-3 py-1.5 text-sm text-foreground"
                  >
                    <RadioGroupItem value={option.value} id={`${tableId}-${option.value}`} />
                    <span>{option.label}</span>
                  </label>
                ))}
              </RadioGroup>
            </div>
          ) : null}

          {activeView === "table" && tableRows.length === 0 ? (
            <div className="rounded-lg border border-dashed border-border bg-background/70 px-4 py-8 text-sm text-muted-foreground">
              No counsel leaderboard data is available for that year yet.
            </div>
          ) : activeView === "table" ? (
            <div className="overflow-x-auto rounded-lg border border-border bg-background/80">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[52%] min-w-[12rem] sm:min-w-[14rem]">Firm</TableHead>
                    <TableHead className="text-right">Deals</TableHead>
                    <TableHead className="text-right">Total value</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {tableRows.map((row) => (
                    <TableRow key={row.counsel}>
                      <TableCell className="font-medium">{row.counsel}</TableCell>
                      <TableCell className="text-right font-mono tabular-nums">
                        {formatDealCount(row.deal_count)}
                      </TableCell>
                      <TableCell className="text-right font-mono tabular-nums">
                        {formatTransactionValue(row.total_transaction_value)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          ) : chartData.length === 0 ? (
            <div className="rounded-lg border border-dashed border-border bg-background/70 px-4 py-8 text-sm text-muted-foreground">
              No counsel leaderboard data is available yet.
            </div>
          ) : (
            <>
              <p id={tableId} className="sr-only">
                100 percent stacked area chart showing yearly share of the selected metric across the
                current top 15 firms.
              </p>
              <CounselLeaderboardChart
                data={chartData}
                describedBy={descriptionId}
                metricLabel={metric === "deal_count" ? "deal count" : "transaction value"}
                series={chartSeries}
                tableId={tableId}
                valueFormatter={
                  metric === "deal_count" ? formatDealCount : formatTransactionValue
                }
              />
            </>
          )}
        </CardContent>
      </Card>
    </section>
  );
}

function LeaderboardsSkeleton() {
  return (
    <div className="space-y-6">
      {[0, 1].map((index) => (
        <Card key={index} variant="subtle">
          <CardHeader>
            <Skeleton className="h-7 w-48" />
            <Skeleton className="h-4 w-full max-w-3xl" />
            <Skeleton className="h-4 w-full max-w-2xl" />
          </CardHeader>
          <CardContent className="space-y-3">
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-64 w-full" />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

export default function Leaderboards() {
  const { data, isLoading: loading, error } = useCounselLeaderboards();

  return (
    <PageShell
      title="Leaderboards"
      size="xl"
      subtitle={
        <span className="prose-copy">
          Canonical counsel rankings for acquirer and target sides. The metric toggle swaps between
          distinct top-15 sets for deal count and total transaction value.
        </span>
      }
    >
      <div className="space-y-6">
        {loading ? <LeaderboardsSkeleton /> : null}

        {!loading && error ? (
          <Card variant="subtle">
            <CardHeader>
              <CardTitle className="text-xl">Unavailable</CardTitle>
              <CardDescription>{error}</CardDescription>
            </CardHeader>
          </Card>
        ) : null}

        {!loading && !error && data ? (
          <>
            <LeaderboardSection
              title="Buy-Side Counsel"
              description="Top acquirer-side counsel across public-eligible agreements."
              side={data.buy_side}
            />
            <LeaderboardSection
              title="Sell-Side Counsel"
              description="Top target-side counsel across public-eligible agreements."
              side={data.sell_side}
            />
          </>
        ) : null}
      </div>
    </PageShell>
  );
}
