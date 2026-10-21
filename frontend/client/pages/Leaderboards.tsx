import { useId, useMemo, useState } from "react";

import { CounselLeaderboardChart } from "@/components/CounselLeaderboardChart";
import { PageShell } from "@/components/PageShell";
import { ChartPanel } from "@/components/ChartPanel";
import { EmptyState, ErrorState } from "@/components/StateMessage";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Eyebrow } from "@/components/ui/eyebrow";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useIsMobile } from "@/hooks/use-mobile";
import { useCounselLeaderboards } from "@/hooks/use-insights";
import {
  CHART_MAX_CATEGORICAL_SERIES,
  CHART_NEUTRAL_SERIES_COLOR,
  CHART_SERIES_PALETTE,
} from "@/lib/chart-palette";
import { formatCompactCurrencyValue } from "@/lib/format-utils";
import type { CounselLeaderboardSide } from "@shared/insights";

type LeaderboardMetric = "deal_count" | "total_transaction_value";
type LeaderboardView = "table" | "chart";

type CounselSectionProps = {
  description: string;
  side: CounselLeaderboardSide;
  title: string;
};

const OTHER_SERIES_KEY = "firm_other";

/**
 * The response ranks up to 15 firms, but the categorical palette has 8 hues.
 * When there are more firms than hues, the top 7 keep their own colors and
 * the rest fold into one neutral "Other" band, so no two firms share a color.
 */
function splitChartFirms<T>(rows: readonly T[]) {
  if (rows.length <= CHART_MAX_CATEGORICAL_SERIES) {
    return { named: rows, other: [] as T[] };
  }
  const namedCount = CHART_MAX_CATEGORICAL_SERIES - 1;
  return { named: rows.slice(0, namedCount), other: rows.slice(namedCount) };
}

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
  const chartFirms = useMemo(() => splitChartFirms(chartRows), [chartRows]);
  const chartSeries = useMemo(() => {
    const series = chartFirms.named.map((row, index) => ({
      key: `firm_${index}`,
      label: row.counsel,
      color: CHART_SERIES_PALETTE[index],
    }));
    if (chartFirms.other.length > 0) {
      const firstRank = chartFirms.named.length + 1;
      const lastRank = firstRank + chartFirms.other.length - 1;
      series.push({
        key: OTHER_SERIES_KEY,
        label: `Other (ranks ${firstRank}–${lastRank})`,
        color: CHART_NEUTRAL_SERIES_COLOR,
      });
    }
    return series;
  }, [chartFirms]);
  const chartData = useMemo(() => {
    const years = new Set<number>();
    chartRows.forEach((row) => {
      row.years.forEach((yearRow) => {
        years.add(yearRow.year);
      });
    });
    const valueFor = (row: (typeof chartRows)[number], year: number) =>
      row.years.find((item) => item.year === year)?.[metric] ?? 0;

    return Array.from(years)
      .sort((a, b) => a - b)
      .map((year) => {
        const nextRow: { year: number } & Record<string, number> = { year };
        chartFirms.named.forEach((row, index) => {
          nextRow[`firm_${index}`] = valueFor(row, year);
        });
        if (chartFirms.other.length > 0) {
          nextRow[OTHER_SERIES_KEY] = chartFirms.other.reduce(
            (sum, row) => sum + valueFor(row, year),
            0,
          );
        }
        return nextRow;
      });
  }, [chartRows, chartFirms, metric]);
  const activeView: LeaderboardView = isMobile ? "table" : view;

  const metricOptions = [
    { value: "deal_count", label: "Deal count" },
    { value: "total_transaction_value", label: "Value" },
  ] as const;
  const viewOptions = [
    { value: "table", label: "Table" },
    { value: "chart", label: "Chart" },
  ] as const;

  return (
    <ChartPanel
      id={descriptionId}
      title={title}
      description={<span className="prose-copy">{description}</span>}
      controls={
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end">
          <div className="flex flex-col gap-1">
            <Eyebrow>Rank by</Eyebrow>
            <SegmentedControl<LeaderboardMetric>
              value={metric}
              onValueChange={setMetric}
              options={metricOptions}
              aria-label={`${title} ranking metric`}
            />
          </div>
          {!isMobile ? (
            <div className="flex flex-col gap-1">
              <Eyebrow>View</Eyebrow>
              <SegmentedControl<LeaderboardView>
                value={view}
                onValueChange={setView}
                options={viewOptions}
                aria-label={`${title} view`}
              />
            </div>
          ) : null}
        </div>
      }
    >
      <div className="space-y-4">
        {activeView === "table" ? (
          <div className="flex flex-col gap-1">
            <Eyebrow>Table window</Eyebrow>
            <SegmentedControl
              value={tableYear}
              onValueChange={setTableYear}
              options={yearOptions}
              aria-label={`${title} table year`}
              size="sm"
              className="flex-wrap self-start"
            />
          </div>
        ) : null}

        {activeView === "table" && tableRows.length === 0 ? (
          <EmptyState
            variant="inline"
            title="No counsel leaderboard data is available for that year yet."
          />
        ) : activeView === "table" ? (
          <div className="overflow-x-auto rounded-lg border border-border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-12 text-right">#</TableHead>
                  <TableHead className="w-[52%] min-w-[12rem] sm:min-w-[14rem]">Firm</TableHead>
                  <TableHead className="text-right">Deals</TableHead>
                  <TableHead className="text-right">Total value</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {tableRows.map((row, index) => (
                  <TableRow key={row.counsel}>
                    <TableCell className="text-right font-mono tabular-nums text-muted-foreground">
                      {index + 1}
                    </TableCell>
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
          <EmptyState variant="inline" title="No counsel leaderboard data is available yet." />
        ) : (
          <>
            <p id={tableId} className="sr-only">
              100 percent stacked area chart showing yearly share of the selected metric across the
              current top {chartRows.length} firms.
              {chartFirms.other.length > 0
                ? ` The top ${chartFirms.named.length} are shown individually; the rest are combined as Other.`
                : null}
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
      </div>
    </ChartPanel>
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
  const { data, isLoading: loading, error, apiError, refetch } = useCounselLeaderboards();

  return (
    <PageShell
      layout="data"
      title="Leaderboards"
      description={
        <span className="prose-copy">
          Canonical counsel rankings for acquirer and target sides. The metric toggle swaps between
          distinct top-15 sets for deal count and total transaction value.
        </span>
      }
    >
      <div className="space-y-6">
        {loading ? <LeaderboardsSkeleton /> : null}

        {!loading && error ? <ErrorState error={apiError ?? error} onRetry={() => void refetch()} /> : null}

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
