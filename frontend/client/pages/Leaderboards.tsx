import { useEffect, useId, useMemo, useState } from "react";

import { CounselLeaderboardChart } from "@/components/CounselLeaderboardChart";
import { PageShell } from "@/components/PageShell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
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
import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";
import { formatCompactCurrencyValue } from "@/lib/format-utils";

type LeaderboardMetric = "deal_count" | "total_transaction_value";
type LeaderboardView = "table" | "chart";

type CounselLeaderboardYear = {
  year: number;
  deal_count: number;
  total_transaction_value: number;
};

type CounselLeaderboardEntry = {
  counsel: string;
  deal_count: number;
  total_transaction_value: number;
  years: CounselLeaderboardYear[];
};

type CounselLeaderboardSide = {
  top_by_count: CounselLeaderboardEntry[];
  top_by_value: CounselLeaderboardEntry[];
};

type CounselLeaderboardResponse = {
  buy_side: CounselLeaderboardSide;
  sell_side: CounselLeaderboardSide;
};

type CounselSectionProps = {
  description: string;
  side: CounselLeaderboardSide;
  title: string;
};

const LEADERBOARD_COLORS = [
  "hsl(212 93% 50%)",
  "hsl(170 84% 36%)",
  "hsl(35 92% 52%)",
  "hsl(0 84% 60%)",
  "hsl(196 83% 42%)",
  "hsl(262 83% 58%)",
  "hsl(142 71% 45%)",
  "hsl(24 95% 53%)",
  "hsl(221 83% 53%)",
  "hsl(12 76% 61%)",
  "hsl(187 72% 41%)",
  "hsl(271 81% 56%)",
  "hsl(43 96% 56%)",
  "hsl(151 55% 42%)",
  "hsl(215 70% 59%)",
];

function formatDealCount(value: number) {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0,
  }).format(value);
}

function formatTransactionValue(value: number) {
  return formatCompactCurrencyValue(value);
}

function LeaderboardSection({ description, side, title }: CounselSectionProps) {
  const [metric, setMetric] = useState<LeaderboardMetric>("deal_count");
  const [view, setView] = useState<LeaderboardView>("table");
  const descriptionId = useId();
  const tableId = useId();

  const rows = metric === "deal_count" ? side.top_by_count : side.top_by_value;
  const chartSeries = useMemo(
    () =>
      rows.map((row, index) => ({
        key: `firm_${index}`,
        label: row.counsel,
        color: LEADERBOARD_COLORS[index % LEADERBOARD_COLORS.length],
      })),
    [rows],
  );
  const chartData = useMemo(() => {
    const years = new Set<number>();
    rows.forEach((row) => {
      row.years.forEach((yearRow) => {
        years.add(yearRow.year);
      });
    });

    return Array.from(years)
      .sort((a, b) => a - b)
      .map((year) => {
        const nextRow: { year: number } & Record<string, number> = { year };
        rows.forEach((row, index) => {
          const yearRow = row.years.find((item) => item.year === year);
          nextRow[`firm_${index}`] = yearRow ? yearRow[metric] : 0;
        });
        return nextRow;
      });
  }, [metric, rows]);

  return (
    <section aria-labelledby={descriptionId} className="space-y-4">
      <Card variant="subtle">
        <CardHeader className="gap-4 md:flex-row md:items-end md:justify-between">
          <div className="space-y-2">
            <CardTitle id={descriptionId} className="text-xl sm:text-2xl">
              {title}
            </CardTitle>
            <CardDescription className="max-w-3xl text-sm sm:text-base">
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
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {rows.length === 0 ? (
            <div className="rounded-lg border border-dashed border-border/80 bg-background/70 px-4 py-8 text-sm text-muted-foreground">
              No counsel leaderboard data is available yet.
            </div>
          ) : view === "table" ? (
            <div className="overflow-x-auto rounded-lg border border-border/60 bg-background/80">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-[60%] min-w-[16rem]">Firm</TableHead>
                    <TableHead className="text-right">Deals</TableHead>
                    <TableHead className="text-right">Total transaction value</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {rows.map((row) => (
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
  const [data, setData] = useState<CounselLeaderboardResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    const fetchLeaderboards = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await authFetch(apiUrl("v1/counsel-leaderboards"), {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Leaderboards request failed (${response.status})`);
        }
        const nextData = (await response.json()) as CounselLeaderboardResponse;
        if (!cancelled) {
          setData(nextData);
        }
      } catch (err) {
        if (!cancelled && !(err instanceof DOMException && err.name === "AbortError")) {
          setError(
            err instanceof Error ? err.message : "Unable to load counsel leaderboards.",
          );
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    fetchLeaderboards();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, []);

  return (
    <PageShell
      title="Leaderboards"
      size="xl"
      subtitle="Canonical counsel rankings for acquirer and target sides. The metric toggle swaps between distinct top-15 sets for deal count and total transaction value."
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
