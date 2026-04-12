import { Suspense, lazy, useEffect, useId, useMemo, useState } from "react";

import type { TrendsChartSeries } from "@/components/AgreementTrendsCharts";
import { MobileChartModal } from "@/components/MobileChartModal";
import { PageShell } from "@/components/PageShell";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { useIsMobile } from "@/hooks/use-mobile";
import { apiUrl } from "@/lib/api-config";
import { formatCompactCurrencyValue, formatEnumValue } from "@/lib/format-utils";
import { readSessionCache, writeSessionCache } from "@/lib/session-cache";
import { cn } from "@/lib/utils";

type OwnershipMetric = "deal_count" | "total_transaction_value";
type HeatmapMetric = "deal_count" | "median_transaction_value";

type OwnershipMixRow = {
  year: number;
  public_deal_count: number;
  private_deal_count: number;
  public_total_transaction_value: number;
  private_total_transaction_value: number;
};

type OwnershipDealSizeRow = {
  year: number;
  public_deal_count: number;
  private_deal_count: number;
  public_p25_transaction_value: number | null;
  public_median_transaction_value: number | null;
  public_p75_transaction_value: number | null;
  private_p25_transaction_value: number | null;
  private_median_transaction_value: number | null;
  private_p75_transaction_value: number | null;
};

type BuyerTypeMatrixRow = {
  target_bucket: string;
  buyer_bucket: string;
  deal_count: number;
  median_transaction_value: number | null;
};

type TargetIndustryByYearRow = {
  year: number;
  industry: string;
  deal_count: number;
  total_transaction_value: number;
};

type IndustryPairingRow = {
  target_industry: string;
  acquirer_industry: string;
  deal_count: number;
  total_transaction_value: number;
};

type AgreementTrendsResponse = {
  ownership: {
    mix_by_year: OwnershipMixRow[];
    deal_size_by_year: OwnershipDealSizeRow[];
    buyer_type_matrix: BuyerTypeMatrixRow[];
  };
  industries: {
    target_industries_by_year: TargetIndustryByYearRow[];
    pairings: IndustryPairingRow[];
  };
};
const TRENDS_CACHE_KEY = "agreement-trends:v1";
const TRENDS_CACHE_TTL_MS = 5 * 60 * 1000;

const TrendsStackedShareAreaChart = lazy(async () => {
  const module = await import("@/components/AgreementTrendsCharts");
  return { default: module.TrendsStackedShareAreaChart };
});

const TrendsMedianBandChart = lazy(async () => {
  const module = await import("@/components/AgreementTrendsCharts");
  return { default: module.TrendsMedianBandChart };
});

const TrendsPercentLineChart = lazy(async () => {
  const module = await import("@/components/AgreementTrendsCharts");
  return { default: module.TrendsPercentLineChart };
});

type LabeledKey = {
  key: string;
  label: string;
};

type TrendsHeatmapCell = {
  displayValue: string;
  intensity: number;
  rawValue: number | null;
};

const OWNERSHIP_SERIES: TrendsChartSeries[] = [
  {
    key: "public",
    label: "Public targets",
    color: "hsl(212 93% 50%)",
  },
  {
    key: "private",
    label: "Private targets",
    color: "hsl(170 84% 36%)",
  },
];
const INDUSTRY_COLORS = [
  "hsl(212 93% 50%)",
  "hsl(170 84% 36%)",
  "hsl(35 92% 52%)",
  "hsl(0 84% 60%)",
  "hsl(196 83% 42%)",
  "hsl(262 83% 58%)",
  "hsl(142 71% 45%)",
];

function formatMoney(value: number | null | undefined) {
  return formatCompactCurrencyValue(value ?? null);
}

function formatCount(value: number | null | undefined) {
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0,
  }).format(value ?? 0);
}

function formatPercent(value: number) {
  return `${value.toFixed(1)}%`;
}

function formatIndustryLabel(value: string) {
  return value === "Unspecified" ? value : formatEnumValue(value);
}

function formatBuyerBucketLabel(value: string) {
  switch (value) {
    case "public_buyer":
      return "Public buyer";
    case "private_strategic":
      return "Private strategic";
    case "private_equity":
      return "Private equity";
    default:
      return "Other";
  }
}

function formatTargetBucketLabel(value: string) {
  return value === "public" ? "Public targets" : "Private targets";
}

function buildIndustrySeriesKey(index: number) {
  return `industry_${index}`;
}

function TrendsSkeleton() {
  return (
    <div className="space-y-6">
      <Card variant="subtle">
        <CardHeader>
          <Skeleton className="h-7 w-56" />
          <Skeleton className="h-4 w-full max-w-3xl" />
        </CardHeader>
      </Card>
      {[0, 1, 2].map((index) => (
        <Card key={index} variant="subtle">
          <CardHeader>
            <Skeleton className="h-7 w-64" />
            <Skeleton className="h-4 w-full max-w-3xl" />
          </CardHeader>
          <CardContent className="space-y-3">
            <Skeleton className="h-10 w-56" />
            <Skeleton className="h-72 w-full" />
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

function ChartSkeleton({ className = "h-[260px] sm:h-[320px] lg:h-[360px]" }: { className?: string }) {
  return (
    <div className="rounded-lg border border-border/60 bg-muted/20 p-3">
      <Skeleton className={`w-full ${className}`} />
    </div>
  );
}

function TrendsHeatmapTable({
  caption,
  className,
  columns,
  formatterLabel,
  getCell,
  rows,
}: {
  caption: string;
  className?: string;
  columns: string[];
  formatterLabel: string;
  getCell: (row: string, column: string) => TrendsHeatmapCell;
  rows: string[];
}) {
  return (
    <div className={cn("overflow-x-auto rounded-lg border border-border/60 bg-background/80", className)}>
      <table className="w-full min-w-[56rem] table-fixed border-collapse text-sm">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="border-b border-border/60">
            <th className="w-64 px-3 py-2 text-left font-semibold text-foreground">
              Segment
            </th>
            {columns.map((column) => (
              <th
                key={column}
                className="w-44 px-3 py-2 text-center font-semibold text-foreground"
                title={column}
              >
                <span className="block overflow-hidden text-ellipsis whitespace-nowrap">
                  {column}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row} className="border-b border-border/40 last:border-0">
              <th className="w-64 px-3 py-3 text-left font-medium text-foreground" title={row}>
                <span className="block overflow-hidden text-ellipsis whitespace-nowrap">
                  {row}
                </span>
              </th>
              {columns.map((column) => {
                const cell = getCell(row, column);
                const opacity = 0.1 + (cell.intensity * 0.75);
                const backgroundColor = `hsl(212 93% 50% / ${opacity})`;
                const foregroundClass =
                  cell.intensity > 0.58 ? "text-white" : "text-foreground";

                return (
                  <td key={`${row}-${column}`} className="w-44 px-2 py-2 align-top">
                    <div
                      className={cn(
                        "flex min-h-[6.75rem] w-full flex-col items-center justify-center rounded-md border border-border/50 px-3 py-3 text-center shadow-sm transition-colors",
                        foregroundClass,
                      )}
                      style={
                        cell.rawValue === null || cell.rawValue === 0
                          ? undefined
                          : { backgroundColor }
                      }
                      aria-label={`${row}, ${column}, ${formatterLabel}: ${cell.displayValue}`}
                    >
                      <div className="text-xs font-medium uppercase tracking-wide opacity-75">
                        {formatterLabel}
                      </div>
                      <div className="mt-1 font-mono text-sm tabular-nums">
                        {cell.displayValue}
                      </div>
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function OwnershipStructurePanel({
  ownership,
}: {
  ownership: AgreementTrendsResponse["ownership"];
}) {
  const isMobile = useIsMobile();
  const [mixMetric, setMixMetric] = useState<OwnershipMetric>("deal_count");
  const [matrixMetric, setMatrixMetric] = useState<HeatmapMetric>("deal_count");
  const [isMixChartModalOpen, setIsMixChartModalOpen] = useState(false);
  const [isDealSizeChartModalOpen, setIsDealSizeChartModalOpen] = useState(false);
  const mixDescriptionId = useId();
  const dealSizeDescriptionId = useId();
  const matrixDescriptionId = useId();
  const mixTableId = useId();
  const dealSizeTableId = useId();

  const mixChartData = useMemo(
    () =>
      ownership.mix_by_year.map((row) => ({
        year: row.year,
        public:
          mixMetric === "deal_count"
            ? row.public_deal_count
            : row.public_total_transaction_value,
        private:
          mixMetric === "deal_count"
            ? row.private_deal_count
            : row.private_total_transaction_value,
      })),
    [mixMetric, ownership.mix_by_year],
  );

  const dealSizeChartData = useMemo(
    () =>
      ownership.deal_size_by_year.map((row) => ({
        year: row.year,
        public_low: row.public_p25_transaction_value ?? 0,
        public_band:
          row.public_p25_transaction_value !== null &&
          row.public_p75_transaction_value !== null
            ? Math.max(
                0,
                row.public_p75_transaction_value - row.public_p25_transaction_value,
              )
            : 0,
        public_median: row.public_median_transaction_value,
        private_low: row.private_p25_transaction_value ?? 0,
        private_band:
          row.private_p25_transaction_value !== null &&
          row.private_p75_transaction_value !== null
            ? Math.max(
                0,
                row.private_p75_transaction_value - row.private_p25_transaction_value,
              )
            : 0,
        private_median: row.private_median_transaction_value,
      })),
    [ownership.deal_size_by_year],
  );

  const matrixRows = useMemo(() => {
    const rowKeys = ["public", "private"];
    const columnKeys = ["public_buyer", "private_strategic", "private_equity", "other"];
    const valueByKey = new Map(
      ownership.buyer_type_matrix.map((row) => [
        `${row.target_bucket}:${row.buyer_bucket}`,
        row,
      ]),
    );
    const visibleColumns = columnKeys.filter((columnKey) =>
      ownership.buyer_type_matrix.some(
        (row) => row.buyer_bucket === columnKey && row.deal_count > 0,
      ),
    );
    const maxValue = Math.max(
      0,
      ...ownership.buyer_type_matrix.map((row) =>
        matrixMetric === "deal_count"
          ? row.deal_count
          : row.median_transaction_value ?? 0,
      ),
    );

    return {
      columns: visibleColumns.map((key) => ({
        key,
        label: formatBuyerBucketLabel(key),
      })),
      rows: rowKeys.map((key) => ({
        key,
        label: formatTargetBucketLabel(key),
      })),
      getCell: (rowKey: string, columnKey: string) => {
        const row = valueByKey.get(`${rowKey}:${columnKey}`);
        const rawValue =
          matrixMetric === "deal_count"
            ? row?.deal_count ?? 0
            : row?.median_transaction_value ?? null;
        const numericValue = rawValue ?? 0;
        return {
          rawValue,
          displayValue:
            matrixMetric === "deal_count"
              ? formatCount(numericValue)
              : rawValue === null
                ? "—"
                : formatMoney(numericValue),
          intensity: maxValue > 0 ? numericValue / maxValue : 0,
        };
      },
    };
  }, [matrixMetric, ownership.buyer_type_matrix]);

  return (
    <div className="space-y-6">
      <Card variant="subtle">
        <CardHeader className="gap-4 md:flex-row md:items-end md:justify-between">
          <div className="space-y-2">
            <CardTitle id={mixDescriptionId} className="text-xl sm:text-2xl">
              Public vs. Private Target Mix Over Time
            </CardTitle>
            <CardDescription className="text-sm sm:text-base">
              Share of public vs. private targets by filing year.
            </CardDescription>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Metric
            </span>
            <ToggleGroup
              type="single"
              value={mixMetric}
              onValueChange={(value) => {
                if (value === "deal_count" || value === "total_transaction_value") {
                  setMixMetric(value);
                }
              }}
              variant="outline"
              aria-label="Ownership mix metric"
              className="justify-start"
            >
              <ToggleGroupItem value="deal_count">Deal count</ToggleGroupItem>
              <ToggleGroupItem value="total_transaction_value">Value</ToggleGroupItem>
            </ToggleGroup>
          </div>
        </CardHeader>
        <CardContent>
          <p id={mixTableId} className="sr-only">
            100 percent stacked area chart showing public-target and private-target share by filing
            year.
          </p>
          {isMobile ? (
            <>
              <Button
                type="button"
                variant="outline"
                className="w-full"
                onClick={() => setIsMixChartModalOpen(true)}
                aria-haspopup="dialog"
                aria-describedby={mixDescriptionId}
              >
                Open chart
              </Button>
              <MobileChartModal
                open={isMixChartModalOpen}
                onOpenChange={setIsMixChartModalOpen}
                title="Public vs. Private Target Mix Over Time"
                description="Shows the share of public versus private targets by filing year."
              >
                <div className="mx-auto w-full max-w-[980px]">
                  <h2 className="mb-2 text-base font-semibold">
                    Public vs. Private Target Mix Over Time
                  </h2>
                  <Suspense fallback={<ChartSkeleton />}>
                    <TrendsStackedShareAreaChart
                      className="border-0 bg-background p-0"
                      ariaLabel="100 percent stacked area chart showing public-target and private-target share by filing year."
                      data={mixChartData}
                      describedBy={mixDescriptionId}
                      series={OWNERSHIP_SERIES}
                      tableId={mixTableId}
                      valueFormatter={mixMetric === "deal_count" ? formatCount : formatMoney}
                    />
                  </Suspense>
                </div>
              </MobileChartModal>
            </>
          ) : (
            <Suspense fallback={<ChartSkeleton />}>
              <TrendsStackedShareAreaChart
                ariaLabel="100 percent stacked area chart showing public-target and private-target share by filing year."
                data={mixChartData}
                describedBy={mixDescriptionId}
                series={OWNERSHIP_SERIES}
                tableId={mixTableId}
                valueFormatter={mixMetric === "deal_count" ? formatCount : formatMoney}
              />
            </Suspense>
          )}
        </CardContent>
      </Card>

      <Card variant="subtle">
        <CardHeader>
          <div className="space-y-2">
            <CardTitle id={dealSizeDescriptionId} className="text-xl sm:text-2xl">
              Public vs. Private Deal Size
            </CardTitle>
            <CardDescription className="text-sm sm:text-base">
              Median deal value by year; shaded bands show the 25th to 75th percentile range.
            </CardDescription>
          </div>
        </CardHeader>
        <CardContent>
          <p id={dealSizeTableId} className="sr-only">
            Line chart showing median reported deal size with 25th to 75th percentile bands for
            public and private targets by filing year.
          </p>
          {isMobile ? (
            <>
              <Button
                type="button"
                variant="outline"
                className="w-full"
                onClick={() => setIsDealSizeChartModalOpen(true)}
                aria-haspopup="dialog"
                aria-describedby={dealSizeDescriptionId}
              >
                Open chart
              </Button>
              <MobileChartModal
                open={isDealSizeChartModalOpen}
                onOpenChange={setIsDealSizeChartModalOpen}
                title="Public vs. Private Deal Size"
                description="Shows median reported deal value with percentile bands for public and private targets by filing year."
              >
                <div className="mx-auto w-full max-w-[980px]">
                  <h2 className="mb-2 text-base font-semibold">
                    Public vs. Private Deal Size
                  </h2>
                  <Suspense fallback={<ChartSkeleton />}>
                    <TrendsMedianBandChart
                      className="border-0 bg-background p-0"
                      ariaLabel="Line chart showing median reported deal size with percentile bands for public and private targets by filing year."
                      data={dealSizeChartData}
                      describedBy={dealSizeDescriptionId}
                      tableId={dealSizeTableId}
                      valueFormatter={(value) => formatMoney(value)}
                    />
                  </Suspense>
                </div>
              </MobileChartModal>
            </>
          ) : (
            <Suspense fallback={<ChartSkeleton />}>
              <TrendsMedianBandChart
                ariaLabel="Line chart showing median reported deal size with percentile bands for public and private targets by filing year."
                data={dealSizeChartData}
                describedBy={dealSizeDescriptionId}
                tableId={dealSizeTableId}
                valueFormatter={(value) => formatMoney(value)}
              />
            </Suspense>
          )}
        </CardContent>
      </Card>

      <Card variant="subtle">
        <CardHeader className="gap-4 md:flex-row md:items-end md:justify-between">
          <div className="space-y-2">
            <CardTitle id={matrixDescriptionId} className="text-xl sm:text-2xl">
              Target Type by Buyer Type
            </CardTitle>
            <CardDescription className="text-sm sm:text-base">
              Counts or median deal value for each target-buyer bucket combination.
            </CardDescription>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Heatmap metric
            </span>
            <ToggleGroup
              type="single"
              value={matrixMetric}
              onValueChange={(value) => {
                if (value === "deal_count" || value === "median_transaction_value") {
                  setMatrixMetric(value);
                }
              }}
              variant="outline"
              aria-label="Buyer matrix metric"
              className="justify-start"
            >
              <ToggleGroupItem value="deal_count">Deal count</ToggleGroupItem>
              <ToggleGroupItem value="median_transaction_value">Median value</ToggleGroupItem>
            </ToggleGroup>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <p className="text-sm text-muted-foreground">
            Private targets include PE-backed targets. Private-equity buyers are broken out
            separately from private strategic buyers.
          </p>
          <TrendsHeatmapTable
            caption="Target type by buyer type heatmap"
            columns={matrixRows.columns.map((item) => item.label)}
            formatterLabel={
              matrixMetric === "deal_count" ? "Deal count" : "Median value"
            }
            getCell={(rowLabel, columnLabel) => {
              const rowKey = matrixRows.rows.find((item) => item.label === rowLabel)?.key ?? "";
              const columnKey =
                matrixRows.columns.find((item) => item.label === columnLabel)?.key ?? "";
              return matrixRows.getCell(rowKey, columnKey);
            }}
            rows={matrixRows.rows.map((item) => item.label)}
          />
        </CardContent>
      </Card>
    </div>
  );
}

function IndustryDynamicsPanel({
  industries,
}: {
  industries: AgreementTrendsResponse["industries"];
}) {
  const isMobile = useIsMobile();
  const [compositionMetric, setCompositionMetric] =
    useState<OwnershipMetric>("deal_count");
  const [pairingsMetric, setPairingsMetric] =
    useState<OwnershipMetric>("deal_count");
  const [concentrationMetric, setConcentrationMetric] =
    useState<OwnershipMetric>("deal_count");
  const [isCompositionChartModalOpen, setIsCompositionChartModalOpen] =
    useState(false);
  const [isConcentrationChartModalOpen, setIsConcentrationChartModalOpen] =
    useState(false);
  const compositionDescriptionId = useId();
  const pairingsDescriptionId = useId();
  const concentrationDescriptionId = useId();
  const compositionTableId = useId();
  const concentrationTableId = useId();

  const industryComposition = useMemo(() => {
    const totalsByIndustry = new Map<string, number>();
    industries.target_industries_by_year.forEach((row) => {
      const metricValue =
        compositionMetric === "deal_count"
          ? row.deal_count
          : row.total_transaction_value;
      totalsByIndustry.set(
        row.industry,
        (totalsByIndustry.get(row.industry) ?? 0) + metricValue,
      );
    });

    const topIndustries = Array.from(totalsByIndustry.entries())
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, 6)
      .map(([industry]) => industry);
    const years = Array.from(
      new Set(industries.target_industries_by_year.map((row) => row.year)),
    ).sort((a, b) => a - b);

    const data = years.map((year) => {
      const nextRow: { year: number } & Record<string, number> = { year };
      let otherValue = 0;
      topIndustries.forEach((_industry, index) => {
        nextRow[buildIndustrySeriesKey(index)] = 0;
      });
      industries.target_industries_by_year
        .filter((row) => row.year === year)
        .forEach((row) => {
          const metricValue =
            compositionMetric === "deal_count"
              ? row.deal_count
              : row.total_transaction_value;
          const industryIndex = topIndustries.indexOf(row.industry);
          if (industryIndex >= 0) {
            nextRow[buildIndustrySeriesKey(industryIndex)] = metricValue;
          } else {
            otherValue += metricValue;
          }
        });
      if (otherValue > 0) {
        nextRow.Other = otherValue;
      }
      return nextRow;
    });

    const series = topIndustries.map((industry, index) => ({
      key: buildIndustrySeriesKey(index),
      label: formatIndustryLabel(industry),
      color: INDUSTRY_COLORS[index % INDUSTRY_COLORS.length],
    }));
    if (data.some((row) => Number(row.Other ?? 0) > 0)) {
      series.push({
        key: "Other",
        label: "Other",
        color: "hsl(220 9% 60%)",
      });
    }

    return { data, series };
  }, [compositionMetric, industries.target_industries_by_year]);

  const pairingsHeatmap = useMemo(() => {
    const rowTotals = new Map<string, number>();
    const columnTotals = new Map<string, number>();
    const cellValues = new Map<string, number>();
    industries.pairings.forEach((row) => {
      const metricValue =
        pairingsMetric === "deal_count" ? row.deal_count : row.total_transaction_value;
      rowTotals.set(
        row.target_industry,
        (rowTotals.get(row.target_industry) ?? 0) + metricValue,
      );
      columnTotals.set(
        row.acquirer_industry,
        (columnTotals.get(row.acquirer_industry) ?? 0) + metricValue,
      );
      cellValues.set(`${row.target_industry}:${row.acquirer_industry}`, metricValue);
    });

    const rows = Array.from(rowTotals.entries())
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, 5)
      .map(([industry]) => ({
        key: industry,
        label: formatIndustryLabel(industry),
      }));
    const columns = Array.from(columnTotals.entries())
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, 5)
      .map(([industry]) => ({
        key: industry,
        label: formatIndustryLabel(industry),
      }));

    const maxValue = Math.max(
      0,
      ...rows.flatMap((row) =>
        columns.map(
          (column) => cellValues.get(`${row.key}:${column.key}`) ?? 0,
        ),
      ),
    );

    return {
      columns,
      rows,
      getCell: (rowKey: string, columnKey: string) => {
        const rawValue = cellValues.get(`${rowKey}:${columnKey}`) ?? 0;
        return {
          rawValue,
          displayValue:
            pairingsMetric === "deal_count"
              ? formatCount(rawValue)
              : formatMoney(rawValue),
          intensity: maxValue > 0 ? rawValue / maxValue : 0,
        };
      },
    };
  }, [industries.pairings, pairingsMetric]);

  const concentrationTrend = useMemo(() => {
    const totalsByIndustry = new Map<string, number>();
    industries.target_industries_by_year.forEach((row) => {
      const metricValue =
        concentrationMetric === "deal_count"
          ? row.deal_count
          : row.total_transaction_value;
      totalsByIndustry.set(
        row.industry,
        (totalsByIndustry.get(row.industry) ?? 0) + metricValue,
      );
    });
    const topIndustries = Array.from(totalsByIndustry.entries())
      .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
      .slice(0, 5)
      .map(([industry]) => industry);
    const years = Array.from(
      new Set(industries.target_industries_by_year.map((row) => row.year)),
    ).sort((a, b) => a - b);

    return {
      topIndustries,
      data: years.map((year) => {
        const rows = industries.target_industries_by_year.filter((row) => row.year === year);
        const total = rows.reduce(
          (sum, row) =>
            sum +
            (concentrationMetric === "deal_count"
              ? row.deal_count
              : row.total_transaction_value),
          0,
        );
        const topTotal = rows.reduce((sum, row) => {
          if (!topIndustries.includes(row.industry)) return sum;
          return (
            sum +
            (concentrationMetric === "deal_count"
              ? row.deal_count
              : row.total_transaction_value)
          );
        }, 0);
        return {
          year,
          share: total > 0 ? Math.round((topTotal / total) * 1000) / 10 : 0,
        };
      }),
    };
  }, [concentrationMetric, industries.target_industries_by_year]);

  return (
    <div className="space-y-6">
      <Card variant="subtle">
        <CardHeader className="gap-4 md:flex-row md:items-end md:justify-between">
          <div className="space-y-2">
            <CardTitle id={compositionDescriptionId} className="text-xl sm:text-2xl">
              Industry Composition Over Time
            </CardTitle>
            <CardDescription className="text-sm sm:text-base">
              Share of deals by target industry over time.
            </CardDescription>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Metric
            </span>
            <ToggleGroup
              type="single"
              value={compositionMetric}
              onValueChange={(value) => {
                if (value === "deal_count" || value === "total_transaction_value") {
                  setCompositionMetric(value);
                }
              }}
              variant="outline"
              aria-label="Industry composition metric"
              className="justify-start"
            >
              <ToggleGroupItem value="deal_count">Deal count</ToggleGroupItem>
              <ToggleGroupItem value="total_transaction_value">Value</ToggleGroupItem>
            </ToggleGroup>
          </div>
        </CardHeader>
        <CardContent>
          <p id={compositionTableId} className="sr-only">
            100 percent stacked area chart showing target-industry composition by filing year.
          </p>
          {isMobile ? (
            <>
              <Button
                type="button"
                variant="outline"
                className="w-full"
                onClick={() => setIsCompositionChartModalOpen(true)}
                aria-haspopup="dialog"
                aria-describedby={compositionDescriptionId}
              >
                Open chart
              </Button>
              <MobileChartModal
                open={isCompositionChartModalOpen}
                onOpenChange={setIsCompositionChartModalOpen}
                title="Industry Composition Over Time"
                description="Shows the share of deals by target industry over time."
              >
                <div className="mx-auto w-full max-w-[980px]">
                  <h2 className="mb-2 text-base font-semibold">
                    Industry Composition Over Time
                  </h2>
                  <Suspense fallback={<ChartSkeleton />}>
                    <TrendsStackedShareAreaChart
                      className="border-0 bg-background p-0"
                      ariaLabel="100 percent stacked area chart showing target-industry composition by filing year."
                      data={industryComposition.data}
                      describedBy={compositionDescriptionId}
                      series={industryComposition.series}
                      tableId={compositionTableId}
                      valueFormatter={
                        compositionMetric === "deal_count" ? formatCount : formatMoney
                      }
                    />
                  </Suspense>
                </div>
              </MobileChartModal>
            </>
          ) : (
            <Suspense fallback={<ChartSkeleton />}>
              <TrendsStackedShareAreaChart
                ariaLabel="100 percent stacked area chart showing target-industry composition by filing year."
                data={industryComposition.data}
                describedBy={compositionDescriptionId}
                series={industryComposition.series}
                tableId={compositionTableId}
                valueFormatter={
                  compositionMetric === "deal_count" ? formatCount : formatMoney
                }
              />
            </Suspense>
          )}
        </CardContent>
      </Card>

      <Card variant="subtle">
        <CardHeader className="gap-4 md:flex-row md:items-end md:justify-between">
          <div className="space-y-2">
            <CardTitle id={pairingsDescriptionId} className="text-xl sm:text-2xl">
              Top Industry Pairings
            </CardTitle>
            <CardDescription className="text-sm sm:text-base">
              Most common target and acquirer industry combinations.
            </CardDescription>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Heatmap metric
            </span>
            <ToggleGroup
              type="single"
              value={pairingsMetric}
              onValueChange={(value) => {
                if (value === "deal_count" || value === "total_transaction_value") {
                  setPairingsMetric(value);
                }
              }}
              variant="outline"
              aria-label="Industry pairing metric"
              className="justify-start"
            >
              <ToggleGroupItem value="deal_count">Deal count</ToggleGroupItem>
              <ToggleGroupItem value="total_transaction_value">Value</ToggleGroupItem>
            </ToggleGroup>
          </div>
        </CardHeader>
        <CardContent>
          <TrendsHeatmapTable
            caption="Target versus acquirer industry pairing heatmap"
            columns={pairingsHeatmap.columns.map((item) => item.label)}
            formatterLabel={pairingsMetric === "deal_count" ? "Deal count" : "Value"}
            getCell={(rowLabel, columnLabel) => {
              const rowKey = pairingsHeatmap.rows.find((item) => item.label === rowLabel)?.key ?? "";
              const columnKey =
                pairingsHeatmap.columns.find((item) => item.label === columnLabel)?.key ?? "";
              return pairingsHeatmap.getCell(rowKey, columnKey);
            }}
            rows={pairingsHeatmap.rows.map((item) => item.label)}
          />
        </CardContent>
      </Card>

      <Card variant="subtle">
        <CardHeader className="gap-4 md:flex-row md:items-end md:justify-between">
          <div className="space-y-2">
            <CardTitle id={concentrationDescriptionId} className="text-xl sm:text-2xl">
              Sector Concentration Trend
            </CardTitle>
            <CardDescription className="text-sm sm:text-base">
              Share of annual activity accounted for by the five largest target industries.
            </CardDescription>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Metric
            </span>
            <ToggleGroup
              type="single"
              value={concentrationMetric}
              onValueChange={(value) => {
                if (value === "deal_count" || value === "total_transaction_value") {
                  setConcentrationMetric(value);
                }
              }}
              variant="outline"
              aria-label="Sector concentration metric"
              className="justify-start"
            >
              <ToggleGroupItem value="deal_count">Deal count</ToggleGroupItem>
              <ToggleGroupItem value="total_transaction_value">Value</ToggleGroupItem>
            </ToggleGroup>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <p id={concentrationTableId} className="sr-only">
            Line chart showing the share of annual activity accounted for by the top five target
            industries.
          </p>
          {isMobile ? (
            <>
              <Button
                type="button"
                variant="outline"
                className="w-full"
                onClick={() => setIsConcentrationChartModalOpen(true)}
                aria-haspopup="dialog"
                aria-describedby={concentrationDescriptionId}
              >
                Open chart
              </Button>
              <MobileChartModal
                open={isConcentrationChartModalOpen}
                onOpenChange={setIsConcentrationChartModalOpen}
                title="Sector Concentration Trend"
                description="Shows the share of annual activity accounted for by the five largest target industries."
              >
                <div className="mx-auto w-full max-w-[980px]">
                  <h2 className="mb-2 text-base font-semibold">
                    Sector Concentration Trend
                  </h2>
                  <Suspense
                    fallback={<ChartSkeleton className="h-[220px] sm:h-[280px] lg:h-[320px]" />}
                  >
                    <TrendsPercentLineChart
                      className="border-0 bg-background p-0"
                      ariaLabel="Line chart showing the share of annual activity accounted for by the top five target industries."
                      data={concentrationTrend.data}
                      describedBy={concentrationDescriptionId}
                      lineColor="hsl(12 76% 61%)"
                      tableId={concentrationTableId}
                    />
                  </Suspense>
                </div>
              </MobileChartModal>
            </>
          ) : (
            <Suspense fallback={<ChartSkeleton className="h-[220px] sm:h-[280px] lg:h-[320px]" />}>
              <TrendsPercentLineChart
                ariaLabel="Line chart showing the share of annual activity accounted for by the top five target industries."
                data={concentrationTrend.data}
                describedBy={concentrationDescriptionId}
                lineColor="hsl(12 76% 61%)"
                tableId={concentrationTableId}
              />
            </Suspense>
          )}
          <div className="flex flex-wrap gap-2">
            {concentrationTrend.topIndustries.map((industry) => (
              <span
                key={industry}
                className="rounded-full border border-border/60 bg-background/80 px-3 py-1 text-xs font-medium text-foreground"
              >
                {formatIndustryLabel(industry)}
              </span>
            ))}
          </div>
          <p className="text-sm text-muted-foreground">
            Current top-five basket share:{" "}
            <span className="font-mono tabular-nums text-foreground">
              {concentrationTrend.data.length > 0
                ? formatPercent(
                    concentrationTrend.data[concentrationTrend.data.length - 1].share,
                  )
                : "—"}
            </span>
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

export default function TrendsAnalyses() {
  const [data, setData] = useState<AgreementTrendsResponse | null>(() =>
    readSessionCache<AgreementTrendsResponse>(
      TRENDS_CACHE_KEY,
      TRENDS_CACHE_TTL_MS,
    ),
  );
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(data === null);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    const fetchTrends = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch(apiUrl("v1/agreement-trends"), {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Agreement trends request failed (${response.status})`);
        }
        const nextData = (await response.json()) as AgreementTrendsResponse;
        if (!cancelled) {
          setData(nextData);
          writeSessionCache(TRENDS_CACHE_KEY, nextData);
        }
      } catch (err) {
        if (!cancelled && !(err instanceof DOMException && err.name === "AbortError")) {
          setError(
            err instanceof Error ? err.message : "Unable to load trend analyses.",
          );
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    fetchTrends();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, []);

  return (
    <PageShell
      title="Trends & Analyses"
      size="xl"
      subtitle="A set of deeper cuts on ownership structure and sector composition, built from public-eligible agreements and designed to complement the headline leaderboards."
    >
      <div className="space-y-6">
        {loading ? <TrendsSkeleton /> : null}

        {!loading && error ? (
          <Card variant="subtle">
            <CardHeader>
              <CardTitle className="text-xl">Unavailable</CardTitle>
              <CardDescription>{error}</CardDescription>
            </CardHeader>
          </Card>
        ) : null}

        {!loading && !error && data ? (
          <Tabs defaultValue="ownership" className="space-y-4">
            <TabsList className="grid h-auto w-full grid-cols-2">
              <TabsTrigger value="ownership">Ownership Structure</TabsTrigger>
              <TabsTrigger value="industry">Industry Dynamics</TabsTrigger>
            </TabsList>
            <TabsContent value="ownership" className="space-y-6">
              <OwnershipStructurePanel ownership={data.ownership} />
            </TabsContent>
            <TabsContent value="industry" className="space-y-6">
              <IndustryDynamicsPanel industries={data.industries} />
            </TabsContent>
          </Tabs>
        ) : null}
      </div>
    </PageShell>
  );
}
