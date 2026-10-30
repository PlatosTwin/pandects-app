import { Suspense, lazy, useId, useMemo, useState } from "react";

import type { TrendsChartSeries } from "@/components/AgreementTrendsCharts";
import { ChartPanel } from "@/components/ChartPanel";
import { PageShell } from "@/components/PageShell";
import { ErrorState } from "@/components/StateMessage";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Eyebrow } from "@/components/ui/eyebrow";
import {
  SegmentedControl,
  type SegmentedControlOption,
} from "@/components/ui/segmented-control";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useAgreementTrends } from "@/hooks/use-insights";
import {
  CHART_NEUTRAL_SERIES_COLOR,
  INDUSTRY_CHART_PALETTE,
  OWNERSHIP_SERIES_COLORS,
  SECTOR_CONCENTRATION_LINE_COLOR,
  heatmapCellFill,
} from "@/lib/chart-palette";
import { formatCompactCurrencyValue, formatEnumValue } from "@/lib/format-utils";
import type { AgreementTrendsResponse } from "@shared/insights";
import { cn } from "@/lib/utils";

type OwnershipMetric = "deal_count" | "total_transaction_value";
type HeatmapMetric = "deal_count" | "median_transaction_value";


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

type TrendsHeatmapCell = {
  displayValue: string;
  intensity: number;
  rawValue: number | null;
};

const OWNERSHIP_SERIES: TrendsChartSeries[] = [
  {
    key: "public",
    label: "Public targets",
    color: OWNERSHIP_SERIES_COLORS.public,
  },
  {
    key: "private",
    label: "Private targets",
    color: OWNERSHIP_SERIES_COLORS.private,
  },
];
const INDUSTRY_COLORS = INDUSTRY_CHART_PALETTE;

const VALUE_METRIC_OPTIONS: ReadonlyArray<SegmentedControlOption<OwnershipMetric>> = [
  { value: "deal_count", label: "Deal count" },
  { value: "total_transaction_value", label: "Value" },
];
const HEATMAP_METRIC_OPTIONS: ReadonlyArray<SegmentedControlOption<HeatmapMetric>> = [
  { value: "deal_count", label: "Deal count" },
  { value: "median_transaction_value", label: "Median value" },
];

/** Eyebrow-labeled metric switch used in chart panel controls. */
function MetricSwitch<T extends string>({
  label,
  value,
  onValueChange,
  options,
  ariaLabel,
}: {
  label: string;
  value: T;
  onValueChange: (value: T) => void;
  options: ReadonlyArray<SegmentedControlOption<T>>;
  ariaLabel: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <Eyebrow>{label}</Eyebrow>
      <SegmentedControl<T>
        value={value}
        onValueChange={onValueChange}
        options={options}
        aria-label={ariaLabel}
      />
    </div>
  );
}

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
    <div className="rounded-lg border border-border bg-muted/20 p-3">
      <Skeleton className={`w-full ${className}`} />
    </div>
  );
}

// The segment column stays put while the cells scroll sideways, so each row
// keeps its label on narrow screens.
// Narrow on phones (labels wrap) so a full data cell still fits beside it.
const STICKY_SEGMENT_CELL =
  "sticky left-0 z-10 w-24 bg-background px-2 text-left text-foreground shadow-[1px_0_0_hsl(var(--foreground)/0.12)] sm:w-64 sm:px-3";
const SEGMENT_LABEL = "block break-words sm:overflow-hidden sm:text-ellipsis sm:whitespace-nowrap";

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
    <div className={cn("overflow-x-auto rounded-lg border border-border bg-background/80", className)}>
      <table className="w-full min-w-[40rem] table-fixed border-collapse text-sm sm:min-w-[56rem]">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="border-b border-border">
            <th className={cn(STICKY_SEGMENT_CELL, "py-2 font-semibold")}>
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
            <tr key={row} className="border-b border-border/50 last:border-0">
              <th className={cn(STICKY_SEGMENT_CELL, "py-3 font-medium")} title={row}>
                <span className={SEGMENT_LABEL}>
                  {row}
                </span>
              </th>
              {columns.map((column) => {
                const cell = getCell(row, column);
                const opacity = 0.1 + (cell.intensity * 0.75);
                const backgroundColor = heatmapCellFill(opacity);
                // text-foreground clears WCAG AA against the blue fill at every
                // intensity in both themes; white-on-blue fails in light mode.
                const foregroundClass = "text-foreground";

                return (
                  <td key={`${row}-${column}`} className="w-44 px-2 py-2 align-top">
                    <div
                      className={cn(
                        "flex min-h-[6.75rem] w-full flex-col items-center justify-center rounded-md border border-border/50 px-3 py-3 text-center shadow-raised transition-colors",
                        foregroundClass,
                      )}
                      style={
                        cell.rawValue === null || cell.rawValue === 0
                          ? undefined
                          : { backgroundColor }
                      }
                      aria-label={`${row}, ${column}, ${formatterLabel}: ${cell.displayValue}`}
                    >
                      <Eyebrow className="text-current">{formatterLabel}</Eyebrow>
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
  const [mixMetric, setMixMetric] = useState<OwnershipMetric>("deal_count");
  const [matrixMetric, setMatrixMetric] = useState<HeatmapMetric>("deal_count");
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

  const mixChart = (inModal: boolean) => (
    <Suspense fallback={<ChartSkeleton />}>
      <TrendsStackedShareAreaChart
        className={inModal ? "border-0 bg-background p-0" : undefined}
        ariaLabel="100 percent stacked area chart showing public-target and private-target share by filing year."
        data={mixChartData}
        describedBy={mixDescriptionId}
        series={OWNERSHIP_SERIES}
        tableId={mixTableId}
        valueFormatter={mixMetric === "deal_count" ? formatCount : formatMoney}
      />
    </Suspense>
  );
  const dealSizeChart = (inModal: boolean) => (
    <Suspense fallback={<ChartSkeleton />}>
      <TrendsMedianBandChart
        className={inModal ? "border-0 bg-background p-0" : undefined}
        ariaLabel="Line chart showing median reported deal size with percentile bands for public and private targets by filing year."
        data={dealSizeChartData}
        describedBy={dealSizeDescriptionId}
        tableId={dealSizeTableId}
        valueFormatter={(value) => formatMoney(value)}
      />
    </Suspense>
  );

  return (
    <div className="space-y-6">
      <ChartPanel
        id={mixDescriptionId}
        title="Public vs. Private Target Mix Over Time"
        description={
          <span className="prose-copy">Share of public vs. private targets by filing year.</span>
        }
        controls={
          <MetricSwitch<OwnershipMetric>
            label="Metric"
            value={mixMetric}
            onValueChange={setMixMetric}
            options={VALUE_METRIC_OPTIONS}
            ariaLabel="Ownership mix metric"
          />
        }
        mobileExpand={() => mixChart(true)}
      >
        <p id={mixTableId} className="sr-only">
          100 percent stacked area chart showing public-target and private-target share by filing
          year.
        </p>
        {mixChart(false)}
      </ChartPanel>

      <ChartPanel
        id={dealSizeDescriptionId}
        title="Public vs. Private Deal Size"
        description={
          <span className="prose-copy">
            Median deal value by year; shaded bands show the 25th to 75th percentile range.
          </span>
        }
        mobileExpand={() => dealSizeChart(true)}
      >
        <p id={dealSizeTableId} className="sr-only">
          Line chart showing median reported deal size with 25th to 75th percentile bands for
          public and private targets by filing year.
        </p>
        {dealSizeChart(false)}
      </ChartPanel>

      <ChartPanel
        id={matrixDescriptionId}
        title="Target Type by Buyer Type"
        description={
          <span className="prose-copy">
            Counts or median deal value for each target-buyer bucket combination.
          </span>
        }
        controls={
          <MetricSwitch<HeatmapMetric>
            label="Heatmap metric"
            value={matrixMetric}
            onValueChange={setMatrixMetric}
            options={HEATMAP_METRIC_OPTIONS}
            ariaLabel="Buyer matrix metric"
          />
        }
        footnote="Private targets include PE-backed targets. Private-equity buyers are broken out separately from private strategic buyers."
      >
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
      </ChartPanel>
    </div>
  );
}

function IndustryDynamicsPanel({
  industries,
}: {
  industries: AgreementTrendsResponse["industries"];
}) {
  const [compositionMetric, setCompositionMetric] =
    useState<OwnershipMetric>("deal_count");
  const [pairingsMetric, setPairingsMetric] =
    useState<OwnershipMetric>("deal_count");
  const [concentrationMetric, setConcentrationMetric] =
    useState<OwnershipMetric>("deal_count");
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
        color: CHART_NEUTRAL_SERIES_COLOR,
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

  const compositionChart = (inModal: boolean) => (
    <Suspense fallback={<ChartSkeleton />}>
      <TrendsStackedShareAreaChart
        className={inModal ? "border-0 bg-background p-0" : undefined}
        ariaLabel="100 percent stacked area chart showing target-industry composition by filing year."
        data={industryComposition.data}
        describedBy={compositionDescriptionId}
        series={industryComposition.series}
        tableId={compositionTableId}
        valueFormatter={compositionMetric === "deal_count" ? formatCount : formatMoney}
      />
    </Suspense>
  );
  const concentrationChart = (inModal: boolean) => (
    <Suspense fallback={<ChartSkeleton className="h-[220px] sm:h-[280px] lg:h-[320px]" />}>
      <TrendsPercentLineChart
        className={inModal ? "border-0 bg-background p-0" : undefined}
        ariaLabel="Line chart showing the share of annual activity accounted for by the top five target industries."
        data={concentrationTrend.data}
        describedBy={concentrationDescriptionId}
        lineColor={SECTOR_CONCENTRATION_LINE_COLOR}
        tableId={concentrationTableId}
      />
    </Suspense>
  );

  return (
    <div className="space-y-6">
      <ChartPanel
        id={compositionDescriptionId}
        title="Industry Composition Over Time"
        description={
          <span className="prose-copy">Share of deals by target industry over time.</span>
        }
        controls={
          <MetricSwitch<OwnershipMetric>
            label="Metric"
            value={compositionMetric}
            onValueChange={setCompositionMetric}
            options={VALUE_METRIC_OPTIONS}
            ariaLabel="Industry composition metric"
          />
        }
        mobileExpand={() => compositionChart(true)}
      >
        <p id={compositionTableId} className="sr-only">
          100 percent stacked area chart showing target-industry composition by filing year.
        </p>
        {compositionChart(false)}
      </ChartPanel>

      <ChartPanel
        id={pairingsDescriptionId}
        title="Top Industry Pairings"
        description={
          <span className="prose-copy">
            Most common target and acquirer industry combinations.
          </span>
        }
        controls={
          <MetricSwitch<OwnershipMetric>
            label="Heatmap metric"
            value={pairingsMetric}
            onValueChange={setPairingsMetric}
            options={VALUE_METRIC_OPTIONS}
            ariaLabel="Industry pairing metric"
          />
        }
      >
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
      </ChartPanel>

      <ChartPanel
        id={concentrationDescriptionId}
        title="Sector Concentration Trend"
        description={
          <span className="prose-copy">
            Share of annual activity accounted for by the five largest target industries.
          </span>
        }
        controls={
          <MetricSwitch<OwnershipMetric>
            label="Metric"
            value={concentrationMetric}
            onValueChange={setConcentrationMetric}
            options={VALUE_METRIC_OPTIONS}
            ariaLabel="Sector concentration metric"
          />
        }
        mobileExpand={() => concentrationChart(true)}
        footnote={
          <>
            Current top-five basket share:{" "}
            <span className="font-mono tabular-nums text-foreground">
              {concentrationTrend.data.length > 0
                ? formatPercent(
                    concentrationTrend.data[concentrationTrend.data.length - 1].share,
                  )
                : "—"}
            </span>
          </>
        }
      >
        <div className="space-y-4">
          <p id={concentrationTableId} className="sr-only">
            Line chart showing the share of annual activity accounted for by the top five target
            industries.
          </p>
          {concentrationChart(false)}
          <div className="flex flex-wrap gap-2">
            {concentrationTrend.topIndustries.map((industry) => (
              <Badge key={industry} variant="metadata">
                {formatIndustryLabel(industry)}
              </Badge>
            ))}
          </div>
        </div>
      </ChartPanel>
    </div>
  );
}

export default function TrendsAnalyses() {
  const { data, isLoading: loading, error, apiError, refetch } = useAgreementTrends();

  return (
    <PageShell
      layout="data"
      title="Trends & Analyses"
      description={
        <span className="prose-copy">
          A set of deeper cuts on ownership structure and sector composition, built from
          public-eligible agreements and designed to complement the headline leaderboards.
        </span>
      }
    >
      <div className="space-y-6">
        {loading ? <TrendsSkeleton /> : null}

        {!loading && error ? <ErrorState error={apiError ?? error} onRetry={() => void refetch()} /> : null}

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
