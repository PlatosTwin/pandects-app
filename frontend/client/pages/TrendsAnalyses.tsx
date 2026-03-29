import { useEffect, useId, useMemo, useState } from "react";

import {
  TrendsHeatmapTable,
  TrendsMedianBandChart,
  TrendsPercentLineChart,
  TrendsStackedShareAreaChart,
  type TrendsChartSeries,
} from "@/components/AgreementTrendsCharts";
import { PageShell } from "@/components/PageShell";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";
import { formatCurrencyValue, formatEnumValue } from "@/lib/format-utils";

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

type LabeledKey = {
  key: string;
  label: string;
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
  return formatCurrencyValue(value ?? null).replace(".00", "");
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
            <CardTitle className="text-xl sm:text-2xl">
              Public vs. Private Target Mix Over Time
            </CardTitle>
            <CardDescription id={mixDescriptionId} className="max-w-3xl text-sm sm:text-base">
              Tracks how the target mix shifts by filing year. PE-backed targets are grouped into
              the private bucket, and the metric toggle swaps between share of deal count and share
              of reported transaction value.
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
          <TrendsStackedShareAreaChart
            ariaLabel="100 percent stacked area chart showing public-target and private-target share by filing year."
            data={mixChartData}
            describedBy={mixDescriptionId}
            series={OWNERSHIP_SERIES}
            tableId={mixTableId}
            valueFormatter={mixMetric === "deal_count" ? formatCount : formatMoney}
          />
        </CardContent>
      </Card>

      <Card variant="subtle">
        <CardHeader>
          <CardTitle className="text-xl sm:text-2xl">
            Public vs. Private Deal Size
          </CardTitle>
          <CardDescription
            id={dealSizeDescriptionId}
            className="max-w-3xl text-sm sm:text-base"
          >
            Median reported deal value by year, with an interquartile band to show how tightly or
            loosely transaction sizes cluster for each target bucket.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p id={dealSizeTableId} className="sr-only">
            Line chart showing median reported deal size with 25th to 75th percentile bands for
            public and private targets by filing year.
          </p>
          <TrendsMedianBandChart
            ariaLabel="Line chart showing median reported deal size with percentile bands for public and private targets by filing year."
            data={dealSizeChartData}
            describedBy={dealSizeDescriptionId}
            tableId={dealSizeTableId}
            valueFormatter={(value) => formatMoney(value)}
          />
        </CardContent>
      </Card>

      <Card variant="subtle">
        <CardHeader className="gap-4 md:flex-row md:items-end md:justify-between">
          <div className="space-y-2">
            <CardTitle className="text-xl sm:text-2xl">
              Target Type by Buyer Type
            </CardTitle>
            <CardDescription id={matrixDescriptionId} className="max-w-3xl text-sm sm:text-base">
              A compact cross-section of who buys what: public and private targets on the rows,
              buyer buckets on the columns, with private equity sourced from the `acquirer_pe`
              field rather than inferred from type text.
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
      topIndustries.forEach((industry) => {
        nextRow[industry] = 0;
      });
      industries.target_industries_by_year
        .filter((row) => row.year === year)
        .forEach((row) => {
          const metricValue =
            compositionMetric === "deal_count"
              ? row.deal_count
              : row.total_transaction_value;
          if (topIndustries.includes(row.industry)) {
            nextRow[row.industry] = metricValue;
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
      key: industry,
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
            <CardTitle className="text-xl sm:text-2xl">
              Industry Composition Over Time
            </CardTitle>
            <CardDescription
              id={compositionDescriptionId}
              className="max-w-3xl text-sm sm:text-base"
            >
              Shows how target-industry mix evolves across filing years. The chart keeps the top
              industries in view and rolls the long tail into an `Other` bucket to stay readable on
              desktop and mobile.
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
        </CardContent>
      </Card>

      <Card variant="subtle">
        <CardHeader className="gap-4 md:flex-row md:items-end md:justify-between">
          <div className="space-y-2">
            <CardTitle className="text-xl sm:text-2xl">Top Industry Pairings</CardTitle>
            <CardDescription
              id={pairingsDescriptionId}
              className="max-w-3xl text-sm sm:text-base"
            >
              A ranked matrix of target and acquirer industry intersections. Darker cells indicate
              the pairings where activity or value is concentrating most heavily.
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
            <CardTitle className="text-xl sm:text-2xl">Sector Concentration Trend</CardTitle>
            <CardDescription
              id={concentrationDescriptionId}
              className="max-w-3xl text-sm sm:text-base"
            >
              Tracks how much of annual activity sits inside the top five target industries. Rising
              share means M&amp;A is becoming more concentrated in a narrower set of sectors.
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
          <TrendsPercentLineChart
            ariaLabel="Line chart showing the share of annual activity accounted for by the top five target industries."
            data={concentrationTrend.data}
            describedBy={concentrationDescriptionId}
            lineColor="hsl(12 76% 61%)"
            tableId={concentrationTableId}
          />
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
  const [data, setData] = useState<AgreementTrendsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    const fetchTrends = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await authFetch(apiUrl("v1/agreement-trends"), {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Agreement trends request failed (${response.status})`);
        }
        const nextData = (await response.json()) as AgreementTrendsResponse;
        if (!cancelled) {
          setData(nextData);
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
        <Card variant="subtle">
          <CardHeader>
            <CardTitle className="text-xl">Method</CardTitle>
            <CardDescription className="max-w-4xl text-sm sm:text-base">
              Ownership views use `target_type`, `acquirer_type`, `target_pe`, and `acquirer_pe`.
              Public/private target charts collapse PE-backed targets into the private bucket, while
              the buyer matrix breaks out private-equity acquirers separately. Industry views use
              the existing target and acquirer industry fields and emphasize compositional change
              rather than raw table scans.
            </CardDescription>
          </CardHeader>
        </Card>

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
