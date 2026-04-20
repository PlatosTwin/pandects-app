import { useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  XAxis,
  YAxis,
} from "recharts";

import { cn } from "@/lib/utils";
import {
  type ChartConfig,
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import {
  buildVerticalYearCoordinatesGenerator,
  buildYearAxisGuides,
} from "@/lib/year-axis";

export type TrendsChartSeries = {
  color: string;
  key: string;
  label: string;
};

export type TrendsStackedRow = {
  year: number;
} & Record<string, number>;

type TrendsStackedShareAreaChartProps = {
  ariaLabel: string;
  className?: string;
  data: TrendsStackedRow[];
  describedBy: string;
  series: TrendsChartSeries[];
  tableId: string;
  valueFormatter: (value: number) => string;
};

type TrendsMedianBandChartRow = {
  year: number;
  public_low: number;
  public_band: number;
  public_median: number | null;
  private_low: number;
  private_band: number;
  private_median: number | null;
};

type TrendsMedianBandChartProps = {
  ariaLabel: string;
  className?: string;
  data: TrendsMedianBandChartRow[];
  describedBy: string;
  tableId: string;
  valueFormatter: (value: number) => string;
};

type TrendsPercentLineChartRow = {
  share: number;
  year: number;
};

type TrendsPercentLineChartProps = {
  ariaLabel: string;
  className?: string;
  data: TrendsPercentLineChartRow[];
  describedBy: string;
  lineColor: string;
  tableId: string;
};

type TrendsHeatmapCell = {
  displayValue: string;
  intensity: number;
  rawValue: number | null;
};

type TrendsHeatmapTableProps = {
  caption: string;
  className?: string;
  columns: string[];
  formatterLabel: string;
  getCell: (row: string, column: string) => TrendsHeatmapCell;
  rows: string[];
};

const PERCENT_AXIS_TICKS = [0, 25, 50, 75, 100];
const MEDIAN_CHART_CONFIG = {
  public_median: {
    label: "Public targets",
    color: "hsl(212 93% 50%)",
  },
  private_median: {
    label: "Private targets",
    color: "hsl(170 84% 36%)",
  },
} satisfies ChartConfig;

function chartColorVar(key: string) {
  return `var(--color-${key.replace(/[^a-zA-Z0-9_-]/g, "")})`;
}

function TrendsMedianBandTooltipContent({
  active,
  payload,
  valueFormatter,
}: {
  active?: boolean;
  payload?: Array<{ payload?: TrendsMedianBandChartRow }>;
  valueFormatter: (value: number) => string;
}) {
  const row = payload?.[0]?.payload;
  if (!active || !row) {
    return null;
  }

  const renderBucket = (
    title: string,
    p25: number | null,
    median: number | null,
    p75: number | null,
  ) => (
    <div className="space-y-1">
      <div className="font-medium text-foreground">{title}</div>
      <div className="font-mono text-xs text-muted-foreground">
        P25 {p25 !== null ? valueFormatter(p25) : "—"} · Median{" "}
        {median !== null ? valueFormatter(median) : "—"} · P75{" "}
        {p75 !== null ? valueFormatter(p75) : "—"}
      </div>
    </div>
  );

  return (
    <div className="grid min-w-[12rem] gap-3 rounded-lg border border-border/50 bg-background px-3 py-2 text-xs shadow-xl">
      <div className="font-medium text-foreground">Filing year {row.year}</div>
      {renderBucket(
        "Public targets",
        row.public_median !== null ? row.public_low : null,
        row.public_median,
        row.public_median !== null ? row.public_low + row.public_band : null,
      )}
      {renderBucket(
        "Private targets",
        row.private_median !== null ? row.private_low : null,
        row.private_median,
        row.private_median !== null ? row.private_low + row.private_band : null,
      )}
    </div>
  );
}

export function TrendsStackedShareAreaChart({
  ariaLabel,
  className,
  data,
  describedBy,
  series,
  tableId,
  valueFormatter,
}: TrendsStackedShareAreaChartProps) {
  const chartConfig = useMemo<ChartConfig>(
    () =>
      series.reduce<ChartConfig>((acc, item) => {
        acc[item.key] = {
          label: item.label,
          color: item.color,
        };
        return acc;
      }, {}),
    [series],
  );

  const dataByYear = useMemo(
    () => new Map(data.map((row) => [row.year, row])),
    [data],
  );

  const percentData = useMemo(() => {
    return data.map((row) => {
      const total = series.reduce(
        (sum, item) => sum + Number(row[item.key] ?? 0),
        0,
      );
      const nextRow: TrendsStackedRow = { year: row.year };
      series.forEach((item) => {
        const value = Number(row[item.key] ?? 0);
        nextRow[item.key] =
          total > 0 ? Math.round((value / total) * 1000) / 10 : 0;
      });
      return nextRow;
    });
  }, [data, series]);
  const yearAxisGuides = useMemo(
    () => buildYearAxisGuides(percentData),
    [percentData],
  );

  return (
    <div className={cn("rounded-lg border border-border bg-muted/20 p-3", className)}>
      <ChartContainer
        className="h-[260px] w-full min-w-0 aspect-auto sm:h-[320px] lg:h-[360px]"
        config={chartConfig}
        role="img"
        aria-label={ariaLabel}
        aria-describedby={`${describedBy} ${tableId}`}
      >
        <AreaChart data={percentData} margin={{ top: 6, right: 24, left: 8, bottom: 0 }}>
          <CartesianGrid vertical={false} stroke="hsl(var(--border) / 0.4)" />
          <CartesianGrid
            horizontal={false}
            stroke="hsl(var(--border) / 0.18)"
            strokeDasharray="2 4"
            verticalCoordinatesGenerator={buildVerticalYearCoordinatesGenerator(
              yearAxisGuides.minorYears,
            )}
          />
          <CartesianGrid
            horizontal={false}
            stroke="hsl(var(--border) / 0.32)"
            verticalCoordinatesGenerator={buildVerticalYearCoordinatesGenerator(
              yearAxisGuides.majorYears,
            )}
          />
          <XAxis
            dataKey="year"
            type="number"
            allowDecimals={false}
            domain={["dataMin", "dataMax"]}
            ticks={yearAxisGuides.majorYears}
            tickFormatter={(value) => String(value)}
            tickMargin={6}
            minTickGap={8}
            interval={0}
          />
          <YAxis
            domain={[0, 100]}
            allowDecimals={false}
            width={44}
            tickMargin={6}
            ticks={PERCENT_AXIS_TICKS}
            tickFormatter={(value) => `${Math.round(Number(value))}%`}
          />
          <ChartTooltip
            content={
              <ChartTooltipContent
                indicator="dashed"
                labelFormatter={(_, payload) => {
                  const year = payload?.[0]?.payload?.year;
                  return `Filing year ${year ?? "—"}`;
                }}
                formatter={(value, _name, item) => {
                  const indicatorColor = item?.payload?.fill || item?.color;
                  const payload = item?.payload as
                    | { year?: number }
                    | undefined;
                  const year = Number(payload?.year ?? NaN);
                  const dataKey =
                    typeof item.dataKey === "string" ? item.dataKey : "";
                  const rawRow = Number.isFinite(year) ? dataByYear.get(year) : undefined;
                  const rawValue = Number(
                    rawRow && dataKey ? (rawRow[dataKey] ?? 0) : 0,
                  );

                  return (
                    <div className="flex items-center gap-3">
                      <span
                        className="inline-block h-2.5 w-2.5 shrink-0 rounded-[2px]"
                        style={
                          indicatorColor
                            ? { backgroundColor: indicatorColor }
                            : undefined
                        }
                        aria-hidden="true"
                      />
                      <span className="w-72 shrink-0 truncate text-left text-foreground">
                        {item.name}
                      </span>
                      <span className="w-10 shrink-0 text-right font-mono font-medium tabular-nums text-foreground">
                        {valueFormatter(rawValue)}
                      </span>
                      <span className="w-24 shrink-0 text-right font-mono text-xs tabular-nums text-muted-foreground">
                        {Number(value).toFixed(1)}% of year
                      </span>
                    </div>
                  );
                }}
              />
            }
          />
          <ChartLegend
            content={
              <ChartLegendContent className="flex-wrap justify-start gap-x-4 gap-y-1 sm:justify-center" />
            }
          />
          {series.map((item) => (
            <Area
              key={item.key}
              dataKey={item.key}
              type="monotone"
              stackId="share"
              stroke={chartColorVar(item.key)}
              fill={chartColorVar(item.key)}
              fillOpacity={0.95}
              name={item.label}
            />
          ))}
        </AreaChart>
      </ChartContainer>
    </div>
  );
}

export function TrendsMedianBandChart({
  ariaLabel,
  className,
  data,
  describedBy,
  tableId,
  valueFormatter,
}: TrendsMedianBandChartProps) {
  const yearAxisGuides = useMemo(
    () => buildYearAxisGuides(data),
    [data],
  );

  return (
    <div className={cn("rounded-lg border border-border bg-muted/20 p-3", className)}>
      <ChartContainer
        className="h-[260px] w-full min-w-0 aspect-auto sm:h-[320px] lg:h-[360px]"
        config={MEDIAN_CHART_CONFIG}
        role="img"
        aria-label={ariaLabel}
        aria-describedby={`${describedBy} ${tableId}`}
      >
        <AreaChart data={data} margin={{ top: 6, right: 24, left: 8, bottom: 0 }}>
          <CartesianGrid vertical={false} stroke="hsl(var(--border) / 0.4)" />
          <CartesianGrid
            horizontal={false}
            stroke="hsl(var(--border) / 0.18)"
            strokeDasharray="2 4"
            verticalCoordinatesGenerator={buildVerticalYearCoordinatesGenerator(
              yearAxisGuides.minorYears,
            )}
          />
          <CartesianGrid
            horizontal={false}
            stroke="hsl(var(--border) / 0.32)"
            verticalCoordinatesGenerator={buildVerticalYearCoordinatesGenerator(
              yearAxisGuides.majorYears,
            )}
          />
          <XAxis
            dataKey="year"
            type="number"
            allowDecimals={false}
            domain={["dataMin", "dataMax"]}
            ticks={yearAxisGuides.majorYears}
            tickFormatter={(value) => String(value)}
            tickMargin={6}
            minTickGap={8}
            interval={0}
          />
          <YAxis
            tickMargin={6}
            width={68}
            tickFormatter={(value) => valueFormatter(Number(value))}
          />
          <ChartTooltip content={<TrendsMedianBandTooltipContent valueFormatter={valueFormatter} />} />
          <ChartLegend
            content={
              <ChartLegendContent className="flex-wrap justify-start gap-x-4 gap-y-1 sm:justify-center" />
            }
          />
          <Area
            dataKey="public_low"
            stackId="public"
            stroke="transparent"
            fill="transparent"
            name="Public targets"
            isAnimationActive={false}
          />
          <Area
            dataKey="public_band"
            stackId="public"
            stroke="transparent"
            fill="var(--color-public_median)"
            fillOpacity={0.18}
            name="Public targets"
            isAnimationActive={false}
          />
          <Area
            dataKey="private_low"
            stackId="private"
            stroke="transparent"
            fill="transparent"
            name="Private targets"
            isAnimationActive={false}
          />
          <Area
            dataKey="private_band"
            stackId="private"
            stroke="transparent"
            fill="var(--color-private_median)"
            fillOpacity={0.18}
            name="Private targets"
            isAnimationActive={false}
          />
          <Line
            dataKey="public_median"
            type="monotone"
            stroke="var(--color-public_median)"
            strokeWidth={2.5}
            dot={{ r: 3, fill: "var(--color-public_median)" }}
            activeDot={{ r: 4 }}
            connectNulls={false}
            name="Public targets"
          />
          <Line
            dataKey="private_median"
            type="monotone"
            stroke="var(--color-private_median)"
            strokeWidth={2.5}
            dot={{ r: 3, fill: "var(--color-private_median)" }}
            activeDot={{ r: 4 }}
            connectNulls={false}
            name="Private targets"
          />
        </AreaChart>
      </ChartContainer>
    </div>
  );
}

export function TrendsPercentLineChart({
  ariaLabel,
  className,
  data,
  describedBy,
  lineColor,
  tableId,
}: TrendsPercentLineChartProps) {
  const yearAxisGuides = useMemo(
    () => buildYearAxisGuides(data),
    [data],
  );

  return (
    <div className={cn("rounded-lg border border-border bg-muted/20 p-3", className)}>
      <ChartContainer
        className="h-[220px] w-full min-w-0 aspect-auto sm:h-[280px] lg:h-[320px]"
        config={{
          share: {
            label: "Share",
            color: lineColor,
          },
        }}
        role="img"
        aria-label={ariaLabel}
        aria-describedby={`${describedBy} ${tableId}`}
      >
        <LineChart data={data} margin={{ top: 6, right: 24, left: 8, bottom: 0 }}>
          <CartesianGrid vertical={false} stroke="hsl(var(--border) / 0.4)" />
          <CartesianGrid
            horizontal={false}
            stroke="hsl(var(--border) / 0.18)"
            strokeDasharray="2 4"
            verticalCoordinatesGenerator={buildVerticalYearCoordinatesGenerator(
              yearAxisGuides.minorYears,
            )}
          />
          <CartesianGrid
            horizontal={false}
            stroke="hsl(var(--border) / 0.32)"
            verticalCoordinatesGenerator={buildVerticalYearCoordinatesGenerator(
              yearAxisGuides.majorYears,
            )}
          />
          <XAxis
            dataKey="year"
            type="number"
            allowDecimals={false}
            domain={["dataMin", "dataMax"]}
            ticks={yearAxisGuides.majorYears}
            tickFormatter={(value) => String(value)}
            tickMargin={6}
            minTickGap={8}
            interval={0}
          />
          <YAxis
            domain={[0, 100]}
            allowDecimals={false}
            width={44}
            tickMargin={6}
            ticks={PERCENT_AXIS_TICKS}
            tickFormatter={(value) => `${Math.round(Number(value))}%`}
          />
          <ChartTooltip
            content={
              <ChartTooltipContent
                indicator="line"
                labelFormatter={(_, payload) => {
                  const year = payload?.[0]?.payload?.year;
                  return `Filing year ${year ?? "—"}`;
                }}
                formatter={(value) => (
                  <span className="font-mono font-medium tabular-nums text-foreground">
                    {Number(value).toFixed(1)}%
                  </span>
                )}
              />
            }
          />
          <Line
            dataKey="share"
            type="monotone"
            stroke="var(--color-share)"
            strokeWidth={2.5}
            dot={{ r: 3, fill: "var(--color-share)" }}
            activeDot={{ r: 4 }}
            name="Share"
          />
        </LineChart>
      </ChartContainer>
    </div>
  );
}

export function TrendsHeatmapTable({
  caption,
  className,
  columns,
  formatterLabel,
  getCell,
  rows,
}: TrendsHeatmapTableProps) {
  return (
    <div className={cn("overflow-x-auto rounded-lg border border-border bg-background/80", className)}>
      <table className="w-full min-w-[56rem] table-fixed border-collapse text-sm">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="border-b border-border">
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
            <tr key={row} className="border-b border-border/50 last:border-0">
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
