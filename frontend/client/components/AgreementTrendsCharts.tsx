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

  return (
    <div className={cn("rounded-lg border border-border/60 bg-muted/20 p-3", className)}>
      <ChartContainer
        className="h-[260px] w-full min-w-0 aspect-auto sm:h-[320px] lg:h-[360px]"
        config={chartConfig}
        role="img"
        aria-label={ariaLabel}
        aria-describedby={`${describedBy} ${tableId}`}
      >
        <AreaChart data={percentData} margin={{ top: 6, right: 24, left: 8, bottom: 0 }}>
          <CartesianGrid vertical={false} />
          <XAxis
            dataKey="year"
            type="number"
            allowDecimals={false}
            domain={["dataMin", "dataMax"]}
            tickFormatter={(value) => String(value)}
            tickMargin={6}
            minTickGap={16}
            interval="preserveStartEnd"
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
                    <div className="grid grid-cols-[auto_minmax(0,6rem)_minmax(0,1fr)] items-center gap-x-3">
                      <span
                        className="inline-block h-2.5 w-2.5 shrink-0 rounded-[2px]"
                        style={
                          indicatorColor
                            ? { backgroundColor: indicatorColor }
                            : undefined
                        }
                        aria-hidden="true"
                      />
                      <span className="text-left font-mono font-medium tabular-nums text-foreground">
                        {valueFormatter(rawValue)}
                      </span>
                      <span className="text-right font-mono text-xs tabular-nums text-muted-foreground">
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
              stroke={`var(--color-${item.key})`}
              fill={`var(--color-${item.key})`}
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
  return (
    <div className={cn("rounded-lg border border-border/60 bg-muted/20 p-3", className)}>
      <ChartContainer
        className="h-[260px] w-full min-w-0 aspect-auto sm:h-[320px] lg:h-[360px]"
        config={MEDIAN_CHART_CONFIG}
        role="img"
        aria-label={ariaLabel}
        aria-describedby={`${describedBy} ${tableId}`}
      >
        <AreaChart data={data} margin={{ top: 6, right: 24, left: 8, bottom: 0 }}>
          <CartesianGrid vertical={false} />
          <XAxis
            dataKey="year"
            type="number"
            allowDecimals={false}
            domain={["dataMin", "dataMax"]}
            tickFormatter={(value) => String(value)}
            tickMargin={6}
            minTickGap={16}
            interval="preserveStartEnd"
          />
          <YAxis
            tickMargin={6}
            width={68}
            tickFormatter={(value) => valueFormatter(Number(value))}
          />
          <ChartTooltip
            content={
              <ChartTooltipContent
                indicator="dashed"
                labelFormatter={(_, payload) => {
                  const year = payload?.[0]?.payload?.year;
                  return `Filing year ${year ?? "—"}`;
                }}
                formatter={(_value, name, item) => {
                  const payload = item?.payload as TrendsMedianBandChartRow | undefined;
                  if (!payload) return null;
                  if (name === "Public targets") {
                    return (
                      <div className="space-y-1">
                        <div className="font-medium text-foreground">Public targets</div>
                        <div className="font-mono text-xs text-muted-foreground">
                          P25 {valueFormatter(payload.public_low)} · Median{" "}
                          {valueFormatter(payload.public_median)} · P75{" "}
                          {valueFormatter(payload.public_low + payload.public_band)}
                        </div>
                      </div>
                    );
                  }
                  if (name === "Private targets") {
                    return (
                      <div className="space-y-1">
                        <div className="font-medium text-foreground">Private targets</div>
                        <div className="font-mono text-xs text-muted-foreground">
                          P25 {valueFormatter(payload.private_low)} · Median{" "}
                          {valueFormatter(payload.private_median)} · P75{" "}
                          {valueFormatter(payload.private_low + payload.private_band)}
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
            }
          />
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
  return (
    <div className={cn("rounded-lg border border-border/60 bg-muted/20 p-3", className)}>
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
          <CartesianGrid vertical={false} />
          <XAxis
            dataKey="year"
            type="number"
            allowDecimals={false}
            domain={["dataMin", "dataMax"]}
            tickFormatter={(value) => String(value)}
            tickMargin={6}
            minTickGap={16}
            interval="preserveStartEnd"
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
    <div className={cn("overflow-x-auto rounded-lg border border-border/60 bg-background/80", className)}>
      <table className="w-full min-w-[38rem] border-collapse text-sm">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="border-b border-border/60">
            <th className="px-3 py-2 text-left font-semibold text-foreground">
              Segment
            </th>
            {columns.map((column) => (
              <th
                key={column}
                className="px-3 py-2 text-center font-semibold text-foreground"
              >
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row} className="border-b border-border/40 last:border-0">
              <th className="px-3 py-3 text-left font-medium text-foreground">
                {row}
              </th>
              {columns.map((column) => {
                const cell = getCell(row, column);
                const opacity = 0.1 + (cell.intensity * 0.75);
                const backgroundColor = `hsl(212 93% 50% / ${opacity})`;
                const foregroundClass =
                  cell.intensity > 0.58 ? "text-white" : "text-foreground";

                return (
                  <td key={`${row}-${column}`} className="px-2 py-2">
                    <div
                      className={cn(
                        "rounded-md border border-border/50 px-3 py-3 text-center shadow-sm transition-colors",
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
