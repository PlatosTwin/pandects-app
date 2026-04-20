import { useMemo } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
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

export type CounselLeaderboardChartSeries = {
  key: string;
  label: string;
  color: string;
};

export type CounselLeaderboardChartRow = {
  year: number;
} & Record<string, number>;

type CounselLeaderboardChartProps = {
  className?: string;
  data: CounselLeaderboardChartRow[];
  describedBy: string;
  metricLabel: string;
  series: CounselLeaderboardChartSeries[];
  tableId: string;
  valueFormatter: (value: number) => string;
};

const PERCENT_AXIS_TICKS = [0, 25, 50, 75, 100];

export function CounselLeaderboardChart({
  className,
  data,
  describedBy,
  metricLabel,
  series,
  tableId,
  valueFormatter,
}: CounselLeaderboardChartProps) {
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
      const nextRow: CounselLeaderboardChartRow = { year: row.year };
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
    <div
      className={cn(
        "rounded-lg border border-border bg-muted/20 p-3",
        className,
      )}
    >
      <ChartContainer
        className="h-[260px] w-full min-w-0 aspect-auto sm:h-[320px] lg:h-[380px]"
        config={chartConfig}
        role="img"
        aria-label={`100 percent stacked area chart showing ${metricLabel} share by filing year for the selected counsel firms.`}
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
                  const rawRow = Number.isFinite(year)
                    ? dataByYear.get(year)
                    : undefined;
                  const rawValue = Number(
                    rawRow && dataKey ? (rawRow[dataKey] ?? 0) : 0,
                  );
                  const rawTotal = series.reduce(
                    (sum, currentSeries) =>
                      sum + Number(rawRow?.[currentSeries.key] ?? 0),
                    0,
                  );
                  const pct =
                    rawTotal > 0
                      ? Math.round((rawValue / rawTotal) * 1000) / 10
                      : 0;
                  const valueNumber = Number(value);

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
                        {valueNumber.toFixed(1)}% of year
                        {pct !== valueNumber ? ` (${pct.toFixed(1)}%)` : ""}
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
              stackId="leaderboard"
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
