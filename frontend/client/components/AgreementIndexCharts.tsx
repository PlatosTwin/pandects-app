import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts";

import { cn } from "@/lib/utils";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  type ChartConfig,
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";

type YearRange = {
  minYear: number;
  maxYear: number;
} | null;

type ProcessingStatusRow = {
  year: number;
  processed: number;
  staged: number;
  awaiting: number;
  notPaginated: number;
};

type DealTypeChartMode = "count" | "percent";

type DealTypeSeries = {
  dealType: string;
  key: string;
  label: string;
  color: string;
  total: number;
};

type DealTypeChartRow = {
  year: number;
} & Record<string, number>;

type ProcessingStatusChartProps = {
  className?: string;
  data: ProcessingStatusRow[];
  describedBy: string;
  isMobile: boolean;
  showSourceSplit: boolean;
  tableId: string;
  yearRange: YearRange;
  yearTicks?: number[];
};

type DealTypesChartProps = {
  className?: string;
  data: DealTypeChartRow[];
  describedBy: string;
  isMobile: boolean;
  mode: DealTypeChartMode;
  onModeChange: (value: DealTypeChartMode) => void;
  series: DealTypeSeries[];
  showSourceSplit: boolean;
  tableId: string;
  yearRange: YearRange;
  yearTicks?: number[];
};

const PERCENT_AXIS_TICKS = [0, 20, 40, 60, 80, 100];

function SplitMarkerLabel({
  isMobile,
  viewBox,
}: {
  isMobile: boolean;
  viewBox?: { x: number; y: number } | null;
}) {
  if (!viewBox) return null;

  const text = isMobile ? "2020/21" : "2020/2021 split";
  const paddingX = isMobile ? 6 : 4;
  const rectHeight = isMobile ? 16 : 14;
  const charWidth = isMobile ? 6.1 : 5.4;
  const textWidth = text.length * charWidth;
  const rectWidth = textWidth + paddingX * 2;
  const rectX = viewBox.x - rectWidth - 8;
  const rectY = viewBox.y + 8;
  const textX = rectX + rectWidth / 2;
  const textY = rectY + rectHeight / 2 + 0.5;

  return (
    <g pointerEvents="none">
      <rect
        x={rectX}
        y={rectY}
        width={rectWidth}
        height={rectHeight}
        rx={4}
        fill="hsl(var(--background))"
        stroke="hsl(var(--border))"
      />
      <text
        x={textX}
        y={textY}
        fill="hsl(var(--muted-foreground))"
        fontSize={10}
        textAnchor="middle"
        dominantBaseline="middle"
      >
        {text}
      </text>
    </g>
  );
}

export function ProcessingStatusChart({
  className,
  data,
  describedBy,
  isMobile,
  showSourceSplit,
  tableId,
  yearRange,
  yearTicks,
}: ProcessingStatusChartProps) {
  return (
    <div
      className={cn(
        "rounded-lg border border-border bg-muted/20 p-3",
        className,
      )}
    >
      <ChartContainer
        className="h-[240px] w-full min-w-0 aspect-auto sm:h-[300px] lg:h-[340px]"
        config={{
          processed: {
            label: "Processed",
            color: "hsl(142 71% 45%)",
          },
          staged: {
            label: "Staged",
            color: "hsl(38 92% 55%)",
          },
          awaiting: {
            label: "Awaiting validation",
            color: "hsl(0 84% 60%)",
          },
          notPaginated: {
            label: "Not paginated",
            color: "hsl(220 9% 60%)",
          },
        }}
        role="img"
        aria-label="Stacked bar chart showing processed, staged, awaiting validation, and not paginated agreements by filing year."
        aria-describedby={`${describedBy} ${tableId}`}
      >
        <BarChart data={data} margin={{ top: 6, right: 24, left: 8, bottom: 0 }}>
          <CartesianGrid vertical={false} />
          <XAxis
            dataKey="year"
            type="number"
            allowDecimals={false}
            domain={yearRange ? [yearRange.minYear, yearRange.maxYear] : ["dataMin", "dataMax"]}
            padding={{ left: 20, right: 20 }}
            tickFormatter={(value) => String(value)}
            tickMargin={6}
            minTickGap={16}
            interval="preserveStartEnd"
            ticks={yearTicks}
          />
          <YAxis allowDecimals={false} tickMargin={6} width={32} />
          <ChartTooltip
            content={
              <ChartTooltipContent
                indicator="dashed"
                labelFormatter={(_, payload) => {
                  const year = payload?.[0]?.payload?.year;
                  return `Filing year ${year ?? "—"}`;
                }}
                formatter={(value, name, item) => {
                  const indicatorColor = item?.payload?.fill || item?.color;
                  const payload = item?.payload as
                    | ProcessingStatusRow
                    | undefined;
                  const processed = Number(payload?.processed ?? 0);
                  const staged = Number(payload?.staged ?? 0);
                  const awaiting = Number(payload?.awaiting ?? 0);
                  const notPaginated = Number(payload?.notPaginated ?? 0);
                  const total = processed + staged + awaiting + notPaginated;
                  const getPct = (count: number) =>
                    total > 0 ? Math.round((count / total) * 1000) / 10 : 0;
                  const countValue = Number(value);
                  const pct =
                    name === "Processed"
                      ? getPct(processed)
                      : name === "Staged"
                        ? getPct(staged)
                        : name === "Awaiting validation"
                          ? getPct(awaiting)
                          : getPct(notPaginated);

                  return (
                    <div className="grid grid-cols-[auto_minmax(0,4.5rem)_minmax(0,1fr)] items-center gap-x-3">
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
                        {countValue.toLocaleString()}
                      </span>
                      <span className="text-right font-mono text-xs tabular-nums text-muted-foreground">
                        {pct.toFixed(1)}%
                      </span>
                    </div>
                  );
                }}
              />
            }
          />
          <ChartLegend content={<ChartLegendContent />} />
          <Bar
            dataKey="processed"
            stackId="agreements"
            fill="var(--color-processed)"
            name="Processed"
          />
          <Bar
            dataKey="staged"
            stackId="agreements"
            fill="var(--color-staged)"
            name="Staged"
          />
          <Bar
            dataKey="awaiting"
            stackId="agreements"
            fill="var(--color-awaiting)"
            name="Awaiting validation"
          />
          <Bar
            dataKey="notPaginated"
            stackId="agreements"
            fill="var(--color-notPaginated)"
            name="Not paginated"
          />
          {showSourceSplit ? (
            <ReferenceLine
              x={2020.5}
              stroke="hsl(var(--foreground))"
              strokeWidth={1.5}
              strokeDasharray="4 4"
              isFront
              label={(props) => (
                <SplitMarkerLabel
                  isMobile={isMobile}
                  viewBox={props.viewBox ?? null}
                />
              )}
            />
          ) : null}
        </BarChart>
      </ChartContainer>
    </div>
  );
}

export function DealTypesChart({
  className,
  data,
  describedBy,
  isMobile,
  mode,
  onModeChange,
  series,
  showSourceSplit,
  tableId,
  yearRange,
  yearTicks,
}: DealTypesChartProps) {
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

  const displayData = useMemo(() => {
    if (mode === "count") return data;

    return data.map((row) => {
      const total = series.reduce(
        (sum, item) => sum + Number(row[item.key] ?? 0),
        0,
      );
      const nextRow: DealTypeChartRow = { year: row.year };
      series.forEach((item) => {
        const value = Number(row[item.key] ?? 0);
        nextRow[item.key] =
          total > 0 ? Math.round((value / total) * 1000) / 10 : 0;
      });
      return nextRow;
    });
  }, [data, mode, series]);

  return (
    <div
      className={cn(
        "rounded-lg border border-border bg-muted/20 p-3",
        className,
      )}
    >
      <div className="mb-3 flex items-center justify-end gap-2">
        <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          Chart mode
        </span>
        <ToggleGroup
          type="single"
          value={mode}
          onValueChange={(value) => {
            if (value === "count" || value === "percent") {
              onModeChange(value);
            }
          }}
          variant="outline"
          size="xs"
          aria-label="Deal type chart mode"
          className="justify-start"
        >
          <ToggleGroupItem
            value="count"
            aria-label="Stacked counts"
            className="text-muted-foreground data-[state=on]:text-foreground"
          >
            Counts
          </ToggleGroupItem>
          <ToggleGroupItem
            value="percent"
            aria-label="100 percent stacked"
            className="text-muted-foreground data-[state=on]:text-foreground"
          >
            100%
          </ToggleGroupItem>
        </ToggleGroup>
      </div>
      <ChartContainer
        className="h-[240px] w-full min-w-0 aspect-auto sm:h-[300px] lg:h-[340px]"
        config={chartConfig}
        role="img"
        aria-label={
          mode === "percent"
            ? "100 percent stacked bar chart showing deal type share by filing year."
            : "Stacked bar chart showing deal type counts by filing year."
        }
        aria-describedby={`${describedBy} ${tableId}`}
      >
        <BarChart
          data={displayData}
          margin={{ top: 6, right: 24, left: 8, bottom: 0 }}
        >
          <CartesianGrid vertical={false} />
          <XAxis
            dataKey="year"
            type="number"
            allowDecimals={false}
            domain={yearRange ? [yearRange.minYear, yearRange.maxYear] : ["dataMin", "dataMax"]}
            padding={{ left: 20, right: 20 }}
            tickFormatter={(value) => String(value)}
            tickMargin={6}
            minTickGap={16}
            interval="preserveStartEnd"
            ticks={yearTicks}
          />
          <YAxis
            tickMargin={6}
            width={mode === "percent" ? 44 : 32}
            allowDecimals={mode !== "percent"}
            domain={mode === "percent" ? [0, 100] : undefined}
            ticks={mode === "percent" ? PERCENT_AXIS_TICKS : undefined}
            tickFormatter={(value) => {
              const numericValue = Number(value);
              if (!Number.isFinite(numericValue)) return "";
              if (mode === "percent") {
                return `${Math.round(numericValue)}%`;
              }
              return String(numericValue);
            }}
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
                  const rawCount = Number(
                    rawRow && dataKey ? (rawRow[dataKey] ?? 0) : 0,
                  );
                  const rawTotal = series.reduce(
                    (sum, currentSeries) =>
                      sum + Number(rawRow?.[currentSeries.key] ?? 0),
                    0,
                  );
                  const rawPct =
                    rawTotal > 0
                      ? Math.round((rawCount / rawTotal) * 1000) / 10
                      : 0;
                  const valueNumber = Number(value);
                  const valueLabel =
                    mode === "percent"
                      ? `${valueNumber.toFixed(1)}%`
                      : rawCount.toLocaleString();
                  const metaLabel =
                    mode === "percent"
                      ? rawCount.toLocaleString()
                      : `${rawPct.toFixed(1)}%`;

                  return (
                    <div className="grid grid-cols-[auto_3rem_minmax(0,1fr)] items-center gap-x-3">
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
                        {valueLabel}
                      </span>
                      <span className="text-right font-mono text-xs tabular-nums text-muted-foreground">
                        {metaLabel}
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
            <Bar
              key={item.key}
              dataKey={item.key}
              stackId="deal-types"
              fill={`var(--color-${item.key})`}
              name={item.label}
            />
          ))}
          {showSourceSplit ? (
            <ReferenceLine
              x={2020.5}
              stroke="hsl(var(--foreground))"
              strokeWidth={1.5}
              strokeDasharray="4 4"
              isFront
              label={(props) => (
                <SplitMarkerLabel
                  isMobile={isMobile}
                  viewBox={props.viewBox ?? null}
                />
              )}
            />
          ) : null}
        </BarChart>
      </ChartContainer>
    </div>
  );
}

export type {
  DealTypeChartMode,
  DealTypeSeries,
  ProcessingStatusRow,
};
