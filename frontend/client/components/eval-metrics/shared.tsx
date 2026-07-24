import type { ReactNode } from "react";

import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export const formatEvalMetric = (value: number) => `${(value * 100).toFixed(2)}%`;

/** Badge pill shown next to a model title (e.g. "Binary Classifier"). */
export function EvalBadge({ children }: { children: ReactNode }) {
  return (
    <span className="rounded-full border border-border bg-muted/40 px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
      {children}
    </span>
  );
}

/** Title + badge header row used in model cards and accordion triggers. */
export function EvalModelHeader({
  badge,
  className,
  title,
}: {
  badge: string;
  className?: string;
  title: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-wrap items-center justify-between gap-3",
        className,
      )}
    >
      <div className="text-sm font-semibold text-foreground">{title}</div>
      <EvalBadge>{badge}</EvalBadge>
    </div>
  );
}

/** Card wrapper for a model's metrics: header row plus content. */
export function EvalModelCard({
  badge,
  children,
  contentClassName = "mt-4",
  title,
}: {
  badge: string;
  children: ReactNode;
  contentClassName?: string;
  title: string;
}) {
  return (
    <Card className="rounded-2xl border border-border bg-card/60 p-6">
      <EvalModelHeader badge={badge} title={title} />
      <div className={contentClassName}>{children}</div>
    </Card>
  );
}

/**
 * Bordered metrics panel. With `right` set, the title renders in a header row
 * with the extra element (children then need no top margin); without it, the
 * title stands alone and children carry their own `mt-3` spacing.
 */
export function EvalPanel({
  children,
  right,
  title,
}: {
  children: ReactNode;
  right?: ReactNode;
  title: string;
}) {
  const titleElement = (
    <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
      {title}
    </div>
  );
  return (
    <div className="min-w-0 rounded-xl border border-border bg-background/60 p-4">
      {right ? (
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          {titleElement}
          {right}
        </div>
      ) : (
        titleElement
      )}
      {children}
    </div>
  );
}

/** Single summary metric tile (label over large value). */
export function EvalSummaryTile({
  label,
  muted = false,
  value,
}: {
  label: string;
  muted?: boolean;
  value: string;
}) {
  return (
    <div className="text-center sm:text-left">
      <div
        className={cn(
          "text-[11px] font-semibold uppercase tracking-wide",
          muted
            ? "text-muted-foreground"
            : "text-emerald-800 dark:text-emerald-200",
        )}
      >
        {label}
      </div>
      <div className="mt-1 text-2xl font-semibold text-foreground">{value}</div>
    </div>
  );
}

/** Emerald summary strip holding a grid of EvalSummaryTiles. */
export function EvalSummaryGrid({
  columnsClassName,
  metrics,
}: {
  columnsClassName: string;
  metrics: Array<{ label: string; value: string }>;
}) {
  return (
    <div className="rounded-lg bg-emerald-500/10 p-3 lg:col-span-2">
      <div
        className={cn(
          "grid gap-3 text-emerald-900 dark:text-emerald-100",
          columnsClassName,
        )}
      >
        {metrics.map((metric) => (
          <EvalSummaryTile
            key={metric.label}
            label={metric.label}
            value={metric.value}
          />
        ))}
      </div>
    </div>
  );
}

/** Confusion matrix panel; `labels` name both axes in matrix order. */
export function ConfusionMatrixPanel({
  caption,
  labels,
  matrix,
  minTableWidthClass,
}: {
  caption: string;
  labels: string[];
  matrix: number[][];
  minTableWidthClass: string;
}) {
  return (
    <EvalPanel title="Confusion Matrix">
      <div className="mt-3 space-y-2">
        <div className="text-center text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
          Predicted
        </div>
        <div className="w-full max-w-full overflow-x-auto">
          <div className="flex items-stretch gap-0">
            <div className="relative w-0">
              <span className="absolute right-1 top-1/2 -translate-y-1/2 -rotate-90 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
                Actual
              </span>
            </div>
            <table
              className={cn(
                "w-full table-fixed border-separate border-spacing-1 text-[11px]",
                minTableWidthClass,
              )}
            >
              <colgroup>
                <col className="w-8" />
                {labels.map((label) => (
                  <col key={label} className="w-12" />
                ))}
              </colgroup>
              <caption className="sr-only">{caption}</caption>
              <thead>
                <tr>
                  <th
                    aria-hidden="true"
                    className="p-1 text-left text-muted-foreground"
                  />
                  {labels.map((label) => (
                    <th
                      key={label}
                      scope="col"
                      className="p-1 text-center font-mono text-muted-foreground"
                    >
                      {label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {matrix.map((row, rowIndex) => (
                  <tr key={labels[rowIndex]}>
                    <th
                      scope="row"
                      className="p-1 pl-0 text-left font-mono text-muted-foreground"
                    >
                      {labels[rowIndex]}
                    </th>
                    {row.map((value, colIndex) => {
                      const isDiagonal = rowIndex === colIndex;
                      const hasValue = value > 0;
                      const cellClass = isDiagonal
                        ? "bg-emerald-500/20 text-foreground"
                        : hasValue
                          ? "bg-rose-500/15 text-foreground"
                          : "bg-muted/40 text-muted-foreground/60";
                      return (
                        <td
                          key={`${rowIndex}-${colIndex}`}
                          className={`rounded-md px-2 py-1 text-center font-mono ${cellClass}`}
                        >
                          {value}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </EvalPanel>
  );
}

export type PerClassMetricRow = {
  /** Stable row key (the raw class label). */
  id: string;
  /** Display label shown in the Class column. */
  label: string;
  acc: number;
  p: number;
  r: number;
  f1: number;
};

/** Per-class Acc/P/R/F1 table with F1-status dots. */
export function PerClassMetricsPanel({
  caption,
  rows,
}: {
  caption: string;
  rows: PerClassMetricRow[];
}) {
  return (
    <EvalPanel title="Per-class Metrics">
      <div className="mt-3 w-full overflow-x-auto">
        <table className="w-full min-w-[320px] text-xs">
          <caption className="sr-only">{caption}</caption>
          <thead>
            <tr className="border-b border-border text-left text-[11px] uppercase tracking-wide text-muted-foreground">
              <th scope="col" className="pb-2 pr-3">
                Class
              </th>
              <th scope="col" className="pb-2 pr-3 text-right">
                Acc
              </th>
              <th scope="col" className="pb-2 pr-3 text-right">
                P
              </th>
              <th scope="col" className="pb-2 pr-3 text-right">
                R
              </th>
              <th scope="col" className="pb-2 text-right">
                F1
              </th>
            </tr>
          </thead>
          <tbody className="font-mono text-muted-foreground">
            {rows.map((row) => {
              const statusClass =
                row.f1 >= 0.95
                  ? "bg-emerald-500"
                  : row.f1 < 0.9
                    ? "bg-amber-400"
                    : "bg-muted-foreground/40";
              return (
                <tr key={row.id} className="border-b border-border/50">
                  <th
                    scope="row"
                    className="py-2 pr-3 text-left text-foreground font-normal"
                  >
                    <span
                      className={`mr-2 inline-flex h-2 w-2 rounded-full ${statusClass}`}
                    />
                    {row.label}
                  </th>
                  <td className="py-2 pr-3 text-right">
                    {formatEvalMetric(row.acc)}
                  </td>
                  <td className="py-2 pr-3 text-right">
                    {formatEvalMetric(row.p)}
                  </td>
                  <td className="py-2 pr-3 text-right">
                    {formatEvalMetric(row.r)}
                  </td>
                  <td className="py-2 text-right">{formatEvalMetric(row.f1)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </EvalPanel>
  );
}
