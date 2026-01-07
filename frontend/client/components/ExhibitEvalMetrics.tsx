import { Card } from "@/components/ui/card";
import type { ExhibitEvalData } from "@/lib/model-metrics-types";

type ExhibitEvalMetricsProps = {
  data: ExhibitEvalData;
};

const formatMetric = (value: number) => `${(value * 100).toFixed(2)}%`;

export function ExhibitEvalMetrics({ data }: ExhibitEvalMetricsProps) {
  const { summary, confusionMatrix, perClass } = data;

  const renderMetricsGrid = () => (
    <div className="rounded-lg bg-emerald-500/10 p-3 lg:col-span-2">
      <div className="grid gap-3 text-emerald-900 dark:text-emerald-100 sm:grid-cols-2 lg:grid-cols-5">
        <div className="text-center sm:text-left">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-emerald-800 dark:text-emerald-200">
            Accuracy
          </div>
          <div className="mt-1 text-2xl font-semibold text-foreground">
            {formatMetric(summary.accuracy)}
          </div>
        </div>
        <div className="text-center sm:text-left">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-emerald-800 dark:text-emerald-200">
            Precision
          </div>
          <div className="mt-1 text-2xl font-semibold text-foreground">
            {formatMetric(summary.precision)}
          </div>
        </div>
        <div className="text-center sm:text-left">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-emerald-800 dark:text-emerald-200">
            Recall
          </div>
          <div className="mt-1 text-2xl font-semibold text-foreground">
            {formatMetric(summary.recall)}
          </div>
        </div>
        <div className="text-center sm:text-left">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-emerald-800 dark:text-emerald-200">
            F1 Score
          </div>
          <div className="mt-1 text-2xl font-semibold text-foreground">
            {formatMetric(summary.f1)}
          </div>
        </div>
        <div className="text-center sm:text-left">
          <div className="text-[11px] font-semibold uppercase tracking-wide text-emerald-800 dark:text-emerald-200">
            ROC AUC
          </div>
          <div className="mt-1 text-2xl font-semibold text-foreground">
            {formatMetric(summary.roc_auc)}
          </div>
        </div>
      </div>
    </div>
  );

  const renderConfusionMatrix = () => (
    <div className="min-w-0 rounded-xl border border-border/60 bg-background/60 p-4">
      <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        Confusion Matrix
      </div>
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
            <table className="w-full min-w-[200px] table-fixed border-separate border-spacing-1 text-[11px]">
              <colgroup>
                <col className="w-8" />
                <col className="w-12" />
                <col className="w-12" />
              </colgroup>
              <caption className="sr-only">Exhibit classifier confusion matrix</caption>
              <thead>
                <tr>
                  <th
                    aria-hidden="true"
                    className="p-1 text-left text-muted-foreground"
                  />
                  <th
                    scope="col"
                    className="p-1 text-center font-mono text-muted-foreground"
                  >
                    Not M&A
                  </th>
                  <th
                    scope="col"
                    className="p-1 text-center font-mono text-muted-foreground"
                  >
                    M&A
                  </th>
                </tr>
              </thead>
              <tbody>
                {confusionMatrix.map((row, rowIndex) => (
                  <tr key={rowIndex}>
                    <th
                      scope="row"
                      className="p-1 pl-0 text-left font-mono text-muted-foreground"
                    >
                      {rowIndex === 0 ? "Not M&A" : "M&A"}
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
    </div>
  );

  const renderPerClass = () => (
    <div className="min-w-0 rounded-xl border border-border/60 bg-background/60 p-4">
      <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        Per-class Metrics
      </div>
      <div className="mt-3 w-full overflow-x-auto">
        <table className="w-full min-w-[320px] text-xs">
          <caption className="sr-only">Exhibit classifier per-class metrics</caption>
          <thead>
            <tr className="border-b border-border/60 text-left text-[11px] uppercase tracking-wide text-muted-foreground">
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
            {perClass.map((metric) => {
              const statusClass =
                metric.f1 >= 0.95
                  ? "bg-emerald-500"
                  : metric.f1 < 0.9
                    ? "bg-amber-400"
                    : "bg-muted-foreground/40";
              return (
                <tr key={metric.label} className="border-b border-border/40">
                  <th
                    scope="row"
                    className="py-2 pr-3 text-left text-foreground font-normal"
                  >
                    <span
                      className={`mr-2 inline-flex h-2 w-2 rounded-full ${statusClass}`}
                    />
                    {metric.label === "class_0" ? "Not M&A" : "M&A"}
                  </th>
                  <td className="py-2 pr-3 text-right">
                    {formatMetric(metric.accuracy)}
                  </td>
                  <td className="py-2 pr-3 text-right">
                    {formatMetric(metric.precision)}
                  </td>
                  <td className="py-2 pr-3 text-right">
                    {formatMetric(metric.recall)}
                  </td>
                  <td className="py-2 text-right">
                    {formatMetric(metric.f1)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );

  return (
    <Card className="rounded-2xl border border-border/70 bg-card/60 p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="text-sm font-semibold text-foreground">
          Model Metrics
        </div>
        <span className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
          Binary Classifier
        </span>
      </div>
      <div className="mt-4">
        <div className="grid gap-6 lg:grid-cols-2">
          {renderMetricsGrid()}
          {renderConfusionMatrix()}
          {renderPerClass()}
        </div>
      </div>
    </Card>
  );
}

