import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Card } from "@/components/ui/card";
import type { ClassifierEvalData } from "@/lib/model-metrics-types";

type ClassifierEvalMetricsProps = {
  data: ClassifierEvalData;
};

const formatMetric = (value: number) => `${(value * 100).toFixed(2)}%`;

export function ClassifierEvalMetrics({ data }: ClassifierEvalMetricsProps) {
  const accordionModels = data.models.filter(
    (model) => model.layout === "accordion",
  );
  const cardModels = data.models.filter((model) => model.layout === "card");

  const renderMetricsGrid = (summary: ClassifierEvalData["models"][number]["summary"]) => (
    <div className="rounded-lg bg-emerald-500/10 p-3 lg:col-span-2">
      <div className="grid gap-3 text-emerald-900 dark:text-emerald-100 sm:grid-cols-2 lg:grid-cols-4">
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
      </div>
    </div>
  );

  const renderConfusionMatrix = (
    model: ClassifierEvalData["models"][number],
  ) => (
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
            <table className="w-full min-w-[320px] table-fixed border-separate border-spacing-1 text-[11px]">
              <colgroup>
                <col className="w-8" />
                {data.abbreviations.map((label) => (
                  <col key={label} className="w-12" />
                ))}
              </colgroup>
              <caption className="sr-only">{model.matrixCaption}</caption>
              <thead>
                <tr>
                  <th
                    aria-hidden="true"
                    className="p-1 text-left text-muted-foreground"
                  />
                  {data.abbreviations.map((label) => (
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
                {model.confusionMatrix.map((row, rowIndex) => (
                  <tr key={data.labels[rowIndex]}>
                    <th
                      scope="row"
                      className="p-1 pl-0 text-left font-mono text-muted-foreground"
                    >
                      {data.abbreviations[rowIndex]}
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

  const renderPerClass = (model: ClassifierEvalData["models"][number]) => (
    <div className="min-w-0 rounded-xl border border-border/60 bg-background/60 p-4">
      <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        Per-class Metrics
      </div>
      <div className="mt-3 w-full overflow-x-auto">
        <table className="w-full min-w-[320px] text-xs">
          <caption className="sr-only">{model.perClassCaption}</caption>
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
            {model.perClass.map((metric) => {
              const statusClass =
                metric.f1 >= 0.95
                  ? "bg-emerald-500"
                  : metric.f1 < 0.9
                    ? "bg-amber-400"
                    : "bg-muted-foreground/40";
              const labelIndex = data.labels.indexOf(metric.label);
              const shortLabel = data.abbreviations[labelIndex];
              return (
                <tr key={metric.label} className="border-b border-border/40">
                  <th
                    scope="row"
                    className="py-2 pr-3 text-left text-foreground font-normal"
                  >
                    <span
                      className={`mr-2 inline-flex h-2 w-2 rounded-full ${statusClass}`}
                    />
                    {shortLabel}
                  </th>
                  <td className="py-2 pr-3 text-right">
                    {formatMetric(metric.acc)}
                  </td>
                  <td className="py-2 pr-3 text-right">
                    {formatMetric(metric.p)}
                  </td>
                  <td className="py-2 pr-3 text-right">
                    {formatMetric(metric.r)}
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

  const renderModelContent = (model: ClassifierEvalData["models"][number]) => (
    <div className="grid gap-6 lg:grid-cols-2">
      {renderMetricsGrid(model.summary)}
      {renderConfusionMatrix(model)}
      {renderPerClass(model)}
    </div>
  );

  return (
    <div className="space-y-6">
      <Accordion type="multiple" className="space-y-4">
        {accordionModels.map((model) => (
          <AccordionItem
            key={model.id}
            value={model.id}
            className="rounded-2xl border border-border/70 bg-card/60"
          >
            <AccordionTrigger className="px-5 py-4 text-left">
              <div className="flex w-full flex-wrap items-center justify-between gap-3">
                <div className="text-sm font-semibold text-foreground">
                  {model.title}
                </div>
                <span className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
                  {model.badge}
                </span>
              </div>
            </AccordionTrigger>
            <AccordionContent className="px-5 pb-5">
              {renderModelContent(model)}
            </AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>
      {cardModels.map((model) => (
        <Card
          key={model.id}
          className="rounded-2xl border border-border/70 bg-card/60 p-5"
        >
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="text-sm font-semibold text-foreground">
              {model.title}
            </div>
            <span className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
              {model.badge}
            </span>
          </div>
          <div className="mt-4">{renderModelContent(model)}</div>
        </Card>
      ))}
    </div>
  );
}
