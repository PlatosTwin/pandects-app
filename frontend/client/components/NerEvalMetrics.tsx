import { Card } from "@/components/ui/card";
import type { NerEvalData } from "@/lib/model-metrics-types";

type NerEvalMetricsProps = {
  data: NerEvalData;
  showValidationBlocks?: boolean;
};

const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`;

const SummaryMetric = ({ label, value }: { label: string; value: string }) => (
  <div className="text-center sm:text-left">
    <div className="text-[11px] font-semibold uppercase tracking-wide text-emerald-800 dark:text-emerald-200">
      {label}
    </div>
    <div className="mt-1 text-2xl font-semibold text-foreground">{value}</div>
  </div>
);

const MutedMetric = ({ label, value }: { label: string; value: string }) => (
  <div className="text-center sm:text-left">
    <div className="text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
      {label}
    </div>
    <div className="mt-1 text-2xl font-semibold text-foreground">{value}</div>
  </div>
);

export function NerEvalMetrics({
  data,
  showValidationBlocks: _showValidationBlocks = true,
}: NerEvalMetricsProps) {
  const { summary, perEntity, boundaries } = data;
  const summaryMetrics = [
    { label: "Precision", key: "precision" as const },
    { label: "Recall", key: "recall" as const },
    { label: "F1 Score", key: "f1" as const },
  ];
  const entityKeyByLabel = {
    Article: "ARTICLE",
    Section: "SECTION",
    Page: "PAGE",
  } as const;

  return (
    <Card className="rounded-2xl border border-border/70 bg-card/60 p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="text-sm font-semibold text-foreground">
          Model Metrics
        </div>
        <span className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
          Tagging Model
        </span>
      </div>

      <div className="mt-4 space-y-6">
        <div className="rounded-lg bg-emerald-500/10 p-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2 text-emerald-900 dark:text-emerald-100">
            <div className="text-[11px] font-semibold uppercase tracking-wide">
              Entity-level micro metrics
            </div>
            <span className="rounded-full border border-emerald-500/40 bg-emerald-500/15 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-emerald-900 dark:text-emerald-100">
              Strict
            </span>
          </div>
          <div className="grid gap-3 text-emerald-900 dark:text-emerald-100 sm:grid-cols-2 lg:grid-cols-3">
            {summaryMetrics.map((metric) => (
              <SummaryMetric
                key={metric.label}
                label={metric.label}
                value={formatPercent(summary.strict[metric.key])}
              />
            ))}
          </div>
        </div>

        <div className="min-w-0 rounded-xl border border-border/60 bg-background/60 p-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Entity-level micro metrics
            </div>
            <span className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
              Lenient
            </span>
          </div>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {summaryMetrics.map((metric) => (
              <MutedMetric
                key={metric.label}
                label={metric.label}
                value={formatPercent(summary.lenient[metric.key])}
              />
            ))}
          </div>
        </div>

        <div className="min-w-0 rounded-xl border border-border/60 bg-background/60 p-4">
          <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Article, Section, Page metrics
          </div>
          <div className="mt-3 w-full overflow-x-auto">
            <table className="w-full min-w-0 text-xs table-auto sm:min-w-[520px] sm:table-fixed">
              <caption className="sr-only">
                Article, Section, and Page metrics for strict and lenient
                evaluation
              </caption>
              <colgroup className="hidden sm:table-column-group">
                <col className="w-[28%]" />
                <col className="w-[18%]" />
                <col className="w-[18%]" />
                <col className="w-[18%]" />
                <col className="w-[18%]" />
              </colgroup>
              <thead>
                <tr className="border-b border-border/60 text-left text-[11px] uppercase tracking-wide text-muted-foreground">
                  <th
                    scope="col"
                    className="w-[1%] whitespace-nowrap pb-2 pr-2 sm:w-auto sm:pr-3"
                  >
                    Entity
                  </th>
                  <th scope="col" className="hidden pb-2 pr-2 sm:table-cell sm:pr-3">
                    Mode
                  </th>
                  <th scope="col" className="pb-2 pr-2 text-right sm:pr-3">
                    P
                  </th>
                  <th scope="col" className="pb-2 pr-2 text-right sm:pr-3">
                    R
                  </th>
                  <th scope="col" className="pb-2 text-right">
                    F1
                  </th>
                </tr>
              </thead>
              <tbody className="font-mono text-muted-foreground">
                {(["Article", "Section", "Page"] as const).flatMap((entity) => {
                  const entityKey = entityKeyByLabel[entity];
                  return (["Strict", "Lenient"] as const).map((mode) => {
                    const modeKey = mode === "Strict" ? "strict" : "lenient";
                    const metrics = perEntity[modeKey][entityKey];
                    return (
                      <tr
                        key={`${entity}-${mode}`}
                        className="border-b border-border/40"
                      >
                        <th
                          scope="row"
                          className="w-[1%] whitespace-nowrap py-1.5 pr-2 text-left text-foreground font-normal sm:w-auto sm:py-2 sm:pr-3"
                        >
                          <div>{entity}</div>
                          <div className="mt-1 text-[11px] uppercase tracking-wide text-muted-foreground sm:hidden">
                            {mode}
                          </div>
                        </th>
                        <td className="hidden py-1.5 pr-2 text-left text-muted-foreground sm:table-cell sm:py-2 sm:pr-3">
                          {mode}
                        </td>
                        <td className="py-1.5 pr-2 text-right sm:py-2 sm:pr-3">
                          {formatPercent(metrics.precision)}
                        </td>
                        <td className="py-1.5 pr-2 text-right sm:py-2 sm:pr-3">
                          {formatPercent(metrics.recall)}
                        </td>
                        <td className="py-1.5 text-right sm:py-2">
                          {formatPercent(metrics.f1)}
                        </td>
                      </tr>
                    );
                  });
                })}
              </tbody>
            </table>
          </div>
        </div>

        <div className="min-w-0 rounded-xl border border-border/60 bg-background/60 p-4">
          <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Boundary metrics (F1)
          </div>
          <div className="mt-3 w-full overflow-x-auto">
            <table className="w-full min-w-0 text-xs table-auto sm:min-w-[420px] sm:table-fixed">
              <caption className="sr-only">
                Boundary F1 scores for Article, Section, and Page entities
              </caption>
              <colgroup className="hidden sm:table-column-group">
                <col className="w-[28%]" />
                <col className="w-[18%]" />
                <col className="w-[18%]" />
                <col className="w-[18%]" />
                <col className="w-[18%]" />
              </colgroup>
              <thead>
                <tr className="border-b border-border/60 text-left text-[11px] uppercase tracking-wide text-muted-foreground">
                  <th
                    scope="col"
                    className="w-[1%] whitespace-nowrap pb-2 pr-2 sm:w-auto sm:pr-3"
                  >
                    Entity
                  </th>
                  {(["B", "I", "E", "S"] as const).map((metric) => (
                    <th
                      key={metric}
                      scope="col"
                      className="pb-2 pr-2 text-right last:pr-0 sm:pr-3"
                    >
                      {metric}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="font-mono text-muted-foreground">
                {[
                  { label: "Article", key: "ARTICLE", showS: false },
                  { label: "Section", key: "SECTION", showS: false },
                  { label: "Page", key: "PAGE", showS: true },
                ].map((row) => (
                  <tr key={row.label} className="border-b border-border/40">
                    <th
                      scope="row"
                      className="w-[1%] whitespace-nowrap py-1.5 pr-2 text-left text-foreground font-normal sm:w-auto sm:py-2 sm:pr-3"
                    >
                      {row.label}
                    </th>
                    {(["B", "I", "E", "S"] as const).map((metric) => {
                      const value =
                        metric === "S"
                          ? row.showS
                            ? boundaries.PAGE.S
                            : null
                          : boundaries[row.key][metric];
                      return (
                        <td
                          key={`${row.label}-${metric}`}
                          className="py-1.5 pr-2 text-right last:pr-0 sm:py-2 sm:pr-3"
                        >
                          {value === null ? "N/A" : formatPercent(value)}
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
    </Card>
  );
}
