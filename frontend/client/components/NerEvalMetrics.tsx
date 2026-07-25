import {
  EvalModelCard,
  EvalPanel,
  EvalSummaryTile,
  formatEvalMetric,
} from "@/components/eval-metrics/shared";
import { Badge } from "@/components/ui/badge";
import type { NerEvalData } from "@/lib/model-metrics-types";

type NerEvalMetricsProps = {
  data: NerEvalData;
};

const getStatusDotClass = (value: number) =>
  value >= 0.95 ? "bg-emerald-500" : "bg-muted-foreground/40";

export function NerEvalMetrics({ data }: NerEvalMetricsProps) {
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
    <EvalModelCard
      badge="Tagging Model"
      contentClassName="mt-4 space-y-6"
      title="Model Metrics"
    >
      <div className="rounded-lg bg-emerald-500/10 p-4">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2 text-emerald-900 dark:text-emerald-100">
          <div className="text-[11px] font-semibold uppercase tracking-wide">
            Entity-level micro metrics
          </div>
          <Badge
            variant="chip"
            className="border-emerald-500/40 bg-emerald-500/15 uppercase tracking-wide text-emerald-900 dark:text-emerald-100"
          >
            Strict
          </Badge>
        </div>
        <div className="grid gap-3 text-emerald-900 dark:text-emerald-100 sm:grid-cols-2 lg:grid-cols-3">
          {summaryMetrics.map((metric) => (
            <EvalSummaryTile
              key={metric.label}
              label={metric.label}
              value={formatEvalMetric(summary.strict[metric.key])}
            />
          ))}
        </div>
      </div>

      <EvalPanel
        title="Entity-level micro metrics"
        right={
          <Badge
            variant="chip"
            className="uppercase tracking-wide text-muted-foreground"
          >
            Lenient
          </Badge>
        }
      >
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {summaryMetrics.map((metric) => (
            <EvalSummaryTile
              key={metric.label}
              label={metric.label}
              muted
              value={formatEvalMetric(summary.lenient[metric.key])}
            />
          ))}
        </div>
      </EvalPanel>

      <EvalPanel title="Article, Section, Page metrics">
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
              <tr className="border-b border-border text-left text-[11px] uppercase tracking-wide text-muted-foreground">
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
                  const statusDotClass = getStatusDotClass(metrics.f1);
                  return (
                    <tr
                      key={`${entity}-${mode}`}
                      className="border-b border-border/50"
                    >
                      <th
                        scope="row"
                        className="w-[1%] whitespace-nowrap py-1.5 pr-2 text-left text-foreground font-normal sm:w-auto sm:py-2 sm:pr-3"
                      >
                        <div className="flex items-center gap-2">
                          <span
                            className={`inline-flex h-2 w-2 rounded-full ${statusDotClass}`}
                          />
                          <span>{entity}</span>
                        </div>
                        <div className="mt-1 text-[11px] uppercase tracking-wide text-muted-foreground sm:hidden">
                          {mode}
                        </div>
                      </th>
                      <td className="hidden py-1.5 pr-2 text-left text-muted-foreground sm:table-cell sm:py-2 sm:pr-3">
                        {mode}
                      </td>
                      <td className="py-1.5 pr-2 text-right sm:py-2 sm:pr-3">
                        {formatEvalMetric(metrics.precision)}
                      </td>
                      <td className="py-1.5 pr-2 text-right sm:py-2 sm:pr-3">
                        {formatEvalMetric(metrics.recall)}
                      </td>
                      <td className="py-1.5 text-right sm:py-2">
                        {formatEvalMetric(metrics.f1)}
                      </td>
                    </tr>
                  );
                });
              })}
            </tbody>
          </table>
        </div>
      </EvalPanel>

      <EvalPanel title="Boundary metrics (F1)">
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
              <tr className="border-b border-border text-left text-[11px] uppercase tracking-wide text-muted-foreground">
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
                <tr key={row.label} className="border-b border-border/50">
                  {(() => {
                    const availableValues = (["B", "I", "E", "S"] as const)
                      .map((metric) =>
                        metric === "S"
                          ? row.showS
                            ? boundaries.PAGE.S
                            : null
                          : boundaries[row.key][metric],
                      )
                      .filter((value): value is number => value !== null);
                    const statusDotClass = getStatusDotClass(
                      availableValues.reduce((sum, value) => sum + value, 0) /
                        availableValues.length,
                    );

                    return (
                      <th
                        scope="row"
                        className="w-[1%] whitespace-nowrap py-1.5 pr-2 text-left text-foreground font-normal sm:w-auto sm:py-2 sm:pr-3"
                      >
                        <div className="flex items-center gap-2">
                          <span
                            className={`inline-flex h-2 w-2 rounded-full ${statusDotClass}`}
                          />
                          <span>{row.label}</span>
                        </div>
                      </th>
                    );
                  })()}
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
                        {value === null ? "N/A" : formatEvalMetric(value)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </EvalPanel>
    </EvalModelCard>
  );
}
