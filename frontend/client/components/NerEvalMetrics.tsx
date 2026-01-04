import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Card } from "@/components/ui/card";
import type { NerEvalData } from "@/lib/model-metrics-types";

type NerEvalMetricsProps = {
  data: NerEvalData;
  showValidationBlocks?: boolean;
};

const MODE_LABELS: Record<"raw" | "regex" | "regex+snap", string> = {
  raw: "Raw",
  regex: "Regex",
  "regex+snap": "Regex + Snap",
};

const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`;
const formatCount = (value: number) =>
  new Intl.NumberFormat("en-US").format(value);

const MetricCard = ({ label, value }: { label: string; value: string }) => (
  <div className="text-center sm:text-left">
    <div className="text-[11px] font-semibold uppercase tracking-wide text-emerald-800 dark:text-emerald-200">
      {label}
    </div>
    <div className="mt-1 text-2xl font-semibold text-foreground">{value}</div>
  </div>
);

const PerTypeTable = ({
  rows,
  caption,
}: {
  rows: NerEvalData["finalTest"]["perTypeStrict"];
  caption: string;
}) => (
  <div className="min-w-0 rounded-xl border border-border/60 bg-background/60 p-4">
    <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
      Per-type Metrics
    </div>
    <div className="mt-3 w-full max-w-full overflow-x-auto">
      <table className="w-max table-auto text-xs">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr className="border-b border-border/60 text-left text-[11px] uppercase tracking-wide text-muted-foreground">
            <th scope="col" className="whitespace-nowrap pb-2 pr-3">
              Type
            </th>
            <th scope="col" className="whitespace-nowrap pb-2 pr-3 text-right">
              P
            </th>
            <th scope="col" className="whitespace-nowrap pb-2 pr-3 text-right">
              R
            </th>
            <th scope="col" className="whitespace-nowrap pb-2 pr-3 text-right">
              F1
            </th>
            <th scope="col" className="whitespace-nowrap pb-2 text-right">
              Support
            </th>
          </tr>
        </thead>
        <tbody className="font-mono text-muted-foreground">
          {rows.map((row) => (
            <tr key={row.type} className="border-b border-border/40">
              <th
                scope="row"
                className="whitespace-nowrap py-2 pr-3 text-left text-foreground font-normal"
              >
                {row.type}
              </th>
              <td className="py-2 pr-3 text-right">
                {formatPercent(row.precision)}
              </td>
              <td className="py-2 pr-3 text-right">
                {formatPercent(row.recall)}
              </td>
              <td className="py-2 pr-3 text-right">{formatPercent(row.f1)}</td>
              <td className="py-2 text-right">{formatCount(row.support)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  </div>
);

export function NerEvalMetrics({
  data,
  showValidationBlocks = true,
}: NerEvalMetricsProps) {
  if (!showValidationBlocks) {
    return null;
  }
  const { meta, finalTest, baselineVal, learningCurveVal, weightSweepVal } =
    data;
  const primarySummary = finalTest.summaryByMode[finalTest.primaryMode];
  const hasBaselineArticle = Object.values(baselineVal.byMode).some(
    (mode) => mode.articleStrict,
  );
  const hasLearningArticle = learningCurveVal.some((row) =>
    Object.values(row.byMode).some((mode) => mode.articleStrictF1 !== undefined),
  );
  const hasWeightEntity = weightSweepVal.some((row) =>
    Object.values(row.byMode).some((mode) => mode.entityStrictF1 !== undefined),
  );

  return (
    <div className="space-y-8 min-w-0">
      <Accordion type="multiple" className="space-y-4 min-w-0">
        <AccordionItem
          value="baseline-validation"
          className="rounded-2xl border border-border/70 bg-card/60"
        >
          <AccordionTrigger className="px-5 py-4 text-left">
            <div className="flex w-full flex-wrap items-center justify-between gap-3">
              <div className="text-sm font-semibold text-foreground">
                Baseline
              </div>
              <span className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
                Validation
              </span>
            </div>
          </AccordionTrigger>
          <AccordionContent className="px-5 pb-5">
            <div className="grid gap-5 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)]">
              <div className="w-full">
                <table className="w-full table-fixed text-xs">
                  <caption className="sr-only">
                    Baseline validation metrics by gating mode
                  </caption>
                  <colgroup>
                    <col className="w-[40%]" />
                    <col className="w-[20%]" />
                    <col className="w-[20%]" />
                    <col className="w-[20%]" />
                  </colgroup>
                  <thead>
                    <tr className="border-b border-border/60 text-left text-[11px] uppercase tracking-wide text-muted-foreground">
                      <th scope="col" className="pb-2 pr-3">
                        Metric
                      </th>
                      <th scope="col" className="pb-2 pr-3 text-right">
                        {MODE_LABELS.raw}
                      </th>
                      <th scope="col" className="pb-2 pr-3 text-right">
                        {MODE_LABELS.regex}
                      </th>
                      <th scope="col" className="pb-2 text-right">
                        {MODE_LABELS["regex+snap"]}
                      </th>
                    </tr>
                  </thead>
                  <tbody className="font-mono text-muted-foreground">
                    {[
                      {
                        label: "Strict P",
                        get: (mode: typeof baselineVal.byMode.raw) =>
                          mode.entityStrict.precision,
                      },
                      {
                        label: "Strict R",
                        get: (mode: typeof baselineVal.byMode.raw) =>
                          mode.entityStrict.recall,
                      },
                      {
                        label: "Strict F1",
                        get: (mode: typeof baselineVal.byMode.raw) =>
                          mode.entityStrict.f1,
                      },
                      {
                        label: "Lenient F1",
                        get: (mode: typeof baselineVal.byMode.raw) =>
                          mode.entityLenient.f1,
                      },
                      hasBaselineArticle
                        ? {
                            label: "ARTICLE strict F1",
                            get: (mode: typeof baselineVal.byMode.raw) =>
                              mode.articleStrict?.f1,
                          }
                        : null,
                    ]
                      .filter(Boolean)
                      .map((row) => (
                        <tr
                          key={(row as { label: string }).label}
                          className="border-b border-border/40"
                        >
                          <th
                            scope="row"
                            className="py-2 pr-3 text-left text-foreground font-normal"
                          >
                            {(row as { label: string }).label}
                          </th>
                          {(["raw", "regex", "regex+snap"] as const).map(
                            (modeKey) => {
                              const value = (
                                row as {
                                  get: (mode: typeof baselineVal.byMode.raw) =>
                                    | number
                                    | undefined;
                                }
                              ).get(baselineVal.byMode[modeKey]);
                              return (
                                <td
                                  key={modeKey}
                                  className="py-2 pr-3 text-right last:pr-0"
                                >
                                  {value === undefined
                                    ? "—"
                                    : formatPercent(value)}
                                </td>
                              );
                            },
                          )}
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
              <div className="rounded-xl border border-border/60 bg-background/60 p-4">
                <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Notes
                </div>
                <ul className="mt-3 list-disc space-y-2 pl-5 text-sm text-muted-foreground">
                  <li>Baseline uses no post-processing constraints.</li>
                  <li>Validation scores guide the next-stage comparisons.</li>
                  <li>All modes share the same val split.</li>
                </ul>
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem
          value="learning-curve-validation"
          className="rounded-2xl border border-border/70 bg-card/60 min-w-0"
        >
          <AccordionTrigger className="px-5 py-4 text-left">
            <div className="flex w-full flex-wrap items-center justify-between gap-3">
              <div className="text-sm font-semibold text-foreground">
                Learning curve
              </div>
              <span className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
                Validation
              </span>
            </div>
          </AccordionTrigger>
          <AccordionContent className="px-5 pb-5 min-w-0">
            <div className="text-xs text-muted-foreground">
              <span className="font-semibold text-foreground">Decision:</span>{" "}
              Placeholder summary of the selected train set size.
            </div>
            <div className="mt-3 w-full max-w-full overflow-x-auto overflow-y-hidden min-w-0">
              <table className="w-max min-w-[640px] table-auto text-xs">
                <caption className="sr-only">
                  Validation learning curve by train set size
                </caption>
                <thead>
                  <tr className="border-b border-border/60 text-left text-[11px] uppercase tracking-wide text-muted-foreground">
                    <th scope="col" className="pb-2 pr-3 leading-tight">
                      Train docs
                    </th>
                    <th scope="col" className="pb-2 pr-3 text-right leading-tight">
                      {MODE_LABELS.raw} strict F1
                    </th>
                    <th scope="col" className="pb-2 pr-3 text-right leading-tight">
                      {MODE_LABELS.regex} strict F1
                    </th>
                    <th scope="col" className="pb-2 pr-3 text-right leading-tight">
                      {MODE_LABELS["regex+snap"]} strict F1
                    </th>
                    {hasLearningArticle && (
                      <>
                        <th scope="col" className="pb-2 pr-3 text-right leading-tight">
                          {MODE_LABELS.raw} ARTICLE F1
                        </th>
                        <th scope="col" className="pb-2 pr-3 text-right leading-tight">
                          {MODE_LABELS.regex} ARTICLE F1
                        </th>
                        <th scope="col" className="pb-2 text-right leading-tight">
                          {MODE_LABELS["regex+snap"]} ARTICLE F1
                        </th>
                      </>
                    )}
                  </tr>
                </thead>
                <tbody className="font-mono text-muted-foreground">
                  {learningCurveVal.map((row) => (
                    <tr
                      key={row.trainDocs}
                      className="border-b border-border/40"
                    >
                      <th
                        scope="row"
                        className="py-2 pr-3 text-left text-foreground font-normal"
                      >
                        {formatCount(row.trainDocs)}
                      </th>
                      <td className="py-2 pr-3 text-right">
                        {formatPercent(row.byMode.raw.entityStrictF1)}
                      </td>
                      <td className="py-2 pr-3 text-right">
                        {formatPercent(row.byMode.regex.entityStrictF1)}
                      </td>
                      <td className="py-2 pr-3 text-right">
                        {formatPercent(row.byMode["regex+snap"].entityStrictF1)}
                      </td>
                      {hasLearningArticle && (
                        <>
                          <td className="py-2 pr-3 text-right">
                            {row.byMode.raw.articleStrictF1 === undefined
                              ? "—"
                              : formatPercent(row.byMode.raw.articleStrictF1)}
                          </td>
                          <td className="py-2 pr-3 text-right">
                            {row.byMode.regex.articleStrictF1 === undefined
                              ? "—"
                              : formatPercent(row.byMode.regex.articleStrictF1)}
                          </td>
                          <td className="py-2 pr-3 text-right">
                            {row.byMode["regex+snap"].articleStrictF1 ===
                            undefined
                              ? "—"
                              : formatPercent(
                                  row.byMode["regex+snap"].articleStrictF1,
                                )}
                          </td>
                        </>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </AccordionContent>
        </AccordionItem>

        <AccordionItem
          value="weight-sweep-validation"
          className="rounded-2xl border border-border/70 bg-card/60 min-w-0"
        >
          <AccordionTrigger className="px-5 py-4 text-left">
            <div className="flex w-full flex-wrap items-center justify-between gap-3">
              <div className="text-sm font-semibold text-foreground">
                ARTICLE weighting sweep
              </div>
              <span className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
                Validation
              </span>
            </div>
          </AccordionTrigger>
          <AccordionContent className="px-5 pb-5 min-w-0">
            <div className="text-xs text-muted-foreground">
              <span className="font-semibold text-foreground">Decision:</span>{" "}
              Placeholder summary of the chosen article weight.
            </div>
            <div className="mt-3 w-full max-w-full overflow-x-auto overflow-y-hidden min-w-0">
              <table className="w-max min-w-[640px] table-auto text-xs">
                <caption className="sr-only">
                  Validation sweep over article weighting
                </caption>
                <thead>
                  <tr className="border-b border-border/60 text-left text-[11px] uppercase tracking-wide text-muted-foreground">
                    <th scope="col" className="pb-2 pr-3 leading-tight">
                      Weight
                    </th>
                    <th scope="col" className="pb-2 pr-3 text-right leading-tight">
                      {MODE_LABELS.raw} ARTICLE F1
                    </th>
                    <th scope="col" className="pb-2 pr-3 text-right leading-tight">
                      {MODE_LABELS.regex} ARTICLE F1
                    </th>
                    <th scope="col" className="pb-2 pr-3 text-right leading-tight">
                      {MODE_LABELS["regex+snap"]} ARTICLE F1
                    </th>
                    {hasWeightEntity && (
                      <>
                        <th scope="col" className="pb-2 pr-3 text-right leading-tight">
                          {MODE_LABELS.raw} strict F1
                        </th>
                        <th scope="col" className="pb-2 pr-3 text-right leading-tight">
                          {MODE_LABELS.regex} strict F1
                        </th>
                        <th scope="col" className="pb-2 text-right leading-tight">
                          {MODE_LABELS["regex+snap"]} strict F1
                        </th>
                      </>
                    )}
                  </tr>
                </thead>
                <tbody className="font-mono text-muted-foreground">
                  {weightSweepVal.map((row) => (
                    <tr
                      key={row.articleWeight}
                      className="border-b border-border/40"
                    >
                      <th
                        scope="row"
                        className="py-2 pr-3 text-left text-foreground font-normal"
                      >
                        {row.articleWeight}
                      </th>
                      <td className="py-2 pr-3 text-right">
                        {formatPercent(row.byMode.raw.articleStrictF1)}
                      </td>
                      <td className="py-2 pr-3 text-right">
                        {formatPercent(row.byMode.regex.articleStrictF1)}
                      </td>
                      <td className="py-2 pr-3 text-right">
                        {formatPercent(row.byMode["regex+snap"].articleStrictF1)}
                      </td>
                      {hasWeightEntity && (
                        <>
                          <td className="py-2 pr-3 text-right">
                            {formatPercent(row.byMode.raw.entityStrictF1)}
                          </td>
                          <td className="py-2 pr-3 text-right">
                            {formatPercent(row.byMode.regex.entityStrictF1)}
                          </td>
                          <td className="py-2 pr-3 text-right">
                            {formatPercent(
                              row.byMode["regex+snap"].entityStrictF1,
                            )}
                          </td>
                        </>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>

      <Card className="rounded-2xl border-border/70 bg-card/60 p-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h4 className="text-lg font-semibold text-foreground">
              NER Model Evaluation
            </h4>
            <p className="mt-1 text-sm text-muted-foreground">
              Model selection is driven by validation splits, with final
              reporting on a held-out test set.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 text-[11px] font-mono text-muted-foreground">
            <span className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5">
              Scheme: {meta.labelScheme}
            </span>
            <span className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5">
              Split: {meta.splitVersion}
            </span>
            <span className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5">
              Train docs: {formatCount(meta.finalTrainDocs)}
            </span>
            <span className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5">
              Article wt: {meta.finalArticleWeight}
            </span>
            <span className="rounded-full border border-border/70 bg-muted/40 px-2 py-0.5">
              Gating: {MODE_LABELS[meta.finalGatingMode]}
            </span>
          </div>
        </div>
        <div className="mt-5 rounded-lg bg-emerald-500/10 p-3">
          <div className="grid gap-3 text-emerald-900 dark:text-emerald-100 sm:grid-cols-2 lg:grid-cols-4">
            <MetricCard
              label="Entity F1 (strict)"
              value={formatPercent(primarySummary.entityStrict.f1)}
            />
            <MetricCard
              label="Precision (strict)"
              value={formatPercent(primarySummary.entityStrict.precision)}
            />
            <MetricCard
              label="Recall (strict)"
              value={formatPercent(primarySummary.entityStrict.recall)}
            />
            <MetricCard
              label="Lenient F1"
              value={formatPercent(primarySummary.entityLenient.f1)}
            />
          </div>
        </div>
        <div className="mt-5 grid gap-5 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)]">
          <PerTypeTable
            rows={finalTest.perTypeStrict}
            caption="NER per-type strict metrics"
          />
          <div className="rounded-xl border border-border/60 bg-background/60 p-4">
            <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Notes
            </div>
            <ul className="mt-3 list-disc space-y-2 pl-5 text-sm text-muted-foreground">
              <li>Primary mode selected via validation performance.</li>
              <li>Support counts reflect strict entity matches.</li>
              <li>Gating mode applied at inference time.</li>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
}
