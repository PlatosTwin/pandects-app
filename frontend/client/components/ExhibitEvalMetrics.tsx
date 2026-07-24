import {
  ConfusionMatrixPanel,
  EvalModelCard,
  EvalSummaryGrid,
  PerClassMetricsPanel,
  formatEvalMetric,
} from "@/components/eval-metrics/shared";
import type { ExhibitEvalData } from "@/lib/model-metrics-types";

type ExhibitEvalMetricsProps = {
  data: ExhibitEvalData;
};

const CLASS_LABELS = ["Not M&A", "M&A"];

export function ExhibitEvalMetrics({ data }: ExhibitEvalMetricsProps) {
  const { summary, confusionMatrix, perClass } = data;

  return (
    <EvalModelCard badge="Binary Classifier" title="Model Metrics">
      <div className="grid gap-6 lg:grid-cols-2">
        <EvalSummaryGrid
          columnsClassName="sm:grid-cols-2 lg:grid-cols-5"
          metrics={[
            { label: "Accuracy", value: formatEvalMetric(summary.accuracy) },
            { label: "Precision", value: formatEvalMetric(summary.precision) },
            { label: "Recall", value: formatEvalMetric(summary.recall) },
            { label: "F1 Score", value: formatEvalMetric(summary.f1) },
            { label: "ROC AUC", value: formatEvalMetric(summary.roc_auc) },
          ]}
        />
        <ConfusionMatrixPanel
          caption="Exhibit classifier confusion matrix"
          labels={CLASS_LABELS}
          matrix={confusionMatrix}
          minTableWidthClass="min-w-[200px]"
        />
        <PerClassMetricsPanel
          caption="Exhibit classifier per-class metrics"
          rows={perClass.map((metric) => ({
            id: metric.label,
            label: metric.label === "class_0" ? "Not M&A" : "M&A",
            acc: metric.accuracy,
            p: metric.precision,
            r: metric.recall,
            f1: metric.f1,
          }))}
        />
      </div>
    </EvalModelCard>
  );
}
