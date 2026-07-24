import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
  ConfusionMatrixPanel,
  EvalModelCard,
  EvalModelHeader,
  EvalSummaryGrid,
  PerClassMetricsPanel,
  formatEvalMetric,
} from "@/components/eval-metrics/shared";
import type { ClassifierEvalData } from "@/lib/model-metrics-types";

type ClassifierEvalMetricsProps = {
  data: ClassifierEvalData;
};

type ClassifierModel = ClassifierEvalData["models"][number];

export function ClassifierEvalMetrics({ data }: ClassifierEvalMetricsProps) {
  const accordionModels = data.models.filter(
    (model) => model.layout === "accordion",
  );
  const cardModels = data.models.filter((model) => model.layout === "card");

  const renderModelContent = (model: ClassifierModel) => (
    <div className="grid gap-6 lg:grid-cols-2">
      <EvalSummaryGrid
        columnsClassName="sm:grid-cols-2 lg:grid-cols-4"
        metrics={[
          { label: "Accuracy", value: formatEvalMetric(model.summary.accuracy) },
          { label: "Precision", value: formatEvalMetric(model.summary.precision) },
          { label: "Recall", value: formatEvalMetric(model.summary.recall) },
          { label: "F1 Score", value: formatEvalMetric(model.summary.f1) },
        ]}
      />
      <ConfusionMatrixPanel
        caption={model.matrixCaption}
        labels={data.abbreviations}
        matrix={model.confusionMatrix}
        minTableWidthClass="min-w-[320px]"
      />
      <PerClassMetricsPanel
        caption={model.perClassCaption}
        rows={model.perClass.map((metric) => ({
          id: metric.label,
          label: data.abbreviations[data.labels.indexOf(metric.label)],
          acc: metric.acc,
          p: metric.p,
          r: metric.r,
          f1: metric.f1,
        }))}
      />
    </div>
  );

  return (
    <div className="space-y-6">
      <Accordion type="multiple" className="space-y-4">
        {accordionModels.map((model) => (
          <AccordionItem
            key={model.id}
            value={model.id}
            className="rounded-2xl border border-border bg-card/60"
          >
            <AccordionTrigger className="px-5 py-4 text-left">
              <EvalModelHeader
                badge={model.badge}
                className="w-full"
                title={model.title}
              />
            </AccordionTrigger>
            <AccordionContent className="px-5 pb-5">
              {renderModelContent(model)}
            </AccordionContent>
          </AccordionItem>
        ))}
      </Accordion>
      {cardModels.map((model) => (
        <EvalModelCard key={model.id} badge={model.badge} title={model.title}>
          {renderModelContent(model)}
        </EvalModelCard>
      ))}
    </div>
  );
}
