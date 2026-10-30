import { useState, type ReactNode } from "react";
import { Maximize2 } from "lucide-react";

import { MobileChartModal } from "@/components/MobileChartModal";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export interface ChartPanelProps {
  title: string;
  description?: ReactNode;
  /** Right-aligned controls (SegmentedControl, Select...). Wrap under the title on mobile. */
  controls?: ReactNode;
  /** The chart. Rendered again inside the full-screen modal when expanded. */
  children: ReactNode;
  /** Small print under the chart (sources, definitions). */
  footnote?: ReactNode;
  /**
   * Show an "Expand" button below `sm` that opens the chart full screen.
   * Pass a function to render a different (e.g. taller) chart in the modal.
   */
  mobileExpand?: boolean | (() => ReactNode);
  /** Heading level for the title. Defaults to h2. */
  titleAs?: "h2" | "h3";
  /** id for the title (e.g. for deep links). */
  id?: string;
  className?: string;
}

/** Bordered panel for one chart: title, description, controls, chart, footnote. */
export function ChartPanel({
  title,
  description,
  controls,
  children,
  footnote,
  mobileExpand = false,
  titleAs: Title = "h2",
  id,
  className,
}: ChartPanelProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <section
      aria-labelledby={id}
      className={cn("rounded-xl border border-border bg-card shadow-raised p-4 sm:p-6", className)}
    >
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between sm:gap-6">
        <div className="min-w-0">
          <Title id={id} className="text-lg font-semibold tracking-tight text-foreground">
            {title}
          </Title>
          {description ? (
            <div className="mt-1 text-sm text-muted-foreground">{description}</div>
          ) : null}
        </div>
        {controls ? (
          <div className="flex flex-wrap items-center gap-2 sm:justify-end">{controls}</div>
        ) : null}
      </div>
      <div className="mt-4 min-w-0">{children}</div>
      {mobileExpand ? (
        <div className="mt-3 sm:hidden">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => setExpanded(true)}
            aria-label={`Expand ${title} chart`}
          >
            <Maximize2 aria-hidden="true" />
            Expand
          </Button>
          <MobileChartModal
            open={expanded}
            onOpenChange={setExpanded}
            title={title}
            description={typeof description === "string" ? description : undefined}
          >
            <h2 className="mb-4 text-lg font-semibold tracking-tight">{title}</h2>
            {typeof mobileExpand === "function" ? mobileExpand() : children}
          </MobileChartModal>
        </div>
      ) : null}
      {footnote ? (
        <div className="mt-4 text-xs text-muted-foreground">{footnote}</div>
      ) : null}
    </section>
  );
}
