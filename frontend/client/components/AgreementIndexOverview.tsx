import { Suspense, lazy, useEffect, useId, useMemo, useState } from "react";
import { ChartPanel } from "@/components/ChartPanel";
import { EmptyState, ErrorState } from "@/components/StateMessage";
import { AdaptiveTooltip } from "@/components/ui/adaptive-tooltip";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type {
  AgreementDealTypeYearRow,
  AgreementStatusYearRow,
  MetadataFieldCoverageRow,
} from "@shared/insights";
import {
  DEAL_TYPE_COLORS,
  DEAL_TYPE_FALLBACK_COLORS,
} from "@/lib/chart-palette";
import {
  formatEnumValue,
  formatNumberValue,
} from "@/lib/format-utils";
import { cn } from "@/lib/utils";
import { buildYearAxisGuides } from "@/lib/year-axis";
import { useIsMobile } from "@/hooks/use-mobile";
import {
  useAgreementDealTypesSummary,
  useAgreementStatusSummary,
} from "@/hooks/use-insights";

const ProcessingStatusChart = lazy(() =>
  import("@/components/AgreementIndexCharts").then((mod) => ({
    default: mod.ProcessingStatusChart,
  })),
);
const DealTypesChart = lazy(() =>
  import("@/components/AgreementIndexCharts").then((mod) => ({
    default: mod.DealTypesChart,
  })),
);

const STAGE_TOOLTIP_COPY = {
  "0_staging":
    "Agreements awaiting pre-processing (splitting into pages, classifying, etc.).",
  "1_pre_processing":
    "Agreements that have been pre-processed and are awaiting tagging via the NER model.",
  "2_tagging":
    "Agreements that have been tagged and are awaiting compilation into XML. All tag validation is done at the XML level.",
  "3_xml":
    "Agreements that have been compiled into XML and are awaiting verification (via AI or manually, if that fails) before an upsert to the sections table.",
} as const;

/** The green bucket of the status summary (`color = 'green'` in the ETL). */
const COMPLETE_STAGE_TOOLTIP =
  "Passed every pipeline stage: verified XML, sections stored, and no stale tags.";

const DEAL_TYPE_DISPLAY_ORDER = [
  "merger",
  "stock_acquisition",
  "asset_acquisition",
  "membership_interest_purchase",
  "tender_offer",
  "unknown",
];

const EMPTY_STATUS_ROWS: AgreementStatusYearRow[] = [];
const EMPTY_FIELD_COVERAGE: MetadataFieldCoverageRow[] = [];
const EMPTY_DEAL_TYPE_ROWS: AgreementDealTypeYearRow[] = [];

type DealTypeSeries = {
  dealType: string;
  key: string;
  label: string;
  color: string;
  total: number;
};

type DealTypeChartMode = "count" | "percent";

const DEAL_TYPE_MODE_OPTIONS = [
  { value: "count", label: "Counts" },
  { value: "percent", label: "100%" },
] as const satisfies ReadonlyArray<{ value: DealTypeChartMode; label: string }>;

/** Chart height, shared by the lazy chart and its fallback so nothing shifts. */
const CHART_HEIGHT_CLASS = "h-[240px] w-full sm:h-[300px] lg:h-[340px]";

/**
 * True once `ref` comes within `rootMargin` of the viewport, then stays true.
 * Without IntersectionObserver (old browsers, tests) it is true at once.
 */
function useNearViewport<T extends Element>(rootMargin = "300px") {
  // A callback ref (state) rather than useRef, so the effect reruns on attach.
  const [node, ref] = useState<T | null>(null);
  const [near, setNear] = useState(
    () => typeof IntersectionObserver === "undefined",
  );
  useEffect(() => {
    if (near || !node) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setNear(true);
          observer.disconnect();
        }
      },
      { rootMargin },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [near, node, rootMargin]);
  return [ref, near] as const;
}

const normalizeDealType = (dealType: string | null | undefined) => {
  if (!dealType) return "unknown";
  const normalized = dealType.trim();
  return normalized.length ? normalized : "unknown";
};

const formatDealTypeLabel = (dealType: string) =>
  dealType === "unknown" ? "Unclassified" : formatEnumValue(dealType);

const dealTypeSeriesKey = (dealType: string) =>
  `dealType_${dealType.replace(/[^a-z0-9]+/gi, "_")}`;

const formatCoverageRatio = (
  covered: number,
  eligible: number,
  label: string,
) =>
  `${formatNumberValue(covered, { maximumFractionDigits: 0 })} / ${formatNumberValue(
    eligible,
    {
      maximumFractionDigits: 0,
    },
  )} ${label}`;

const formatCoveragePct = (pct: number | null) =>
  pct === null ? "—" : `${pct.toFixed(1)}%`;

export function AgreementIndexOverview() {
  const isMobile = useIsMobile();
  const [dealTypeChartMode, setDealTypeChartMode] =
    useState<DealTypeChartMode>("count");
  const stagedChartDescriptionId = useId();
  const stagedChartTableId = useId();
  const dealTypeChartDescriptionId = useId();
  const dealTypeChartTableId = useId();
  const stagedTitleId = useId();
  const dealTypeTitleId = useId();
  // The deal-types panel sits below the fold: fetch its data as it nears view.
  const [dealTypesPanelRef, dealTypesPanelNear] = useNearViewport<HTMLDivElement>();
  const statusSummaryQuery = useAgreementStatusSummary();
  const statusSummaryData = statusSummaryQuery.data;
  const statusSummary = statusSummaryData?.years ?? EMPTY_STATUS_ROWS;
  const statusSummaryLoading = statusSummaryQuery.isLoading;
  const statusSummaryError = statusSummaryQuery.error;
  const statusSummaryMetadataCoveredAgreements =
    statusSummaryData?.metadata_covered_agreements ?? null;
  const statusSummaryMetadataCoveragePct = statusSummaryData?.metadata_coverage_pct ?? null;
  const statusSummaryMetadataFieldCoverage =
    statusSummaryData?.metadata_field_coverage ?? EMPTY_FIELD_COVERAGE;
  const statusSummaryTaxonomyCoveredSections =
    statusSummaryData?.taxonomy_covered_sections ?? null;
  const statusSummaryTaxonomyCoveragePct = statusSummaryData?.taxonomy_coverage_pct ?? null;
  const dealTypeSummaryQuery = useAgreementDealTypesSummary({ enabled: dealTypesPanelNear });
  const dealTypeSummary = dealTypeSummaryQuery.data?.years ?? EMPTY_DEAL_TYPE_ROWS;
  const dealTypeSummaryLoading = dealTypeSummaryQuery.isLoading;
  const dealTypeSummaryLoaded = dealTypeSummaryQuery.isLoaded;
  const dealTypeSummaryError = dealTypeSummaryQuery.error;

  const renderMetadataFieldCoverageTooltip = () => (
    <div className="space-y-3">
      <div className="space-y-1">
        <p className="text-xs font-medium text-foreground">Coverage by field</p>
        <p className="text-xs prose-copy">
          Pricing fields use consideration-aware denominators for both ingested
          and processed deals.
        </p>
      </div>
      <div className="space-y-2">
        {statusSummaryMetadataFieldCoverage.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            Field-level coverage is unavailable.
          </p>
        ) : (
          statusSummaryMetadataFieldCoverage.map((row) => (
            <div
              key={row.field}
              className="rounded-md border border-border bg-background/70 p-2"
            >
              <div className="text-xs font-medium text-foreground">
                {row.label}
              </div>
              <div className="mt-1 grid grid-cols-[minmax(0,1fr)_auto] items-start gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
                <div className="min-w-0">
                  {formatCoverageRatio(
                    row.processed_covered_agreements,
                    row.processed_eligible_agreements,
                    "processed deals",
                  )}
                </div>
                <div className="text-right font-mono tabular-nums">
                  {formatCoveragePct(row.processed_coverage_pct)}
                </div>
                <div className="min-w-0">
                  {formatCoverageRatio(
                    row.ingested_covered_agreements,
                    row.ingested_eligible_agreements,
                    "ingested deals",
                  )}
                </div>
                <div className="text-right font-mono tabular-nums">
                  {formatCoveragePct(row.ingested_coverage_pct)}
                </div>
              </div>
              {row.note ? (
                <div className="mt-1 text-[11px] leading-relaxed prose-copy">
                  {row.note}
                </div>
              ) : null}
            </div>
          ))
        )}
      </div>
    </div>
  );

  const renderAwaitingValidationTooltip = () => (
    <div className="space-y-3">
      <p className="text-xs leading-relaxed prose-copy">
        The vast majority of agreements awaiting validation are awaiting{" "}
        <em>XML validation</em>, which happens when they fail either our hard
        validation rules or are rejected by the LLM validator.
      </p>
      <p className="text-xs leading-relaxed prose-copy">
        Not all agreements that fail validation are invalid, however. We&apos;ve
        found that a non-negligible number of agreements awaiting XML validation
        accurately represent the original agreement, and it is the original
        agreement that fails validation.
      </p>
      <p className="text-xs leading-relaxed prose-copy">
        For instance,{" "}
        <a
          href="https://www.sec.gov/Archives/edgar/data/862861/000095017023013095/jan-ex10_95.htm"
          target="_blank"
          rel="noreferrer"
          className="text-foreground underline underline-offset-2"
        >
          this agreement
        </a>{" "}
        progresses straight from 4.10 to 4.12.{" "}
        <a
          href="https://www.sec.gov/Archives/edgar/data/1863181/000114036123012443/ny20008306x2_ex2-1.htm"
          target="_blank"
          rel="noreferrer"
          className="text-foreground underline underline-offset-2"
        >
          This one
        </a>{" "}
        has two sections labeled 12.1.{" "}
        <a
          href="https://www.sec.gov/Archives/edgar/data/1078799/000107997321000583/ex2x1.htm"
          target="_blank"
          rel="noreferrer"
          className="text-foreground underline underline-offset-2"
        >
          This one
        </a>{" "}
        skips straight from 4.12 to 4.14. And{" "}
        <a
          href="https://www.sec.gov/Archives/edgar/data/1820143/000119312521053301/d102219dex21.htm"
          target="_blank"
          rel="noreferrer"
          className="text-foreground underline underline-offset-2"
        >
          this one
        </a>{" "}
        skips straight from 8.07 to 8.09.
      </p>
      <p className="text-xs leading-relaxed prose-copy">
        We have implemented several measures to keep such agreements from being
        screened out, and are slowly running invalid agreements through this new
        system. In the meantime, only{" "}
        <span className="font-mono tabular-nums text-foreground">
          {xmlAwaitingValidationPct === null
            ? "—"
            : `${xmlAwaitingValidationPct.toFixed(1)}%`}
        </span>{" "}
        of ingested agreements are awaiting XML validation.
      </p>
    </div>
  );

  const renderMetricLabel = (
    label: string,
    metricKey: string,
  ): React.ReactNode => {
    if (metricKey === "processed") {
      return (
        <span className="inline-flex items-center gap-1">
          <span>{label}</span>
          <AdaptiveTooltip
            trigger={
              <button
                type="button"
                aria-label="Complete details"
                className="tooltip-help-trigger-compact min-h-[24px] min-w-[24px] sm:min-h-4 sm:min-w-4"
              >
                ?
              </button>
            }
            content={COMPLETE_STAGE_TOOLTIP}
            tooltipProps={{ side: "top", align: "start", className: "max-w-[320px] text-xs" }}
            popoverProps={{
              side: "top",
              align: "start",
              className: "w-[min(20rem,calc(100vw-2rem))] p-3 text-xs",
            }}
            delayDuration={0}
          />
        </span>
      );
    }
    if (metricKey === "metadata-coverage") {
      return (
        <span className="inline-flex items-center gap-1">
          <span>{label}</span>
          <AdaptiveTooltip
            trigger={
              <button
                type="button"
                aria-label="Metadata coverage details by field"
                className="tooltip-help-trigger-compact min-h-[24px] min-w-[24px] sm:min-h-4 sm:min-w-4"
              >
                ?
              </button>
            }
            content={renderMetadataFieldCoverageTooltip()}
            tooltipProps={{
              side: "top",
              align: "start",
              className:
                "max-h-[min(28rem,calc(100vh-6rem))] max-w-[340px] overflow-y-auto text-xs",
            }}
            popoverProps={{
              side: "top",
              align: "start",
              className:
                "max-h-[min(28rem,calc(100vh-6rem))] w-[min(22rem,calc(100vw-2rem))] max-w-[22rem] overflow-y-auto p-3 text-xs",
            }}
            delayDuration={0}
          />
        </span>
      );
    }
    if (metricKey === "awaiting") {
      return (
        <span className="inline-flex items-center gap-1">
          <span>{label}</span>
          <AdaptiveTooltip
            trigger={
              <button
                type="button"
                aria-label="Awaiting validation details"
                className="tooltip-help-trigger-compact min-h-[24px] min-w-[24px] sm:min-h-4 sm:min-w-4"
              >
                ?
              </button>
            }
            content={renderAwaitingValidationTooltip()}
            tooltipProps={{
              side: "top",
              align: "start",
              className:
                "max-h-[min(28rem,calc(100vh-6rem))] max-w-[360px] overflow-y-auto text-xs",
            }}
            popoverProps={{
              side: "top",
              align: "start",
              className:
                "max-h-[min(28rem,calc(100vh-6rem))] w-[min(24rem,calc(100vw-2rem))] max-w-[24rem] overflow-y-auto p-3 text-xs",
            }}
            delayDuration={0}
          />
        </span>
      );
    }
    return label;
  };

  const stagedChartData = useMemo(() => {
    const yearMap = new Map<
      number,
      {
        year: number;
        processed: number;
        staged: number;
        awaiting: number;
        notPaginated: number;
      }
    >();
    statusSummary.forEach((row) => {
      const year = Number(row.year);
      if (!Number.isFinite(year)) return;
      const count = Math.max(0, Number(row.count || 0));
      const entry = yearMap.get(year) ?? {
        year,
        processed: 0,
        staged: 0,
        awaiting: 0,
        notPaginated: 0,
      };
      if (row.color === "green") {
        entry.processed += count;
      } else if (row.color === "yellow") {
        entry.staged += count;
      } else if (row.color === "red") {
        entry.awaiting += count;
      } else if (row.color === "gray") {
        entry.notPaginated += count;
      }
      yearMap.set(year, entry);
    });
    return Array.from(yearMap.values()).sort((a, b) => a.year - b.year);
  }, [statusSummary]);
  const stagedTotals = useMemo(() => {
    return stagedChartData.reduce(
      (acc, row) => {
        acc.processed += row.processed;
        acc.staged += row.staged;
        acc.awaiting += row.awaiting;
        acc.notPaginated += row.notPaginated;
        acc.total +=
          row.processed + row.staged + row.awaiting + row.notPaginated;
        return acc;
      },
      { processed: 0, staged: 0, awaiting: 0, notPaginated: 0, total: 0 },
    );
  }, [stagedChartData]);
  const stagedSummaryMetrics = useMemo(() => {
    const total = stagedTotals.total;
    const pctOfTotal = (value: number) =>
      total > 0 ? Math.round((value / total) * 1000) / 10 : 0;
    const formatPctDetail = (
      value: number | null,
      denominatorLabel: "searchable" | "sections",
    ) =>
      value === null
        ? `—% of ${denominatorLabel}`
        : `${value.toFixed(1)}% of ${denominatorLabel}`;
    return [
      {
        key: "staged",
        label: "Staged",
        value: stagedTotals.staged,
        detail: `${pctOfTotal(stagedTotals.staged).toFixed(1)}% of total`,
      },
      {
        key: "awaiting",
        label: "Awaiting validation",
        value: stagedTotals.awaiting,
        detail: `${pctOfTotal(stagedTotals.awaiting).toFixed(1)}% of total`,
      },
      {
        key: "processed",
        label: "Complete",
        value: stagedTotals.processed,
        detail: `${pctOfTotal(stagedTotals.processed).toFixed(1)}% of total`,
      },
      {
        key: "not-paginated",
        label: "Not paginated",
        value: stagedTotals.notPaginated,
        detail: `${pctOfTotal(stagedTotals.notPaginated).toFixed(1)}% of total`,
      },
      {
        key: "metadata-coverage",
        label: "Metadata coverage",
        value: statusSummaryMetadataCoveredAgreements ?? "—",
        detail: formatPctDetail(statusSummaryMetadataCoveragePct, "searchable"),
      },
      {
        key: "taxonomy-coverage",
        label: "Taxonomy coverage",
        value: statusSummaryTaxonomyCoveredSections ?? "—",
        detail: formatPctDetail(statusSummaryTaxonomyCoveragePct, "sections"),
      },
    ] as const;
  }, [
    stagedTotals,
    statusSummaryMetadataCoveredAgreements,
    statusSummaryMetadataCoveragePct,
    statusSummaryTaxonomyCoveredSections,
    statusSummaryTaxonomyCoveragePct,
  ]);
  const stagedPrimarySummaryMetrics = stagedSummaryMetrics.slice(0, 4);
  const stagedSecondarySummaryMetrics = stagedSummaryMetrics.slice(4);
  const stagedYearRange = useMemo(() => {
    if (stagedChartData.length === 0) return null;
    const minYear = stagedChartData[0].year;
    const maxYear = stagedChartData[stagedChartData.length - 1].year;
    return { minYear, maxYear };
  }, [stagedChartData]);
  const showSourceSplit =
    stagedYearRange !== null &&
    stagedYearRange.minYear <= 2020 &&
    stagedYearRange.maxYear >= 2021;
  const stagedYearTicks = useMemo(
    () =>
      stagedChartData.length > 0
        ? buildYearAxisGuides(stagedChartData).majorYears
        : undefined,
    [stagedChartData],
  );
  const dealTypeSeries = useMemo<DealTypeSeries[]>(() => {
    const totalsByType = new Map<string, number>();
    dealTypeSummary.forEach((row) => {
      const dealType = normalizeDealType(row.deal_type);
      const count = Math.max(0, Number(row.count || 0));
      totalsByType.set(dealType, (totalsByType.get(dealType) ?? 0) + count);
    });

    const orderedKnown = DEAL_TYPE_DISPLAY_ORDER.filter((dealType) =>
      totalsByType.has(dealType),
    );
    const orderedRemaining = Array.from(totalsByType.keys())
      .filter((dealType) => !DEAL_TYPE_DISPLAY_ORDER.includes(dealType))
      .sort((a, b) => a.localeCompare(b));
    const orderedDealTypes = [...orderedKnown, ...orderedRemaining];

    return orderedDealTypes.map((dealType, index) => ({
      dealType,
      key: dealTypeSeriesKey(dealType),
      label: formatDealTypeLabel(dealType),
      color:
        DEAL_TYPE_COLORS[dealType] ??
        DEAL_TYPE_FALLBACK_COLORS[index % DEAL_TYPE_FALLBACK_COLORS.length],
      total: totalsByType.get(dealType) ?? 0,
    }));
  }, [dealTypeSummary]);
  const dealTypeChartData = useMemo(() => {
    const seriesByType = new Map(
      dealTypeSeries.map((series) => [series.dealType, series]),
    );
    const yearMap = new Map<
      number,
      { year: number } & Record<string, number>
    >();

    dealTypeSummary.forEach((row) => {
      const year = Number(row.year);
      if (!Number.isFinite(year)) return;
      const dealType = normalizeDealType(row.deal_type);
      const series = seriesByType.get(dealType);
      if (!series) return;
      const count = Math.max(0, Number(row.count || 0));
      const entry = yearMap.get(year) ?? { year };
      entry[series.key] = (entry[series.key] ?? 0) + count;
      yearMap.set(year, entry);
    });

    const rows = Array.from(yearMap.values()).sort((a, b) => a.year - b.year);
    rows.forEach((row) => {
      dealTypeSeries.forEach((series) => {
        row[series.key] = row[series.key] ?? 0;
      });
    });
    return rows;
  }, [dealTypeSeries, dealTypeSummary]);
  const dealTypeTotals = useMemo(() => {
    const total = dealTypeSeries.reduce((sum, series) => sum + series.total, 0);
    const metrics = dealTypeSeries.map((series) => {
      const pct =
        total > 0 ? Math.round((series.total / total) * 1000) / 10 : 0;
      return {
        key: series.key,
        label: series.label,
        value: series.total,
        pct,
      };
    });
    return { total, metrics };
  }, [dealTypeSeries]);
  const dealTypeYearRange = useMemo(() => {
    if (dealTypeChartData.length === 0) return null;
    const minYear = dealTypeChartData[0].year;
    const maxYear = dealTypeChartData[dealTypeChartData.length - 1].year;
    return { minYear, maxYear };
  }, [dealTypeChartData]);
  const showDealTypeSourceSplit =
    dealTypeYearRange !== null &&
    dealTypeYearRange.minYear <= 2020 &&
    dealTypeYearRange.maxYear >= 2021;
  const dealTypeYearTicks = useMemo(
    () =>
      dealTypeChartData.length > 0
        ? buildYearAxisGuides(dealTypeChartData).majorYears
        : undefined,
    [dealTypeChartData],
  );
  const stageSummaryRows = useMemo(() => {
    const stageOrder = [
      { key: "0_staging", label: "Staging" },
      { key: "1_pre_processing", label: "Pre-processing" },
      { key: "2_tagging", label: "Tagging" },
      { key: "3_xml", label: "XML validation" },
    ] as const;
    type StageKey = (typeof stageOrder)[number]["key"];
    const stageMap = new Map<
      StageKey,
      { key: StageKey; label: string; staged: number; awaiting: number }
    >(
      stageOrder.map((stage) => [
        stage.key,
        { ...stage, staged: 0, awaiting: 0 },
      ]),
    );
    statusSummary.forEach((row) => {
      const entry = stageMap.get(row.current_stage as StageKey);
      if (!entry) return;
      const count = Math.max(0, Number(row.count || 0));
      if (row.color === "yellow") {
        entry.staged += count;
      } else if (row.color === "red") {
        entry.awaiting += count;
      }
    });
    return stageOrder.map((stage) => stageMap.get(stage.key)!);
  }, [statusSummary]);
  const xmlAwaitingValidationPct = useMemo(() => {
    const xmlAwaitingCount =
      stageSummaryRows.find((row) => row.key === "3_xml")?.awaiting ?? 0;
    return stagedTotals.total > 0
      ? (xmlAwaitingCount / stagedTotals.total) * 100
      : null;
  }, [stageSummaryRows, stagedTotals.total]);

  const renderStageLabel = (row: (typeof stageSummaryRows)[number]) => (
    <span className="inline-flex items-center gap-1">
      <span>{row.label}</span>
      <AdaptiveTooltip
        trigger={
          <button
            type="button"
            aria-label={`${row.label} stage details`}
            className="tooltip-help-trigger-compact min-h-[24px] min-w-[24px] sm:min-h-4 sm:min-w-4"
          >
            ?
          </button>
        }
        content={<p>{STAGE_TOOLTIP_COPY[row.key]}</p>}
        tooltipProps={{
          side: "top",
          className: "max-w-[260px] text-xs",
        }}
        popoverProps={{
          side: "top",
          className: "w-auto max-w-[260px] p-2 text-xs",
        }}
        delayDuration={0}
      />
    </span>
  );

  const renderStagedChart = (className?: string) => (
    <Suspense
      fallback={<Skeleton className={CHART_HEIGHT_CLASS} />}
    >
      <ProcessingStatusChart
        className={className}
        data={stagedChartData}
        describedBy={stagedChartDescriptionId}
        isMobile={isMobile}
        showSourceSplit={showSourceSplit}
        tableId={stagedChartTableId}
        yearRange={stagedYearRange}
        yearTicks={stagedYearTicks}
      />
    </Suspense>
  );

  const renderStagedSummaryTable = (className?: string) => (
    <div className={className}>
      <div className="grid gap-2 sm:hidden">
        {stagedSummaryMetrics.map((metric) => (
          <dl
            key={metric.key}
            className="rounded-lg border border-border p-3"
          >
            <dt className="text-xs font-semibold text-muted-foreground">
              {renderMetricLabel(metric.label, metric.key)}
            </dt>
            <dd className="mt-1 text-base font-semibold text-foreground">
              {typeof metric.value === "number"
                ? metric.value.toLocaleString("en-US")
                : metric.value}
            </dd>
            <dd className="text-xs text-muted-foreground">{metric.detail}</dd>
          </dl>
        ))}
      </div>
      <div className="hidden overflow-x-auto rounded-lg border border-border sm:grid">
        <div className="grid min-w-[520px] grid-cols-4">
          {stagedPrimarySummaryMetrics.map((metric, index) => (
            <div
              key={metric.key}
              className={cn("p-4", index > 0 && "border-l border-border")}
            >
              <div className="text-xs font-semibold text-muted-foreground">
                {renderMetricLabel(metric.label, metric.key)}
              </div>
              <div className="mt-2 text-base font-semibold text-foreground">
                {typeof metric.value === "number"
                  ? metric.value.toLocaleString("en-US")
                  : metric.value}
              </div>
              <div className="text-xs text-muted-foreground">
                {metric.detail}
              </div>
            </div>
          ))}
        </div>
        <div className="grid min-w-[520px] grid-cols-2 border-t border-border">
          {stagedSecondarySummaryMetrics.map((metric, index) => (
            <div
              key={metric.key}
              className={cn("p-4", index > 0 && "border-l border-border")}
            >
              <div className="text-xs font-semibold text-muted-foreground">
                {renderMetricLabel(metric.label, metric.key)}
              </div>
              <div className="mt-2 text-base font-semibold text-foreground">
                {typeof metric.value === "number"
                  ? metric.value.toLocaleString("en-US")
                  : metric.value}
              </div>
              <div className="text-xs text-muted-foreground">
                {metric.detail}
              </div>
            </div>
          ))}
        </div>
      </div>
      <div className="sr-only"><table>
        <caption>
          Summary totals for staged, awaiting validation, complete, and not
          paginated agreements, plus metadata coverage and taxonomy coverage.
        </caption>
        <thead>
          <tr>
            <th scope="col">Metric</th>
            <th scope="col">Value</th>
            <th scope="col">Detail</th>
          </tr>
        </thead>
        <tbody>
          {stagedSummaryMetrics.map((metric) => (
            <tr key={`staged-summary-${metric.key}`}>
              <th scope="row">{metric.label}</th>
              <td>
                {typeof metric.value === "number"
                  ? metric.value.toLocaleString("en-US")
                  : metric.value}
              </td>
              <td>{metric.detail}</td>
            </tr>
          ))}
        </tbody>
      </table></div>
    </div>
  );

  const renderDealTypeSummaryTable = (className?: string) => (
    <div className={className}>
      <div className="grid gap-2 sm:hidden">
        {dealTypeTotals.metrics.map((metric) => (
          <dl
            key={metric.key}
            className="rounded-lg border border-border p-3"
          >
            <dt className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              {metric.label}
            </dt>
            <dd className="mt-1 text-base font-semibold text-foreground">
              {metric.value.toLocaleString("en-US")}
            </dd>
            <dd className="text-xs text-muted-foreground">
              {metric.pct.toFixed(1)}% of total
            </dd>
          </dl>
        ))}
      </div>
      <div className="hidden overflow-x-auto rounded-lg border border-border sm:block">
        <Table className="min-w-[520px]">
          <caption className="sr-only">
            Deal counts by deal type across all filing years.
          </caption>
          <TableHeader>
            <TableRow>
              {dealTypeTotals.metrics.map((metric) => (
                <TableHead
                  key={metric.key}
                  scope="col"
                  className="text-xs uppercase tracking-wide text-muted-foreground"
                >
                  {metric.label}
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            <TableRow>
              {dealTypeTotals.metrics.map((metric) => (
                <TableCell key={metric.key} className="align-top">
                  <div className="text-base font-semibold text-foreground">
                    {metric.value.toLocaleString("en-US")}
                  </div>
                  <div className="text-xs font-mono tabular-nums text-muted-foreground">
                    {metric.pct.toFixed(1)}% of total
                  </div>
                </TableCell>
              ))}
            </TableRow>
          </TableBody>
        </Table>
      </div>
    </div>
  );

  const renderDealTypeChart = (className?: string) => (
    <Suspense
      fallback={<Skeleton className={CHART_HEIGHT_CLASS} />}
    >
      <DealTypesChart
        className={className}
        data={dealTypeChartData}
        describedBy={dealTypeChartDescriptionId}
        isMobile={isMobile}
        mode={dealTypeChartMode}
        series={dealTypeSeries}
        showSourceSplit={showDealTypeSourceSplit}
        tableId={dealTypeChartTableId}
        yearRange={dealTypeYearRange}
        yearTicks={dealTypeYearTicks}
      />
    </Suspense>
  );

  const renderStageFunnelTable = (className?: string) => (
    <div className={className}>
      <div className="grid gap-2 sm:hidden">
        {stageSummaryRows.map((row) => (
          <dl
            key={row.key}
            className="rounded-lg border border-border p-3"
          >
            <dt className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              {renderStageLabel(row)}
            </dt>
            <div className="mt-2 grid grid-cols-2 gap-3">
              <div>
                <dd className="text-xs uppercase tracking-wide text-muted-foreground">
                  Staged
                </dd>
                <dd className="text-base font-semibold text-foreground">
                  {row.staged.toLocaleString("en-US")}
                </dd>
              </div>
              <div>
                <dd className="text-xs uppercase tracking-wide text-muted-foreground">
                  Awaiting validation
                </dd>
                <dd className="text-base font-semibold text-foreground">
                  {row.awaiting.toLocaleString("en-US")}
                </dd>
              </div>
            </div>
          </dl>
        ))}
      </div>
      <div className="hidden overflow-x-auto rounded-lg border border-border sm:block">
        <Table className="min-w-[520px]">
          <caption className="sr-only">
            Staged versus awaiting validation agreements by pipeline stage.
          </caption>
          <TableHeader>
            <TableRow>
              <TableHead
                scope="col"
                className="text-xs uppercase tracking-wide text-muted-foreground"
              >
                Stage
              </TableHead>
              <TableHead
                scope="col"
                className="text-xs uppercase tracking-wide text-muted-foreground"
              >
                Staged
              </TableHead>
              <TableHead
                scope="col"
                className="text-xs uppercase tracking-wide text-muted-foreground"
              >
                Awaiting validation
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {stageSummaryRows.map((row) => (
              <TableRow key={row.key}>
                <TableCell className="font-medium text-foreground">
                  {renderStageLabel(row)}
                </TableCell>
                <TableCell className="font-mono tabular-nums text-foreground">
                  {row.staged.toLocaleString("en-US")}
                </TableCell>
                <TableCell className="font-mono tabular-nums text-foreground">
                  {row.awaiting.toLocaleString("en-US")}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );

  const panelSkeleton = (
    <div className="space-y-4" aria-hidden="true">
      <Skeleton className="h-[88px] w-full" />
      <Skeleton className={CHART_HEIGHT_CLASS} />
    </div>
  );

  // Footnotes and chart controls describe a drawn chart, so they wait for one.
  const stagedChartShown =
    !statusSummaryLoading && !statusSummaryError && stagedChartData.length > 0;
  const dealTypeChartShown =
    !dealTypeSummaryLoading &&
    dealTypeSummaryLoaded &&
    !dealTypeSummaryError &&
    dealTypeChartData.length > 0 &&
    dealTypeSeries.length > 0;

  return (
    <div className="space-y-6">
      <ChartPanel
        id={stagedTitleId}
        title="Processing status"
        description={
          <span id={stagedChartDescriptionId} className="prose-copy">
            Every ingested agreement by pipeline stage and filing year. Staged
            agreements have not yet gone through our pipelines; agreements
            awaiting validation tripped one of our validations and are waiting
            for review. Neither appears in the{" "}
            <span className="font-mono text-foreground">/v1/sections/*</span> or{" "}
            <span className="font-mono text-foreground">/v1/agreements/*</span>{" "}
            routes.
          </span>
        }
        footnote={
          stagedChartShown
            ? "The dashed divider marks the 2020/2021 boundary: from 2000 through 2020 the data comes from the DMA Corpus; from 2021 we source it from EDGAR."
            : undefined
        }
        mobileExpand={stagedChartData.length > 0 ? () => renderStagedChart() : false}
      >
        {statusSummaryLoading ? (
          panelSkeleton
        ) : statusSummaryError ? (
          <ErrorState
            error={statusSummaryQuery.apiError ?? statusSummaryError}
            onRetry={() => void statusSummaryQuery.refetch()}
            variant="inline"
          />
        ) : stagedChartData.length === 0 ? (
          <EmptyState title="No processing status data yet." variant="inline" />
        ) : (
          <div className="space-y-4">
            {renderStagedSummaryTable()}
            {renderStagedChart()}
            {renderStageFunnelTable()}
            <div className="sr-only"><table id={stagedChartTableId}>
              <caption>
                Complete, awaiting validation, staged, and not paginated
                agreements by filing year
              </caption>
              <thead>
                <tr>
                  <th scope="col">Year</th>
                  <th scope="col">Complete</th>
                  <th scope="col">Awaiting validation</th>
                  <th scope="col">Staged</th>
                  <th scope="col">Not paginated</th>
                  <th scope="col">Total</th>
                </tr>
              </thead>
              <tbody>
                {stagedChartData.map((row) => (
                  <tr key={`staged-row-${row.year}`}>
                    <th scope="row">{row.year}</th>
                    <td>{row.processed.toLocaleString("en-US")}</td>
                    <td>{row.awaiting.toLocaleString("en-US")}</td>
                    <td>{row.staged.toLocaleString("en-US")}</td>
                    <td>{row.notPaginated.toLocaleString("en-US")}</td>
                    <td>
                      {(
                        row.processed +
                        row.awaiting +
                        row.staged +
                        row.notPaginated
                      ).toLocaleString("en-US")}
                    </td>
                  </tr>
                ))}
                <tr>
                  <th scope="row">Total</th>
                  <td>{stagedTotals.processed.toLocaleString("en-US")}</td>
                  <td>{stagedTotals.awaiting.toLocaleString("en-US")}</td>
                  <td>{stagedTotals.staged.toLocaleString("en-US")}</td>
                  <td>{stagedTotals.notPaginated.toLocaleString("en-US")}</td>
                  <td>{stagedTotals.total.toLocaleString("en-US")}</td>
                </tr>
              </tbody>
            </table></div>
          </div>
        )}
      </ChartPanel>

      <div ref={dealTypesPanelRef}>
        <ChartPanel
          id={dealTypeTitleId}
          title="Deal types"
          description={
            <span id={dealTypeChartDescriptionId} className="prose-copy">
              Processed agreements by deal type and filing year.
            </span>
          }
          controls={
            dealTypeChartShown ? (
            <SegmentedControl<DealTypeChartMode>
              value={dealTypeChartMode}
              onValueChange={setDealTypeChartMode}
              options={DEAL_TYPE_MODE_OPTIONS}
              aria-label="Deal type chart mode"
              size="sm"
            />
            ) : undefined
          }
          footnote={
            dealTypeChartShown
              ? "The dashed divider marks the 2020/2021 boundary between DMA Corpus and EDGAR-sourced data."
              : undefined
          }
          mobileExpand={
            dealTypeChartData.length > 0 && dealTypeSeries.length > 0
              ? () => renderDealTypeChart()
              : false
          }
        >
          {dealTypeSummaryLoading ||
          (!dealTypeSummaryLoaded && !dealTypeSummaryError) ? (
            panelSkeleton
          ) : dealTypeSummaryError ? (
            <ErrorState
              error={dealTypeSummaryQuery.apiError ?? dealTypeSummaryError}
              onRetry={() => void dealTypeSummaryQuery.refetch()}
              variant="inline"
            />
          ) : dealTypeChartData.length === 0 || dealTypeSeries.length === 0 ? (
            <EmptyState title="No deal type data yet." variant="inline" />
          ) : (
            <div className="space-y-4">
              {renderDealTypeSummaryTable()}
              {renderDealTypeChart()}
              <div className="sr-only"><table id={dealTypeChartTableId}>
                <caption>Deal type counts by filing year</caption>
                <thead>
                  <tr>
                    <th scope="col">Year</th>
                    {dealTypeSeries.map((series) => (
                      <th key={`deal-type-head-${series.key}`} scope="col">
                        {series.label}
                      </th>
                    ))}
                    <th scope="col">Total</th>
                  </tr>
                </thead>
                <tbody>
                  {dealTypeChartData.map((row) => {
                    const rowTotal = dealTypeSeries.reduce(
                      (sum, series) => sum + Number(row[series.key] ?? 0),
                      0,
                    );
                    return (
                      <tr key={`deal-type-row-${row.year}`}>
                        <th scope="row">{row.year}</th>
                        {dealTypeSeries.map((series) => (
                          <td key={`deal-type-cell-${row.year}-${series.key}`}>
                            {Number(row[series.key] ?? 0).toLocaleString("en-US")}
                          </td>
                        ))}
                        <td>{rowTotal.toLocaleString("en-US")}</td>
                      </tr>
                    );
                  })}
                  <tr>
                    <th scope="row">Total</th>
                    {dealTypeSeries.map((series) => (
                      <td key={`deal-type-total-${series.key}`}>
                        {series.total.toLocaleString("en-US")}
                      </td>
                    ))}
                    <td>{dealTypeTotals.total.toLocaleString("en-US")}</td>
                  </tr>
                </tbody>
              </table></div>
            </div>
          )}
        </ChartPanel>
      </div>
    </div>
  );
}
