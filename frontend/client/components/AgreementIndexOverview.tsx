import { Suspense, lazy, useEffect, useId, useMemo, useState } from "react";
import { AdaptiveTooltip } from "@/components/ui/adaptive-tooltip";
import { Button } from "@/components/ui/button";
import { MobileChartModal } from "@/components/MobileChartModal";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";
import { formatDateValue, formatEnumValue, formatNumberValue } from "@/lib/format-utils";
import { cn } from "@/lib/utils";
import { useIsMobile } from "@/hooks/use-mobile";

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

const DEAL_TYPE_DISPLAY_ORDER = [
  "merger",
  "stock_acquisition",
  "asset_acquisition",
  "membership_interest_purchase",
  "tender_offer",
  "unknown",
];

const DEAL_TYPE_COLORS: Record<string, string> = {
  merger: "hsl(212 93% 50%)",
  stock_acquisition: "hsl(170 84% 36%)",
  asset_acquisition: "hsl(35 92% 52%)",
  membership_interest_purchase: "hsl(196 83% 42%)",
  tender_offer: "hsl(0 84% 60%)",
  unknown: "hsl(220 9% 60%)",
};

type AgreementStatusYearRow = {
  year: number;
  color: "green" | "yellow" | "red" | "gray";
  current_stage: string;
  count: number;
};

type AgreementStatusSummaryResponse = {
  years: AgreementStatusYearRow[];
  latest_filing_date: string | null;
  metadata_covered_agreements: number | null;
  metadata_coverage_pct: number | null;
  metadata_field_coverage: MetadataFieldCoverageRow[];
  taxonomy_covered_sections: number | null;
  taxonomy_coverage_pct: number | null;
};

type MetadataFieldCoverageRow = {
  field: string;
  label: string;
  ingested_eligible_agreements: number;
  ingested_covered_agreements: number;
  ingested_coverage_pct: number | null;
  processed_eligible_agreements: number;
  processed_covered_agreements: number;
  processed_coverage_pct: number | null;
  note: string | null;
};

type AgreementDealTypeYearRow = {
  year: number;
  deal_type: string;
  count: number;
};

type AgreementDealTypeSummaryResponse = {
  years: AgreementDealTypeYearRow[];
};

type DealTypeSeries = {
  dealType: string;
  key: string;
  label: string;
  color: string;
  total: number;
};

type OverviewTab = "processing-status" | "deal-types";
type DealTypeChartMode = "count" | "percent";

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
  `${formatNumberValue(covered, { maximumFractionDigits: 0 })} / ${formatNumberValue(eligible, {
    maximumFractionDigits: 0,
  })} ${label}`;

const formatCoveragePct = (pct: number | null) =>
  pct === null ? "—" : `${pct.toFixed(1)}%`;

export function AgreementIndexOverview() {
  const isMobile = useIsMobile();
  const [isProcessingChartModalOpen, setIsProcessingChartModalOpen] =
    useState(false);
  const [isDealTypesChartModalOpen, setIsDealTypesChartModalOpen] =
    useState(false);
  const [overviewTab, setOverviewTab] =
    useState<OverviewTab>("processing-status");
  const [dealTypeChartMode, setDealTypeChartMode] =
    useState<DealTypeChartMode>("count");
  const [statusSummary, setStatusSummary] = useState<AgreementStatusYearRow[]>(
    [],
  );
  const [statusSummaryLoading, setStatusSummaryLoading] = useState(false);
  const [statusSummaryLoaded, setStatusSummaryLoaded] = useState(false);
  const [statusSummaryError, setStatusSummaryError] = useState<string | null>(
    null,
  );
  const [statusSummaryLatestFilingDate, setStatusSummaryLatestFilingDate] =
    useState<string | null>(null);
  const [
    statusSummaryMetadataCoveredAgreements,
    setStatusSummaryMetadataCoveredAgreements,
  ] = useState<number | null>(null);
  const [statusSummaryMetadataCoveragePct, setStatusSummaryMetadataCoveragePct] =
    useState<number | null>(null);
  const [statusSummaryMetadataFieldCoverage, setStatusSummaryMetadataFieldCoverage] =
    useState<MetadataFieldCoverageRow[]>([]);
  const [
    statusSummaryTaxonomyCoveredSections,
    setStatusSummaryTaxonomyCoveredSections,
  ] = useState<number | null>(null);
  const [statusSummaryTaxonomyCoveragePct, setStatusSummaryTaxonomyCoveragePct] =
    useState<number | null>(null);
  const [dealTypeSummary, setDealTypeSummary] = useState<
    AgreementDealTypeYearRow[]
  >([]);
  const [dealTypeSummaryLoading, setDealTypeSummaryLoading] = useState(false);
  const [dealTypeSummaryLoaded, setDealTypeSummaryLoaded] = useState(false);
  const [dealTypeSummaryError, setDealTypeSummaryError] = useState<
    string | null
  >(null);
  const stagedChartDescriptionId = useId();
  const stagedChartTableId = useId();
  const dealTypeChartDescriptionId = useId();
  const dealTypeChartTableId = useId();
  const dealTypesTabOpen = overviewTab === "deal-types";

  const renderMetadataFieldCoverageTooltip = () => (
    <div className="space-y-3">
      <div className="space-y-1">
        <p className="text-xs font-medium text-foreground">
          Coverage by field
        </p>
        <p className="text-xs text-muted-foreground">
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
              className="rounded-md border border-border/60 bg-background/70 p-2"
            >
              <div className="text-xs font-medium text-foreground">{row.label}</div>
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
                <div className="mt-1 text-[11px] leading-relaxed text-muted-foreground">
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
      <p className="text-xs leading-relaxed text-muted-foreground">
        The vast majority of agreements awaiting validation are awaiting{" "}
        <em>XML validation</em>, which happens when they fail either our hard
        validation rules or are rejected by the LLM validator.
      </p>
      <p className="text-xs leading-relaxed text-muted-foreground">
        Not all agreements that fail validation are invalid, however. We&apos;ve
        found that a non-negligible number of agreements awaiting XML
        validation accurately represent the original agreement, and it is the
        original agreement that fails validation.
      </p>
      <p className="text-xs leading-relaxed text-muted-foreground">
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
      <p className="text-xs leading-relaxed text-muted-foreground">
        We have not yet upgraded our validation pipelines to validate based on
        the table of contents rather than using hard validation rules only, but
        hope to do so soon.
      </p>
      <p className="text-xs leading-relaxed text-muted-foreground">
        In the meantime, only{" "}
        <span className="font-mono tabular-nums text-foreground">
          {xmlAwaitingValidationPct === null ? "—" : `${xmlAwaitingValidationPct.toFixed(1)}%`}
        </span>{" "}
        of ingested agreements are awaiting XML validation.
      </p>
    </div>
  );

  const renderMetricLabel = (
    label: string,
    metricKey: string,
  ): React.ReactNode => {
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
              className: "max-h-[min(28rem,calc(100vh-6rem))] max-w-[340px] overflow-y-auto text-xs",
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
              className: "max-h-[min(28rem,calc(100vh-6rem))] max-w-[360px] overflow-y-auto text-xs",
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

  useEffect(() => {
    if (statusSummaryLoaded) return;

    let cancelled = false;
    const controller = new AbortController();

    const fetchStatusSummary = async () => {
      try {
        setStatusSummaryLoading(true);
        setStatusSummaryError(null);
        const res = await authFetch(apiUrl("v1/agreements-status-summary"), {
          signal: controller.signal,
        });
        if (!res.ok) {
          throw new Error(`Status summary request failed (${res.status})`);
        }
        const data = (await res.json()) as AgreementStatusSummaryResponse;
        if (!cancelled) {
          setStatusSummary(data.years ?? []);
          setStatusSummaryLatestFilingDate(data.latest_filing_date ?? null);
          setStatusSummaryMetadataCoveredAgreements(
            data.metadata_covered_agreements ?? null,
          );
          setStatusSummaryMetadataCoveragePct(data.metadata_coverage_pct ?? null);
          setStatusSummaryMetadataFieldCoverage(
            data.metadata_field_coverage ?? [],
          );
          setStatusSummaryTaxonomyCoveredSections(
            data.taxonomy_covered_sections ?? null,
          );
          setStatusSummaryTaxonomyCoveragePct(data.taxonomy_coverage_pct ?? null);
          setStatusSummaryLoaded(true);
        }
      } catch (err) {
        if (
          !cancelled &&
          !(err instanceof DOMException && err.name === "AbortError")
        ) {
          setStatusSummaryError(
            err instanceof Error
              ? err.message
              : "Unable to load staging summary.",
          );
        }
      } finally {
        if (!cancelled) {
          setStatusSummaryLoading(false);
        }
      }
    };

    fetchStatusSummary();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [statusSummaryLoaded]);

  useEffect(() => {
    if (!dealTypesTabOpen || dealTypeSummaryLoaded) {
      return;
    }

    let cancelled = false;
    const controller = new AbortController();

    const fetchDealTypeSummary = async () => {
      try {
        setDealTypeSummaryLoading(true);
        setDealTypeSummaryError(null);
        const res = await authFetch(
          apiUrl("v1/agreements-deal-types-summary"),
          {
            signal: controller.signal,
          },
        );
        if (!res.ok) {
          throw new Error(`Deal type summary request failed (${res.status})`);
        }
        const data = (await res.json()) as AgreementDealTypeSummaryResponse;
        if (!cancelled) {
          setDealTypeSummary(data.years ?? []);
          setDealTypeSummaryLoaded(true);
        }
      } catch (err) {
        if (
          !cancelled &&
          !(err instanceof DOMException && err.name === "AbortError")
        ) {
          setDealTypeSummaryError(
            err instanceof Error
              ? err.message
              : "Unable to load deal type summary.",
          );
        }
      } finally {
        if (!cancelled) {
          setDealTypeSummaryLoading(false);
        }
      }
    };

    fetchDealTypeSummary();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [dealTypeSummaryLoaded, dealTypesTabOpen]);

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
      denominatorLabel: "processed" | "sections",
    ) =>
      value === null ? `—% of ${denominatorLabel}` : `${value.toFixed(1)}% of ${denominatorLabel}`;
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
        label: "Processed",
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
        detail: formatPctDetail(statusSummaryMetadataCoveragePct, "processed"),
      },
      {
        key: "taxonomy-coverage",
        label: "Taxonomy coverage",
        value: statusSummaryTaxonomyCoveredSections ?? "—",
        detail: formatPctDetail(statusSummaryTaxonomyCoveragePct, "sections"),
      },
      {
        key: "latest",
        label: "Latest ingested",
        value: statusSummaryLatestFilingDate
          ? formatDateValue(statusSummaryLatestFilingDate)
          : "—",
        detail: "Max filing date",
      },
    ] as const;
  }, [
    stagedTotals,
    statusSummaryLatestFilingDate,
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
  const stagedYearTicks = useMemo(() => {
    if (!stagedYearRange) return undefined;
    const start = 2000;
    const end = stagedYearRange.maxYear;
    const availableYears = new Set(stagedChartData.map((row) => row.year));
    const ticks: number[] = [];
    for (let year = start; year <= end; year += 5) {
      if (year >= stagedYearRange.minYear && availableYears.has(year)) {
        ticks.push(year);
      }
    }
    return ticks.length ? ticks : undefined;
  }, [stagedChartData, stagedYearRange]);
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
        (index % 2 === 0 ? "hsl(226 80% 58%)" : "hsl(191 82% 45%)"),
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
  const dealTypeYearTicks = useMemo(() => {
    if (!dealTypeYearRange) return undefined;
    const start = 2000;
    const end = dealTypeYearRange.maxYear;
    const availableYears = new Set(dealTypeChartData.map((row) => row.year));
    const ticks: number[] = [];
    for (let year = start; year <= end; year += 5) {
      if (year >= dealTypeYearRange.minYear && availableYears.has(year)) {
        ticks.push(year);
      }
    }
    return ticks.length ? ticks : undefined;
  }, [dealTypeChartData, dealTypeYearRange]);
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
      fallback={
        <div className="rounded-lg border border-border/60 bg-muted/20 p-4">
          <Skeleton className="h-[240px] w-full sm:h-[300px] lg:h-[340px]" />
        </div>
      }
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
    <div
      className={cn(
        "rounded-lg border border-border/60 bg-muted/20 p-3",
        className,
      )}
    >
      <div className="grid gap-2 sm:hidden">
        {stagedSummaryMetrics.map((metric) => (
          <dl
            key={metric.key}
            className="rounded-md border border-border/60 bg-background/70 p-3"
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
      <div className="hidden gap-0 overflow-hidden rounded-md border border-border/60 bg-background/70 sm:grid">
        <div className="grid min-w-[520px] grid-cols-4">
          {stagedPrimarySummaryMetrics.map((metric, index) => (
            <div
              key={metric.key}
              className={cn(
                "p-4",
                index > 0 && "border-l border-border/60",
              )}
            >
              <div className="text-xs font-semibold text-muted-foreground">
                {renderMetricLabel(metric.label, metric.key)}
              </div>
              <div className="mt-2 text-base font-semibold text-foreground">
                {typeof metric.value === "number"
                  ? metric.value.toLocaleString("en-US")
                  : metric.value}
              </div>
              <div className="text-xs text-muted-foreground">{metric.detail}</div>
            </div>
          ))}
        </div>
        <div className="grid min-w-[520px] grid-cols-3 border-t border-border/60">
          {stagedSecondarySummaryMetrics.map((metric, index) => (
            <div
              key={metric.key}
              className={cn(
                "p-4",
                index > 0 && "border-l border-border/60",
              )}
            >
              <div className="text-xs font-semibold text-muted-foreground">
                {renderMetricLabel(metric.label, metric.key)}
              </div>
              <div className="mt-2 text-base font-semibold text-foreground">
                {typeof metric.value === "number"
                  ? metric.value.toLocaleString("en-US")
                  : metric.value}
              </div>
              <div className="text-xs text-muted-foreground">{metric.detail}</div>
            </div>
          ))}
        </div>
      </div>
      <table className="sr-only">
        <caption>
          Summary totals for staged, awaiting validation, processed, and not
          paginated agreements, plus metadata coverage, taxonomy coverage, and
          latest ingested filing date.
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
      </table>
    </div>
  );

  const renderDealTypeSummaryTable = (className?: string) => (
    <div
      className={cn(
        "rounded-lg border border-border/60 bg-muted/20 p-3",
        className,
      )}
    >
      <div className="grid gap-2 sm:hidden">
        {dealTypeTotals.metrics.map((metric) => (
          <dl
            key={metric.key}
            className="rounded-md border border-border/60 bg-background/70 p-3"
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
      <div className="hidden overflow-x-auto sm:block">
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
      fallback={
        <div className="rounded-lg border border-border/60 bg-muted/20 p-4">
          <Skeleton className="mb-3 h-8 w-40" />
          <Skeleton className="h-[240px] w-full sm:h-[300px] lg:h-[340px]" />
        </div>
      }
    >
      <DealTypesChart
        className={className}
        data={dealTypeChartData}
        describedBy={dealTypeChartDescriptionId}
        isMobile={isMobile}
        mode={dealTypeChartMode}
        onModeChange={setDealTypeChartMode}
        series={dealTypeSeries}
        showSourceSplit={showDealTypeSourceSplit}
        tableId={dealTypeChartTableId}
        yearRange={dealTypeYearRange}
        yearTicks={dealTypeYearTicks}
      />
    </Suspense>
  );

  const renderStageFunnelTable = (className?: string) => (
    <div
      className={cn(
        "rounded-lg border border-border/60 bg-muted/20 p-3",
        className,
      )}
    >
      <div className="grid gap-2 sm:hidden">
        {stageSummaryRows.map((row) => (
          <dl
            key={row.key}
            className="rounded-md border border-border/60 bg-background/70 p-3"
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
      <div className="hidden overflow-x-auto sm:block">
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

  return (
    <Tabs
      value={overviewTab}
      onValueChange={(value) => {
        if (value === "processing-status" || value === "deal-types") {
          setOverviewTab(value);
        }
      }}
      className="space-y-3"
    >
      <TabsList className="grid h-auto w-full grid-cols-2">
        <TabsTrigger value="processing-status">Processing status</TabsTrigger>
        <TabsTrigger value="deal-types">Deal types</TabsTrigger>
      </TabsList>
      <TabsContent value="processing-status" className="mt-0 space-y-3">
        <p
          id={stagedChartDescriptionId}
          className="text-base text-muted-foreground"
        >
          Staged agreements have not yet gone through our pipelines. Agreements
          that are awaiting validation have made it through at least one step of
          the pipeline but tripped one of our validations and are awaiting
          manual review. Agreements that are staged or awaiting validation do
          not show up in the <span className="font-mono text-sm text-foreground">/v1/sections/*</span>{" "}or
          <span className="font-mono text-sm text-foreground">/v1/agreements/*</span>{" "}
          routes. The dashed vertical divider marks the 2020/2021 boundary:
          from 2000 through 2020, we use data from the DMA Corpus; beginning
          2021, we source data ourselves from EDGAR.
        </p>
        {statusSummaryLoading ? (
          <div className="rounded-lg border border-border/60 bg-muted/20 p-4">
            <Skeleton className="h-4 w-40" />
            <Skeleton className="mt-4 h-[220px] w-full sm:h-[260px]" />
          </div>
        ) : statusSummaryError ? (
          <div
            className="rounded-lg border border-border/60 bg-muted/10 p-4 text-sm text-muted-foreground"
            role="alert"
          >
            {statusSummaryError}
          </div>
        ) : stagedChartData.length === 0 ? (
          <div
            className="rounded-lg border border-border/60 bg-muted/10 p-4 text-sm text-muted-foreground"
            role="status"
          >
            No staged agreement data available yet.
          </div>
        ) : (
          <div className="space-y-3">
            {renderStagedSummaryTable()}
            {isMobile ? (
              <>
                <Button
                  type="button"
                  variant="outline"
                  className="w-full"
                  onClick={() => setIsProcessingChartModalOpen(true)}
                  aria-haspopup="dialog"
                  aria-describedby={stagedChartDescriptionId}
                >
                  Open processing status chart
                </Button>
                <MobileChartModal
                  open={isProcessingChartModalOpen}
                  onOpenChange={setIsProcessingChartModalOpen}
                  title="Processing status"
                  description="Shows annual counts for processed, awaiting validation, staged, and not paginated agreements."
                >
                  <div className="mx-auto w-full max-w-[980px]">
                    <h2 className="mb-2 text-base font-semibold">
                      Processing status
                    </h2>
                    {renderStagedChart("border-0 bg-background p-0")}
                  </div>
                </MobileChartModal>
              </>
            ) : (
              renderStagedChart()
            )}
            {renderStageFunnelTable()}
            <table id={stagedChartTableId} className="sr-only">
              <caption>
                Processed, awaiting validation, staged, and not paginated
                agreements by filing year
              </caption>
              <thead>
                <tr>
                  <th scope="col">Year</th>
                  <th scope="col">Processed</th>
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
            </table>
          </div>
        )}
      </TabsContent>
      <TabsContent value="deal-types" className="mt-0 space-y-3">
        <p
          id={dealTypeChartDescriptionId}
          className="text-base text-muted-foreground"
        >
          Deal type counts are precomputed from processed agreements and grouped
          by filing year. The dashed vertical divider marks the 2020/2021
          boundary: from 2000 through 2020, we use data from the DMA Corpus;
          beginning 2021, we source data ourselves from EDGAR.
        </p>
        {dealTypeSummaryLoading || (!dealTypeSummaryLoaded && !dealTypeSummaryError) ? (
          <div className="rounded-lg border border-border/60 bg-muted/20 p-4">
            <Skeleton className="h-4 w-40" />
            <Skeleton className="mt-4 h-[220px] w-full sm:h-[260px]" />
          </div>
        ) : dealTypeSummaryError ? (
          <div
            className="rounded-lg border border-border/60 bg-muted/10 p-4 text-sm text-muted-foreground"
            role="alert"
          >
            {dealTypeSummaryError}
          </div>
        ) : dealTypeChartData.length === 0 || dealTypeSeries.length === 0 ? (
          <div
            className="rounded-lg border border-border/60 bg-muted/10 p-4 text-sm text-muted-foreground"
            role="status"
          >
            No deal type data available yet.
          </div>
        ) : (
          <div className="space-y-3">
            {renderDealTypeSummaryTable()}
            {isMobile ? (
              <>
                <Button
                  type="button"
                  variant="outline"
                  className="w-full"
                  onClick={() => setIsDealTypesChartModalOpen(true)}
                  aria-haspopup="dialog"
                  aria-describedby={dealTypeChartDescriptionId}
                >
                  Open deal types chart
                </Button>
                <MobileChartModal
                  open={isDealTypesChartModalOpen}
                  onOpenChange={setIsDealTypesChartModalOpen}
                  title="Deal types"
                  description="Shows annual deal type counts for processed agreements."
                >
                  <div className="mx-auto w-full max-w-[980px]">
                    <h2 className="mb-2 text-base font-semibold">
                      Deal types
                    </h2>
                    {renderDealTypeChart("border-0 bg-background p-0")}
                  </div>
                </MobileChartModal>
              </>
            ) : (
              renderDealTypeChart()
            )}
            <table id={dealTypeChartTableId} className="sr-only">
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
            </table>
          </div>
        )}
      </TabsContent>
    </Tabs>
  );
}
