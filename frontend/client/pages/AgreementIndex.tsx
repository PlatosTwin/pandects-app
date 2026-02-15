import { useEffect, useId, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  BadgeCheck,
  BookOpen,
  FileText,
  Layers,
  Search,
  X,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts";
import { PageShell } from "@/components/PageShell";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Pagination,
  PaginationContent,
  PaginationEllipsis,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from "@/components/ui/pagination";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  type ChartConfig,
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";
import { cn } from "@/lib/utils";
import { AgreementModal } from "@/components/AgreementModal";
import { formatDateValue, formatEnumValue } from "@/lib/format-utils";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { AdaptiveTooltip } from "@/components/ui/adaptive-tooltip";
import { useIsMobile } from "@/hooks/use-mobile";

const STAGE_TOOLTIP_COPY = {
  "0_staging":
    "Agreements awaiting pre-processing (splitting into pages, classifying, etc.).",
  "1_pre_processing":
    "Agreements that have been pre-processed and are awaiting tagging via the NER model.",
  "2_tagging":
    "Agreements that have been tagged and are awaiting compilation into XML.",
  "3_xml":
    "Agreements that have been compiled into XML and are awaiting an upsert to the sections table.",
} as const;

type AgreementIndexRow = {
  agreement_uuid: string;
  year: string | null;
  target: string | null;
  acquirer: string | null;
  consideration_type: string | null;
  total_consideration: string | number | null;
  target_industry: string | null;
  acquirer_industry: string | null;
  verified: boolean;
};

type AgreementIndexResponse = {
  results: AgreementIndexRow[];
  page: number;
  page_size: number;
  total_count: number;
  total_pages: number;
  has_next: boolean;
  has_prev: boolean;
};

type AgreementIndexSummary = {
  agreements: number;
  sections: number;
  pages: number;
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

type SortColumn = "year" | "target" | "acquirer";
type SortDirection = "asc" | "desc";
type OverviewTab = "processing-status" | "deal-types";
type DealTypeChartMode = "count" | "percent";

const formatValue = (value: string | number | null | undefined) => {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value === "number") return value.toLocaleString("en-US");
  return value;
};

const formatYear = (value: string | number | null | undefined) => {
  if (value === null || value === undefined || value === "") return "—";
  return typeof value === "number" ? String(value) : value;
};

const summaryCards = [
  {
    key: "agreements",
    label: "Agreements",
    icon: FileText,
    description: "Total agreements ingested",
  },
  {
    key: "sections",
    label: "Sections",
    icon: Layers,
    description: "Clause sections indexed",
  },
  {
    key: "pages",
    label: "Pages",
    icon: BookOpen,
    description: "Pages parsed across filings",
  },
] as const;

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

const normalizeDealType = (dealType: string | null | undefined) => {
  if (!dealType) return "unknown";
  const normalized = dealType.trim();
  return normalized.length ? normalized : "unknown";
};

const formatDealTypeLabel = (dealType: string) =>
  dealType === "unknown" ? "Unclassified" : formatEnumValue(dealType);

const dealTypeSeriesKey = (dealType: string) =>
  `dealType_${dealType.replace(/[^a-z0-9]+/gi, "_")}`;
const PERCENT_AXIS_TICKS = [0, 20, 40, 60, 80, 100];

type MobileChartModalProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  describedBy?: string;
  children: React.ReactNode;
};

function MobileChartModal({
  open,
  onOpenChange,
  title,
  describedBy,
  children,
}: MobileChartModalProps) {
  const closeButtonRef = useRef<HTMLButtonElement | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!open || !mounted) return;

    const activeElement = document.activeElement as HTMLElement | null;
    const scrollY = window.scrollY;
    const body = document.body;
    const root = document.getElementById("root");
    const currentModalCount = Number(body.dataset.mobileChartModalCount ?? "0");
    const prev = {
      position: body.style.position,
      top: body.style.top,
      left: body.style.left,
      right: body.style.right,
      width: body.style.width,
      overflow: body.style.overflow,
      touchAction: body.style.touchAction,
    };

    body.style.position = "fixed";
    body.style.top = `-${scrollY}px`;
    body.style.left = "0";
    body.style.right = "0";
    body.style.width = "100%";
    body.style.overflow = "hidden";
    body.style.touchAction = "none";
    body.dataset.mobileChartModalCount = String(currentModalCount + 1);
    if (root) {
      root.setAttribute("aria-hidden", "true");
      root.setAttribute("inert", "");
    }

    const frame = window.requestAnimationFrame(() => {
      closeButtonRef.current?.focus();
    });

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      onOpenChange(false);
    };

    document.addEventListener("keydown", onKeyDown);

    return () => {
      window.cancelAnimationFrame(frame);
      document.removeEventListener("keydown", onKeyDown);
      body.style.position = prev.position;
      body.style.top = prev.top;
      body.style.left = prev.left;
      body.style.right = prev.right;
      body.style.width = prev.width;
      body.style.overflow = prev.overflow;
      body.style.touchAction = prev.touchAction;
      const nextModalCount = Math.max(
        0,
        Number(body.dataset.mobileChartModalCount ?? "1") - 1,
      );
      body.dataset.mobileChartModalCount = String(nextModalCount);
      if (nextModalCount === 0 && root) {
        root.removeAttribute("aria-hidden");
        root.removeAttribute("inert");
      }
      window.scrollTo(0, scrollY);
      activeElement?.focus?.();
    };
  }, [open, mounted, onOpenChange]);

  if (!mounted || !open) return null;

  return createPortal(
    <div className="fixed inset-0 z-50">
      <div
        className="absolute inset-0 bg-black/80"
        onClick={() => onOpenChange(false)}
        aria-hidden="true"
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        aria-describedby={describedBy}
        className="absolute inset-0 bg-background"
      >
        <button
          ref={closeButtonRef}
          type="button"
          onClick={() => onOpenChange(false)}
          className="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
          aria-label="Close"
        >
          <X className="h-4 w-4" aria-hidden="true" />
        </button>
        <div className="flex h-full w-full items-center justify-center p-4">
          <div className="w-full">{children}</div>
        </div>
      </div>
    </div>,
    document.body,
  );
}

export default function AgreementIndex() {
  const isMobile = useIsMobile();
  const [isProcessingChartModalOpen, setIsProcessingChartModalOpen] =
    useState(false);
  const [isDealTypesChartModalOpen, setIsDealTypesChartModalOpen] =
    useState(false);
  const [overviewTab, setOverviewTab] =
    useState<OverviewTab>("processing-status");
  const [dealTypeChartMode, setDealTypeChartMode] =
    useState<DealTypeChartMode>("count");
  const [summary, setSummary] = useState<AgreementIndexSummary | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(true);
  const [summaryError, setSummaryError] = useState<string | null>(null);
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
  const [dealTypeSummary, setDealTypeSummary] = useState<
    AgreementDealTypeYearRow[]
  >([]);
  const [dealTypeSummaryLoading, setDealTypeSummaryLoading] = useState(false);
  const [dealTypeSummaryLoaded, setDealTypeSummaryLoaded] = useState(false);
  const [dealTypeSummaryError, setDealTypeSummaryError] = useState<
    string | null
  >(null);
  const [statusAccordionValue, setStatusAccordionValue] = useState<
    string | undefined
  >(undefined);

  const [agreements, setAgreements] = useState<AgreementIndexRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [page_size] = useState(25);
  const [total_count, setTotalCount] = useState(0);
  const [total_pages, setTotalPages] = useState(1);
  const [sort_by, setSortBy] = useState<SortColumn>("year");
  const [sort_dir, setSortDir] = useState<SortDirection>("desc");
  const [filterInput, setFilterInput] = useState("");
  const [filterQuery, setFilterQuery] = useState("");
  const [selectedAgreement, setSelectedAgreement] = useState<{
    agreement_uuid: string;
    year: string;
    target: string;
    acquirer: string;
  } | null>(null);
  const stagedChartDescriptionId = useId();
  const stagedChartTableId = useId();
  const dealTypeChartDescriptionId = useId();
  const dealTypeChartTableId = useId();

  useEffect(() => {
    let cancelled = false;
    const fetchSummary = async () => {
      try {
        setSummaryLoading(true);
        setSummaryError(null);
        const res = await authFetch(apiUrl("v1/agreements-summary"));
        if (!res.ok) {
          throw new Error(`Summary request failed (${res.status})`);
        }
        const data = (await res.json()) as AgreementIndexSummary;
        if (!cancelled) {
          setSummary(data);
        }
      } catch (err) {
        if (!cancelled) {
          setSummaryError(
            err instanceof Error
              ? err.message
              : "Unable to load agreement summary.",
          );
        }
      } finally {
        if (!cancelled) {
          setSummaryLoading(false);
        }
      }
    };

    fetchSummary();
    return () => {
      cancelled = true;
    };
  }, []);

  const statusAccordionOpen = statusAccordionValue === "staged";
  const dealTypesTabOpen = overviewTab === "deal-types";

  useEffect(() => {
    if (!statusAccordionOpen || statusSummaryLoaded || statusSummaryLoading)
      return;
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
  }, [statusAccordionOpen, statusSummaryLoaded]);

  useEffect(() => {
    if (
      !statusAccordionOpen ||
      !dealTypesTabOpen ||
      dealTypeSummaryLoaded ||
      dealTypeSummaryLoading
    ) {
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
  }, [statusAccordionOpen, dealTypesTabOpen, dealTypeSummaryLoaded]);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    const fetchAgreements = async () => {
      try {
        setLoading(true);
        setError(null);
        const params = new URLSearchParams();
        params.set("page", String(page));
        params.set("page_size", String(page_size));
        params.set("sort_by", sort_by);
        params.set("sort_dir", sort_dir);
        if (filterQuery.trim()) {
          params.set("query", filterQuery.trim());
        }

        const res = await authFetch(
          apiUrl(`v1/agreements-index?${params.toString()}`),
          { signal: controller.signal },
        );
        if (!res.ok) {
          throw new Error(`Agreement index request failed (${res.status})`);
        }
        const data = (await res.json()) as AgreementIndexResponse;
        if (!cancelled) {
          setAgreements(data.results);
          setTotalCount(data.total_count);
          setTotalPages(Math.max(1, data.total_pages));
        }
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof Error
              ? err.message
              : "Unable to load agreement index.",
          );
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    fetchAgreements();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [page, page_size, sort_by, sort_dir, filterQuery]);

  useEffect(() => {
    const handle = window.setTimeout(() => {
      setFilterQuery(filterInput);
      setPage(1);
    }, 300);
    return () => window.clearTimeout(handle);
  }, [filterInput]);

  const handleSort = (column: SortColumn) => {
    if (column === sort_by) {
      setSortDir((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortBy(column);
      setSortDir("asc");
    }
    setPage(1);
  };

  const pageItems = useMemo(() => {
    if (total_pages <= 7) {
      return Array.from({ length: total_pages }, (_, i) => i + 1);
    }

    const items: Array<number | "ellipsis"> = [];
    items.push(1);

    const start = Math.max(2, page - 1);
    const end = Math.min(total_pages - 1, page + 1);

    if (start > 2) items.push("ellipsis");
    for (let i = start; i <= end; i += 1) items.push(i);
    if (end < total_pages - 1) items.push("ellipsis");

    items.push(total_pages);
    return items;
  }, [page, total_pages]);

  const filteredLabel = filterQuery.trim()
    ? `Filtered by "${filterQuery.trim()}"`
    : "";
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
      if (!Number.isFinite(row.year)) return;
      const year = Number(row.year);
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
    const pct = (value: number) =>
      total > 0 ? Math.round((value / total) * 1000) / 10 : 0;
    return [
      {
        key: "staged",
        label: "Staged",
        value: stagedTotals.staged,
        pct: pct(stagedTotals.staged),
      },
      {
        key: "awaiting",
        label: "Awaiting validation",
        value: stagedTotals.awaiting,
        pct: pct(stagedTotals.awaiting),
      },
      {
        key: "processed",
        label: "Processed",
        value: stagedTotals.processed,
        pct: pct(stagedTotals.processed),
      },
      {
        key: "not-paginated",
        label: "Not paginated",
        value: stagedTotals.notPaginated,
        pct: pct(stagedTotals.notPaginated),
      },
      {
        key: "latest",
        label: "Latest ingested",
        value: statusSummaryLatestFilingDate
          ? formatDateValue(statusSummaryLatestFilingDate)
          : "—",
        pct: null,
      },
    ] as const;
  }, [stagedTotals, statusSummaryLatestFilingDate]);
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
  }, [stagedYearRange, stagedChartData]);
  const dealTypeSeries = useMemo(() => {
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
      if (!Number.isFinite(row.year)) return;
      const year = Number(row.year);
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
  }, [dealTypeSummary, dealTypeSeries]);
  const dealTypeChartDataByYear = useMemo(
    () => new Map(dealTypeChartData.map((row) => [row.year, row])),
    [dealTypeChartData],
  );
  const dealTypeChartDisplayData = useMemo(() => {
    if (dealTypeChartMode === "count") {
      return dealTypeChartData;
    }
    return dealTypeChartData.map((row) => {
      const total = dealTypeSeries.reduce(
        (sum, series) => sum + Number(row[series.key] ?? 0),
        0,
      );
      const pctRow: { year: number } & Record<string, number> = {
        year: row.year,
      };
      dealTypeSeries.forEach((series) => {
        const value = Number(row[series.key] ?? 0);
        pctRow[series.key] =
          total > 0 ? Math.round((value / total) * 1000) / 10 : 0;
      });
      return pctRow;
    });
  }, [dealTypeChartData, dealTypeSeries, dealTypeChartMode]);
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
  }, [dealTypeYearRange, dealTypeChartData]);
  const dealTypeChartConfig = useMemo<ChartConfig>(
    () =>
      dealTypeSeries.reduce<ChartConfig>((acc, series) => {
        acc[series.key] = {
          label: series.label,
          color: series.color,
        };
        return acc;
      }, {}),
    [dealTypeSeries],
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

  const renderStageLabel = (row: (typeof stageSummaryRows)[number]) => (
    <span className="inline-flex items-center gap-1">
      <span>{row.label}</span>
      <AdaptiveTooltip
        trigger={
          <button
            type="button"
            aria-label={`${row.label} stage details`}
            className="tooltip-help-trigger-compact"
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
    <div
      className={cn(
        "rounded-lg border border-border/60 bg-muted/20 p-3",
        className,
      )}
    >
      <ChartContainer
        className="h-[240px] w-full min-w-0 aspect-auto sm:h-[300px] lg:h-[340px]"
        config={{
          processed: {
            label: "Processed",
            color: "hsl(142 71% 45%)",
          },
          staged: {
            label: "Staged",
            color: "hsl(38 92% 55%)",
          },
          awaiting: {
            label: "Awaiting validation",
            color: "hsl(0 84% 60%)",
          },
          notPaginated: {
            label: "Not paginated",
            color: "hsl(220 9% 60%)",
          },
        }}
        role="img"
        aria-label="Stacked bar chart showing processed, staged, awaiting validation, and not paginated agreements by filing year."
        aria-describedby={`${stagedChartDescriptionId} ${stagedChartTableId}`}
      >
        <BarChart
          data={stagedChartData}
          margin={{ top: 6, right: 24, left: 8, bottom: 0 }}
        >
          <CartesianGrid vertical={false} />
          <XAxis
            dataKey="year"
            type="number"
            allowDecimals={false}
            domain={
              stagedYearRange
                ? [stagedYearRange.minYear, stagedYearRange.maxYear]
                : ["dataMin", "dataMax"]
            }
            padding={{ left: 20, right: 20 }}
            tickFormatter={(value) => String(value)}
            tickMargin={6}
            minTickGap={16}
            interval="preserveStartEnd"
            ticks={stagedYearTicks}
          />
          <YAxis allowDecimals={false} tickMargin={6} width={32} />
          <ChartTooltip
            content={
              <ChartTooltipContent
                indicator="dashed"
                labelFormatter={(_, payload) => {
                  const year = payload?.[0]?.payload?.year;
                  return `Filing year ${year ?? "—"}`;
                }}
                formatter={(value, name, item) => {
                  const indicatorColor = item?.payload?.fill || item?.color;
                  const payload = item?.payload as
                    | {
                        processed?: number;
                        staged?: number;
                        awaiting?: number;
                        notPaginated?: number;
                      }
                    | undefined;
                  const processed = Number(payload?.processed ?? 0);
                  const staged = Number(payload?.staged ?? 0);
                  const awaiting = Number(payload?.awaiting ?? 0);
                  const notPaginated = Number(payload?.notPaginated ?? 0);
                  const total = processed + staged + awaiting + notPaginated;
                  const processedPct =
                    total > 0 ? Math.round((processed / total) * 1000) / 10 : 0;
                  const getPct = (count: number) =>
                    total > 0 ? Math.round((count / total) * 1000) / 10 : 0;
                  const colorBlock = (
                    <span
                      className="inline-block h-2.5 w-2.5 shrink-0 rounded-[2px]"
                      style={
                        indicatorColor
                          ? { backgroundColor: indicatorColor }
                          : undefined
                      }
                      aria-hidden="true"
                    />
                  );

                  const countValue = Number(value);
                  const pct =
                    name === "Processed"
                      ? processedPct
                      : name === "Staged"
                        ? getPct(staged)
                        : name === "Awaiting validation"
                          ? getPct(awaiting)
                          : getPct(notPaginated);

                  return (
                    <div className="grid grid-cols-[auto_minmax(0,4.5rem)_minmax(0,1fr)] items-center gap-x-3">
                      {colorBlock}
                      <span className="text-left font-mono font-medium tabular-nums text-foreground">
                        {countValue.toLocaleString()}
                      </span>
                      <span className="text-right font-mono text-xs tabular-nums text-muted-foreground">
                        {pct.toFixed(1)}%
                      </span>
                    </div>
                  );
                }}
              />
            }
          />
          <ChartLegend content={<ChartLegendContent />} />
          <Bar
            dataKey="processed"
            stackId="agreements"
            fill="var(--color-processed)"
            name="Processed"
          />
          <Bar
            dataKey="staged"
            stackId="agreements"
            fill="var(--color-staged)"
            name="Staged"
          />
          <Bar
            dataKey="awaiting"
            stackId="agreements"
            fill="var(--color-awaiting)"
            name="Awaiting validation"
          />
          <Bar
            dataKey="notPaginated"
            stackId="agreements"
            fill="var(--color-notPaginated)"
            name="Not paginated"
          />
          {showSourceSplit ? (
            <ReferenceLine
              x={2020.5}
              stroke="hsl(var(--foreground))"
              strokeWidth={1.5}
              strokeDasharray="4 4"
              isFront
              label={(props) => {
                if (!props.viewBox) return null;
                const text = isMobile ? "2020/21" : "2020/2021 split";
                const paddingX = isMobile ? 6 : 4;
                const rectHeight = isMobile ? 16 : 14;
                const charWidth = isMobile ? 6.1 : 5.4;
                const textWidth = text.length * charWidth;
                const rectWidth = textWidth + paddingX * 2;
                const rectX = props.viewBox.x - rectWidth - 8;
                const rectY = props.viewBox.y + 8;
                const textX = rectX + rectWidth / 2;
                const textY = rectY + rectHeight / 2 + 0.5;
                return (
                  <g pointerEvents="none">
                    <rect
                      x={rectX}
                      y={rectY}
                      width={rectWidth}
                      height={rectHeight}
                      rx={4}
                      fill="hsl(var(--background))"
                      stroke="hsl(var(--border))"
                    />
                    <text
                      x={textX}
                      y={textY}
                      fill="hsl(var(--muted-foreground))"
                      fontSize={10}
                      textAnchor="middle"
                      dominantBaseline="middle"
                    >
                      {text}
                    </text>
                  </g>
                );
              }}
            />
          ) : null}
        </BarChart>
      </ChartContainer>
    </div>
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
            <dt className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              {metric.label}
            </dt>
            <dd className="mt-1 text-base font-semibold text-foreground">
              {typeof metric.value === "number"
                ? metric.value.toLocaleString("en-US")
                : metric.value}
            </dd>
            <dd className="text-xs text-muted-foreground">
              {metric.pct !== null
                ? `${metric.pct.toFixed(1)}% of total`
                : "Max filing date"}
            </dd>
          </dl>
        ))}
      </div>
      <div className="hidden overflow-x-auto sm:block">
        <Table className="min-w-[520px]">
          <caption className="sr-only">
            Summary totals for staged, awaiting validation, and processed
            agreements, plus the latest ingested filing date.
          </caption>
          <TableHeader>
            <TableRow>
              {stagedSummaryMetrics.map((metric) => (
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
              {stagedSummaryMetrics.map((metric) => (
                <TableCell key={metric.key} className="align-top">
                  <div className="text-base font-semibold text-foreground">
                    {typeof metric.value === "number"
                      ? metric.value.toLocaleString("en-US")
                      : metric.value}
                  </div>
                  {metric.pct !== null ? (
                    <div className="text-xs font-mono tabular-nums text-muted-foreground">
                      {metric.pct.toFixed(1)}% of total
                    </div>
                  ) : (
                    <div className="text-xs text-muted-foreground">
                      Max filing date
                    </div>
                  )}
                </TableCell>
              ))}
            </TableRow>
          </TableBody>
        </Table>
      </div>
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
    <div
      className={cn(
        "rounded-lg border border-border/60 bg-muted/20 p-3",
        className,
      )}
    >
      <div className="mb-3 flex items-center justify-end gap-2">
        <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          Chart mode
        </span>
        <ToggleGroup
          type="single"
          value={dealTypeChartMode}
          onValueChange={(value) => {
            if (value === "count" || value === "percent") {
              setDealTypeChartMode(value);
            }
          }}
          variant="outline"
          size="xs"
          aria-label="Deal type chart mode"
          className="justify-start"
        >
          <ToggleGroupItem
            value="count"
            aria-label="Stacked counts"
            className="text-muted-foreground data-[state=on]:text-foreground"
          >
            Counts
          </ToggleGroupItem>
          <ToggleGroupItem
            value="percent"
            aria-label="100 percent stacked"
            className="text-muted-foreground data-[state=on]:text-foreground"
          >
            100%
          </ToggleGroupItem>
        </ToggleGroup>
      </div>
      <ChartContainer
        className="h-[240px] w-full min-w-0 aspect-auto sm:h-[300px] lg:h-[340px]"
        config={dealTypeChartConfig}
        role="img"
        aria-label={
          dealTypeChartMode === "percent"
            ? "100 percent stacked bar chart showing deal type share by filing year."
            : "Stacked bar chart showing deal type counts by filing year."
        }
        aria-describedby={`${dealTypeChartDescriptionId} ${dealTypeChartTableId}`}
      >
        <BarChart
          data={dealTypeChartDisplayData}
          margin={{ top: 6, right: 24, left: 8, bottom: 0 }}
        >
          <CartesianGrid vertical={false} />
          <XAxis
            dataKey="year"
            type="number"
            allowDecimals={false}
            domain={
              dealTypeYearRange
                ? [dealTypeYearRange.minYear, dealTypeYearRange.maxYear]
                : ["dataMin", "dataMax"]
            }
            padding={{ left: 20, right: 20 }}
            tickFormatter={(value) => String(value)}
            tickMargin={6}
            minTickGap={16}
            interval="preserveStartEnd"
            ticks={dealTypeYearTicks}
          />
          <YAxis
            tickMargin={6}
            width={dealTypeChartMode === "percent" ? 44 : 32}
            allowDecimals={dealTypeChartMode !== "percent"}
            domain={dealTypeChartMode === "percent" ? [0, 100] : undefined}
            ticks={
              dealTypeChartMode === "percent" ? PERCENT_AXIS_TICKS : undefined
            }
            tickFormatter={(value) => {
              const numericValue = Number(value);
              if (!Number.isFinite(numericValue)) return "";
              if (dealTypeChartMode === "percent") {
                return `${Math.round(numericValue)}%`;
              }
              return String(numericValue);
            }}
          />
          <ChartTooltip
            content={
              <ChartTooltipContent
                indicator="dashed"
                labelFormatter={(_, payload) => {
                  const year = payload?.[0]?.payload?.year;
                  return `Filing year ${year ?? "—"}`;
                }}
                formatter={(value, _name, item) => {
                  const indicatorColor = item?.payload?.fill || item?.color;
                  const payload = item?.payload as
                    | { year?: number }
                    | undefined;
                  const year = Number(payload?.year ?? NaN);
                  const dataKey =
                    typeof item.dataKey === "string" ? item.dataKey : "";
                  const rawRow = Number.isFinite(year)
                    ? dealTypeChartDataByYear.get(year)
                    : undefined;
                  const rawCount = Number(
                    rawRow && dataKey ? (rawRow[dataKey] ?? 0) : 0,
                  );
                  const rawTotal = dealTypeSeries.reduce(
                    (sum, series) => sum + Number(rawRow?.[series.key] ?? 0),
                    0,
                  );
                  const rawPct =
                    rawTotal > 0
                      ? Math.round((rawCount / rawTotal) * 1000) / 10
                      : 0;
                  const valueNumber = Number(value);
                  const valueLabel =
                    dealTypeChartMode === "percent"
                      ? `${valueNumber.toFixed(1)}%`
                      : rawCount.toLocaleString();
                  const metaLabel =
                    dealTypeChartMode === "percent"
                      ? rawCount.toLocaleString()
                      : `${rawPct.toFixed(1)}%`;
                  return (
                    <div className="grid grid-cols-[auto_3rem_minmax(0,1fr)] items-center gap-x-3">
                      <span
                        className="inline-block h-2.5 w-2.5 shrink-0 rounded-[2px]"
                        style={
                          indicatorColor
                            ? { backgroundColor: indicatorColor }
                            : undefined
                        }
                        aria-hidden="true"
                      />
                      <span className="text-left font-mono font-medium tabular-nums text-foreground">
                        {valueLabel}
                      </span>
                      <span className="text-right font-mono text-xs tabular-nums text-muted-foreground">
                        {metaLabel}
                      </span>
                    </div>
                  );
                }}
              />
            }
          />
          <ChartLegend content={<ChartLegendContent />} />
          {dealTypeSeries.map((series) => (
            <Bar
              key={series.key}
              dataKey={series.key}
              stackId="deal-types"
              fill={`var(--color-${series.key})`}
              name={series.label}
            />
          ))}
          {showDealTypeSourceSplit ? (
            <ReferenceLine
              x={2020.5}
              stroke="hsl(var(--foreground))"
              strokeWidth={1.5}
              strokeDasharray="4 4"
              isFront
              label={(props) => {
                if (!props.viewBox) return null;
                const text = isMobile ? "2020/21" : "2020/2021 split";
                const paddingX = isMobile ? 6 : 4;
                const rectHeight = isMobile ? 16 : 14;
                const charWidth = isMobile ? 6.1 : 5.4;
                const textWidth = text.length * charWidth;
                const rectWidth = textWidth + paddingX * 2;
                const rectX = props.viewBox.x - rectWidth - 8;
                const rectY = props.viewBox.y + 8;
                const textX = rectX + rectWidth / 2;
                const textY = rectY + rectHeight / 2 + 0.5;
                return (
                  <g pointerEvents="none">
                    <rect
                      x={rectX}
                      y={rectY}
                      width={rectWidth}
                      height={rectHeight}
                      rx={4}
                      fill="hsl(var(--background))"
                      stroke="hsl(var(--border))"
                    />
                    <text
                      x={textX}
                      y={textY}
                      fill="hsl(var(--muted-foreground))"
                      fontSize={10}
                      textAnchor="middle"
                      dominantBaseline="middle"
                    >
                      {text}
                    </text>
                  </g>
                );
              }}
            />
          ) : null}
        </BarChart>
      </ChartContainer>
    </div>
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
    <PageShell size="xl" title="Agreement Index">
      <div
        className="mb-10 grid gap-4 md:grid-cols-3"
        aria-busy={summaryLoading}
        aria-live="polite"
      >
        {summaryCards.map((card) => {
          const Icon = card.icon;
          const value = summary?.[card.key] ?? null;
          return (
            <Card
              key={card.key}
              className="relative overflow-hidden border-border/60 bg-gradient-to-br from-background via-background to-muted/40 shadow-sm"
            >
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,_rgba(99,102,241,0.18),_transparent_55%)] opacity-70" />
              <CardContent className="relative flex items-center gap-4 p-6">
                <div className="flex h-11 w-11 items-center justify-center rounded-full bg-primary/10 text-primary shadow-sm">
                  <Icon className="h-5 w-5" aria-hidden="true" />
                </div>
                <div className="min-w-0">
                  <div className="text-xs uppercase tracking-wide text-muted-foreground">
                    {card.label}
                  </div>
                  <div className="text-2xl font-semibold text-foreground">
                    {summaryLoading ? (
                      <>
                        <Skeleton className="h-7 w-24" />
                        <span className="sr-only">Loading summary</span>
                      </>
                    ) : summaryError ? (
                      "—"
                    ) : (
                      (value ?? 0).toLocaleString("en-US")
                    )}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {card.description}
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      <Card className="mb-10 border-border/60 shadow-sm">
        <CardContent className="p-6">
          <Accordion
            type="single"
            collapsible
            value={statusAccordionValue}
            onValueChange={setStatusAccordionValue}
          >
            <AccordionItem value="staged" className="border-border/60">
              <AccordionTrigger
                headingLevel="h2"
                className="py-3 text-2xl font-semibold tracking-tight"
              >
                Agreement overview
              </AccordionTrigger>
              <AccordionContent className="pt-3">
                <Tabs
                  value={overviewTab}
                  onValueChange={(value) => {
                    if (
                      value === "processing-status" ||
                      value === "deal-types"
                    ) {
                      setOverviewTab(value);
                    }
                  }}
                  className="space-y-3"
                >
                  <TabsList className="grid h-auto w-full grid-cols-2">
                    <TabsTrigger value="processing-status">
                      Processing status
                    </TabsTrigger>
                    <TabsTrigger value="deal-types">Deal types</TabsTrigger>
                  </TabsList>
                  <TabsContent
                    value="processing-status"
                    className="mt-0 space-y-3"
                  >
                    <p
                      id={stagedChartDescriptionId}
                      className="text-base text-muted-foreground"
                    >
                      Staged agreements have not yet gone through our pipelines.
                      Agreements that are awaiting validation have made it
                      through at least one step of the pipeline but tripped one
                      of our validations and are awaiting manual review.
                      Agreements that are staged or awaiting validation do not
                      show up in the{" "}
                      <span className="font-mono text-sm text-foreground">
                        /v1/search
                      </span>
                      ,{" "}
                      <span className="font-mono text-sm text-foreground">
                        /v1/agreements
                      </span>
                      , or{" "}
                      <span className="font-mono text-sm text-foreground">
                        /v1/sections
                      </span>{" "}
                      routes. The dashed vertical divider marks the 2020/2021
                      boundary: from 2000 through 2020, we use data from the DMA
                      Corpus; beginning 2021, we source data ourselves from
                      EDGAR.
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
                              onClick={() =>
                                setIsProcessingChartModalOpen(true)
                              }
                              aria-haspopup="dialog"
                              aria-describedby={stagedChartDescriptionId}
                            >
                              Click to view on mobile
                            </Button>
                            <MobileChartModal
                              open={isProcessingChartModalOpen}
                              onOpenChange={setIsProcessingChartModalOpen}
                              title="Processing status"
                              describedBy={stagedChartDescriptionId}
                            >
                              <div className="mx-auto w-full max-w-[980px]">
                                <h2 className="mb-2 text-base font-semibold">
                                  Processing status
                                </h2>
                                {renderStagedChart(
                                  "border-0 bg-background p-0",
                                )}
                              </div>
                            </MobileChartModal>
                          </>
                        ) : (
                          renderStagedChart()
                        )}
                        {renderStageFunnelTable()}
                        <table id={stagedChartTableId} className="sr-only">
                          <caption>
                            Processed, awaiting validation, staged, and not
                            paginated agreements by filing year
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
                                <td>
                                  {row.notPaginated.toLocaleString("en-US")}
                                </td>
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
                              <td>
                                {stagedTotals.processed.toLocaleString("en-US")}
                              </td>
                              <td>
                                {stagedTotals.awaiting.toLocaleString("en-US")}
                              </td>
                              <td>
                                {stagedTotals.staged.toLocaleString("en-US")}
                              </td>
                              <td>
                                {stagedTotals.notPaginated.toLocaleString(
                                  "en-US",
                                )}
                              </td>
                              <td>
                                {stagedTotals.total.toLocaleString("en-US")}
                              </td>
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
                      Deal type counts are precomputed from processed agreements
                      and grouped by filing year. The dashed vertical divider
                      marks the 2020/2021 boundary: from 2000 through 2020, we
                      use data from the DMA Corpus; beginning 2021, we source
                      data ourselves from EDGAR.
                    </p>
                    {dealTypeSummaryLoading ||
                    (!dealTypeSummaryLoaded && !dealTypeSummaryError) ? (
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
                    ) : dealTypeChartData.length === 0 ||
                      dealTypeSeries.length === 0 ? (
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
                              Click to view on mobile
                            </Button>
                            <MobileChartModal
                              open={isDealTypesChartModalOpen}
                              onOpenChange={setIsDealTypesChartModalOpen}
                              title="Deal types"
                              describedBy={dealTypeChartDescriptionId}
                            >
                              <div className="mx-auto w-full max-w-[980px]">
                                <h2 className="mb-2 text-base font-semibold">
                                  Deal types
                                </h2>
                                {renderDealTypeChart(
                                  "border-0 bg-background p-0",
                                )}
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
                                <th
                                  key={`deal-type-head-${series.key}`}
                                  scope="col"
                                >
                                  {series.label}
                                </th>
                              ))}
                              <th scope="col">Total</th>
                            </tr>
                          </thead>
                          <tbody>
                            {dealTypeChartData.map((row) => {
                              const rowTotal = dealTypeSeries.reduce(
                                (sum, series) =>
                                  sum + Number(row[series.key] ?? 0),
                                0,
                              );
                              return (
                                <tr key={`deal-type-row-${row.year}`}>
                                  <th scope="row">{row.year}</th>
                                  {dealTypeSeries.map((series) => (
                                    <td
                                      key={`deal-type-cell-${row.year}-${series.key}`}
                                    >
                                      {Number(
                                        row[series.key] ?? 0,
                                      ).toLocaleString("en-US")}
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
                              <td>
                                {dealTypeTotals.total.toLocaleString("en-US")}
                              </td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    )}
                  </TabsContent>
                </Tabs>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </CardContent>
      </Card>

      <Card className="border-border/60 shadow-sm hover:shadow-md transition-shadow duration-200">
        <CardContent className="p-6">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <h2 className="text-2xl font-semibold tracking-tight text-foreground">
                Index of processed agreements
              </h2>
              {filteredLabel ? (
                <p className="text-sm text-muted-foreground" aria-live="polite">
                  {filteredLabel}
                </p>
              ) : null}
            </div>
            <div className="flex flex-1 flex-col gap-3 sm:flex-row sm:items-center sm:justify-end">
              <div className="relative w-full sm:max-w-xs">
                <Label htmlFor="agreement-filter" className="sr-only">
                  Filter agreements
                </Label>
                <Search
                  className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
                  aria-hidden="true"
                />
                <Input
                  id="agreement-filter"
                  value={filterInput}
                  onChange={(event) => {
                    setFilterInput(event.target.value);
                  }}
                  placeholder="Filter by year, target, or acquirer"
                  className="pl-9"
                />
              </div>
              <Badge variant="secondary" className="self-start sm:self-auto">
                {total_count.toLocaleString("en-US")} agreements
              </Badge>
            </div>
          </div>

          <div className="mt-6 hidden lg:block rounded-lg border border-border/60 bg-muted/20">
            <Table className="min-w-[900px]">
              <TableHeader className="sticky top-0 bg-muted/50 backdrop-blur">
                <TableRow>
                  <TableHead
                    scope="col"
                    aria-sort={
                      sort_by === "year"
                        ? sort_dir === "asc"
                          ? "ascending"
                          : "descending"
                        : "none"
                    }
                  >
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleSort("year")}
                      className="h-8 px-2 hover:bg-muted/40"
                    >
                      Year
                      {sort_by === "year" ? (
                        sort_dir === "asc" ? (
                          <ArrowUp
                            className="ml-2 h-4 w-4"
                            aria-hidden="true"
                          />
                        ) : (
                          <ArrowDown
                            className="ml-2 h-4 w-4"
                            aria-hidden="true"
                          />
                        )
                      ) : (
                        <ArrowUpDown
                          className="ml-2 h-4 w-4 opacity-50"
                          aria-hidden="true"
                        />
                      )}
                    </Button>
                  </TableHead>
                  <TableHead
                    scope="col"
                    aria-sort={
                      sort_by === "target"
                        ? sort_dir === "asc"
                          ? "ascending"
                          : "descending"
                        : "none"
                    }
                  >
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleSort("target")}
                      className="h-8 px-2 hover:bg-muted/40"
                    >
                      Target
                      {sort_by === "target" ? (
                        sort_dir === "asc" ? (
                          <ArrowUp
                            className="ml-2 h-4 w-4"
                            aria-hidden="true"
                          />
                        ) : (
                          <ArrowDown
                            className="ml-2 h-4 w-4"
                            aria-hidden="true"
                          />
                        )
                      ) : (
                        <ArrowUpDown
                          className="ml-2 h-4 w-4 opacity-50"
                          aria-hidden="true"
                        />
                      )}
                    </Button>
                  </TableHead>
                  <TableHead
                    scope="col"
                    aria-sort={
                      sort_by === "acquirer"
                        ? sort_dir === "asc"
                          ? "ascending"
                          : "descending"
                        : "none"
                    }
                  >
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleSort("acquirer")}
                      className="h-8 px-2 hover:bg-muted/40"
                    >
                      Acquirer
                      {sort_by === "acquirer" ? (
                        sort_dir === "asc" ? (
                          <ArrowUp
                            className="ml-2 h-4 w-4"
                            aria-hidden="true"
                          />
                        ) : (
                          <ArrowDown
                            className="ml-2 h-4 w-4"
                            aria-hidden="true"
                          />
                        )
                      ) : (
                        <ArrowUpDown
                          className="ml-2 h-4 w-4 opacity-50"
                          aria-hidden="true"
                        />
                      )}
                    </Button>
                  </TableHead>
                  <TableHead scope="col" className="hidden lg:table-cell">
                    Consideration type
                  </TableHead>
                  <TableHead scope="col" className="hidden xl:table-cell">
                    Total consideration
                  </TableHead>
                  <TableHead scope="col" className="w-28 text-right">
                    Verified
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {loading ? (
                  Array.from({ length: 6 }).map((_, index) => (
                    <TableRow key={`skeleton-${index}`}>
                      <TableCell colSpan={6} className="py-6">
                        <Skeleton className="h-4 w-full" />
                      </TableCell>
                    </TableRow>
                  ))
                ) : error ? (
                  <TableRow>
                    <TableCell colSpan={6} className="py-10 text-center">
                      <div
                        className="text-sm text-muted-foreground"
                        role="alert"
                      >
                        {error}
                      </div>
                    </TableCell>
                  </TableRow>
                ) : agreements.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={6} className="py-10 text-center">
                      <div
                        className="text-sm text-muted-foreground"
                        role="status"
                      >
                        No agreements match this filter.
                      </div>
                    </TableCell>
                  </TableRow>
                ) : (
                  agreements.map((agreement) => (
                    <TableRow key={agreement.agreement_uuid}>
                      <TableCell className="whitespace-nowrap">
                        <Badge variant="outline">
                          {formatYear(agreement.year)}
                        </Badge>
                      </TableCell>
                      <TableCell className="max-w-[320px] truncate font-semibold text-foreground">
                        <button
                          type="button"
                          onClick={() =>
                            setSelectedAgreement({
                              agreement_uuid: agreement.agreement_uuid,
                              year: agreement.year ?? "",
                              target: agreement.target ?? "",
                              acquirer: agreement.acquirer ?? "",
                            })
                          }
                          className="group inline-flex items-center gap-2 text-left text-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                        >
                          <span className="truncate">
                            {formatValue(agreement.target)}
                          </span>
                          <span className="text-xs font-semibold text-primary/80 opacity-0 transition-opacity group-hover:opacity-100">
                            View
                          </span>
                        </button>
                      </TableCell>
                      <TableCell className="max-w-[320px] truncate font-semibold text-muted-foreground">
                        {formatValue(agreement.acquirer)}
                      </TableCell>
                      <TableCell className="hidden lg:table-cell text-muted-foreground">
                        {formatValue(agreement.consideration_type)}
                      </TableCell>
                      <TableCell className="hidden xl:table-cell text-muted-foreground">
                        {formatValue(agreement.total_consideration)}
                      </TableCell>
                      <TableCell className="text-right">
                        {agreement.verified ? (
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <button
                                type="button"
                                aria-label="Verified agreement"
                                className="inline-flex items-center justify-end gap-1 rounded-full bg-emerald-500/10 px-2 py-0.5 text-xs font-medium text-emerald-700 ring-1 ring-emerald-500/20"
                              >
                                <BadgeCheck
                                  className="h-3.5 w-3.5"
                                  aria-hidden="true"
                                />
                                <span className="hidden sm:inline">
                                  Verified
                                </span>
                              </button>
                            </TooltipTrigger>
                            <TooltipContent side="left">
                              <p>This agreement has been verified by hand.</p>
                            </TooltipContent>
                          </Tooltip>
                        ) : (
                          <span className="text-xs text-muted-foreground">
                            —
                          </span>
                        )}
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>

          <div className="mt-6 space-y-4 lg:hidden">
            {loading ? (
              Array.from({ length: 4 }).map((_, index) => (
                <Card
                  key={`mobile-skeleton-${index}`}
                  className="border-border/60"
                >
                  <CardContent className="p-4">
                    <Skeleton className="h-4 w-32" />
                    <Skeleton className="mt-3 h-4 w-full" />
                    <Skeleton className="mt-2 h-4 w-3/4" />
                  </CardContent>
                </Card>
              ))
            ) : error ? (
              <Card className="border-border/60">
                <CardContent className="p-4">
                  <div className="text-sm text-muted-foreground" role="alert">
                    {error}
                  </div>
                </CardContent>
              </Card>
            ) : agreements.length === 0 ? (
              <Card className="border-border/60">
                <CardContent className="p-4">
                  <div className="text-sm text-muted-foreground" role="status">
                    No agreements match this filter.
                  </div>
                </CardContent>
              </Card>
            ) : (
              agreements.map((agreement) => (
                <Card
                  key={`mobile-${agreement.agreement_uuid}`}
                  className="border-border/60 hover:shadow-md transition-shadow duration-200"
                >
                  <CardContent className="space-y-3 p-4">
                    <div className="flex items-center justify-between">
                      <Badge variant="outline">
                        {formatYear(agreement.year)}
                      </Badge>
                      {agreement.verified ? (
                        <span className="inline-flex items-center gap-1 rounded-full bg-emerald-500/10 px-2 py-0.5 text-xs font-medium text-emerald-700 ring-1 ring-emerald-500/20">
                          <BadgeCheck
                            className="h-3.5 w-3.5"
                            aria-hidden="true"
                          />
                          Verified
                        </span>
                      ) : (
                        <span className="text-xs text-muted-foreground">—</span>
                      )}
                    </div>
                    <div className="space-y-2 text-sm">
                      <div>
                        <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                          Target
                        </div>
                        <button
                          type="button"
                          onClick={() =>
                            setSelectedAgreement({
                              agreement_uuid: agreement.agreement_uuid,
                              year: agreement.year ?? "",
                              target: agreement.target ?? "",
                              acquirer: agreement.acquirer ?? "",
                            })
                          }
                          className="mt-1 text-left font-semibold text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                        >
                          {formatValue(agreement.target)}
                        </button>
                      </div>
                      <div>
                        <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                          Acquirer
                        </div>
                        <div className="mt-1 text-muted-foreground">
                          {formatValue(agreement.acquirer)}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                          Consideration type
                        </div>
                        <div className="mt-1 text-muted-foreground">
                          {formatValue(agreement.consideration_type)}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                          Total consideration
                        </div>
                        <div className="mt-1 text-muted-foreground">
                          {formatValue(agreement.total_consideration)}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))
            )}
          </div>

          <div className="mt-6 flex flex-col items-center gap-4 sm:flex-row sm:justify-between">
            <div className="text-sm text-muted-foreground">
              Page {page} of {total_pages}
            </div>
            <Pagination className="justify-end sm:justify-center">
              <PaginationContent>
                <PaginationItem>
                  <PaginationPrevious
                    href="#"
                    onClick={(event) => {
                      event.preventDefault();
                      if (page > 1) setPage(page - 1);
                    }}
                    className={cn(
                      page === 1 && "pointer-events-none opacity-50",
                      "text-foreground",
                    )}
                  />
                </PaginationItem>
                {pageItems.map((item, idx) =>
                  item === "ellipsis" ? (
                    <PaginationItem key={`ellipsis-${idx}`}>
                      <PaginationEllipsis />
                    </PaginationItem>
                  ) : (
                    <PaginationItem key={`page-${item}`}>
                      <PaginationLink
                        href="#"
                        isActive={item === page}
                        onClick={(event) => {
                          event.preventDefault();
                          setPage(item);
                        }}
                      >
                        {item}
                      </PaginationLink>
                    </PaginationItem>
                  ),
                )}
                <PaginationItem>
                  <PaginationNext
                    href="#"
                    onClick={(event) => {
                      event.preventDefault();
                      if (page < total_pages) setPage(page + 1);
                    }}
                    className={cn(
                      page === total_pages && "pointer-events-none opacity-50",
                      "text-foreground",
                    )}
                  />
                </PaginationItem>
              </PaginationContent>
            </Pagination>
          </div>
        </CardContent>
      </Card>

      {selectedAgreement ? (
        <AgreementModal
          isOpen={!!selectedAgreement}
          onClose={() => setSelectedAgreement(null)}
          agreement_uuid={selectedAgreement.agreement_uuid}
          agreementMetadata={{
            year: selectedAgreement.year,
            target: selectedAgreement.target,
            acquirer: selectedAgreement.acquirer,
          }}
        />
      ) : null}
    </PageShell>
  );
}
