import { useEffect, useId, useMemo, useState } from "react";
import {
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  BadgeCheck,
  BookOpen,
  FileText,
  Layers,
  Search,
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
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import {
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
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useIsMobile } from "@/hooks/use-mobile";
import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";

type AgreementIndexRow = {
  agreementUuid: string;
  year: string | null;
  target: string | null;
  acquirer: string | null;
  considerationType: string | null;
  totalConsideration: string | number | null;
  targetIndustry: string | null;
  acquirerIndustry: string | null;
  verified: boolean;
};

type AgreementIndexResponse = {
  results: AgreementIndexRow[];
  page: number;
  pageSize: number;
  totalCount: number;
  totalPages: number;
  hasNext: boolean;
  hasPrev: boolean;
};

type AgreementIndexSummary = {
  agreements: number;
  sections: number;
  pages: number;
};

type AgreementStatusYearRow = {
  year: number;
  processed: number;
  pagesOnly: number;
  noPages: number;
};

type AgreementStatusSummaryResponse = {
  years: AgreementStatusYearRow[];
};

type SortColumn = "year" | "target" | "acquirer";
type SortDirection = "asc" | "desc";

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
    description: "Total agreements processed",
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

export default function AgreementIndex() {
  const isMobile = useIsMobile();
  const [isChartModalOpen, setIsChartModalOpen] = useState(false);
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
  const [statusAccordionValue, setStatusAccordionValue] = useState<
    string | undefined
  >(undefined);

  const [agreements, setAgreements] = useState<AgreementIndexRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(25);
  const [totalCount, setTotalCount] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [sortBy, setSortBy] = useState<SortColumn>("year");
  const [sortDir, setSortDir] = useState<SortDirection>("desc");
  const [filterInput, setFilterInput] = useState("");
  const [filterQuery, setFilterQuery] = useState("");
  const [selectedAgreement, setSelectedAgreement] = useState<{
    agreementUuid: string;
    year: string;
    target: string;
    acquirer: string;
  } | null>(null);
  const stagedChartDescriptionId = useId();
  const stagedChartTableId = useId();

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
    let cancelled = false;
    const controller = new AbortController();

    const fetchAgreements = async () => {
      try {
        setLoading(true);
        setError(null);
        const params = new URLSearchParams();
        params.set("page", String(page));
        params.set("pageSize", String(pageSize));
        params.set("sortBy", sortBy);
        params.set("sortDir", sortDir);
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
          setTotalCount(data.totalCount);
          setTotalPages(Math.max(1, data.totalPages));
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
  }, [page, pageSize, sortBy, sortDir, filterQuery]);

  useEffect(() => {
    const handle = window.setTimeout(() => {
      setFilterQuery(filterInput);
      setPage(1);
    }, 300);
    return () => window.clearTimeout(handle);
  }, [filterInput]);

  const handleSort = (column: SortColumn) => {
    if (column === sortBy) {
      setSortDir((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortBy(column);
      setSortDir("asc");
    }
    setPage(1);
  };

  const pageItems = useMemo(() => {
    if (totalPages <= 7) {
      return Array.from({ length: totalPages }, (_, i) => i + 1);
    }

    const items: Array<number | "ellipsis"> = [];
    items.push(1);

    const start = Math.max(2, page - 1);
    const end = Math.min(totalPages - 1, page + 1);

    if (start > 2) items.push("ellipsis");
    for (let i = start; i <= end; i += 1) items.push(i);
    if (end < totalPages - 1) items.push("ellipsis");

    items.push(totalPages);
    return items;
  }, [page, totalPages]);

  const filteredLabel = filterQuery.trim()
    ? `Filtered by "${filterQuery.trim()}"`
    : "";
  const stagedChartData = useMemo(() => {
    return statusSummary
      .filter((row) => Number.isFinite(row.year))
      .map((row) => ({
        year: Number(row.year),
        processed: Math.max(0, Number(row.processed || 0)),
        pagesOnly: Math.max(0, Number(row.pagesOnly || 0)),
        noPages: Math.max(0, Number(row.noPages || 0)),
      }))
      .sort((a, b) => a.year - b.year);
  }, [statusSummary]);
  const stagedTotals = useMemo(() => {
    return stagedChartData.reduce(
      (acc, row) => {
        acc.processed += row.processed;
        acc.pagesOnly += row.pagesOnly;
        acc.noPages += row.noPages;
        acc.total += row.processed + row.pagesOnly + row.noPages;
        return acc;
      },
      { processed: 0, pagesOnly: 0, noPages: 0, total: 0 },
    );
  }, [stagedChartData]);
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

  const renderStagedChart = (className?: string) => (
    <div
      className={cn(
        "rounded-lg border border-border/60 bg-muted/20 p-3",
        className,
      )}
    >
      <ChartContainer
        className="h-[240px] w-full aspect-auto sm:h-[300px] lg:h-[340px]"
        config={{
          processed: {
            label: "Processed",
            color: "hsl(142 71% 45%)",
          },
          pagesOnly: {
            label: "Awaiting validation",
            color: "hsl(0 84% 60%)",
          },
          noPages: {
            label: "Staged",
            color: "hsl(38 92% 55%)",
          },
        }}
        role="img"
        aria-label="Stacked bar chart showing processed versus staged agreements by filing year."
        aria-describedby={`${stagedChartDescriptionId} ${stagedChartTableId}`}
      >
        <BarChart
          data={stagedChartData}
          margin={{ top: 6, right: 16, left: 6, bottom: 0 }}
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
                        pagesOnly?: number;
                        noPages?: number;
                      }
                    | undefined;
                  const processed = Number(payload?.processed ?? 0);
                  const pagesOnly = Number(payload?.pagesOnly ?? 0);
                  const noPages = Number(payload?.noPages ?? 0);
                  const total = processed + noPages + pagesOnly;
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
                      : name === "Awaiting validation"
                        ? getPct(pagesOnly)
                        : getPct(noPages);

                  return (
                    <div className="grid grid-cols-[auto_3rem_minmax(0,1fr)] items-center gap-x-3">
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
            dataKey="pagesOnly"
            stackId="agreements"
            fill="var(--color-pagesOnly)"
            name="Awaiting validation"
          />
          <Bar
            dataKey="noPages"
            stackId="agreements"
            fill="var(--color-noPages)"
            name="Staged"
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
                <div className="space-y-3">
                  <p
                    id={stagedChartDescriptionId}
                    className="text-base text-muted-foreground"
                  >
                    Staged agreements have not yet gone through our pipelines.
                    Agreements that are awaiting validation have made it through
                    at least one step of the pipeline but tripped one of our
                    validations and are awaiting manual review. Agreements that
                    are staged or awaiting validation do not show up in the{" "}
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
                    routes, and are not included in the agreement statistics at
                    the top of this page. The dashed vertical divider marks the 2020/2021
                    boundary: from 2000 through 2020, we use data from the DMA
                    Corpus; beginning 2021, we source data ourselves from EDGAR.
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
                      {isMobile ? (
                        <>
                          <Button
                            type="button"
                            variant="outline"
                            className="w-full"
                            onClick={() => setIsChartModalOpen(true)}
                            aria-haspopup="dialog"
                            aria-describedby={stagedChartDescriptionId}
                          >
                            Click to view on mobile
                          </Button>
                          <Dialog
                            open={isChartModalOpen}
                            onOpenChange={setIsChartModalOpen}
                          >
                            <DialogContent
                              className="left-0 top-0 h-[100dvh] w-[100dvw] max-w-none translate-x-0 translate-y-0 rounded-none p-4"
                              aria-describedby={stagedChartDescriptionId}
                            >
                              <DialogTitle className="text-base font-semibold">
                                Agreement overview
                              </DialogTitle>
                              {renderStagedChart("border-0 bg-background p-0")}
                            </DialogContent>
                          </Dialog>
                        </>
                      ) : (
                        renderStagedChart()
                      )}
                      <table id={stagedChartTableId} className="sr-only">
                        <caption>
                          Processed, awaiting validation, and staged agreements
                          by filing year
                        </caption>
                        <thead>
                          <tr>
                            <th scope="col">Year</th>
                            <th scope="col">Processed</th>
                            <th scope="col">Awaiting validation</th>
                            <th scope="col">Staged</th>
                            <th scope="col">Total</th>
                          </tr>
                        </thead>
                        <tbody>
                          {stagedChartData.map((row) => (
                            <tr key={`staged-row-${row.year}`}>
                              <th scope="row">{row.year}</th>
                              <td>{row.processed.toLocaleString("en-US")}</td>
                              <td>{row.pagesOnly.toLocaleString("en-US")}</td>
                              <td>{row.noPages.toLocaleString("en-US")}</td>
                              <td>
                                {(
                                  row.processed +
                                  row.pagesOnly +
                                  row.noPages
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
                              {stagedTotals.pagesOnly.toLocaleString("en-US")}
                            </td>
                            <td>
                              {stagedTotals.noPages.toLocaleString("en-US")}
                            </td>
                            <td>
                              {stagedTotals.total.toLocaleString("en-US")}
                            </td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
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
                {totalCount.toLocaleString("en-US")} agreements
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
                      sortBy === "year"
                        ? sortDir === "asc"
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
                      {sortBy === "year" ? (
                        sortDir === "asc" ? (
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
                      sortBy === "target"
                        ? sortDir === "asc"
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
                      {sortBy === "target" ? (
                        sortDir === "asc" ? (
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
                      sortBy === "acquirer"
                        ? sortDir === "asc"
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
                      {sortBy === "acquirer" ? (
                        sortDir === "asc" ? (
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
                    <TableRow key={agreement.agreementUuid}>
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
                              agreementUuid: agreement.agreementUuid,
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
                        {formatValue(agreement.considerationType)}
                      </TableCell>
                      <TableCell className="hidden xl:table-cell text-muted-foreground">
                        {formatValue(agreement.totalConsideration)}
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
                  key={`mobile-${agreement.agreementUuid}`}
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
                              agreementUuid: agreement.agreementUuid,
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
                          {formatValue(agreement.considerationType)}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                          Total consideration
                        </div>
                        <div className="mt-1 text-muted-foreground">
                          {formatValue(agreement.totalConsideration)}
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
              Page {page} of {totalPages}
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
                      if (page < totalPages) setPage(page + 1);
                    }}
                    className={cn(
                      page === totalPages && "pointer-events-none opacity-50",
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
          agreementUuid={selectedAgreement.agreementUuid}
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
