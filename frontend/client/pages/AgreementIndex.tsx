import { Suspense, lazy, useEffect, useMemo, useState } from "react";
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
import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";
import { readSessionCache, writeSessionCache } from "@/lib/session-cache";
import { cn } from "@/lib/utils";

const AgreementModal = lazy(() =>
  import("@/components/AgreementModal").then((mod) => ({
    default: mod.AgreementModal,
  })),
);
const AgreementIndexOverview = lazy(() =>
  import("@/components/AgreementIndexOverview").then((mod) => ({
    default: mod.AgreementIndexOverview,
  })),
);

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
const AGREEMENT_SUMMARY_CACHE_KEY = "agreement-index-summary:v1";
const AGREEMENT_SUMMARY_CACHE_TTL_MS = 5 * 60 * 1000;

export default function AgreementIndex() {
  const [summary, setSummary] = useState<AgreementIndexSummary | null>(() =>
    readSessionCache<AgreementIndexSummary>(
      AGREEMENT_SUMMARY_CACHE_KEY,
      AGREEMENT_SUMMARY_CACHE_TTL_MS,
    ),
  );
  const [summaryLoading, setSummaryLoading] = useState(summary === null);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const [hasLoadedOverview, setHasLoadedOverview] = useState(false);

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

  useEffect(() => {
    let cancelled = false;
    let timeoutId: number | null = null;
    let idleId: number | null = null;

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
          writeSessionCache(AGREEMENT_SUMMARY_CACHE_KEY, data);
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

    const scheduleFetch = () => {
      if (cancelled) return;
      void fetchSummary();
    };

    const browserWindow = typeof window !== "undefined" ? window : null;

    if (browserWindow && "requestIdleCallback" in browserWindow) {
      idleId = browserWindow.requestIdleCallback(scheduleFetch, { timeout: 1500 });
    } else {
      timeoutId = window.setTimeout(scheduleFetch, 800);
    }

    return () => {
      cancelled = true;
      if (idleId !== null && browserWindow && "cancelIdleCallback" in browserWindow) {
        browserWindow.cancelIdleCallback(idleId);
      }
      if (timeoutId !== null) {
        window.clearTimeout(timeoutId);
      }
    };
  }, []);

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
  }, [filterQuery, page, page_size, sort_by, sort_dir]);

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
  const isInitialAgreementLoad = loading && agreements.length === 0 && !error;

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
                  <div className="flex min-h-7 items-center text-2xl font-semibold tabular-nums text-foreground">
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
            onValueChange={(value) => {
              if (value === "staged") {
                setHasLoadedOverview(true);
              }
            }}
          >
            <AccordionItem value="staged" className="border-border/60">
              <AccordionTrigger
                headingLevel="h2"
                className="py-3 text-2xl font-semibold tracking-tight"
              >
                Agreement overview
              </AccordionTrigger>
              <AccordionContent
                disableAnimation
                className="pt-3"
              >
                {hasLoadedOverview ? (
                  <Suspense
                    fallback={
                      <div className="space-y-3">
                        <Skeleton className="h-10 w-full" />
                        <Skeleton className="h-20 w-full" />
                        <Skeleton className="h-[260px] w-full" />
                      </div>
                    }
                  >
                    <AgreementIndexOverview />
                  </Suspense>
                ) : null}
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </CardContent>
      </Card>

      <Card className="border-border/60 shadow-sm transition-shadow duration-200 hover:shadow-md">
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
              <Badge
                variant="secondary"
                className="self-start tabular-nums sm:self-auto"
              >
                {total_count.toLocaleString("en-US")} agreements
              </Badge>
            </div>
          </div>

          <div className="mt-6 hidden rounded-lg border border-border/60 bg-muted/20 lg:block">
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
              <TableBody aria-busy={loading}>
                {isInitialAgreementLoad ? (
                  Array.from({ length: page_size }).map((_, index) => (
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
                      <TableCell className="hidden text-muted-foreground lg:table-cell">
                        {formatValue(agreement.consideration_type)}
                      </TableCell>
                      <TableCell className="hidden text-muted-foreground xl:table-cell">
                        {formatValue(agreement.total_consideration)}
                      </TableCell>
                      <TableCell className="text-right">
                        {agreement.verified ? (
                          <span
                            title="This agreement has been verified by hand."
                            aria-label="Verified agreement. This agreement has been verified by hand."
                            className="inline-flex items-center justify-end gap-1 rounded-full bg-emerald-500/10 px-2 py-0.5 text-xs font-medium text-emerald-700 ring-1 ring-emerald-500/20"
                          >
                            <BadgeCheck
                              className="h-3.5 w-3.5"
                              aria-hidden="true"
                            />
                            <span className="hidden sm:inline">
                              Verified
                            </span>
                          </span>
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
            {isInitialAgreementLoad ? (
              Array.from({ length: page_size }).map((_, index) => (
                <Card key={`mobile-skeleton-${index}`} className="border-border/60">
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
                  className="border-border/60 transition-shadow duration-200 hover:shadow-md"
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
        <Suspense fallback={null}>
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
        </Suspense>
      ) : null}
    </PageShell>
  );
}
