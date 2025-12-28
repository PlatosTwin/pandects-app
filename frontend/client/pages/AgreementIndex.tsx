import { useEffect, useMemo, useState } from "react";
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
import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";
import { cn } from "@/lib/utils";
import { AgreementModal } from "@/components/AgreementModal";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

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

export default function AgreementIndex() {
  const [summary, setSummary] = useState<AgreementIndexSummary | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(true);
  const [summaryError, setSummaryError] = useState<string | null>(null);

  const [agreements, setAgreements] = useState<AgreementIndexRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(25);
  const [totalCount, setTotalCount] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [sortBy, setSortBy] = useState<SortColumn>("year");
  const [sortDir, setSortDir] = useState<SortDirection>("desc");
  const [filter, setFilter] = useState("");
  const [selectedAgreement, setSelectedAgreement] = useState<{
    agreementUuid: string;
    year: string;
    target: string;
    acquirer: string;
  } | null>(null);

  useEffect(() => {
    let cancelled = false;
    const fetchSummary = async () => {
      try {
        setSummaryLoading(true);
        setSummaryError(null);
        const res = await authFetch(apiUrl("api/agreements-summary"));
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
        if (filter.trim()) {
          params.set("query", filter.trim());
        }

        const res = await authFetch(
          apiUrl(`api/agreements-index?${params.toString()}`),
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
  }, [page, pageSize, sortBy, sortDir, filter]);

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

  const filteredLabel = filter.trim()
    ? `Filtered by "${filter.trim()}"`
    : "Showing all agreements";

  return (
    <PageShell
      size="xl"
      title="Agreement Index"
      subtitle="Explore the full agreement universe and drill into deal-level metadata."
    >
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
              className="relative overflow-hidden border-border/60 bg-gradient-to-br from-background via-background to-muted/40"
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

      <Card className="border-border/60 shadow-sm">
        <CardContent className="p-6">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <h2 className="text-lg font-semibold text-foreground">
                Agreements
              </h2>
              <p className="text-sm text-muted-foreground" aria-live="polite">
                {filteredLabel}
              </p>
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
                  value={filter}
                  onChange={(event) => {
                    setFilter(event.target.value);
                    setPage(1);
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

          <div className="mt-6 rounded-lg border border-border/60 bg-background/60">
            <Table className="min-w-[900px]">
              <TableHeader className="sticky top-0 bg-background/95 backdrop-blur">
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
                      className="h-8 px-2"
                    >
                      Year
                      {sortBy === "year" ? (
                        sortDir === "asc" ? (
                          <ArrowUp className="ml-2 h-4 w-4" aria-hidden="true" />
                        ) : (
                          <ArrowDown className="ml-2 h-4 w-4" aria-hidden="true" />
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
                      className="h-8 px-2"
                    >
                      Target
                      {sortBy === "target" ? (
                        sortDir === "asc" ? (
                          <ArrowUp className="ml-2 h-4 w-4" aria-hidden="true" />
                        ) : (
                          <ArrowDown className="ml-2 h-4 w-4" aria-hidden="true" />
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
                      className="h-8 px-2"
                    >
                      Acquirer
                      {sortBy === "acquirer" ? (
                        sortDir === "asc" ? (
                          <ArrowUp className="ml-2 h-4 w-4" aria-hidden="true" />
                        ) : (
                          <ArrowDown className="ml-2 h-4 w-4" aria-hidden="true" />
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
                      <div className="text-sm text-muted-foreground" role="alert">
                        {error}
                      </div>
                    </TableCell>
                  </TableRow>
                ) : agreements.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={6} className="py-10 text-center">
                      <div className="text-sm text-muted-foreground" role="status">
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
                              <span className="inline-flex items-center justify-end gap-1 rounded-full bg-emerald-500/10 px-2 py-0.5 text-xs font-medium text-emerald-700 ring-1 ring-emerald-500/20">
                                <BadgeCheck className="h-3.5 w-3.5" aria-hidden="true" />
                                <span className="hidden sm:inline">Verified</span>
                              </span>
                            </TooltipTrigger>
                            <TooltipContent side="left">
                              <p>This agreement has been verified by hand.</p>
                            </TooltipContent>
                          </Tooltip>
                        ) : (
                          <span className="text-xs text-muted-foreground">—</span>
                        )}
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
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
