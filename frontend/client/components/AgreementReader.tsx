import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  ArrowLeft,
  Building2,
  Calendar,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  CircleDollarSign,
  ExternalLink,
  FileSearch,
  FileText,
  Info,
  ListTree,
  Search,
  SlidersHorizontal,
  ShieldCheck,
  Tag,
  X,
} from "lucide-react";
import { useAgreement } from "@/hooks/use-agreement";
import { useFilterOptions } from "@/hooks/use-filter-options";
import { useIsMobile } from "@/hooks/use-mobile";
import { TableOfContents } from "@/components/TableOfContents";
import { XMLRenderer } from "@/components/XMLRenderer";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { LoadingSpinner } from "@/components/ui/loading-spinner";
import { Skeleton } from "@/components/ui/skeleton";
import { indexClauseTypeLabels } from "@/lib/clause-type-index";
import { extractAgreementTextSections } from "@/lib/agreement-search";
import {
  formatBooleanValue,
  formatCompactCurrencyValue,
  formatDateValue,
  formatEnumValue,
  formatTextValue,
} from "@/lib/format-utils";
import { apiUrl } from "@/lib/api-config";
import { authFetch } from "@/lib/auth-fetch";
import { logger } from "@/lib/logger";
import { cn } from "@/lib/utils";
import type {
  AgreementSectionIndexResponse,
  AgreementSectionIndexItem,
} from "@shared/transactions";

interface AgreementReaderProps {
  agreementUuid: string;
  focusSectionUuid?: string | null;
  backTo?: string | null;
}

interface JumpItem {
  sectionUuid: string;
  title: string;
  subtitle: string;
  preview?: string | null;
}

type AgreementTextSize = "small" | "medium" | "large";

function HeaderFactChip({
  icon: Icon,
  label,
  value,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
}) {
  return (
    <div className="inline-flex max-w-full min-w-0 items-center gap-1.5 rounded-full border border-border bg-background/80 px-2.5 py-1 text-xs">
      <Icon className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
      <span className="text-muted-foreground">{label}</span>
      <span className="min-w-0 truncate font-medium text-foreground">{value}</span>
    </div>
  );
}

function ReaderSearch({
  textQuery,
  onTextQueryChange,
  sectionType,
  onSectionTypeChange,
  sectionTypeOptions,
  jumpItems,
  activeSectionUuid,
  onJumpToSection,
  matchCount,
  isTextQueryActive,
}: {
  textQuery: string;
  onTextQueryChange: (value: string) => void;
  sectionType: string;
  onSectionTypeChange: (value: string) => void;
  sectionTypeOptions: Array<{ value: string; label: string; count: number }>;
  jumpItems: JumpItem[];
  activeSectionUuid: string | null;
  onJumpToSection: (sectionUuid: string) => void;
  matchCount: number | null;
  isTextQueryActive: boolean;
}) {
  return (
    <div className="space-y-4">
      <div className="space-y-1.5">
        <label
          className="text-xs font-semibold uppercase tracking-wide text-muted-foreground"
          htmlFor="agreement-text-search"
        >
          Search this agreement
        </label>
        <div className="relative">
          <Search
            className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
            aria-hidden="true"
          />
          <Input
            id="agreement-text-search"
            value={textQuery}
            onChange={(event) => onTextQueryChange(event.target.value)}
            placeholder="Find text in this document"
            className="h-11 pl-9 pr-9 sm:h-9"
            aria-describedby="agreement-text-search-hint"
          />
          {textQuery ? (
            <button
              type="button"
              onClick={() => onTextQueryChange("")}
              className="absolute right-1.5 top-1/2 flex h-7 w-7 -translate-y-1/2 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
              aria-label="Clear text search"
            >
              <X className="h-3.5 w-3.5" aria-hidden="true" />
            </button>
          ) : null}
        </div>
        {isTextQueryActive ? (
          <p id="agreement-text-search-hint" className="text-xs text-muted-foreground">
            {matchCount === 0
              ? "No matches"
              : `${matchCount} match${matchCount === 1 ? "" : "es"}`}
          </p>
        ) : null}
      </div>

      <div className="space-y-1.5">
        <label
          className="text-xs font-semibold uppercase tracking-wide text-muted-foreground"
          htmlFor="agreement-section-type"
        >
          Filter by section type
        </label>
        <Select value={sectionType} onValueChange={onSectionTypeChange}>
          <SelectTrigger
            id="agreement-section-type"
            className="h-11 sm:h-9"
            aria-label="Filter agreement navigation by section type"
          >
            <SelectValue placeholder="All section types" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All section types</SelectItem>
            {sectionTypeOptions.map((option) => (
              <SelectItem key={option.value} value={option.value}>
                <span className="flex items-center gap-2">
                  <span className="truncate">{option.label}</span>
                  <span className="ml-auto shrink-0 text-xs tabular-nums text-muted-foreground">
                    {option.count}
                  </span>
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {isTextQueryActive
              ? "Matching text"
              : sectionType === "all"
                ? "Jump to section"
                : "Matching sections"}
          </div>
          {jumpItems.length > 0 ? (
            <span className="text-xs tabular-nums text-muted-foreground">
              {jumpItems.length}
            </span>
          ) : null}
        </div>
        {jumpItems.length === 0 ? (
          <div className="rounded-md border border-dashed border-border bg-muted/20 p-3 text-center text-xs text-muted-foreground">
            {isTextQueryActive
              ? "No text matches yet."
              : sectionType === "all"
                ? "Use text search or a section-type filter to jump."
                : "No sections of this type."}
          </div>
        ) : (
          <ul className="space-y-1.5">
            {jumpItems.map((item) => {
              const isActive = item.sectionUuid === activeSectionUuid;
              return (
                <li key={`${item.sectionUuid}-${item.title}`}>
                  <button
                    type="button"
                    onClick={() => onJumpToSection(item.sectionUuid)}
                    className={cn(
                      "group min-h-11 w-full rounded-lg border p-2.5 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
                      isActive
                        ? "border-primary/40 bg-primary/5"
                        : "border-border bg-muted/20 hover:border-border hover:bg-muted/40",
                    )}
                    aria-current={isActive ? "true" : undefined}
                  >
                    <div className="flex items-start gap-2">
                      <FileText
                        className={cn(
                          "mt-0.5 h-3.5 w-3.5 shrink-0",
                          isActive ? "text-primary" : "text-muted-foreground",
                        )}
                        aria-hidden="true"
                      />
                      <div className="min-w-0 flex-1">
                        <div
                          className={cn(
                            "truncate text-sm font-medium",
                            isActive ? "text-primary" : "text-foreground",
                          )}
                        >
                          {item.title}
                        </div>
                        <div className="truncate text-xs text-muted-foreground">
                          {item.subtitle}
                        </div>
                        {item.preview ? (
                          <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-muted-foreground">
                            {item.preview}
                          </p>
                        ) : null}
                      </div>
                    </div>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}

function ReaderDetails({
  metadata,
  filingUrl,
}: {
  metadata: Array<{ label: string; value: string }>;
  filingUrl?: string | null;
}) {
  return (
    <div className="space-y-4">
      <dl className="divide-y divide-border/50 rounded-lg border border-border bg-card">
        {metadata.map((item) => (
          <div
            key={item.label}
            className="grid gap-1 px-3 py-2 text-sm sm:grid-cols-[110px_minmax(0,1fr)] sm:gap-3"
          >
            <dt className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
              {item.label}
            </dt>
            <dd className="break-words text-foreground">{item.value}</dd>
          </div>
        ))}
      </dl>
      {filingUrl ? (
        <Button asChild variant="outline" size="sm" className="w-full gap-2">
          <a href={filingUrl} target="_blank" rel="noopener noreferrer">
            <ExternalLink className="h-4 w-4" aria-hidden="true" />
            View SEC filing
          </a>
        </Button>
      ) : null}
    </div>
  );
}

export function AgreementReader({
  agreementUuid,
  focusSectionUuid,
  backTo,
}: AgreementReaderProps) {
  const { agreement, isLoading, error, fetchAgreement } = useAgreement();
  const [sectionIndex, setSectionIndex] = useState<AgreementSectionIndexItem[]>([]);
  const [sectionIndexLoading, setSectionIndexLoading] = useState(true);
  const [sectionIndexError, setSectionIndexError] = useState<string | null>(null);
  const [highlightedSection, setHighlightedSection] = useState<string | null>(
    focusSectionUuid ?? null,
  );
  const [textQuery, setTextQuery] = useState("");
  const [sectionType, setSectionType] = useState("all");
  const [sidebarTab, setSidebarTab] = useState<"search" | "details">("search");
  const [isTocSheetOpen, setIsTocSheetOpen] = useState(false);
  const [isDetailsSheetOpen, setIsDetailsSheetOpen] = useState(false);
  const [leftOpen, setLeftOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(true);
  const [headerCollapsed, setHeaderCollapsed] = useState(false);
  const [showBodyOnly, setShowBodyOnly] = useState(false);
  const [textSize, setTextSize] = useState<AgreementTextSize>("medium");
  const [stickyHeaderBottom, setStickyHeaderBottom] = useState(240);
  const highlightTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const contentRef = useRef<HTMLDivElement>(null);
  const headerRef = useRef<HTMLDivElement>(null);
  const isMobile = useIsMobile();

  useEffect(() => {
    if (isLoading) return;
    const element = headerRef.current;
    if (!element) return;
    const update = () => {
      const rect = element.getBoundingClientRect();
      // sticky top-16 → header occupies [64px, 64+height] once pinned
      setStickyHeaderBottom(64 + rect.height);
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(element);
    window.addEventListener("resize", update);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", update);
    };
  }, [isLoading]);

  useEffect(() => {
    const element = headerRef.current;
    if (!element) return;
    const raf = requestAnimationFrame(() => {
      const rect = element.getBoundingClientRect();
      setStickyHeaderBottom(64 + rect.height);
    });
    return () => cancelAnimationFrame(raf);
  }, [headerCollapsed, agreement?.is_redacted]);

  const { clause_types } = useFilterOptions({ fields: ["clause_types"] });
  const clauseTypeLabelById = useMemo(
    () => indexClauseTypeLabels(clause_types),
    [clause_types],
  );

  useEffect(() => {
    fetchAgreement(agreementUuid, focusSectionUuid ?? undefined);
  }, [agreementUuid, fetchAgreement, focusSectionUuid]);

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();

    const fetchSectionIndex = async () => {
      try {
        setSectionIndexLoading(true);
        setSectionIndexError(null);
        const response = await authFetch(
          apiUrl(`v1/agreements/${agreementUuid}/sections`),
          { signal: controller.signal },
        );
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        const payload = (await response.json()) as AgreementSectionIndexResponse;
        if (!cancelled) setSectionIndex(payload.results);
      } catch (fetchError) {
        if (!cancelled) {
          logger.error("Failed to fetch agreement sections index:", fetchError);
          setSectionIndex([]);
          setSectionIndexError(
            fetchError instanceof Error
              ? fetchError.message
              : "Unable to load agreement sections.",
          );
        }
      } finally {
        if (!cancelled) setSectionIndexLoading(false);
      }
    };

    void fetchSectionIndex();
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [agreementUuid]);

  const scrollToSection = (sectionUuid: string) => {
    setHighlightedSection(sectionUuid);
    if (highlightTimeoutRef.current) {
      clearTimeout(highlightTimeoutRef.current);
    }
    highlightTimeoutRef.current = setTimeout(() => {
      setHighlightedSection(null);
      highlightTimeoutRef.current = null;
    }, 2200);
    const sectionElement = contentRef.current?.querySelector(
      `[data-section-uuid="${sectionUuid}"]`,
    ) as HTMLElement | null;
    if (!sectionElement) return;
    const rect = sectionElement.getBoundingClientRect();
    const targetY = window.scrollY + rect.top - stickyHeaderBottom - 16;
    window.scrollTo({ top: Math.max(0, targetY), behavior: "smooth" });
    setIsTocSheetOpen(false);
    setIsDetailsSheetOpen(false);
  };

  const scrollToAnchor = (anchorId: string) => {
    const element = document.getElementById(anchorId);
    if (!element) return;

    const targetY =
      element.getBoundingClientRect().top +
      window.scrollY -
      stickyHeaderBottom -
      16;
    window.scrollTo({ top: Math.max(0, targetY), behavior: "smooth" });
  };

  useEffect(() => {
    return () => {
      if (highlightTimeoutRef.current) clearTimeout(highlightTimeoutRef.current);
    };
  }, []);

  useEffect(() => {
    if (!focusSectionUuid || !agreement?.xml) return;
    const timer = window.setTimeout(() => scrollToSection(focusSectionUuid), 120);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agreement?.xml, focusSectionUuid]);

  const textSections = useMemo(
    () => (agreement?.xml ? extractAgreementTextSections(agreement.xml) : []),
    [agreement?.xml],
  );

  const normalizedTextQuery = textQuery.trim().toLowerCase();
  const isTextQueryActive = normalizedTextQuery.length > 0;

  const textMatches = useMemo(() => {
    if (!isTextQueryActive) return [];
    return textSections
      .filter((section) => section.text.toLowerCase().includes(normalizedTextQuery))
      .map((section) => {
        const matchIndex = section.text.toLowerCase().indexOf(normalizedTextQuery);
        const previewStart = Math.max(0, matchIndex - 40);
        const previewEnd = Math.min(section.text.length, matchIndex + 140);
        const prefix = previewStart > 0 ? "\u2026" : "";
        const suffix = previewEnd < section.text.length ? "\u2026" : "";
        return {
          sectionUuid: section.sectionUuid,
          title: section.sectionTitle || section.articleTitle || "Matched section",
          subtitle: section.articleTitle || "Text match",
          preview: `${prefix}${section.text.slice(previewStart, previewEnd)}${suffix}`,
        };
      });
  }, [isTextQueryActive, normalizedTextQuery, textSections]);

  const sectionTypeOptions = useMemo(() => {
    const counts = new Map<string, number>();
    for (const item of sectionIndex) {
      for (const id of item.standard_id) {
        counts.set(id, (counts.get(id) ?? 0) + 1);
      }
    }
    return Array.from(counts.entries())
      .map(([value, count]) => ({
        value,
        label: clauseTypeLabelById[value] ?? value,
        count,
      }))
      .sort((a, b) => a.label.localeCompare(b.label));
  }, [clauseTypeLabelById, sectionIndex]);

  const sectionTypeMatches = useMemo(() => {
    if (sectionType === "all") return [];
    return sectionIndex
      .filter((item) => item.standard_id.includes(sectionType))
      .map((item) => ({
        sectionUuid: item.section_uuid,
        title: item.section_title || item.article_title || "Matched section",
        subtitle: clauseTypeLabelById[sectionType] ?? sectionType,
      }));
  }, [clauseTypeLabelById, sectionIndex, sectionType]);

  const jumpItems: JumpItem[] = isTextQueryActive
    ? textMatches.slice(0, 50)
    : sectionTypeMatches.slice(0, 50);
  const matchCount = isTextQueryActive ? textMatches.length : null;

  const metadata = [
    { label: "Target", value: formatTextValue(agreement?.target) },
    { label: "Acquirer", value: formatTextValue(agreement?.acquirer) },
    { label: "Filed", value: formatDateValue(agreement?.filing_date) },
    { label: "Announced", value: formatDateValue(agreement?.announce_date) },
    { label: "Closed", value: formatDateValue(agreement?.close_date) },
    { label: "Deal type", value: formatEnumValue(agreement?.deal_type) },
    { label: "Status", value: formatEnumValue(agreement?.deal_status) },
    { label: "Attitude", value: formatEnumValue(agreement?.attitude) },
    { label: "Purpose", value: formatEnumValue(agreement?.purpose) },
    { label: "Consideration", value: formatEnumValue(agreement?.transaction_consideration) },
    { label: "Value", value: formatCompactCurrencyValue(agreement?.transaction_price_total) },
    { label: "Target industry", value: formatTextValue(agreement?.target_industry) },
    { label: "Acquirer industry", value: formatTextValue(agreement?.acquirer_industry) },
    { label: "Target PE", value: formatBooleanValue(agreement?.target_pe) },
    { label: "Acquirer PE", value: formatBooleanValue(agreement?.acquirer_pe) },
  ];

  const targetName = agreement?.target?.trim() || "";
  const acquirerName = agreement?.acquirer?.trim() || "";
  const title = targetName || acquirerName || "Agreement";
  const subtitle = targetName && acquirerName ? `Acquired by ${acquirerName}` : null;

  if (isLoading) {
    return (
      <div className="mx-auto max-w-[1600px] px-4 py-8 sm:px-6 lg:px-8">
        <div className="space-y-6">
          <Skeleton className="h-8 w-40" />
          <div className="space-y-3">
            <Skeleton className="h-9 w-3/4" />
            <div className="flex flex-wrap gap-2">
              <Skeleton className="h-6 w-24" />
              <Skeleton className="h-6 w-32" />
              <Skeleton className="h-6 w-28" />
              <Skeleton className="h-6 w-36" />
            </div>
          </div>
          <div className="grid gap-6 lg:grid-cols-[280px_minmax(0,1fr)_320px]">
            <Skeleton className="h-96 w-full" />
            <Skeleton className="h-96 w-full" />
            <Skeleton className="h-96 w-full" />
          </div>
        </div>
      </div>
    );
  }

  if (error || !agreement) {
    return (
      <div className="mx-auto max-w-2xl px-4 py-10 sm:px-6 lg:px-8">
        <Alert variant="destructive">
          <AlertTitle>Agreement unavailable</AlertTitle>
          <AlertDescription>
            {error ?? "Unable to load this agreement."}
          </AlertDescription>
        </Alert>
        <div className="mt-4">
          <Button asChild variant="outline" size="sm" className="gap-2">
            <Link to={backTo || "/search"}>
              <ArrowLeft className="h-4 w-4" aria-hidden="true" />
              Back to results
            </Link>
          </Button>
        </div>
      </div>
    );
  }

  const headerFacts: Array<{
    icon: React.ComponentType<{ className?: string }>;
    label: string;
    value: string;
  }> = [];
  if (agreement.year) {
    headerFacts.push({
      icon: Calendar,
      label: "Year",
      value: String(agreement.year),
    });
  }
  if (agreement.transaction_price_total !== null) {
    headerFacts.push({
      icon: CircleDollarSign,
      label: "Value",
      value: formatCompactCurrencyValue(agreement.transaction_price_total),
    });
  }
  if (agreement.deal_type) {
    headerFacts.push({
      icon: Tag,
      label: "Type",
      value: formatEnumValue(agreement.deal_type),
    });
  }
  if (agreement.deal_status) {
    headerFacts.push({
      icon: ShieldCheck,
      label: "Status",
      value: formatEnumValue(agreement.deal_status),
    });
  }
  if (agreement.filing_date) {
    headerFacts.push({
      icon: Calendar,
      label: "Filed",
      value: formatDateValue(agreement.filing_date),
    });
  }
  if (agreement.target_industry) {
    headerFacts.push({
      icon: Building2,
      label: "Industry",
      value: agreement.target_industry,
    });
  }

  const readerSearch = (
    <ReaderSearch
      textQuery={textQuery}
      onTextQueryChange={setTextQuery}
      sectionType={sectionType}
      onSectionTypeChange={setSectionType}
      sectionTypeOptions={sectionTypeOptions}
      jumpItems={jumpItems}
      activeSectionUuid={highlightedSection}
      onJumpToSection={scrollToSection}
      matchCount={matchCount}
      isTextQueryActive={isTextQueryActive}
    />
  );

  const readerDetails = (
    <ReaderDetails metadata={metadata} filingUrl={agreement.url ?? null} />
  );

  const agreementTextSizeClass =
    textSize === "small"
      ? "text-sm"
      : textSize === "large"
        ? "text-base sm:text-lg"
        : "text-sm sm:text-base";

  return (
    <div className="overflow-x-hidden" style={{ overflowX: "clip" }}>
      {/* Sticky header */}
      <div
        ref={headerRef}
        className="sticky top-16 z-30 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80"
      >
        <div className="mx-auto max-w-[1600px] px-4 sm:px-6 lg:px-8">
          {/* Single row: back | title+subtitle | SEC filing */}
          <div className="flex flex-wrap items-center gap-2 py-2 sm:flex-nowrap sm:py-2.5">
            <Button
              asChild
              variant="ghost"
              size="sm"
              className="-ml-2 h-11 shrink-0 gap-1.5 px-2 text-muted-foreground hover:text-foreground sm:h-8"
            >
              <Link
                to={backTo || "/search"}
                onClick={() => window.scrollTo({ top: 0, left: 0 })}
              >
                <ArrowLeft className="h-4 w-4" aria-hidden="true" />
                <span className="hidden sm:inline">Back to results</span>
                <span className="sm:hidden">Back</span>
              </Link>
            </Button>
            <div className="flex min-w-0 flex-1 items-baseline gap-1.5">
              <span className="shrink-0 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground">
                Target
              </span>
              <h1
                id="agreement-reader-title"
                className="min-w-0 truncate text-base font-semibold text-foreground sm:text-lg"
                title={subtitle ? `${title} — ${subtitle}` : title}
              >
                {title}
              </h1>
              {subtitle ? (
                <span
                  className="hidden min-w-0 shrink truncate text-sm text-muted-foreground sm:inline"
                  title={subtitle}
                >
                  · {subtitle}
                </span>
              ) : null}
            </div>
            <DropdownMenu modal={false}>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-11 w-11 shrink-0 gap-1.5 px-2 text-muted-foreground hover:text-foreground sm:h-8 sm:w-auto"
                  aria-label="Open view options"
                >
                  <SlidersHorizontal className="h-4 w-4" aria-hidden="true" />
                  <span className="hidden sm:inline">View options</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-48">
                <DropdownMenuCheckboxItem
                  checked={showBodyOnly}
                  onCheckedChange={(checked) => setShowBodyOnly(checked === true)}
                  onSelect={(event) => event.preventDefault()}
                >
                  Show body only
                </DropdownMenuCheckboxItem>
                <DropdownMenuSeparator />
                <DropdownMenuLabel>Text size</DropdownMenuLabel>
                <DropdownMenuRadioGroup
                  value={textSize}
                  onValueChange={(value) =>
                    setTextSize(value as AgreementTextSize)
                  }
                >
                  <DropdownMenuRadioItem value="small">Small</DropdownMenuRadioItem>
                  <DropdownMenuRadioItem value="medium">Medium</DropdownMenuRadioItem>
                  <DropdownMenuRadioItem value="large">Large</DropdownMenuRadioItem>
                </DropdownMenuRadioGroup>
              </DropdownMenuContent>
            </DropdownMenu>
            {agreement.url ? (
              <Button
                asChild
                variant="ghost"
                size="sm"
                className="hidden h-8 shrink-0 gap-1.5 px-2 text-muted-foreground hover:text-foreground sm:inline-flex"
              >
                <a href={agreement.url} target="_blank" rel="noopener noreferrer">
                  <ExternalLink className="h-4 w-4" aria-hidden="true" />
                  SEC filing
                </a>
              </Button>
            ) : null}
          </div>

          {/* Expanded: mobile subtitle + fact chips + mobile sheet buttons + redacted alert */}
          {!headerCollapsed ? (
            <div className="space-y-2 pb-2 sm:pb-2.5">
              {subtitle ? (
                <div className="truncate text-sm text-muted-foreground sm:hidden">
                  {subtitle}
                </div>
              ) : null}
              {headerFacts.length > 0 ? (
                <div className="flex flex-wrap gap-1.5">
                  {headerFacts.map((fact) => (
                    <HeaderFactChip
                      key={fact.label}
                      icon={fact.icon}
                      label={fact.label}
                      value={fact.value}
                    />
                  ))}
                </div>
              ) : null}
              {isMobile ? (
                <div className="flex gap-2 pt-0.5">
                  <Sheet open={isTocSheetOpen} onOpenChange={setIsTocSheetOpen}>
                    <SheetTrigger asChild>
                      <Button variant="outline" size="sm" className="h-11 flex-1 gap-2">
                        <ListTree className="h-4 w-4" aria-hidden="true" />
                        Contents
                      </Button>
                    </SheetTrigger>
                    <SheetContent side="left" className="w-[min(360px,100vw)] p-0">
                      <SheetTitle className="sr-only">Agreement contents</SheetTitle>
                      <SheetDescription className="sr-only">
                        Browse agreement sections and jump to a selected section.
                      </SheetDescription>
                      <TableOfContents
                        xmlContent={agreement.xml}
                        targetSectionUuid={highlightedSection ?? undefined}
                        onSectionClick={scrollToSection}
                        onAnchorClick={scrollToAnchor}
                        className="h-full"
                        scrollable={true}
                      />
                    </SheetContent>
                  </Sheet>
                  <Sheet open={isDetailsSheetOpen} onOpenChange={setIsDetailsSheetOpen}>
                    <SheetTrigger asChild>
                      <Button variant="outline" size="sm" className="h-11 flex-1 gap-2">
                        <Info className="h-4 w-4" aria-hidden="true" />
                        Search & details
                      </Button>
                    </SheetTrigger>
                    <SheetContent
                      side="right"
                      className="w-[min(380px,100vw)] overflow-y-auto p-4"
                    >
                      <SheetTitle className="sr-only">Search and details</SheetTitle>
                      <SheetDescription className="sr-only">
                        Search within the agreement and review deal metadata.
                      </SheetDescription>
                      <Tabs
                        value={sidebarTab}
                        onValueChange={(value) =>
                          setSidebarTab(value === "details" ? "details" : "search")
                        }
                      >
                        <TabsList className="grid w-full grid-cols-2">
                          <TabsTrigger value="search">Search</TabsTrigger>
                          <TabsTrigger value="details">Details</TabsTrigger>
                        </TabsList>
                        <TabsContent value="search" className="mt-4">
                          {readerSearch}
                        </TabsContent>
                        <TabsContent value="details" className="mt-4">
                          {readerDetails}
                        </TabsContent>
                      </Tabs>
                    </SheetContent>
                  </Sheet>
                </div>
              ) : null}
              {agreement.is_redacted ? (
                <Alert className="border-amber-500/30 bg-amber-500/5 py-2.5 text-amber-900 dark:text-amber-100">
                  <FileSearch className="h-4 w-4" aria-hidden="true" />
                  <AlertTitle className="text-sm">Limited agreement text</AlertTitle>
                  <AlertDescription className="text-xs">
                    Full-text access is limited in anonymous mode. Your matched section
                    stays in view, but some surrounding text may be redacted.
                  </AlertDescription>
                </Alert>
              ) : null}
            </div>
          ) : null}
        </div>
        {/* Full-width collapse bar */}
        <button
          type="button"
          onClick={() => setHeaderCollapsed((v) => !v)}
          aria-label={headerCollapsed ? "Expand header" : "Collapse header"}
          aria-expanded={!headerCollapsed}
          className="group flex min-h-[22px] w-full items-center justify-center gap-1.5 border-t border-border bg-muted/40 py-0 text-muted-foreground/60 transition-colors hover:bg-muted/70 hover:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring"
        >
          <div className="h-px w-8 rounded-full bg-current transition-colors" aria-hidden="true" />
          <span className="flex items-center gap-1 text-[10px] font-medium uppercase tracking-wide">
            {headerCollapsed ? (
              <>
                <ChevronDown className="h-3 w-3" aria-hidden="true" />
                Expand
              </>
            ) : (
              <>
                <ChevronUp className="h-3 w-3" aria-hidden="true" />
                Collapse
              </>
            )}
          </span>
          <div className="h-px w-8 rounded-full bg-current transition-colors" aria-hidden="true" />
        </button>
      </div>

      {/* Main layout */}
      <main
        aria-labelledby="agreement-reader-title"
        className={cn(
          "mx-auto grid max-w-[1600px] gap-6 px-4 sm:px-6 lg:px-8",
          headerCollapsed ? "py-3" : "py-6 sm:py-8",
          !isMobile &&
            (leftOpen && rightOpen
              ? "lg:grid-cols-[280px_minmax(0,1fr)_320px]"
              : !leftOpen && rightOpen
                ? "lg:grid-cols-[44px_minmax(0,1fr)_320px]"
                : leftOpen && !rightOpen
                  ? "lg:grid-cols-[280px_minmax(0,1fr)_44px]"
                  : "lg:grid-cols-[44px_minmax(0,1fr)_44px]"),
        )}
      >
        {/* Left: TOC */}
        {!isMobile ? (
          <aside
            aria-label="Table of contents"
            className={cn(
              "sticky self-start",
              leftOpen ? "overflow-y-auto" : "overflow-visible",
            )}
            style={{
              top: `${stickyHeaderBottom + 12}px`,
              maxHeight: `calc(100vh - ${stickyHeaderBottom + 44}px)`,
            }}
          >
            {leftOpen ? (
              <div className="rounded-xl border border-border bg-card shadow-sm">
                <div className="flex items-center gap-2 border-b border-border px-4 py-3">
                  <ListTree
                    className="h-4 w-4 text-muted-foreground"
                    aria-hidden="true"
                  />
                  <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Contents
                  </span>
                  <button
                    type="button"
                    onClick={() => setLeftOpen(false)}
                    aria-label="Collapse contents"
                    className="ml-auto inline-flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    <ChevronLeft className="h-4 w-4" aria-hidden="true" />
                  </button>
                </div>
                <TableOfContents
                  xmlContent={agreement.xml}
                  targetSectionUuid={highlightedSection ?? undefined}
                  onSectionClick={scrollToSection}
                  onAnchorClick={scrollToAnchor}
                  scrollable={false}
                />
              </div>
            ) : (
              <button
                type="button"
                onClick={() => setLeftOpen(true)}
                aria-label="Expand contents"
                className="flex w-11 flex-col items-center gap-2 rounded-xl border border-border bg-card py-3 text-muted-foreground shadow-sm transition-colors hover:border-border hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <ChevronRight className="h-4 w-4" aria-hidden="true" />
                <ListTree className="h-4 w-4" aria-hidden="true" />
                <span
                  className="text-[10px] font-semibold uppercase tracking-wide"
                  style={{ writingMode: "vertical-rl" }}
                >
                  Contents
                </span>
              </button>
            )}
          </aside>
        ) : null}

        {/* Center: document */}
        <div className="min-w-0">
          <article
            className="rounded-lg border border-border bg-card shadow-sm"
            aria-labelledby="agreement-reader-title"
          >
            <div
              ref={contentRef}
              className="agreement-reader-content max-h-none overflow-visible p-4 sm:p-8 lg:px-10"
            >
              <XMLRenderer
                xmlContent={agreement.xml}
                mode="agreement"
                highlightedSection={highlightedSection}
                isMobile={isMobile}
                className={cn(
                  "leading-relaxed text-foreground break-words",
                  agreementTextSizeClass,
                )}
                showBodyOnly={showBodyOnly}
              />
            </div>
          </article>
        </div>

        {/* Right: Search + Details */}
        {!isMobile ? (
          <aside
            aria-label="Search and details"
            className={cn(
              "sticky self-start",
              rightOpen ? "overflow-y-auto" : "overflow-visible",
            )}
            style={{
              top: `${stickyHeaderBottom + 12}px`,
              maxHeight: `calc(100vh - ${stickyHeaderBottom + 44}px)`,
            }}
          >
            {rightOpen ? (
              sectionIndexLoading ? (
                <div className="rounded-xl border border-border bg-card p-4 shadow-sm">
                  <div className="flex h-32 items-center justify-center">
                    <LoadingSpinner aria-label="Loading agreement index" />
                  </div>
                </div>
              ) : sectionIndexError ? (
                <Alert variant="destructive">
                  <AlertTitle>Section index unavailable</AlertTitle>
                  <AlertDescription>{sectionIndexError}</AlertDescription>
                </Alert>
              ) : (
                <div className="rounded-xl border border-border bg-card shadow-sm">
                  <Tabs
                    value={sidebarTab}
                    onValueChange={(value) =>
                      setSidebarTab(value === "details" ? "details" : "search")
                    }
                  >
                    <div className="flex items-center gap-2 border-b border-border p-3">
                      <TabsList className="grid flex-1 grid-cols-2">
                        <TabsTrigger value="search">Search</TabsTrigger>
                        <TabsTrigger value="details">Details</TabsTrigger>
                      </TabsList>
                      <button
                        type="button"
                        onClick={() => setRightOpen(false)}
                        aria-label="Collapse panel"
                        className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                      >
                        <ChevronRight className="h-4 w-4" aria-hidden="true" />
                      </button>
                    </div>
                    <TabsContent value="search" className="m-0 p-4">
                      {readerSearch}
                    </TabsContent>
                    <TabsContent value="details" className="m-0 p-4">
                      {readerDetails}
                    </TabsContent>
                  </Tabs>
                </div>
              )
            ) : (
              <button
                type="button"
                onClick={() => setRightOpen(true)}
                aria-label="Expand panel"
                className="flex w-11 flex-col items-center gap-2 rounded-xl border border-border bg-card py-3 text-muted-foreground shadow-sm transition-colors hover:border-border hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <ChevronLeft className="h-4 w-4" aria-hidden="true" />
                <Search className="h-4 w-4" aria-hidden="true" />
                <span
                  className="text-[10px] font-semibold uppercase tracking-wide"
                  style={{ writingMode: "vertical-rl" }}
                >
                  Search &amp; details
                </span>
              </button>
            )}
          </aside>
        ) : null}
      </main>
    </div>
  );
}
