import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import * as AccordionPrimitive from "@radix-ui/react-accordion";
import { ChevronDown, ListTree, Search } from "lucide-react";

import { PageShell } from "@/components/PageShell";
import { EmptyState, ErrorState } from "@/components/StateMessage";
import { TaxonomyExportButtons } from "@/components/TaxonomyExportButtons";
import { TaxonomyNodePanel } from "@/components/taxonomy/TaxonomyNodePanel";
import {
  countNoun,
  formatCount,
  taxonomySearchHref,
} from "@/components/taxonomy/taxonomy-nodes";
import { Accordion, AccordionContent, AccordionItem } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { CopyButton } from "@/components/ui/copy-button";
import { Eyebrow } from "@/components/ui/eyebrow";
import { ExternalLink } from "@/components/ui/external-link";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { useMediaQuery } from "@/components/reader/use-media-query";
import { useIsMobile } from "@/hooks/use-mobile";
import { useTaxonomy, useTaxonomyCounts } from "@/hooks/use-taxonomy";
import { scheduleWhenBrowserIdle } from "@/lib/analytics";
import { getDocsUrl } from "@/lib/docs-url";
import type { TaxonomyKind } from "@/lib/query-keys";
import {
  buildTaxonomyEntries,
  buildTaxonomySearchEntries,
  filterTaxonomySearchEntries,
  normalizeForTaxonomySearch,
  type TaxonomySearchEntry,
} from "@/lib/taxonomy-search";
import { formatTaxonomyTreeText, taxonomyTextFileName } from "@/lib/taxonomy-export";
import { cn } from "@/lib/utils";

type TaxonomyViewMode = "tile" | "tree";

const TAB_OPTIONS = [
  { value: "main", label: "Main" },
  { value: "tax", label: "Tax" },
] as const;

const VIEW_OPTIONS = [
  { value: "tree", label: "Tree" },
  { value: "tile", label: "Tile" },
] as const;

/** The detail panel docks beside the tree at `lg`; below it, it opens as a sheet. */
const DOCKED_PANEL_QUERY = "(min-width: 1024px)";

function useDockedPanel(): boolean {
  // Prerendered HTML has the docked panel; phones switch to the sheet after
  // hydration instead of mismatching on the first render.
  return useMediaQuery(DOCKED_PANEL_QUERY, true);
}

/**
 * Accordion header whose trigger is a full-row overlay button instead of a
 * wrapper around the row content. This keeps interactive children (the
 * copy-ID buttons) out of the trigger's DOM subtree — a <button> may not
 * contain another <button>. Row content sits above the overlay with
 * pointer-events disabled, except for controls that opt back in via
 * `pointer-events-auto`.
 */
function TileAccordionHeader({
  ariaLabel,
  children,
  className,
  headingLevel: HeadingTag,
}: {
  ariaLabel: string;
  children: ReactNode;
  className?: string;
  headingLevel: "h3" | "h4";
}) {
  return (
    <AccordionPrimitive.Header asChild>
      <HeadingTag className="relative flex">
        <AccordionPrimitive.Trigger
          aria-label={ariaLabel}
          className="absolute inset-0 rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 [&[data-state=open]+div_.taxonomy-accordion-chevron]:rotate-180"
        />
        <div
          className={cn(
            "pointer-events-none relative flex w-full items-center justify-between gap-4 text-left",
            className,
          )}
        >
          <div className="min-w-0 flex-1">{children}</div>
          <ChevronDown
            className="taxonomy-accordion-chevron h-4 w-4 shrink-0 transition-transform duration-200"
            aria-hidden="true"
          />
        </div>
      </HeadingTag>
    </AccordionPrimitive.Header>
  );
}

export default function Taxonomy() {
  const docsUrl = getDocsUrl();
  const [searchParams, setSearchParams] = useSearchParams();
  const currentTab: TaxonomyKind = searchParams.get("tab") === "tax" ? "tax" : "main";
  const selectedId = searchParams.get("node");
  const { taxonomyTree, isLoading, error } = useTaxonomy({ kind: currentTab });
  const { counts } = useTaxonomyCounts({ kind: currentTab });
  const [searchQuery, setSearchQuery] = useState("");
  const isMobile = useIsMobile();
  const isPanelDocked = useDockedPanel();
  const [hasActivatedSearch, setHasActivatedSearch] = useState(false);
  const [openLevel1, setOpenLevel1] = useState<string[]>([]);
  const [openLevel2ByParent, setOpenLevel2ByParent] = useState<Record<string, string[]>>(
    {},
  );
  const [viewMode, setViewMode] = useState<TaxonomyViewMode>("tree");
  const [highlightedId, setHighlightedId] = useState<string | null>(null);
  const [shouldRenderTree, setShouldRenderTree] = useState(false);
  const highlightTimerRef = useRef<number | null>(null);
  const endpointPath = currentTab === "tax" ? "/v1/taxonomy/tax-clauses" : "/v1/taxonomy";
  const endpointDocsUrl =
    currentTab === "tax"
      ? `${docsUrl}/docs/pandects/get-tax-clause-taxonomy`
      : `${docsUrl}/docs/pandects/get-taxonomy`;
  const treeTitle = currentTab === "tax" ? "Tax Taxonomy Tree" : "Taxonomy Tree";
  const taxonomySearchLabel =
    currentTab === "tax" ? "Find a clause class" : "Find a section class";
  const taxonomySearchPlaceholder =
    currentTab === "tax"
      ? "Search clause types or taxonomy IDs"
      : "Search section types or taxonomy IDs";
  const emptySearchMessage =
    currentTab === "tax" ? "No matching tax clause types yet." : "No matching section types yet.";
  const taxonomyEntries = useMemo(
    () => (taxonomyTree ? buildTaxonomyEntries(taxonomyTree) : []),
    [taxonomyTree],
  );
  const flattenedEntries = useMemo(
    () =>
      hasActivatedSearch && taxonomyEntries.length > 0
        ? buildTaxonomySearchEntries(taxonomyEntries)
        : [],
    [hasActivatedSearch, taxonomyEntries],
  );
  const normalizedQueryValue = normalizeForTaxonomySearch(searchQuery);
  const hasQuery = normalizedQueryValue.length > 0;
  const searchResults = useMemo(() => {
    if (!hasQuery || flattenedEntries.length === 0) {
      return [];
    }
    return filterTaxonomySearchEntries(flattenedEntries, searchQuery);
  }, [flattenedEntries, hasQuery, searchQuery]);
  const orderedResults = useMemo(() => {
    const results = [...searchResults];
    const depthRank = (entry: TaxonomySearchEntry) => (entry.l3 ? 2 : entry.l2 ? 1 : 0);
    results.sort((a, b) => {
      const level = depthRank(a) - depthRank(b);
      if (level !== 0) return level;
      const l1 = a.l1.localeCompare(b.l1, "en", { sensitivity: "base" });
      if (l1 !== 0) return l1;
      const l2 = (a.l2 ?? "").localeCompare(b.l2 ?? "", "en", { sensitivity: "base" });
      if (l2 !== 0) return l2;
      return (a.l3 ?? "").localeCompare(b.l3 ?? "", "en", { sensitivity: "base" });
    });
    return results;
  }, [searchResults]);

  useEffect(() => {
    return () => {
      if (highlightTimerRef.current) {
        window.clearTimeout(highlightTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (isLoading || taxonomyEntries.length === 0) {
      setShouldRenderTree(false);
      return;
    }
    if (hasQuery) {
      setShouldRenderTree(true);
      return;
    }
    const cancelDeferredTree = scheduleWhenBrowserIdle(() => {
      setShouldRenderTree(true);
    }, 1800);
    return cancelDeferredTree;
  }, [hasQuery, isLoading, taxonomyEntries.length]);

  const renderHighlighted = (label: string) => {
    if (!hasQuery) {
      return label;
    }

    let normalizedLabel = "";
    const indexMap: number[] = [];
    for (let i = 0; i < label.length; i += 1) {
      const char = label[i];
      if (/[a-z0-9]/i.test(char)) {
        normalizedLabel += char.toLowerCase();
        indexMap.push(i);
      }
    }

    if (!normalizedLabel) {
      return label;
    }

    const ranges: Array<{ start: number; end: number }> = [];
    let searchStart = 0;
    while (true) {
      const matchIndex = normalizedLabel.indexOf(normalizedQueryValue, searchStart);
      if (matchIndex === -1) {
        break;
      }
      const start = indexMap[matchIndex];
      const end = indexMap[matchIndex + normalizedQueryValue.length - 1] + 1;
      ranges.push({ start, end });
      searchStart = matchIndex + normalizedQueryValue.length;
    }

    if (ranges.length === 0) {
      return label;
    }

    const parts: ReactNode[] = [];
    let cursor = 0;
    ranges.forEach((range, index) => {
      if (range.start > cursor) {
        parts.push(<span key={`text-${index}`}>{label.slice(cursor, range.start)}</span>);
      }
      parts.push(
        <span
          key={`highlight-${index}`}
          className="rounded-sm bg-primary/10 font-semibold text-foreground"
        >
          {label.slice(range.start, range.end)}
        </span>,
      );
      cursor = range.end;
    });
    if (cursor < label.length) {
      parts.push(<span key="text-tail">{label.slice(cursor)}</span>);
    }
    return parts;
  };

  const getResultLevel = (result: TaxonomySearchEntry) =>
    result.l3 ? "L3" : result.l2 ? "L2" : "L1";

  // Depth encoded as darkness steps off the primary token: L3 is primary,
  // L2 and L1 mix progressively more black into it.
  const getResultLevelStyles = (level: string) => {
    switch (level) {
      case "L3":
        return "border-transparent bg-primary text-white";
      case "L2":
        return "border-transparent bg-[color-mix(in_srgb,hsl(var(--primary))_72%,black)] text-white";
      default:
        return "border-transparent bg-[color-mix(in_srgb,hsl(var(--primary))_45%,black)] text-white";
    }
  };

  const updateParams = (mutate: (params: URLSearchParams) => void) => {
    const nextParams = new URLSearchParams(searchParams);
    mutate(nextParams);
    setSearchParams(nextParams, { replace: true });
  };

  const selectNode = (standardId: string | null) => {
    updateParams((params) => {
      if (standardId) params.set("node", standardId);
      else params.delete("node");
    });
  };

  const addUnique = (items: string[], value: string) =>
    items.includes(value) ? items : [...items, value];

  const handleResultClick = (result: TaxonomySearchEntry) => {
    setShouldRenderTree(true);
    const level1Id = result.l1Id;
    const level2Id = result.l2Id;
    const level3Id = result.l3Id;
    const targetId = level3Id
      ? `taxonomy-l3-${level3Id}`
      : level2Id
        ? `taxonomy-l2-${level2Id}`
        : `taxonomy-l1-${level1Id}`;

    selectNode(level3Id ?? level2Id ?? level1Id);
    setOpenLevel1((prev) => addUnique(prev, level1Id));
    if (level2Id) {
      setOpenLevel2ByParent((prev) => ({
        ...prev,
        [level1Id]: addUnique(prev[level1Id] ?? [], level2Id),
      }));
    }

    if (highlightTimerRef.current) {
      window.clearTimeout(highlightTimerRef.current);
    }
    setHighlightedId(targetId);
    highlightTimerRef.current = window.setTimeout(() => {
      setHighlightedId(null);
      highlightTimerRef.current = null;
    }, 2500);

    window.setTimeout(() => {
      const target = document.getElementById(targetId);
      if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    }, 25);
  };

  const highlightRingClass = (id: string) =>
    highlightedId === id ? "bg-primary/10 ring-2 ring-primary/35" : "ring-1 ring-transparent";

  const totals = useMemo(() => {
    const l2 = taxonomyEntries.reduce((sum, entry) => sum + entry.children.length, 0);
    const l3 = taxonomyEntries.reduce((sum, entry) => sum + entry.l3Count, 0);
    return { l1: taxonomyEntries.length, l2, l3 };
  }, [taxonomyEntries]);

  const handleTabChange = (nextTab: TaxonomyKind) => {
    updateParams((params) => {
      if (nextTab === "main") params.delete("tab");
      else params.set("tab", nextTab);
      params.delete("node");
    });
    setSearchQuery("");
    setHasActivatedSearch(false);
    setOpenLevel1([]);
    setOpenLevel2ByParent({});
    setHighlightedId(null);
    if (highlightTimerRef.current) {
      window.clearTimeout(highlightTimerRef.current);
      highlightTimerRef.current = null;
    }
  };

  const nodeCount = (id: string) =>
    counts ? (
      <span
        className="shrink-0 text-xs tabular-nums text-muted-foreground"
        aria-label={`${formatCount(counts[id] ?? 0)} ${countNoun(currentTab, counts[id] ?? 0)}`}
      >
        {formatCount(counts[id] ?? 0)}
      </span>
    ) : null;

  const searchLink = (id: string, label: string, className?: string) => (
    <Link
      to={taxonomySearchHref(currentTab, id)}
      aria-label={`Find ${label} in search`}
      title="Find in search"
      className={cn(
        "inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-accent/60 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        className,
      )}
    >
      <Search className="h-4 w-4" aria-hidden="true" />
    </Link>
  );

  /** One selectable row of the tree view. */
  const treeRow = (
    id: string,
    label: string,
    level: 1 | 2 | 3,
    domId: string,
  ) => {
    const isSelected = selectedId === id;
    return (
      <div
        id={domId}
        className={cn(
          "group flex items-center gap-2 rounded-lg pl-2 pr-1 transition-colors duration-700",
          isSelected ? "bg-primary/10" : "hover:bg-accent/60",
          highlightRingClass(domId),
        )}
      >
        <button
          type="button"
          onClick={() => selectNode(id)}
          aria-pressed={isSelected}
          className="flex min-h-9 min-w-0 flex-1 flex-wrap items-baseline gap-x-2 rounded-md py-1.5 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <span
            className={cn(
              "text-foreground",
              level === 1 ? "font-semibold" : level === 2 ? "font-medium" : "",
            )}
          >
            {label}
          </span>
          {!isMobile ? (
            <span className="font-mono text-[11px] text-muted-foreground">{id}</span>
          ) : null}
        </button>
        {nodeCount(id)}
        {level === 3 ? searchLink(id, label) : <span className="w-8 shrink-0" aria-hidden="true" />}
      </div>
    );
  };

  const panel = selectedId ? (
    <TaxonomyNodePanel
      key={`${currentTab}:${selectedId}`}
      kind={currentTab}
      standardId={selectedId}
      onSelectNode={selectNode}
      titleId="taxonomy-node-title"
    />
  ) : null;

  return (
    <PageShell
      layout="data"
      title="Taxonomy"
      description={
        <>
          Pull the latest taxonomy via the{" "}
          <span className="font-mono text-sm text-foreground">{endpointPath}</span> API
          route. See the <ExternalLink href={endpointDocsUrl}>Docs</ExternalLink> for usage
          details.
        </>
      }
      actions={
        <SegmentedControl
          aria-label="Taxonomy mode"
          value={currentTab}
          onValueChange={handleTabChange}
          options={TAB_OPTIONS}
        />
      }
    >
      <div className="lg:grid lg:grid-cols-[minmax(0,1fr)_360px] lg:gap-8">
        <div className="grid min-w-0 gap-6">
          <section aria-labelledby="taxonomy-controls" role="search">
            <Label id="taxonomy-controls" htmlFor="taxonomy-search-input" className="sr-only">
              {taxonomySearchLabel}
            </Label>
            <div className="relative">
              <Search
                className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
                aria-hidden="true"
              />
              <Input
                id="taxonomy-search-input"
                value={searchQuery}
                onChange={(event) => {
                  setHasActivatedSearch(true);
                  setSearchQuery(event.target.value);
                }}
                onFocus={() => setHasActivatedSearch(true)}
                placeholder={taxonomySearchPlaceholder}
                className="h-11 pl-10 pr-4 text-base md:text-sm"
                autoComplete="off"
              />
            </div>

            {hasQuery && (
              <div
                className="mt-3 rounded-xl border border-border bg-card p-3"
                aria-live="polite"
                aria-atomic="true"
              >
                {isLoading ? (
                  <div className="space-y-2" role="status" aria-live="polite">
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-5/6" />
                    <Skeleton className="h-4 w-3/4" />
                  </div>
                ) : searchResults.length === 0 ? (
                  <p className="text-sm text-foreground">{emptySearchMessage}</p>
                ) : (
                  <div className="space-y-2">
                    <p className="text-xs font-medium text-muted-foreground">
                      {searchResults.length} {searchResults.length === 1 ? "result" : "results"}
                    </p>
                    <ul className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                      {orderedResults.map((result) => (
                        <li key={result.id} className="rounded-lg border border-border bg-card">
                          <button
                            type="button"
                            onClick={() => handleResultClick(result)}
                            className="flex w-full items-start gap-2 rounded-lg px-3 py-2 text-left text-sm text-foreground transition-colors hover:bg-accent/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                            aria-label={`View clause: ${[result.l1, result.l2, result.l3].filter(Boolean).join(" > ")}`}
                          >
                            <Badge
                              variant="outline"
                              className={`min-w-[2.5rem] shrink-0 justify-center text-[11px] ${getResultLevelStyles(
                                getResultLevel(result),
                              )}`}
                            >
                              {getResultLevel(result)}
                            </Badge>
                            <div className="flex-1 space-y-0.5">
                              <div className="font-medium leading-snug">
                                {renderHighlighted(result.l1)}
                              </div>
                              {result.l2 && (
                                <div className="ml-1 mt-0.5 border-l border-border pl-3">
                                  <div className="leading-snug">{renderHighlighted(result.l2)}</div>
                                  {result.l3 && (
                                    <div className="ml-3 mt-0.5 border-l border-border pl-3 leading-snug">
                                      {renderHighlighted(result.l3)}
                                    </div>
                                  )}
                                </div>
                              )}
                              {!isMobile ? (
                                <div className="font-mono text-xs text-muted-foreground">
                                  {renderHighlighted(result.id)}
                                </div>
                              ) : null}
                            </div>
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </section>

          <section aria-labelledby="taxonomy-tree">
            <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
              <div className="space-y-1">
                <h2 id="taxonomy-tree" className="text-xl">
                  {viewMode === "tree" ? `${treeTitle} View` : treeTitle}
                </h2>
                {!isLoading && taxonomyEntries.length > 0 ? (
                  <p className="text-sm text-muted-foreground">
                    {totals.l1} categories · {totals.l2} groups · {totals.l3} types
                  </p>
                ) : null}
              </div>
              <div className="flex flex-wrap items-center gap-3 sm:justify-end">
                <TaxonomyExportButtons
                  // Remount on tab change so a "Copied" flash from the previous
                  // taxonomy can't linger over the one now loading, matching how
                  // handleTabChange clears the page's other transient state.
                  key={currentTab}
                  getText={() => formatTaxonomyTreeText(taxonomyEntries)}
                  fileName={taxonomyTextFileName(currentTab)}
                  disabled={isLoading || taxonomyEntries.length === 0}
                />
                <SegmentedControl<TaxonomyViewMode>
                  aria-label="Taxonomy view"
                  value={viewMode}
                  onValueChange={setViewMode}
                  options={VIEW_OPTIONS}
                />
              </div>
            </div>

            <div className="mt-4" aria-busy={isLoading}>
              {error ? (
                <ErrorState variant="card" error={error} />
              ) : isLoading || (taxonomyEntries.length > 0 && !shouldRenderTree) ? (
                <div className="min-h-[28rem] space-y-4" role="status" aria-live="polite">
                  <Skeleton className="h-32 w-full" />
                  <Skeleton className="h-32 w-full" />
                  <Skeleton className="h-32 w-full" />
                </div>
              ) : taxonomyEntries.length === 0 ? (
                <EmptyState
                  variant="card"
                  icon={ListTree}
                  title="No taxonomy entries are available right now."
                />
              ) : viewMode === "tile" ? (
                <Accordion
                  type="multiple"
                  value={openLevel1}
                  onValueChange={setOpenLevel1}
                  className="space-y-3"
                >
                  {taxonomyEntries.map((entry) => (
                    <AccordionItem
                      key={entry.id}
                      value={entry.id}
                      id={`taxonomy-l1-${entry.id}`}
                      className={`rounded-xl border border-border bg-card px-0 transition-shadow duration-700 ${highlightRingClass(
                        `taxonomy-l1-${entry.id}`,
                      )}`}
                    >
                      <TileAccordionHeader
                        headingLevel="h3"
                        ariaLabel={`Toggle ${entry.label}`}
                        className="px-5 py-4"
                      >
                        <div className="flex w-full flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                            <span className="text-lg font-semibold text-foreground">
                              {entry.label}
                            </span>
                            {!isMobile ? (
                              <span className="inline-flex items-center gap-1">
                                <span className="font-mono text-[13px] text-foreground/80">
                                  {entry.id}
                                </span>
                                <CopyButton
                                  value={entry.id}
                                  label="Copy level 1 ID"
                                  className="pointer-events-auto"
                                />
                              </span>
                            ) : null}
                          </div>
                          <div className="flex items-center gap-2">
                            <Badge variant="count">{entry.l2Count} Groups</Badge>
                            <Badge variant="count">{entry.l3Count} Types</Badge>
                          </div>
                        </div>
                      </TileAccordionHeader>
                      <AccordionContent className="px-5 pb-5 pt-0">
                        <Accordion
                          type="multiple"
                          value={openLevel2ByParent[entry.id] ?? []}
                          onValueChange={(value) => {
                            setOpenLevel2ByParent((prev) => ({ ...prev, [entry.id]: value }));
                          }}
                          className="space-y-2"
                        >
                          {entry.children.map((child) => (
                            <AccordionItem
                              key={child.id}
                              value={child.id}
                              id={`taxonomy-l2-${child.id}`}
                              className={`rounded-lg border border-border bg-muted/20 transition-shadow duration-700 ${highlightRingClass(
                                `taxonomy-l2-${child.id}`,
                              )}`}
                            >
                              <TileAccordionHeader
                                headingLevel="h4"
                                ariaLabel={`Toggle ${child.label}`}
                                className="px-4 py-3"
                              >
                                <div className="flex w-full flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                                  <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
                                    <span className="text-sm font-semibold text-foreground">
                                      {child.label}
                                    </span>
                                    {!isMobile ? (
                                      <span className="inline-flex items-center gap-1">
                                        <span className="font-mono text-[11px] text-foreground/80">
                                          {child.id}
                                        </span>
                                        <CopyButton
                                          value={child.id}
                                          label="Copy level 2 ID"
                                          className="pointer-events-auto"
                                        />
                                      </span>
                                    ) : null}
                                  </div>
                                  <Badge variant="count">{child.children.length} Types</Badge>
                                </div>
                              </TileAccordionHeader>
                              <AccordionContent className="px-4 pb-4 pt-0">
                                <ul className="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-3">
                                  {child.children.map((leaf) => {
                                    const isSelected = selectedId === leaf.id;
                                    return (
                                      <li
                                        key={leaf.id}
                                        id={`taxonomy-l3-${leaf.id}`}
                                        className={cn(
                                          "flex items-start gap-1 rounded-lg border bg-card p-1 transition-colors duration-700",
                                          isSelected ? "border-primary/60 bg-primary/5" : "border-border",
                                          highlightRingClass(`taxonomy-l3-${leaf.id}`),
                                        )}
                                      >
                                        <button
                                          type="button"
                                          onClick={() => selectNode(leaf.id)}
                                          aria-pressed={isSelected}
                                          className="min-w-0 flex-1 rounded-md px-2 py-1.5 text-left hover:bg-accent/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                                        >
                                          <span className="block text-sm font-medium text-foreground">
                                            {leaf.label}
                                          </span>
                                          <span className="mt-0.5 flex flex-wrap items-baseline gap-x-2 text-xs text-muted-foreground">
                                            {!isMobile ? (
                                              <span className="font-mono">{leaf.id}</span>
                                            ) : null}
                                            {counts ? (
                                              <span className="tabular-nums">
                                                {formatCount(counts[leaf.id] ?? 0)}{" "}
                                                {countNoun(currentTab, counts[leaf.id] ?? 0)}
                                              </span>
                                            ) : null}
                                          </span>
                                        </button>
                                        {searchLink(leaf.id, leaf.label)}
                                      </li>
                                    );
                                  })}
                                </ul>
                              </AccordionContent>
                            </AccordionItem>
                          ))}
                        </Accordion>
                      </AccordionContent>
                    </AccordionItem>
                  ))}
                </Accordion>
              ) : (
                <div className="rounded-xl border border-border bg-card p-2 sm:p-3">
                  {counts ? (
                    <div className="flex justify-end pb-1 pr-10" aria-hidden="true">
                      <Eyebrow>{currentTab === "tax" ? "Clauses" : "Sections"}</Eyebrow>
                    </div>
                  ) : null}
                  <ul className="space-y-3">
                    {taxonomyEntries.map((entry) => (
                      <li key={entry.id}>
                        {treeRow(entry.id, entry.label, 1, `taxonomy-l1-${entry.id}`)}
                        <ul className="ml-3 mt-1 space-y-1 border-l border-border pl-3">
                          {entry.children.map((child) => (
                            <li key={child.id}>
                              {treeRow(child.id, child.label, 2, `taxonomy-l2-${child.id}`)}
                              <ul className="ml-3 mt-0.5 space-y-0.5 border-l border-dashed border-border pl-3">
                                {child.children.map((leaf) => (
                                  <li key={leaf.id}>
                                    {treeRow(leaf.id, leaf.label, 3, `taxonomy-l3-${leaf.id}`)}
                                  </li>
                                ))}
                              </ul>
                            </li>
                          ))}
                        </ul>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </section>
        </div>

        {isPanelDocked ? (
          <aside aria-labelledby={panel ? "taxonomy-node-title" : undefined} className="hidden lg:block">
            <div className="sticky top-24 max-h-[calc(100vh-7rem)] overflow-y-auto rounded-xl border border-border bg-card p-5">
              {panel ?? (
                <EmptyState
                  variant="inline"
                  icon={ListTree}
                  title="Select a node"
                  body="Its path, corpus count, and an example appear here."
                />
              )}
            </div>
          </aside>
        ) : null}
      </div>

      {!isPanelDocked ? (
        <Sheet
          open={Boolean(selectedId)}
          onOpenChange={(open) => {
            if (!open) selectNode(null);
          }}
        >
          <SheetContent
            side="bottom"
            className="max-h-[85vh] overflow-y-auto rounded-t-2xl"
            aria-describedby={undefined}
          >
            <SheetTitle className="sr-only">Taxonomy node</SheetTitle>
            {panel}
          </SheetContent>
        </Sheet>
      ) : null}
    </PageShell>
  );
}
