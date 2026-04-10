import { type ReactNode, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { PageShell } from "@/components/PageShell";
import brandLinks from "@branding/links.json";
import { useTaxonomy } from "@/hooks/use-taxonomy";
import { ArrowRight, Copy, Folder, Layers, Search, Tag } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
} from "@/components/ui/card";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { logger } from "@/lib/logger";
import {
  buildTaxonomyEntries,
  buildTaxonomySearchEntries,
  filterTaxonomySearchEntries,
  normalizeForTaxonomySearch,
  type TaxonomySearchEntry,
} from "@/lib/taxonomy-search";

type TaxonomyTab = "main" | "tax";

export default function Taxonomy() {
  const docsUrl = import.meta.env.DEV ? "http://localhost:3001" : brandLinks.docsSiteUrl;
  const [searchParams, setSearchParams] = useSearchParams();
  const currentTab = searchParams.get("tab") === "tax" ? "tax" : "main";
  const { taxonomyTree, isLoading, error } = useTaxonomy({ kind: currentTab });
  const [searchQuery, setSearchQuery] = useState("");
  const [hasActivatedSearch, setHasActivatedSearch] = useState(false);
  const [openLevel1, setOpenLevel1] = useState<string[]>([]);
  const [openLevel2ByParent, setOpenLevel2ByParent] = useState<
    Record<string, string[]>
  >({});
  const [highlightedId, setHighlightedId] = useState<string | null>(null);
  const highlightTimerRef = useRef<number | null>(null);
  const endpointPath =
    currentTab === "tax" ? "/v1/taxonomy/tax-clauses" : "/v1/taxonomy";
  const overviewTitle =
    currentTab === "tax" ? "Tax Taxonomy Overview" : "Taxonomy Overview";
  const treeTitle = currentTab === "tax" ? "Tax Taxonomy Tree" : "Taxonomy Tree";
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
    const depthRank = (entry: TaxonomySearchEntry) => {
      if (entry.l3) {
        return 2;
      }
      if (entry.l2) {
        return 1;
      }
      return 0;
    };
    results.sort((a, b) => {
      const level = depthRank(a) - depthRank(b);
      if (level !== 0) {
        return level;
      }
      const l1 = a.l1.localeCompare(b.l1, "en", { sensitivity: "base" });
      if (l1 !== 0) {
        return l1;
      }
      const l2 = (a.l2 ?? "").localeCompare(b.l2 ?? "", "en", {
        sensitivity: "base",
      });
      if (l2 !== 0) {
        return l2;
      }
      return (a.l3 ?? "").localeCompare(b.l3 ?? "", "en", {
        sensitivity: "base",
      });
    });
    return results;
  }, [searchResults]);

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
      const matchIndex = normalizedLabel.indexOf(
        normalizedQueryValue,
        searchStart,
      );
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
        parts.push(
          <span key={`text-${index}`}>{label.slice(cursor, range.start)}</span>,
        );
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

  const getResultLevel = (result: TaxonomySearchEntry) => {
    if (result.l3) {
      return "L3";
    }
    if (result.l2) {
      return "L2";
    }
    return "L1";
  };

  const getResultLevelStyles = (level: string) => {
    switch (level) {
      case "L3":
        return "border-[#2664EB] bg-[#2664EB] text-white";
      case "L2":
        return "border-[#19439E] bg-[#19439E] text-white";
      default:
        return "border-[#0E265A] bg-[#0E265A] text-white";
    }
  };

  const addUnique = (items: string[], value: string) =>
    items.includes(value) ? items : [...items, value];

  const handleResultClick = (result: TaxonomySearchEntry) => {
    const level1Id = result.l1Id;
    const level2Id = result.l2Id;
    const level3Id = result.l3Id;
    const targetId = level3Id
      ? `taxonomy-l3-${level3Id}`
      : level2Id
        ? `taxonomy-l2-${level2Id}`
        : `taxonomy-l1-${level1Id}`;

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

  const totals = useMemo(() => {
    const l1 = taxonomyEntries.length;
    const l2 = taxonomyEntries.reduce(
      (sum, entry) => sum + entry.children.length,
      0,
    );
    const l3 = taxonomyEntries.reduce(
      (sum, entry) =>
        sum +
        entry.children.reduce(
          (childSum, child) => childSum + child.children.length,
          0,
        ),
      0,
    );
    return { l1, l2, l3 };
  }, [taxonomyEntries]);

  const copyToClipboard = async (value: string) => {
    try {
      await navigator.clipboard.writeText(value);
    } catch {
      logger.warn("Clipboard write failed.");
    }
  };

  const renderCopyControl = (value: string, label: string) => (
    <button
      type="button"
      onClick={(event) => {
        event.stopPropagation();
        void copyToClipboard(value);
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          event.stopPropagation();
          void copyToClipboard(value);
        }
      }}
      className="inline-flex min-h-8 min-w-8 items-center justify-center rounded-md border border-border/50 bg-background/80 text-foreground/80 transition-colors hover:bg-accent/60 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
      aria-label={label}
      title={label}
    >
      <Copy className="h-3.5 w-3.5" aria-hidden="true" />
    </button>
  );

  const handleTabChange = (value: string) => {
    const nextTab: TaxonomyTab = value === "tax" ? "tax" : "main";
    const nextParams = new URLSearchParams(searchParams);
    if (nextTab === "main") {
      nextParams.delete("tab");
    } else {
      nextParams.set("tab", nextTab);
    }
    setSearchParams(nextParams, { replace: true });
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

  return (
    <PageShell
      size="xl"
      title="Taxonomy"
    >
      <div className="mb-4">
        <div className="rounded-lg bg-muted/20 pb-3 pt-0 text-sm text-foreground/80 sm:text-base">
          Pull the latest taxonomy via the{" "}
          <span className="font-mono text-sm text-foreground">{endpointPath}</span>{" "}
          API route. See the{" "}
          <a
            href={docsUrl}
            className="text-primary underline underline-offset-2"
          >
            Docs
          </a>{" "}
          for usage details.
        </div>
      </div>

      <div className="grid gap-6">
        <section aria-label="Taxonomy mode">
          <Tabs
            value={currentTab}
            onValueChange={handleTabChange}
            className="space-y-0"
          >
            <TabsList className="grid h-auto w-full grid-cols-2">
              <TabsTrigger value="main">Main</TabsTrigger>
              <TabsTrigger value="tax">Tax</TabsTrigger>
            </TabsList>
          </Tabs>
        </section>

        <section aria-labelledby="taxonomy-search">
          <div className="rounded-lg border border-border/60 bg-card p-4 shadow-sm">
            <Label
              id="taxonomy-search"
              htmlFor="taxonomy-search-input"
              className="text-sm font-semibold text-foreground"
            >
              Find a clause class
            </Label>
            <div className="relative mt-2">
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
                placeholder="Search clause types or taxonomy IDs"
                className="pl-9"
                autoComplete="off"
              />
            </div>
            {hasQuery && (
              <div
                className="mt-4 rounded-lg border border-border/60 bg-muted/20 p-3"
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
                  <div className="text-sm text-foreground/80">
                    No matching clause types yet.
                  </div>
                ) : (
                  <div className="space-y-2">
                    <div className="text-xs font-semibold uppercase tracking-wide text-foreground/80">
                      {searchResults.length}{" "}
                      {searchResults.length === 1 ? "result" : "results"}
                    </div>
                    <ul className="grid gap-2 sm:grid-cols-2">
                      {orderedResults.map((result) => (
                        <li
                          key={result.id}
                          className="rounded-md border border-border/60 bg-card shadow-sm"
                        >
                          <button
                            type="button"
                            onClick={() => handleResultClick(result)}
                            className="flex w-full items-start gap-2 rounded-md px-3 py-2 text-left text-sm text-foreground transition hover:bg-muted/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                            aria-label={`View clause: ${[result.l1, result.l2, result.l3].filter(Boolean).join(" > ")}`}
                          >
                            <Badge
                              variant="outline"
                              className={`shrink-0 min-w-[2.5rem] justify-center text-[11px] ${getResultLevelStyles(
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
                                <div className="ml-1 mt-0.5 border-l border-foreground/35 pl-3">
                                  <div className="relative leading-snug text-foreground/80">
                                    <span
                                      aria-hidden="true"
                                      className="absolute -left-3 top-1/2 h-px w-2 -translate-y-1/2 bg-foreground/35"
                                    />
                                    <span>{renderHighlighted(result.l2)}</span>
                                  </div>
                                  {result.l3 && (
                                    <div className="ml-3 mt-0.5 border-l border-foreground/30 pl-3">
                                      <div className="relative leading-snug">
                                        <span
                                          aria-hidden="true"
                                          className="absolute -left-3 top-1/2 h-px w-2 -translate-y-1/2 bg-foreground/30"
                                        />
                                        <span>{renderHighlighted(result.l3)}</span>
                                      </div>
                                    </div>
                                  )}
                                </div>
                              )}
                              <div className="font-mono text-xs text-foreground/75">
                                {renderHighlighted(result.id)}
                              </div>
                            </div>
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        </section>

        <section
          aria-labelledby="taxonomy-overview"
          aria-describedby="taxonomy-overview-desc"
        >
          <Card className="border-border/60 bg-card">
            <CardHeader>
              <h2
                id="taxonomy-overview"
                className="text-xl font-semibold leading-none tracking-tight"
              >
                {overviewTitle}
              </h2>
              <CardDescription id="taxonomy-overview-desc">
                Expand a level to see how categories roll up to standard IDs and
                child labels.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div
                  className="grid gap-4 sm:grid-cols-3"
                  role="status"
                  aria-live="polite"
                >
                  <Skeleton className="h-20 w-full" />
                  <Skeleton className="h-20 w-full" />
                  <Skeleton className="h-20 w-full" />
                </div>
              ) : (
                <div className="grid gap-3 rounded-lg border border-border/60 bg-muted/20 p-4 sm:grid-cols-[1fr_auto_1fr_auto_1fr] sm:items-center">
                  <div className="rounded-lg border border-border/60 bg-card p-4">
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-foreground/80">
                      <Folder className="h-4 w-4" aria-hidden="true" />
                      Level 1 Categories
                    </div>
                    <div className="mt-2 text-4xl font-bold text-foreground">
                      {totals.l1}
                    </div>
                  </div>
                  <div className="hidden justify-center text-muted-foreground sm:flex">
                    <ArrowRight className="h-4 w-4" aria-hidden="true" />
                  </div>
                  <div className="rounded-lg border border-border/60 bg-card p-4">
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-foreground/80">
                      <Layers className="h-4 w-4" aria-hidden="true" />
                      Level 2 Groups
                    </div>
                    <div className="mt-2 text-4xl font-bold text-foreground">
                      {totals.l2}
                    </div>
                  </div>
                  <div className="hidden justify-center text-muted-foreground sm:flex">
                    <ArrowRight className="h-4 w-4" aria-hidden="true" />
                  </div>
                  <div className="rounded-lg border border-border/60 bg-card p-4">
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-foreground/80">
                      <Tag className="h-4 w-4" aria-hidden="true" />
                      Level 3 Types
                    </div>
                    <div className="mt-2 text-4xl font-bold text-foreground">
                      {totals.l3}
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </section>

        {error && (
          <Alert variant="destructive">
            <AlertTitle>Taxonomy unavailable</AlertTitle>
            <AlertDescription>
              {error} Please refresh the page to try again.
            </AlertDescription>
          </Alert>
        )}

        <section aria-labelledby="taxonomy-tree">
          <div className="flex items-center justify-between">
            <h2
              id="taxonomy-tree"
              className="text-xl font-semibold leading-none tracking-tight"
            >
              {treeTitle}
            </h2>
          </div>

          <div className="mt-4" aria-busy={isLoading}>
            {isLoading ? (
              <div className="space-y-4" role="status" aria-live="polite">
                <Skeleton className="h-24 w-full" />
                <Skeleton className="h-24 w-full" />
                <Skeleton className="h-24 w-full" />
              </div>
            ) : taxonomyEntries.length === 0 ? (
              <Card className="border-border/60 bg-card">
                <CardContent className="py-6 text-sm text-foreground/80">
                  No taxonomy entries are available right now.
                </CardContent>
              </Card>
            ) : (
              <Accordion
                type="multiple"
                value={openLevel1}
                onValueChange={setOpenLevel1}
                className="space-y-4"
              >
                {taxonomyEntries.map((entry) => (
                  <AccordionItem
                    key={entry.id}
                    value={entry.id}
                    id={`taxonomy-l1-${entry.id}`}
                    className={`rounded-lg border border-border/60 bg-card px-0 transition-all duration-300 hover:-translate-y-1 hover:shadow-md ${
                      highlightedId === `taxonomy-l1-${entry.id}`
                        ? "ring-2 ring-primary/30"
                        : ""
                    }`}
                  >
                    <AccordionTrigger
                      headingLevel="h2"
                      className="px-6 py-4 text-left text-xl font-semibold hover:no-underline"
                    >
                      <div className="flex w-full flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                        <div>
                          <div className="text-xl font-semibold text-foreground">
                            {entry.label}
                          </div>
                          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-foreground/80">
                            <span className="inline-flex items-center gap-1.5 rounded-full border border-border/60 bg-muted/40 px-2 py-0.5 text-xs text-foreground/80">
                              <span className="font-mono text-[11px] text-foreground/75">
                                {entry.id}
                              </span>
                              {renderCopyControl(entry.id, "Copy level 1 ID")}
                            </span>
                            <span>{entry.l2Count} groups</span>
                            <span>{entry.l3Count} types</span>
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge className="min-w-[5.5rem] justify-center rounded-full border border-primary/20 bg-primary/10 text-xs text-primary hover:bg-primary/10">
                            {entry.l2Count} Groups
                          </Badge>
                          <Badge className="min-w-[5.5rem] justify-center rounded-full border border-primary/20 bg-primary/10 text-xs text-primary hover:bg-primary/10">
                            {entry.l3Count} Types
                          </Badge>
                        </div>
                      </div>
                    </AccordionTrigger>
                    <AccordionContent className="px-6 pb-6 pt-0 transition-all duration-300 data-[state=closed]:animate-[accordion-up_0.3s_ease-out] data-[state=open]:animate-[accordion-down_0.3s_ease-out]">
                      <div className="rounded-lg border border-border/60 bg-muted/20 p-4">
                        <Accordion
                          type="multiple"
                          value={openLevel2ByParent[entry.id] ?? []}
                          onValueChange={(value) => {
                            setOpenLevel2ByParent((prev) => ({
                              ...prev,
                              [entry.id]: value,
                            }));
                          }}
                          className="space-y-3"
                        >
                          {entry.children.map((child) => (
                            <AccordionItem
                              key={child.id}
                              value={child.id}
                              id={`taxonomy-l2-${child.id}`}
                              className={`rounded-lg border border-border/60 bg-muted/20 transition-all duration-300 hover:-translate-y-1 hover:shadow-md ${
                                highlightedId === `taxonomy-l2-${child.id}`
                                  ? "ring-2 ring-primary/30"
                                  : ""
                              }`}
                            >
                              <AccordionTrigger
                                headingLevel="h3"
                                className="px-5 py-3 text-left text-sm font-semibold hover:no-underline"
                              >
                                <div className="flex w-full flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                                  <div>
                                    <div className="text-sm font-semibold text-foreground">
                                      {child.label}
                                    </div>
                                  </div>
                                  <Badge className="min-w-[5.5rem] justify-center rounded-full border border-primary/20 bg-primary/10 text-xs text-primary hover:bg-primary/10">
                                    {child.children.length} Types
                                  </Badge>
                                </div>
                              </AccordionTrigger>
                              <AccordionContent className="px-5 pb-4 pt-0 transition-all duration-300 data-[state=closed]:animate-[accordion-up_0.3s_ease-out] data-[state=open]:animate-[accordion-down_0.3s_ease-out]">
                                <div className="mb-3">
                                  <span className="inline-flex items-center gap-1.5 rounded-full border border-border/60 bg-muted/40 px-2 py-0.5 text-xs text-foreground/80">
                                    <span className="font-mono text-[11px] text-foreground/75">
                                      {child.id}
                                    </span>
                                    {renderCopyControl(child.id, "Copy level 2 ID")}
                                  </span>
                                </div>
                                <ul className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                                  {child.children.map((leaf) => (
                                    <li
                                      key={leaf.id}
                                      id={`taxonomy-l3-${leaf.id}`}
                                      className={`group rounded-md border border-border/60 bg-card px-3 py-3 shadow-sm transition-all duration-300 hover:-translate-y-1 hover:border-primary/60 hover:shadow-md ${
                                        highlightedId === `taxonomy-l3-${leaf.id}`
                                          ? "ring-2 ring-primary/30"
                                          : ""
                                      }`}
                                    >
                                      <div className="text-sm font-medium text-foreground">
                                        {leaf.label}
                                      </div>
                                      <div className="mt-1 flex items-center gap-1.5 text-xs text-foreground/80">
                                        <span className="font-mono text-xs text-foreground/75">
                                          {leaf.id}
                                        </span>
                                        {renderCopyControl(leaf.id, "Copy level 3 ID")}
                                      </div>
                                    </li>
                                  ))}
                                </ul>
                              </AccordionContent>
                            </AccordionItem>
                          ))}
                        </Accordion>
                      </div>
                    </AccordionContent>
                  </AccordionItem>
                ))}
              </Accordion>
            )}
          </div>
        </section>
      </div>
    </PageShell>
  );
}
