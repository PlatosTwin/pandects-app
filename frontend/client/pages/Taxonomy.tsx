import { type ReactNode, useMemo, useRef, useState } from "react";
import { PageShell } from "@/components/PageShell";
import { useTaxonomy } from "@/hooks/use-taxonomy";
import type { ClauseTypeNode, ClauseTypeTree } from "@/lib/clause-types";
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

type ClauseTypeValue = ClauseTypeTree[keyof ClauseTypeTree];

type TaxonomyLevel3 = {
  label: string;
  id: string;
};

type TaxonomyLevel2 = {
  label: string;
  id: string;
  children: TaxonomyLevel3[];
};

type TaxonomyLevel1 = {
  label: string;
  id: string;
  children: TaxonomyLevel2[];
  l2Count: number;
  l3Count: number;
};

type SearchEntry = {
  id: string;
  l1: string;
  l2?: string;
  l3?: string;
  l1Id: string;
  l2Id?: string;
  l3Id?: string;
  l1Normalized: string;
  l2Normalized?: string;
  l3Normalized?: string;
};

const normalizeForSearch = (value: string) =>
  value.toLowerCase().replace(/[^a-z0-9]+/g, "");

const toNode = (value: ClauseTypeValue): ClauseTypeNode => value as ClauseTypeNode;

const buildTaxonomyEntries = (tree: ClauseTypeTree): TaxonomyLevel1[] =>
  Object.entries(tree).map(([l1Label, l1Value]) => {
    const l1Node = toNode(l1Value);
    const l2Tree = l1Node.children ?? {};
    const l2Entries = Object.entries(l2Tree).map(([l2Label, l2Value]) => {
      const l2Node = toNode(l2Value);
      const l3Tree = l2Node.children ?? {};
      const l3Entries = Object.entries(l3Tree).map(([l3Label, l3Value]) => {
        const l3Node = toNode(l3Value);
        return { label: l3Label, id: l3Node.id };
      });
      return { label: l2Label, id: l2Node.id, children: l3Entries };
    });
    const l3Count = l2Entries.reduce(
      (sum, entry) => sum + entry.children.length,
      0,
    );
    return {
      label: l1Label,
      id: l1Node.id,
      children: l2Entries,
      l2Count: l2Entries.length,
      l3Count,
    };
  });

export default function Taxonomy() {
  const { taxonomyTree, isLoading, error } = useTaxonomy({ fresh: true });
  const [searchQuery, setSearchQuery] = useState("");
  const [openLevel1, setOpenLevel1] = useState<string[]>([]);
  const [openLevel2ByParent, setOpenLevel2ByParent] = useState<
    Record<string, string[]>
  >({});
  const [highlightedId, setHighlightedId] = useState<string | null>(null);
  const highlightTimerRef = useRef<number | null>(null);
  const taxonomyEntries = useMemo(
    () => (taxonomyTree ? buildTaxonomyEntries(taxonomyTree) : []),
    [taxonomyTree],
  );
  const flattenedEntries = useMemo(() => {
    const entries: SearchEntry[] = [];
    taxonomyEntries.forEach((entry) => {
      entries.push({
        id: entry.id,
        l1: entry.label,
        l1Id: entry.id,
        l1Normalized: normalizeForSearch(entry.label),
      });
      entry.children.forEach((child) => {
        entries.push({
          id: child.id,
          l1: entry.label,
          l2: child.label,
          l1Id: entry.id,
          l2Id: child.id,
          l1Normalized: normalizeForSearch(entry.label),
          l2Normalized: normalizeForSearch(child.label),
        });
        child.children.forEach((leaf) => {
          entries.push({
            id: leaf.id,
            l1: entry.label,
            l2: child.label,
            l3: leaf.label,
            l1Id: entry.id,
            l2Id: child.id,
            l3Id: leaf.id,
            l1Normalized: normalizeForSearch(entry.label),
            l2Normalized: normalizeForSearch(child.label),
            l3Normalized: normalizeForSearch(leaf.label),
          });
        });
      });
    });
    return entries;
  }, [taxonomyEntries]);
  const normalizedQueryValue = normalizeForSearch(searchQuery);
  const hasQuery = normalizedQueryValue.length > 0;
  const searchResults = useMemo(() => {
    if (!hasQuery) {
      return [];
    }

    return flattenedEntries.filter(
      (entry) =>
        entry.l1Normalized.includes(normalizedQueryValue) ||
        entry.l2Normalized?.includes(normalizedQueryValue) ||
        entry.l3Normalized?.includes(normalizedQueryValue),
    );
  }, [flattenedEntries, hasQuery, normalizedQueryValue]);
  const orderedResults = useMemo(() => {
    const results = [...searchResults];
    const depthRank = (entry: SearchEntry) => {
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

  const getResultLevel = (result: SearchEntry) => {
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

  const handleResultClick = (result: SearchEntry) => {
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
      if (import.meta.env.DEV) {
        console.warn("Clipboard write failed.");
      }
    }
  };

  const renderCopyControl = (value: string, label: string) => (
    <span
      role="button"
      tabIndex={0}
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
      className="cursor-pointer rounded-sm opacity-0 transition-opacity group-hover:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
      aria-label={label}
      title={label}
    >
      <Copy className="h-3 w-3" aria-hidden="true" />
    </span>
  );

  return (
    <PageShell
      size="xl"
      title="Taxonomy"
    >
      <div className="mb-4">
        <div className="rounded-lg bg-muted/20 pb-3 pt-0 text-sm text-muted-foreground sm:text-base">
          Pull the latest taxonomy via the{" "}
          <span className="font-mono text-sm text-foreground">/v1/taxonomy</span>{" "}
          API route. See the{" "}
          <a href="/docs" className="text-primary underline underline-offset-2">
            Docs
          </a>{" "}
          for usage details.
        </div>
      </div>

      <div className="grid gap-6">
        <section aria-labelledby="taxonomy-search">
          <div className="rounded-lg border border-border/60 bg-card p-4 shadow-sm">
            <Label
              htmlFor="taxonomy-search-input"
              className="text-sm font-semibold text-foreground"
            >
              Find a clause
            </Label>
            <div className="relative mt-2">
              <Search
                className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground"
                aria-hidden="true"
              />
              <Input
                id="taxonomy-search-input"
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                placeholder="Search clause types"
                className="pl-9"
                autoComplete="off"
              />
            </div>
            {hasQuery && (
              <div className="mt-4 rounded-lg border border-border/60 bg-muted/20 p-3">
                {isLoading ? (
                  <div className="space-y-2" role="status" aria-live="polite">
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-5/6" />
                    <Skeleton className="h-4 w-3/4" />
                  </div>
                ) : searchResults.length === 0 ? (
                  <div className="text-sm text-muted-foreground">
                    No matching clause types yet.
                  </div>
                ) : (
                  <div className="space-y-2">
                    <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
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
                          >
                            <Badge
                              variant="outline"
                              className={`shrink-0 min-w-[2.5rem] justify-center text-[11px] ${getResultLevelStyles(
                                getResultLevel(result),
                              )}`}
                            >
                              {getResultLevel(result)}
                            </Badge>
                            <div className="flex flex-wrap items-center">
                              <span className="font-medium">
                                {renderHighlighted(result.l1)}
                              </span>
                              {result.l2 && (
                                <>
                                  <span className="mx-1 text-muted-foreground">
                                    {">"}
                                  </span>
                                  <span className="text-muted-foreground">
                                    {renderHighlighted(result.l2)}
                                  </span>
                                </>
                              )}
                              {result.l3 && (
                                <>
                                  <span className="mx-1 text-muted-foreground">
                                    {">"}
                                  </span>
                                  <span>{renderHighlighted(result.l3)}</span>
                                </>
                              )}
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
                Taxonomy Overview
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
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
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
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
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
                    <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
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
              Taxonomy Tree
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
                <CardContent className="py-6 text-sm text-muted-foreground">
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
                      className="px-6 py-4 text-left text-lg font-semibold hover:no-underline"
                    >
                      <div className="flex w-full flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                        <div>
                          <div className="text-lg font-semibold text-foreground">
                            {entry.label}
                          </div>
                          <div className="group mt-1 flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                            <span className="inline-flex items-center gap-1 rounded-full border border-border/60 bg-muted/40 px-2 py-0.5 font-mono text-xs text-muted-foreground/70">
                              {entry.id}
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
                                    <div className="group mt-1 text-xs text-muted-foreground">
                                      <span className="inline-flex items-center gap-1 font-mono text-xs text-muted-foreground/70">
                                        {child.id}
                                        {renderCopyControl(child.id, "Copy level 2 ID")}
                                      </span>
                                    </div>
                                  </div>
                                  <Badge className="min-w-[5.5rem] justify-center rounded-full border border-primary/20 bg-primary/10 text-xs text-primary hover:bg-primary/10">
                                    {child.children.length} Types
                                  </Badge>
                                </div>
                              </AccordionTrigger>
                              <AccordionContent className="px-5 pb-4 pt-0 transition-all duration-300 data-[state=closed]:animate-[accordion-up_0.3s_ease-out] data-[state=open]:animate-[accordion-down_0.3s_ease-out]">
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
                                      <div className="mt-1 flex items-center gap-1 text-xs text-muted-foreground">
                                        <span className="font-mono text-xs text-muted-foreground/70">
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
