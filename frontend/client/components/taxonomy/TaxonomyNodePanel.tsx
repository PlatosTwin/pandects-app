import { useMemo } from "react";
import { Link, useLocation } from "react-router-dom";
import { ChevronRight, Search } from "lucide-react";

import { EmptyState, ErrorState } from "@/components/StateMessage";
import { Button } from "@/components/ui/button";
import { CopyButton } from "@/components/ui/copy-button";
import { Eyebrow } from "@/components/ui/eyebrow";
import { Skeleton } from "@/components/ui/skeleton";
import {
  useTaxonomy,
  useTaxonomyCounts,
  useTaxonomyNodeExample,
  type TaxonomyKind,
} from "@/hooks/use-taxonomy";
import { cn } from "@/lib/utils";
import {
  TAXONOMY_LEVEL_NAMES,
  countNoun,
  formatCount,
  indexTaxonomyNodes,
  plainTextFromXml,
  sectionReaderHref,
  taxonomySearchHref,
  truncateAtWord,
} from "@/components/taxonomy/taxonomy-nodes";

const EXAMPLE_MAX_CHARS = 420;

export interface TaxonomyNodePanelProps {
  /** `main` = clause (section) taxonomy, `tax` = tax-clause taxonomy. */
  kind: TaxonomyKind;
  standardId: string;
  /**
   * Called when a parent or child node is chosen inside the panel. Without
   * it, the path and children render as plain text.
   */
  onSelectNode?: (standardId: string) => void;
  /** Heading level of the node title (default h2). */
  titleAs?: "h2" | "h3";
  /** Id for the title element, e.g. for `aria-labelledby`. */
  titleId?: string;
  className?: string;
}

/**
 * Detail panel for one taxonomy node: its path, corpus count, standard id, an
 * example section (or tax clause), and a link to search. Everything is
 * resolved from `standardId` through the shared taxonomy, counts, and example
 * caches, so it can render anywhere (the Taxonomy page, a Search popover).
 */
export function TaxonomyNodePanel({
  kind,
  standardId,
  onSelectNode,
  titleAs: TitleTag = "h2",
  titleId,
  className,
}: TaxonomyNodePanelProps) {
  const {
    taxonomyTree,
    isLoading: isTreeLoading,
    error: treeError,
  } = useTaxonomy({ kind });
  const { counts, isLoading: isCountsLoading } = useTaxonomyCounts({ kind });
  const nodes = useMemo(
    () => (taxonomyTree ? indexTaxonomyNodes(taxonomyTree) : null),
    [taxonomyTree],
  );
  const node = nodes?.get(standardId) ?? null;
  const count = counts ? (counts[standardId] ?? 0) : null;

  if (isTreeLoading) {
    return (
      <div
        className={cn("space-y-3", className)}
        role="status"
        aria-live="polite"
      >
        <span className="sr-only">Loading taxonomy node</span>
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-6 w-3/4" />
        <Skeleton className="h-20 w-full" />
      </div>
    );
  }
  if (treeError) {
    return (
      <ErrorState variant="inline" error={treeError} className={className} />
    );
  }
  if (!node) {
    return (
      <EmptyState
        variant="inline"
        title="Unknown taxonomy ID"
        body={<span className="font-mono">{standardId}</span>}
        className={className}
      />
    );
  }

  const nodeLink = (id: string, label: string) =>
    onSelectNode ? (
      <button
        type="button"
        onClick={() => onSelectNode(id)}
        className="rounded-sm text-left text-primary underline-offset-2 hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      >
        {label}
      </button>
    ) : (
      <span>{label}</span>
    );

  return (
    <div className={cn("space-y-5 text-sm", className)}>
      <div className="space-y-1.5">
        <Eyebrow>{TAXONOMY_LEVEL_NAMES[kind][node.level]}</Eyebrow>
        <TitleTag
          id={titleId}
          className="text-lg font-semibold leading-snug text-foreground"
        >
          {node.label}
        </TitleTag>
        {node.path.length > 0 ? (
          <nav aria-label="Parent path">
            <ol className="flex flex-wrap items-center gap-x-1 gap-y-0.5 text-muted-foreground">
              {node.path.map((ancestor, index) => (
                <li
                  key={ancestor.id}
                  className="inline-flex items-center gap-1"
                >
                  {index > 0 ? (
                    <ChevronRight
                      className="h-3.5 w-3.5 shrink-0"
                      aria-hidden="true"
                    />
                  ) : null}
                  {nodeLink(ancestor.id, ancestor.label)}
                </li>
              ))}
            </ol>
          </nav>
        ) : null}
      </div>

      <dl className="grid grid-cols-[auto_minmax(0,1fr)] items-center gap-x-4 gap-y-2">
        <dt className="text-muted-foreground">In corpus</dt>
        <dd className="font-medium tabular-nums text-foreground">
          {isCountsLoading ? (
            <Skeleton className="h-4 w-24" />
          ) : count === null ? (
            "Unavailable"
          ) : (
            `${formatCount(count)} ${countNoun(kind, count)}`
          )}
        </dd>
        <dt className="text-muted-foreground">Standard ID</dt>
        <dd className="flex min-w-0 items-center gap-1">
          <span className="truncate font-mono text-xs text-foreground">
            {node.id}
          </span>
          <CopyButton
            value={node.id}
            label="Copy standard ID"
            className="-my-1"
          />
        </dd>
      </dl>

      <Button asChild size="sm" className="w-full sm:w-auto">
        <Link to={taxonomySearchHref(kind, node.id)}>
          <Search aria-hidden="true" />
          Find in search
        </Link>
      </Button>

      {node.children.length > 0 ? (
        <section aria-label="Contains" className="space-y-2">
          <Eyebrow>Contains</Eyebrow>
          <ul className="space-y-1">
            {node.children.map((child) => (
              <li
                key={child.id}
                className="flex items-baseline justify-between gap-3"
              >
                {nodeLink(child.id, child.label)}
                {counts ? (
                  <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
                    {formatCount(counts[child.id] ?? 0)}
                  </span>
                ) : null}
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {count === 0 ? null : <NodeExample kind={kind} standardId={node.id} />}
    </div>
  );
}

function NodeExample({
  kind,
  standardId,
}: {
  kind: TaxonomyKind;
  standardId: string;
}) {
  const location = useLocation();
  const { example, isLoading, error } = useTaxonomyNodeExample(
    kind,
    standardId,
  );
  const heading = kind === "tax" ? "Example clause" : "Example section";

  let body;
  if (isLoading) {
    body = (
      <div className="space-y-2" role="status" aria-live="polite">
        <span className="sr-only">Loading example</span>
        <Skeleton className="h-4 w-2/3" />
        <Skeleton className="h-16 w-full" />
      </div>
    );
  } else if (error) {
    body = <ErrorState variant="inline" error={error as Error} />;
  } else if (!example) {
    body = <p className="text-muted-foreground">No example found.</p>;
  } else {
    const text = truncateAtWord(
      example.bodyFormat === "xml"
        ? plainTextFromXml(example.body)
        : example.body,
      EXAMPLE_MAX_CHARS,
    );
    body = (
      <figure className="space-y-2 rounded-xl border border-border bg-card shadow-raised p-4">
        <figcaption className="space-y-0.5">
          <p className="font-medium text-foreground">
            {example.target} / {example.acquirer}
          </p>
          <p className="text-xs text-muted-foreground">
            {[example.year, example.heading].filter(Boolean).join(" · ")}
          </p>
        </figcaption>
        <blockquote className="text-sm leading-relaxed text-secondary-foreground">
          {text}
        </blockquote>
        <Link
          to={sectionReaderHref(
            example.agreementUuid,
            example.sectionUuid,
            `${location.pathname}${location.search}`,
          )}
          className="inline-flex text-sm font-medium text-primary underline-offset-2 hover:underline"
        >
          Open in agreement
        </Link>
      </figure>
    );
  }

  return (
    <section aria-label={heading} className="space-y-2">
      <Eyebrow>{heading}</Eyebrow>
      {body}
    </section>
  );
}
