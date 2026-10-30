import { memo, useCallback, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { decodeXmlEntities } from "@/lib/text-utils";
import { Badge } from "@/components/ui/badge";
import {
  buildFindRegex,
  renderFindHighlights,
} from "@/components/reader/find-highlight";
import {
  buildXmlTreeIndex,
  findFirstTagNode,
  parseXMLContent,
  setsAgreeWithin,
  type XMLNode,
  type XmlNodeMeta,
} from "@/lib/xml-parser";
import {
  anchorText,
  buildReaderOverlay,
  collectTextAnchors,
  isLinkedXref,
  type OverlayRange,
  type ReaderIndex,
  type ReaderOverlay,
} from "@/lib/reader-index";
import type { TermHighlightMode } from "@/components/reader/reader-preferences";

interface XMLRendererProps {
  xmlContent: string;
  className?: string;
  mode?: "search" | "agreement";
  isMobile?: boolean;
  showBodyOnly?: boolean;
  /**
   * Find-in-document query. Every hit in rendered text and headings is
   * wrapped in `<mark data-find-hit>`; collapsed sections render expanded
   * while a query is active so every hit is in the DOM.
   */
  findQuery?: string;
  /** Sections that matched the originating search: persistent left rule + "Matched" label. */
  matchedSectionUuids?: ReadonlySet<string> | null;
  /** The match the reader is focused on (stronger marker). */
  currentMatchUuid?: string | null;
  /**
   * Expand the collapsed ancestors of `uuid`. Bump `seq` to re-request the
   * same section after the user collapsed it again.
   */
  revealRequest?: { uuid: string; seq: number } | null;
  /** Per-section heading actions (agreement mode, `<section>` nodes only). Must be stable. */
  renderSectionActions?: (section: { uuid: string; title: string }) => ReactNode;
  /**
   * Smart-reader index for this XML (agreement mode). Term uses and
   * cross-references render as `.reader-term` / `.reader-xref` spans; the
   * reader handles their events by delegation.
   */
  readerIndex?: ReaderIndex | null;
  termMode?: TermHighlightMode;
  /** Called with the overlay built from `readerIndex` (null without one). */
  onOverlayChange?: (overlay: ReaderOverlay | null) => void;
}

const EMPTY_SET: ReadonlySet<string> = new Set<string>();

interface NormalizedAgreementTocRow {
  kind: "entry" | "heading";
  text: string;
  pageNumber?: string;
}

const XML_TAG_COLORS = {
  text: "text-info",
  definition: "text-success",
} as const;

const SEARCH_COLLAPSIBLE_TAGS = new Set(["text", "definition"]);
const AGREEMENT_COLLAPSIBLE_TAGS = new Set(["article", "section"]);
const NON_BREAKING_CHARS_RE = /[\u00a0\u2007\u202f\u2060\ufeff]/g;
const TOC_HEADER_LINE_RE = /^(?:TABLE OF CONTENTS|\(continued\)|Page)$/i;
const TOC_HEADING_LINE_RE = /^TABLE OF /i;
const TOC_PAGE_NUMBER_RE = /^(?:\d{1,4}|[ivxlcdm]+)$/i;
const TOC_TRAILING_PAGE_RE = /^(.*\S)\s+(?<page>(?:\d{1,4}|[ivxlcdm]+))$/i;
const TOC_DOT_LEADER_RE = /[.\u00b7\u2022\u2024\u2025\u2026\s]+$/u;
const TOC_ENTRY_START_PATTERN =
  "(?:TABLE OF CONTENTS|TABLE OF SCHEDULES AND EXHIBITS|\\(continued\\)|ARTICLE\\s+[IVXLCDM]+\\b|Section\\s+\\d+(?:\\.\\d+)*\\b|Annex\\s+[A-Z0-9]+\\b|Exhibit\\s+[A-Z0-9]+\\b|Schedule\\s+[A-Z0-9]+\\b|Appendix\\s+[A-Z0-9]+\\b)";
const TOC_ENTRY_BREAK_RE = new RegExp(`\\s+(?=${TOC_ENTRY_START_PATTERN})`, "gi");
const TOC_SUBENTRY_START_RE =
  /^(?:Section|Annex|Exhibit|Schedule|Appendix|\d+(?:\.\d+)+)\b/i;
const AGREEMENT_REGION_LABELS = new Map([
  ["frontMatter", "Front Matter"],
  ["tableOfContents", "Table of Contents"],
  ["body", "Body"],
  ["sigPages", "Signature Pages"],
  ["backMatter", "Back Matter"],
]);

export function normalizeXmlText(content: string) {
  return decodeXmlEntities(content.replace(NON_BREAKING_CHARS_RE, " "));
}

export function normalizeAgreementTableOfContentsText(
  textChunks: string[],
): NormalizedAgreementTocRow[] {
  const rows: NormalizedAgreementTocRow[] = [];
  const seenHeadings = new Set<string>();

  for (const chunk of textChunks) {
    const lines = normalizeAgreementTocChunk(chunk);

    for (const line of lines) {
      if (TOC_HEADER_LINE_RE.test(line)) {
        continue;
      }

      if (TOC_PAGE_NUMBER_RE.test(line)) {
        const previousRow = rows.at(-1);
        if (previousRow?.kind === "entry" && !previousRow.pageNumber) {
          previousRow.pageNumber = line;
        }
        continue;
      }

      if (TOC_HEADING_LINE_RE.test(line)) {
        const normalizedHeading = line.toUpperCase();
        if (!seenHeadings.has(normalizedHeading)) {
          seenHeadings.add(normalizedHeading);
          rows.push({ kind: "heading", text: line });
        }
        continue;
      }

      const entry = splitAgreementTocLine(line);
      rows.push({
        kind: "entry",
        text: entry.text,
        pageNumber: entry.pageNumber,
      });
    }
  }

  return rows;
}

// Props shared by every node in the tree. `meta` is recreated only when the
// document (or showBodyOnly) changes, and `onToggleCollapse` is stable, so
// the memo comparator only has to reason about the three stateful props.
interface SharedNodeProps {
  mode: "search" | "agreement";
  isMobile: boolean;
  meta: Map<XMLNode, XmlNodeMeta>;
  collapsedTags: ReadonlySet<string>;
  onToggleCollapse: (tagId: string) => void;
  findQuery: string;
  findRegex: RegExp | null;
  /** Nodes whose subtree contains a hit for `findRegex` (see collectFindHitNodes). */
  findHitNodes: ReadonlySet<XMLNode>;
  matchedSectionUuids: ReadonlySet<string>;
  currentMatchUuid: string | null;
  renderSectionActions?: (section: { uuid: string; title: string }) => ReactNode;
  overlay: ReaderOverlay | null;
  /** `<text>` tag node -> anchor key, only while there is an overlay. */
  anchorKeys: Map<XMLNode, string> | null;
  termMode: TermHighlightMode;
}

interface XMLNodeViewProps extends SharedNodeProps {
  node: XMLNode;
  index: number;
  depth: number;
  siblings: XMLNode[];
  parentTagName?: string;
}

function renderNodeList(
  children: XMLNode[] | undefined,
  depth: number,
  parentTagName: string | undefined,
  shared: SharedNodeProps,
) {
  return children?.map((child, childIndex) => (
    <XMLNodeView
      key={
        child.type === "text"
          ? `${depth}-${childIndex}-text`
          : (shared.meta.get(child)?.tagId ?? childIndex)
      }
      node={child}
      index={childIndex}
      depth={depth}
      siblings={children}
      parentTagName={parentTagName}
      {...shared}
    />
  ));
}

function renderAgreementTableOfContents(
  children: XMLNode[] | undefined,
  shared: SharedNodeProps,
) {
  const textChunks = extractAgreementTocTextChunks(children);
  const rows = normalizeAgreementTableOfContentsText(textChunks);

  if (rows.length === 0) {
    return renderNodeList(children, 0, "tableOfContents", shared);
  }

  return (
    <div
      className="space-y-1.5"
      role="list"
      aria-label="Agreement table of contents entries"
    >
      {rows.map((row, rowIndex) => {
        if (row.kind === "heading") {
          return (
            <div
              key={`toc-heading-${rowIndex}`}
              className="pt-4 text-center text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground first:pt-0"
            >
              {renderFindHighlights(row.text, shared.findRegex)}
            </div>
          );
        }

        const isSubentry = TOC_SUBENTRY_START_RE.test(row.text);
        const isArticle = /^ARTICLE\b/i.test(row.text);

        return (
          <div
            key={`toc-entry-${rowIndex}`}
            role="listitem"
            className={cn(
              "flex items-baseline gap-4 py-0.5",
              isSubentry ? "pl-4 sm:pl-6" : "pt-2 first:pt-0",
            )}
          >
            <span
              className={cn(
                "min-w-0 flex-1 break-words [overflow-wrap:anywhere]",
                isArticle && "font-medium tracking-[0.02em]",
              )}
            >
              {renderFindHighlights(row.text, shared.findRegex)}
            </span>
            {row.pageNumber ? (
              <span className="w-10 flex-shrink-0 text-right tabular-nums text-muted-foreground">
                {row.pageNumber}
              </span>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

function XMLNodeViewInner(props: XMLNodeViewProps): ReactNode {
  const {
    node,
    index,
    depth,
    siblings,
    parentTagName,
    mode,
    isMobile,
    meta,
    collapsedTags,
    onToggleCollapse,
    findQuery,
    findRegex,
    findHitNodes,
    matchedSectionUuids,
    currentMatchUuid,
    renderSectionActions,
    overlay,
    anchorKeys,
    termMode,
  } = props;

  if (node.type === "text") {
    const normalizedContent = normalizeXmlText(node.content);
    return (
      <span
        className={cn(
          isMobile
            ? "whitespace-pre-line break-words [overflow-wrap:anywhere]"
            : "whitespace-pre-wrap break-words [overflow-wrap:anywhere]",
        )}
      >
        {renderFindHighlights(normalizedContent, findRegex)}
      </span>
    );
  }

  const nodeMeta = meta.get(node);
  if (!nodeMeta) return null;

  const { tagId } = nodeMeta;
  const sectionUuid =
    node.tagName === "article" || node.tagName === "section"
      ? node.uuid
      : undefined;
  const isCollapsed = collapsedTags.has(tagId);
  const collapsibleTags =
    mode === "agreement" ? AGREEMENT_COLLAPSIBLE_TAGS : SEARCH_COLLAPSIBLE_TAGS;
  const isCollapsible = collapsibleTags.has(node.tagName || "");
  const colorClass =
    XML_TAG_COLORS[node.tagName as keyof typeof XML_TAG_COLORS] ||
    "text-muted-foreground";

  const dataAttributes = sectionUuid
    ? { "data-section-uuid": sectionUuid }
    : {};

  const shared: SharedNodeProps = {
    mode,
    isMobile,
    meta,
    collapsedTags,
    onToggleCollapse,
    findQuery,
    findRegex,
    findHitNodes,
    matchedSectionUuids,
    currentMatchUuid,
    renderSectionActions,
    overlay,
    anchorKeys,
    termMode,
  };

  if (mode === "agreement" && node.tagName === "metadata") {
    return null;
  }

  // Handle text tags - remove tags and just render content
  if (node.tagName === "text") {
    if (mode === "search") {
      // In search mode, keep text collapsible
      return (
        <div className="my-1">
          <div className="flex items-start">
            <div className="w-4 flex-shrink-0">
              <button
                type="button"
                onClick={() => onToggleCollapse(tagId)}
                data-collapse-toggle="true"
                className="inline-flex min-h-8 min-w-8 items-center justify-center rounded-md p-0.5 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:min-h-0 sm:min-w-0"
                aria-expanded={!isCollapsed}
                aria-label={isCollapsed ? "Expand section" : "Collapse section"}
              >
                {isCollapsed ? (
                  <ChevronRight className="w-3 h-3" aria-hidden="true" />
                ) : (
                  <ChevronDown className="w-3 h-3" aria-hidden="true" />
                )}
              </button>
            </div>
            <div className="min-w-0 flex-1">
              {!isCollapsed && node.children && (
                <div className="leading-relaxed">
                  {renderNodeList(node.children, depth + 1, node.tagName, shared)}
                </div>
              )}
            </div>
          </div>
        </div>
      );
    } else {
      // In agreement mode, just render content without tags. With a reader
      // index the leading text gets its term and cross-reference spans,
      // except while find is open (find marks win).
      const anchor = anchorKeys?.get(node);
      const ranges = anchor && !findRegex ? overlay?.rangesByKey.get(anchor) : undefined;
      const [first, ...rest] = node.children ?? [];
      if (ranges && first?.type === "text") {
        return (
          <div className="my-1 leading-relaxed" data-anchor={anchor}>
            <span
              className={
                isMobile
                  ? "whitespace-pre-line break-words [overflow-wrap:anywhere]"
                  : "whitespace-pre-wrap break-words [overflow-wrap:anywhere]"
              }
            >
              {renderOverlayText(anchorText(first.content), ranges, overlay!, termMode)}
            </span>
            {renderNodeList(rest, depth + 1, node.tagName, shared)}
          </div>
        );
      }
      return (
        <div className="my-1 leading-relaxed" data-anchor={anchor}>
          {renderNodeList(node.children, depth + 1, node.tagName, shared)}
        </div>
      );
    }
  }

  if (mode === "agreement" && node.tagName) {
    const regionLabel = AGREEMENT_REGION_LABELS.get(node.tagName);
    if (regionLabel) {
      return (
        <section
          id={`agreement-region-${node.tagName}`}
          data-reader-region={node.tagName}
          className="scroll-mt-3"
        >
          <div className="my-8 flex items-center gap-4" aria-hidden="true">
            <div className="h-0.5 flex-1 bg-foreground/30" />
            <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              {regionLabel}
            </div>
            <div className="h-0.5 flex-1 bg-foreground/30" />
          </div>
          <div className="min-w-0">
            {node.tagName === "tableOfContents"
              ? renderAgreementTableOfContents(node.children, shared)
              : renderNodeList(node.children, depth + 1, node.tagName, shared)}
          </div>
        </section>
      );
    }
  }

  // Handle pageUUID and page tags - render page separators
  if (node.tagName === "pageUUID" || node.tagName === "page") {
    if (node.tagName === "page") return null;
    if (mode === "agreement") {
      if (isAdjacentToAgreementRegionBreak(siblings, index)) return null;
      if (isAgreementRegionBoundaryPageBreak(siblings, index, parentTagName)) {
        return null;
      }
    }

    return (
      <div className="my-5 border-t border-foreground/20" aria-hidden="true" />
    );
  }

  // Handle article and section tags in agreement mode
  if (
    mode === "agreement" &&
    (node.tagName === "article" || node.tagName === "section")
  ) {
    const title =
      node.title !== undefined
        ? decodeXmlEntities(node.title)
        : `${node.tagName} ${index + 1}`;

    const headerLevel =
      node.tagName === "article"
        ? "text-lg font-semibold"
        : "text-base font-medium";
    // Articles sit under the agreement's h1, and their sections under them.
    const HeadingTag = node.tagName === "article" ? "h2" : "h3";
    const isMatched = sectionUuid != null && matchedSectionUuids.has(sectionUuid);
    const isCurrentMatch = sectionUuid != null && currentMatchUuid === sectionUuid;

    // Add article header attribute for scroll targeting
    const additionalAttributes =
      node.tagName === "article" ? { "data-article-header": "true" } : {};
    const matchAttributes = isMatched
      ? { "data-matched": isCurrentMatch ? "current" : "true" }
      : {};

    const containerProps = {
      className: cn(
        "my-4 scroll-mt-3 relative",
        // Persistent match marker: a left rule in the gutter, so the
        // section text keeps the same left edge as its neighbors.
        isMatched && "-ml-3.5 border-l-2 border-primary/30 pl-3",
        isCurrentMatch && "rounded-r-lg border-primary bg-primary/5 py-1 pr-2",
      ),
      ...dataAttributes,
      ...additionalAttributes,
      ...matchAttributes,
    };

    const headingExtras = (
      <>
        {isMatched ? (
          <Badge variant="count" className="mt-0.5 shrink-0 px-2 py-0 text-[11px]">
            Matched
          </Badge>
        ) : null}
        {node.tagName === "section" && sectionUuid && renderSectionActions
          ? renderSectionActions({ uuid: sectionUuid, title })
          : null}
      </>
    );

    if (isMobile) {
      return (
        <div {...containerProps}>
          <div className="group/heading flex items-start gap-2" data-section-heading="">
            {isCollapsible && (
              <button
                type="button"
                onClick={() => onToggleCollapse(tagId)}
                data-collapse-toggle="true"
                className="inline-flex min-h-8 min-w-8 items-center justify-center rounded-md p-0.5 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:min-h-0 sm:min-w-0"
                aria-expanded={!isCollapsed}
                aria-label={isCollapsed ? "Expand section" : "Collapse section"}
              >
                {isCollapsed ? (
                  <ChevronRight className="w-4 h-4" aria-hidden="true" />
                ) : (
                  <ChevronDown className="w-4 h-4" aria-hidden="true" />
                )}
              </button>
            )}

            <HeadingTag className={cn(headerLevel, "min-w-0 flex-1 break-words text-foreground [overflow-wrap:anywhere]")}>
              {renderFindHighlights(title, findRegex)}
            </HeadingTag>
            {headingExtras}
          </div>

          {!isCollapsed && node.children && node.children.length > 0 && (
            <div className="agreement-children mt-2 min-w-0">
              {renderNodeList(node.children, depth + 1, node.tagName, shared)}
            </div>
          )}
        </div>
      );
    }

    return (
      <div {...containerProps}>
        <div className="flex items-start gap-1.5">
          <div className="flex-shrink-0">
            {isCollapsible && (
              <button
                type="button"
                onClick={() => onToggleCollapse(tagId)}
                data-collapse-toggle="true"
                className="inline-flex min-h-8 min-w-8 items-center justify-center rounded-md p-0.5 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:min-h-0 sm:min-w-0"
                aria-expanded={!isCollapsed}
                aria-label={isCollapsed ? "Expand section" : "Collapse section"}
              >
                {isCollapsed ? (
                  <ChevronRight className="w-4 h-4" aria-hidden="true" />
                ) : (
                  <ChevronDown className="w-4 h-4" aria-hidden="true" />
                )}
              </button>
            )}
          </div>

          <div className="min-w-0 flex-1">
            <div
              className="group/heading mb-2 flex items-start gap-2"
              data-section-heading=""
            >
              <HeadingTag className={cn(headerLevel, "min-w-0 flex-1 break-words text-foreground [overflow-wrap:anywhere]")}>
                {renderFindHighlights(title, findRegex)}
              </HeadingTag>
              {headingExtras}
            </div>

            {!isCollapsed && node.children && node.children.length > 0 && (
              <div className="agreement-children ml-2 min-w-0">
                {renderNodeList(node.children, depth + 1, node.tagName, shared)}
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (mode === "agreement") {
    return (
      <div className="my-1 min-w-0 scroll-mt-3" {...dataAttributes}>
        {renderNodeList(node.children, depth + 1, node.tagName, shared)}
      </div>
    );
  }

  // For other tags in search mode or non-collapsible tags
  return (
    <div className="my-1 scroll-mt-3" {...dataAttributes}>
      <div className="flex items-start gap-1">
        <div className="flex-shrink-0">
          {isCollapsible && (
            <button
              type="button"
              onClick={() => onToggleCollapse(tagId)}
              data-collapse-toggle="true"
              className="inline-flex min-h-8 min-w-8 items-center justify-center rounded-md p-0.5 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:min-h-0 sm:min-w-0"
              aria-expanded={!isCollapsed}
              aria-label={isCollapsed ? "Expand section" : "Collapse section"}
            >
              {isCollapsed ? (
                <ChevronRight className="w-3 h-3" aria-hidden="true" />
              ) : (
                <ChevronDown className="w-3 h-3" aria-hidden="true" />
              )}
            </button>
          )}
        </div>

        <div className="min-w-0 flex-1">
          <span className={cn("text-xs font-light", colorClass)}>
            &lt;{node.tagName}&gt;
          </span>

          {!isCollapsed && node.children && node.children.length > 0 && (
            <div className="ml-4 mt-1 min-w-0">
              {renderNodeList(node.children, depth + 1, node.tagName, shared)}
            </div>
          )}

          {!isCollapsed && (
            <div className="mt-1">
              <span className={cn("text-xs font-light", colorClass)}>
                &lt;/{node.tagName}&gt;
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function renderOverlayText(
  text: string,
  ranges: OverlayRange[],
  overlay: ReaderOverlay,
  termMode: TermHighlightMode,
): ReactNode[] {
  const out: ReactNode[] = [];
  let cursor = 0;
  for (const range of ranges) {
    let element: ReactNode = null;
    if (range.kind === "xref") {
      if (isLinkedXref(range.xref)) {
        element = (
          <a
            key={range.start}
            href={`#${encodeURIComponent(range.xref.target!)}`}
            className="reader-xref"
            data-xref-index={range.index}
          >
            {text.slice(range.start, range.end)}
          </a>
        );
      }
    } else if (termMode !== "off") {
      const term = overlay.termsById.get(range.termId);
      if (term && (termMode === "all" || term.confidence !== "low")) {
        element = (
          <span key={range.start} className="reader-term" data-term-id={range.termId}>
            {text.slice(range.start, range.end)}
          </span>
        );
      }
    }
    if (!element) continue;
    if (range.start > cursor) out.push(text.slice(cursor, range.start));
    out.push(element);
    cursor = range.end;
  }
  if (cursor < text.length) out.push(text.slice(cursor));
  return out;
}

function regexHits(regex: RegExp, text: string): boolean {
  regex.lastIndex = 0;
  const hit = regex.test(text);
  regex.lastIndex = 0;
  return hit;
}

const EMPTY_NODE_SET: ReadonlySet<XMLNode> = new Set<XMLNode>();

/** Every node (text leaf or tag) whose rendered subtree contains a find hit. */
function collectFindHitNodes(
  nodes: XMLNode[],
  regex: RegExp | null,
): ReadonlySet<XMLNode> {
  if (!regex) return EMPTY_NODE_SET;
  const hits = new Set<XMLNode>();
  const visit = (node: XMLNode): boolean => {
    let hit = false;
    if (node.type === "text") {
      hit = regexHits(regex, normalizeXmlText(node.content));
    } else {
      if (node.title && regexHits(regex, decodeXmlEntities(node.title))) hit = true;
      for (const child of node.children ?? []) {
        if (visit(child)) hit = true;
      }
    }
    if (hit) hits.add(node);
    return hit;
  };
  nodes.forEach(visit);
  return hits;
}

// Skip re-rendering a subtree unless a collapse/match change actually
// touches an id inside it. All identity props are stable per parsed document,
// so a match or collapse change only re-renders the affected paths.
function xmlNodeViewPropsEqual(
  prev: XMLNodeViewProps,
  next: XMLNodeViewProps,
): boolean {
  if (
    prev.node !== next.node ||
    prev.index !== next.index ||
    prev.depth !== next.depth ||
    prev.siblings !== next.siblings ||
    prev.parentTagName !== next.parentTagName ||
    prev.mode !== next.mode ||
    prev.isMobile !== next.isMobile ||
    prev.meta !== next.meta ||
    prev.onToggleCollapse !== next.onToggleCollapse ||
    prev.overlay !== next.overlay ||
    prev.termMode !== next.termMode
  ) {
    return false;
  }

  if (prev.findQuery !== next.findQuery) {
    // Only subtrees holding a hit for the old or new query re-render, so a
    // query change on a long agreement touches a few paths, not every node.
    if (prev.findHitNodes.has(next.node) || next.findHitNodes.has(next.node)) {
      return false;
    }
    if (next.node.type !== "tag") return true;
  }
  if (next.node.type !== "tag") return true;
  const nodeMeta = next.meta.get(next.node);
  if (!nodeMeta) return false;

  const { subtreeTagIds, subtreeUuids } = nodeMeta;
  if (prev.renderSectionActions !== next.renderSectionActions) return false;
  if (
    prev.currentMatchUuid !== next.currentMatchUuid &&
    ((prev.currentMatchUuid != null && subtreeUuids.has(prev.currentMatchUuid)) ||
      (next.currentMatchUuid != null && subtreeUuids.has(next.currentMatchUuid)))
  ) {
    return false;
  }
  if (
    prev.matchedSectionUuids !== next.matchedSectionUuids &&
    !setsAgreeWithin(prev.matchedSectionUuids, next.matchedSectionUuids, subtreeUuids)
  ) {
    return false;
  }
  if (!setsAgreeWithin(prev.collapsedTags, next.collapsedTags, subtreeTagIds)) {
    return false;
  }
  return true;
}

const XMLNodeView = memo(XMLNodeViewInner, xmlNodeViewPropsEqual);

export function XMLRenderer({
  xmlContent,
  className,
  mode = "search",
  isMobile = false,
  showBodyOnly = false,
  findQuery = "",
  matchedSectionUuids,
  currentMatchUuid = null,
  revealRequest = null,
  renderSectionActions,
  readerIndex = null,
  termMode = "distinctive",
  onOverlayChange,
}: XMLRendererProps) {
  const [collapsedTags, setCollapsedTags] = useState<Set<string>>(new Set());
  const parsedDocument = useMemo(() => parseXMLContent(xmlContent), [xmlContent]);
  const {
    nodes: parsedXML,
    meta,
    uuidPaths,
  } = useMemo(() => {
    const nodes = showBodyOnly
      ? (findFirstTagNode(parsedDocument, "body")?.children ?? parsedDocument)
      : parsedDocument;
    return { nodes, ...buildXmlTreeIndex(nodes) };
  }, [parsedDocument, showBodyOnly]);

  // Anchors follow the whole document, so the body-only view keeps them.
  const { overlay, anchorKeys } = useMemo(() => {
    if (!readerIndex || mode !== "agreement") return { overlay: null, anchorKeys: null };
    const anchors = collectTextAnchors(parsedDocument);
    return {
      overlay: buildReaderOverlay(readerIndex, anchors),
      anchorKeys: anchors.keyByNode,
    };
  }, [mode, parsedDocument, readerIndex]);

  useEffect(() => {
    onOverlayChange?.(overlay);
  }, [onOverlayChange, overlay]);

  // Path-based collapse ids shift when the rendered tree changes, so reset
  // collapse state whenever the document or the body-only view changes.
  useEffect(() => {
    setCollapsedTags(new Set());
  }, [xmlContent, showBodyOnly]);

  // On a reader jump (TOC, match stepping, deep link) expand any collapsed
  // ancestors so the target actually renders and can be scrolled to.
  // Collapse state for uuid-bearing nodes is keyed by uuid, so deleting the
  // uuids on the root->target path expands every ancestor of the target.
  const revealUuid = revealRequest?.uuid;
  const revealSeq = revealRequest?.seq;
  useEffect(() => {
    const pathUuids = revealUuid ? (uuidPaths.get(revealUuid) ?? []) : [];
    if (pathUuids.length === 0) return;
    setCollapsedTags((prev) => {
      let changed = false;
      const next = new Set(prev);
      for (const uuid of pathUuids) {
        if (next.delete(uuid)) changed = true;
      }
      return changed ? next : prev;
    });
  }, [revealUuid, revealSeq, uuidPaths]);

  const normalizedFindQuery = findQuery.trim();
  const findRegex = useMemo(
    () => buildFindRegex(normalizedFindQuery),
    [normalizedFindQuery],
  );
  const findHitNodes = useMemo(
    () => collectFindHitNodes(parsedXML, findRegex),
    [findRegex, parsedXML],
  );

  const toggleCollapse = useCallback((tagId: string) => {
    setCollapsedTags((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(tagId)) {
        newSet.delete(tagId);
      } else {
        newSet.add(tagId);
      }
      return newSet;
    });
  }, []);

  if (parsedXML.length === 0) {
    return (
      <div className={cn("xml-renderer min-w-0 max-w-full", className)}>
        <div className="py-8 text-center" role="alert">
          <p className="font-medium text-destructive">
            Unable to render this document
          </p>
          <p className="mt-1 text-sm text-muted-foreground">
            The document content is empty or could not be parsed.
          </p>
        </div>
      </div>
    );
  }

  const shared: SharedNodeProps = {
    mode,
    isMobile,
    meta,
    // While find is active every section renders expanded so every hit is in
    // the DOM; the user's collapse state comes back when find closes.
    collapsedTags: findRegex ? EMPTY_SET : collapsedTags,
    onToggleCollapse: toggleCollapse,
    findQuery: normalizedFindQuery,
    findRegex,
    findHitNodes,
    matchedSectionUuids: matchedSectionUuids ?? EMPTY_SET,
    currentMatchUuid,
    renderSectionActions,
    overlay,
    anchorKeys,
    termMode,
  };

  return (
    <div className={cn("xml-renderer min-w-0 max-w-full", className)}>
      {renderNodeList(parsedXML, 0, undefined, shared)}
    </div>
  );
}

function isAdjacentToAgreementRegionBreak(
  siblings: XMLNode[],
  index: number,
): boolean {
  return (
    isAgreementRegionNode(siblings[index - 1]) ||
    isAgreementRegionNode(siblings[index + 1])
  );
}

function isAgreementRegionNode(node: XMLNode | undefined): boolean {
  return (
    node?.type === "tag" &&
    (node.tagName ? AGREEMENT_REGION_LABELS.has(node.tagName) : false)
  );
}

function isAgreementRegionBoundaryPageBreak(
  siblings: XMLNode[],
  index: number,
  parentTagName?: string,
): boolean {
  if (!parentTagName || !AGREEMENT_REGION_LABELS.has(parentTagName)) {
    return false;
  }

  const hasPreviousContent = siblings
    .slice(0, index)
    .some(hasSubstantiveReaderContent);
  const hasNextContent = siblings
    .slice(index + 1)
    .some(hasSubstantiveReaderContent);

  return !hasPreviousContent || !hasNextContent;
}

function hasSubstantiveReaderContent(node: XMLNode): boolean {
  if (node.type === "text") return node.content.trim().length > 0;
  if (isPageMarkerNode(node) || node.tagName === "metadata") return false;
  return node.children ? node.children.some(hasSubstantiveReaderContent) : false;
}

function isPageMarkerNode(node: XMLNode): boolean {
  return (
    node.type === "tag" &&
    (node.tagName === "page" || node.tagName === "pageUUID")
  );
}

function normalizeAgreementTocChunk(content: string): string[] {
  const normalized = normalizeXmlText(content)
    .replace(/[^\S\r\n]+/g, " ")
    .replace(/\s*\n\s*/g, "\n")
    .trim();

  if (!normalized) {
    return [];
  }

  const withEntryBreaks = normalized
    .replace(
      /^Page\s+(?=(?:ARTICLE|Section|Annex|Exhibit|Schedule|Appendix)\b)/i,
      "Page\n",
    )
    .replace(TOC_ENTRY_BREAK_RE, "\n");

  return withEntryBreaks
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean);
}

function extractAgreementTocTextChunks(children: XMLNode[] | undefined): string[] {
  if (!children) {
    return [];
  }

  return children.flatMap((child) => {
    if (child.type === "text") {
      return child.content;
    }

    if (child.tagName !== "text" || !child.children) {
      return [];
    }

    return child.children
      .filter((grandchild) => grandchild.type === "text")
      .map((grandchild) => grandchild.content);
  });
}

function splitAgreementTocLine(
  line: string,
): { text: string; pageNumber?: string } {
  const match = line.match(TOC_TRAILING_PAGE_RE);
  if (!match?.groups?.page) {
    return { text: line };
  }

  return {
    text: match[1].replace(TOC_DOT_LEADER_RE, "").trim(),
    pageNumber: match.groups.page,
  };
}
