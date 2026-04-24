import { useState, useMemo, useEffect } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface XMLNode {
  type: "text" | "tag";
  content: string;
  tagName?: string;
  children?: XMLNode[];
}

interface XMLRendererProps {
  xmlContent: string;
  className?: string;
  mode?: "search" | "agreement";
  highlightedSection?: string | null;
  isMobile?: boolean;
  showBodyOnly?: boolean;
}

interface NormalizedAgreementTocRow {
  kind: "entry" | "heading";
  text: string;
  pageNumber?: string;
}

const XML_TAG_COLORS = {
  text: "text-blue-600",
  definition: "text-green-600",
  page: "text-purple-600",
  pageUUID: "text-orange-600",
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
  return content.replace(NON_BREAKING_CHARS_RE, " ");
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

export function XMLRenderer({
  xmlContent,
  className,
  mode = "search",
  highlightedSection,
  isMobile = false,
  showBodyOnly = false,
}: XMLRendererProps) {
  const [collapsedTags, setCollapsedTags] = useState<Set<string>>(new Set());
  const [fadingHighlights, setFadingHighlights] = useState<Set<string>>(
    new Set(),
  );

  // Handle highlight transitions
  const [previousHighlightedSection, setPreviousHighlightedSection] = useState<
    string | null
  >(null);

  useEffect(() => {
    if (
      previousHighlightedSection &&
      previousHighlightedSection !== highlightedSection
    ) {
      setFadingHighlights((prev) =>
        new Set(prev).add(previousHighlightedSection),
      );

      const timer = setTimeout(() => {
        setFadingHighlights((prev) => {
          const newSet = new Set(prev);
          newSet.delete(previousHighlightedSection);
          return newSet;
        });
      }, 1000);

      setPreviousHighlightedSection(highlightedSection);
      return () => clearTimeout(timer);
    }

    setPreviousHighlightedSection(highlightedSection);
    return undefined;
  }, [highlightedSection, previousHighlightedSection]);

  const parsedXML = useMemo(() => {
    const nodes = parseXMLContent(xmlContent);
    if (!showBodyOnly) return nodes;

    const bodyNode = findFirstTagNode(nodes, "body");
    return bodyNode?.children ?? nodes;
  }, [showBodyOnly, xmlContent]);

  const toggleCollapse = (tagId: string) => {
    setCollapsedTags((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(tagId)) {
        newSet.delete(tagId);
      } else {
        newSet.add(tagId);
      }
      return newSet;
    });
  };

  const renderAgreementTableOfContents = (children: XMLNode[] | undefined) => {
    const textChunks = extractAgreementTocTextChunks(children);
    const rows = normalizeAgreementTableOfContentsText(textChunks);

    if (rows.length === 0) {
      return renderChildren(children, 0, "tableOfContents");
    }

    return (
      <div className="space-y-1.5">
        {rows.map((row, rowIndex) => {
          if (row.kind === "heading") {
            return (
              <div
                key={`toc-heading-${rowIndex}`}
                className="pt-4 text-center text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground first:pt-0"
              >
                {row.text}
              </div>
            );
          }

          const isSubentry = TOC_SUBENTRY_START_RE.test(row.text);
          const isArticle = /^ARTICLE\b/i.test(row.text);

          return (
            <div
              key={`toc-entry-${rowIndex}`}
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
                {row.text}
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
  };

  const renderChildren = (
    children: XMLNode[] | undefined,
    depth: number,
    parentTagName?: string,
  ) =>
    children?.map((child, childIndex) =>
      renderNode(child, childIndex, depth, children, parentTagName),
    );

  const renderNode = (
    node: XMLNode,
    index: number,
    depth: number = 0,
    siblings: XMLNode[] = [],
    parentTagName?: string,
  ): React.ReactNode => {
    if (node.type === "text") {
      const normalizedContent = normalizeXmlText(node.content);
      return (
        <span
          key={`${depth}-${index}-text`}
          className={cn(
            isMobile
              ? "whitespace-pre-line break-words [overflow-wrap:anywhere]"
              : "whitespace-pre-wrap break-words [overflow-wrap:anywhere]",
          )}
        >
          {normalizedContent}
        </span>
      );
    }

    if (node.type === "tag") {
      const tagId = `${node.tagName}-${index}-${depth}`;
      const isCollapsed = collapsedTags.has(tagId);
      const collapsibleTags =
        mode === "agreement"
          ? AGREEMENT_COLLAPSIBLE_TAGS
          : SEARCH_COLLAPSIBLE_TAGS;
      const isCollapsible = collapsibleTags.has(node.tagName || "");
      const colorClass =
        XML_TAG_COLORS[node.tagName as keyof typeof XML_TAG_COLORS] ||
        "text-muted-foreground";

      // Extract UUID from attributes for sections/articles for scroll-to functionality
      let sectionUuid: string | undefined;

      // Check if this node has a uuid attribute (for article/section tags)
      if (node.tagName === "article" || node.tagName === "section") {
        const uuidMatch = node.content.match(/uuid="([^"]*)"/);
        sectionUuid = uuidMatch ? uuidMatch[1] : undefined;
      }

      const dataAttributes = sectionUuid
        ? { "data-section-uuid": sectionUuid }
        : {};

      if (mode === "agreement" && node.tagName === "metadata") {
        return null;
      }

      // Handle text tags - remove tags and just render content
      if (node.tagName === "text") {
        if (mode === "search") {
          // In search mode, keep text collapsible
          return (
            <div key={tagId} className="my-1">
              <div className="flex items-start">
                <div className="w-4 flex-shrink-0">
                  <button
                    type="button"
                    onClick={() => toggleCollapse(tagId)}
                    data-collapse-toggle="true"
                    className="inline-flex min-h-8 min-w-8 items-center justify-center rounded-md p-0.5 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:min-h-0 sm:min-w-0"
                    aria-expanded={!isCollapsed}
                    aria-label={
                      isCollapsed ? "Expand section" : "Collapse section"
                    }
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
                      {renderChildren(node.children, depth + 1, node.tagName)}
                    </div>
                  )}
                </div>
              </div>
            </div>
          );
        } else {
          // In agreement mode, just render content without tags
          return (
            <div key={tagId} className="my-1 leading-relaxed">
              {renderChildren(node.children, depth + 1, node.tagName)}
            </div>
          );
        }
      }

      if (mode === "agreement" && node.tagName) {
        const regionLabel = AGREEMENT_REGION_LABELS.get(node.tagName);
        if (regionLabel) {
          return (
            <section
              key={tagId}
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
                  ? renderAgreementTableOfContents(node.children)
                  : renderChildren(node.children, depth + 1, node.tagName)}
              </div>
            </section>
          );
        }
      }

      // Handle pageUUID and page tags - render page separators in agreement mode
      if (node.tagName === "pageUUID" || node.tagName === "page") {
        if (mode === "agreement") {
          if (node.tagName === "page") return null;
          if (isAdjacentToAgreementRegionBreak(siblings, index)) return null;
          if (isAgreementRegionBoundaryPageBreak(siblings, index, parentTagName)) {
            return null;
          }

          return (
            <div
              key={tagId}
              className="my-5 border-t border-foreground/20"
              aria-hidden="true"
            />
          );
        }

        const content =
          node.children?.find((child) => child.type === "text")?.content || "";
        return (
          <span
            key={tagId}
            className={cn(
              "text-xs font-light inline",
              colorClass,
              "break-words [overflow-wrap:anywhere]",
            )}
          >
            &lt;{node.tagName}&gt;{content}&lt;/{node.tagName}&gt;
          </span>
        );
      }

      // Handle article and section tags in agreement mode
      if (
        mode === "agreement" &&
        (node.tagName === "article" || node.tagName === "section")
      ) {
        // Extract title from attributes
        const titleMatch = node.content.match(/title="([^"]*)"/);
        const title = titleMatch
          ? titleMatch[1]
          : `${node.tagName} ${index + 1}`;

        const headerLevel =
          node.tagName === "article"
            ? "text-lg font-semibold"
            : "text-base font-medium";
        const isHighlighted = highlightedSection === sectionUuid;
        const isFading = fadingHighlights.has(sectionUuid || "");
        const showHighlight = isHighlighted || isFading;

        // Add article header attribute for scroll targeting
        const additionalAttributes =
          node.tagName === "article" ? { "data-article-header": "true" } : {};

        const containerProps = {
          className: cn(
            "my-4 scroll-mt-3 relative",
            showHighlight && "z-10",
            showHighlight && isMobile && "pl-1 pr-1 -ml-1 -mr-1",
            showHighlight && !isMobile && "pr-2 -mr-2",
          ),
          ...dataAttributes,
          ...additionalAttributes,
        };

        const highlightOverlay = showHighlight ? (
          <div
            className={cn(
              "absolute inset-0 bg-primary/10 border border-primary/30 rounded-lg pointer-events-none -z-10 transition-opacity duration-1000 ease-out",
              isHighlighted ? "opacity-100" : "opacity-0",
            )}
          />
        ) : null;

        if (isMobile) {
          return (
            <div key={tagId} {...containerProps}>
              {highlightOverlay}
              <div className="flex items-start gap-2">
                {isCollapsible && (
                  <button
                    type="button"
                    onClick={() => toggleCollapse(tagId)}
                    data-collapse-toggle="true"
                    className="inline-flex min-h-8 min-w-8 items-center justify-center rounded-md p-0.5 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:min-h-0 sm:min-w-0"
                    aria-expanded={!isCollapsed}
                    aria-label={
                      isCollapsed ? "Expand section" : "Collapse section"
                    }
                  >
                    {isCollapsed ? (
                      <ChevronRight className="w-4 h-4" aria-hidden="true" />
                    ) : (
                      <ChevronDown className="w-4 h-4" aria-hidden="true" />
                    )}
                  </button>
                )}

                <h3 className={cn(headerLevel, "min-w-0 flex-1 break-words text-foreground [overflow-wrap:anywhere]")}>
                  {title}
                </h3>
              </div>

              {!isCollapsed && node.children && node.children.length > 0 && (
                <div className="agreement-children mt-2 min-w-0">
                  {renderChildren(node.children, depth + 1, node.tagName)}
                </div>
              )}
            </div>
          );
        }

        return (
          <div key={tagId} {...containerProps}>
            {highlightOverlay}
            <div className="flex items-start gap-1.5">
              <div className="flex-shrink-0">
                {isCollapsible && (
                  <button
                    type="button"
                    onClick={() => toggleCollapse(tagId)}
                    data-collapse-toggle="true"
                    className="inline-flex min-h-8 min-w-8 items-center justify-center rounded-md p-0.5 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:min-h-0 sm:min-w-0"
                    aria-expanded={!isCollapsed}
                    aria-label={
                      isCollapsed ? "Expand section" : "Collapse section"
                    }
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
                <h3 className={cn(headerLevel, "mb-2 break-words text-foreground [overflow-wrap:anywhere]")}>
                  {title}
                </h3>

                {!isCollapsed && node.children && node.children.length > 0 && (
                  <div className="agreement-children ml-2 min-w-0">
                    {renderChildren(node.children, depth + 1, node.tagName)}
                  </div>
                )}
              </div>
            </div>
          </div>
        );
      }

      if (mode === "agreement") {
        return (
          <div key={tagId} className="my-1 min-w-0 scroll-mt-3" {...dataAttributes}>
            {renderChildren(node.children, depth + 1, node.tagName)}
          </div>
        );
      }

      // For other tags in search mode or non-collapsible tags
      return (
        <div key={tagId} className="my-1 scroll-mt-3" {...dataAttributes}>
          <div className="flex items-start gap-1">
            <div className="flex-shrink-0">
              {isCollapsible && (
                <button
                  type="button"
                  onClick={() => toggleCollapse(tagId)}
                  data-collapse-toggle="true"
                  className="inline-flex min-h-8 min-w-8 items-center justify-center rounded-md p-0.5 text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:min-h-0 sm:min-w-0"
                  aria-expanded={!isCollapsed}
                  aria-label={
                    isCollapsed ? "Expand section" : "Collapse section"
                  }
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
                  {renderChildren(node.children, depth + 1, node.tagName)}
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

    return null;
  };

  return (
    <div className={cn("xml-renderer min-w-0 max-w-full", className)}>
      {renderChildren(parsedXML, 0)}
    </div>
  );
}

function parseXMLContent(xmlContent: string): XMLNode[] {
  const nodes: XMLNode[] = [];
  let currentIndex = 0;

  while (currentIndex < xmlContent.length) {
    // Find the next opening tag
    const tagStart = xmlContent.indexOf("<", currentIndex);

    if (tagStart === -1) {
      // No more tags, add remaining text
      if (currentIndex < xmlContent.length) {
        const remainingText = xmlContent.slice(currentIndex);
        if (remainingText.trim()) {
          nodes.push({
            type: "text",
            content: remainingText,
          });
        }
      }
      break;
    }

    // Add text before the tag if any
    if (tagStart > currentIndex) {
      const textContent = xmlContent.slice(currentIndex, tagStart);
      if (textContent.trim()) {
        nodes.push({
          type: "text",
          content: textContent,
        });
      }
    }

    // Skip XML declarations and comments
    if (xmlContent.slice(tagStart, tagStart + 4) === "<!--") {
      const commentEnd = xmlContent.indexOf("-->", tagStart);
      if (commentEnd !== -1) {
        currentIndex = commentEnd + 3;
        continue;
      }
    }

    if (xmlContent.slice(tagStart, tagStart + 5) === "<?xml") {
      const declEnd = xmlContent.indexOf("?>", tagStart);
      if (declEnd !== -1) {
        currentIndex = declEnd + 2;
        continue;
      }
    }

    // Find the end of the opening tag
    const tagEnd = xmlContent.indexOf(">", tagStart);
    if (tagEnd === -1) {
      // Malformed XML, treat as text
      nodes.push({
        type: "text",
        content: xmlContent.slice(tagStart),
      });
      break;
    }

    // Extract tag content and name
    const tagContent = xmlContent.slice(tagStart + 1, tagEnd);
    const tagName = tagContent.split(/\s/)[0]; // Get tag name before any attributes

    // Check if it's a self-closing tag
    if (tagContent.endsWith("/")) {
      nodes.push({
        type: "tag",
        content: `<${tagContent}>`,
        tagName: tagName.replace("/", ""),
        children: [],
      });
      currentIndex = tagEnd + 1;
      continue;
    }

    // Find the corresponding closing tag
    const closingTag = `</${tagName}>`;
    const closingTagStart = xmlContent.indexOf(closingTag, tagEnd + 1);

    if (closingTagStart === -1) {
      // No closing tag found, treat as text
      nodes.push({
        type: "text",
        content: xmlContent.slice(tagStart),
      });
      break;
    }

    // Extract content between opening and closing tags
    const innerContent = xmlContent.slice(tagEnd + 1, closingTagStart);
    const children = innerContent.trim() ? parseXMLContent(innerContent) : [];

    nodes.push({
      type: "tag",
      content: `<${tagContent}>`,
      tagName: tagName,
      children: children,
    });

    currentIndex = closingTagStart + closingTag.length;
  }

  return nodes;
}

function findFirstTagNode(nodes: XMLNode[], tagName: string): XMLNode | null {
  for (const node of nodes) {
    if (node.type !== "tag") continue;
    if (node.tagName === tagName) return node;

    const childMatch = node.children
      ? findFirstTagNode(node.children, tagName)
      : null;
    if (childMatch) return childMatch;
  }

  return null;
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
