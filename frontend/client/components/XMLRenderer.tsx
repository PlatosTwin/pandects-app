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

const XML_TAG_COLORS = {
  text: "text-blue-600",
  definition: "text-green-600",
  page: "text-purple-600",
  pageUUID: "text-orange-600",
} as const;

const SEARCH_COLLAPSIBLE_TAGS = new Set(["text", "definition"]);
const AGREEMENT_COLLAPSIBLE_TAGS = new Set(["article", "section"]);
const NON_BREAKING_CHARS_RE = /[\u00a0\u2007\u202f\u2060\ufeff]/g;
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
                    className="text-muted-foreground hover:text-foreground transition-colors p-0.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
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
                {renderChildren(node.children, depth + 1, node.tagName)}
              </div>
            </section>
          );
        }
      }

      // Handle pageUUID and page tags - render page separators in agreement mode
      if (node.tagName === "pageUUID" || node.tagName === "page") {
        if (mode === "agreement") {
          if (node.tagName === "page") return null;
          if (isAdjacentToAgreementHeavyBreak(siblings, index)) return null;
          if (isAgreementContainerBoundaryPageBreak(siblings, index, parentTagName)) {
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
            "my-6 border-t-2 border-foreground/20 pt-5 scroll-mt-3 relative",
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
                    className="text-muted-foreground hover:text-foreground transition-colors p-0.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
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

                <h3 className={cn(headerLevel, "text-foreground min-w-0 flex-1")}>
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
                    className="text-muted-foreground hover:text-foreground transition-colors p-0.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
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
                <h3 className={cn(headerLevel, "text-foreground mb-2")}>
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
                  className="text-muted-foreground hover:text-foreground transition-colors p-0.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
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

function isAdjacentToAgreementHeavyBreak(
  siblings: XMLNode[],
  index: number,
): boolean {
  return (
    isAgreementHeavyBreakNode(siblings[index - 1]) ||
    isAgreementHeavyBreakNode(siblings[index + 1])
  );
}

function isAgreementHeavyBreakNode(node: XMLNode | undefined): boolean {
  return (
    node?.type === "tag" &&
    (node.tagName === "article" ||
      node.tagName === "section" ||
      (node.tagName ? AGREEMENT_REGION_LABELS.has(node.tagName) : false))
  );
}

function isAgreementContainerBoundaryPageBreak(
  siblings: XMLNode[],
  index: number,
  parentTagName?: string,
): boolean {
  if (!parentTagName || !isAgreementContentContainer(parentTagName)) {
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

function isAgreementContentContainer(tagName: string): boolean {
  return (
    tagName === "article" ||
    tagName === "section" ||
    AGREEMENT_REGION_LABELS.has(tagName)
  );
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
