import { useState, useMemo } from "react";
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
}

const XML_TAG_COLORS = {
  text: "text-blue-600",
  definition: "text-green-600",
  page: "text-purple-600",
  pageUUID: "text-orange-600",
} as const;

const SEARCH_COLLAPSIBLE_TAGS = new Set(["text", "definition"]);
const AGREEMENT_COLLAPSIBLE_TAGS = new Set(["article", "section"]);

export function XMLRenderer({
  xmlContent,
  className,
  mode = "search",
  highlightedSection,
}: XMLRendererProps) {
  const [collapsedTags, setCollapsedTags] = useState<Set<string>>(new Set());

  const parsedXML = useMemo(() => {
    return parseXMLContent(xmlContent);
  }, [xmlContent]);

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

  const renderNode = (
    node: XMLNode,
    index: number,
    depth: number = 0,
  ): React.ReactNode => {
    if (node.type === "text") {
      return (
        <span key={index} className="whitespace-pre-wrap">
          {node.content}
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
        "text-gray-600";

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

      // Handle text tags - remove tags and just render content
      if (node.tagName === "text") {
        if (mode === "search") {
          // In search mode, keep text collapsible
          return (
            <div key={tagId} className="my-1">
              <div className="flex items-start">
                <div className="w-4 flex-shrink-0">
                  <button
                    onClick={() => toggleCollapse(tagId)}
                    className="text-gray-400 hover:text-gray-600 transition-colors p-0.5"
                  >
                    {isCollapsed ? (
                      <ChevronRight className="w-3 h-3" />
                    ) : (
                      <ChevronDown className="w-3 h-3" />
                    )}
                  </button>
                </div>
                <div className="flex-1">
                  {!isCollapsed && node.children && (
                    <div className="leading-relaxed">
                      {node.children.map((child, childIndex) =>
                        renderNode(child, childIndex, depth + 1),
                      )}
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
              {node.children &&
                node.children.map((child, childIndex) =>
                  renderNode(child, childIndex, depth + 1),
                )}
            </div>
          );
        }
      }

      // Handle pageUUID and page tags - render inline
      if (node.tagName === "pageUUID" || node.tagName === "page") {
        const content =
          node.children?.find((child) => child.type === "text")?.content || "";
        return (
          <span
            key={tagId}
            className={cn("text-xs font-light inline", colorClass)}
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

        // Add article header attribute for scroll targeting
        const additionalAttributes =
          node.tagName === "article" ? { "data-article-header": "true" } : {};

        return (
          <div
            key={tagId}
            className={cn(
              "my-4 transition-all duration-1000",
              isHighlighted &&
                "bg-blue-50 border-2 border-blue-300 rounded-lg p-4 shadow-lg",
            )}
            {...dataAttributes}
            {...additionalAttributes}
          >
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0">
                {isCollapsible && (
                  <button
                    onClick={() => toggleCollapse(tagId)}
                    className="text-gray-400 hover:text-gray-600 transition-colors p-0.5"
                  >
                    {isCollapsed ? (
                      <ChevronRight className="w-4 h-4" />
                    ) : (
                      <ChevronDown className="w-4 h-4" />
                    )}
                  </button>
                )}
              </div>

              <div className="flex-1">
                <h3 className={cn(headerLevel, "text-gray-900 mb-2")}>
                  {title}
                </h3>

                {!isCollapsed && node.children && node.children.length > 0 && (
                  <div className="ml-2">
                    {node.children.map((child, childIndex) =>
                      renderNode(child, childIndex, depth + 1),
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        );
      }
      // For other tags in search mode or non-collapsible tags
      return (
        <div key={tagId} className="my-1" {...dataAttributes}>
          <div className="flex items-start">
            <div className="w-4 flex-shrink-0">
              {isCollapsible && (
                <button
                  onClick={() => toggleCollapse(tagId)}
                  className="text-gray-400 hover:text-gray-600 transition-colors p-0.5"
                >
                  {isCollapsed ? (
                    <ChevronRight className="w-3 h-3" />
                  ) : (
                    <ChevronDown className="w-3 h-3" />
                  )}
                </button>
              )}
            </div>

            <div className="flex-1">
              <span className={cn("text-xs font-light", colorClass)}>
                &lt;{node.tagName}&gt;
              </span>

              {!isCollapsed && node.children && node.children.length > 0 && (
                <div className="ml-4 mt-1">
                  {node.children.map((child, childIndex) =>
                    renderNode(child, childIndex, depth + 1),
                  )}
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
    <div className={cn("xml-renderer", className)}>
      {parsedXML.map((node, index) => renderNode(node, index))}
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
