import { useState, useMemo } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface XMLNode {
  type: "text" | "tag";
  content: string;
  tagName?: string;
  children?: XMLNode[];
  isCollapsible?: boolean;
}

interface XMLRendererProps {
  xmlContent: string;
  className?: string;
}

const XML_TAG_COLORS = {
  text: "text-blue-600",
  definition: "text-green-600",
  page: "text-purple-600",
  pageUUID: "text-orange-600",
} as const;

const COLLAPSIBLE_TAGS = new Set(["text", "definition"]);

export function XMLRenderer({ xmlContent, className }: XMLRendererProps) {
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
      const isCollapsible = COLLAPSIBLE_TAGS.has(node.tagName || "");
      const colorClass =
        XML_TAG_COLORS[node.tagName as keyof typeof XML_TAG_COLORS] ||
        "text-gray-600";

      // Extract UUID for sections/articles for scroll-to functionality
      const uuidMatch = node.children?.find(
        (child) => child.type === "tag" && child.tagName === "uuid",
      );
      const sectionUuid = uuidMatch?.children?.find(
        (child) => child.type === "text",
      )?.content;

      const dataAttributes = sectionUuid
        ? { "data-section-uuid": sectionUuid }
        : {};

      return (
        <div key={tagId} className="my-1" {...dataAttributes}>
          <div className="flex items-start">
            {/* Chevron in gutter */}
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

            {/* Tag content */}
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

    // Extract tag name
    const tagContent = xmlContent.slice(tagStart + 1, tagEnd);
    const tagName = tagContent.split(" ")[0]; // Handle tags with attributes

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
      isCollapsible: COLLAPSIBLE_TAGS.has(tagName),
    });

    currentIndex = closingTagStart + closingTag.length;
  }

  return nodes;
}
