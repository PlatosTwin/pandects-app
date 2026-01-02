import { useState, useMemo, useEffect } from "react";
import { ChevronDown, ChevronRight, FileText } from "lucide-react";
import { cn } from "@/lib/utils";
import { TOCItem } from "@shared/agreement";

interface TableOfContentsProps {
  xmlContent: string;
  targetSectionUuid?: string;
  onSectionClick: (sectionUuid: string) => void;
  className?: string;
}

export function TableOfContents({
  xmlContent,
  targetSectionUuid,
  onSectionClick,
  className,
}: TableOfContentsProps) {
  const [expandedItems, setExpandedItems] = useState<Set<string>>(new Set());

  const tocItems = useMemo(() => {
    return extractTOCFromXML(xmlContent);
  }, [xmlContent]);

  useEffect(() => {
    if (!targetSectionUuid) return;
    const expandedSet = new Set<string>();
    findAndExpandParents(tocItems, targetSectionUuid, expandedSet);
    setExpandedItems(expandedSet);
  }, [tocItems, targetSectionUuid]);

  const toggleExpanded = (itemId: string) => {
    setExpandedItems((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(itemId)) {
        newSet.delete(itemId);
      } else {
        newSet.add(itemId);
      }
      return newSet;
    });
  };

  const renderTOCItem = (item: TOCItem, depth: number = 0): React.ReactNode => {
    const isExpanded = expandedItems.has(item.id);
    const hasChildren = item.children && item.children.length > 0;
    const isTarget = item.sectionUuid === targetSectionUuid;

    return (
      <div key={item.id} className="select-none">
        <button
          type="button"
          className={cn(
            "flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm text-foreground transition-colors",
            "hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
            isTarget && "bg-primary/10 text-primary font-medium",
            depth > 0 && "ml-4",
          )}
          onClick={() => {
            if (hasChildren) {
              toggleExpanded(item.id);
            }
            if (item.sectionUuid) {
              onSectionClick(item.sectionUuid);
            }
          }}
          aria-expanded={hasChildren ? isExpanded : undefined}
        >
          <div className="w-4 flex-shrink-0 flex items-center justify-center">
            {hasChildren ? (
              isExpanded ? (
                <ChevronDown className="w-3 h-3 text-muted-foreground" aria-hidden="true" />
              ) : (
                <ChevronRight className="w-3 h-3 text-muted-foreground" aria-hidden="true" />
              )
            ) : (
              <FileText className="w-3 h-3 text-muted-foreground" aria-hidden="true" />
            )}
          </div>
          <span
            className="truncate leading-relaxed"
            title={item.title}
          >
            {item.title}
          </span>
        </button>

        {hasChildren && isExpanded && (
          <div className="ml-2">
            {item.children!.map((child) => renderTOCItem(child, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className={cn("overflow-y-auto", className)}>
      <div className="p-4">
        <h3 className="mb-3 text-sm font-medium text-foreground">
          Table of Contents
        </h3>
        <nav aria-label="Agreement table of contents">
          <div className="space-y-1">
            {tocItems.map((item) => renderTOCItem(item))}
          </div>
        </nav>
      </div>
    </div>
  );
}

function findAndExpandParents(
  items: TOCItem[],
  targetUuid: string,
  expandedSet: Set<string>,
): boolean {
  for (const item of items) {
    if (item.sectionUuid === targetUuid) {
      return true;
    }

    if (
      item.children &&
      findAndExpandParents(item.children, targetUuid, expandedSet)
    ) {
      expandedSet.add(item.id);
      return true;
    }
  }
  return false;
}

function extractTOCFromXML(xmlContent: string): TOCItem[] {
  const items: TOCItem[] = [];
  let itemCounter = 0;

  try {
    // First, let's check if there's a body tag and extract content from it
    const bodyMatch = xmlContent.match(/<body[^>]*>(.*?)<\/body>/s);
    const contentToScan = bodyMatch ? bodyMatch[1] : xmlContent;

    // Extract all articles from the body
    const articleMatches = contentToScan.matchAll(
      /<article([^>]*)>(.*?)<\/article>/gs,
    );

    for (const articleMatch of articleMatches) {
      const articleAttributes = articleMatch[1];
      const articleContent = articleMatch[2];

      // Extract title from article attributes
      const titleMatch = articleAttributes.match(/title="([^"]*)"/);
      const articleTitle = titleMatch
        ? titleMatch[1]
        : `Article ${itemCounter + 1}`;

      // Extract UUID from article attributes
      const uuidMatch = articleAttributes.match(/uuid="([^"]*)"/);
      const articleUuid = uuidMatch ? uuidMatch[1] : undefined;

      const articleItem: TOCItem = {
        id: `article-${itemCounter++}`,
        title: articleTitle,
        level: 1,
        sectionUuid: articleUuid,
        children: [],
      };

      // Extract sections within this article
      const sectionMatches = articleContent.matchAll(
        /<section([^>]*)>(.*?)<\/section>/gs,
      );

      for (const sectionMatch of sectionMatches) {
        const sectionAttributes = sectionMatch[1];

        // Extract title from section attributes
        const sectionTitleMatch = sectionAttributes.match(/title="([^"]*)"/);
        const sectionTitle = sectionTitleMatch
          ? sectionTitleMatch[1]
          : `Section ${articleItem.children!.length + 1}`;

        // Extract UUID from section attributes
        const sectionUuidMatch = sectionAttributes.match(/uuid="([^"]*)"/);
        const sectionUuid = sectionUuidMatch ? sectionUuidMatch[1] : undefined;

        // Extract order attribute if present for better sorting
        const orderMatch = sectionAttributes.match(/order="([^"]*)"/);
        const order = orderMatch
          ? parseInt(orderMatch[1], 10)
          : articleItem.children!.length + 1;

        articleItem.children!.push({
          id: `section-${itemCounter++}`,
          title: sectionTitle,
          level: 2,
          sectionUuid: sectionUuid,
        });
      }

      // Sort sections by order if available
      if (articleItem.children!.length > 0) {
        articleItem.children!.sort((a, b) => {
          // If we can extract order numbers from titles, use those
          const aOrder = extractOrderFromTitle(a.title);
          const bOrder = extractOrderFromTitle(b.title);
          return aOrder - bOrder;
        });
      }

      items.push(articleItem);
    }

    // Also look for any standalone sections that might not be in articles
    const standaloneSectionMatches = contentToScan.matchAll(
      /<section([^>]*)>(?!.*<\/article>)/gs,
    );

    for (const sectionMatch of standaloneSectionMatches) {
      const sectionAttributes = sectionMatch[1];

      const titleMatch = sectionAttributes.match(/title="([^"]*)"/);
      const sectionTitle = titleMatch
        ? titleMatch[1]
        : `Section ${itemCounter + 1}`;

      const uuidMatch = sectionAttributes.match(/uuid="([^"]*)"/);
      const sectionUuid = uuidMatch ? uuidMatch[1] : undefined;

      items.push({
        id: `standalone-section-${itemCounter++}`,
        title: sectionTitle,
        level: 1,
        sectionUuid: sectionUuid,
      });
    }
  } catch (error) {
    if (import.meta.env.DEV) {
      console.error("Error parsing XML for TOC:", error);
    }
    // Return a basic structure if parsing fails
    return [
      {
        id: "error-item",
        title: "Unable to parse document structure",
        level: 1,
      },
    ];
  }

  return items;
}

function extractOrderFromTitle(title: string): number {
  // Try to extract section numbers like "1.1", "2.3", etc.
  const match = title.match(/^(\d+)\.?(\d*)/);
  if (match) {
    const major = parseInt(match[1], 10) || 0;
    const minor = parseInt(match[2], 10) || 0;
    return major * 1000 + minor; // This ensures proper sorting
  }
  return 9999; // Put items without numbers at the end
}
