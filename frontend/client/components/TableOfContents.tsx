import { useState, useMemo } from "react";
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
        <div
          className={cn(
            "flex items-center gap-1 py-1 px-2 text-sm cursor-pointer rounded hover:bg-gray-100 transition-colors",
            isTarget && "bg-blue-50 text-blue-700 font-medium",
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
        >
          <div className="w-4 flex-shrink-0 flex items-center justify-center">
            {hasChildren ? (
              isExpanded ? (
                <ChevronDown className="w-3 h-3 text-gray-400" />
              ) : (
                <ChevronRight className="w-3 h-3 text-gray-400" />
              )
            ) : (
              <FileText className="w-3 h-3 text-gray-400" />
            )}
          </div>
          <span className="truncate text-gray-700" title={item.title}>
            {item.title}
          </span>
        </div>

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
        <h3 className="text-sm font-medium text-gray-900 mb-3">
          Table of Contents
        </h3>
        <div className="space-y-1">
          {tocItems.map((item) => renderTOCItem(item))}
        </div>
      </div>
    </div>
  );
}

function extractTOCFromXML(xmlContent: string): TOCItem[] {
  const items: TOCItem[] = [];
  let itemCounter = 0;

  // Simple XML parsing to extract sections and articles
  const articleMatches = xmlContent.matchAll(
    /<article[^>]*>(.*?)<\/article>/gs,
  );

  for (const articleMatch of articleMatches) {
    const articleContent = articleMatch[1];

    // Try to extract article title
    const titleMatch = articleContent.match(/<title[^>]*>(.*?)<\/title>/s);
    const articleTitle = titleMatch
      ? titleMatch[1].replace(/<[^>]*>/g, "").trim()
      : `Article ${itemCounter + 1}`;

    // Extract article UUID if present
    const uuidMatch = articleContent.match(/<uuid[^>]*>(.*?)<\/uuid>/s);
    const articleUuid = uuidMatch ? uuidMatch[1].trim() : undefined;

    const articleItem: TOCItem = {
      id: `article-${itemCounter++}`,
      title: articleTitle,
      level: 1,
      sectionUuid: articleUuid,
      children: [],
    };

    // Extract sections within this article
    const sectionMatches = articleContent.matchAll(
      /<section[^>]*>(.*?)<\/section>/gs,
    );

    for (const sectionMatch of sectionMatches) {
      const sectionContent = sectionMatch[1];

      const sectionTitleMatch = sectionContent.match(
        /<title[^>]*>(.*?)<\/title>/s,
      );
      const sectionTitle = sectionTitleMatch
        ? sectionTitleMatch[1].replace(/<[^>]*>/g, "").trim()
        : `Section ${articleItem.children!.length + 1}`;

      const sectionUuidMatch = sectionContent.match(
        /<uuid[^>]*>(.*?)<\/uuid>/s,
      );
      const sectionUuid = sectionUuidMatch
        ? sectionUuidMatch[1].trim()
        : undefined;

      articleItem.children!.push({
        id: `section-${itemCounter++}`,
        title: sectionTitle,
        level: 2,
        sectionUuid: sectionUuid,
      });
    }

    items.push(articleItem);
  }

  return items;
}
