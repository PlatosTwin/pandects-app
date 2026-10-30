import { useState, useMemo, useEffect, useRef } from "react";
import { ChevronDown, ChevronRight, FileText } from "lucide-react";
import { cn } from "@/lib/utils";
import { decodeXmlEntities } from "@/lib/text-utils";
import { logger } from "@/lib/logger";
import { prefersReducedMotion } from "@/lib/scroll";
import { TOCItem } from "@shared/agreement";
import type { ReaderOutlineNode } from "@/lib/reader-index";

interface TableOfContentsProps {
  xmlContent: string;
  targetSectionUuid?: string;
  /** Scroll-spy id (section uuid or region anchor id) of the current section. */
  activeItemId?: string | null;
  onSectionClick: (sectionUuid: string) => void;
  onAnchorClick?: (anchorId: string) => void;
  className?: string;
  scrollable?: boolean;
  /** Render the "Table of Contents" heading (off when a panel supplies its own). */
  showTitle?: boolean;
  /**
   * When set, only these sections (and the articles that contain them) are
   * listed, fully expanded. Used by the reader's section-type filter.
   */
  visibleSectionUuids?: ReadonlySet<string> | null;
  /** Sections that matched the originating search; marked with a dot. */
  matchedSectionUuids?: ReadonlySet<string> | null;
  /**
   * Smart reader: subsections of a section. The current section lists its
   * subsections under it; other sections stay one line each.
   */
  subsectionsFor?: (sectionUuid: string) => ReaderOutlineNode[];
  onSubsectionClick?: (outlineId: string) => void;
}

function filterTocItems(
  items: TOCItem[],
  visible: ReadonlySet<string>,
): TOCItem[] {
  const result: TOCItem[] = [];
  for (const item of items) {
    const children = item.children ? filterTocItems(item.children, visible) : [];
    const selfVisible = item.sectionUuid != null && visible.has(item.sectionUuid);
    if (selfVisible || children.length > 0) {
      result.push({ ...item, children: item.children ? children : undefined });
    }
  }
  return result;
}

function collectExpandableIds(items: TOCItem[], into: Set<string>): Set<string> {
  for (const item of items) {
    if (item.children && item.children.length > 0) {
      into.add(item.id);
      collectExpandableIds(item.children, into);
    }
  }
  return into;
}

const REGION_ITEMS: Array<{ tagName: string; title: string }> = [
  { tagName: "frontMatter", title: "Front Matter" },
  { tagName: "tableOfContents", title: "Table of Contents" },
  { tagName: "body", title: "Body" },
  { tagName: "sigPages", title: "Signature Pages" },
  { tagName: "backMatter", title: "Back Matter" },
];

export function TableOfContents({
  xmlContent,
  targetSectionUuid,
  activeItemId,
  onSectionClick,
  onAnchorClick,
  className,
  scrollable = true,
  showTitle = true,
  visibleSectionUuids = null,
  matchedSectionUuids = null,
  subsectionsFor,
  onSubsectionClick,
}: TableOfContentsProps) {
  const [expandedItems, setExpandedItems] = useState<Set<string>>(
    () => new Set(["region-body"]),
  );
  const rootRef = useRef<HTMLDivElement>(null);
  const isPointerOverRef = useRef(false);
  const lastInteractionRef = useRef(0);

  const allTocItems = useMemo(() => {
    return extractTOCFromXML(xmlContent);
  }, [xmlContent]);

  const isFiltered = visibleSectionUuids != null;
  const tocItems = useMemo(
    () =>
      visibleSectionUuids ? filterTocItems(allTocItems, visibleSectionUuids) : allTocItems,
    [allTocItems, visibleSectionUuids],
  );
  // A filtered list is short, so show it fully expanded.
  const filteredExpanded = useMemo(
    () => (isFiltered ? collectExpandableIds(tocItems, new Set()) : null),
    [isFiltered, tocItems],
  );

  useEffect(() => {
    if (!targetSectionUuid) return;
    setExpandedItems((prev) => {
      const next = new Set(prev);
      findAndExpandParents(tocItems, targetSectionUuid, next);
      return next;
    });
  }, [tocItems, targetSectionUuid]);

  useEffect(() => {
    if (!activeItemId) return;
    setExpandedItems((prev) => {
      const next = new Set(prev);
      findAndExpandParents(tocItems, activeItemId, next);
      return next.size === prev.size ? prev : next;
    });
  }, [tocItems, activeItemId]);

  // Keep the active entry visible inside the TOC's scroll container, unless
  // the user is interacting with the TOC themselves.
  useEffect(() => {
    if (!activeItemId) return;
    const root = rootRef.current;
    if (!root) return;
    // Wait a frame so an entry revealed by ancestor auto-expand exists.
    const frame = requestAnimationFrame(() => {
      if (isPointerOverRef.current) return;
      if (Date.now() - lastInteractionRef.current < 1000) return;
      const button = root.querySelector<HTMLElement>('[aria-current="true"]');
      if (!button) return;
      const scroller = findScrollContainer(button);
      if (!scroller) return;
      const scrollerRect = scroller.getBoundingClientRect();
      const buttonRect = button.getBoundingClientRect();
      let delta = 0;
      if (buttonRect.top < scrollerRect.top + 8) {
        delta = buttonRect.top - (scrollerRect.top + 8);
      } else if (buttonRect.bottom > scrollerRect.bottom - 8) {
        delta = buttonRect.bottom - (scrollerRect.bottom - 8);
      }
      if (delta === 0) return;
      scroller.scrollTo({
        top: scroller.scrollTop + delta,
        behavior: prefersReducedMotion() ? "auto" : "smooth",
      });
    });
    return () => cancelAnimationFrame(frame);
  }, [activeItemId, expandedItems]);

  const markInteraction = () => {
    lastInteractionRef.current = Date.now();
  };

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
    const isExpanded = (filteredExpanded ?? expandedItems).has(item.id);
    const hasChildren = item.children && item.children.length > 0;
    const isTarget =
      item.sectionUuid != null && item.sectionUuid === targetSectionUuid;
    const isActive =
      activeItemId != null &&
      (item.sectionUuid === activeItemId || item.anchorId === activeItemId);
    const isMatched =
      item.sectionUuid != null && matchedSectionUuids?.has(item.sectionUuid) === true;

    return (
      <div key={item.id} className="select-none">
        <button
          type="button"
          className={cn(
            "flex w-full items-start gap-2 rounded-md px-2.5 py-1.5 text-left text-sm text-foreground transition-colors",
            "hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
            (isActive || isTarget) && "bg-primary/10 text-primary font-medium",
            depth > 0 && "ml-3 w-[calc(100%-0.75rem)]",
          )}
          onClick={() => {
            if (hasChildren && !isFiltered) {
              toggleExpanded(item.id);
            }
            if (item.sectionUuid) {
              onSectionClick(item.sectionUuid);
            } else if (item.anchorId) {
              onAnchorClick?.(item.anchorId);
            }
          }}
          aria-current={isActive ? "true" : undefined}
          aria-expanded={hasChildren ? isExpanded : undefined}
          aria-label={
            item.sectionUuid || item.anchorId
              ? `Go to section: ${item.title}`
              : item.title
          }
        >
          <div className="mt-[0.3em] flex h-3 w-4 flex-shrink-0 items-center justify-center">
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
          <span className="min-w-0 flex-1 break-words leading-snug [overflow-wrap:anywhere]">
            {item.title}
          </span>
          {isMatched ? (
            <span
              className="mt-[0.45em] h-1.5 w-1.5 shrink-0 rounded-full bg-primary"
              title="Matched"
            >
              <span className="sr-only">(matched)</span>
            </span>
          ) : null}
        </button>

        {hasChildren && isExpanded && (
          <div className="ml-2">
            {item.children!.map((child) => renderTOCItem(child, depth + 1))}
          </div>
        )}
        {isActive && !isFiltered && item.sectionUuid ? renderSubsections(item.sectionUuid) : null}
      </div>
    );
  };

  const renderSubsections = (sectionUuid: string): React.ReactNode => {
    const subs = subsectionsFor?.(sectionUuid);
    if (!subs || subs.length === 0) return null;
    return (
      <ul className="ml-9 mt-0.5 space-y-px border-l border-border pl-2" aria-label="Subsections">
        {subs.map((sub) => (
          <li key={sub.id}>
            <button
              type="button"
              className="w-full truncate rounded-md px-2 py-1 text-left text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              onClick={() => onSubsectionClick?.(sub.id)}
              title={decodeXmlEntities(sub.title)}
            >
              {decodeXmlEntities(sub.title) || sub.label}
            </button>
          </li>
        ))}
      </ul>
    );
  };

  return (
    // These listeners only observe user activity to pause the auto-scroll
    // nudge; they add no behavior that needs a keyboard equivalent.
    // eslint-disable-next-line jsx-a11y/no-static-element-interactions
    <div
      ref={rootRef}
      className={cn(scrollable && "overflow-y-auto", className)}
      onPointerEnter={() => {
        isPointerOverRef.current = true;
      }}
      onPointerLeave={() => {
        isPointerOverRef.current = false;
      }}
      onPointerDown={markInteraction}
      onWheel={markInteraction}
      onTouchMove={markInteraction}
      onKeyDown={markInteraction}
    >
      <div className={showTitle ? "p-4" : "p-2"}>
        {showTitle ? (
          <h2 className="mb-3 text-sm font-medium text-foreground">
            Table of Contents
          </h2>
        ) : null}
        <nav aria-label="Agreement table of contents">
          <div className="space-y-0.5">
            {tocItems.length === 0 && isFiltered ? (
              <p className="px-2.5 py-2 text-sm text-muted-foreground">
                No sections of this type.
              </p>
            ) : (
              tocItems.map((item) => renderTOCItem(item))
            )}
          </div>
        </nav>
      </div>
    </div>
  );
}

// targetId matches either a section uuid or a region anchor id.
function findAndExpandParents(
  items: TOCItem[],
  targetId: string,
  expandedSet: Set<string>,
): boolean {
  for (const item of items) {
    if (item.sectionUuid === targetId || item.anchorId === targetId) {
      if (item.children && item.children.length > 0) {
        expandedSet.add(item.id);
      }
      return true;
    }

    if (
      item.children &&
      findAndExpandParents(item.children, targetId, expandedSet)
    ) {
      expandedSet.add(item.id);
      return true;
    }
  }
  return false;
}

// The TOC's scroll container differs by surface: the component root when
// scrollable, otherwise an ancestor (the desktop sidebar `<aside>`).
function findScrollContainer(element: HTMLElement): HTMLElement | null {
  let node = element.parentElement;
  while (node) {
    if (node.scrollHeight > node.clientHeight) {
      const overflowY = window.getComputedStyle(node).overflowY;
      if (overflowY === "auto" || overflowY === "scroll") return node;
    }
    node = node.parentElement;
  }
  return null;
}

function extractTOCFromXML(xmlContent: string): TOCItem[] {
  const items: TOCItem[] = [];
  let itemCounter = 0;

  try {
    const bodyMatch = xmlContent.match(/<body[^>]*>(.*?)<\/body>/s);
    const contentToScan = bodyMatch ? bodyMatch[1] : xmlContent;
    const bodyChildren: TOCItem[] = [];

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
        ? decodeXmlEntities(titleMatch[1])
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
          ? decodeXmlEntities(sectionTitleMatch[1])
          : `Section ${articleItem.children!.length + 1}`;

        // Extract UUID from section attributes
        const sectionUuidMatch = sectionAttributes.match(/uuid="([^"]*)"/);
        const sectionUuid = sectionUuidMatch ? sectionUuidMatch[1] : undefined;

        articleItem.children!.push({
          id: `section-${itemCounter++}`,
          title: sectionTitle,
          level: 2,
          sectionUuid: sectionUuid,
        });
      }

      // Sections are already in document order from the XML scan; keep that
      // order so the TOC matches the rendered document.
      bodyChildren.push(articleItem);
    }

    // Also look for any standalone sections that might not be in articles
    const standaloneSectionMatches = contentToScan.matchAll(
      /<section([^>]*)>(?!.*<\/article>)/gs,
    );

    for (const sectionMatch of standaloneSectionMatches) {
      const sectionAttributes = sectionMatch[1];

      const titleMatch = sectionAttributes.match(/title="([^"]*)"/);
      const sectionTitle = titleMatch
        ? decodeXmlEntities(titleMatch[1])
        : `Section ${itemCounter + 1}`;

      const uuidMatch = sectionAttributes.match(/uuid="([^"]*)"/);
      const sectionUuid = uuidMatch ? uuidMatch[1] : undefined;

      bodyChildren.push({
        id: `standalone-section-${itemCounter++}`,
        title: sectionTitle,
        level: 1,
        sectionUuid: sectionUuid,
      });
    }

    for (const region of REGION_ITEMS) {
      if (!xmlContent.includes(`<${region.tagName}`)) continue;

      items.push({
        id: `region-${region.tagName}`,
        title: region.title,
        level: 1,
        anchorId: `agreement-region-${region.tagName}`,
        children: region.tagName === "body" ? bodyChildren : undefined,
      });
    }

    if (items.length === 0) return bodyChildren;
  } catch (error) {
    logger.error("Error parsing XML for TOC:", error);
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
