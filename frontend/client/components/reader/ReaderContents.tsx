import { useMemo, useState } from "react";
import { TableOfContents } from "@/components/TableOfContents";
import { badgeVariants } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { ReaderOutlineNode } from "@/lib/reader-index";

export interface SectionTypeOption {
  value: string;
  label: string;
  count: number;
  /** Full taxonomy path ("Parent › Leaf"), for the chip tooltip. */
  title?: string;
}

const PATH_SEPARATOR = " › ";

/**
 * One chip per standard id in the section index. Leaf labels can repeat
 * across parents, so the tooltip always carries the taxonomy path and, when
 * two chips share a leaf label, the visible label adds the parent.
 */
export function buildSectionTypeOptions(
  sectionIndex: ReadonlyArray<{
    standard_id: readonly string[];
    standard_id_labels?: Record<string, string> | null;
  }>,
  pathById: Record<string, readonly string[]> = {},
): SectionTypeOption[] {
  const counts = new Map<string, number>();
  const leafLabels = new Map<string, string>();
  for (const item of sectionIndex) {
    for (const id of item.standard_id) {
      counts.set(id, (counts.get(id) ?? 0) + 1);
      const label = item.standard_id_labels?.[id];
      if (label && !leafLabels.has(id)) leafLabels.set(id, label);
    }
  }
  const leafOf = (id: string) => leafLabels.get(id) ?? pathById[id]?.at(-1) ?? id;
  const leafUses = new Map<string, number>();
  for (const id of counts.keys()) {
    const leaf = leafOf(id);
    leafUses.set(leaf, (leafUses.get(leaf) ?? 0) + 1);
  }
  return Array.from(counts.entries()).map(([value, count]) => {
    const leaf = leafOf(value);
    const path = pathById[value];
    const parent = path && path.length > 1 ? path[path.length - 2] : null;
    const collides = (leafUses.get(leaf) ?? 0) > 1;
    return {
      value,
      label: collides && parent ? `${parent}${PATH_SEPARATOR}${leaf}` : leaf,
      count,
      title: path && path.length > 1 ? path.join(PATH_SEPARATOR) : leaf,
    };
  });
}

interface ReaderContentsProps {
  xmlContent: string;
  activeItemId: string | null;
  targetSectionUuid?: string;
  onSectionClick: (sectionUuid: string) => void;
  onAnchorClick: (anchorId: string) => void;
  sectionTypeOptions: SectionTypeOption[];
  sectionType: string;
  onSectionTypeChange: (value: string) => void;
  /** Uuids of the sections tagged with `sectionType` (null = no filter). */
  visibleSectionUuids: ReadonlySet<string> | null;
  matchedSectionUuids: ReadonlySet<string> | null;
  /** Section-index failure: the type filter is unavailable, contents still work. */
  sectionIndexError?: string | null;
  /** Smart reader: a section's subsections, listed under the current section. */
  subsectionsFor?: (sectionUuid: string) => ReaderOutlineNode[];
  onSubsectionClick?: (outlineId: string) => void;
}

const COLLAPSED_CHIP_COUNT = 6;

/** Contents list with section-type filter chips. */
export function ReaderContents({
  xmlContent,
  activeItemId,
  targetSectionUuid,
  onSectionClick,
  onAnchorClick,
  sectionTypeOptions,
  sectionType,
  onSectionTypeChange,
  visibleSectionUuids,
  matchedSectionUuids,
  sectionIndexError,
  subsectionsFor,
  onSubsectionClick,
}: ReaderContentsProps) {
  const [showAllTypes, setShowAllTypes] = useState(false);
  const rankedOptions = useMemo(
    () =>
      [...sectionTypeOptions].sort(
        (a, b) => b.count - a.count || a.label.localeCompare(b.label),
      ),
    [sectionTypeOptions],
  );
  const visibleOptions = useMemo(() => {
    if (showAllTypes || rankedOptions.length <= COLLAPSED_CHIP_COUNT + 1) {
      return rankedOptions;
    }
    const head = rankedOptions.slice(0, COLLAPSED_CHIP_COUNT);
    // Keep the selected chip visible even when it ranks below the cut.
    const selected = rankedOptions.find((option) => option.value === sectionType);
    if (selected && !head.includes(selected)) head.push(selected);
    return head;
  }, [rankedOptions, sectionType, showAllTypes]);
  const hiddenCount = rankedOptions.length - visibleOptions.length;

  const chipClass = (active: boolean) =>
    cn(
      badgeVariants({ variant: active ? "count" : "chip" }),
      "max-w-full cursor-pointer gap-1 py-0.5 font-medium",
      !active && "hover:bg-muted",
    );

  return (
    <div className="space-y-2">
      {rankedOptions.length > 0 ? (
        <div
          role="group"
          aria-label="Filter contents by section type"
          className="flex flex-wrap gap-1.5 px-2 pt-1"
        >
          <button
            type="button"
            className={chipClass(sectionType === "all")}
            aria-pressed={sectionType === "all"}
            onClick={() => onSectionTypeChange("all")}
          >
            All sections
          </button>
          {visibleOptions.map((option) => {
            const active = option.value === sectionType;
            return (
              <button
                key={option.value}
                type="button"
                className={chipClass(active)}
                aria-pressed={active}
                onClick={() => onSectionTypeChange(active ? "all" : option.value)}
                title={option.title ?? option.label}
              >
                <span className="min-w-0 truncate">{option.label}</span>
                <span className="shrink-0 tabular-nums opacity-70">{option.count}</span>
              </button>
            );
          })}
          {hiddenCount > 0 || showAllTypes ? (
            <button
              type="button"
              className="rounded-full px-2 py-0.5 text-[11px] font-medium text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              onClick={() => setShowAllTypes((value) => !value)}
              aria-expanded={showAllTypes}
            >
              {showAllTypes ? "Fewer types" : `${hiddenCount} more`}
            </button>
          ) : null}
        </div>
      ) : sectionIndexError ? (
        <p className="px-3 pt-1 text-xs text-muted-foreground">
          Section types are unavailable right now.
        </p>
      ) : null}
      <TableOfContents
        xmlContent={xmlContent}
        targetSectionUuid={targetSectionUuid}
        activeItemId={activeItemId}
        onSectionClick={onSectionClick}
        onAnchorClick={onAnchorClick}
        scrollable={false}
        showTitle={false}
        visibleSectionUuids={visibleSectionUuids}
        matchedSectionUuids={matchedSectionUuids}
        subsectionsFor={subsectionsFor}
        onSubsectionClick={onSubsectionClick}
      />
    </div>
  );
}
