import { useMemo, useState } from "react";
import { TableOfContents } from "@/components/TableOfContents";
import { badgeVariants } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export interface SectionTypeOption {
  value: string;
  label: string;
  count: number;
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
                title={option.label}
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
      />
    </div>
  );
}
