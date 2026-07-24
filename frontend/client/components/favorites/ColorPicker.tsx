import { Check } from "lucide-react";

import { TagSwatch } from "@/components/favorites/TagPill";
import { TAG_COLORS, type TagColor } from "@/lib/favorites-api";

/**
 * Row of tag-color swatch buttons. Renders as a fragment so callers keep their
 * own flex container (some place a sibling action button in the same row).
 */
export function ColorPicker({
  value,
  onChange,
  buttonClassName = "sm:h-6 sm:w-6",
}: {
  value: TagColor;
  onChange: (color: TagColor) => void;
  /** Compact-viewport size overrides for each swatch button. */
  buttonClassName?: string;
}) {
  return (
    <>
      {TAG_COLORS.map((color) => (
        <button
          key={color}
          type="button"
          onClick={() => onChange(color)}
          aria-label={`Use ${color} color`}
          aria-pressed={value === color}
          className={
            `relative inline-grid h-11 w-11 place-items-center rounded-full ring-1 ring-transparent transition-shadow ${buttonClassName} ` +
            (value === color
              ? "ring-2 ring-foreground ring-offset-2 ring-offset-background"
              : "hover:ring-muted-foreground/40")
          }
        >
          <TagSwatch color={color} className="h-4 w-4" />
          {value === color ? (
            <Check className="pointer-events-none absolute left-1/2 top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 text-white drop-shadow" />
          ) : null}
        </button>
      ))}
    </>
  );
}
