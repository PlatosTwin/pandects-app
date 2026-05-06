import { Check, X } from "lucide-react";
import { cn } from "@/lib/utils";
import type { TagColor } from "@/lib/favorites-api";

const COLOR_CLASSES: Record<TagColor, string> = {
  slate:
    "bg-slate-100 text-slate-800 ring-slate-200 dark:bg-slate-800/40 dark:text-slate-200 dark:ring-slate-700",
  red: "bg-red-100 text-red-800 ring-red-200 dark:bg-red-900/30 dark:text-red-200 dark:ring-red-800",
  orange:
    "bg-orange-100 text-orange-800 ring-orange-200 dark:bg-orange-900/30 dark:text-orange-200 dark:ring-orange-800",
  amber:
    "bg-amber-100 text-amber-800 ring-amber-200 dark:bg-amber-900/30 dark:text-amber-200 dark:ring-amber-800",
  green:
    "bg-green-100 text-green-800 ring-green-200 dark:bg-green-900/30 dark:text-green-200 dark:ring-green-800",
  teal: "bg-teal-100 text-teal-800 ring-teal-200 dark:bg-teal-900/30 dark:text-teal-200 dark:ring-teal-800",
  blue: "bg-blue-100 text-blue-800 ring-blue-200 dark:bg-blue-900/30 dark:text-blue-200 dark:ring-blue-800",
  violet:
    "bg-violet-100 text-violet-800 ring-violet-200 dark:bg-violet-900/30 dark:text-violet-200 dark:ring-violet-800",
};

const SWATCH_CLASSES: Record<TagColor, string> = {
  slate: "bg-slate-400",
  red: "bg-red-500",
  orange: "bg-orange-500",
  amber: "bg-amber-500",
  green: "bg-green-500",
  teal: "bg-teal-500",
  blue: "bg-blue-500",
  violet: "bg-violet-500",
};

export function tagPillClassName(color: TagColor): string {
  return COLOR_CLASSES[color] ?? COLOR_CLASSES.slate;
}

export function TagSwatch({
  color,
  className,
}: {
  color: TagColor;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-block h-2.5 w-2.5 rounded-full",
        SWATCH_CLASSES[color] ?? SWATCH_CLASSES.slate,
        className,
      )}
      aria-hidden="true"
    />
  );
}

export interface TagPillProps {
  name: string;
  color: TagColor;
  onRemove?: () => void;
  selected?: boolean;
  onClick?: () => void;
  className?: string;
}

export function TagPill({
  name,
  color,
  onRemove,
  onClick,
  selected,
  className,
}: TagPillProps) {
  const interactive = Boolean(onClick);
  const Tag: "button" | "span" = interactive ? "button" : "span";
  return (
    <Tag
      type={interactive ? "button" : undefined}
      onClick={onClick}
      aria-pressed={interactive ? Boolean(selected) : undefined}
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset",
        tagPillClassName(color),
        interactive &&
          "min-h-11 cursor-pointer transition-shadow hover:shadow-sm sm:min-h-0",
        selected &&
          "ring-2 ring-foreground ring-offset-2 ring-offset-background shadow-sm",
        onRemove && "py-1 pr-1",
        className,
      )}
    >
      <TagSwatch color={color} className="h-1.5 w-1.5" />
      <span className="truncate">{name}</span>
      {interactive && selected ? (
        <Check className="h-3 w-3 shrink-0" aria-hidden="true" />
      ) : null}
      {onRemove ? (
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
          aria-label={`Remove tag ${name}`}
          className="ml-0.5 inline-flex h-11 w-11 items-center justify-center rounded-full hover:bg-black/5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring sm:h-5 sm:w-5 dark:hover:bg-white/10"
        >
          <X className="h-3 w-3" />
        </button>
      ) : null}
    </Tag>
  );
}
