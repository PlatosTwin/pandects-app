import type { ReactNode } from "react";
import * as ToggleGroupPrimitive from "@radix-ui/react-toggle-group";

import { cn } from "@/lib/utils";

export interface SegmentedControlOption<T extends string> {
  value: T;
  label: ReactNode;
  disabled?: boolean;
}

export interface SegmentedControlProps<T extends string> {
  value: T;
  onValueChange: (value: T) => void;
  options: ReadonlyArray<SegmentedControlOption<T>>;
  /** Accessible name for the group, e.g. "Chart metric". */
  "aria-label": string;
  size?: "sm" | "default";
  className?: string;
}

/**
 * Single-choice pill switcher (a Radix ToggleGroup that can never be
 * deselected). Use for view/metric switches; use Tabs when the choice swaps
 * whole panels of content.
 */
export function SegmentedControl<T extends string>({
  value,
  onValueChange,
  options,
  size = "default",
  className,
  ...rest
}: SegmentedControlProps<T>) {
  return (
    <ToggleGroupPrimitive.Root
      type="single"
      value={value}
      onValueChange={(next) => {
        // Radix emits "" when the active item is clicked again; ignore it.
        if (next) onValueChange(next as T);
      }}
      aria-label={rest["aria-label"]}
      className={cn(
        "inline-flex max-w-full items-center gap-0.5 overflow-x-auto rounded-lg border border-border bg-muted/50 p-0.5",
        className,
      )}
    >
      {options.map((option) => (
        <ToggleGroupPrimitive.Item
          key={option.value}
          value={option.value}
          disabled={option.disabled}
          className={cn(
            "inline-flex shrink-0 items-center justify-center whitespace-nowrap rounded-md font-medium text-muted-foreground transition-colors",
            "hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-1 focus-visible:ring-offset-background",
            "disabled:pointer-events-none disabled:opacity-50",
            "data-[state=on]:bg-background data-[state=on]:text-foreground data-[state=on]:shadow-raised",
            size === "sm" ? "h-7 px-2.5 text-xs" : "h-8 px-3 text-sm",
          )}
        >
          {option.label}
        </ToggleGroupPrimitive.Item>
      ))}
    </ToggleGroupPrimitive.Root>
  );
}
