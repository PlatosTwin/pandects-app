import * as React from "react";
import { Slot } from "@radix-ui/react-slot";

import { cn } from "@/lib/utils";

/**
 * Small uppercase label above a title, a table column group, or a stat.
 * One size and one tracking value for the whole app; use this instead of
 * hand-rolled `text-[11px] uppercase tracking-[...]` combinations.
 */
export const eyebrowClassName =
  "text-xs font-medium uppercase tracking-wider text-muted-foreground";

export interface EyebrowProps extends React.HTMLAttributes<HTMLElement> {
  /** Render as the child element (e.g. a `<dt>` or `<h2>`) instead of a `<p>`. */
  asChild?: boolean;
}

export const Eyebrow = React.forwardRef<HTMLParagraphElement, EyebrowProps>(
  ({ asChild = false, className, ...props }, ref) => {
    const Comp = asChild ? Slot : "p";
    return (
      <Comp ref={ref} className={cn(eyebrowClassName, className)} {...props} />
    );
  },
);
Eyebrow.displayName = "Eyebrow";
