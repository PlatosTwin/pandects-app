import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
  {
    variants: {
      variant: {
        default:
          "border-transparent bg-primary text-primary-foreground hover:bg-primary/80",
        secondary:
          "border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80",
        destructive:
          "border-transparent bg-destructive text-destructive-foreground hover:bg-destructive/80",
        outline: "text-foreground",
        /** Status chip (e.g. "Verified"): emerald tint with a soft ring. */
        success:
          "border-transparent bg-emerald-500/10 px-2 font-medium text-emerald-700 ring-1 ring-emerald-500/20 dark:text-emerald-300",
        /** Neutral metadata pill: on-background with a border ring. */
        metadata:
          "border-transparent bg-background px-2 font-medium text-muted-foreground ring-1 ring-border",
        /** Soft muted pill (article/section titles, enum values, counts). */
        muted:
          "border-transparent bg-muted px-2 font-medium text-muted-foreground",
        /** Micro-label chip (11px, bordered, muted fill): eval labels, doc tags. */
        chip: "border-border bg-muted/40 px-2 text-[11px] text-foreground",
        /** Primary-tinted count pill (taxonomy group/type counts). */
        count: "border-primary/20 bg-primary/10 text-primary",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

/** Renders a span so badges are valid inside buttons, headings, and prose. */
function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <span className={cn(badgeVariants({ variant }), className)} {...props} />
  );
}

export { Badge, badgeVariants };
