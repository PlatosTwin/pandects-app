import * as React from "react";
import { ArrowUpRight } from "lucide-react";

import { cn } from "@/lib/utils";

export interface ExternalLinkProps
  extends React.AnchorHTMLAttributes<HTMLAnchorElement> {
  href: string;
  /** Hide the trailing glyph (e.g. when the link wraps an icon already). */
  hideIcon?: boolean;
}

/**
 * Link to another site: opens in a new tab with `rel="noopener noreferrer"`,
 * shows a small arrow glyph, and tells screen readers it opens a new tab.
 */
export const ExternalLink = React.forwardRef<HTMLAnchorElement, ExternalLinkProps>(
  ({ className, children, hideIcon = false, ...props }, ref) => (
    <a
      ref={ref}
      target="_blank"
      rel="noopener noreferrer"
      className={cn("inline-flex items-baseline gap-0.5", className)}
      {...props}
    >
      {children}
      {hideIcon ? null : (
        <ArrowUpRight
          className="h-3.5 w-3.5 shrink-0 self-center opacity-70"
          aria-hidden="true"
        />
      )}
      <span className="sr-only"> (opens in a new tab)</span>
    </a>
  ),
);
ExternalLink.displayName = "ExternalLink";
