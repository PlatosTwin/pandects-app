import * as React from "react";
import { Link, type LinkProps } from "react-router-dom";
import { ArrowUpRight } from "lucide-react";

export interface ReferenceLinkProps extends Omit<LinkProps, "target" | "rel"> {
  /** Hide the trailing glyph (e.g. when the link is an icon-only trigger). */
  hideIcon?: boolean;
}

/**
 * Link from a working surface (search, the reader, a sign-up prompt) to an
 * in-app reference page such as Taxonomy or Sources & Methods. It opens in a
 * new tab so the user keeps their results, filters and scroll position, and it
 * says so to screen readers. Use `ExternalLink` for other sites.
 */
export const ReferenceLink = React.forwardRef<HTMLAnchorElement, ReferenceLinkProps>(
  ({ className, children, hideIcon = false, ...props }, ref) => (
    <Link
      ref={ref}
      target="_blank"
      rel="noopener"
      className={className}
      {...props}
    >
      {children}
      {hideIcon ? null : (
        // Inline, not flex, so the link text can wrap inside running prose.
        <ArrowUpRight className="ml-0.5 inline-block h-3.5 w-3.5 align-[-0.125em] opacity-70" aria-hidden="true" />
      )}
      <span className="sr-only"> (opens in a new tab)</span>
    </Link>
  ),
);
ReferenceLink.displayName = "ReferenceLink";
