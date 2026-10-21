import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

import { PageHeader } from "@/components/PageHeader";

export type PageShellLayout = "docs" | "data" | "form" | "settings";

export interface PageShellProps {
  /**
   * Page frame. The outer frame always matches the navigation bar
   * (`max-w-7xl px-4 sm:px-6 lg:px-8`). Layouts narrower than the frame are
   * centered in it, with the title inside the column so it shares the
   * content's edges; only `data` pages line the title up with the logo.
   * - `docs`: one 720px reading measure, centered; pass `toc` for a sticky
   *   240px table of contents on the left (lg+), and the whole 1008px
   *   block is centered instead.
   * - `data`: full `max-w-7xl` for tables, charts and dashboards.
   * - `form`: a centered `max-w-md` column.
   * - `settings`: a centered `max-w-4xl` column for account-style pages.
   */
  layout: PageShellLayout;
  eyebrow?: ReactNode;
  title?: ReactNode;
  /** Text under the title. */
  description?: ReactNode;
  actions?: ReactNode;
  /** Docs layout only: table of contents (e.g. `<DocsTableOfContents>`). */
  toc?: ReactNode;
  className?: string;
  /** Extra classes for the content column (not the header). */
  contentClassName?: string;
  children: ReactNode;
}

const FRAME = "mx-auto w-full max-w-7xl flex-1 px-4 py-8 sm:px-6 lg:px-8";

export function PageShell({
  layout,
  eyebrow,
  title,
  description,
  actions,
  toc,
  className,
  contentClassName,
  children,
}: PageShellProps) {
  const header =
    title || description || actions || eyebrow ? (
      <PageHeader
        eyebrow={eyebrow}
        title={title}
        description={description}
        actions={actions}
      />
    ) : null;

  if (layout === "docs") {
    return (
      <div className={cn(FRAME, className)}>
        {toc ? (
          <div className="mx-auto max-w-[1008px]">
            {header}
            <div className="lg:grid lg:grid-cols-[240px_minmax(0,720px)] lg:gap-12 print:block">
              <aside className="mb-6 lg:mb-0 print:hidden">
                <div className="lg:sticky lg:top-24">{toc}</div>
              </aside>
              <div className={cn("min-w-0", contentClassName)}>{children}</div>
            </div>
          </div>
        ) : (
          <div className={cn("mx-auto max-w-[720px]", contentClassName)}>
            {header}
            {children}
          </div>
        )}
      </div>
    );
  }

  if (layout === "form" || layout === "settings") {
    return (
      <div className={cn(FRAME, className)}>
        <div
          className={cn("mx-auto", layout === "form" ? "max-w-md" : "max-w-4xl")}
        >
          {header}
          <div className={contentClassName}>{children}</div>
        </div>
      </div>
    );
  }

  return (
    <div className={cn(FRAME, className)}>
      {header}
      <div className={cn("w-full", contentClassName)}>{children}</div>
    </div>
  );
}
