import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

import { PageHeader } from "@/components/PageHeader";

/** @deprecated Use `layout` instead. Kept until every page is migrated. */
type PageShellSize = "md" | "lg" | "xl" | "full";

export type PageShellLayout = "docs" | "data" | "form";

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
   */
  layout?: PageShellLayout;
  eyebrow?: ReactNode;
  title?: ReactNode;
  /** Text under the title. Alias: `subtitle` (deprecated name). */
  description?: ReactNode;
  /** @deprecated Use `description`. */
  subtitle?: ReactNode;
  actions?: ReactNode;
  /** Docs layout only: table of contents (e.g. `<DocsTableOfContents>`). */
  toc?: ReactNode;
  /**
   * @deprecated Use `layout`. Only honored when `layout` is not set; renders
   * the old centered frame so pages that are not migrated yet keep their look.
   */
  size?: PageShellSize;
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
  subtitle,
  actions,
  toc,
  size = "lg",
  className,
  contentClassName,
  children,
}: PageShellProps) {
  const body = description ?? subtitle;
  const header =
    title || body || actions || eyebrow ? (
      <PageHeader
        eyebrow={eyebrow}
        title={title}
        description={body}
        actions={actions}
      />
    ) : null;

  if (!layout) {
    return (
      <LegacyPageShell size={size} className={className} header={header}>
        {children}
      </LegacyPageShell>
    );
  }

  if (layout === "docs") {
    return (
      <div className={cn(FRAME, className)}>
        {toc ? (
          <div className="mx-auto max-w-[1008px]">
            {header}
            <div className="lg:grid lg:grid-cols-[240px_minmax(0,720px)] lg:gap-12">
              <aside className="mb-6 lg:mb-0">
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

  if (layout === "form") {
    return (
      <div className={cn(FRAME, className)}>
        <div className="mx-auto max-w-md">
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

function LegacyPageShell({
  size,
  className,
  header,
  children,
}: {
  size: PageShellSize;
  className?: string;
  header: ReactNode;
  children: ReactNode;
}) {
  const sizeClass =
    size === "md"
      ? "max-w-4xl"
      : size === "lg"
        ? "max-w-5xl"
        : size === "xl"
          ? "max-w-7xl"
          : "max-w-none";

  return (
    <div
      className={cn(
        "mx-auto w-full flex-1 px-4 py-8 sm:px-6 lg:px-8",
        sizeClass,
        className,
      )}
    >
      {header}
      {children}
    </div>
  );
}
