import type { ReactNode } from "react";

import { Eyebrow } from "@/components/ui/eyebrow";
import { cn } from "@/lib/utils";

export interface PageHeaderProps {
  /** Optional small label above the title. */
  eyebrow?: ReactNode;
  title: ReactNode;
  /** One or two sentences under the title. */
  description?: ReactNode;
  /** Right-aligned controls on desktop; stacked under the text on mobile. */
  actions?: ReactNode;
  /** id for the h1 (e.g. for aria-labelledby). */
  titleId?: string;
  className?: string;
}

/**
 * Page title block. The h1 is sized with clamp() so short titles never wrap
 * at desktop width. PageShell renders it for you when given `title`; render it
 * directly only for pages that build their own frame.
 */
export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
  titleId,
  className,
}: PageHeaderProps) {
  return (
    <header
      className={cn(
        "mb-6 flex flex-col gap-3 sm:mb-8 sm:flex-row sm:items-end sm:justify-between sm:gap-6",
        className,
      )}
    >
      <div className="min-w-0">
        {eyebrow ? <Eyebrow className="mb-2">{eyebrow}</Eyebrow> : null}
        <h1
          id={titleId}
          className="text-[clamp(1.625rem,1.25rem+1.25vw,2rem)] font-semibold leading-tight tracking-tight text-foreground"
        >
          {title}
        </h1>
        {description ? (
          <div className="mt-2 text-sm text-muted-foreground sm:text-base">
            {description}
          </div>
        ) : null}
      </div>
      {actions ? (
        <div className="flex flex-shrink-0 flex-wrap items-center gap-2">
          {actions}
        </div>
      ) : null}
    </header>
  );
}
