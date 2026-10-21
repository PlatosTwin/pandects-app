import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";

import { Eyebrow } from "@/components/ui/eyebrow";
import { cn } from "@/lib/utils";

export interface FactCardProps {
  icon?: LucideIcon;
  eyebrow?: ReactNode;
  title: ReactNode;
  description?: ReactNode;
  /** Makes the whole card a link. Internal paths use the router; http(s) opens a new tab. */
  href?: string;
  /** Heading level for the title. Defaults to h3. */
  titleAs?: "h2" | "h3" | "h4";
  className?: string;
  children?: ReactNode;
}

/**
 * Small bordered card: icon, eyebrow, title, one or two sentences, and an
 * optional link. Used for channel lists, feature facts and "next step" cards.
 */
export function FactCard({
  icon: Icon,
  eyebrow,
  title,
  description,
  href,
  titleAs: Title = "h3",
  className,
  children,
}: FactCardProps) {
  // On phones the icon sits beside the text so a stack of cards stays short;
  // from sm up it sits above the text.
  const body = (
    <div className="flex items-start gap-4 sm:block">
      {Icon ? (
        <span className="inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary sm:mb-3">
          <Icon className="h-4 w-4" aria-hidden="true" />
        </span>
      ) : null}
      <div className="min-w-0">
        {eyebrow ? <Eyebrow className="mb-1">{eyebrow}</Eyebrow> : null}
        <Title className="flex items-center gap-1.5 text-base font-semibold tracking-tight text-foreground">
          {title}
          {href ? (
            <ArrowRight
              className="h-4 w-4 text-muted-foreground transition-transform group-hover:translate-x-0.5"
              aria-hidden="true"
            />
          ) : null}
        </Title>
        {description ? (
          <div className="mt-1.5 text-sm leading-relaxed text-muted-foreground">
            {description}
          </div>
        ) : null}
        {children}
      </div>
    </div>
  );

  const frame = cn(
    "block rounded-xl border border-border bg-card p-5",
    href &&
      "group transition-colors hover:border-primary/40 hover:bg-accent/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background",
    className,
  );

  if (!href) return <div className={frame}>{body}</div>;
  if (/^(https?:|mailto:)/.test(href)) {
    const external = href.startsWith("http");
    return (
      <a
        href={href}
        className={frame}
        {...(external ? { target: "_blank", rel: "noopener noreferrer" } : {})}
      >
        {body}
        {external ? <span className="sr-only"> (opens in a new tab)</span> : null}
      </a>
    );
  }
  return (
    <Link to={href} className={frame}>
      {body}
    </Link>
  );
}
