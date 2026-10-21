import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";

import { cn } from "@/lib/utils";

export interface StatusPageProps {
  /** Big code ("404") or an icon shown above the title. */
  code?: ReactNode;
  icon?: LucideIcon;
  title: ReactNode;
  /** One sentence. */
  description?: ReactNode;
  /** Primary and secondary action (usually two Buttons). */
  actions?: ReactNode;
  /** Extra content under the actions (e.g. a dev-only error dump). */
  children?: ReactNode;
  /** Use "h2" when the page already has an h1. */
  titleAs?: "h1" | "h2";
  role?: "alert" | "status";
  className?: string;
}

/**
 * Frame shared with the route skeleton. At least a viewport tall (minus the
 * h-16 nav), with the message centered in it, so a short status page reads as
 * deliberate rather than a strip at the top of an empty screen.
 */
export const STATUS_FRAME =
  "flex min-h-[calc(100svh-4rem)] app-frame flex-1 flex-col justify-center py-16";

/**
 * Full-page status message: 404, crash fallback, temporary lockdown. Centered
 * in the frame horizontally and vertically.
 */
export function StatusPage({
  code,
  icon: Icon,
  title,
  description,
  actions,
  children,
  titleAs: Title = "h1",
  role,
  className,
}: StatusPageProps) {
  return (
    <div role={role} className={cn(STATUS_FRAME, className)}>
      <div className="mx-auto w-full max-w-xl text-center">
        {code ? (
          <p className="font-mono text-sm font-medium text-primary">{code}</p>
        ) : null}
        {Icon ? (
          <Icon
            className="mx-auto h-8 w-8 text-muted-foreground"
            aria-hidden="true"
          />
        ) : null}
        <Title
          className={cn(
            "text-[clamp(1.625rem,1.25rem+1.25vw,2rem)] font-semibold leading-tight tracking-tight text-foreground",
            (code || Icon) && "mt-3",
          )}
        >
          {title}
        </Title>
        {description ? (
          <div className="mt-3 text-base text-muted-foreground">
            {description}
          </div>
        ) : null}
        {actions ? (
          <div className="mt-8 flex flex-wrap justify-center gap-3">
            {actions}
          </div>
        ) : null}
        {children ? <div className="text-left">{children}</div> : null}
      </div>
    </div>
  );
}
