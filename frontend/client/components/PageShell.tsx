import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

type PageShellSize = "md" | "lg" | "xl" | "full";

export function PageShell({
  title,
  subtitle,
  actions,
  size = "lg",
  className,
  children,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  size?: PageShellSize;
  className?: string;
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
      {(title || subtitle || actions) && (
        <div className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
          <div className="min-w-0">
            {title && (
              <h1 className="text-3xl font-bold text-foreground">{title}</h1>
            )}
            {subtitle && (
              <div className="mt-2 text-sm text-muted-foreground sm:text-base">
                {subtitle}
              </div>
            )}
          </div>
          {actions && <div className="flex-shrink-0">{actions}</div>}
        </div>
      )}

      {children}
    </div>
  );
}

