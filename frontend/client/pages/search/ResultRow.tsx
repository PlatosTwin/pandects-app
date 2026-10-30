import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { cn } from "@/lib/utils";
import type { ResultsDensity } from "@/components/ResultsToolbar";

export interface ResultRowProps {
  selected: boolean;
  onToggleSelected: () => void;
  /** Accessible name for the row checkbox, e.g. "Select Section 7.08 from Custom Health". */
  selectLabel: string;
  /** One line of context above the title (year, parties, value, badges). */
  meta: ReactNode;
  title: ReactNode;
  /** Small line under the title (e.g. taxonomy path). */
  subtitle?: ReactNode;
  /** Trailing action cluster (see RowActions). */
  actions: ReactNode;
  children?: ReactNode;
  density?: ResultsDensity;
  className?: string;
}

/**
 * The one result shell every search mode uses: selection checkbox, meta
 * line, title, body, and a consistent trailing action cluster. On phones the
 * actions move under the body; the checkbox stays visible at every width.
 */
export function ResultRow({
  selected,
  onToggleSelected,
  selectLabel,
  meta,
  title,
  subtitle,
  actions,
  children,
  density = "comfy",
  className,
}: ResultRowProps) {
  return (
    <div
      className={cn(
        "grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 rounded-lg border bg-card shadow-raised transition-[border-color,box-shadow] duration-150 motion-safe:animate-enter sm:grid-cols-[auto_minmax(0,1fr)_auto]",
        density === "compact" ? "px-3 py-2.5" : "px-4 py-4",
        selected ? "border-primary/50 bg-primary/5 shadow-lifted" : "border-border hover:border-primary/30 hover:shadow-lifted",
        className,
      )}
    >
      <div className="row-span-2 pt-0.5 sm:row-span-1">
        <Checkbox
          checked={selected}
          onCheckedChange={onToggleSelected}
          aria-label={selectLabel}
          className="h-5 w-5 sm:h-4 sm:w-4"
        />
      </div>
      <div className="min-w-0">
        <div className="flex min-w-0 flex-wrap items-center gap-x-1.5 gap-y-1 text-sm text-muted-foreground">
          {meta}
        </div>
        <div className={cn("min-w-0", density === "compact" ? "mt-0.5" : "mt-1")}>{title}</div>
        {subtitle ? <div className="mt-0.5 min-w-0 text-xs text-muted-foreground">{subtitle}</div> : null}
        {children ? <div className={density === "compact" ? "mt-1.5" : "mt-2.5"}>{children}</div> : null}
      </div>
      <div className="col-start-2 mt-3 flex items-start gap-1 sm:col-start-3 sm:row-start-1 sm:mt-0">
        {actions}
      </div>
    </div>
  );
}

/** Star · Flag · (extra) · Open, in that order, in every mode. */
export function RowActions({
  star,
  flag,
  extra,
  openHref,
  openLabel,
  onOpen,
}: {
  star: ReactNode;
  flag: ReactNode;
  extra?: ReactNode;
  openHref: string;
  /** Accessible name for the Open link. */
  openLabel: string;
  onOpen?: () => void;
}) {
  return (
    <>
      {star}
      {flag}
      {extra}
      <Button asChild size="sm" variant="outline" className="ml-auto h-9 px-3 sm:ml-1 sm:h-8">
        <Link to={openHref} onClick={onOpen} aria-label={openLabel}>
          Open
        </Link>
      </Button>
    </>
  );
}
