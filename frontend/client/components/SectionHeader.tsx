import type { ReactNode } from "react";

type SectionHeaderProps = {
  title: string;
  id: string;
  /** Optional numbered chip ("01"). Off by default; use only for real sequences. */
  number?: string;
  /** Optional sentence under the title. */
  description?: ReactNode;
  actions?: ReactNode;
};

export function SectionHeader({
  number,
  title,
  id,
  description,
  actions,
}: SectionHeaderProps) {
  const heading = (
    <div className="min-w-0">
      <h2
        id={id}
        className="scroll-mt-24 text-xl font-semibold tracking-tight text-foreground sm:text-2xl"
      >
        {title}
      </h2>
      {description ? (
        <div className="mt-1 text-sm text-muted-foreground">{description}</div>
      ) : null}
    </div>
  );

  return (
    <div className="flex items-center gap-3">
      {number ? (
        <span className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-primary/10 text-xs font-medium tabular-nums text-primary">
          {number}
        </span>
      ) : null}
      {actions ? (
        <div className="flex flex-1 flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          {heading}
          {actions}
        </div>
      ) : (
        heading
      )}
    </div>
  );
}
