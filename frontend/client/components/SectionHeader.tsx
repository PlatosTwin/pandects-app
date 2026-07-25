import type { ReactNode } from "react";

type SectionHeaderProps = {
  number: string;
  title: string;
  id: string;
  actions?: ReactNode;
};

export function SectionHeader({ number, title, id, actions }: SectionHeaderProps) {
  const heading = (
    <h2
      id={id}
      className="text-2xl font-semibold tracking-tight text-foreground"
    >
      {title}
    </h2>
  );

  return (
    <div className="flex items-center gap-3">
      <span className="inline-flex h-7 w-7 items-center justify-center rounded-md bg-primary/10 text-primary text-xs font-medium tabular-nums">
        {number}
      </span>
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
