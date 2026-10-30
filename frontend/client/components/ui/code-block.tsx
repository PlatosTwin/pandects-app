import * as React from "react";

import { CopyButton } from "@/components/ui/copy-button";
import { cn } from "@/lib/utils";

export interface CodeBlockProps {
  /** The code shown and copied. */
  code: string;
  /** Optional caption above the code (e.g. "bash", "Python"). */
  label?: React.ReactNode;
  /** Accessible name for the copy button. Defaults to "Copy code". */
  copyLabel?: string;
  /** Hide the copy button (e.g. while `code` is a placeholder). */
  copyDisabled?: boolean;
  /** Called after each copy attempt (e.g. for analytics). */
  onCopied?: (ok: boolean) => void;
  /** Wrap long lines instead of scrolling horizontally. */
  wrap?: boolean;
  className?: string;
}

/** Monospace block with a copy button; scrolls horizontally on overflow. */
export function CodeBlock({
  code,
  label,
  copyLabel = "Copy code",
  copyDisabled = false,
  wrap = false,
  onCopied,
  className,
}: CodeBlockProps) {
  // Unique per block where callers give unique copy labels ("Copy X" -> "X").
  const regionLabel =
    typeof label === "string"
      ? label
      : copyLabel.replace(/^copy\s+/i, "").replace(/^\w/, (c) => c.toUpperCase());

  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-lg border border-border bg-muted/50",
        className,
      )}
    >
      {label ? (
        <div className="border-b border-border px-3 py-1.5 text-xs font-medium text-muted-foreground">
          {label}
        </div>
      ) : null}
      <div className="flex items-start gap-2">
        <pre
          // Focusable so keyboard users can scroll a long line into view;
          // named as a region so screen readers announce what it is.
          tabIndex={wrap ? undefined : 0}
          role={wrap ? undefined : "region"}
          aria-label={wrap ? undefined : regionLabel}
          className={cn(
            "min-w-0 flex-1 overflow-x-auto rounded-md px-3 py-2.5 font-mono text-[13px] leading-relaxed text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring",
            wrap && "whitespace-pre-wrap break-all",
          )}
        >
          <code>{code}</code>
        </pre>
        <div className="shrink-0 p-1">
          <CopyButton
            value={code}
            label={copyLabel}
            disabled={copyDisabled}
            onCopied={onCopied}
          />
        </div>
      </div>
    </div>
  );
}
