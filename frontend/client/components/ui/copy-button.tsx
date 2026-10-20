import * as React from "react";
import { Check, Copy } from "lucide-react";

import { Button, type ButtonProps } from "@/components/ui/button";
import { logger } from "@/lib/logger";
import { cn } from "@/lib/utils";

export type CopyState = "idle" | "copied" | "failed";

/**
 * Clipboard write with a transient "copied"/"failed" state that resets after
 * `resetMs`. Shared by CopyButton and any control that needs its own look.
 */
export function useCopyToClipboard(resetMs = 2000) {
  const [state, setState] = React.useState<CopyState>("idle");
  const timerRef = React.useRef<number | null>(null);

  React.useEffect(
    () => () => {
      if (timerRef.current) window.clearTimeout(timerRef.current);
    },
    [],
  );

  const copy = React.useCallback(
    async (text: string): Promise<boolean> => {
      let ok = true;
      try {
        await navigator.clipboard.writeText(text);
      } catch {
        logger.warn("Clipboard write failed.");
        ok = false;
      }
      setState(ok ? "copied" : "failed");
      if (timerRef.current) window.clearTimeout(timerRef.current);
      timerRef.current = window.setTimeout(() => {
        setState("idle");
        timerRef.current = null;
      }, resetMs);
      return ok;
    },
    [resetMs],
  );

  return { state, copy };
}

export interface CopyButtonProps
  extends Omit<ButtonProps, "onClick" | "children" | "value"> {
  /** Text to copy, or a function that builds it on click. */
  value: string | (() => string);
  /** Accessible name in the idle state, e.g. "Copy wget command". */
  label?: string;
  /** Show a visible text label next to the icon ("Copy" / "Copied"). */
  showLabel?: boolean;
  /** Visible idle label when `showLabel` is set, e.g. "Copy wget". Defaults to "Copy". */
  idleText?: string;
  /** Text announced to screen readers after a successful copy. */
  copiedMessage?: string;
  /** Text announced after a failed copy. */
  failedMessage?: string;
  onCopied?: (ok: boolean) => void;
}

/**
 * Icon (or icon + label) button that copies text. At least a 32px target.
 * The transient state is folded into the accessible name so it never
 * contradicts the visible label (WCAG 2.5.3), and announced via a live region.
 */
export const CopyButton = React.forwardRef<HTMLButtonElement, CopyButtonProps>(
  (
    {
      value,
      label = "Copy to clipboard",
      showLabel = false,
      idleText = "Copy",
      copiedMessage = "Copied to clipboard",
      failedMessage = "Copying failed",
      onCopied,
      className,
      variant = "ghost",
      size,
      disabled,
      ...props
    },
    ref,
  ) => {
    const { state, copy } = useCopyToClipboard();
    const visibleLabel =
      state === "copied" ? "Copied" : state === "failed" ? "Copy failed" : idleText;

    return (
      <>
        <Button
          ref={ref}
          type="button"
          variant={variant}
          size={size ?? (showLabel ? "sm" : "icon")}
          disabled={disabled}
          className={cn(
            showLabel ? "h-8 min-w-8 px-2.5" : "h-8 w-8",
            "text-muted-foreground hover:text-foreground",
            className,
          )}
          aria-label={state === "idle" ? label : visibleLabel}
          title={state === "idle" ? label : visibleLabel}
          onClick={async () => {
            const text = typeof value === "function" ? value() : value;
            const ok = await copy(text);
            onCopied?.(ok);
          }}
          {...props}
        >
          {state === "copied" ? (
            <Check className="text-success" aria-hidden="true" />
          ) : (
            <Copy aria-hidden="true" />
          )}
          {showLabel ? visibleLabel : null}
        </Button>
        <span className="sr-only" role="status" aria-live="polite">
          {state === "copied" ? copiedMessage : state === "failed" ? failedMessage : ""}
        </span>
      </>
    );
  },
);
CopyButton.displayName = "CopyButton";
