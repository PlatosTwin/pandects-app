import { useEffect, useRef, useState, type ReactNode } from "react";
import { BadgeCheck } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { Highlight } from "./Highlight";

/** "Target ← Acquirer", with the relationship spelled out for screen readers. */
export function Parties({ target, acquirer }: { target?: string | null; acquirer?: string | null }) {
  const t = target?.trim() || "Unknown target";
  const a = acquirer?.trim() || "Unknown acquirer";
  return (
    <span className="min-w-0 max-w-full truncate" title={`${t}, acquired by ${a}`}>
      <span className="font-medium text-foreground">{t}</span>
      <span aria-hidden="true"> ← </span>
      <span className="sr-only">, acquired by </span>
      <span>{a}</span>
    </span>
  );
}

/** Joins meta-line items with a middle dot, skipping empty ones. */
export function MetaLine({ items }: { items: ReactNode[] }) {
  const present = items.filter((item) => item !== null && item !== undefined && item !== false && item !== "");
  return (
    <>
      {present.map((item, index) => (
        <span key={index} className="inline-flex min-w-0 max-w-full items-center gap-1.5">
          {index > 0 ? (
            <span aria-hidden="true" className="text-muted-foreground/60">
              ·
            </span>
          ) : null}
          {item}
        </span>
      ))}
    </>
  );
}

export function VerifiedBadge() {
  return (
    <Badge variant="success" className="gap-1" title="This agreement has been verified by hand.">
      <BadgeCheck className="h-3 w-3" aria-hidden="true" />
      Verified
    </Badge>
  );
}

/**
 * Where to start a collapsed preview so the first keyword match is visible:
 * the paragraph holding the match, or a nearby sentence start when the match
 * sits deep in a long paragraph. 0 when the match is already near the top.
 */
export function previewStart(text: string, pattern: RegExp | null | undefined, lead = 320): number {
  if (!pattern) return 0;
  const re = new RegExp(pattern.source, pattern.flags.replace("g", ""));
  const index = text.search(re);
  if (index < lead) return 0;
  const paragraph = text.lastIndexOf("\n", index);
  const paragraphStart = paragraph < 0 ? 0 : paragraph + 1;
  if (index - paragraphStart < lead) return paragraphStart;
  const window = text.slice(index - 200, index);
  const sentence = window.search(/[.;:]\s+[^.;:]*$/);
  return sentence >= 0 ? index - 200 + sentence + 2 : index - 120;
}

/**
 * Plain text clamped to `lines`, with a "Show full …" toggle when it
 * overflows. No inner scroll box, so the page wheel never gets trapped.
 */
export function ClampedText({
  text,
  pattern,
  lines = 6,
  moreLabel = "Show full section",
  className,
}: {
  text: string;
  pattern?: RegExp | null;
  lines?: 3 | 4 | 6;
  moreLabel?: string;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [overflowing, setOverflowing] = useState(false);

  useEffect(() => {
    const node = ref.current;
    if (!node || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => {
      setOverflowing(node.scrollHeight > node.clientHeight + 1);
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, [text, lines]);

  const clampClass = lines === 3 ? "line-clamp-3" : lines === 4 ? "line-clamp-4" : "line-clamp-6";
  // Collapsed, start at the first keyword match (with a leading ellipsis).
  const start = expanded ? 0 : previewStart(text, pattern);
  const shown = start > 0 ? `… ${text.slice(start).trimStart()}` : text;
  return (
    <div className={className}>
      <div
        ref={ref}
        className={cn(
          "whitespace-pre-line text-sm leading-relaxed text-foreground/90",
          !expanded && clampClass,
        )}
      >
        <Highlight text={shown} pattern={pattern ?? null} />
      </div>
      {overflowing || expanded || start > 0 ? (
        <button
          type="button"
          onClick={() => setExpanded((prev) => !prev)}
          aria-expanded={expanded}
          className="mt-1 text-sm font-medium text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {expanded ? "Show less" : moreLabel}
        </button>
      ) : null}
    </div>
  );
}
