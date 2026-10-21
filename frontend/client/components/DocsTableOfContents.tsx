import { useEffect, useState } from "react";
import { ChevronDown } from "lucide-react";

import { eyebrowClassName } from "@/components/ui/eyebrow";
import { cn } from "@/lib/utils";

export interface DocsTocItem {
  /** id of the target heading/section on the page. */
  id: string;
  label: string;
  /** 0 = section, 1 = subsection (indented). */
  level?: 0 | 1;
}

/**
 * Tracks which section is in view. The band between 20% and 30% from the top
 * of the viewport is the "reading line"; of the observed elements crossing it,
 * the one listed last is active (so nested subsections beat their parent).
 */
export function useScrollSpy(ids: readonly string[]): string | null {
  const [active, setActive] = useState<string | null>(ids[0] ?? null);
  const key = ids.join("|");

  useEffect(() => {
    if (typeof IntersectionObserver === "undefined") return;
    const elements = ids
      .map((id) => document.getElementById(id))
      .filter((el): el is HTMLElement => el !== null);
    if (elements.length === 0) return;

    const visible = new Set<string>();
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) visible.add(entry.target.id);
          else visible.delete(entry.target.id);
        }
        // Last match in list order wins, so a subsection beats the parent
        // section that contains it.
        const current = [...ids].reverse().find((id) => visible.has(id));
        if (current) setActive(current);
      },
      { rootMargin: "-20% 0px -70% 0px" },
    );
    elements.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
    // `key` captures the id list; `ids` identity may change every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return active;
}

export interface DocsTableOfContentsProps {
  items: readonly DocsTocItem[];
  /** Heading above the list. Defaults to "On this page". */
  title?: string;
  className?: string;
}

/**
 * Docs-page table of contents with scroll-spy. On lg+ it renders a plain list
 * (PageShell `layout="docs"` makes it sticky); below lg it collapses into an
 * "On this page" disclosure. Not the agreement reader's TOC.
 */
export function DocsTableOfContents({
  items,
  title = "On this page",
  className,
}: DocsTableOfContentsProps) {
  const active = useScrollSpy(items.map((item) => item.id));
  const [open, setOpen] = useState(false);

  const list = (onNavigate?: () => void) => (
    <ul className="space-y-0.5 border-l border-border">
      {items.map((item) => {
        const isActive = item.id === active;
        return (
          <li key={item.id}>
            <a
              href={`#${item.id}`}
              aria-current={isActive ? "location" : undefined}
              onClick={onNavigate}
              className={cn(
                "-ml-px block border-l-2 py-1.5 pr-2 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                item.level === 1 ? "pl-6" : "pl-3",
                isActive
                  ? "border-primary font-medium text-foreground"
                  : "border-transparent text-muted-foreground hover:border-border hover:text-foreground",
              )}
            >
              {item.label}
            </a>
          </li>
        );
      })}
    </ul>
  );

  return (
    <nav aria-label={title} className={className}>
      <div className="hidden lg:block">
        <p className={cn(eyebrowClassName, "mb-3")}>{title}</p>
        {list()}
      </div>
      <div className="rounded-lg border border-border bg-card shadow-raised lg:hidden">
        <button
          type="button"
          aria-expanded={open}
          onClick={() => setOpen((value) => !value)}
          className="flex h-11 w-full items-center justify-between px-4 text-sm font-medium text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {title}
          <ChevronDown
            className={cn("h-4 w-4 text-muted-foreground transition-transform", open && "rotate-180")}
            aria-hidden="true"
          />
        </button>
        {open ? <div className="px-4 pb-3">{list(() => setOpen(false))}</div> : null}
      </div>
    </nav>
  );
}
