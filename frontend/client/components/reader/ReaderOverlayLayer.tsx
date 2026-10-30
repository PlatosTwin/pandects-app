import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { RefObject } from "react";
import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { Popover, PopoverAnchor, PopoverContent } from "@/components/ui/popover";
import { Sheet, SheetContent, SheetDescription, SheetTitle } from "@/components/ui/sheet";
import { buildAccountPathWithNext } from "@/lib/auth-next";
import { decodeXmlEntities } from "@/lib/text-utils";
import {
  anchorKey,
  declLocation,
  excerpt,
  type ReaderOutlineNode,
  type ReaderOverlay,
  type ReaderXref,
} from "@/lib/reader-index";

/** Hover intent before a preview opens, and grace before it closes. */
const OPEN_DELAY_MS = 300;
const CLOSE_DELAY_MS = 200;
const MAX_DECLS = 3;
const REDACTED_TEXT = "[REDACTED]";

// `pinned`: opened by a click or tap, so it stays until dismissed rather
// than closing when the pointer leaves.
type Active = (
  | { kind: "term"; termId: number; element: HTMLElement }
  | { kind: "xref"; index: number; element: HTMLElement }
) & { pinned?: boolean };

interface ReaderOverlayLayerProps {
  overlay: ReaderOverlay;
  containerRef: RefObject<HTMLElement | null>;
  isMobile: boolean;
  /** Jump to an outline node (a cross-reference), recording where it came from. */
  onFollowXref: (xref: ReaderXref, source: HTMLElement) => void;
  /** Jump to a `<text>` element by anchor key ("Go to definition"). */
  onGoToAnchor: (key: string, revealUuid: string) => void;
}

export function outlineNodeName(node: ReaderOutlineNode): string {
  if (node.kind === "article") return `Article ${node.label}`;
  if (node.kind === "attachment") return node.label;
  return `Section ${node.label}`;
}

/**
 * Previews for defined terms and cross-references. Text spans carry
 * `data-term-id` / `data-xref-index` and this layer listens on the reader
 * container, so a long agreement mounts one popover, not thousands.
 */
export function ReaderOverlayLayer({
  overlay,
  containerRef,
  isMobile,
  onFollowXref,
  onGoToAnchor,
}: ReaderOverlayLayerProps) {
  const [active, setActive] = useState<Active | null>(null);
  const openTimer = useRef<number | null>(null);
  const closeTimer = useRef<number | null>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const activeRef = useRef(active);
  useEffect(() => {
    activeRef.current = active;
  }, [active]);

  const clearTimers = useCallback(() => {
    if (openTimer.current !== null) window.clearTimeout(openTimer.current);
    if (closeTimer.current !== null) window.clearTimeout(closeTimer.current);
    openTimer.current = null;
    closeTimer.current = null;
  }, []);

  const close = useCallback(() => {
    clearTimers();
    setActive(null);
  }, [clearTimers]);

  const scheduleClose = useCallback(() => {
    if (openTimer.current !== null) window.clearTimeout(openTimer.current);
    openTimer.current = null;
    if (closeTimer.current !== null) window.clearTimeout(closeTimer.current);
    closeTimer.current = window.setTimeout(
      () => setActive((current) => (current?.pinned ? current : null)),
      CLOSE_DELAY_MS,
    );
  }, []);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const targetOf = (event: Event): Active | null => {
      const element = (event.target as Element | null)?.closest<HTMLElement>(
        "[data-term-id],[data-xref-index]",
      );
      if (!element || !container.contains(element)) return null;
      if (element.dataset.termId !== undefined) {
        return { kind: "term", termId: Number(element.dataset.termId), element };
      }
      return { kind: "xref", index: Number(element.dataset.xrefIndex), element };
    };

    // Hover previews are for fine pointers; on phones a card opens on tap,
    // as a sheet whose backdrop would otherwise read as the pointer leaving.
    const onPointerOver = (event: PointerEvent) => {
      if (isMobile || event.pointerType === "touch") return;
      const target = targetOf(event);
      if (!target) return;
      if (closeTimer.current !== null) window.clearTimeout(closeTimer.current);
      closeTimer.current = null;
      // A card opened by a click stays put until dismissed.
      if (activeRef.current?.pinned || activeRef.current?.element === target.element) return;
      if (openTimer.current !== null) window.clearTimeout(openTimer.current);
      openTimer.current = window.setTimeout(() => setActive(target), OPEN_DELAY_MS);
    };

    const onPointerOut = (event: PointerEvent) => {
      if (isMobile || event.pointerType === "touch") return;
      if (!targetOf(event)) return;
      const next = event.relatedTarget as Node | null;
      if (next && cardRef.current?.contains(next)) return;
      scheduleClose();
    };

    const onClick = (event: MouseEvent) => {
      // Cmd/Ctrl/Shift-click keeps the browser's own link handling (new tab).
      if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
        return;
      }
      const target = targetOf(event);
      if (!target) return;
      event.preventDefault();
      if (target.kind === "xref") {
        const xref = overlay.index.xrefs[target.index];
        close();
        if (xref) onFollowXref(xref, target.element);
        return;
      }
      clearTimers();
      setActive((current) =>
        current?.element === target.element && current.pinned ? null : { ...target, pinned: true },
      );
    };

    // Keyboard users reach cross-references as links; previews follow focus.
    const onFocusIn = (event: FocusEvent) => {
      const target = targetOf(event);
      if (target?.kind === "xref" && !isMobile) {
        clearTimers();
        setActive(target);
      }
    };

    container.addEventListener("pointerover", onPointerOver);
    container.addEventListener("pointerout", onPointerOut);
    container.addEventListener("click", onClick);
    container.addEventListener("focusin", onFocusIn);
    return () => {
      container.removeEventListener("pointerover", onPointerOver);
      container.removeEventListener("pointerout", onPointerOut);
      container.removeEventListener("click", onClick);
      container.removeEventListener("focusin", onFocusIn);
      clearTimers();
    };
  }, [clearTimers, close, containerRef, isMobile, onFollowXref, overlay, scheduleClose]);

  // Mark the term whose card is open; the overlay re-renders rarely, so a
  // data attribute on the element is enough.
  const activeTerm = active?.kind === "term" ? active.element : null;
  useEffect(() => {
    if (!activeTerm) return;
    activeTerm.setAttribute("data-active", "true");
    return () => activeTerm.removeAttribute("data-active");
  }, [activeTerm]);

  const virtualRef = useMemo(() => ({ current: active?.element ?? null }), [active]);
  // Text that re-renders (find opened, settings changed) detaches the element
  // a card points at; such a card just stays closed.
  const open = active !== null && active.element.isConnected;

  const body = active ? (
    active.kind === "term" ? (
      <TermCard
        overlay={overlay}
        termId={active.termId}
        inSheet={isMobile}
        onGoToAnchor={(key, uuid) => {
          close();
          onGoToAnchor(key, uuid);
        }}
      />
    ) : (
      <XrefCard
        overlay={overlay}
        xref={overlay.index.xrefs[active.index]}
        inSheet={isMobile}
        onFollow={() => {
          const element = active.element;
          const xref = overlay.index.xrefs[active.index];
          close();
          if (xref) onFollowXref(xref, element);
        }}
      />
    )
  ) : null;

  if (isMobile) {
    return (
      <Sheet open={open} onOpenChange={(open) => (open ? null : close())}>
        <SheetContent side="bottom" className="max-h-[70vh] overflow-y-auto">
          {body}
        </SheetContent>
      </Sheet>
    );
  }

  return (
    <Popover open={open} onOpenChange={(open) => (open ? null : close())} modal={false}>
      <PopoverAnchor virtualRef={virtualRef as RefObject<HTMLElement>} />
      <PopoverContent
        ref={cardRef}
        side="bottom"
        align="start"
        collisionPadding={12}
        className="w-[min(26rem,calc(100vw-2rem))] p-4 text-sm"
        // A hover preview must not pull focus out of the text.
        onOpenAutoFocus={(event) => event.preventDefault()}
        // A click on the open card's own term toggles it closed in onClick;
        // treating it as an outside click would close and then reopen it.
        onPointerDownOutside={(event) => {
          if (active?.element.contains(event.target as Node)) event.preventDefault();
        }}
        onCloseAutoFocus={(event) => event.preventDefault()}
        onPointerEnter={() => {
          if (closeTimer.current !== null) window.clearTimeout(closeTimer.current);
          closeTimer.current = null;
        }}
        onPointerLeave={scheduleClose}
      >
        {body}
      </PopoverContent>
    </Popover>
  );
}

function SignInHint() {
  return (
    <Link
      to={buildAccountPathWithNext(
        `${window.location.pathname}${window.location.search}${window.location.hash}`,
      )}
      className="font-medium text-primary underline underline-offset-2"
    >
      Sign in to read it
    </Link>
  );
}

function TermCard({
  overlay,
  termId,
  inSheet,
  onGoToAnchor,
}: {
  overlay: ReaderOverlay;
  inSheet: boolean;
  termId: number;
  onGoToAnchor: (key: string, revealUuid: string) => void;
}) {
  const term = overlay.termsById.get(termId);
  if (!term) return null;
  const uses = overlay.useCounts.get(termId) ?? 0;
  const decls = term.decls.slice(0, MAX_DECLS);
  return (
    <div className="space-y-3">
      <CardHeading inSheet={inSheet}>{term.term}</CardHeading>
      {decls.length === 0 ? (
        <p className="text-muted-foreground">No definition was found for this term.</p>
      ) : null}
      {decls.map((decl, i) => {
        const where = declLocation(decl);
        if (decl.redacted || !decl.anchor) {
          return (
            <p key={i} className="text-muted-foreground">
              {where ? `Defined in ${where}. ` : "Defined in text hidden in anonymous mode. "}
              <SignInHint />
            </p>
          );
        }
        const key = anchorKey(decl.anchor);
        const aligned = overlay.alignedDecls.has(decl);
        const text = aligned ? overlay.textByKey.get(key) : undefined;
        const preview = text && !text.includes(REDACTED_TEXT) ? excerpt(text, decl.span?.[0] ?? 0) : null;
        return (
          <div key={i} className="space-y-1.5">
            {where || decl.pointer ? (
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                {decl.pointer ? `Defined by reference to ${decl.pointer}` : `Defined in ${where}`}
              </p>
            ) : null}
            {preview ? (
              <p className="leading-relaxed text-foreground">
                {preview.text}
                {preview.truncated ? "…" : null}
              </p>
            ) : null}
            {aligned ? (
              <button
                type="button"
                className="inline-flex items-center gap-1 text-sm font-medium text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                onClick={() => onGoToAnchor(key, decl.anchor!.s)}
              >
                Go to definition
                <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
              </button>
            ) : null}
          </div>
        );
      })}
      {term.decls.length > MAX_DECLS ? (
        <p className="text-xs text-muted-foreground">
          Defined {term.decls.length} times; showing the first {MAX_DECLS}.
        </p>
      ) : null}
      <p className="border-t border-border pt-2 text-xs text-muted-foreground">
        {uses === 1 ? "Used once in this agreement" : `Used ${uses} times in this agreement`}
      </p>
    </div>
  );
}

function XrefCard({
  overlay,
  xref,
  inSheet,
  onFollow,
}: {
  overlay: ReaderOverlay;
  inSheet: boolean;
  xref: ReaderXref | undefined;
  onFollow: () => void;
}) {
  const node = xref?.target ? overlay.outlineById.get(xref.target) : undefined;
  if (!xref || !node) return null;
  const isSub = node.anchor.t != null;
  const text = isSub && !overlay.alignedOutline.has(node.id)
    ? undefined
    : overlay.textByKey.get(anchorKey(node.anchor));
  // A redacted section keeps one placeholder paragraph, so a subsection
  // inside it has no text of its own; its section's first paragraph says so.
  const hidden =
    (text ?? overlay.textByKey.get(`${node.anchor.s}/0`) ?? "").includes(REDACTED_TEXT);
  const preview = text && !hidden ? excerpt(text, 0, 320) : null;
  const heading = isSub ? outlineNodeName(node) : decodeXmlEntities(node.title) || outlineNodeName(node);
  return (
    <div className="space-y-2">
      <CardHeading inSheet={inSheet}>{heading}</CardHeading>
      {xref.status === "section_fallback" ? (
        <p className="text-xs text-muted-foreground">
          {xref.text} was not found as its own subsection; this links to the section.
        </p>
      ) : null}
      {preview ? (
        <p className="leading-relaxed text-foreground">
          {preview.text}
          {preview.truncated ? "…" : null}
        </p>
      ) : hidden ? (
        <p className="text-muted-foreground">
          This text is hidden in anonymous mode. <SignInHint />
        </p>
      ) : null}
      <button
        type="button"
        className="inline-flex items-center gap-1 text-sm font-medium text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        onClick={onFollow}
      >
        Go to {outlineNodeName(node)}
        <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
      </button>
    </div>
  );
}

// The card's title is the Sheet's accessible title on phones and a plain
// heading in the desktop popover.
function CardHeading({ inSheet, children }: { inSheet: boolean; children: string }) {
  if (inSheet) {
    return (
      <>
        <SheetTitle className="pr-8 text-base">{children}</SheetTitle>
        <SheetDescription className="sr-only">Preview</SheetDescription>
      </>
    );
  }
  return <p className="font-semibold text-foreground">{children}</p>;
}
