import { useCallback, useEffect, useRef, useState } from "react";
import type { RefObject } from "react";

// DOM contract with XMLRenderer: article/section containers carry
// data-section-uuid; region wrappers carry data-reader-region and an
// id of `agreement-region-<region>`.
const SPY_SELECTOR = "[data-section-uuid], [data-reader-region]";

// The spy line sits just below where click-jumps land (sticky header bottom +
// 16px padding), so a jumped-to section immediately spans it.
const LINE_OFFSET_PX = 24;
// Initial grace covers scrollToSection's expand-retry loop (up to 5 x 80ms)
// before the jump scroll even starts emitting scroll events.
const JUMP_START_GRACE_MS = 600;
const JUMP_SETTLE_MS = 180;
const JUMP_MAX_MS = 2500;

export function spyIdForElement(element: Element): string | null {
  const sectionUuid = element.getAttribute("data-section-uuid");
  if (sectionUuid) return sectionUuid;
  const region = element.getAttribute("data-reader-region");
  if (region) return `agreement-region-${region}`;
  return null;
}

export function pickActiveElement(
  elementsInOrder: readonly Element[],
  intersecting: ReadonlySet<Element>,
  current: Element | null,
): Element | null {
  // Nested containers (region > article > section) all span the spy line at
  // once, and elementsInOrder is document order, so the last intersecting
  // element is the innermost one.
  let deepest: Element | null = null;
  for (const element of elementsInOrder) {
    if (intersecting.has(element)) deepest = element;
  }
  if (!deepest) {
    // The line sits in a gap (inter-section margins, collapsed titles): keep
    // the nearest preceding section active unless it left the DOM entirely
    // (e.g. its ancestor article collapsed).
    return current && current.isConnected ? current : null;
  }
  if (
    current &&
    current.isConnected &&
    deepest !== current &&
    deepest.contains(current)
  ) {
    // Only an ancestor spans the line — the gap between its children — so
    // keep the child active instead of flickering to the ancestor.
    return current;
  }
  return deepest;
}

interface UseScrollSpyOptions {
  containerRef: RefObject<HTMLElement | null>;
  /** Bottom edge of the sticky header in viewport px. */
  topOffset: number;
  /** Identity of the rendered document; a change forces a full re-observe. */
  contentKey: unknown;
  enabled?: boolean;
}

interface UseScrollSpyResult {
  /** Section uuid or region anchor id of the section under the reading line. */
  activeId: string | null;
  /**
   * Call when a programmatic jump starts: adopts the target as active and
   * suppresses observer-driven updates until the jump scroll settles.
   */
  notifyJump: (targetId: string) => void;
}

export function useScrollSpy({
  containerRef,
  topOffset,
  contentKey,
  enabled = true,
}: UseScrollSpyOptions): UseScrollSpyResult {
  const [activeId, setActiveId] = useState<string | null>(null);
  const orderedElementsRef = useRef<Element[]>([]);
  const intersectingRef = useRef<Set<Element>>(new Set());
  const activeElementRef = useRef<Element | null>(null);
  const suppressedRef = useRef(false);
  const settleTimerRef = useRef<number | null>(null);
  const capTimerRef = useRef<number | null>(null);
  const endJumpRef = useRef<() => void>(() => {});

  const recompute = useCallback(() => {
    const next = pickActiveElement(
      orderedElementsRef.current,
      intersectingRef.current,
      activeElementRef.current,
    );
    activeElementRef.current = next;
    setActiveId(next ? spyIdForElement(next) : null);
  }, []);

  const handleJumpScroll = useCallback(() => {
    if (settleTimerRef.current !== null) {
      window.clearTimeout(settleTimerRef.current);
    }
    settleTimerRef.current = window.setTimeout(
      () => endJumpRef.current(),
      JUMP_SETTLE_MS,
    );
  }, []);

  const endJump = useCallback(() => {
    if (!suppressedRef.current) return;
    suppressedRef.current = false;
    window.removeEventListener("scroll", handleJumpScroll);
    if (settleTimerRef.current !== null) {
      window.clearTimeout(settleTimerRef.current);
      settleTimerRef.current = null;
    }
    if (capTimerRef.current !== null) {
      window.clearTimeout(capTimerRef.current);
      capTimerRef.current = null;
    }
    recompute();
  }, [handleJumpScroll, recompute]);

  useEffect(() => {
    endJumpRef.current = endJump;
  }, [endJump]);

  const notifyJump = useCallback(
    (targetId: string) => {
      // Region anchors resolve by element id; sections via the observed set.
      // The target may still be inside a collapsed article and absent from the
      // DOM — the settle recompute resolves it once the jump lands.
      const target =
        document.getElementById(targetId) ??
        orderedElementsRef.current.find(
          (element) => element.getAttribute("data-section-uuid") === targetId,
        ) ??
        null;
      activeElementRef.current = target;
      setActiveId(targetId);
      suppressedRef.current = true;
      window.addEventListener("scroll", handleJumpScroll, { passive: true });
      if (settleTimerRef.current !== null) {
        window.clearTimeout(settleTimerRef.current);
      }
      settleTimerRef.current = window.setTimeout(
        () => endJumpRef.current(),
        JUMP_START_GRACE_MS,
      );
      if (capTimerRef.current !== null) {
        window.clearTimeout(capTimerRef.current);
      }
      capTimerRef.current = window.setTimeout(
        () => endJumpRef.current(),
        JUMP_MAX_MS,
      );
    },
    [handleJumpScroll],
  );

  useEffect(() => {
    return () => {
      window.removeEventListener("scroll", handleJumpScroll);
      if (settleTimerRef.current !== null) {
        window.clearTimeout(settleTimerRef.current);
      }
      if (capTimerRef.current !== null) {
        window.clearTimeout(capTimerRef.current);
      }
    };
  }, [handleJumpScroll]);

  useEffect(() => {
    if (!enabled) {
      orderedElementsRef.current = [];
      intersectingRef.current = new Set();
      activeElementRef.current = null;
      setActiveId(null);
      return;
    }
    const container = containerRef.current;
    if (!container || typeof IntersectionObserver === "undefined") return;

    let observer: IntersectionObserver | null = null;
    let rescanFrame: number | null = null;
    let resizeTimer: number | null = null;

    const onIntersect: IntersectionObserverCallback = (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting) intersectingRef.current.add(entry.target);
        else intersectingRef.current.delete(entry.target);
      }
      if (!suppressedRef.current) recompute();
    };

    const buildObserver = () => {
      observer?.disconnect();
      intersectingRef.current = new Set();
      // A 1px horizontal band just below the sticky header: an element counts
      // as intersecting while it spans the reading line.
      const line = topOffset + LINE_OFFSET_PX;
      const bottomMargin = Math.max(0, window.innerHeight - line - 1);
      observer = new IntersectionObserver(onIntersect, {
        rootMargin: `-${line}px 0px -${bottomMargin}px 0px`,
      });
      orderedElementsRef.current = Array.from(
        container.querySelectorAll(SPY_SELECTOR),
      );
      for (const element of orderedElementsRef.current) {
        observer.observe(element);
      }
    };

    buildObserver();

    // Collapse toggles and body-only mode add/remove observed elements without
    // changing contentKey — rescan whenever the rendered tree mutates. Skip
    // rebuilds when the observed set is unchanged (e.g. highlight overlays).
    const mutationObserver = new MutationObserver(() => {
      if (rescanFrame !== null) return;
      rescanFrame = requestAnimationFrame(() => {
        rescanFrame = null;
        const next = container.querySelectorAll(SPY_SELECTOR);
        const previous = orderedElementsRef.current;
        const unchanged =
          next.length === previous.length &&
          previous.every((element, index) => element === next[index]);
        if (!unchanged) buildObserver();
      });
    });
    mutationObserver.observe(container, { childList: true, subtree: true });

    // rootMargin bakes in the viewport height, so viewport resizes need a
    // rebuilt observer.
    const onResize = () => {
      if (resizeTimer !== null) window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(buildObserver, 150);
    };
    window.addEventListener("resize", onResize);

    return () => {
      observer?.disconnect();
      mutationObserver.disconnect();
      window.removeEventListener("resize", onResize);
      if (rescanFrame !== null) cancelAnimationFrame(rescanFrame);
      if (resizeTimer !== null) window.clearTimeout(resizeTimer);
    };
  }, [containerRef, contentKey, enabled, recompute, topOffset]);

  return { activeId, notifyJump };
}
