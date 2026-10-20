import { useCallback, useEffect, useRef, useState, useTransition } from "react";
import type { RefObject } from "react";
import { prefersReducedMotion } from "@/lib/scroll";
import { FIND_CURRENT_ATTR, FIND_HIT_ATTR } from "@/components/reader/find-highlight";

const DEBOUNCE_MS = 200;

interface UseFindInDocumentOptions {
  /** Element that contains the highlighted document. */
  containerRef: RefObject<HTMLElement | null>;
  /** Raw input value. */
  query: string;
  /** Viewport px covered by sticky chrome; hits are scrolled below it. */
  topOffset: number;
  /** Identity of the rendered document; a change re-attaches the observer. */
  contentKey?: unknown;
  /** Debounce delay; tests pass 0. */
  debounceMs?: number;
}

export interface FindInDocumentState {
  /** Query the document is currently highlighted with (debounced, trimmed). */
  appliedQuery: string;
  /** True while typing has outpaced the applied query, or the highlight render is in flight. */
  isPending: boolean;
  /** Hits in the rendered document. */
  count: number;
  /** 0-based index of the current hit, or -1. */
  currentIndex: number;
  next: () => void;
  previous: () => void;
}

function scrollHitIntoView(element: HTMLElement, topOffset: number) {
  const rect = element.getBoundingClientRect();
  const viewportHeight = window.innerHeight || document.documentElement.clientHeight;
  const visible = rect.top >= topOffset + 8 && rect.bottom <= viewportHeight - 8;
  if (visible) return;
  // Put the hit a third of the way down the area below the sticky chrome.
  const target = window.scrollY + rect.top - topOffset - (viewportHeight - topOffset) / 3;
  window.scrollTo({
    top: Math.max(0, target),
    behavior: prefersReducedMotion() ? "auto" : "smooth",
  });
}

function markCurrent(hits: HTMLElement[], index: number) {
  hits.forEach((hit, hitIndex) => {
    if (hitIndex === index) {
      if (!hit.hasAttribute(FIND_CURRENT_ATTR)) hit.setAttribute(FIND_CURRENT_ATTR, "");
    } else if (hit.hasAttribute(FIND_CURRENT_ATTR)) {
      hit.removeAttribute(FIND_CURRENT_ATTR);
    }
  });
}

/**
 * Browser-style find over the reader's rendered text. The renderer wraps hits
 * in `<mark data-find-hit>`; this hook debounces the query, counts the marks
 * whenever the rendered document changes, tracks the current hit
 * (`data-find-current`) and scrolls to it on `next` / `previous`.
 */
export function useFindInDocument({
  containerRef,
  query,
  topOffset,
  contentKey,
  debounceMs = DEBOUNCE_MS,
}: UseFindInDocumentOptions): FindInDocumentState {
  const trimmed = query.trim();
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [isRendering, startTransition] = useTransition();
  // The tally records which query it counted, so a count from the previous query
  // is never shown as the new one's result.
  const [tally, setTally] = useState({ query: "", count: 0, index: -1 });
  const indexRef = useRef(-1);
  const pickInitialRef = useRef(false);
  const topOffsetRef = useRef(topOffset);

  useEffect(() => {
    topOffsetRef.current = topOffset;
  }, [topOffset]);

  // A cleared query applies immediately; a typed one after the debounce, in a
  // transition so the input stays responsive while a long document re-renders.
  const appliedQuery = trimmed ? debouncedQuery : "";

  useEffect(() => {
    if (!trimmed) return;
    const apply = () =>
      startTransition(() => {
        pickInitialRef.current = true;
        setDebouncedQuery(trimmed);
      });
    const timer = window.setTimeout(apply, Math.max(0, debounceMs));
    return () => window.clearTimeout(timer);
  }, [trimmed, debounceMs]);

  const getHits = useCallback((): HTMLElement[] => {
    const container = containerRef.current;
    if (!container || !appliedQuery) return [];
    return Array.from(container.querySelectorAll<HTMLElement>(`mark[${FIND_HIT_ATTR}]`));
  }, [appliedQuery, containerRef]);

  const sync = useCallback(() => {
    const hits = getHits();
    let index = indexRef.current;
    if (hits.length === 0) {
      index = -1;
    } else if (pickInitialRef.current) {
      pickInitialRef.current = false;
      // Like a browser: start from the first hit below the sticky chrome.
      const firstVisible = hits.findIndex(
        (hit) => hit.getBoundingClientRect().top >= topOffsetRef.current,
      );
      index = firstVisible === -1 ? 0 : firstVisible;
      scrollHitIntoView(hits[index], topOffsetRef.current);
    } else if (index < 0 || index >= hits.length) {
      index = 0;
    }
    indexRef.current = index;
    setTally({ query: appliedQuery, count: hits.length, index });
    markCurrent(hits, index);
  }, [appliedQuery, getHits]);

  // The renderer re-renders marks on a new query, collapse toggles or the
  // body-only view; recount from the DOM whenever its nodes change.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const observer = new MutationObserver(() => sync());
    observer.observe(container, { childList: true, subtree: true, characterData: true });
    // Marks committed together with this effect predate the observer.
    const timer = window.setTimeout(sync, 0);
    return () => {
      observer.disconnect();
      window.clearTimeout(timer);
    };
  }, [containerRef, contentKey, sync]);

  const step = useCallback(
    (delta: 1 | -1) => {
      const hits = getHits();
      if (hits.length === 0) return;
      const current = indexRef.current;
      const base = current < 0 ? (delta === 1 ? -1 : 0) : current;
      const nextIndex = (base + delta + hits.length) % hits.length;
      indexRef.current = nextIndex;
      setTally((prev) => ({ ...prev, index: nextIndex }));
      markCurrent(hits, nextIndex);
      scrollHitIntoView(hits[nextIndex], topOffsetRef.current);
    },
    [getHits],
  );

  const next = useCallback(() => step(1), [step]);
  const previous = useCallback(() => step(-1), [step]);

  return {
    appliedQuery,
    isPending: trimmed !== appliedQuery || isRendering || tally.query !== appliedQuery,
    count: appliedQuery && tally.query === appliedQuery ? tally.count : 0,
    currentIndex: appliedQuery && tally.query === appliedQuery ? tally.index : -1,
    next,
    previous,
  };
}
