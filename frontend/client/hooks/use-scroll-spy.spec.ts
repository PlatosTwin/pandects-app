// @vitest-environment jsdom
/**
 * Tests for the agreement reader scroll-spy: active-element selection against
 * the nested region > article > section DOM contract, and click-jump
 * suppression handoff back to the observer.
 */
import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  pickActiveElement,
  spyIdForElement,
  useScrollSpy,
} from "./use-scroll-spy";

class IntersectionObserverStub {
  static instances: IntersectionObserverStub[] = [];
  observed: Element[] = [];

  constructor(
    public callback: IntersectionObserverCallback,
    public options?: IntersectionObserverInit,
  ) {
    IntersectionObserverStub.instances.push(this);
  }

  observe(element: Element) {
    this.observed.push(element);
  }

  unobserve(element: Element) {
    this.observed = this.observed.filter((observed) => observed !== element);
  }

  disconnect() {
    this.observed = [];
  }

  takeRecords(): IntersectionObserverEntry[] {
    return [];
  }

  emit(states: Array<[Element, boolean]>) {
    this.callback(
      states.map(
        ([target, isIntersecting]) =>
          ({ target, isIntersecting }) as IntersectionObserverEntry,
      ),
      this as unknown as IntersectionObserver,
    );
  }
}

function buildReaderDom() {
  const container = document.createElement("div");
  container.innerHTML = [
    '<section data-reader-region="body" id="agreement-region-body">',
    '<div data-section-uuid="article-1">',
    '<div data-section-uuid="sec-1"></div>',
    '<div data-section-uuid="sec-2"></div>',
    "</div>",
    "</section>",
  ].join("");
  document.body.appendChild(container);
  const [body, article, sec1, sec2] = Array.from(
    container.querySelectorAll("[data-section-uuid], [data-reader-region]"),
  ) as Element[];
  return { container, body, article, sec1, sec2 };
}

beforeEach(() => {
  IntersectionObserverStub.instances = [];
  vi.stubGlobal("IntersectionObserver", IntersectionObserverStub);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
  document.body.innerHTML = "";
});

describe("spyIdForElement", () => {
  it("maps sections to their uuid and regions to their anchor id", () => {
    const { body, sec1 } = buildReaderDom();
    expect(spyIdForElement(sec1)).toBe("sec-1");
    expect(spyIdForElement(body)).toBe("agreement-region-body");
    expect(spyIdForElement(document.createElement("div"))).toBeNull();
  });
});

describe("pickActiveElement", () => {
  it("picks the innermost intersecting element in document order", () => {
    const { body, article, sec1, sec2 } = buildReaderDom();
    const ordered = [body, article, sec1, sec2];
    expect(
      pickActiveElement(ordered, new Set([body, article, sec1]), null),
    ).toBe(sec1);
    expect(
      pickActiveElement(ordered, new Set([body, article, sec2]), sec1),
    ).toBe(sec2);
  });

  it("keeps the nearest preceding section when the line sits in a gap", () => {
    const { body, article, sec1, sec2 } = buildReaderDom();
    const ordered = [body, article, sec1, sec2];
    // Nothing spans the line at all (e.g. margins between collapsed titles).
    expect(pickActiveElement(ordered, new Set(), sec1)).toBe(sec1);
    // Only an ancestor spans the line (the margin between its children).
    expect(
      pickActiveElement(ordered, new Set([body, article]), sec2),
    ).toBe(sec2);
  });

  it("drops a stale current element that left the DOM", () => {
    const { body, article, sec1, sec2 } = buildReaderDom();
    const ordered = [body, article, sec1, sec2];
    sec2.remove();
    // The collapsed ancestor takes over instead of the removed child.
    expect(
      pickActiveElement(ordered, new Set([body, article]), sec2),
    ).toBe(article);
    expect(pickActiveElement(ordered, new Set(), sec2)).toBeNull();
  });
});

describe("useScrollSpy", () => {
  it("tracks the innermost section under the reading line", () => {
    const { container, body, article, sec1, sec2 } = buildReaderDom();
    const { result } = renderHook(() =>
      useScrollSpy({
        containerRef: { current: container },
        topOffset: 100,
        contentKey: "doc",
      }),
    );

    const observer = IntersectionObserverStub.instances.at(-1)!;
    expect(observer.observed).toEqual([body, article, sec1, sec2]);

    act(() =>
      observer.emit([
        [body, true],
        [article, true],
        [sec1, true],
      ]),
    );
    expect(result.current.activeId).toBe("sec-1");

    act(() =>
      observer.emit([
        [sec1, false],
        [sec2, true],
      ]),
    );
    expect(result.current.activeId).toBe("sec-2");

    // Gap between sections: only the ancestors span the line.
    act(() => observer.emit([[sec2, false]]));
    expect(result.current.activeId).toBe("sec-2");
  });

  it("suppresses observer updates during a jump, then hands back on settle", () => {
    vi.useFakeTimers();
    const { container, sec1 } = buildReaderDom();
    const { result } = renderHook(() =>
      useScrollSpy({
        containerRef: { current: container },
        topOffset: 100,
        contentKey: "doc",
      }),
    );
    const observer = IntersectionObserverStub.instances.at(-1)!;

    act(() => result.current.notifyJump("sec-2"));
    expect(result.current.activeId).toBe("sec-2");

    // Mid-scroll intersections must not steal the clicked state.
    act(() => observer.emit([[sec1, true]]));
    expect(result.current.activeId).toBe("sec-2");

    // After the jump settles, the observer's view of the line wins again.
    act(() => vi.advanceTimersByTime(700));
    expect(result.current.activeId).toBe("sec-1");
  });

  it("disconnects on unmount and clears state when disabled", () => {
    const { container, sec1 } = buildReaderDom();
    const { result, rerender, unmount } = renderHook(
      ({ enabled }: { enabled: boolean }) =>
        useScrollSpy({
          containerRef: { current: container },
          topOffset: 100,
          contentKey: "doc",
          enabled,
        }),
      { initialProps: { enabled: true } },
    );
    const observer = IntersectionObserverStub.instances.at(-1)!;
    act(() => observer.emit([[sec1, true]]));
    expect(result.current.activeId).toBe("sec-1");

    rerender({ enabled: false });
    expect(observer.observed).toEqual([]);
    expect(result.current.activeId).toBeNull();

    rerender({ enabled: true });
    const reenabled = IntersectionObserverStub.instances.at(-1)!;
    expect(reenabled.observed.length).toBe(4);
    unmount();
    expect(reenabled.observed).toEqual([]);
  });
});
