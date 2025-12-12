type EasingFn = (t: number) => number;

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

const easeOutCubic: EasingFn = (t) => 1 - Math.pow(1 - t, 3);
const easeOutExpo: EasingFn = (t) =>
  t === 1 ? 1 : 1 - Math.pow(2, -10 * t);

export function prefersReducedMotion() {
  return (
    typeof window !== "undefined" &&
    window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches === true
  );
}

export function getScrollTopForElementInContainer(
  container: HTMLElement,
  element: HTMLElement,
  options?: { offsetPx?: number },
) {
  const offsetPx = options?.offsetPx ?? 0;
  const containerRect = container.getBoundingClientRect();
  const elementRect = element.getBoundingClientRect();

  const rawTop = elementRect.top - containerRect.top + container.scrollTop;
  const unclampedTarget = rawTop - offsetPx;
  const maxScrollTop = container.scrollHeight - container.clientHeight;

  return clamp(unclampedTarget, 0, Math.max(0, maxScrollTop));
}

export function animateScrollTop(
  container: HTMLElement,
  targetScrollTop: number,
  options?: { durationMs?: number; easing?: EasingFn },
) {
  const startScrollTop = container.scrollTop;
  const distance = targetScrollTop - startScrollTop;

  if (distance === 0) return () => {};

  if (prefersReducedMotion()) {
    container.scrollTop = targetScrollTop;
    return () => {};
  }

  if (Math.abs(distance) < 4) {
    container.scrollTop = targetScrollTop;
    return () => {};
  }

  const durationMs = options?.durationMs ?? (() => {
    const absDistance = Math.abs(distance);
    // Log curve keeps long jumps snappy while still readable.
    const raw = 120 + Math.log2(absDistance + 1) * 18;
    return clamp(raw, 140, 420);
  })();
  const easing = options?.easing ?? easeOutExpo;

  let rafId = 0;
  let canceled = false;
  const startedAt = performance.now();

  const tick = (now: number) => {
    if (canceled) return;

    const elapsed = now - startedAt;
    const t = clamp(elapsed / durationMs, 0, 1);
    const eased = easing(t);

    container.scrollTop = startScrollTop + distance * eased;

    if (t < 1) rafId = requestAnimationFrame(tick);
  };

  rafId = requestAnimationFrame(tick);

  return () => {
    canceled = true;
    if (rafId) cancelAnimationFrame(rafId);
  };
}
