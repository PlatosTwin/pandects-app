import { useCallback, useSyncExternalStore } from "react";

/**
 * True while `query` matches. Server/no-matchMedia environments report
 * `serverValue` (default false), which is also what the hydration pass of a
 * prerendered page uses, so the first client render matches the HTML.
 */
export function useMediaQuery(query: string, serverValue = false): boolean {
  const subscribe = useCallback(
    (onChange: () => void) => {
      if (typeof window === "undefined" || !window.matchMedia) return () => {};
      const list = window.matchMedia(query);
      list.addEventListener("change", onChange);
      return () => list.removeEventListener("change", onChange);
    },
    [query],
  );
  const getSnapshot = () =>
    typeof window !== "undefined" && window.matchMedia
      ? window.matchMedia(query).matches
      : serverValue;
  return useSyncExternalStore(subscribe, getSnapshot, () => serverValue);
}
