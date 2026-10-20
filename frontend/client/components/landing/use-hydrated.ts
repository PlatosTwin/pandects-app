import { useSyncExternalStore } from "react";

const subscribe = () => () => undefined;

/**
 * False during prerender and the hydration render, true afterwards. Lets a
 * component render the same placeholder as the server HTML on the first
 * client paint, then switch to live data without a hydration mismatch.
 */
export function useHydrated(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => true,
    () => false,
  );
}
