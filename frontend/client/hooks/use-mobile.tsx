import { useMediaQuery } from "@/components/reader/use-media-query";

const MOBILE_BREAKPOINT = 768;
const MOBILE_QUERY = `(max-width: ${MOBILE_BREAKPOINT - 1}px)`;

/**
 * True below the mobile breakpoint. Prerendered HTML and the hydration pass
 * use the server value (false); the client value takes over right after, so
 * phones don't hit a hydration mismatch.
 */
export function useIsMobile(): boolean {
  return useMediaQuery(MOBILE_QUERY);
}
