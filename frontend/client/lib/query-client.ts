import { QueryClient } from "@tanstack/react-query";

/**
 * True when the hook is currently running on a server renderer (e.g. the
 * prerender pipeline). All `useQuery` callers should gate `enabled` on this:
 * during SSR the QueryClient is fresh per `renderPage()` call and any data
 * fetched is discarded, so a real network round-trip is wasted work — and
 * historically has hung the build when the upstream API is slow.
 */
export const IS_SERVER_RENDER = typeof window === "undefined";

/**
 * Creates a configured QueryClient.
 *
 * Defaults mirror the behavior of the hand-rolled hooks we are replacing:
 * - no automatic refetch on window focus (none of the legacy hooks did this)
 * - one retry on failure (legacy hooks logged + bailed; one retry is a mild improvement)
 * - 30s default staleTime so flipping between routes doesn't immediately refetch
 *
 * Per-query overrides should set staleTime explicitly when the data is known
 * to be long-lived (filter options, taxonomy) or short-lived (search results).
 */
export function createQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        refetchOnWindowFocus: false,
        retry: 1,
        staleTime: 30_000,
      },
      mutations: {
        retry: 0,
      },
    },
  });
}
