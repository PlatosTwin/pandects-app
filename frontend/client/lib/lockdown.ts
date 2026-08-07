// TEMPORARY (September reopen): paths whose pages hydrate via the locked-down data
// API. <AccessGate> swaps these for the lockdown notice for visitors who aren't
// signed in, so they get a clean message instead of API error states. To reopen
// public access, empty this set (and remove the AccessGate wiring in main.tsx
// and prerender/entry.tsx).
export const ACCESS_GATED_PATHS: ReadonlySet<string> = new Set([
  "/search",
  "/agreements/:agreementUuid",
  "/bulk-data",
  "/agreement-index",
  "/taxonomy",
  "/leaderboards",
  "/trends-analyses",
]);
