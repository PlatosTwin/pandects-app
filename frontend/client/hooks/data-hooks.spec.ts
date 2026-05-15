import { readdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const HOOKS_DIR = dirname(fileURLToPath(import.meta.url));

/**
 * Drift guard: ensure no new data-fetching hook regresses to the pre–React
 * Query pattern (hand-rolled `useState` + `useEffect` + `fetch`/`authFetch`).
 *
 * The rule, per AGENTS.md "Data Fetching":
 *   All server reads go through React Query.
 *
 * If a hook needs to fetch, it should call `useQuery` or
 * `useQueryClient().fetchQuery(...)`. If a hook in `client/hooks/` uses all of
 * `useState`, `useEffect`, and a fetch call without also importing
 * `@tanstack/react-query`, this test fails — that combination is the
 * fingerprint of a hand-rolled hook.
 *
 * Add the file to ALLOWED_EXCEPTIONS only with a comment explaining why it
 * isn't a data fetch (e.g. local-only state, DOM subscriptions, toast plumbing).
 */
const ALLOWED_EXCEPTIONS = new Set<string>([
  "use-toast.ts", // toast notification plumbing, no network
  "use-mobile.tsx", // matchMedia subscription, no network
  "use-auth.ts", // context consumer, no fetch
]);

interface HookViolation {
  file: string;
  hasState: boolean;
  hasEffect: boolean;
  hasFetch: boolean;
  usesReactQuery: boolean;
}

function inspectHook(absPath: string): HookViolation {
  const source = readFileSync(absPath, "utf-8");
  const hasState = /\buseState\s*[<(]/.test(source);
  const hasEffect = /\buseEffect\s*\(/.test(source);
  const hasFetch = /\b(?:authFetch|fetch)\s*\(/.test(source);
  const usesReactQuery = /from\s+["']@tanstack\/react-query["']/.test(source);
  return {
    file: absPath.split("/").slice(-1)[0]!,
    hasState,
    hasEffect,
    hasFetch,
    usesReactQuery,
  };
}

describe("hooks data-fetching convention", () => {
  it("no hand-rolled fetch hooks (useState + useEffect + fetch without React Query)", () => {
    const offenders: string[] = [];
    for (const entry of readdirSync(HOOKS_DIR, { withFileTypes: true })) {
      if (!entry.isFile()) continue;
      if (!entry.name.endsWith(".ts") && !entry.name.endsWith(".tsx")) continue;
      if (entry.name.endsWith(".spec.ts") || entry.name.endsWith(".spec.tsx")) {
        continue;
      }
      if (ALLOWED_EXCEPTIONS.has(entry.name)) continue;

      const v = inspectHook(join(HOOKS_DIR, entry.name));
      if (v.hasState && v.hasEffect && v.hasFetch && !v.usesReactQuery) {
        offenders.push(entry.name);
      }
    }
    expect(offenders).toEqual([]);
  });
});
