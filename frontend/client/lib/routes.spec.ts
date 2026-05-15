import { describe, expect, it } from "vitest";
import {
  ROUTE_DEFINITION_BY_PATH,
  // eslint-disable-next-line @typescript-eslint/ban-ts-comment
  // @ts-ignore - .mjs has no .d.ts; this is a runtime import for tests
} from "../../shared/route-manifest.mjs";
import { STATIC_ROUTE_PATHS } from "./routes";

/**
 * Cross-check: every static runtime route in `client/lib/routes.tsx` must have
 * a matching entry in `shared/route-manifest.mjs`. If they drift, the SEO
 * pipeline falls back to a "Not Found | Pandects" title on a working page
 * (see CLAUDE.md). This test makes that drift fail CI instead of shipping.
 */
describe("routes ↔ manifest cross-check", () => {
  it("every static runtime route has a matching SEO manifest entry", () => {
    const manifest = ROUTE_DEFINITION_BY_PATH as Map<string, unknown>;
    const missing = STATIC_ROUTE_PATHS.filter((path) => !manifest.has(path));
    expect(missing).toEqual([]);
  });
});
