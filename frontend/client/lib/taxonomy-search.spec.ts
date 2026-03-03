import { describe, expect, it } from "vitest";
import type { ClauseTypeTree } from "@/lib/clause-types";
import {
  buildTaxonomyEntries,
  buildTaxonomySearchEntries,
  filterTaxonomySearchEntries,
} from "@/lib/taxonomy-search";

describe("taxonomy search", () => {
  const tree: ClauseTypeTree = {
    General: {
      id: "aaa111",
      children: {
        Definitions: {
          id: "bbb222",
          children: {
            "Board Recommendation": {
              id: "f758e6a8d32690bd",
            },
          },
        },
      },
    },
  };

  it("matches taxonomy ids", () => {
    const entries = buildTaxonomySearchEntries(buildTaxonomyEntries(tree));

    expect(filterTaxonomySearchEntries(entries, "f758e6a8d32690bd")).toEqual([
      expect.objectContaining({
        id: "f758e6a8d32690bd",
        l3: "Board Recommendation",
      }),
    ]);
  });

  it("still matches labels", () => {
    const entries = buildTaxonomySearchEntries(buildTaxonomyEntries(tree));

    expect(filterTaxonomySearchEntries(entries, "board")).toEqual([
      expect.objectContaining({
        id: "f758e6a8d32690bd",
        l3: "Board Recommendation",
      }),
    ]);
  });
});
