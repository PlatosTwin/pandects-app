import { describe, expect, it } from "vitest";

import {
  formatTaxonomyTreeText,
  taxonomyTextFileName,
} from "./taxonomy-export";
import type { TaxonomyLevel1 } from "./taxonomy-search";

const entries: TaxonomyLevel1[] = [
  {
    label: "Corporate",
    id: "L1-CORP",
    l2Count: 2,
    l3Count: 3,
    children: [
      {
        label: "Capital Structure",
        id: "L2-CAP",
        children: [
          { label: "Authorized Capital", id: "L3-AUTH" },
          { label: "Outstanding Shares", id: "L3-OUT" },
        ],
      },
      {
        label: "Subsidiaries",
        id: "L2-SUB",
        children: [{ label: "Subsidiary List", id: "L3-LIST" }],
      },
    ],
  },
  {
    label: "Tax",
    id: "L1-TAX",
    l2Count: 0,
    l3Count: 0,
    children: [],
  },
];

describe("formatTaxonomyTreeText", () => {
  it("renders the full three-level tree with ids and counts", () => {
    expect(formatTaxonomyTreeText(entries)).toBe(
      [
        "Corporate [L1-CORP] (2 groups, 3 types)",
        "├── Capital Structure [L2-CAP] (2 types)",
        "│   ├── Authorized Capital [L3-AUTH]",
        "│   └── Outstanding Shares [L3-OUT]",
        "└── Subsidiaries [L2-SUB] (1 types)",
        "    └── Subsidiary List [L3-LIST]",
        "Tax [L1-TAX] (0 groups, 0 types)",
      ].join("\n"),
    );
  });

  it("returns an empty string when there is nothing to export", () => {
    expect(formatTaxonomyTreeText([])).toBe("");
  });

  it("keeps the trunk connector open while more level 2 groups follow", () => {
    const lines = formatTaxonomyTreeText(entries).split("\n");
    // Leaves under the non-final group stay attached to the trunk...
    expect(lines[2].startsWith("│   ")).toBe(true);
    // ...and leaves under the final group do not.
    expect(lines[5].startsWith("    ")).toBe(true);
  });
});

describe("taxonomyTextFileName", () => {
  it("names the main and tax exports distinctly and stamps the date", () => {
    const day = new Date("2026-08-06T12:00:00Z");
    expect(taxonomyTextFileName("main", day)).toBe("taxonomy_2026-08-06.txt");
    expect(taxonomyTextFileName("tax", day)).toBe(
      "tax_clause_taxonomy_2026-08-06.txt",
    );
  });

  // Pinned timezones on both sides of UTC: a UTC stamp would name these
  // files after the wrong day for a third of every day.
  it.each([
    ["America/Los_Angeles", "2026-08-06T01:00:00Z", "taxonomy_2026-08-05.txt"],
    ["Pacific/Auckland", "2026-08-05T22:00:00Z", "taxonomy_2026-08-06.txt"],
  ])("stamps the local day in %s, not the UTC one", (zone, instant, expected) => {
    const originalZone = process.env.TZ;
    process.env.TZ = zone;
    try {
      expect(taxonomyTextFileName("main", new Date(instant))).toBe(expected);
    } finally {
      process.env.TZ = originalZone;
    }
  });

  it("zero-pads single-digit months and days", () => {
    expect(taxonomyTextFileName("main", new Date(2026, 0, 9))).toBe(
      "taxonomy_2026-01-09.txt",
    );
  });
});
