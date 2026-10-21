import { describe, expect, it } from "vitest";
import { groupChips, useActiveChips } from "./SearchActiveFilters";

describe("active filter chips", () => {
  it("collapses many values of one filter into a single run", () => {
    const chips = useActiveChips(
      {
        clauseType: ["a", "b", "c"],
        transaction_consideration: ["cash", "stock"],
        year: ["2024"],
      },
      "sections",
      { a: "Defined Terms", b: "Governing Law", c: "Waiver of Jury Trial" },
    );
    const runs = groupChips(chips);
    expect(runs.map((run) => [run.key, run.chips.length])).toEqual([
      ["clauseType", 3],
      ["year", 1],
      ["transaction_consideration", 2],
    ]);
    expect(runs[0].chips[0].group?.removeGroup).toEqual({ clauseType: [] });
    expect(runs[2].chips[0].group?.removeGroup).toEqual({ transaction_consideration: [] });
  });
});
