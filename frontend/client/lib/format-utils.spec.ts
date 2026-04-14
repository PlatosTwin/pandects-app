import { describe, expect, it } from "vitest";
import { formatDateValue } from "@/lib/format-utils";

describe("formatDateValue", () => {
  it("formats date-only strings without timezone drift", () => {
    expect(formatDateValue("2023-04-01")).toBe("Apr 01, 2023");
  });

  it("formats datetime strings", () => {
    expect(formatDateValue("2023-04-01T12:34:56Z")).toBe("Apr 01, 2023");
  });

  it("returns an em dash for invalid values", () => {
    expect(formatDateValue("not-a-date")).toBe("—");
  });
});
