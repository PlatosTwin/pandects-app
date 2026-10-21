import { describe, expect, it } from "vitest";

import { formatIndustry } from "@/lib/naics";

describe("formatIndustry", () => {
  it("shows the server label with the code", () => {
    expect(formatIndustry("62", "Health Care and Social Assistance")).toBe(
      "Health Care and Social Assistance (62)",
    );
    expect(formatIndustry(" 621 ", "Ambulatory Health Care Services")).toBe(
      "Ambulatory Health Care Services (621)",
    );
  });

  it("falls back to the raw value when there is no label", () => {
    expect(formatIndustry("99", null)).toBe("99");
    expect(formatIndustry("Pharmaceuticals", undefined)).toBe("Pharmaceuticals");
  });

  it("returns null for empty values", () => {
    expect(formatIndustry("", "Anything")).toBeNull();
    expect(formatIndustry(null, null)).toBeNull();
    expect(formatIndustry(undefined, null)).toBeNull();
  });
});
