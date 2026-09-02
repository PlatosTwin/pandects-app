import { describe, expect, it } from "vitest";
import { formatDateValue } from "@/lib/format-utils";

describe("formatDateValue", () => {
  it("formats date-only strings without timezone drift", () => {
    expect(formatDateValue("2023-04-01")).toBe("Apr 01, 2023");
  });

  // Pinned, because which calendar day an instant falls on is the viewer's
  // local business: an unpinned expectation here fails above UTC+12.
  it("formats datetime strings", () => {
    expect(formatDateValue("2023-04-01T12:34:56Z", "UTC")).toBe("Apr 01, 2023");
  });

  it("renders a datetime in the requested zone, not the ambient one", () => {
    // 22:00Z on the 5th is already the 6th in Auckland and still the 5th in LA.
    const instant = "2026-08-05T22:00:00Z";
    expect(formatDateValue(instant, "Pacific/Auckland")).toBe("Aug 06, 2026");
    expect(formatDateValue(instant, "America/Los_Angeles")).toBe("Aug 05, 2026");
  });

  it("returns an em dash for invalid values", () => {
    expect(formatDateValue("not-a-date")).toBe("—");
  });
});
