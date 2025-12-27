import { describe, expect, it } from "vitest";
import { normalizeXmlText } from "./XMLRenderer";

describe("normalizeXmlText", () => {
  it("replaces non-breaking and word-joining spaces with regular spaces", () => {
    const input = "A\u00a0B\u2007C\u202fD\u2060E\ufeffF";
    expect(normalizeXmlText(input)).toBe("A B C D E F");
  });

  it("leaves normal whitespace intact", () => {
    const input = "Line 1\n  Line 2\tLine 3";
    expect(normalizeXmlText(input)).toBe(input);
  });
});
