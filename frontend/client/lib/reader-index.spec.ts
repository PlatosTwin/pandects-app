import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { parseXMLContent } from "@/lib/xml-parser";
import {
  buildReaderOverlay,
  collectTextAnchors,
  excerpt,
  type ReaderIndex,
} from "@/lib/reader-index";

// The same fixture etl/tests/test_reader_offset_contract.py pins on the
// enricher side: expected.json is the enricher's text per anchor key and the
// payload it built (docs/plans/smart-reader.md, Section 6.3).
const FIXTURE = new URL(
  "../../../etl/tests/fixtures/reader_offset_contract/",
  import.meta.url,
);
const xml = readFileSync(new URL("agreement.xml", FIXTURE), "utf-8");
const expected = JSON.parse(readFileSync(new URL("expected.json", FIXTURE), "utf-8")) as {
  texts: Record<string, string>;
  payload: Pick<ReaderIndex, "outline" | "terms" | "uses" | "xrefs">;
};

function indexFrom(payload: typeof expected.payload): ReaderIndex {
  return {
    schema: 1,
    enricher: "test",
    agreement_uuid: "contract-doc",
    xml_version: 1,
    stats: {},
    ...payload,
  };
}

describe("reader index anchor contract", () => {
  const anchors = collectTextAnchors(parseXMLContent(xml));

  it("gives every <text> element the enricher's key and text", () => {
    expect(Object.fromEntries(anchors.textByKey)).toEqual(expected.texts);
  });

  it("keeps every term use and cross-reference of the enricher's payload", () => {
    const overlay = buildReaderOverlay(indexFrom(expected.payload), anchors);
    const uses = Object.values(expected.payload.uses).reduce((n, rows) => n + rows.length, 0);
    const keptUses = [...overlay.useCounts.values()].reduce((n, c) => n + c, 0);
    expect(keptUses).toBe(uses);
    const keptXrefs = [...overlay.rangesByKey.values()]
      .flat()
      .filter((range) => range.kind === "xref").length;
    expect(keptXrefs).toBe(expected.payload.xrefs.length);
  });

  it("slices the same words the enricher matched, after a non-BMP character", () => {
    const overlay = buildReaderOverlay(indexFrom(expected.payload), anchors);
    const text = anchors.textByKey.get("sec-1-1/1")!;
    const ranges = overlay.rangesByKey.get("sec-1-1/1") ?? [];
    const words = ranges.map((range) => text.slice(range.start, range.end));
    expect(words).toContain("Company");
    expect(words).toContain("Section 2.1(b)");
  });

  it("drops ranges that no longer line up with the text", () => {
    const shifted = structuredClone(expected.payload);
    for (const rows of Object.values(shifted.uses)) {
      for (const row of rows) {
        row[0] += 5;
        row[1] += 5;
      }
    }
    for (const xref of shifted.xrefs) xref.span = [xref.span[0] + 1, xref.span[1] + 1];
    const overlay = buildReaderOverlay(indexFrom(shifted), anchors);
    expect(overlay.useCounts.size).toBe(0);
    expect([...overlay.rangesByKey.values()].flat()).toHaveLength(0);
  });
});

describe("excerpt", () => {
  it("starts at the sentence holding the offset and cuts at a word", () => {
    const text = `First sentence. ${"word ".repeat(200)}`;
    const result = excerpt(text, 20, 50);
    expect(result.text.startsWith("word")).toBe(true);
    expect(result.truncated).toBe(true);
    expect(result.text.length).toBeLessThanOrEqual(50);
  });
});
