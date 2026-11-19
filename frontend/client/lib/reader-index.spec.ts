import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { parseXMLContent } from "@/lib/xml-parser";
import {
  buildReaderOverlay,
  collectTextAnchors,
  excerpt,
  tocEntryOutlineId,
  type ReaderIndex,
  type ReaderOutlineNode,
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

describe("declaration and subsection anchors", () => {
  const anchors = collectTextAnchors(parseXMLContent(xml));

  it("accepts every declaration and subsection of the enricher's payload", () => {
    const overlay = buildReaderOverlay(indexFrom(expected.payload), anchors);
    const decls = expected.payload.terms.flatMap((term) => term.decls).filter((d) => d.anchor?.t != null);
    expect(overlay.alignedDecls.size).toBe(decls.length);
    const subs = [...overlay.outlineById.values()].filter((node) => node.anchor.t != null);
    expect(subs.length).toBeGreaterThan(0);
    expect(subs.every((node) => overlay.alignedOutline.has(node.id))).toBe(true);
  });

  it("rejects a declaration or subsection whose ordinal moved", () => {
    const moved = structuredClone(expected.payload);
    for (const term of moved.terms) {
      for (const decl of term.decls) if (decl.anchor?.t != null) decl.anchor.t += 1;
    }
    const shiftOutline = (nodes: typeof moved.outline) => {
      for (const node of nodes) {
        if (node.anchor.t != null) node.anchor.t += 1;
        shiftOutline(node.children ?? []);
      }
    };
    shiftOutline(moved.outline);
    const overlay = buildReaderOverlay(indexFrom(moved), anchors);
    expect(overlay.alignedDecls.size).toBe(0);
    expect(overlay.alignedOutline.size).toBe(0);
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

describe("printed table of contents entries", () => {
  const node = (
    kind: "article" | "section",
    id: string,
    label: string,
    title: string,
    children?: ReaderOutlineNode[],
  ): ReaderOutlineNode => ({ id, kind, label, title, anchor: { s: `uuid-${id}` }, children });
  const overlayOf = (outline: ReaderOutlineNode[]) =>
    buildReaderOverlay(
      indexFrom({ outline, terms: [], uses: {}, xrefs: [] }),
      collectTextAnchors(parseXMLContent("<document><body></body></document>")),
    );
  const overlay = overlayOf([
    node("article", "a1", "I", "ARTICLE I DEFINITIONS"),
    node("article", "a2", "II", "ARTICLE II SALE AND PURCHASE", [
      node("section", "s2.1", "2.1", "2.1 Purchase and Sale of the Company Capital Stock; Purchase Price."),
      node("section", "s2.2", "2.2", "2.2 Closing Payments."),
    ]),
    node("article", "a5", "V", "ARTICLE V Covenants", [
      node("section", "s5.8", "5.8", "Section 5.08. Retention of Books &amp; Records."),
    ]),
  ]);

  it("links articles and sections by their number", () => {
    expect(tocEntryOutlineId(overlay, "ARTICLE II SALE AND PURCHASE")).toBe("a2");
    expect(
      tocEntryOutlineId(overlay, "2.1 Purchase and Sale of the Company Capital Stock; Purchase Price"),
    ).toBe("s2.1");
    expect(tocEntryOutlineId(overlay, "Section 5.08 Retention of Books & Records")).toBe("s5.8");
    expect(tocEntryOutlineId(overlay, "Article 5 Covenants")).toBe("a5");
  });

  it("accepts a shortened printed heading and a bare number", () => {
    expect(tocEntryOutlineId(overlay, "2.1 Purchase and Sale")).toBe("s2.1");
    expect(tocEntryOutlineId(overlay, "2.2")).toBe("s2.2");
    expect(tocEntryOutlineId(overlay, "1. DEFINITIONS")).toBe("a1");
  });

  it("leaves entries the outline does not hold as plain text", () => {
    expect(tocEntryOutlineId(overlay, "2.5 Outstanding Receivables Amount")).toBeNull();
    expect(tocEntryOutlineId(overlay, "Exhibit A Form of Escrow Agreement")).toBeNull();
    // Same number, different words: an exhibit's own Section 2.2.
    expect(tocEntryOutlineId(overlay, "2.2 Governing Law")).toBeNull();
  });

  it("does not link a number that two sections share", () => {
    const twice = overlayOf([
      node("section", "s1.1", "1.1", "1.1 Definitions."),
      node("section", "s1.1~2", "1.1", "1.1 Interpretation."),
      node("article", "a5", "5", "ARTICLE 5"),
      node("section", "s5", "5", "5 Covenants."),
      node("section", "s5~2", "5", "5 Conditions."),
    ]);
    expect(tocEntryOutlineId(twice, "1.1 Definitions")).toBeNull();
    // Not Article 5 either: the shared section number is the better reading.
    expect(tocEntryOutlineId(twice, "5 Covenants")).toBeNull();
  });

  it("does not cut a longer number down to an article", () => {
    const bare = overlayOf([
      node("article", "a2", "II", "ARTICLE II"),
      node("article", "a10", "X", "ARTICLE X"),
    ]);
    expect(tocEntryOutlineId(bare, "2.1A Purchase Price Adjustment")).toBeNull();
    expect(tocEntryOutlineId(bare, "10.1(a) Conditions")).toBeNull();
    expect(tocEntryOutlineId(bare, "2. Sale and Purchase")).toBe("a2");
  });
});

