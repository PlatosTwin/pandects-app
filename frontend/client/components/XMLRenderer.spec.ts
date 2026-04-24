import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import {
  XMLRenderer,
  normalizeAgreementTableOfContentsText,
  normalizeXmlText,
} from "./XMLRenderer";

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

describe("normalizeAgreementTableOfContentsText", () => {
  it("attaches split page-number lines to the preceding TOC entry", () => {
    expect(
      normalizeAgreementTableOfContentsText([
        "TABLE OF CONTENTS",
        "ARTICLE I THE MERGER",
        "6",
        "1.1 The Merger 6",
        "1.2 The Closing 6",
        "1.3 Effective Date and Time",
        "6",
        "1.4 Certificate of Formation and Bylaws of the Surviving Corporation",
        "6",
      ]),
    ).toEqual([
      { kind: "entry", text: "ARTICLE I THE MERGER", pageNumber: "6" },
      { kind: "entry", text: "1.1 The Merger", pageNumber: "6" },
      { kind: "entry", text: "1.2 The Closing", pageNumber: "6" },
      {
        kind: "entry",
        text: "1.3 Effective Date and Time",
        pageNumber: "6",
      },
      {
        kind: "entry",
        text: "1.4 Certificate of Formation and Bylaws of the Surviving Corporation",
        pageNumber: "6",
      },
    ]);
  });

  it("splits OCR-flattened TOC runs into distinct rows", () => {
    expect(
      normalizeAgreementTableOfContentsText([
        "Page ARTICLE I DEFINITIONS AND RULES OF CONSTRUCTION 2 Section 1.1 Defined Terms 2 Section 1.2 Certain References 2 ARTICLE II THE MERGER 3 Section 2.1 The Merger 3",
      ]),
    ).toEqual([
      {
        kind: "entry",
        text: "ARTICLE I DEFINITIONS AND RULES OF CONSTRUCTION",
        pageNumber: "2",
      },
      { kind: "entry", text: "Section 1.1 Defined Terms", pageNumber: "2" },
      {
        kind: "entry",
        text: "Section 1.2 Certain References",
        pageNumber: "2",
      },
      { kind: "entry", text: "ARTICLE II THE MERGER", pageNumber: "3" },
      { kind: "entry", text: "Section 2.1 The Merger", pageNumber: "3" },
    ]);
  });
});

describe("XMLRenderer table of contents rendering", () => {
  it("normalizes TOC rows from nested text tags in agreement mode", () => {
    const markup = renderToStaticMarkup(
      createElement(XMLRenderer, {
        mode: "agreement",
        xmlContent: `
          <document>
            <tableOfContents>
              <text>ARTICLE I THE MERGER                                                                                                   6</text>
              <text>1.4     Certificate of Formation and Bylaws of the Surviving Corporation                                               6</text>
              <text>1.5     Directors and Officers                                                                                         6</text>
            </tableOfContents>
          </document>
        `,
      }),
    );

    expect(markup).toContain("text-right tabular-nums");
    expect(markup).toContain("ARTICLE I THE MERGER</span><span class=\"w-10 flex-shrink-0 text-right tabular-nums text-muted-foreground\">6</span>");
    expect(markup).toContain("1.4 Certificate of Formation and Bylaws of the Surviving Corporation</span><span class=\"w-10 flex-shrink-0 text-right tabular-nums text-muted-foreground\">6</span>");
    expect(markup).not.toContain(">1.4     Certificate of Formation and Bylaws of the Surviving Corporation");
  });
});
