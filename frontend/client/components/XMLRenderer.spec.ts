// @vitest-environment jsdom
import { createElement, Profiler, act } from "react";
import type { ProfilerProps } from "react";
import { createRoot } from "react-dom/client";
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

  it("deduplicates repeated TOC heading rows", () => {
    expect(
      normalizeAgreementTableOfContentsText([
        "TABLE OF SCHEDULES AND EXHIBITS",
        "Schedule A - Definitions ix",
        "TABLE OF SCHEDULES AND EXHIBITS",
        "Exhibit B - Form of Certificate x",
      ]),
    ).toEqual([
      { kind: "heading", text: "TABLE OF SCHEDULES AND EXHIBITS" },
      { kind: "entry", text: "Schedule A - Definitions", pageNumber: "ix" },
      {
        kind: "entry",
        text: "Exhibit B - Form of Certificate",
        pageNumber: "x",
      },
    ]);
  });

  it("attaches standalone roman-numeral page lines to preceding entries", () => {
    expect(
      normalizeAgreementTableOfContentsText([
        "ARTICLE I DEFINITIONS",
        "iv",
        "Section 1.1 Defined Terms",
        "v",
      ]),
    ).toEqual([
      { kind: "entry", text: "ARTICLE I DEFINITIONS", pageNumber: "iv" },
      { kind: "entry", text: "Section 1.1 Defined Terms", pageNumber: "v" },
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
    expect(markup).toContain("role=\"list\"");
    expect(markup).toContain("role=\"listitem\"");
    expect(markup).toContain("ARTICLE I THE MERGER</span><span class=\"w-10 flex-shrink-0 text-right tabular-nums text-muted-foreground\">6</span>");
    expect(markup).toContain("1.4 Certificate of Formation and Bylaws of the Surviving Corporation</span><span class=\"w-10 flex-shrink-0 text-right tabular-nums text-muted-foreground\">6</span>");
    expect(markup).not.toContain(">1.4     Certificate of Formation and Bylaws of the Surviving Corporation");
  });

  it("renders a deduplicated heading and aligned entry rows", () => {
    const markup = renderToStaticMarkup(
      createElement(XMLRenderer, {
        mode: "agreement",
        xmlContent: `
          <document>
            <tableOfContents>
              <text>TABLE OF SCHEDULES AND EXHIBITS</text>
              <text>Schedule A - Definitions ix</text>
              <text>TABLE OF SCHEDULES AND EXHIBITS</text>
              <text>Exhibit B - Form of Certificate x</text>
            </tableOfContents>
          </document>
        `,
      }),
    );

    expect(markup).toContain(
      "text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground",
    );
    expect(markup).toContain(
      ">TABLE OF SCHEDULES AND EXHIBITS</div>",
    );
    expect(markup).toContain(
      "Schedule A - Definitions</span><span class=\"w-10 flex-shrink-0 text-right tabular-nums text-muted-foreground\">ix</span>",
    );
    expect(markup).toContain(
      "Exhibit B - Form of Certificate</span><span class=\"w-10 flex-shrink-0 text-right tabular-nums text-muted-foreground\">x</span>",
    );
    expect(markup.match(/TABLE OF SCHEDULES AND EXHIBITS/g)).toHaveLength(1);
  });
});

const AGREEMENT_FIXTURE = `<?xml version="1.0"?>
<document>
  <metadata><text>meta content</text></metadata>
  <frontMatter><text>AGREEMENT AND PLAN OF MERGER</text><pageUUID/></frontMatter>
  <body>
    <article uuid="a-1" title="ARTICLE I &amp; THE MERGER">
      <section uuid="s-1" title="Section 1.1 The Merger">
        <text>The merger text.</text>
        <pageUUID/>
        <definition>Some definition</definition>
      </section>
      <section uuid="s-2" title="Section 1.2 Closing">
        <text>Closing text.</text>
      </section>
    </article>
  </body>
  <sigPages><text>Signatures here</text></sigPages>
</document>`;

describe("XMLRenderer agreement mode anchors and structure", () => {
  const markup = renderToStaticMarkup(
    createElement(XMLRenderer, {
      mode: "agreement",
      xmlContent: AGREEMENT_FIXTURE,
    }),
  );

  it("renders region anchors with stable ids and data attributes", () => {
    expect(markup).toContain('id="agreement-region-frontMatter"');
    expect(markup).toContain('data-reader-region="frontMatter"');
    expect(markup).toContain('id="agreement-region-body"');
    expect(markup).toContain('id="agreement-region-sigPages"');
  });

  it("renders section anchors as data-section-uuid attributes", () => {
    expect(markup).toContain('data-section-uuid="a-1"');
    expect(markup).toContain('data-section-uuid="s-1"');
    expect(markup).toContain('data-section-uuid="s-2"');
    expect(markup).toContain('data-article-header="true"');
  });

  it("decodes entities in article and section titles", () => {
    expect(markup).toContain("ARTICLE I &amp; THE MERGER</h3>");
    expect(markup).toContain("Section 1.1 The Merger</h3>");
  });

  it("omits metadata content", () => {
    expect(markup).not.toContain("meta content");
  });

  it("renders collapse toggles for articles and sections", () => {
    expect(markup.match(/data-collapse-toggle="true"/g)).toHaveLength(3);
  });

  it("highlights the requested section only", () => {
    const highlighted = renderToStaticMarkup(
      createElement(XMLRenderer, {
        mode: "agreement",
        xmlContent: AGREEMENT_FIXTURE,
        highlightedSection: "s-2",
      }),
    );
    const sectionTwo = highlighted.slice(
      highlighted.indexOf('data-section-uuid="s-2"'),
    );
    expect(sectionTwo).toContain("bg-primary/10");
    const sectionOne = highlighted.slice(
      highlighted.indexOf('data-section-uuid="s-1"'),
      highlighted.indexOf('data-section-uuid="s-2"'),
    );
    expect(sectionOne).not.toContain("bg-primary/10");
  });

  it("renders only body content when showBodyOnly is set", () => {
    const bodyOnly = renderToStaticMarkup(
      createElement(XMLRenderer, {
        mode: "agreement",
        xmlContent: AGREEMENT_FIXTURE,
        showBodyOnly: true,
      }),
    );
    expect(bodyOnly).not.toContain("agreement-region");
    expect(bodyOnly).not.toContain("AGREEMENT AND PLAN OF MERGER");
    expect(bodyOnly).toContain('data-section-uuid="s-1"');
  });
});

describe("XMLRenderer search mode", () => {
  it("renders tag markers and collapsible text content", () => {
    const markup = renderToStaticMarkup(
      createElement(XMLRenderer, {
        mode: "search",
        xmlContent:
          "<result><text>Some snippet with a <definition>Defined Term</definition> inside.</text></result>",
      }),
    );
    expect(markup).toContain("&lt;result&gt;");
    expect(markup).toContain("&lt;definition&gt;");
    expect(markup).toContain("Defined Term");
    expect(markup).toContain('data-collapse-toggle="true"');
  });
});

describe("XMLRenderer unparseable content fallback", () => {
  const expectFallback = (xmlContent: string) => {
    const markup = renderToStaticMarkup(
      createElement(XMLRenderer, { mode: "agreement", xmlContent }),
    );
    expect(markup).toContain("Unable to render this document");
    expect(markup).toContain('role="alert"');
  };

  it("shows the fallback for empty content", () => {
    expectFallback("");
  });

  it("shows the fallback for whitespace-only content", () => {
    expectFallback("   \n  ");
  });

  it("shows the fallback for content with nothing renderable", () => {
    expectFallback('<?xml version="1.0"?><!-- only a comment -->');
  });

  it("still renders recoverable malformed content as text", () => {
    const markup = renderToStaticMarkup(
      createElement(XMLRenderer, {
        mode: "search",
        xmlContent: "<text>abc",
      }),
    );
    expect(markup).not.toContain("Unable to render this document");
    expect(markup).toContain("&lt;text&gt;abc");
  });
});

describe("XMLRenderer memoized subtree rendering", () => {
  it("re-renders only affected subtrees on highlight change, and keeps collapse/auto-expand working", () => {
    (
      globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }
    ).IS_REACT_ACT_ENVIRONMENT = true;

    let body = "";
    for (let i = 0; i < 300; i++) {
      body += `<article uuid="a-${i}" title="ARTICLE ${i}"><section uuid="s-${i}" title="Section ${i}"><text>Body text for section ${i}. Lorem ipsum dolor sit amet.</text></section></article>`;
    }
    const xml = `<document><body>${body}</body></document>`;

    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);

    const durations: number[] = [];
    const onRender: ProfilerProps["onRender"] = (_id, _phase, actualDuration) => {
      durations.push(actualDuration);
    };
    const render = (highlighted: string | null) =>
      act(() => {
        root.render(
          createElement(
            Profiler,
            { id: "xml", onRender },
            createElement(XMLRenderer, {
              mode: "agreement",
              xmlContent: xml,
              highlightedSection: highlighted,
            }),
          ),
        );
      });

    render(null);
    const mountDuration = durations[0];

    durations.length = 0;
    render("s-150");
    expect(container.innerHTML).toContain("bg-primary/10");
    const highlightDuration = Math.max(...durations);

    // Memoized subtrees make a highlight change render only the path to the
    // highlighted section, not the whole 300-section document (~68x cheaper
    // when measured; assert a conservative 10x to avoid CI timing flakes).
    expect(highlightDuration).toBeLessThan(mountDuration / 10);

    // Collapse a section via its toggle and verify only it collapses.
    const target = container.querySelector('[data-section-uuid="s-10"]');
    const toggle = target?.querySelector(
      '[data-collapse-toggle="true"]',
    ) as HTMLButtonElement;
    act(() => {
      toggle.click();
    });
    expect(
      container.querySelector('[data-section-uuid="s-10"]')?.textContent,
    ).not.toContain("Body text for section 10.");
    expect(container.innerHTML).toContain("Body text for section 11.");

    // Highlighting a section inside a collapsed ancestor auto-expands it.
    render("s-10");
    expect(
      container.querySelector('[data-section-uuid="s-10"]')?.textContent,
    ).toContain("Body text for section 10.");

    act(() => {
      root.unmount();
    });
    container.remove();
  });
});
