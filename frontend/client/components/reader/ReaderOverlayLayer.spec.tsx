// @vitest-environment jsdom
import { useRef } from "react";
import { MemoryRouter } from "react-router-dom";
import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { ReaderOverlayLayer } from "./ReaderOverlayLayer";
import {
  buildReaderOverlay,
  collectTextAnchors,
  type ReaderOutlineNode,
  type ReaderXref,
} from "@/lib/reader-index";
import { parseXMLContent } from "@/lib/xml-parser";

afterEach(cleanup);

const LINK_TEXT = "the cited place";

function overlayFor(target: ReaderOutlineNode, xref: Partial<ReaderXref>) {
  return buildReaderOverlay(
    {
      schema: 1,
      enricher: "test",
      agreement_uuid: "doc",
      xml_version: 1,
      stats: {},
      outline: [target],
      terms: [],
      uses: {},
      xrefs: [
        {
          at: { s: "s9", t: 0 },
          span: [0, 11],
          text: "Section 8.15",
          status: "resolved",
          target: target.id,
          ...xref,
        },
      ],
    },
    collectTextAnchors(parseXMLContent("<document><body></body></document>")),
  );
}

function Harness({ target, xref = {} }: { target: ReaderOutlineNode; xref?: Partial<ReaderXref> }) {
  const containerRef = useRef<HTMLDivElement>(null);
  return (
    <MemoryRouter>
      <div ref={containerRef}>
        <a href="#target" data-xref-index="0">
          {LINK_TEXT}
        </a>
      </div>
      <ReaderOverlayLayer
        overlay={overlayFor(target, xref)}
        containerRef={containerRef}
        isMobile={false}
        onFollowXref={() => {}}
        onFollowOutline={() => {}}
        onGoToAnchor={() => {}}
      />
    </MemoryRouter>
  );
}

const section = (extra: Partial<ReaderOutlineNode> = {}): ReaderOutlineNode => ({
  id: "s8.15",
  kind: "section",
  label: "8.15",
  title: "Section 1.7 Parent Guarantor Guaranty.",
  anchor: { s: "u815" },
  ...extra,
});

function openCard() {
  act(() => {
    screen.getByText(LINK_TEXT).focus();
  });
}

describe("cross-reference card", () => {
  it("names the number the filing does not print", () => {
    render(<Harness target={section({ inferred: true })} />);
    openCard();
    expect(screen.getByText("Go to Section 8.15")).toBeTruthy();
    expect(
      screen.getByText(
        "The filed text does not print the number 8.15. It comes from the agreement's own references and section order.",
      ),
    ).toBeTruthy();
  });

  it("adds nothing for a printed number or a payload without the field", () => {
    render(<Harness target={section()} />);
    openCard();
    expect(screen.getByText("Go to Section 8.15")).toBeTruthy();
    expect(screen.queryByText(/does not print the number/)).toBeNull();
  });

  it("names a section target of a section fallback", () => {
    render(
      <Harness
        target={section({ id: "s6.4", label: "6.4", title: "Section 6.4 Covenants." })}
        xref={{ status: "section_fallback", text: "Section 6.4(e)(ii)" }}
      />,
    );
    openCard();
    expect(
      screen.getByText(
        "Section 6.4(e)(ii) was not found as its own subsection; this links to Section 6.4.",
      ),
    ).toBeTruthy();
  });

  it("names a paragraph target of a section fallback", () => {
    render(
      <Harness
        target={section({
          id: "s6.4(e)",
          kind: "sub",
          label: "6.4(e)",
          title: "(e) Each party shall",
          anchor: { s: "u64", t: 3 },
        })}
        xref={{ status: "section_fallback", text: "Section 6.4(e)(ii)" }}
      />,
    );
    openCard();
    expect(
      screen.getByText(
        "Section 6.4(e)(ii) was not found as its own subsection; this links to Section 6.4(e).",
      ),
    ).toBeTruthy();
  });

  it("ends the fallback sentence with one period", () => {
    render(
      <Harness
        target={{ id: "x-a", kind: "attachment", label: "Exhibit A.", title: "", anchor: { s: "ua" } }}
        xref={{ status: "section_fallback", text: "Exhibit A-1" }}
      />,
    );
    openCard();
    expect(
      screen.getByText("Exhibit A-1 was not found as its own subsection; this links to Exhibit A."),
    ).toBeTruthy();
  });
});
