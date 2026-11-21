// @vitest-environment jsdom
import { useRef } from "react";
import { MemoryRouter } from "react-router-dom";
import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { ReaderOverlayLayer } from "./ReaderOverlayLayer";
import {
  INFERRED_NUMBER_NOTE,
  buildReaderOverlay,
  collectTextAnchors,
  type ReaderOutlineNode,
} from "@/lib/reader-index";
import { parseXMLContent } from "@/lib/xml-parser";

afterEach(cleanup);

function overlayFor(target: ReaderOutlineNode) {
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
        { at: { s: "s9", t: 0 }, span: [0, 11], text: "Section 1.2", status: "resolved", target: target.id },
      ],
    },
    collectTextAnchors(parseXMLContent("<document><body></body></document>")),
  );
}

function Harness({ target }: { target: ReaderOutlineNode }) {
  const containerRef = useRef<HTMLDivElement>(null);
  return (
    <MemoryRouter>
      <div ref={containerRef}>
        <a href="#s1.2" data-xref-index="0">
          Section 1.2
        </a>
      </div>
      <ReaderOverlayLayer
        overlay={overlayFor(target)}
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
  id: "s1.2",
  kind: "section",
  label: "1.2",
  title: "Section i.. Closing.",
  anchor: { s: "u12" },
  ...extra,
});

function openCard() {
  act(() => {
    screen.getByText("Section 1.2").focus();
  });
}

describe("cross-reference card", () => {
  it("says when the filing does not print the target's number", () => {
    render(<Harness target={section({ inferred: true })} />);
    openCard();
    expect(screen.getByText("Go to Section 1.2")).toBeTruthy();
    expect(screen.getByText(INFERRED_NUMBER_NOTE)).toBeTruthy();
  });

  it("adds nothing for a printed number or a payload without the field", () => {
    render(<Harness target={section()} />);
    openCard();
    expect(screen.getByText("Go to Section 1.2")).toBeTruthy();
    expect(screen.queryByText(INFERRED_NUMBER_NOTE)).toBeNull();
  });
});
