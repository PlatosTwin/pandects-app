// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it } from "vitest";

import { TableOfContents } from "@/components/TableOfContents";
import type { ReaderOutlineNode } from "@/lib/reader-index";

afterEach(cleanup);
beforeAll(() => {
  // jsdom has no layout; the panel scrolls the current entry into view.
  Element.prototype.scrollIntoView ??= () => {};
});

const XML =
  '<document><body><article title="ARTICLE II EFFECT OF THE MERGER" uuid="a2">' +
  '<section title="Section 1.07 Effect of the Merger." uuid="s21"><text>[REDACTED]</text></section>' +
  "</article></body></document>";

const sub = (label: string, extra: Partial<ReaderOutlineNode> = {}): ReaderOutlineNode => ({
  id: `s${label}`,
  kind: "sub",
  label,
  title: "",
  anchor: { s: "s21", t: 1 },
  ...extra,
});

function renderWith(subs: ReaderOutlineNode[]) {
  render(
    <TableOfContents
      xmlContent={XML}
      activeItemId="s21"
      onSectionClick={() => {}}
      subsectionsFor={(uuid) => (uuid === "s21" ? subs : [])}
    />,
  );
}

describe("subsections in the contents panel", () => {
  it("brackets a number the filing does not print and names it for screen readers", () => {
    renderWith([sub("2.1(a)", { inferred: true })]);
    const button = screen.getByRole("button", {
      name: "2.1(a), number not printed in the filing",
    });
    expect(button.textContent).toBe("[2.1(a)]");
    expect(button.getAttribute("title")).toBe(
      "The filed text does not print the number 2.1(a). It comes from the agreement's own references and section order.",
    );
  });

  it("leaves a printed number and printed words as they are", () => {
    renderWith([sub("2.1(a)"), sub("2.1(b)", { inferred: true, title: "(b) Each share" })]);
    const plain = screen.getByRole("button", { name: "2.1(a)" });
    expect(plain.getAttribute("aria-label")).toBeNull();
    const worded = screen.getByRole("button", { name: "(b) Each share" });
    expect(worded.getAttribute("title")).toBe("(b) Each share");
  });
});
