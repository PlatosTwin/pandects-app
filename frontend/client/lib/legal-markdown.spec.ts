import { describe, expect, it } from "vitest";
import { prepareLegalMarkdownForPage, renderLegalMarkdownToHtml } from "./legal-markdown";

describe("legal markdown rendering", () => {
  it("renders headings with stable ids and does not throw on links", () => {
    const source = [
      "# Pandects Terms of Service",
      "",
      "Effective date: **December 21, 2025**",
      "",
      "See [our site](https://pandects.org) and email us at test@example.com.",
      "",
      "Also: https://example.com",
      "",
      "## 1. Intro",
      "",
      "Content.",
    ].join("\n");

    const prepared = prepareLegalMarkdownForPage(source);
    expect(prepared.subtitle).toBe("Effective date: December 21, 2025");
    expect(prepared.markdown).not.toMatch(/^# /m);

    expect(() => renderLegalMarkdownToHtml(prepared.markdown)).not.toThrow();

    const html = renderLegalMarkdownToHtml(prepared.markdown);
    expect(html).toContain('id="intro"');
    expect(html).toContain('href="https://pandects.org"');
    expect(html).toContain('href="https://example.com"');
  });
});
