import { Remarkable } from "remarkable";
import { linkify } from "remarkable/linkify";

export type PreparedMarkdown = {
  markdown: string;
  subtitle?: string;
};

export function prepareLegalMarkdownForPage(source: string): PreparedMarkdown {
  const withoutBom = source.replace(/^\uFEFF/, "");
  const lines = withoutBom.split(/\r?\n/);

  // Drop the leading H1 (we render the page title via PageShell).
  const firstNonEmptyIndex = lines.findIndex((line) => line.trim().length > 0);
  if (firstNonEmptyIndex !== -1 && lines[firstNonEmptyIndex]?.startsWith("# ")) {
    lines.splice(firstNonEmptyIndex, 1);
    if (lines[firstNonEmptyIndex]?.trim() === "") lines.splice(firstNonEmptyIndex, 1);
  }

  // Promote "Effective date: ..." into the subtitle.
  const effectiveDateIndex = lines.findIndex((line) =>
    /^Effective date:\s*/i.test(line.trim()),
  );
  let subtitle: string | undefined;
  if (effectiveDateIndex !== -1) {
    const raw = lines[effectiveDateIndex] ?? "";
    subtitle = raw
      .replace(/^Effective date:\s*/i, "Effective date: ")
      .replace(/\*\*/g, "")
      .trim();

    lines.splice(effectiveDateIndex, 1);
    if (lines[effectiveDateIndex]?.trim() === "") lines.splice(effectiveDateIndex, 1);
  }

  return { markdown: lines.join("\n").trim(), subtitle };
}

function slugifyHeading(text: string): string {
  const unescaped = text.replace(/\\([\\`*_{}\[\]()#+\-.!])/g, "$1");
  const withoutLinks = unescaped.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
  const withoutFormatting = withoutLinks.replace(/[*_`~]/g, "");
  const withoutLeadingNumbering = withoutFormatting.replace(/^\s*\d+(\.\d+)*[.)]?\s*/, "");

  const slug = withoutLeadingNumbering
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");

  return slug.length ? slug : "section";
}

export function renderLegalMarkdownToHtml(markdown: string): string {
  const env: { __usedHeadingIds?: Map<string, number> } = {};

  const md = new Remarkable({ html: false, typographer: true }).use(linkify);
  md.renderer.rules.heading_open = (tokens: any[], idx: number) => {
    const level = tokens[idx]?.hLevel ?? 2;
    const headingInline = tokens[idx + 1];
    const rawText = typeof headingInline?.content === "string" ? headingInline.content : "";

    const used = (env.__usedHeadingIds ??= new Map<string, number>());
    const base = slugifyHeading(rawText);
    const nextCount = (used.get(base) ?? 0) + 1;
    used.set(base, nextCount);
    const id = nextCount === 1 ? base : `${base}-${nextCount}`;

    return `<h${level} id="${id}">`;
  };

  return md.render(markdown, env);
}

