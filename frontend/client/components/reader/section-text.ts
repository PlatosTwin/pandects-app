import { decodeXmlEntities } from "@/lib/text-utils";

const NON_BREAKING_CHARS_RE = new RegExp("[\\u00a0\\u2007\\u202f\\u2060\\ufeff]", "g");

function escapeRegex(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Plain text of one `<section>` for "copy text": the title, a blank line,
 * then each `<text>` block as its own paragraph. Returns null when the
 * section isn't in the document.
 */
export function extractSectionPlainText(xml: string, sectionUuid: string): string | null {
  const pattern = new RegExp(
    `<section\\b([^>]*\\buuid="${escapeRegex(sectionUuid)}"[^>]*)>([\\s\\S]*?)</section>`,
  );
  const match = xml.match(pattern);
  if (!match) return null;
  const title = match[1].match(/\btitle="([^"]*)"/)?.[1];
  const body = match[2]
    // Page markers carry page ids/numbers, not agreement text.
    .replace(/<(pageUUID|page)\b[^>]*>[\s\S]*?<\/\1>/g, " ")
    .replace(/<\/text>/g, "\n\n")
    .replace(/<[^>]+>/g, " ");
  const paragraphs = decodeXmlEntities(body)
    .replace(NON_BREAKING_CHARS_RE, " ")
    .split(/\n{2,}/)
    .map((paragraph) => paragraph.replace(/[^\S\n]+/g, " ").replace(/\s*\n\s*/g, "\n").trim())
    .filter(Boolean);
  const heading = title ? decodeXmlEntities(title).trim() : "";
  return [heading, ...paragraphs].filter(Boolean).join("\n\n");
}

/**
 * Short label for a section: "§7.08" when the title starts with a section
 * number ("Section 7.08 Indemnification", "7.08 Indemnification"), otherwise
 * the title itself.
 */
export function sectionShortLabel(title: string | null | undefined): string {
  const trimmed = (title ?? "").trim();
  if (!trimmed) return "Section";
  const numbered = trimmed.match(
    /^(?:(?:section|§)\s*(\d+(?:\.\d+)*[a-z]?)|(\d+\.\d+(?:\.\d+)*[a-z]?))(?![\w.]*\d)/i,
  );
  const number = numbered?.[1] ?? numbered?.[2];
  if (number) return `§${number}`;
  return trimmed;
}
