// The smart reader's view of the reader index (docs/plans/smart-reader.md,
// Sections 6.3 and 6.5): stand-off anchors into the agreement XML for the
// outline, defined terms and cross-references.
//
// Anchors name a `<text>` element as `<container>/<t>`, where the container is
// the nearest section or article with a uuid (else the region tag) and `t` its
// ordinal in that container. Spans are UTF-16 offsets into the element's text
// after entity decoding, with the non-breaking spaces the reader replaces. The
// walk below has to match the enricher's `_collect_texts`
// (etl/src/etl/domain/reader_index.py) exactly; reader-index.spec.ts pins both
// against the same fixture.

import type { components } from "@shared/api.gen";
import { decodeXmlEntities } from "@/lib/text-utils";
import type { XMLNode } from "@/lib/xml-parser";

type Schemas = components["schemas"];
export type ReaderIndex = Schemas["ReaderIndexResponse"];
export type ReaderAnchor = Schemas["ReaderAnchor"];
export type ReaderOutlineNode = Schemas["ReaderOutlineNode"];
export type ReaderTerm = Schemas["ReaderTerm"];
export type ReaderTermDecl = Schemas["ReaderTermDecl"];
export type ReaderXref = Schemas["ReaderXref"];

const NON_BREAKING_CHARS_RE = new RegExp("[\\u00a0\\u2007\\u202f\\u2060\\ufeff]", "g");
const CONTAINER_TAGS = new Set(["section", "article"]);

/**
 * The enricher's text for one `<text>` element: entities decoded first (so an
 * escaped `&#160;` is replaced too), and CR/CRLF folded to LF as XML parsers do.
 */
export function anchorText(raw: string): string {
  return decodeXmlEntities(raw.replace(/\r\n?/g, "\n")).replace(NON_BREAKING_CHARS_RE, " ");
}

export function anchorKey(anchor: ReaderAnchor): string {
  return anchor.t == null ? `${anchor.s}/0` : `${anchor.s}/${anchor.t}`;
}

export interface TextAnchors {
  /** `<text>` tag node -> its anchor key. */
  keyByNode: Map<XMLNode, string>;
  /** Anchor key -> the element's leading text (what the enricher read). */
  textByKey: Map<string, string>;
}

/**
 * Anchor keys for every `<text>` element of a parsed document. `roots` is the
 * full parse (the document element and nothing above it that matters), not the
 * body-only slice the reader may render.
 */
export function collectTextAnchors(roots: XMLNode[]): TextAnchors {
  const keyByNode = new Map<XMLNode, string>();
  const textByKey = new Map<string, string>();
  const counters = new Map<string, number>();

  const walk = (el: XMLNode, container: string) => {
    for (const child of el.children ?? []) {
      if (child.type !== "tag") continue;
      if (child.tagName === "text") {
        const t = counters.get(container) ?? 0;
        counters.set(container, t + 1);
        const key = `${container}/${t}`;
        keyByNode.set(child, key);
        // ElementTree's `.text` is the text before the first child element.
        const first = child.children?.[0];
        textByKey.set(key, first?.type === "text" ? anchorText(first.content) : "");
        continue;
      }
      const opens = CONTAINER_TAGS.has(child.tagName ?? "") && Boolean(child.uuid);
      walk(child, opens ? child.uuid! : container);
    }
  };

  const root = roots.find((node) => node.type === "tag");
  for (const region of root?.children ?? []) {
    if (region.type !== "tag" || region.tagName === "metadata") continue;
    walk(region, region.tagName ?? "");
  }
  return { keyByNode, textByKey };
}

/** A highlight inside one text element, in UTF-16 offsets. */
export type OverlayRange =
  | { kind: "term"; start: number; end: number; termId: number }
  | { kind: "xref"; start: number; end: number; xref: ReaderXref; index: number };

export interface ReaderOverlay {
  index: ReaderIndex;
  termsById: Map<number, ReaderTerm>;
  /** Number of validated uses per term id. */
  useCounts: Map<number, number>;
  outlineById: Map<string, ReaderOutlineNode>;
  /** Outline id -> ids of its ancestors, outermost first. */
  outlinePath: Map<string, string[]>;
  /** Section or article uuid -> its outline node. */
  outlineByUuid: Map<string, ReaderOutlineNode>;
  /**
   * Declarations and subsection outline nodes whose anchor points at the text
   * they claim (see buildReaderOverlay). Others get no preview and no jump to
   * the exact paragraph.
   */
  alignedDecls: Set<ReaderTermDecl>;
  alignedOutline: Set<string>;
  /** Anchor key -> non-overlapping ranges in document order. */
  rangesByKey: Map<string, OverlayRange[]>;
  textByKey: Map<string, string>;
}

// Uses include plurals, possessives and singulars of plural terms
// ("Subsidiaries" for Subsidiary, "Company’s"), so a use counts as aligned
// when it shares all but the last couple of characters with the term.
function spanMatchesTerm(slice: string, term: string): boolean {
  const a = slice.toLowerCase();
  const b = term.toLowerCase();
  if (a === b) return true;
  let common = 0;
  while (common < a.length && common < b.length && a[common] === b[common]) common++;
  return common >= 3 && common >= Math.min(a.length, b.length) - 2;
}

function indexOutline(nodes: ReaderOutlineNode[], overlay: ReaderOverlay, parents: string[]) {
  for (const node of nodes) {
    overlay.outlineById.set(node.id, node);
    overlay.outlinePath.set(node.id, parents);
    if (node.anchor.t == null && !overlay.outlineByUuid.has(node.anchor.s)) {
      overlay.outlineByUuid.set(node.anchor.s, node);
    }
    if (node.children) indexOutline(node.children, overlay, [...parents, node.id]);
  }
}

/** Statuses whose `target` is a place in this agreement the reader can jump to. */
export function isLinkedXref(xref: ReaderXref): boolean {
  return (xref.status === "resolved" || xref.status === "section_fallback") && Boolean(xref.target);
}

/**
 * Joins the index to the XML the reader holds. Every range is checked against
 * the text it points into and dropped when it doesn't line up, so a stale
 * index or a redacted document that shifted ordinals never underlines the
 * wrong words.
 */
export function buildReaderOverlay(index: ReaderIndex, anchors: TextAnchors): ReaderOverlay {
  const overlay: ReaderOverlay = {
    index,
    termsById: new Map(index.terms.map((term) => [term.id, term])),
    useCounts: new Map(),
    outlineById: new Map(),
    outlinePath: new Map(),
    outlineByUuid: new Map(),
    alignedDecls: new Set(),
    alignedOutline: new Set(),
    rangesByKey: new Map(),
    textByKey: anchors.textByKey,
  };
  indexOutline(index.outline, overlay, []);

  const raw = new Map<string, OverlayRange[]>();
  const push = (key: string, range: OverlayRange) => {
    const list = raw.get(key);
    if (list) list.push(range);
    else raw.set(key, [range]);
  };

  index.xrefs.forEach((xref, i) => {
    if (xref.at.t == null || xref.span.length !== 2) return;
    const key = anchorKey(xref.at);
    const text = anchors.textByKey.get(key);
    const [start, end] = xref.span;
    if (text === undefined || text.slice(start, end) !== xref.text) return;
    push(key, { kind: "xref", start, end, xref, index: i });
  });

  for (const [key, rows] of Object.entries(index.uses)) {
    const text = anchors.textByKey.get(key);
    if (text === undefined) continue;
    for (const row of rows) {
      const [start, end, termId] = row;
      const term = overlay.termsById.get(termId);
      if (!term || !spanMatchesTerm(text.slice(start, end), term.term)) continue;
      push(key, { kind: "term", start, end, termId });
      overlay.useCounts.set(termId, (overlay.useCounts.get(termId) ?? 0) + 1);
    }
  }

  // Cross-references win over term uses they overlap.
  for (const [key, ranges] of raw) {
    ranges.sort((a, b) => a.start - b.start || (a.kind === "xref" ? -1 : 1));
    const kept: OverlayRange[] = [];
    for (const range of ranges) {
      const last = kept.at(-1);
      if (last && range.start < last.end) {
        if (range.kind === "xref" && last.kind === "term") kept[kept.length - 1] = range;
        continue;
      }
      kept.push(range);
    }
    overlay.rangesByKey.set(key, kept);
  }

  // Redaction can shift ordinals in a container that holds a uuid-less
  // section, so a declaration or subsection anchor may name a different
  // paragraph in the XML the reader holds. Check them like uses.
  for (const term of index.terms) {
    for (const decl of term.decls) {
      if (!decl.anchor || decl.anchor.t == null || decl.span?.length !== 2) continue;
      const text = anchors.textByKey.get(anchorKey(decl.anchor));
      const slice = text?.slice(decl.span[0], decl.span[1]);
      if (slice !== undefined && [term.term, decl.as].some((t) => t && spanMatchesTerm(slice, t))) {
        overlay.alignedDecls.add(decl);
      }
    }
  }
  for (const node of overlay.outlineById.values()) {
    if (node.anchor.t == null) continue;
    const text = anchors.textByKey.get(anchorKey(node.anchor));
    const head = (s: string) => s.replace(/\s+/g, " ").trim().slice(0, 24);
    if (text !== undefined && head(text) === head(decodeXmlEntities(node.title))) {
      overlay.alignedOutline.add(node.id);
    }
  }
  return overlay;
}

const REGION_LABELS: Record<string, string> = {
  frontMatter: "the front matter",
  body: "the preamble",
  backMatter: "the back matter",
  sigPages: "the signature pages",
};

/** "Section 1.1", or a readable name for a region (`body` is text before Article I). */
export function declLocation(decl: ReaderTermDecl): string | null {
  const where = decl.redacted ? decl.label : decl.in;
  if (!where) return null;
  return REGION_LABELS[where] ?? where;
}

const EXCERPT_LIMIT = 600;

/**
 * Up to ~600 characters of `text` starting at the sentence that contains
 * `from`, for previews. `truncated` says whether anything was cut at the end.
 */
export function excerpt(
  text: string,
  from = 0,
  limit = EXCERPT_LIMIT,
): { text: string; truncated: boolean } {
  let start = 0;
  if (from > 0) {
    const before = text.slice(Math.max(0, from - 300), from);
    const boundary = Math.max(before.lastIndexOf(". "), before.lastIndexOf("; "), before.lastIndexOf("\n"));
    start = boundary === -1 ? Math.max(0, from - 300) : from - before.length + boundary + 1;
  }
  const body = text.slice(start).replace(/\s+/g, " ").trim();
  if (body.length <= limit) return { text: body, truncated: false };
  const cut = body.lastIndexOf(" ", limit);
  return { text: body.slice(0, cut > limit / 2 ? cut : limit), truncated: true };
}

/** "Article IV › 4.3 › 4.3(b)" pieces for an outline id. */
export function outlineTrail(overlay: ReaderOverlay, id: string): ReaderOutlineNode[] {
  const ids = [...(overlay.outlinePath.get(id) ?? []), id];
  return ids.map((i) => overlay.outlineById.get(i)).filter((n): n is ReaderOutlineNode => !!n);
}

// ---- Printed table of contents ------------------------------------------------
// The agreement's own table of contents is plain text in the XML. With an
// outline, each of its entries can link to the article or section it names.

const ROMAN_VALUES: Record<string, number> = { I: 1, V: 5, X: 10, L: 50, C: 100, D: 500, M: 1000 };
const ROMAN_RE = /^[IVXLCDM]+$/;
const TOC_ARTICLE_RE = /^ARTICLE\s+([IVXLCDM]+|\d+)\b[\s.:\u2013\u2014-]*/i;
const TOC_SECTION_RE = /^(?:Section\s+)?(\d+(?:\.\d+)*)(?=[.\s:)\u2013\u2014-]|$)[\s.:)\u2013\u2014-]*/i;

function romanToInt(roman: string): number {
  let total = 0;
  for (let i = 0; i < roman.length; i++) {
    const value = ROMAN_VALUES[roman[i]] ?? 0;
    total += value < (ROMAN_VALUES[roman[i + 1]] ?? 0) ? -value : value;
  }
  return total;
}

/** "5.08" and "5.8" are one label; so are "IV" and "4". */
function normalizeOutlineLabel(label: string): string {
  const trimmed = label.trim().replace(/[.\s]+$/, "").toUpperCase();
  if (ROMAN_RE.test(trimmed)) return String(romanToInt(trimmed));
  return trimmed
    .split(".")
    .map((part) => (/^\d+$/.test(part) ? String(Number(part)) : part))
    .join(".");
}

function normalizeHeading(text: string): string {
  return decodeXmlEntities(text)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "");
}

/** A heading without its own number: "ARTICLE II SALE AND PURCHASE" -> "saleandpurchase". */
function headingWords(text: string): string {
  const decoded = decodeXmlEntities(text).trim();
  return normalizeHeading(decoded.replace(TOC_ARTICLE_RE, "").replace(TOC_SECTION_RE, ""));
}

/**
 * A printed entry and an outline node with the same number name the same
 * place unless their words plainly differ (a table of contents that also
 * lists an exhibit's own sections, say). Printed entries are often shortened,
 * so only the start of the words is compared.
 */
function headingsAgree(entry: string, node: ReaderOutlineNode): boolean {
  const a = headingWords(entry);
  const b = headingWords(node.title);
  if (!a || !b) return true;
  const length = Math.min(a.length, b.length, 8);
  return a.slice(0, length) === b.slice(0, length);
}

interface TocLookup {
  /** "article:2" / "section:2.1" -> outline node, null when two nodes share the label. */
  byLabel: Map<string, ReaderOutlineNode | null>;
  byHeading: Map<string, ReaderOutlineNode | null>;
}

const tocLookups = new WeakMap<ReaderOverlay, TocLookup>();

function tocLookupFor(overlay: ReaderOverlay): TocLookup {
  const cached = tocLookups.get(overlay);
  if (cached) return cached;
  const lookup: TocLookup = { byLabel: new Map(), byHeading: new Map() };
  const put = (map: Map<string, ReaderOutlineNode | null>, key: string, node: ReaderOutlineNode) => {
    map.set(key, map.has(key) ? null : node);
  };
  for (const node of overlay.outlineById.values()) {
    if (node.kind !== "article" && node.kind !== "section") continue;
    if (node.label) put(lookup.byLabel, `${node.kind}:${normalizeOutlineLabel(node.label)}`, node);
    const heading = normalizeHeading(node.title);
    if (heading) put(lookup.byHeading, heading, node);
  }
  tocLookups.set(overlay, lookup);
  return lookup;
}

/**
 * The outline id a printed table-of-contents entry points at, or null when
 * the entry names no article or section of the outline (an exhibit, a
 * schedule, a number that two sections share).
 */
export function tocEntryOutlineId(overlay: ReaderOverlay, entryText: string): string | null {
  const lookup = tocLookupFor(overlay);
  const text = entryText.trim();
  const accept = (node: ReaderOutlineNode | null | undefined) =>
    node && headingsAgree(text, node) ? node.id : null;

  const article = TOC_ARTICLE_RE.exec(text);
  if (article) {
    return accept(lookup.byLabel.get(`article:${normalizeOutlineLabel(article[1])}`));
  }
  const section = TOC_SECTION_RE.exec(text);
  if (section) {
    const label = normalizeOutlineLabel(section[1]);
    const byNumber =
      lookup.byLabel.get(`section:${label}`) ??
      // "1. DEFINITIONS" in a table whose top level is numbered articles.
      (label.includes(".") || /^section\b/i.test(text) ? undefined : lookup.byLabel.get(`article:${label}`));
    if (byNumber !== undefined) return accept(byNumber);
  }
  return lookup.byHeading.get(normalizeHeading(text))?.id ?? null;
}
