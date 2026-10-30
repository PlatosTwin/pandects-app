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
