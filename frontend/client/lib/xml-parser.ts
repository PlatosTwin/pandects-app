// Single-pass parser and tree index for the agreement XML rendered by
// XMLRenderer. The parser runs once per document; uuid/title attributes are
// extracted here so rendering and highlight bookkeeping never re-scan the
// XML string.

export interface XMLNode {
  type: "text" | "tag";
  content: string;
  tagName?: string;
  /** uuid="..." attribute value, extracted at parse time (tag nodes only). */
  uuid?: string;
  /** Raw title="..." attribute value, extracted at parse time (tag nodes only). */
  title?: string;
  children?: XMLNode[];
}

export interface XmlNodeMeta {
  /** Dot-separated index path from the rendered root, e.g. "0.2.1". */
  path: string;
  /** Collapse-state key: section uuid when available, else `${tagName}-${path}`. */
  tagId: string;
  /** tagIds of this node and every tag descendant. */
  subtreeTagIds: Set<string>;
  /** uuids of this node and every tag descendant that carries one. */
  subtreeUuids: Set<string>;
}

export interface XmlTreeIndex {
  meta: Map<XMLNode, XmlNodeMeta>;
  /** uuid -> uuids on the path from the root to that node (target included). */
  uuidPaths: Map<string, string[]>;
}

const UUID_ATTR_RE = /uuid="([^"]*)"/;
const TITLE_ATTR_RE = /title="([^"]*)"/;

interface OpenFrame {
  node: XMLNode;
  /** Offset of the frame's "<" so unclosed tags can be re-emitted as raw text. */
  start: number;
}

export function parseXMLContent(xmlContent: string): XMLNode[] {
  const roots: XMLNode[] = [];
  const stack: OpenFrame[] = [];
  const len = xmlContent.length;

  const target = () =>
    stack.length ? stack[stack.length - 1].node.children! : roots;

  const pushText = (content: string) => {
    if (content.trim()) {
      target().push({ type: "text", content });
    }
  };

  // Replace the top (unclosed) frame's tag node with the raw source from its
  // opening "<" up to `end`. The raw slice subsumes any parsed descendants,
  // so they are discarded along with the node.
  const unwindTopAsText = (end: number) => {
    const frame = stack.pop()!;
    const siblings = target();
    siblings.pop();
    pushText(xmlContent.slice(frame.start, end));
  };

  let i = 0;
  while (i < len) {
    const tagStart = xmlContent.indexOf("<", i);

    if (tagStart === -1) {
      pushText(xmlContent.slice(i));
      break;
    }

    if (tagStart > i) {
      pushText(xmlContent.slice(i, tagStart));
    }

    if (xmlContent.startsWith("<!--", tagStart)) {
      const commentEnd = xmlContent.indexOf("-->", tagStart + 4);
      i = commentEnd === -1 ? len : commentEnd + 3;
      continue;
    }

    if (xmlContent.startsWith("<?xml", tagStart)) {
      const declEnd = xmlContent.indexOf("?>", tagStart);
      i = declEnd === -1 ? len : declEnd + 2;
      continue;
    }

    const tagEnd = xmlContent.indexOf(">", tagStart);
    if (tagEnd === -1) {
      // Malformed: "<" with no ">", emit the rest as text.
      pushText(xmlContent.slice(tagStart));
      break;
    }

    const tagContent = xmlContent.slice(tagStart + 1, tagEnd);

    if (tagContent.startsWith("/")) {
      const closeName = tagContent.slice(1);
      let matchDepth = -1;
      for (let d = stack.length - 1; d >= 0; d--) {
        if (stack[d].node.tagName === closeName) {
          matchDepth = d;
          break;
        }
      }
      if (matchDepth === -1) {
        // Stray closing tag: emit the rest as text.
        pushText(xmlContent.slice(tagStart));
        break;
      }
      while (stack.length - 1 > matchDepth) {
        unwindTopAsText(tagStart);
      }
      stack.pop();
      i = tagEnd + 1;
      continue;
    }

    const selfClosing = tagContent.endsWith("/");
    const rawName = tagContent.split(/\s/)[0];
    const node: XMLNode = {
      type: "tag",
      content: `<${tagContent}>`,
      tagName: selfClosing ? rawName.replace("/", "") : rawName,
      uuid: UUID_ATTR_RE.exec(tagContent)?.[1],
      title: TITLE_ATTR_RE.exec(tagContent)?.[1],
      children: [],
    };

    target().push(node);
    if (!selfClosing) {
      stack.push({ node, start: tagStart });
    }
    i = tagEnd + 1;
  }

  // Unclosed tags at EOF: re-emit each as raw text, innermost first.
  while (stack.length > 0) {
    unwindTopAsText(len);
  }

  return roots;
}

export function findFirstTagNode(
  nodes: XMLNode[],
  tagName: string,
): XMLNode | null {
  for (const node of nodes) {
    if (node.type !== "tag") continue;
    if (node.tagName === tagName) return node;

    const childMatch = node.children
      ? findFirstTagNode(node.children, tagName)
      : null;
    if (childMatch) return childMatch;
  }

  return null;
}

export function isSectionLikeTag(node: XMLNode): boolean {
  return (
    node.type === "tag" &&
    (node.tagName === "article" || node.tagName === "section")
  );
}

// Walks the rendered tree once, assigning each tag node its path-based
// identity plus the uuid/tagId sets of its subtree. The subtree sets let the
// renderer decide whether a collapse/highlight change can affect a given
// node without re-rendering it.
export function buildXmlTreeIndex(nodes: XMLNode[]): XmlTreeIndex {
  const meta = new Map<XMLNode, XmlNodeMeta>();
  const uuidPaths = new Map<string, string[]>();

  const visit = (
    node: XMLNode,
    index: number,
    parentPath: string,
    uuidAncestors: string[],
  ): XmlNodeMeta | null => {
    if (node.type !== "tag") return null;

    const path = parentPath ? `${parentPath}.${index}` : String(index);
    const sectionUuid = isSectionLikeTag(node) ? node.uuid : undefined;
    const tagId = sectionUuid ?? `${node.tagName}-${path}`;
    const uuidChain = node.uuid ? [...uuidAncestors, node.uuid] : uuidAncestors;
    if (node.uuid && !uuidPaths.has(node.uuid)) {
      uuidPaths.set(node.uuid, uuidChain);
    }

    const subtreeTagIds = new Set([tagId]);
    const subtreeUuids = new Set(node.uuid ? [node.uuid] : []);
    node.children?.forEach((child, childIndex) => {
      const childMeta = visit(child, childIndex, path, uuidChain);
      if (childMeta) {
        for (const id of childMeta.subtreeTagIds) subtreeTagIds.add(id);
        for (const uuid of childMeta.subtreeUuids) subtreeUuids.add(uuid);
      }
    });

    const nodeMeta: XmlNodeMeta = { path, tagId, subtreeTagIds, subtreeUuids };
    meta.set(node, nodeMeta);
    return nodeMeta;
  };

  nodes.forEach((node, index) => visit(node, index, "", []));
  return { meta, uuidPaths };
}

/**
 * True when `a` and `b` agree on every member of `scope` — i.e. swapping one
 * set for the other cannot change rendering decisions keyed on scope members.
 */
export function setsAgreeWithin(
  a: ReadonlySet<string>,
  b: ReadonlySet<string>,
  scope: ReadonlySet<string>,
): boolean {
  if (a === b) return true;
  for (const value of a) {
    if (!b.has(value) && scope.has(value)) return false;
  }
  for (const value of b) {
    if (!a.has(value) && scope.has(value)) return false;
  }
  return true;
}
