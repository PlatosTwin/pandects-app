import { describe, expect, it } from "vitest";
import {
  buildXmlTreeIndex,
  findFirstTagNode,
  parseXMLContent,
  setsAgreeWithin,
  type XMLNode,
} from "./xml-parser";

describe("parseXMLContent", () => {
  it("parses nested tags with text children", () => {
    const nodes = parseXMLContent(
      "<document><body><text>hello</text></body></document>",
    );

    expect(nodes).toHaveLength(1);
    const document = nodes[0];
    expect(document.tagName).toBe("document");
    const body = document.children?.[0];
    expect(body?.tagName).toBe("body");
    const text = body?.children?.[0];
    expect(text?.tagName).toBe("text");
    expect(text?.children).toEqual([{ type: "text", content: "hello" }]);
  });

  it("extracts uuid and title attributes at parse time", () => {
    const nodes = parseXMLContent(
      '<section uuid="s-1" title="Section 1.1 &amp; More"><text>x</text></section>',
    );

    expect(nodes[0].uuid).toBe("s-1");
    expect(nodes[0].title).toBe("Section 1.1 &amp; More");
    expect(nodes[0].content).toBe(
      '<section uuid="s-1" title="Section 1.1 &amp; More">',
    );
  });

  it("leaves uuid and title undefined when absent", () => {
    const nodes = parseXMLContent("<text>plain</text>");
    expect(nodes[0].uuid).toBeUndefined();
    expect(nodes[0].title).toBeUndefined();
  });

  it("handles self-closing tags", () => {
    const nodes = parseXMLContent("<body><pageUUID/><text>x</text></body>");
    const children = nodes[0].children!;
    expect(children[0]).toMatchObject({
      type: "tag",
      tagName: "pageUUID",
      children: [],
    });
    expect(children[1].tagName).toBe("text");
  });

  it("skips XML declarations and comments", () => {
    const nodes = parseXMLContent(
      '<?xml version="1.0"?><!-- note --><text>x</text>',
    );
    expect(nodes).toHaveLength(1);
    expect(nodes[0].tagName).toBe("text");
  });

  it("keeps sibling same-name tags separate", () => {
    const nodes = parseXMLContent("<text>a</text><text>b</text>");
    expect(nodes).toHaveLength(2);
    expect(nodes[0].children).toEqual([{ type: "text", content: "a" }]);
    expect(nodes[1].children).toEqual([{ type: "text", content: "b" }]);
  });

  it("drops whitespace-only text between tags", () => {
    const nodes = parseXMLContent("<body>\n  <text>x</text>\n</body>");
    expect(nodes[0].children).toHaveLength(1);
  });

  it("emits an unclosed tag and its contents as raw text", () => {
    const nodes = parseXMLContent("<text>abc");
    expect(nodes).toEqual([{ type: "text", content: "<text>abc" }]);
  });

  it("emits nested unclosed tags as a single raw text node in the parent", () => {
    const nodes = parseXMLContent("<a><b><c>x</a>");
    expect(nodes).toHaveLength(1);
    expect(nodes[0].tagName).toBe("a");
    expect(nodes[0].children).toEqual([
      { type: "text", content: "<b><c>x" },
    ]);
  });

  it("emits a stray closing tag and the rest as raw text", () => {
    const nodes = parseXMLContent("<text>x</text></a>tail");
    expect(nodes).toEqual([
      expect.objectContaining({ type: "tag", tagName: "text" }),
      { type: "text", content: "</a>tail" },
    ]);
  });

  it("emits a lone bracket as text", () => {
    const nodes = parseXMLContent("before <after");
    expect(nodes).toEqual([
      { type: "text", content: "before " },
      { type: "text", content: "<after" },
    ]);
  });

  it("returns no nodes for empty, whitespace-only, or comment-only input", () => {
    expect(parseXMLContent("")).toEqual([]);
    expect(parseXMLContent("   \n  ")).toEqual([]);
    expect(parseXMLContent("<!-- nothing here -->")).toEqual([]);
  });
});

describe("findFirstTagNode", () => {
  it("finds the first matching tag depth-first", () => {
    const nodes = parseXMLContent(
      "<document><metadata>m</metadata><body><text>x</text></body></document>",
    );
    const body = findFirstTagNode(nodes, "body");
    expect(body?.tagName).toBe("body");
    expect(findFirstTagNode(nodes, "missing")).toBeNull();
  });
});

describe("buildXmlTreeIndex", () => {
  const nodes = parseXMLContent(
    '<document><body><article uuid="a-1" title="A1">' +
      '<section uuid="s-1" title="S1"><text>one</text></section>' +
      '<section uuid="s-2" title="S2"><text>two</text></section>' +
      "</article></body></document>",
  );
  const { meta, uuidPaths } = buildXmlTreeIndex(nodes);

  const document = nodes[0];
  const body = document.children![0];
  const article = body.children![0];
  const [sectionOne, sectionTwo] = article.children!;

  it("assigns dot-separated paths and path-based tagIds", () => {
    expect(meta.get(document)).toMatchObject({
      path: "0",
      tagId: "document-0",
    });
    expect(meta.get(body)).toMatchObject({ path: "0.0", tagId: "body-0.0" });
  });

  it("keys section-like nodes by their uuid", () => {
    expect(meta.get(article)?.tagId).toBe("a-1");
    expect(meta.get(sectionOne)?.tagId).toBe("s-1");
    expect(meta.get(sectionTwo)?.tagId).toBe("s-2");
  });

  it("maps each uuid to its root-to-node uuid path", () => {
    expect(uuidPaths.get("a-1")).toEqual(["a-1"]);
    expect(uuidPaths.get("s-1")).toEqual(["a-1", "s-1"]);
    expect(uuidPaths.get("s-2")).toEqual(["a-1", "s-2"]);
    expect(uuidPaths.get("missing")).toBeUndefined();
  });

  it("collects subtree tagIds and uuids per node", () => {
    const documentMeta = meta.get(document)!;
    expect(documentMeta.subtreeUuids).toEqual(new Set(["a-1", "s-1", "s-2"]));
    expect(documentMeta.subtreeTagIds.has("s-2")).toBe(true);
    expect(documentMeta.subtreeTagIds.has("body-0.0")).toBe(true);

    const sectionOneMeta = meta.get(sectionOne)!;
    expect(sectionOneMeta.subtreeUuids).toEqual(new Set(["s-1"]));
    expect(sectionOneMeta.subtreeUuids.has("s-2")).toBe(false);
    expect(sectionOneMeta.subtreeTagIds.has("s-2")).toBe(false);
  });

  it("keeps the first occurrence when a uuid repeats", () => {
    const dupNodes = parseXMLContent(
      '<body><section uuid="dup"><text>a</text></section>' +
        '<article uuid="a-x"><section uuid="dup"><text>b</text></section></article></body>',
    );
    const { uuidPaths: dupPaths } = buildXmlTreeIndex(dupNodes);
    expect(dupPaths.get("dup")).toEqual(["dup"]);
  });
});

describe("setsAgreeWithin", () => {
  const scope = new Set(["a", "b"]);

  it("treats identical sets as agreeing", () => {
    const s = new Set(["a"]);
    expect(setsAgreeWithin(s, s, scope)).toBe(true);
  });

  it("ignores differences outside the scope", () => {
    expect(
      setsAgreeWithin(new Set(["x"]), new Set(["y"]), scope),
    ).toBe(true);
  });

  it("detects additions and removals inside the scope", () => {
    expect(setsAgreeWithin(new Set(), new Set(["a"]), scope)).toBe(false);
    expect(setsAgreeWithin(new Set(["b"]), new Set(), scope)).toBe(false);
  });
});

describe("parse parity with rendered node shape", () => {
  it("produces the tree the renderer expects for a representative agreement", () => {
    const nodes = parseXMLContent(
      `<?xml version="1.0"?>
<document>
  <metadata><text>meta</text></metadata>
  <body>
    <article uuid="a-1" title="ARTICLE I">
      <section uuid="s-1" title="Section 1.1"><text>The merger text.</text><pageUUID/></section>
    </article>
  </body>
</document>`,
    );

    const walk = (list: XMLNode[]): string[] =>
      list.flatMap((node) =>
        node.type === "text"
          ? [`#text`]
          : [node.tagName!, ...walk(node.children ?? [])],
      );

    expect(walk(nodes)).toEqual([
      "document",
      "metadata",
      "text",
      "#text",
      "body",
      "article",
      "section",
      "text",
      "#text",
      "pageUUID",
    ]);
  });
});
