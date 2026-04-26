const fs = require("node:fs");
const path = require("node:path");

const taxonomyResponse = {
  responses: {
    200: {
      description: "OK",
      content: {
        "application/json": {
          schema: {
            type: "object",
            additionalProperties: {
              type: "object",
              required: ["id"],
              properties: {
                id: { type: "string" },
                children: {
                  type: "object",
                  additionalProperties: { type: "object" },
                },
              },
            },
          },
        },
      },
    },
    default: {
      description: "Default error response",
      content: {
        "application/json": {
          schema: {
            type: "object",
            properties: {
              code: { type: "integer", description: "Error code" },
              status: { type: "string", description: "Error name" },
              message: { type: "string", description: "Error message" },
              errors: {
                type: "object",
                description: "Errors",
                additionalProperties: {},
              },
            },
            title: "Error",
          },
        },
      },
    },
  },
};

for (const relativePath of [
  "docs/pandects/get-taxonomy.StatusCodes.json",
  "docs/pandects/get-tax-clause-taxonomy.StatusCodes.json",
]) {
  const filePath = path.join(__dirname, "..", relativePath);
  fs.writeFileSync(filePath, JSON.stringify(taxonomyResponse));
}

// agreements_search has its own tag entry with a description, which causes
// docusaurus-plugin-openapi-docs to generate a tag page. That page uses
// useCurrentSidebarCategory(), which fails when the doc is a hidden sidebar
// item rather than a real category link. Fix: repoint the generated sidebar
// link directly at the endpoint doc so the tag.mdx is never referenced, then
// delete the orphan tag page.
const sidebarPath = path.join(__dirname, "..", "docs/pandects/sidebar.ts");
if (fs.existsSync(sidebarPath)) {
  const original = fs.readFileSync(sidebarPath, "utf8");
  const patched = original.replace(
    /id:\s*"pandects\/agreements-search"/g,
    'id: "pandects/search-agreements"',
  );
  if (patched !== original) fs.writeFileSync(sidebarPath, patched);
}

const tagMdxPath = path.join(__dirname, "..", "docs/pandects/agreements-search.tag.mdx");
if (fs.existsSync(tagMdxPath)) fs.unlinkSync(tagMdxPath);
