const fs = require("node:fs");
const path = require("node:path");
const yaml = require("js-yaml");

const ROOT = path.join(__dirname, "..");
const SPEC_PATH = path.join(ROOT, "..", "frontend", "public", "openapi.yaml");
const STATIC = path.join(ROOT, "static");
const STATIC_LLMS = path.join(STATIC, "llms", "pandects");
const STATIC_GUIDES = path.join(STATIC, "llms", "guides");
const DOCS_DIR = path.join(ROOT, "docs");
const SITE_URL = "https://docs.pandects.org";

fs.mkdirSync(STATIC_LLMS, { recursive: true });
fs.mkdirSync(STATIC_GUIDES, { recursive: true });

const spec = yaml.load(fs.readFileSync(SPEC_PATH, "utf8"));

fs.copyFileSync(SPEC_PATH, path.join(STATIC, "openapi.yaml"));

const stripFrontmatter = (md) => md.replace(/^---\n[\s\S]*?\n---\n+/, "");

const readGuide = (relPath) => {
  const abs = path.join(DOCS_DIR, relPath);
  const raw = fs.readFileSync(abs, "utf8");
  const fm = raw.match(/^---\n([\s\S]*?)\n---/);
  let title = relPath;
  let description = "";
  if (fm) {
    const t = fm[1].match(/^title:\s*(.+)$/m);
    const d = fm[1].match(/^description:\s*(.+)$/m);
    if (t) title = t[1].trim();
    if (d) description = d[1].trim();
  }
  return { relPath, title, description, body: stripFrontmatter(raw).trim() };
};

const GUIDES = [
  readGuide("intro.md"),
  readGuide("guides/getting-started.md"),
  readGuide("guides/request-patterns.md"),
  readGuide("guides/error-model.md"),
  readGuide("mcp/setup.md"),
  readGuide("mcp/using.md"),
  readGuide("mcp/technical-details.md"),
];

const guideSlug = (relPath) => relPath.replace(/\.md$/, "").replace(/\//g, "__");
const guideUrl = (relPath) => `${SITE_URL}/llms/guides/${guideSlug(relPath)}.md`;

for (const g of GUIDES) {
  const out = `# ${g.title}\n\n${g.description ? `> ${g.description}\n\n` : ""}${g.body}\n`;
  fs.writeFileSync(path.join(STATIC_GUIDES, `${guideSlug(g.relPath)}.md`), out);
}

const METHODS = ["get", "post", "put", "patch", "delete"];

const renderParamSchema = (schema) => {
  if (!schema) return "";
  const parts = [];
  if (schema.type === "array") {
    const items = schema.items?.type || "string";
    parts.push(`array<${items}>`);
  } else if (schema.type) {
    parts.push(schema.type);
  }
  if (schema.enum) parts.push(`enum: ${schema.enum.join(", ")}`);
  if (schema.minimum !== undefined) parts.push(`min: ${schema.minimum}`);
  if (schema.maximum !== undefined) parts.push(`max: ${schema.maximum}`);
  if (schema.default !== undefined && schema.default !== null && schema.default !== "") {
    parts.push(`default: ${JSON.stringify(schema.default)}`);
  }
  if (schema.nullable) parts.push("nullable");
  return parts.join("; ");
};

const renderEndpoint = (method, urlPath, op) => {
  const opId = op.operationId || `${method}_${urlPath}`;
  const lines = [];
  lines.push(`# ${op.summary || opId}`);
  lines.push("");
  lines.push(`\`${method.toUpperCase()} ${urlPath}\``);
  lines.push("");
  lines.push(`Operation ID: \`${opId}\``);
  if (op.tags?.length) {
    lines.push(`Tags: ${op.tags.join(", ")}`);
  }
  lines.push("");
  if (op.description) {
    lines.push(op.description.trim());
    lines.push("");
  }

  const params = op.parameters || [];
  const pathParams = params.filter((p) => p.in === "path");
  const queryParams = params.filter((p) => p.in === "query");
  const headerParams = params.filter((p) => p.in === "header");

  const renderParamList = (heading, items) => {
    if (!items.length) return;
    lines.push(`## ${heading}`);
    lines.push("");
    for (const p of items) {
      const req = p.required ? " (required)" : "";
      const schemaDesc = renderParamSchema(p.schema);
      const head = `- \`${p.name}\`${req}${schemaDesc ? ` — ${schemaDesc}` : ""}`;
      lines.push(head);
      if (p.description) {
        lines.push(`  ${p.description.replace(/\n+/g, " ").trim()}`);
      }
    }
    lines.push("");
  };

  renderParamList("Path parameters", pathParams);
  renderParamList("Query parameters", queryParams);
  renderParamList("Header parameters", headerParams);

  if (op.requestBody) {
    lines.push("## Request body");
    lines.push("");
    const content = op.requestBody.content || {};
    for (const [mime, def] of Object.entries(content)) {
      lines.push(`Content-Type: \`${mime}\``);
      lines.push("");
      if (def.schema) {
        lines.push("```json");
        lines.push(JSON.stringify(def.schema, null, 2));
        lines.push("```");
        lines.push("");
      }
    }
  }

  if (op.responses) {
    lines.push("## Responses");
    lines.push("");
    for (const [status, resp] of Object.entries(op.responses)) {
      lines.push(`### ${status}`);
      lines.push("");
      if (resp.description) {
        lines.push(resp.description.trim());
        lines.push("");
      }
      const content = resp.content || {};
      for (const [mime, def] of Object.entries(content)) {
        lines.push(`Content-Type: \`${mime}\``);
        lines.push("");
        if (def.schema) {
          lines.push("```json");
          lines.push(JSON.stringify(def.schema, null, 2));
          lines.push("```");
          lines.push("");
        }
      }
    }
  }

  return { opId, urlPath, method, summary: op.summary || opId, tags: op.tags || [], markdown: lines.join("\n").trim() + "\n" };
};

const endpoints = [];
for (const [urlPath, item] of Object.entries(spec.paths || {})) {
  for (const method of METHODS) {
    if (!item[method]) continue;
    const ep = renderEndpoint(method, urlPath, item[method]);
    endpoints.push(ep);
    fs.writeFileSync(path.join(STATIC_LLMS, `${ep.opId}.md`), ep.markdown);
  }
}

const endpointUrl = (opId) => `${SITE_URL}/llms/pandects/${opId}.md`;

const llmsTxt = [];
llmsTxt.push("# Pandects");
llmsTxt.push("");
llmsTxt.push("> Public, read-only REST API for searching M&A agreements and reading full agreement or section text. Base URL: https://api.pandects.org. All routes versioned under /v1.");
llmsTxt.push("");
llmsTxt.push("## Guides");
llmsTxt.push("");
for (const g of GUIDES) {
  const desc = g.description ? `: ${g.description}` : "";
  llmsTxt.push(`- [${g.title}](${guideUrl(g.relPath)})${desc}`);
}
llmsTxt.push("");
llmsTxt.push("## API Reference");
llmsTxt.push("");
llmsTxt.push(`- [OpenAPI spec](${SITE_URL}/openapi.yaml): full machine-readable spec`);
for (const ep of endpoints) {
  llmsTxt.push(`- [${ep.summary}](${endpointUrl(ep.opId)}): ${ep.method.toUpperCase()} ${ep.urlPath}`);
}
llmsTxt.push("");
fs.writeFileSync(path.join(STATIC, "llms.txt"), llmsTxt.join("\n"));

const llmsFull = [];
llmsFull.push("# Pandects — Full Documentation\n");
llmsFull.push("> Concatenated guides and API reference for the Pandects API. Base URL: https://api.pandects.org.\n");
llmsFull.push("---\n");
llmsFull.push("# Part 1: Guides\n");
for (const g of GUIDES) {
  llmsFull.push(`<!-- source: docs/${g.relPath} -->`);
  llmsFull.push(g.body);
  llmsFull.push("\n---\n");
}
llmsFull.push("# Part 2: API Reference\n");
llmsFull.push(`Full OpenAPI spec: ${SITE_URL}/openapi.yaml\n`);
for (const ep of endpoints) {
  llmsFull.push(`<!-- endpoint: ${ep.method.toUpperCase()} ${ep.urlPath} -->`);
  llmsFull.push(ep.markdown);
  llmsFull.push("\n---\n");
}
fs.writeFileSync(path.join(STATIC, "llms-full.txt"), llmsFull.join("\n"));

console.log(`Generated llms.txt, llms-full.txt, openapi.yaml, and ${endpoints.length} per-endpoint files in static/`);
