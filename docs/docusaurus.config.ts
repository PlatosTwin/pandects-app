// @ts-check

import type * as Preset from "@docusaurus/preset-classic";
import type { Config } from "@docusaurus/types";
import type * as Plugin from "@docusaurus/types/src/plugin";
import type * as OpenApiPlugin from "docusaurus-plugin-openapi-docs";

const brandLinks = require("../branding/links.json");
const path = require("node:path");
const docsOgImage = `${brandLinks.mainSiteUrl.replace(/\/+$/, "")}/og.jpg`;

function createOpenApiSidebarDocItem(item: any, context: any) {
  const id = item.type === "schema" ? `schemas/${item.id}` : item.id;
  const classNameParts: string[] = [];

  if (item.type === "api") {
    if (item.api?.deprecated) {
      classNameParts.push("menu__list-item--deprecated");
    }
    if (item.api?.method) {
      classNameParts.push("api-method", item.api.method);
    }
  } else if (item.schema?.deprecated) {
    classNameParts.push("menu__list-item--deprecated", "schema");
  }

  const docId =
    context.basePath === "" || context.basePath === undefined
      ? `${id}`
      : `${context.basePath}/${id}`;

  return {
    type: "doc",
    id: docId,
    label:
      item.type === "api"
        ? item.api.path
        : (item.frontMatter?.sidebar_label as string) ?? item.title ?? id,
    customProps: context.sidebarOptions?.customProps,
    className: classNameParts.length > 0 ? classNameParts.join(" ") : undefined,
  };
}

function isOpenApiModule(resource: string): boolean {
  return (
    resource.includes(`${path.sep}docusaurus-theme-openapi-docs${path.sep}`) ||
    resource.includes(`${path.sep}docusaurus-plugin-openapi-docs${path.sep}`) ||
    resource.includes(`${path.sep}postman-collection${path.sep}`) ||
    resource.includes(`${path.sep}react-redux${path.sep}`) ||
    resource.includes(`${path.sep}@reduxjs${path.sep}toolkit${path.sep}`) ||
    resource.includes(`${path.sep}redux${path.sep}`) ||
    resource.includes(`${path.sep}redux-thunk${path.sep}`) ||
    resource.includes(`${path.sep}immer${path.sep}`) ||
    resource.includes(`${path.sep}pako${path.sep}`) ||
    resource.includes(path.join(__dirname, "src", "theme", "Api")) ||
    resource.includes(path.join(__dirname, "src", "theme", "MimeTabs")) ||
    resource.includes(path.join(__dirname, "src", "theme", "SchemaItem")) ||
    resource.includes(path.join(__dirname, "src", "theme", "SchemaTabs")) ||
    resource.includes(path.join(__dirname, "src", "theme", "ApiTabs"))
  );
}

function docsPerfPlugin(): Plugin.PluginModule {
  return {
    name: "docs-perf-plugin",
    configureWebpack(_config, isServer) {
      if (isServer) {
        return {};
      }

      return {
        optimization: {
          splitChunks: {
            cacheGroups: {
              openapi: {
                name: "openapi",
                chunks: "async",
                enforce: true,
                priority: 45,
                reuseExistingChunk: true,
                test(module: {
                  nameForCondition?: (() => string | null | undefined) | undefined;
                }) {
                  const resource = module.nameForCondition?.() ?? "";
                  return resource !== "" && isOpenApiModule(resource);
                },
              },
              prism: {
                name: "prism",
                chunks: "all",
                enforce: true,
                priority: 44,
                reuseExistingChunk: true,
                test(module: {
                  nameForCondition?: (() => string | null | undefined) | undefined;
                }) {
                  const resource = module.nameForCondition?.() ?? "";
                  return (
                    resource.includes(`${path.sep}prismjs${path.sep}`) ||
                    resource.includes(
                      `${path.sep}prism-react-renderer${path.sep}`
                    )
                  );
                },
              },
            },
          },
        },
      };
    },
  };
}

const config: Config = {
  future: {
    // Keep this off for stability while integrating OpenAPI theme assets.
    faster: false,
    v4: true,
  },
  title: "Pandects Docs",
  tagline: "Open-source M&A data, API, taxonomy, and clause research docs",
  url: brandLinks.docsSiteUrl,
  baseUrl: "/",
  trailingSlash: false,
  onBrokenLinks: "throw",
  favicon: "img/pandects-logo-128.png",
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: "warn",
    },
  },
  customFields: {
    mainSiteUrl: brandLinks.mainSiteUrl,
  },
  presets: [
    [
      "classic",
      {
        docs: {
          sidebarPath: require.resolve("./sidebars.ts"),
          docItemComponent: "@theme/DocItem",
        },
        blog: false,
        sitemap: {
          ignorePatterns: ["/"],
        },
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      } satisfies Preset.Options,
    ],
  ],
  themeConfig:
    {
      docs: {
        sidebar: {
          hideable: true,
          autoCollapseCategories: false,
        },
      },
      colorMode: {
        disableSwitch: true,
      },
      image: docsOgImage,
      metadata: [
        {
          name: "description",
          content:
            "Open-source documentation for Pandects covering M&A data, API usage, research workflows, taxonomy structure, and clause-level access.",
        },
        {
          name: "twitter:site",
          content: "@pandects",
        },
        {
          name: "twitter:creator",
          content: "@pandects",
        },
        {
          property: "og:site_name",
          content: "Pandects Docs",
        },
      ],
      navbar: {
        title: "v1.0",
        logo: {
          alt: "Pandects",
          src: "img/pandects-logo-128.png",
        },
        items: [
          {
            type: "doc",
            docId: "guides/getting-started",
            position: "left",
            label: "Guides",
          },
          {
            type: "doc",
            docId: "pandects/list-sections",
            position: "left",
            label: "API Reference",
          },
          {
            type: "doc",
            docId: "mcp/using",
            position: "left",
            label: "MCP",
          },
          {
            label: "App",
            position: "right",
            href: `${brandLinks.mainSiteUrl}/search`,
          },
        ],
      },
      prism: {
        additionalLanguages: ["json", "bash", "python"],
      },
      languageTabs: [
        {
          highlight: "python",
          language: "python",
          logoClass: "python",
          variant: "Requests",
          variants: ["Requests", "http.client"],
        },
        { highlight: "bash", language: "curl", logoClass: "curl" },
        { highlight: "javascript", language: "nodejs", logoClass: "nodejs" },
      ],
    } satisfies Preset.ThemeConfig,

  plugins: [
    [
      "docusaurus-plugin-openapi-docs",
      {
        id: "openapi",
        docsPluginId: "classic",
        config: {
          pandects: {
            specPath: "../frontend/public/openapi.yaml",
            outputDir: "docs/pandects",
            downloadUrl: "/openapi.yaml",
            sidebarOptions: {
              groupPathsBy: "tag",
              categoryLinkSource: "tag",
              sidebarGenerators: {
                createDocItem: createOpenApiSidebarDocItem,
              },
            },
          } satisfies OpenApiPlugin.Options,
        } satisfies Plugin.PluginOptions,
      },
    ],
    docsPerfPlugin,
  ],

  themes: ["docusaurus-theme-openapi-docs"],
};

export default async function createConfig() {
  return config;
}
