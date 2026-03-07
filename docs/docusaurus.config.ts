// @ts-check

import type * as Preset from "@docusaurus/preset-classic";
import type { Config } from "@docusaurus/types";
import type * as Plugin from "@docusaurus/types/src/plugin";
import type * as OpenApiPlugin from "docusaurus-plugin-openapi-docs";

const brandLinks = require("../branding/links.json");

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

const config: Config = {
  future: {
    // Keep this off for stability while integrating OpenAPI theme assets.
    experimental_faster: false,
    v4: true,
  },
  title: "Pandects Docs",
  tagline: "Engineering-grade docs for the Pandects API",
  url: brandLinks.docsSiteUrl,
  baseUrl: "/",
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
          docItemComponent: "@theme/ApiItem",
        },
        blog: false,
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
          autoCollapseCategories: true,
        },
      },
      colorMode: {
        disableSwitch: true,
      },
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
            label: "App",
            position: "right",
            href: `${brandLinks.mainSiteUrl}/sections`,
          },
        ],
      },
      prism: {
        additionalLanguages: [
          "ruby",
          "csharp",
          "php",
          "java",
          "powershell",
          "json",
          "bash",
          "dart",
          "objectivec",
          "r",
        ],
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
  ],

  themes: ["docusaurus-theme-openapi-docs"],
};

export default async function createConfig() {
  return config;
}
