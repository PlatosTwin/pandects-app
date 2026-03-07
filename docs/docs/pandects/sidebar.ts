import type { SidebarsConfig } from "@docusaurus/plugin-content-docs";

const sidebar: SidebarsConfig = {
  apisidebar: [
    {
      type: "doc",
      id: "pandects/pandects-api",
    },
    {
      type: "category",
      label: "agreements",
      link: {
        type: "doc",
        id: "pandects/agreements",
      },
      items: [
        {
          type: "doc",
          id: "pandects/list-agreements",
          label: "/v1/agreements",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "pandects/get-agreement",
          label: "/v1/agreements/{agreement_uuid}",
          className: "api-method get",
        },
      ],
    },
    {
      type: "category",
      label: "sections",
      link: {
        type: "doc",
        id: "pandects/sections",
      },
      items: [
        {
          type: "doc",
          id: "pandects/list-sections",
          label: "/v1/sections",
          className: "api-method get",
        },
        {
          type: "doc",
          id: "pandects/get-section",
          label: "/v1/sections/{section_uuid}",
          className: "api-method get",
        },
      ],
    },
    {
      type: "category",
      label: "taxonomy",
      link: {
        type: "doc",
        id: "pandects/taxonomy",
      },
      items: [
        {
          type: "doc",
          id: "pandects/get-taxonomy",
          label: "/v1/taxonomy",
          className: "api-method get",
        },
      ],
    },
    {
      type: "category",
      label: "naics",
      link: {
        type: "doc",
        id: "pandects/naics",
      },
      items: [
        {
          type: "doc",
          id: "pandects/get-naics",
          label: "/v1/naics",
          className: "api-method get",
        },
      ],
    },
    {
      type: "category",
      label: "dumps",
      link: {
        type: "doc",
        id: "pandects/dumps",
      },
      items: [
        {
          type: "doc",
          id: "pandects/list-dumps",
          label: "/v1/dumps",
          className: "api-method get",
        },
      ],
    },
  ],
};

export default sidebar.apisidebar;
