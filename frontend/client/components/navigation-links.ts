// Shared nav link definitions — single source of truth for the desktop
// dropdowns, the mobile sheet and the footer so they cannot drift apart.
// Pure constants hoisted to module scope so array identity is stable
// across renders without needing useMemo(..., []).
import { getDocsUrl } from "@/lib/docs-url";
import { ACCESS_GATED_PATHS } from "@/lib/lockdown";

export type InternalNavLink = {
  kind: "internal";
  to: string;
  label: string;
  pandaTarget?: string;
};

export type ExternalNavLink = {
  kind: "external";
  href: string;
  label: string;
  pandaTarget?: string;
};

export type NavLinkItem = InternalNavLink | ExternalNavLink;

export type NavGroup = {
  id: "browse" | "insights" | "data";
  label: string;
  pandaTarget?: string;
  links: readonly NavLinkItem[];
};

const DOCS_BASE = `${getDocsUrl()}/docs`;

export const DOCS_HOME_URL = `${DOCS_BASE}/guides/getting-started`;
export const DOCS_MCP_URL = `${DOCS_BASE}/mcp/using`;
export const DOCS_CHANGELOG_URL = `${DOCS_BASE}/guides/changelog`;
export const GITHUB_URL = "https://github.com/PlatosTwin/pandects-app";

export const SEARCH_LINK: InternalNavLink = {
  kind: "internal",
  to: "/search",
  label: "Search",
  pandaTarget: "nav-search",
};

export const DOCS_LINK: ExternalNavLink = {
  kind: "external",
  href: DOCS_HOME_URL,
  label: "Docs",
  pandaTarget: "nav-docs",
};

const AGREEMENT_INDEX: InternalNavLink = {
  kind: "internal",
  to: "/agreement-index",
  label: "Agreement Index",
};
const TAXONOMY: InternalNavLink = {
  kind: "internal",
  to: "/taxonomy",
  label: "Taxonomy",
};
const LEADERBOARDS: InternalNavLink = {
  kind: "internal",
  to: "/leaderboards",
  label: "Leaderboards",
};
const TRENDS: InternalNavLink = {
  kind: "internal",
  to: "/trends-analyses",
  label: "Trends & Analyses",
};
const BULK_DATA: InternalNavLink = {
  kind: "internal",
  to: "/bulk-data",
  label: "Bulk Data",
};
const XML_SCHEMA: InternalNavLink = {
  kind: "internal",
  to: "/xml-schema",
  label: "XML Schema",
};
const SOURCES_METHODS: InternalNavLink = {
  kind: "internal",
  to: "/sources-methods",
  label: "Sources & Methods",
};
const API_MCP: ExternalNavLink = {
  kind: "external",
  href: DOCS_MCP_URL,
  label: "API & MCP",
};

// Top-level dropdown groups, in header order. The panda easter egg uses a few
// data-panda-target names as platforms; they are kept on the group triggers.
export const NAV_GROUPS: readonly NavGroup[] = [
  { id: "browse", label: "Browse", links: [AGREEMENT_INDEX, TAXONOMY] },
  {
    id: "insights",
    label: "Insights",
    pandaTarget: "nav-about",
    links: [LEADERBOARDS, TRENDS],
  },
  {
    id: "data",
    label: "Data",
    pandaTarget: "nav-bulk-data",
    links: [BULK_DATA, XML_SCHEMA, API_MCP, SOURCES_METHODS],
  },
];

export const PROJECT_LINKS: readonly InternalNavLink[] = [
  { kind: "internal", to: "/about", label: "About" },
  { kind: "internal", to: "/feedback", label: "Feedback" },
  { kind: "internal", to: "/support", label: "Support", pandaTarget: "nav-support" },
];

export const LEGAL_LINKS: readonly InternalNavLink[] = [
  { kind: "internal", to: "/license", label: "License" },
  { kind: "internal", to: "/privacy-policy", label: "Privacy Policy" },
  { kind: "internal", to: "/terms", label: "Terms" },
  { kind: "internal", to: "/license/data", label: "Data License" },
  { kind: "internal", to: "/license/software", label: "Software License" },
];

export const FOOTER_COLUMNS: readonly { heading: string; links: readonly NavLinkItem[] }[] = [
  { heading: "Product", links: [SEARCH_LINK, AGREEMENT_INDEX, TAXONOMY] },
  { heading: "Data", links: [BULK_DATA, XML_SCHEMA, SOURCES_METHODS] },
  {
    heading: "Developers",
    links: [
      DOCS_LINK,
      API_MCP,
      { kind: "external", href: DOCS_CHANGELOG_URL, label: "Changelog" },
    ],
  },
  {
    heading: "Project",
    links: [
      ...PROJECT_LINKS,
      { kind: "external", href: GITHUB_URL, label: "GitHub" },
    ],
  },
];

export function linkKey(link: NavLinkItem): string {
  return link.kind === "internal" ? link.to : link.href;
}

export function isGroupActive(group: NavGroup, pathname: string): boolean {
  return group.links.some(
    (link) => link.kind === "internal" && link.to === pathname,
  );
}

// TEMPORARY (access lockdown): mirrors <AccessGate>. The lockdown is on while
// ACCESS_GATED_PATHS is non-empty, unless the dev-only bypass is set.
export const LOCKDOWN_ACTIVE =
  ACCESS_GATED_PATHS.size > 0 &&
  !(import.meta.env.DEV && import.meta.env.VITE_DISABLE_ACCESS_GATE === "1");

export function isGatedLink(link: NavLinkItem): boolean {
  return link.kind === "internal" && ACCESS_GATED_PATHS.has(link.to);
}
