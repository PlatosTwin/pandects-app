// Shared nav link definitions — single source of truth for the desktop
// dropdowns and the mobile sheet so the two menus cannot drift apart.
// Pure constants hoisted to module scope so array identity is stable
// across renders without needing useMemo(..., []).

export type AboutLink = { to: string; label: string; pandaTarget?: string };

export const ABOUT_LINKS: readonly AboutLink[] = [
  { to: "/about", label: "About", pandaTarget: "nav-about" },
  { to: "/feedback", label: "Feedback" },
  { to: "/support", label: "Support", pandaTarget: "nav-support" },
];

export const DATA_LINKS = [
  { type: "link", to: "/bulk-data", label: "Bulk Data", pandaTarget: "nav-bulk-data" },
  { type: "link", to: "/agreement-index", label: "Agreement Index" },
  { type: "link", to: "/sources-methods", label: "Sources & Methods" },
  { type: "link", to: "/xml-schema", label: "XML Schema" },
  { type: "link", to: "/taxonomy", label: "Taxonomy" },
  { type: "separator", key: "data-divider-1" },
  { type: "link", to: "/leaderboards", label: "Leaderboards" },
  { type: "link", to: "/trends-analyses", label: "Trends & Analyses" },
] as const;
