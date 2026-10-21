import { lazy, type LazyExoticComponent, type ComponentType } from "react";

/**
 * Canonical runtime route table.
 *
 * This is the single place to register a SPA route. Add an entry here and
 * `App.tsx` will pick it up. SEO metadata for the same path lives in
 * `shared/route-manifest.mjs`; a unit test cross-checks that every static
 * runtime path also has a manifest entry, so adding a route in one file but
 * forgetting the other fails CI instead of silently shipping a "Not Found"
 * tab title on a working page (see CLAUDE.md).
 */
export interface RouteEntry {
  path: string;
  component: LazyExoticComponent<ComponentType<unknown>>;
  /** When true, wrap the element in <ProtectedRoute>. */
  protected?: boolean;
  /** When true, this is the catch-all; placed last regardless of array order. */
  catchAll?: boolean;
  /**
   * Frame the page renders in, so the loading placeholder matches it and
   * nothing moves sideways when the page arrives. Defaults to `data`.
   */
  shell?: RouteShell;
}

/**
 * Mirrors the PageShell layouts, plus the landing card and StatusPage. `none`
 * draws no placeholder, for full-bleed app pages (search, the reader) whose
 * frame a boxed skeleton would not match.
 */
export type RouteShell =
  | "landing"
  | "data"
  | "docs"
  | "docs-toc"
  | "form"
  | "settings"
  | "status"
  | "none";

const lazyPage = <T extends ComponentType<unknown>>(
  loader: () => Promise<{ default: T }>,
) => lazy(loader) as LazyExoticComponent<ComponentType<unknown>>;

export const ROUTES: ReadonlyArray<RouteEntry> = [
  {
    path: "/",
    component: lazyPage(() => import("@/pages/Landing")),
    shell: "landing",
  },
  {
    path: "/search",
    component: lazyPage(() => import("@/pages/Search")),
    shell: "none",
  },
  {
    path: "/compare/tax",
    component: lazyPage(() => import("@/pages/TaxClauseCompare")),
  },
  {
    path: "/agreements/:agreementUuid",
    component: lazyPage(() => import("@/pages/Agreement")),
    shell: "none",
  },
  { path: "/bulk-data", component: lazyPage(() => import("@/pages/BulkData")) },
  {
    path: "/agreement-index",
    component: lazyPage(() => import("@/pages/AgreementIndex")),
  },
  {
    path: "/sources-methods",
    component: lazyPage(() => import("@/pages/SourcesMethods")),
    shell: "docs-toc",
  },
  {
    path: "/xml-schema",
    component: lazyPage(() => import("@/pages/XmlSchema")),
    shell: "docs",
  },
  { path: "/taxonomy", component: lazyPage(() => import("@/pages/Taxonomy")) },
  {
    path: "/leaderboards",
    component: lazyPage(() => import("@/pages/Leaderboards")),
  },
  {
    path: "/trends-analyses",
    component: lazyPage(() => import("@/pages/TrendsAnalyses")),
  },
  {
    path: "/about",
    component: lazyPage(() => import("@/pages/About")),
    shell: "docs",
  },
  {
    path: "/feedback",
    component: lazyPage(() => import("@/pages/Feedback")),
    shell: "docs",
  },
  {
    path: "/support",
    component: lazyPage(() => import("@/pages/Support")),
    shell: "docs",
  },
  {
    path: "/login",
    component: lazyPage(() => import("@/pages/Login")),
    shell: "form",
  },
  {
    path: "/signup",
    component: lazyPage(() => import("@/pages/Signup")),
    shell: "form",
  },
  {
    path: "/reset-password",
    component: lazyPage(() => import("@/pages/ResetPassword")),
    shell: "form",
  },
  {
    path: "/reset-password/confirm",
    component: lazyPage(() => import("@/pages/ResetPasswordConfirm")),
    shell: "form",
  },
  {
    path: "/verify-email",
    component: lazyPage(() => import("@/pages/VerifyEmail")),
    shell: "form",
  },
  {
    path: "/account",
    component: lazyPage(() => import("@/pages/Account")),
    protected: true,
    shell: "settings",
  },
  {
    path: "/favorites",
    component: lazyPage(() => import("@/pages/Favorites")),
    protected: true,
  },
  {
    path: "/privacy-policy",
    component: lazyPage(() => import("@/pages/PrivacyPolicy")),
    shell: "docs",
  },
  {
    path: "/terms",
    component: lazyPage(() => import("@/pages/Terms")),
    shell: "docs",
  },
  {
    path: "/license",
    component: lazyPage(() => import("@/pages/License")),
    shell: "docs",
  },
  {
    path: "/license/software",
    component: lazyPage(() => import("@/pages/SoftwareLicense")),
    shell: "docs",
  },
  {
    path: "/license/data",
    component: lazyPage(() => import("@/pages/DataLicense")),
    shell: "docs",
  },
  {
    path: "/auth/zitadel/callback",
    component: lazyPage(() => import("@/pages/AuthZitadelCallback")),
    shell: "form",
  },
  {
    path: "/oauth/consent",
    component: lazyPage(() => import("@/pages/OAuthConsent")),
    shell: "form",
    protected: true,
  },
  {
    path: "*",
    component: lazyPage(() => import("@/pages/NotFound")),
    shell: "status",
    catchAll: true,
  },
] as const;

/** Static (non-dynamic, non-catch-all) paths — used by the cross-check test. */
export const STATIC_ROUTE_PATHS: ReadonlyArray<string> = ROUTES.filter(
  (r) => !r.catchAll && !r.path.includes(":"),
).map((r) => r.path);
