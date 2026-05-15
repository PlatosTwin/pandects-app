import { lazy, type LazyExoticComponent, type ComponentType } from "react";

/**
 * Canonical runtime route table.
 *
 * This is the single place to register a SPA route. Add an entry here and
 * `main.tsx` will pick it up. SEO metadata for the same path lives in
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
}

const lazyPage = <T extends ComponentType<unknown>>(
  loader: () => Promise<{ default: T }>,
) => lazy(loader) as LazyExoticComponent<ComponentType<unknown>>;

export const ROUTES: ReadonlyArray<RouteEntry> = [
  { path: "/", component: lazyPage(() => import("@/pages/Landing")) },
  { path: "/search", component: lazyPage(() => import("@/pages/Search")) },
  {
    path: "/compare/tax",
    component: lazyPage(() => import("@/pages/TaxClauseCompare")),
  },
  {
    path: "/agreements/:agreementUuid",
    component: lazyPage(() => import("@/pages/Agreement")),
  },
  { path: "/bulk-data", component: lazyPage(() => import("@/pages/BulkData")) },
  {
    path: "/agreement-index",
    component: lazyPage(() => import("@/pages/AgreementIndex")),
  },
  {
    path: "/sources-methods",
    component: lazyPage(() => import("@/pages/SourcesMethods")),
  },
  { path: "/xml-schema", component: lazyPage(() => import("@/pages/XmlSchema")) },
  { path: "/taxonomy", component: lazyPage(() => import("@/pages/Taxonomy")) },
  {
    path: "/leaderboards",
    component: lazyPage(() => import("@/pages/Leaderboards")),
  },
  {
    path: "/trends-analyses",
    component: lazyPage(() => import("@/pages/TrendsAnalyses")),
  },
  { path: "/about", component: lazyPage(() => import("@/pages/About")) },
  { path: "/feedback", component: lazyPage(() => import("@/pages/Feedback")) },
  { path: "/support", component: lazyPage(() => import("@/pages/Support")) },
  { path: "/login", component: lazyPage(() => import("@/pages/Login")) },
  { path: "/signup", component: lazyPage(() => import("@/pages/Signup")) },
  {
    path: "/reset-password",
    component: lazyPage(() => import("@/pages/ResetPassword")),
  },
  {
    path: "/reset-password/confirm",
    component: lazyPage(() => import("@/pages/ResetPasswordConfirm")),
  },
  {
    path: "/verify-email",
    component: lazyPage(() => import("@/pages/VerifyEmail")),
  },
  {
    path: "/account",
    component: lazyPage(() => import("@/pages/Account")),
    protected: true,
  },
  {
    path: "/favorites",
    component: lazyPage(() => import("@/pages/Favorites")),
    protected: true,
  },
  {
    path: "/privacy-policy",
    component: lazyPage(() => import("@/pages/PrivacyPolicy")),
  },
  { path: "/terms", component: lazyPage(() => import("@/pages/Terms")) },
  { path: "/license", component: lazyPage(() => import("@/pages/License")) },
  {
    path: "/license/software",
    component: lazyPage(() => import("@/pages/SoftwareLicense")),
  },
  {
    path: "/license/data",
    component: lazyPage(() => import("@/pages/DataLicense")),
  },
  {
    path: "/auth/zitadel/callback",
    component: lazyPage(() => import("@/pages/AuthZitadelCallback")),
  },
  {
    path: "*",
    component: lazyPage(() => import("@/pages/NotFound")),
    catchAll: true,
  },
] as const;

/** Static (non-dynamic, non-catch-all) paths — used by the cross-check test. */
export const STATIC_ROUTE_PATHS: ReadonlyArray<string> = ROUTES.filter(
  (r) => !r.catchAll && !r.path.includes(":"),
).map((r) => r.path);
