import type { Request } from "express";
import { buildJsonLd, getSeoConfigForPath } from "../shared/seo-helpers.mjs";

export type SeoPage = {
  title: string;
  description: string;
  canonical: string;
  robots: string;
  ogImage: string;
  jsonLd: string;
  status: number;
  xRobotsTag?: string;
};

const DEFAULT_ORIGIN = "https://pandects.org";
const CONFIGURED_ORIGIN = (process.env.PUBLIC_ORIGIN || "").trim().replace(/\/+$/, "");

const KNOWN_ROUTES = new Set([
  "/",
  "/search",
  "/docs",
  "/bulk-data",
  "/agreement-index",
  "/about",
  "/sources-methods",
  "/feedback",
  "/donate",
  "/privacy-policy",
  "/terms",
  "/license",
  "/account",
  "/auth/forgot-password",
  "/auth/reset-password",
  "/auth/google/callback",
]);

export function isKnownRoute(pathname: string): boolean {
  return KNOWN_ROUTES.has(normalizePathname(pathname));
}

export function getPublicOrigin(req: Request): string {
  const fallback = CONFIGURED_ORIGIN || DEFAULT_ORIGIN;

  const forwardedProto = req.get("x-forwarded-proto");
  const forwardedHost = req.get("x-forwarded-host");
  const hostHeader = forwardedHost ?? req.get("host");
  if (!hostHeader) return fallback;

  const host = hostHeader.split(",", 1)[0]?.trim();
  if (!host) return fallback;

  const protoHeader = (forwardedProto ?? req.protocol).split(",", 1)[0]?.trim().toLowerCase();
  const protocol = protoHeader === "http" || protoHeader === "https" ? protoHeader : "https";

  try {
    const url = new URL(`${protocol}://${host}`);
    if (url.username || url.password) return fallback;
    if (url.protocol !== "http:" && url.protocol !== "https:") return fallback;
    return url.origin.replace(/\/+$/, "");
  } catch {
    return fallback;
  }
}

export function getSeoForPath(pathname: string, origin: string): SeoPage {
  const normalizedPath = normalizePathname(pathname);
  const ogImage = `${origin}/og.jpg`;

  if (!isKnownRoute(normalizedPath)) {
    const canonical = `${origin}${normalizedPath}`;
    return {
      title: "Not Found | Pandects",
      description: "The requested page does not exist.",
      canonical,
      robots: "noindex,nofollow",
      ogImage,
      jsonLd: JSON.stringify(
        {
          "@context": "https://schema.org",
          "@type": "WebPage",
          name: "Not Found",
          url: canonical,
        },
        null,
        0,
      ),
      status: 404,
      xRobotsTag: "noindex, nofollow",
    };
  }

  const seo = getSeoConfigForPath(normalizedPath, origin);

  return {
    title: seo.title,
    description: seo.description,
    canonical: seo.canonical,
    robots: seo.robots,
    ogImage: seo.ogImage,
    jsonLd: JSON.stringify(
      buildJsonLd({
        origin,
        canonical: seo.canonical,
        pageType: seo.pageType,
        pageName: seo.pageName,
        pageDescription: seo.pageDescription,
      }),
      null,
      0,
    ),
    status: 200,
    xRobotsTag: seo.robots.includes("noindex") ? "noindex, nofollow" : undefined,
  };
}

export function injectSeoBlock(indexHtml: string, seo: SeoPage): string {
  const block = buildSeoBlock(seo);
  const withBlock = indexHtml.replace(SEO_BLOCK_REGEX, block);
  return withBlock.replace(TITLE_TAG_REGEX, `<title>${escapeHtmlText(seo.title)}</title>`);
}

const SEO_BLOCK_REGEX = /<!-- SEO:BEGIN -->[\s\S]*?<!-- SEO:END -->/;
const TITLE_TAG_REGEX = /<title>[\s\S]*?<\/title>/i;

function normalizePathname(pathname: string): string {
  const stripped = pathname.split("?")[0]?.split("#")[0] ?? "/";
  if (stripped.length > 1) return stripped.replace(/\/+$/, "");
  return "/";
}

function escapeHtmlAttribute(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function escapeHtmlText(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function buildSeoBlock(seo: SeoPage): string {
  const title = escapeHtmlAttribute(seo.title);
  const description = escapeHtmlAttribute(seo.description);
  const canonical = escapeHtmlAttribute(seo.canonical);
  const robots = escapeHtmlAttribute(seo.robots);
  const ogImage = escapeHtmlAttribute(seo.ogImage);
  const imageAlt = escapeHtmlAttribute("Pandects");
  const jsonLd = escapeJsonForHtmlScript(seo.jsonLd);

  return `<!-- SEO:BEGIN -->
  <meta name="description" content="${description}" />
  <link rel="canonical" href="${canonical}" />
  <meta name="robots" content="${robots}" />

  <meta property="og:type" content="website" />
  <meta property="og:locale" content="en_US" />
  <meta property="og:site_name" content="Pandects" />
  <meta property="og:title" content="${title}" />
  <meta property="og:description" content="${description}" />
  <meta property="og:url" content="${canonical}" />
  <meta property="og:image" content="${ogImage}" />
  <meta property="og:image:alt" content="${imageAlt}" />

  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:title" content="${title}" />
  <meta name="twitter:description" content="${description}" />
  <meta name="twitter:image" content="${ogImage}" />
  <meta name="twitter:image:alt" content="${imageAlt}" />

  <script type="application/ld+json">${jsonLd}</script>
  <!-- SEO:END -->`;
}

function escapeJsonForHtmlScript(value: string): string {
  return value
    .replace(/&/g, "\\u0026")
    .replace(/</g, "\\u003c")
    .replace(/>/g, "\\u003e")
    .replace(/\u2028/g, "\\u2028")
    .replace(/\u2029/g, "\\u2029");
}
