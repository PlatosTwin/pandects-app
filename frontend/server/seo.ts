import type { Request } from "express";

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

const KNOWN_ROUTES = new Set([
  "/",
  "/search",
  "/docs",
  "/bulk-data",
  "/about",
  "/feedback",
  "/donate",
]);

export function isKnownRoute(pathname: string): boolean {
  return KNOWN_ROUTES.has(normalizePathname(pathname));
}

export function getPublicOrigin(req: Request): string {
  const forwardedProto = req.get("x-forwarded-proto");
  const forwardedHost = req.get("x-forwarded-host");
  const host = forwardedHost ?? req.get("host");

  if (!host) return DEFAULT_ORIGIN;

  const protocol = forwardedProto ?? req.protocol;
  return `${protocol}://${host}`.replace(/\/+$/, "");
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

  const canonical = normalizedPath === "/" ? `${origin}/` : `${origin}${normalizedPath}`;

  const page = (() => {
    switch (normalizedPath) {
      case "/":
        return {
          title: "Pandects — M&A Agreement Research",
          description:
            "Pandects is an open-source M&A research platform for exploring and analyzing merger agreement clauses.",
          jsonLd: buildJsonLd({
            origin,
            canonical: `${origin}/`,
            pageType: "WebSite",
            pageName: "Pandects",
            pageDescription:
              "Pandects is an open-source M&A research platform for exploring and analyzing merger agreement clauses.",
          }),
        };
      case "/search":
        return {
          title: "Search Merger Agreements | Pandects",
          description:
            "Search and filter merger agreement clauses across deals, years, and parties in the Pandects database.",
          jsonLd: buildJsonLd({
            origin,
            canonical,
            pageType: "WebPage",
            pageName: "Search Merger Agreements",
            pageDescription:
              "Search and filter merger agreement clauses across deals, years, and parties in the Pandects database.",
          }),
        };
      case "/docs":
        return {
          title: "Docs | Pandects",
          description:
            "Documentation for Pandects: data sources, coverage, methodology, and how to use the platform.",
          jsonLd: buildJsonLd({
            origin,
            canonical,
            pageType: "WebPage",
            pageName: "Docs",
            pageDescription:
              "Documentation for Pandects: data sources, coverage, methodology, and how to use the platform.",
          }),
        };
      case "/bulk-data":
        return {
          title: "Bulk Data | Pandects",
          description:
            "Download bulk datasets and exports from Pandects for research and analysis.",
          jsonLd: buildJsonLd({
            origin,
            canonical,
            pageType: "WebPage",
            pageName: "Bulk Data",
            pageDescription:
              "Download bulk datasets and exports from Pandects for research and analysis.",
          }),
        };
      case "/about":
        return {
          title: "About | Pandects",
          description:
            "Learn what Pandects is, why it exists, and how it’s built as an open-source M&A research platform.",
          jsonLd: buildJsonLd({
            origin,
            canonical,
            pageType: "AboutPage",
            pageName: "About",
            pageDescription:
              "Learn what Pandects is, why it exists, and how it’s built as an open-source M&A research platform.",
          }),
        };
      case "/feedback":
        return {
          title: "Feedback | Pandects",
          description:
            "Share feedback, report issues, or suggest improvements for the Pandects platform.",
          jsonLd: buildJsonLd({
            origin,
            canonical,
            pageType: "WebPage",
            pageName: "Feedback",
            pageDescription:
              "Share feedback, report issues, or suggest improvements for the Pandects platform.",
          }),
        };
      case "/donate":
        return {
          title: "Donate | Pandects",
          description:
            "Support Pandects to help maintain and expand open access to M&A agreement research data.",
          jsonLd: buildJsonLd({
            origin,
            canonical,
            pageType: "WebPage",
            pageName: "Donate",
            pageDescription:
              "Support Pandects to help maintain and expand open access to M&A agreement research data.",
          }),
        };
    }
  })();

  return {
    title: page.title,
    description: page.description,
    canonical,
    robots: "index,follow,max-image-preview:large",
    ogImage,
    jsonLd: JSON.stringify(page.jsonLd, null, 0),
    status: 200,
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

  <script type="application/ld+json">${seo.jsonLd}</script>
  <!-- SEO:END -->`;
}

type JsonLdParams = {
  origin: string;
  canonical: string;
  pageType: "WebSite" | "WebPage" | "AboutPage";
  pageName: string;
  pageDescription: string;
};

function buildJsonLd(params: JsonLdParams): Record<string, unknown> {
  const { origin, canonical, pageType, pageName, pageDescription } = params;
  const siteUrl = `${origin}/`;

  const organizationId = `${origin}/#organization`;
  const websiteId = `${origin}/#website`;

  const organization = {
    "@type": "Organization",
    "@id": organizationId,
    name: "Pandects",
    url: siteUrl,
    logo: {
      "@type": "ImageObject",
      url: `${origin}/og.jpg`,
    },
  };

  const website = {
    "@type": "WebSite",
    "@id": websiteId,
    name: "Pandects",
    url: siteUrl,
    publisher: { "@id": organizationId },
  };

  const page =
    pageType === "WebSite"
      ? {
          "@type": "WebSite",
          "@id": `${siteUrl}#website-home`,
          name: pageName,
          url: siteUrl,
          description: pageDescription,
          publisher: { "@id": organizationId },
        }
      : {
          "@type": pageType,
          "@id": `${canonical}#webpage`,
          name: pageName,
          url: canonical,
          description: pageDescription,
          isPartOf: { "@id": websiteId },
          about: { "@id": organizationId },
        };

  return {
    "@context": "https://schema.org",
    "@graph": [organization, website, page],
  };
}
