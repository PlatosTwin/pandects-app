import { ROUTE_DEFINITION_BY_PATH } from "./route-manifest.mjs";

const DEFAULT_TITLE = "Pandects";
const DEFAULT_DESCRIPTION =
  "Open-source M&A research platform with structured agreement data, clause taxonomy, API access, and downloadable SEC EDGAR merger agreement datasets.";
const OG_IMAGE_WIDTH = 1536;
const OG_IMAGE_HEIGHT = 806;
const OG_IMAGE_TYPE = "image/jpeg";
const SEO_BLOCK_REGEX = /<!-- SEO:BEGIN -->[\s\S]*?<!-- SEO:END -->/;
const TITLE_TAG_REGEX = /<title>[\s\S]*?<\/title>/i;

function normalizePathname(pathname) {
  const stripped = pathname.split("?")[0]?.split("#")[0] ?? "/";
  if (stripped.length > 1) return stripped.replace(/\/+$/, "");
  return "/";
}

function normalizeSearch(search) {
  if (!search || search === "?") return "";
  return search.startsWith("?") ? search : `?${search}`;
}

function getRouteDefinition(pathname) {
  return ROUTE_DEFINITION_BY_PATH.get(normalizePathname(pathname)) ?? null;
}

function getSeoConfigForLocation(pathname, search = "", origin) {
  const normalizedPath = normalizePathname(pathname);
  const normalizedSearch = normalizeSearch(search);
  const route = getRouteDefinition(normalizedPath);
  const ogImage = `${origin}/og.jpg`;
  const canonical = normalizedPath === "/" ? `${origin}/` : `${origin}${normalizedPath}`;

  if (!route) {
    return {
      title: "Not Found | Pandects",
      description: "The requested page does not exist.",
      canonical,
      robots: "noindex,nofollow",
      ogImage,
      pageType: "WebPage",
      pageName: "Not Found",
      pageDescription: "The requested page does not exist.",
      status: 404,
      xRobotsTag: "noindex, nofollow",
    };
  }

  const effectiveRobots =
    route.queryRobots && normalizedSearch.length > 0 ? route.queryRobots : route.robots;

  return {
    title: route.title,
    description: route.description,
    canonical,
    robots: effectiveRobots,
    ogImage,
    pageType: route.pageType,
    pageName: route.pageName,
    pageDescription: route.pageDescription,
    status: 200,
    xRobotsTag: effectiveRobots.startsWith("noindex")
      ? effectiveRobots.replaceAll(",", ", ")
      : undefined,
  };
}

function getSeoConfigForPath(pathname, origin) {
  return getSeoConfigForLocation(pathname, "", origin);
}

function buildJsonLd({ origin, canonical, pageType, pageName, pageDescription }) {
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
      width: OG_IMAGE_WIDTH,
      height: OG_IMAGE_HEIGHT,
    },
    sameAs: ["https://github.com/PlatosTwin/pandects-app"],
    description:
      "Open-source M&A research, data, API, taxonomy, and clause analysis platform",
  };

  const website = {
    "@type": "WebSite",
    "@id": websiteId,
    name: "Pandects",
    url: siteUrl,
    publisher: { "@id": organizationId },
    potentialAction: {
      "@type": "SearchAction",
      target: {
        "@type": "EntryPoint",
        urlTemplate: `${siteUrl}search?q={search_term_string}`,
      },
      "query-input": "required name=search_term_string",
    },
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
          inLanguage: "en-US",
        };

  const graph = [organization, website, page];

  if (canonical.includes("/bulk-data")) {
    graph.push({
      "@type": "Dataset",
      "@id": `${canonical}#dataset`,
      name: "Pandects M&A Agreement Dataset",
      description:
        "Structured M&A agreement data from SEC EDGAR with tagged clauses, extracted terms, metadata, and taxonomy coverage for research and API workflows.",
      url: canonical,
      keywords: [
        "open-source",
        "data API",
        "M&A",
        "merger agreements",
        "taxonomy",
        "clauses",
        "SEC EDGAR",
        "definitive agreements",
        "acquisition agreements",
        "legal research",
        "contract analysis",
      ],
      license: `${origin}/license`,
      creator: { "@id": organizationId },
      publisher: { "@id": organizationId },
      isPartOf: { "@id": websiteId },
      distribution: {
        "@type": "DataDownload",
        encodingFormat: "CSV",
        contentUrl: canonical,
      },
      temporalCoverage: "2010/..",
      spatialCoverage: {
        "@type": "Country",
        name: "United States",
      },
    });
  }

  if (pageType !== "WebSite") {
    graph.push({
      "@type": "BreadcrumbList",
      "@id": `${canonical}#breadcrumbs`,
      itemListElement: [
        {
          "@type": "ListItem",
          position: 1,
          name: "Home",
          item: siteUrl,
        },
        {
          "@type": "ListItem",
          position: 2,
          name: pageName,
          item: canonical,
        },
      ],
    });
  }

  if (pageType === "Article" || pageType === "TechArticle") {
    const articleIndex = graph.findIndex((item) => item["@id"] === `${canonical}#webpage`);
    if (articleIndex !== -1) {
      graph[articleIndex] = {
        ...graph[articleIndex],
        "@type": pageType,
        headline: pageName,
        author: { "@id": organizationId },
        publisher: { "@id": organizationId },
        mainEntityOfPage: canonical,
      };
    }
  }

  return {
    "@context": "https://schema.org",
    "@graph": graph,
  };
}

function buildSeoPage(pathname, search = "", origin) {
  const seo = getSeoConfigForLocation(pathname, search, origin);

  return {
    ...seo,
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
  };
}

function injectSeoDocument(indexHtml, seo) {
  const block = buildSeoHtmlBlock(seo);
  const withBlock = indexHtml.replace(SEO_BLOCK_REGEX, block);
  return withBlock.replace(TITLE_TAG_REGEX, `<title>${escapeHtmlText(seo.title)}</title>`);
}

function buildSeoHtmlBlock(seo) {
  const title = escapeHtmlAttribute(seo.title);
  const description = escapeHtmlAttribute(seo.description);
  const canonical = escapeHtmlAttribute(seo.canonical);
  const robots = escapeHtmlAttribute(seo.robots);
  const ogImage = escapeHtmlAttribute(seo.ogImage);
  const imageAlt = escapeHtmlAttribute("Pandects");
  const ogImageWidth = escapeHtmlAttribute(String(OG_IMAGE_WIDTH));
  const ogImageHeight = escapeHtmlAttribute(String(OG_IMAGE_HEIGHT));
  const ogImageType = escapeHtmlAttribute(OG_IMAGE_TYPE);
  const jsonLd = escapeJsonForHtmlScript(seo.jsonLd);

  return `<!-- SEO:BEGIN -->
  <meta name="description" content="${description}" />
  <link rel="canonical" href="${canonical}" />
  <meta name="robots" content="${robots}" />
  <meta name="googlebot" content="${robots}" />
  <meta name="bingbot" content="${robots}" />
  <meta property="og:type" content="website" />
  <meta property="og:locale" content="en_US" />
  <meta property="og:locale:alternate" content="en_US" />
  <meta property="og:site_name" content="Pandects" />
  <meta property="og:title" content="${title}" />
  <meta property="og:description" content="${description}" />
  <meta property="og:url" content="${canonical}" />
  <meta property="og:image" content="${ogImage}" />
  <meta property="og:image:width" content="${ogImageWidth}" />
  <meta property="og:image:height" content="${ogImageHeight}" />
  <meta property="og:image:type" content="${ogImageType}" />
  <meta property="og:image:alt" content="${imageAlt}" />
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:title" content="${title}" />
  <meta name="twitter:description" content="${description}" />
  <meta name="twitter:image" content="${ogImage}" />
  <meta name="twitter:image:alt" content="${imageAlt}" />
  <meta name="twitter:site" content="@pandects" />
  <meta name="twitter:creator" content="@pandects" />
  <script type="application/ld+json">${jsonLd}</script>
  <!-- SEO:END -->`;
}

function escapeHtmlAttribute(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function escapeHtmlText(value) {
  return value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function escapeJsonForHtmlScript(value) {
  return value
    .replace(/&/g, "\\u0026")
    .replace(/</g, "\\u003c")
    .replace(/>/g, "\\u003e")
    .replace(/\u2028/g, "\\u2028")
    .replace(/\u2029/g, "\\u2029");
}

export {
  DEFAULT_DESCRIPTION,
  DEFAULT_TITLE,
  OG_IMAGE_HEIGHT,
  OG_IMAGE_TYPE,
  OG_IMAGE_WIDTH,
  buildJsonLd,
  buildSeoHtmlBlock,
  buildSeoPage,
  getRouteDefinition,
  getSeoConfigForLocation,
  getSeoConfigForPath,
  injectSeoDocument,
  normalizePathname,
};
