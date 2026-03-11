import { ROUTE_DEFINITION_BY_PATH } from "./route-manifest.mjs";

const DEFAULT_TITLE = "Pandects";
const DEFAULT_DESCRIPTION =
  "Search and download structured M&A agreements from SEC EDGAR. Tag clauses, extract terms, and export CSVs.";
const OG_IMAGE_WIDTH = 1536;
const OG_IMAGE_HEIGHT = 806;
const OG_IMAGE_TYPE = "image/jpeg";

function normalizePathname(pathname) {
  const stripped = pathname.split("?")[0]?.split("#")[0] ?? "/";
  if (stripped.length > 1) return stripped.replace(/\/+$/, "");
  return "/";
}

function getSeoConfigForPath(pathname, origin) {
  const normalizedPath = normalizePathname(pathname);
  const ogImage = `${origin}/og.jpg`;
  const canonical = normalizedPath === "/" ? `${origin}/` : `${origin}${normalizedPath}`;
  const page = ROUTE_DEFINITION_BY_PATH.get(normalizedPath) ?? {
    title: "Not Found | Pandects",
    description: "The requested page does not exist.",
    robots: "noindex,nofollow",
    pageType: "WebPage",
    pageName: "Not Found",
    pageDescription: "The requested page does not exist.",
  };

  return {
    title: page.title,
    description: page.description,
    canonical,
    robots: page.robots,
    ogImage,
    pageType: page.pageType,
    pageName: page.pageName,
    pageDescription: page.pageDescription,
  };
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
      width: 1536,
      height: 806,
    },
    sameAs: [
      "https://github.com/PlatosTwin/pandects-app",
    ],
    description: "Open-source M&A agreement search and data platform",
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
          datePublished: new Date().toISOString(),
          dateModified: new Date().toISOString(),
          inLanguage: "en-US",
        };

  const graph = [organization, website, page];

  // Add Dataset schema for bulk data page
  if (canonical.includes("/bulk-data")) {
    graph.push({
      "@type": "Dataset",
      "@id": `${canonical}#dataset`,
      name: "Pandects M&A Agreement Dataset",
      description:
        "Structured M&A agreements from SEC EDGAR. Includes tagged clauses, extracted terms, and metadata for research and analysis.",
      url: canonical,
      keywords: [
        "M&A",
        "merger agreements",
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

  // Add Article schema for article-type pages
  if (pageType === "Article" || pageType === "TechArticle") {
    const articleIndex = graph.findIndex((item) => item["@id"] === `${canonical}#webpage`);
    if (articleIndex !== -1) {
      graph[articleIndex] = {
        ...graph[articleIndex],
        "@type": pageType,
        headline: pageName,
        author: { "@id": organizationId },
        publisher: { "@id": organizationId },
        mainEntityOfPage: { "@id": `${canonical}#webpage` },
      };
    }
  }

  return {
    "@context": "https://schema.org",
    "@graph": graph,
  };
}

export {
  DEFAULT_DESCRIPTION,
  DEFAULT_TITLE,
  OG_IMAGE_HEIGHT,
  OG_IMAGE_TYPE,
  OG_IMAGE_WIDTH,
  buildJsonLd,
  getSeoConfigForPath,
};
