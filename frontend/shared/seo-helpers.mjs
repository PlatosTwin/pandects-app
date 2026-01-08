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

  const page = (() => {
    switch (normalizedPath) {
      case "/":
        return {
          title: "Pandects - Open-Source M&A Agreement Search & Data",
          description: DEFAULT_DESCRIPTION,
          robots: "index,follow,max-image-preview:large",
          pageType: "WebSite",
          pageName: "Pandects",
          pageDescription: DEFAULT_DESCRIPTION,
        };
      case "/search":
        return {
          title: "Search Merger Agreements | Pandects",
          description:
            "Search and filter merger agreement clauses across deals, years, and parties in the Pandects database.",
          robots: "index,follow,max-image-preview:large",
          pageType: "WebPage",
          pageName: "Search Merger Agreements",
          pageDescription:
            "Search and filter merger agreement clauses across deals, years, and parties in the Pandects database.",
        };
      case "/docs":
        return {
          title: "API Docs | Pandects",
          description:
            "Explore the Pandects API via OpenAPI and learn how to query agreements, clauses, and metadata.",
          robots: "index,follow,max-image-preview:large",
          pageType: "WebPage",
          pageName: "API Docs",
          pageDescription:
            "Explore the Pandects API via OpenAPI and learn how to query agreements, clauses, and metadata.",
        };
      case "/xml-schema":
        return {
          title: "XML Schema | Pandects",
          description:
            "Reference the Pandects XML schema for agreement exports, including structure and element definitions.",
          robots: "index,follow,max-image-preview:large",
          pageType: "WebPage",
          pageName: "XML Schema",
          pageDescription:
            "Reference the Pandects XML schema for agreement exports, including structure and element definitions.",
        };
      case "/bulk-data":
        return {
          title: "Bulk Data | Pandects",
          description:
            "Download bulk datasets and exports from Pandects for research and analysis.",
          robots: "index,follow,max-image-preview:large",
          pageType: "WebPage",
          pageName: "Bulk Data",
          pageDescription:
            "Download bulk datasets and exports from Pandects for research and analysis.",
        };
      case "/agreement-index":
        return {
          title: "Agreement Index | Pandects",
          description:
            "Browse all merger agreements in Pandects with sortable metadata and high-level dataset statistics.",
          robots: "index,follow,max-image-preview:large",
          pageType: "WebPage",
          pageName: "Agreement Index",
          pageDescription:
            "Browse all merger agreements in Pandects with sortable metadata and high-level dataset statistics.",
        };
      case "/sources-methods":
        return {
          title: "Sources & Methods | Pandects",
          description:
            "Learn where Pandects data comes from and how it is processed into structured datasets.",
          robots: "index,follow,max-image-preview:large",
          pageType: "WebPage",
          pageName: "Sources & Methods",
          pageDescription:
            "Learn where Pandects data comes from and how it is processed into structured datasets.",
        };
      case "/about":
        return {
          title: "About | Pandects",
          description:
            "Learn what Pandects is, why it exists, and how it's built as an open-source M&A research platform.",
          robots: "index,follow,max-image-preview:large",
          pageType: "AboutPage",
          pageName: "About",
          pageDescription:
            "Learn what Pandects is, why it exists, and how it's built as an open-source M&A research platform.",
        };
      case "/feedback":
        return {
          title: "Feedback | Pandects",
          description:
            "Share feedback, report issues, or suggest improvements for the Pandects platform.",
          robots: "index,follow,max-image-preview:large",
          pageType: "WebPage",
          pageName: "Feedback",
          pageDescription:
            "Share feedback, report issues, or suggest improvements for the Pandects platform.",
        };
      case "/donate":
        return {
          title: "Donate | Pandects",
          description:
            "Support Pandects to help maintain and expand open access to M&A agreement research data.",
          robots: "index,follow,max-image-preview:large",
          pageType: "WebPage",
          pageName: "Donate",
          pageDescription:
            "Support Pandects to help maintain and expand open access to M&A agreement research data.",
        };
      case "/privacy-policy":
        return {
          title: "Privacy Policy | Pandects",
          description: "Read Pandects' Privacy Policy.",
          robots: "index,follow,max-image-preview:large",
          pageType: "WebPage",
          pageName: "Privacy Policy",
          pageDescription: "Read Pandects' Privacy Policy.",
        };
      case "/terms":
        return {
          title: "Terms of Service | Pandects",
          description: "Read the Pandects Terms of Service.",
          robots: "index,follow,max-image-preview:large",
          pageType: "WebPage",
          pageName: "Terms of Service",
          pageDescription: "Read the Pandects Terms of Service.",
        };
      case "/license":
        return {
          title: "License | Pandects",
          description: "Pandects open-source software license information.",
          robots: "index,follow,max-image-preview:large",
          pageType: "WebPage",
          pageName: "License",
          pageDescription: "Pandects open-source software license information.",
        };
      case "/account":
        return {
          title: "Account | Pandects",
          description: "Manage your Pandects account, access, and saved settings.",
          robots: "noindex,nofollow",
          pageType: "WebPage",
          pageName: "Account",
          pageDescription: "Manage your Pandects account, access, and saved settings.",
        };
      case "/auth/forgot-password":
        return {
          title: "Reset Password | Pandects",
          description: "Reset your Pandects account password.",
          robots: "noindex,nofollow",
          pageType: "WebPage",
          pageName: "Reset Password",
          pageDescription: "Reset your Pandects account password.",
        };
      case "/auth/reset-password":
        return {
          title: "Reset Password | Pandects",
          description: "Reset your Pandects account password.",
          robots: "noindex,nofollow",
          pageType: "WebPage",
          pageName: "Reset Password",
          pageDescription: "Reset your Pandects account password.",
        };
      case "/auth/google/callback":
        return {
          title: "Signing In | Pandects",
          description: "Completing your Pandects sign-in flow.",
          robots: "noindex,nofollow",
          pageType: "WebPage",
          pageName: "Signing In",
          pageDescription: "Completing your Pandects sign-in flow.",
        };
      case "/auth/verify-email":
        return {
          title: "Verify Email | Pandects",
          description: "Verify your Pandects account email address.",
          robots: "noindex,nofollow",
          pageType: "WebPage",
          pageName: "Verify Email",
          pageDescription: "Verify your Pandects account email address.",
        };
      default:
        return {
          title: "Not Found | Pandects",
          description: "The requested page does not exist.",
          robots: "noindex,nofollow",
          pageType: "WebPage",
          pageName: "Not Found",
          pageDescription: "The requested page does not exist.",
        };
    }
  })();

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
