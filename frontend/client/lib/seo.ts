type SeoPage = {
  title: string;
  description: string;
  canonical: string;
  robots: string;
  ogImage: string;
  jsonLd: string;
};

type JsonLdParams = {
  origin: string;
  canonical: string;
  pageType: "WebSite" | "WebPage" | "AboutPage";
  pageName: string;
  pageDescription: string;
};

const DEFAULT_TITLE = "Pandects";
const DEFAULT_DESCRIPTION =
  "Search and download structured M&A agreements from SEC EDGAR. Tag clauses, extract terms, and export CSVs.";

export function applySeoForPath(pathname: string): void {
  const origin = window.location.origin;
  const seo = getSeoForPath(pathname, origin);
  applySeo(seo);
}

export function getSeoForPath(pathname: string, origin: string): SeoPage {
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
          pageType: "WebSite" as const,
          pageName: "Pandects",
          pageDescription: DEFAULT_DESCRIPTION,
        };
      case "/search":
        return {
          title: "Search Merger Agreements | Pandects",
          description:
            "Search and filter merger agreement clauses across deals, years, and parties in the Pandects database.",
          robots: "index,follow,max-image-preview:large",
          pageType: "WebPage" as const,
          pageName: "Search Merger Agreements",
          pageDescription:
            "Search and filter merger agreement clauses across deals, years, and parties in the Pandects database.",
        };
      case "/docs":
        return {
          title: "Docs | Pandects",
          description:
            "Documentation for Pandects: data sources, coverage, methodology, and how to use the platform.",
          robots: "index,follow,max-image-preview:large",
          pageType: "WebPage" as const,
          pageName: "Docs",
          pageDescription:
            "Documentation for Pandects: data sources, coverage, methodology, and how to use the platform.",
        };
      case "/bulk-data":
        return {
          title: "Bulk Data | Pandects",
          description:
            "Download bulk datasets and exports from Pandects for research and analysis.",
          robots: "index,follow,max-image-preview:large",
          pageType: "WebPage" as const,
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
          pageType: "WebPage" as const,
          pageName: "Agreement Index",
          pageDescription:
            "Browse all merger agreements in Pandects with sortable metadata and high-level dataset statistics.",
        };
      case "/about":
        return {
          title: "About | Pandects",
          description:
            "Learn what Pandects is, why it exists, and how it's built as an open-source M&A research platform.",
          robots: "index,follow,max-image-preview:large",
          pageType: "AboutPage" as const,
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
          pageType: "WebPage" as const,
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
          pageType: "WebPage" as const,
          pageName: "Donate",
          pageDescription:
            "Support Pandects to help maintain and expand open access to M&A agreement research data.",
        };
      case "/privacy-policy":
        return {
          title: "Privacy Policy | Pandects",
          description: "Read Pandects' Privacy Policy.",
          robots: "index,follow,max-image-preview:large",
          pageType: "WebPage" as const,
          pageName: "Privacy Policy",
          pageDescription: "Read Pandects' Privacy Policy.",
        };
      case "/terms":
        return {
          title: "Terms of Service | Pandects",
          description: "Read the Pandects Terms of Service.",
          robots: "index,follow,max-image-preview:large",
          pageType: "WebPage" as const,
          pageName: "Terms of Service",
          pageDescription: "Read the Pandects Terms of Service.",
        };
      case "/license":
        return {
          title: "License | Pandects",
          description: "Pandects open-source software license information.",
          robots: "index,follow,max-image-preview:large",
          pageType: "WebPage" as const,
          pageName: "License",
          pageDescription: "Pandects open-source software license information.",
        };
      case "/account":
        return {
          title: "Account | Pandects",
          description: "Manage your Pandects account, access, and saved settings.",
          robots: "noindex,nofollow",
          pageType: "WebPage" as const,
          pageName: "Account",
          pageDescription: "Manage your Pandects account, access, and saved settings.",
        };
      case "/auth/forgot-password":
        return {
          title: "Reset Password | Pandects",
          description: "Reset your Pandects account password.",
          robots: "noindex,nofollow",
          pageType: "WebPage" as const,
          pageName: "Reset Password",
          pageDescription: "Reset your Pandects account password.",
        };
      case "/auth/reset-password":
        return {
          title: "Reset Password | Pandects",
          description: "Reset your Pandects account password.",
          robots: "noindex,nofollow",
          pageType: "WebPage" as const,
          pageName: "Reset Password",
          pageDescription: "Reset your Pandects account password.",
        };
      case "/auth/google/callback":
        return {
          title: "Signing In | Pandects",
          description: "Completing your Pandects sign-in flow.",
          robots: "noindex,nofollow",
          pageType: "WebPage" as const,
          pageName: "Signing In",
          pageDescription: "Completing your Pandects sign-in flow.",
        };
      default:
        return {
          title: "Not Found | Pandects",
          description: "The requested page does not exist.",
          robots: "noindex,nofollow",
          pageType: "WebPage" as const,
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
    jsonLd: JSON.stringify(
      buildJsonLd({
        origin,
        canonical,
        pageType: page.pageType,
        pageName: page.pageName,
        pageDescription: page.pageDescription,
      }),
    ),
  };
}

export function applySeo(seo: SeoPage): void {
  document.title = seo.title || DEFAULT_TITLE;

  setMetaByName("description", seo.description);
  setMetaByName("robots", seo.robots);
  setMetaByName("twitter:card", "summary_large_image");
  setMetaByName("twitter:title", seo.title);
  setMetaByName("twitter:description", seo.description);
  setMetaByName("twitter:image", seo.ogImage);
  setMetaByName("twitter:image:alt", "Pandects");

  setMetaByProperty("og:type", "website");
  setMetaByProperty("og:locale", "en_US");
  setMetaByProperty("og:site_name", "Pandects");
  setMetaByProperty("og:title", seo.title);
  setMetaByProperty("og:description", seo.description);
  setMetaByProperty("og:url", seo.canonical);
  setMetaByProperty("og:image", seo.ogImage);
  setMetaByProperty("og:image:alt", "Pandects");

  const canonicalLink = document.querySelector<HTMLLinkElement>('link[rel="canonical"]');
  if (canonicalLink) {
    canonicalLink.setAttribute("href", seo.canonical);
  }

  const jsonLdScript = document.querySelector<HTMLScriptElement>(
    'script[type="application/ld+json"]',
  );
  if (jsonLdScript) {
    jsonLdScript.textContent = seo.jsonLd;
  }
}

function normalizePathname(pathname: string): string {
  const stripped = pathname.split("?")[0]?.split("#")[0] ?? "/";
  if (stripped.length > 1) return stripped.replace(/\/+$/, "");
  return "/";
}

function setMetaByName(name: string, content: string): void {
  const el = document.querySelector<HTMLMetaElement>(`meta[name="${name}"]`);
  if (el) {
    el.setAttribute("content", content);
  }
}

function setMetaByProperty(property: string, content: string): void {
  const el = document.querySelector<HTMLMetaElement>(`meta[property="${property}"]`);
  if (el) {
    el.setAttribute("content", content);
  }
}

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
