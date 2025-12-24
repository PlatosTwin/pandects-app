import {
  DEFAULT_DESCRIPTION,
  DEFAULT_TITLE,
  buildJsonLd,
  getSeoConfigForPath,
} from "@shared/seo-helpers";

type SeoPage = {
  title: string;
  description: string;
  canonical: string;
  robots: string;
  ogImage: string;
  jsonLd: string;
};

export function applySeoForPath(pathname: string): void {
  const origin = window.location.origin;
  const seo = getSeoForPath(pathname, origin);
  applySeo(seo);
}

export function getSeoForPath(pathname: string, origin: string): SeoPage {
  const seo = getSeoConfigForPath(pathname, origin);

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
