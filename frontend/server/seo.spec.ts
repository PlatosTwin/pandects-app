import { describe, expect, it } from "vitest";
import { buildSeoPage, injectSeoDocument } from "../shared/seo-helpers.mjs";

describe("SEO routing policy", () => {
  const origin = "https://pandects.org";

  it("keeps the base search page indexable", () => {
    const seo = buildSeoPage("/search", "", origin);
    expect(seo.robots).toBe("index,follow,max-image-preview:large");
    expect(seo.canonical).toBe("https://pandects.org/search");
  });

  it("marks filtered search states as noindex,follow", () => {
    const seo = buildSeoPage("/search", "?q=termination+fee&page=2", origin);
    expect(seo.robots).toBe("noindex,follow");
    expect(seo.canonical).toBe("https://pandects.org/search");
  });

  it("keeps account routes noindexed", () => {
    const seo = buildSeoPage("/account", "", origin);
    expect(seo.robots).toBe("noindex,nofollow");
  });

  it("keeps auth flow routes noindexed", () => {
    const seo = buildSeoPage("/auth/reset-password", "", origin);
    expect(seo.robots).toBe("noindex,nofollow");
  });

  it("returns a noindex 404 policy for unknown routes", () => {
    const seo = buildSeoPage("/does-not-exist", "", origin);
    expect(seo.status).toBe(404);
    expect(seo.robots).toBe("noindex,nofollow");
    expect(seo.xRobotsTag).toBe("noindex, nofollow");
  });
});

describe("SEO document injection", () => {
  it("replaces the title and SEO block in the source html", () => {
    const seo = buildSeoPage("/about", "", "https://pandects.org");
    const html = injectSeoDocument(
      "<html><head><title>Old Title</title><!-- SEO:BEGIN --><!-- SEO:END --></head><body><div id=\"root\"></div></body></html>",
      seo,
    );

    expect(html).toContain("<title>About | Pandects</title>");
    expect(html).toContain('rel="canonical" href="https://pandects.org/about"');
    expect(html).toContain('name="description" content="Learn what Pandects is, why it exists, and how it\'s built as an open-source M&amp;A research platform."');
  });
});
