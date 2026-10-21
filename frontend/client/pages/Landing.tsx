import { Link } from "react-router-dom";
import { ArrowRight, Code, Database } from "lucide-react";
import { PageShell } from "@/components/PageShell";
import { FeaturedAgreements } from "@/components/FeaturedAgreements";
import { LandingSearchForm } from "@/components/landing/LandingSearchForm";
import { LandingStats } from "@/components/landing/LandingStats";
import { DOCS_MCP_URL, GITHUB_URL } from "@/components/navigation-links";
import { ExternalLink } from "@/components/ui/external-link";
import { Eyebrow, eyebrowClassName } from "@/components/ui/eyebrow";
import { trackEvent } from "@/lib/analytics";

const API_EXAMPLES_URL = `${GITHUB_URL}/tree/main/examples`;

const BROWSE_LINKS = [
  { to: "/agreement-index", label: "Browse the index" },
  { to: "/taxonomy", label: "Taxonomy" },
  { to: "/bulk-data", label: "Bulk data" },
] as const;

const DEV_LINK =
  "inline-flex min-h-11 items-center gap-2 rounded-md text-sm font-medium text-foreground/80 transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:min-h-0";

export default function Landing() {
  return (
    <PageShell
      layout="data"
      className="sm:py-12 lg:py-16"
      contentClassName="max-w-4xl"
    >
      <section aria-labelledby="landing-heading">
        <h1
          id="landing-heading"
          className="text-[clamp(2rem,1.1rem+3vw,3rem)] font-semibold leading-[1.1] tracking-[-0.025em] text-foreground"
        >
          {"Search Thousands of M&A Agreements"}
        </h1>
        <p className="mt-3 text-base text-muted-foreground sm:text-lg">
          The fastest way to query deal data, open-source.
        </p>

        <div className="mt-6 sm:mt-8">
          <LandingSearchForm />
        </div>

        <LandingStats className="mt-4" />
      </section>

      {/*
        DOM (and mobile) order is browse, developers, latest agreements, so
        when the agreements list drops out (API locked or down) nothing above
        the footer moves. At md+ the grid puts the list beside the browse
        links; both columns have the same height, so its absence moves nothing
        there either.
      */}
      <div className="mt-10 grid grid-cols-1 gap-8 md:mt-12 md:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)] md:gap-x-6 md:gap-y-10">
        <nav
          aria-labelledby="landing-browse-heading"
          className="min-w-0 md:col-start-1 md:row-start-1"
        >
          <div className="mb-2 flex h-6 items-center">
            <h2 id="landing-browse-heading" className={eyebrowClassName}>
              Browse
            </h2>
          </div>
          <ul className="divide-y divide-border overflow-hidden rounded-xl border border-border bg-card">
            {BROWSE_LINKS.map((link) => (
              <li key={link.to}>
                <Link
                  to={link.to}
                  onClick={() =>
                    trackEvent("landing_cta_click", {
                      cta: link.label,
                      to_path: link.to,
                    })
                  }
                  className="group flex min-h-[3.75rem] items-center justify-between gap-3 px-4 py-2.5 text-sm font-medium text-foreground transition-colors hover:bg-accent/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring"
                >
                  {link.label}
                  <ArrowRight
                    className="h-4 w-4 shrink-0 text-muted-foreground transition-colors group-hover:text-foreground"
                    aria-hidden="true"
                  />
                </Link>
              </li>
            ))}
          </ul>
        </nav>

        <section
          aria-labelledby="landing-developers-heading"
          className="flex flex-col gap-x-6 gap-y-1 border-t border-border pt-6 sm:flex-row sm:flex-wrap sm:items-center sm:gap-y-2 md:col-span-2 md:row-start-2"
        >
          <Eyebrow asChild>
            <h2 id="landing-developers-heading">For developers</h2>
          </Eyebrow>
          <ul className="flex flex-col gap-x-6 sm:flex-row sm:flex-wrap sm:items-center sm:gap-y-2">
            <li>
              <ExternalLink
                href={DOCS_MCP_URL}
                className={DEV_LINK}
                onClick={() =>
                  trackEvent("landing_cta_click", {
                    cta: "Connect to MCP",
                    href: DOCS_MCP_URL,
                  })
                }
              >
                <img
                  src="/mcp-logo.svg"
                  alt=""
                  aria-hidden="true"
                  width={16}
                  height={16}
                  className="h-4 w-4 self-center dark:invert"
                />
                Connect to MCP
              </ExternalLink>
            </li>
            <li>
              <ExternalLink
                href={API_EXAMPLES_URL}
                className={DEV_LINK}
                onClick={() =>
                  trackEvent("landing_cta_click", {
                    cta: "See API Examples",
                    href: API_EXAMPLES_URL,
                  })
                }
              >
                <Code className="h-4 w-4 self-center" aria-hidden="true" />
                See API Examples
              </ExternalLink>
            </li>
            <li>
              <Link
                to="/sources-methods"
                className={DEV_LINK}
                onClick={() =>
                  trackEvent("landing_cta_click", {
                    cta: "Sources & Methods",
                    to_path: "/sources-methods",
                  })
                }
              >
                <Database className="h-4 w-4" aria-hidden="true" />
                Sources &amp; Methods
              </Link>
            </li>
          </ul>
        </section>

        <FeaturedAgreements className="md:col-start-2 md:row-start-1" />
      </div>
    </PageShell>
  );
}
