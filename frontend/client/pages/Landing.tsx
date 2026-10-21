import { Link } from "react-router-dom";
import { ArrowRight, Code, Database } from "lucide-react";
import logo128Webp from "../../assets/logo-128.webp";
import logo256Webp from "../../assets/logo-256.webp";
import logo128Png from "../../assets/logo-128.png";
import logo256Png from "../../assets/logo-256.png";
import { FeaturedAgreements } from "@/components/FeaturedAgreements";
import { LandingSearchForm } from "@/components/landing/LandingSearchForm";
import { LandingStats } from "@/components/landing/LandingStats";
import { DOCS_MCP_URL, GITHUB_URL } from "@/components/navigation-links";
import { AdaptiveTooltip } from "@/components/ui/adaptive-tooltip";
import { Card } from "@/components/ui/card";
import { ExternalLink } from "@/components/ui/external-link";
import { Eyebrow, eyebrowClassName } from "@/components/ui/eyebrow";
import { trackEvent } from "@/lib/analytics";

const API_EXAMPLES_URL = `${GITHUB_URL}/tree/main/examples`;

const BROWSE_LINKS = [
  { to: "/agreement-index", label: "Browse the index" },
  { to: "/taxonomy", label: "Taxonomy" },
  { to: "/bulk-data", label: "Bulk data" },
] as const;

/** Faint grid that fades out toward the edges, behind the landing card. */
const GRID_TEXTURE_STYLE = {
  backgroundImage:
    "linear-gradient(to right, hsl(var(--foreground) / 0.07) 1px, transparent 1px), linear-gradient(to bottom, hsl(var(--foreground) / 0.07) 1px, transparent 1px)",
  backgroundSize: "48px 48px",
  maskImage:
    "radial-gradient(ellipse 70% 60% at 50% 45%, black 40%, transparent 100%)",
  WebkitMaskImage:
    "radial-gradient(ellipse 70% 60% at 50% 45%, black 40%, transparent 100%)",
} as const;

const DEV_LINK =
  "inline-flex min-h-11 items-center gap-2 rounded-md text-sm font-medium text-foreground/80 transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:min-h-0";

export default function Landing() {
  return (
    <div className="relative isolate overflow-hidden px-4 py-8 sm:py-12 lg:py-16">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 -z-10"
        style={GRID_TEXTURE_STYLE}
      />
      <Card className="mx-auto w-full max-w-[960px] border-border bg-background/80 motion-safe:animate-enter px-5 py-10 shadow-overlay backdrop-blur supports-[backdrop-filter]:bg-background/75 sm:px-10 sm:py-14">
        <section
          aria-labelledby="landing-heading"
          className="flex flex-col items-center text-center"
        >
          <AdaptiveTooltip
            trigger={
              <button
                type="button"
                aria-label="About the Pandects name"
                className="mb-6 block rounded-2xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background sm:mb-8"
              >
                <picture>
                  <source
                    srcSet={`${logo128Webp} 128w, ${logo256Webp} 256w`}
                    sizes="(min-width: 640px) 112px, 88px"
                    type="image/webp"
                  />
                  <img
                    src={logo128Png}
                    alt="Pandects logo"
                    width={128}
                    height={128}
                    srcSet={`${logo128Png} 128w, ${logo256Png} 256w`}
                    sizes="(min-width: 640px) 112px, 88px"
                    fetchpriority="high"
                    decoding="async"
                    className="h-[88px] w-[88px] rounded-2xl object-cover shadow-raised ring-1 ring-border sm:h-28 sm:w-28"
                  />
                </picture>
              </button>
            }
            content={
              <>
                What&apos;s behind the name? We took a page from Emperor
                Justinian, whose 6th-century compendium, the Pandects, distilled
                centuries of legal wisdom into a single, authoritative digest.
              </>
            }
            tooltipProps={{
              side: "top",
              className:
                "max-w-xs border-transparent bg-foreground text-background",
            }}
            popoverProps={{
              side: "top",
              className:
                "w-auto max-w-xs border-transparent bg-foreground p-2 text-sm text-background",
            }}
          />
          <h1
            id="landing-heading"
            className="text-balance text-[clamp(2rem,1.2rem+2.6vw,2.75rem)] font-semibold leading-[1.1] tracking-[-0.025em] text-foreground"
          >
            {"Search Thousands of M&A\u00A0Agreements"}
          </h1>
          <p className="mt-3 text-base text-muted-foreground sm:text-lg">
            The fastest way to query deal data, open-source.
          </p>

          <div className="mt-8 w-full max-w-2xl">
            <LandingSearchForm />
          </div>

          <LandingStats className="mt-4 justify-center" />
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
            <ul className="divide-y divide-border overflow-hidden rounded-xl border border-border bg-card shadow-raised">
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
                      className="h-4 w-4 shrink-0 text-muted-foreground transition-[color,transform] duration-150 group-hover:text-foreground motion-safe:group-hover:translate-x-0.5"
                      aria-hidden="true"
                    />
                  </Link>
                </li>
              ))}
            </ul>
          </nav>

          <section
            aria-labelledby="landing-developers-heading"
            className="flex flex-col items-center gap-x-6 gap-y-1 border-t border-border pt-6 sm:flex-row sm:flex-wrap sm:justify-center sm:gap-y-2 md:col-span-2 md:row-start-2"
          >
            <Eyebrow asChild>
              <h2 id="landing-developers-heading">For developers</h2>
            </Eyebrow>
            <ul className="flex flex-col items-center gap-x-6 sm:flex-row sm:flex-wrap sm:gap-y-2">
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
      </Card>
    </div>
  );
}
