import logo128 from "../../assets/logo-128.png";
import logo256 from "../../assets/logo-256.png";
import sponsorLogoOne from "../../assets/sponsors/pandects-placeholder-1.png";
import sponsorLogoTwo from "../../assets/sponsors/pandects-placeholder-2.png";
import sponsorLogoThree from "../../assets/sponsors/pandects-placeholder-3.png";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Link } from "react-router-dom";
import { trackEvent } from "@/lib/analytics";
import { Code, Database } from "lucide-react";
import { Card } from "@/components/ui/card";

export default function Landing() {
  const showSponsors = false;
  const sponsorLogos = [sponsorLogoOne, sponsorLogoTwo, sponsorLogoThree];

  return (
    <div className="relative isolate min-h-[80vh] overflow-hidden px-4 py-8 sm:py-10">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 -z-10"
      >
        <div className="absolute left-1/2 top-[-14rem] h-[28rem] w-[28rem] -translate-x-1/2 rounded-full bg-primary/10 blur-3xl" />
        <div className="absolute bottom-[-18rem] right-[-10rem] h-[34rem] w-[34rem] rounded-full bg-foreground/5 blur-3xl" />
      </div>

      <div className="mx-auto flex min-h-[80vh] max-w-5xl items-center justify-center">
        <div className="flex w-full max-w-[860px] flex-col items-center">
          <Card className="w-full animate-fade-in-up border-border/70 bg-background/75 px-6 py-12 text-center backdrop-blur sm:px-10 sm:py-16">
            <div className="mx-auto mb-6 inline-flex items-center rounded-full bg-muted px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.2em] text-foreground/70">
              Sourced from EDGAR • Updated{"\u00A0"}weekly
            </div>

            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  aria-label="About the Pandects name"
                  className="mx-auto mb-8 block rounded-2xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                >
                  <img
                    src={logo128}
                    alt="Pandects Logo"
                    width={128}
                    height={128}
                    srcSet={`${logo128} 128w, ${logo256} 256w`}
                    sizes="(min-width: 640px) 128px, 96px"
                    fetchPriority="high"
                    decoding="async"
                    className="h-24 w-24 rounded-2xl object-cover shadow-sm ring-1 ring-border/70 sm:h-32 sm:w-32"
                  />
                </button>
              </TooltipTrigger>
              <TooltipContent
                side="top"
                className="max-w-xs border-transparent bg-foreground text-background"
              >
                What's behind the name? We took a page from Emperor Justinian,
                whose 6th-century compendium, the Pandects, distilled centuries
                of legal wisdom into a single, authoritative digest.
              </TooltipContent>
            </Tooltip>

            <h1 className="mt-0 text-3xl font-semibold leading-tight tracking-tight text-foreground sm:text-4xl">
            {"Search Thousands of M&A\u00A0Agreements"}
            </h1>

            <p className="mt-4 text-base font-medium text-foreground/70 sm:text-xl">
            The fastest way for academics to query deal data.
            </p>

            <div className="mt-12 flex w-full flex-col items-center gap-3">
              <Button
                asChild
                className="w-full max-w-sm rounded-full px-8 py-3 text-base sm:w-auto"
              >
                <Link
                  to="/search"
                  onClick={() =>
                    trackEvent("landing_cta_click", {
                      cta: "Explore Agreements",
                      to_path: "/search",
                    })
                  }
                >
                  Explore Agreements
                </Link>
              </Button>
            </div>

            <div className="mt-6 flex flex-col items-center justify-center gap-1 text-sm sm:mt-4 sm:flex-row sm:gap-4">
              <Button
                asChild
                variant="link"
                className="h-auto gap-2 p-0 text-sm text-muted-foreground hover:text-foreground"
              >
                <a
                  href="https://github.com/PlatosTwin/pandects-app/tree/main/examples"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="See API Examples (opens in a new tab)"
                  onClick={() =>
                    trackEvent("landing_cta_click", {
                      cta: "See API Examples",
                      href:
                        "https://github.com/PlatosTwin/pandects-app/tree/main/examples",
                    })
                  }
                >
                  <Code className="h-4 w-4" aria-hidden="true" />
                  See API Examples
                </a>
              </Button>
              <span
                aria-hidden="true"
                className="hidden text-muted-foreground/50 sm:inline"
              >
                •
              </span>
              <Button
                asChild
                variant="link"
                className="h-auto gap-2 p-0 text-sm text-muted-foreground hover:text-foreground"
              >
                <Link
                  to="/sources-methods"
                  onClick={() =>
                    trackEvent("landing_cta_click", {
                      cta: "About the Dataset",
                      to_path: "/sources-methods",
                    })
                  }
                >
                  <Database className="h-4 w-4" aria-hidden="true" />
                  About the Dataset
                </Link>
              </Button>
            </div>
          </Card>
          {showSponsors ? (
            <section className="mt-20 text-center">
              <p className="text-xs font-semibold uppercase tracking-[0.2em] text-muted-foreground">
                Sponsors &amp; Supporters
              </p>
              <div className="mt-4 flex flex-wrap items-center justify-center gap-4">
                {sponsorLogos.map((sponsorLogo, index) => (
                  <img
                    key={`${sponsorLogo}-${index}`}
                    src={sponsorLogo}
                    alt="Pandects placeholder sponsor logo"
                    loading="lazy"
                    decoding="async"
                    className="h-10 w-10 rounded-lg object-cover opacity-50 grayscale"
                  />
                ))}
              </div>
            </section>
          ) : null}
        </div>
      </div>
    </div>
  );
}
