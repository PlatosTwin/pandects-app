import logo128Webp from "../../assets/logo-128.webp";
import logo256Webp from "../../assets/logo-256.webp";
import logo128Png from "../../assets/logo-128.png";
import logo256Png from "../../assets/logo-256.png";
import { Button } from "@/components/ui/button";
import { AdaptiveTooltip } from "@/components/ui/adaptive-tooltip";
import { Link } from "react-router-dom";
import { trackEvent } from "@/lib/analytics";
import { Code, Database } from "lucide-react";
import { Card } from "@/components/ui/card";
import { FeaturedAgreements } from "@/components/FeaturedAgreements";
import { useAgreementSummary } from "@/hooks/use-agreement-summary";
import { formatNumber } from "@/lib/format-utils";
import brandLinks from "@branding/links.json";
import { useEffect, useMemo, useRef, useState } from "react";

const LATEST_FILING_FORMATTER = new Intl.DateTimeFormat("en-US", {
  year: "numeric",
  month: "short",
});

function formatLatestFiling(value: string | null): string | null {
  if (!value) return null;
  const dt = new Date(value);
  if (Number.isNaN(dt.getTime())) return null;
  return LATEST_FILING_FORMATTER.format(dt);
}

export default function Landing() {
  const docsUrl = import.meta.env.DEV ? "http://localhost:3001" : brandLinks.docsSiteUrl;
  const { summary } = useAgreementSummary();
  const liveStats = useMemo(() => {
    if (!summary) return null;
    const latestFiling = formatLatestFiling(summary.latest_filing_date);
    return {
      agreements: formatNumber(summary.agreements),
      latestFiling,
    };
  }, [summary]);
  const headerBadgeRef = useRef<HTMLDivElement | null>(null);
  const sourcedTextRef = useRef<HTMLSpanElement | null>(null);
  const updatedTextRef = useRef<HTMLSpanElement | null>(null);
  const separatorRef = useRef<HTMLSpanElement | null>(null);
  const [showHeaderSeparator, setShowHeaderSeparator] = useState(true);

  useEffect(() => {
    const badge = headerBadgeRef.current;
    const sourced = sourcedTextRef.current;
    const updated = updatedTextRef.current;
    const separator = separatorRef.current;
    if (!badge || !sourced || !updated || !separator) return;

    const measureWrap = () => {
      const top = sourced.offsetTop;
      const nextShowHeaderSeparator =
        separator.offsetTop === top && updated.offsetTop === top;

      setShowHeaderSeparator((previous) =>
        previous === nextShowHeaderSeparator ? previous : nextShowHeaderSeparator,
      );
    };

    measureWrap();

    const resizeObserver = new ResizeObserver(() => {
      measureWrap();
    });
    resizeObserver.observe(badge);

    window.addEventListener("resize", measureWrap);
    return () => {
      resizeObserver.disconnect();
      window.removeEventListener("resize", measureWrap);
    };
  }, []);

  return (
    <div className="relative isolate min-h-[80vh] overflow-hidden px-4 py-8 sm:py-10">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 -z-10"
        style={{
          backgroundImage:
            "linear-gradient(to right, hsl(var(--foreground) / 0.07) 1px, transparent 1px), linear-gradient(to bottom, hsl(var(--foreground) / 0.07) 1px, transparent 1px)",
          backgroundSize: "48px 48px",
          maskImage:
            "radial-gradient(ellipse 70% 60% at 50% 45%, black 40%, transparent 100%)",
          WebkitMaskImage:
            "radial-gradient(ellipse 70% 60% at 50% 45%, black 40%, transparent 100%)",
        }}
      />

      <div className="mx-auto flex min-h-[80vh] max-w-5xl items-center justify-center">
        <div className="flex w-full max-w-[860px] flex-col items-center">
          <Card className="w-full animate-fade-in-up border-border bg-background/75 px-6 py-12 text-center backdrop-blur shadow-lg supports-[backdrop-filter]:bg-background/75 sm:px-10 sm:py-16">
            <div
              ref={headerBadgeRef}
              className="mx-auto mb-6 inline-flex flex-wrap items-center justify-center gap-x-1 rounded-full bg-muted px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.2em] text-foreground/70"
            >
              <span ref={sourcedTextRef}>Sourced from EDGAR</span>
              <span
                ref={separatorRef}
                aria-hidden="true"
                className={showHeaderSeparator ? "" : "invisible"}
              >
                •
              </span>
              <span ref={updatedTextRef}>Updated{"\u00A0"}monthly</span>
            </div>

            <AdaptiveTooltip
              trigger={
                <button
                  type="button"
                  aria-label="About the Pandects name"
                  className="mx-auto mb-8 block rounded-2xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
                >
                  <picture>
                    <source
                      srcSet={`${logo128Webp} 128w, ${logo256Webp} 256w`}
                      sizes="(min-width: 640px) 128px, 96px"
                      type="image/webp"
                    />
                    <img
                      src={logo128Png}
                      alt="Pandects Logo"
                      width={128}
                      height={128}
                      srcSet={`${logo128Png} 128w, ${logo256Png} 256w`}
                      sizes="(min-width: 640px) 128px, 96px"
                      fetchpriority="high"
                      decoding="async"
                      className="h-24 w-24 rounded-2xl object-cover shadow-sm ring-1 ring-border sm:h-32 sm:w-32"
                    />
                  </picture>
                </button>
              }
              content={
                <>
                  What&apos;s behind the name? We took a page from Emperor Justinian,
                  whose 6th-century compendium, the Pandects, distilled centuries
                  of legal wisdom into a single, authoritative digest.
                </>
              }
              tooltipProps={{
                side: "top",
                className: "max-w-xs border-transparent bg-foreground text-background",
              }}
              popoverProps={{
                side: "top",
                className: "w-auto max-w-xs border-transparent bg-foreground p-2 text-sm text-background",
              }}
            />

            <h1 className="mt-0 text-4xl font-semibold leading-[1.05] tracking-[-0.025em] text-foreground sm:text-5xl">
              {"Search Thousands of M&A\u00A0Agreements"}
            </h1>

            <p className="mt-5 text-base font-normal text-foreground/65 sm:text-lg">
              The fastest way to query deal data, open-source.
            </p>

            {liveStats ? (
              <p
                className="mt-2 text-sm text-muted-foreground"
                aria-live="polite"
              >
                <span className="font-medium text-foreground/80">
                  {liveStats.agreements}
                </span>{" "}
                agreements indexed
                {liveStats.latestFiling ? (
                  <>
                    <span aria-hidden="true" className="mx-2 text-foreground/40">
                      ·
                    </span>
                    Latest filing {liveStats.latestFiling}
                  </>
                ) : null}
              </p>
            ) : null}

            <div className="mt-12 flex w-full flex-col items-center gap-3">
              <Button
                asChild
                className="w-full max-w-sm rounded-full px-8 py-3 text-base sm:w-auto"
              >
                <Link
                  to="/agreement-index"
                  onClick={() =>
                    trackEvent("landing_cta_click", {
                      cta: "Explore Agreements",
                      to_path: "/agreement-index",
                    })
                  }
                >
                  Explore Agreements
                </Link>
              </Button>
            </div>

            <FeaturedAgreements className="mt-10" />

            <div className="mt-8 flex flex-col items-center gap-3 sm:mt-6">
              <div className="flex flex-wrap items-center justify-center gap-3 text-sm sm:gap-4">
                <Button
                  asChild
                  variant="link"
                  className="min-h-11 gap-2 px-2 py-2 text-sm text-muted-foreground hover:text-foreground sm:min-h-0 sm:p-0"
                >
                  <a
                    href={`${docsUrl}/docs/mcp/using`}
                    target="_blank"
                    rel="noopener noreferrer"
                    aria-label="Connect to MCP (opens in a new tab)"
                    onClick={() =>
                      trackEvent("landing_cta_click", {
                        cta: "Connect to MCP",
                        href: `${docsUrl}/docs/mcp/using`,
                      })
                    }
                  >
                    <img
                      src="/mcp-logo.svg"
                      alt=""
                      aria-hidden="true"
                      className="h-4 w-4"
                    />
                    Connect to MCP
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
                  className="min-h-11 gap-2 px-2 py-2 text-sm text-muted-foreground hover:text-foreground sm:min-h-0 sm:p-0"
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
              </div>
              <Button
                asChild
                variant="link"
                className="min-h-11 gap-2 px-2 py-2 text-sm text-muted-foreground hover:text-foreground sm:min-h-0 sm:p-0"
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
        </div>
      </div>
    </div>
  );
}
