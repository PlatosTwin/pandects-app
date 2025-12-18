import logo from "../../assets/logo.png";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { trackEvent } from "@/lib/analytics";

export default function Landing() {
  return (
    <div className="relative isolate min-h-[80vh] overflow-hidden px-4 py-10 sm:py-12">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 -z-10"
      >
        <div className="absolute left-1/2 top-[-14rem] h-[28rem] w-[28rem] -translate-x-1/2 rounded-full bg-primary/10 blur-3xl" />
        <div className="absolute bottom-[-18rem] right-[-10rem] h-[34rem] w-[34rem] rounded-full bg-foreground/5 blur-3xl" />
      </div>

      <div className="mx-auto flex min-h-[80vh] max-w-5xl items-center justify-center">
        <div className="w-full max-w-[860px] animate-fade-in-up rounded-2xl border border-border/70 bg-background/75 p-6 text-center shadow-sm backdrop-blur sm:rounded-3xl sm:p-10">
          <img
            src={logo}
            alt="Pandects Logo"
            width={128}
            height={128}
            decoding="async"
            className="mx-auto h-24 w-24 rounded-2xl object-cover shadow-sm ring-1 ring-border/70 sm:h-32 sm:w-32"
          />

          <h1 className="mt-6 text-4xl font-extrabold leading-tight tracking-tight text-foreground sm:text-5xl">
            Pandects
          </h1>

          <p className="mt-3 text-base font-medium text-muted-foreground sm:text-xl">
            Welcome to Pandects, the open-source M&A research platform.
          </p>

          <div className="mx-auto mt-6 h-1 w-24 rounded-full bg-primary" />

          <div className="mx-auto mt-6 max-w-md text-sm font-normal leading-relaxed text-muted-foreground sm:text-base">
            <p>
              What's up with the name? We took a page from Emperor Justinian,
              whose 6th‑century compendium—The Pandects—distilled centuries of
              legal wisdom into a single, authoritative digest.
            </p>
          </div>

          <div className="mt-8 grid w-full gap-3 sm:grid-cols-2 sm:gap-4">
            <Button asChild className="w-full rounded-full px-8 py-3 text-base">
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

            <Button
              asChild
              variant="outline"
              className="w-full rounded-full border-primary/60 px-8 py-3 text-base text-primary hover:bg-primary/10"
            >
              <a
                href="https://github.com/PlatosTwin/pandects-app/tree/main/examples"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="See Examples (opens in a new tab)"
                onClick={() =>
                  trackEvent("landing_cta_click", {
                    cta: "See Examples",
                    href:
                      "https://github.com/PlatosTwin/pandects-app/tree/main/examples",
                  })
                }
              >
                See Examples
              </a>
            </Button>

            <Button
              asChild
              variant="outline"
              className="w-full rounded-full border-primary/60 px-8 py-3 text-base text-primary hover:bg-primary/10 sm:col-span-2"
            >
              <Link
                to="/about#data"
                onClick={() =>
                  trackEvent("landing_cta_click", {
                    cta: "Learn About the Data",
                    to_path: "/about#data",
                  })
                }
              >
                Learn About the Data
              </Link>
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
