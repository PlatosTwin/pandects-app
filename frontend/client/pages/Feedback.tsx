import { PageShell } from "@/components/PageShell";
import { trackEvent } from "@/lib/analytics";
import { useEffect, useRef, useState } from "react";
import { Bug, ExternalLink, GitPullRequest, MessageSquare } from "lucide-react";

const generalFeedbackUrl =
  "https://airtable.com/embed/appsaasOdbK3k0JIR/pagX6sJC7D7wihUto/form";

function ensureHeadLink(rel: string, href: string) {
  if (typeof document === "undefined") return;
  const selector = `link[rel="${rel}"][href="${href}"]`;
  if (document.head.querySelector(selector)) return;

  const link = document.createElement("link");
  link.rel = rel;
  link.href = href;
  document.head.appendChild(link);
}

export default function Feedback() {
  const [loaded, setLoaded] = useState(false);
  const [loadTimeout, setLoadTimeout] = useState(false);
  const trackedRef = useRef(false);

  useEffect(() => {
    if (typeof document === "undefined") return;

    if (!trackedRef.current) {
      trackedRef.current = true;
      trackEvent("feedback_form_view", { section: "general_feedback" });
    }

    ensureHeadLink("preconnect", "https://airtable.com");
    ensureHeadLink("preconnect", "https://static.airtable.com");
    ensureHeadLink("dns-prefetch", "https://airtable.com");
    ensureHeadLink("dns-prefetch", "https://static.airtable.com");
  }, []);

  useEffect(() => {
    if (loaded) return;
    const timer = window.setTimeout(() => setLoadTimeout(true), 10000);
    return () => window.clearTimeout(timer);
  }, [loaded]);

  return (
    <PageShell size="lg" title="Feedback">
      <article>
        <section
          id="channels"
          className="scroll-mt-24 first:pt-0"
          aria-labelledby="channels-heading"
        >
          <div className="flex items-center gap-3">
            <span className="inline-flex h-7 w-7 items-center justify-center rounded-md bg-primary/10 text-primary text-xs font-medium tabular-nums">
              01
            </span>
            <h2
              id="channels-heading"
              className="text-2xl font-semibold tracking-tight text-foreground"
            >
              Channels
            </h2>
          </div>

          <p className="prose-copy mt-4">
            Questions, bug reports, and feature requests are welcome. Choose the
            path that matches the shape of the feedback.
          </p>

          <div className="mt-6 grid gap-3 md:grid-cols-3">
            <a
              href="#form"
              className="rounded-lg border border-border bg-card p-4 transition-colors hover:border-primary/40 hover:bg-accent/40"
            >
              <MessageSquare className="mb-2 h-5 w-5 text-primary" />
              <div className="text-xs uppercase tracking-[0.12em] text-muted-foreground">
                Feedback
              </div>
              <div className="mt-1 text-sm font-medium leading-5 text-foreground">
                Use the form
              </div>
              <p className="mt-2 text-sm text-muted-foreground">
                Best for quick notes, questions, and product suggestions.
              </p>
            </a>
            <a
              href="https://github.com/PlatosTwin/pandects-app/issues"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="Open an issue (opens in a new tab)"
              className="rounded-lg border border-border bg-card p-4 transition-colors hover:border-primary/40 hover:bg-accent/40"
            >
              <Bug className="mb-2 h-5 w-5 text-primary" />
              <div className="text-xs uppercase tracking-[0.12em] text-muted-foreground">
                GitHub
              </div>
              <div className="mt-1 text-sm font-medium leading-5 text-foreground">
                Open an issue
                <ExternalLink
                  className="ml-0.5 inline-block h-3 w-3 align-baseline opacity-60"
                  aria-hidden="true"
                />
              </div>
              <p className="mt-2 text-sm text-muted-foreground">
                Best for reproducible bugs and scoped feature requests.
              </p>
            </a>
            <a
              href="https://github.com/PlatosTwin/pandects-app/compare"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="Open a pull request on GitHub (opens in a new tab)"
              className="rounded-lg border border-border bg-card p-4 transition-colors hover:border-primary/40 hover:bg-accent/40"
            >
              <GitPullRequest className="mb-2 h-5 w-5 text-primary" />
              <div className="text-xs uppercase tracking-[0.12em] text-muted-foreground">
                Contributions
              </div>
              <div className="mt-1 text-sm font-medium leading-5 text-foreground">
                Open a PR
                <ExternalLink
                  className="ml-0.5 inline-block h-3 w-3 align-baseline opacity-60"
                  aria-hidden="true"
                />
              </div>
              <p className="mt-2 text-sm text-muted-foreground">
                Best when the fix or improvement is already in code.
              </p>
            </a>
          </div>
        </section>

        <section
          id="form"
          className="scroll-mt-24 pt-12"
          aria-labelledby="form-heading"
        >
          <div className="flex items-center gap-3">
            <span className="inline-flex h-7 w-7 items-center justify-center rounded-md bg-primary/10 text-primary text-xs font-medium tabular-nums">
              02
            </span>
            <div className="flex flex-1 flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <h2
                id="form-heading"
                className="text-2xl font-semibold tracking-tight text-foreground"
              >
                Feedback form
              </h2>
              <a
                href={generalFeedbackUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex w-fit items-center gap-1 text-sm font-medium text-muted-foreground transition-colors hover:text-primary"
              >
                Open in new tab
                <ExternalLink className="h-3 w-3" aria-hidden="true" />
              </a>
            </div>
          </div>
          <div
            className="mt-6 relative"
            aria-busy={!loaded}
            role="status"
            aria-live="polite"
          >
            <span className="sr-only">Loading feedback form</span>
            {!loaded && (
              <div
                className="absolute inset-0 h-[895px] rounded-lg border border-border bg-muted/30 p-6"
                aria-hidden="true"
              >
                <div className="h-5 w-40 bg-muted/60 rounded animate-pulse" />
                <div className="h-10 w-full bg-muted/50 rounded animate-pulse mt-4" />
                <div className="h-10 w-3/4 bg-muted/50 rounded animate-pulse mt-4" />
                <div className="h-10 w-full bg-muted/50 rounded animate-pulse mt-4" />
                <div className="h-24 w-full bg-muted/50 rounded animate-pulse mt-4" />
                <div className="h-10 w-32 bg-muted/60 rounded-md animate-pulse mt-6" />
              </div>
            )}
            <iframe
              loading="eager"
              className="airtable-embed w-full rounded-lg border border-border"
              style={{ background: "transparent", opacity: loaded ? 1 : 0 }}
              src={generalFeedbackUrl}
              title="Pandects general feedback form"
              width="100%"
              height="895"
              onLoad={() => setLoaded(true)}
            />
            {loadTimeout && !loaded && (
              <div className="mt-4 rounded-lg border border-border bg-background/60 p-4 text-sm text-muted-foreground">
                The form is taking longer than expected to load.{" "}
                <a
                  href={generalFeedbackUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary underline underline-offset-2 hover:underline"
                >
                  Open the form in a new tab
                </a>
                <ExternalLink
                  className="ml-0.5 inline-block h-3 w-3 opacity-60"
                  aria-hidden="true"
                />
                .
              </div>
            )}
          </div>
        </section>
      </article>
    </PageShell>
  );
}
