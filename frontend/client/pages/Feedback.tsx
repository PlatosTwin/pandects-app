import { PageShell } from "@/components/PageShell";
import { trackEvent } from "@/lib/analytics";
import { useEffect, useRef, useState } from "react";
import { MessageSquare, Bug, GitPullRequest, ExternalLink } from "lucide-react";

const generalFeedbackUrl =
  "https://airtable.com/embed/appsaasOdbK3k0JIR/pagX6sJC7D7wihUto/form";

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

    if (document.querySelector("script[data-airtable-embed]")) return;

    const loadScript = () => {
      const script = document.createElement("script");
      script.async = true;
      script.src = "https://static.airtable.com/js/embed/embed_snippet_v1.js";
      script.dataset.airtableEmbed = "true";
      document.head.appendChild(script);
    };

    if (window.requestIdleCallback) {
      window.requestIdleCallback(loadScript);
    } else {
      window.setTimeout(loadScript, 500);
    }
  }, []);

  useEffect(() => {
    if (loaded) return;
    const timer = window.setTimeout(() => setLoadTimeout(true), 10000);
    return () => window.clearTimeout(timer);
  }, [loaded]);

  return (
    <PageShell size="xl" title="Feedback">
      <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
        PROJECT · FEEDBACK
      </div>
      <p className="mt-3 text-lg text-muted-foreground max-w-[68ch]">
        Questions, bug reports, and feature requests — all in one place.
      </p>
      <div className="mt-6 h-px w-12 bg-primary/70" />

      <div className="mt-10 grid grid-cols-1 md:grid-cols-3 gap-4">
        <a
          href="#form"
          className="rounded-lg border border-border bg-card p-5 transition-colors hover:border-primary/40 hover:bg-accent/40 block"
        >
          <MessageSquare className="h-5 w-5 text-primary mb-3" />
          <h3 className="text-base font-medium text-foreground">
            Quick feedback
          </h3>
          <p className="text-sm text-muted-foreground mt-1">
            Use the form below.
          </p>
        </a>
        <a
          href="https://github.com/PlatosTwin/pandects-app/issues"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Open an issue (opens in a new tab)"
          className="rounded-lg border border-border bg-card p-5 transition-colors hover:border-primary/40 hover:bg-accent/40 block"
        >
          <Bug className="h-5 w-5 text-primary mb-3" />
          <h3 className="text-base font-medium text-foreground">
            Open an issue
            <ExternalLink
              className="ml-0.5 inline-block h-3 w-3 align-baseline opacity-60"
              aria-hidden="true"
            />
          </h3>
          <p className="text-sm text-muted-foreground mt-1">
            Report bugs or request features on GitHub.
          </p>
        </a>
        <a
          href="https://github.com/PlatosTwin/pandects-app"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Main GitHub repository (opens in a new tab)"
          className="rounded-lg border border-border bg-card p-5 transition-colors hover:border-primary/40 hover:bg-accent/40 block"
        >
          <GitPullRequest className="h-5 w-5 text-primary mb-3" />
          <h3 className="text-base font-medium text-foreground">
            Send a pull request
            <ExternalLink
              className="ml-0.5 inline-block h-3 w-3 align-baseline opacity-60"
              aria-hidden="true"
            />
          </h3>
          <p className="text-sm text-muted-foreground mt-1">
            Be the change you want to see.
          </p>
        </a>
      </div>

      <p className="mt-12 text-sm text-muted-foreground">
        Thanks for helping make Pandects better!
      </p>

      <section id="form" className="pt-8">
        <div className="flex items-center gap-3">
          <span className="inline-flex h-7 w-7 items-center justify-center rounded-md bg-primary/10 text-primary text-xs font-medium tabular-nums">
            01
          </span>
          <h2 className="text-2xl font-semibold tracking-tight text-foreground">
            Feedback form
          </h2>
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
            loading="lazy"
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
    </PageShell>
  );
}
